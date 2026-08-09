"""S-B: Ben-Or/Tiwari sparse interpolation under a Kronecker substitution.

The dense grid (zpbatch / exact-QQ) pays for EVERY cell of the
prod(di+1) x L tensor whether or not the polynomial has a term there. This
backend discovers the true support T (num/den of the flagship k=12 bench:
~10^2..10^3 terms vs 20 736 grid cells) and then pays per TERM:

  1. Kronecker map: kept variable x_a -> y^{K_a} with C-order strides
     K_a = prod_{b>a}(d_b+1), so a monomial's Kronecker exponent IS its
     C-order flat index in the dense tensor. Distinctness needs
     ord(omega) >= K = prod(d_a+1): omega is a primitive root mod p.
  2. Support discovery (one prime): probe dets at y = omega^j in doubling
     batches, project the (s-power x {num,den}) vector values to scalars
     with random weights, Berlekamp-Massey each round; stop when the
     recurrence length L satisfies 2L + pad <= J (Kaltofen-Lee style
     early termination -- no a-priori T bound).
  3. Exponent recovery WITHOUT discrete logs: evaluate the connection
     polynomial at all K candidate values omega^E in one vectorized
     Horner pass (K is the dense grid size, <= ~10^5 in our regime -- the
     brute-force scan is milliseconds; upgrade to Pohlig-Hellman logs only
     if K ever outgrows this).
  4. Coefficients per s-power by the O(T^2) transposed-Vandermonde solve
     (synthetic division of the master polynomial), numpy-vectorized over
     the root array.
  5. Extra primes REUSE the support: T probes each, one solve, no BM
     (the classical Zippel remnant); primes run in parallel via the
     caller's pool. CRT + maximal-quotient rational reconstruction and
     the one-confirming-prime acceptance are shared with zpbatch; the
     caller's off-grid exact probe self-check remains the final authority
     with the dense QQ grid as fallback.

Probe economics at k=12 (~930-bit coefficients -> ~31 primes): support
discovery ~2T + T per extra prime ~ 2.6 k + 19.5 k points, x L x 2 dets,
vs the dense grid's 20 736 x L x 2 PER PRIME EACH -- the sparse path is
where the S-A mod-p kernel finally pays.
"""
from __future__ import annotations

import itertools
import random

import numpy as np
import sympy as sp
from sympy import QQ

from .zpbatch import (ZpBatchError, _batch_det_modp, _confirm,
                      _eval_dets_slab, _lift, _primes, _transform_modp)

__all__ = ["solve_tensors_sparse", "BotError"]


class BotError(RuntimeError):
    pass


_PAD = 8                       # early-termination confirmation probes
_SEED = 20260727               # projection weights: fixed for reproducibility


# ------------------------------------------------------------ small pieces
def _proot(p: int) -> int:
    return int(sp.primitive_root(p))


def _pow_seq(base: int, j0: int, j1: int, p: int) -> np.ndarray:
    """[base^j0, ..., base^(j1-1)] mod p."""
    out = np.empty(j1 - j0, dtype=np.int64)
    v = pow(base, j0, p)
    for i in range(j1 - j0):
        out[i] = v
        v = v * base % p
    return out


def _bm(seq: np.ndarray, p: int) -> tuple[np.ndarray, int]:
    """Berlekamp-Massey over Zp. Returns (C, L) with C[0] = 1 and
    seq[n] + sum_{i=1..L} C[i]*seq[n-i] == 0 (mod p) for all n >= L."""
    N = len(seq)
    C = np.zeros(N + 1, dtype=np.int64)
    B = np.zeros(N + 1, dtype=np.int64)
    C[0] = B[0] = 1
    L, m, b = 0, 1, 1
    for n in range(N):
        if L:
            win = seq[n - L:n][::-1]
            d = (int(seq[n]) + int((C[1:L + 1] * win % p).sum())) % p
        else:
            d = int(seq[n]) % p
        if d == 0:
            m += 1
            continue
        coef = d * pow(b, -1, p) % p
        if 2 * L <= n:
            T0 = C.copy()
            C[m:] = (C[m:] - coef * B[:N + 1 - m]) % p
            B, b = T0, d
            L = n + 1 - L
            m = 1
        else:
            C[m:] = (C[m:] - coef * B[:N + 1 - m]) % p
            m += 1
    return C[:L + 1] % p, L


def _candidate_values(lens: list[int], radK: list[int], g: int,
                      p: int) -> np.ndarray:
    """omega^E for every exponent tuple in the degree box, in C-order (so
    index == Kronecker exponent E)."""
    vals = np.ones(1, dtype=np.int64)
    for ln, Ka in zip(lens, radK):
        col = np.array([pow(g, Ka * e % (p - 1), p) for e in range(ln)],
                       dtype=np.int64)
        vals = (vals[:, None] * col[None, :] % p).reshape(-1)
    return vals


def _roots_to_exponents(C: np.ndarray, L: int, cand: np.ndarray,
                        p: int) -> np.ndarray:
    """Zeros of Lambda(x) = x^L * C(1/x) among the candidates; the surviving
    candidate INDICES are the Kronecker exponents. Count must equal L."""
    acc = np.full(cand.shape[0], int(C[0]), dtype=np.int64)
    for i in range(1, L + 1):
        acc = (acc * cand + int(C[i])) % p
    Es = np.nonzero(acc == 0)[0]
    if Es.shape[0] != L:
        raise BotError(
            f"support recovery found {Es.shape[0]} roots for recurrence "
            f"length {L} (unlucky prime/projection)")
    return Es


def _master_poly(R: np.ndarray, p: int) -> np.ndarray:
    """Monic prod(z - r) as coefficient array a[0..T] (a[T] = 1)."""
    T = R.shape[0]
    a = np.zeros(T + 1, dtype=np.int64)
    a[0] = 1
    for t in range(T):
        r = int(R[t])
        old = a[:t + 1].copy()                     # degree-t poly
        a[1:t + 2] = old                           # z * poly
        a[0] = 0
        a[:t + 1] = (a[:t + 1] - r * old) % p      # - r * poly
    return a


def _tvand_solve(R: np.ndarray, V: np.ndarray, p: int) -> np.ndarray:
    """Solve v_j = sum_t c_t * R_t^j (j = 0..T-1) for every column of V.

    O(T^2) via synthetic division of the master polynomial: with
    q_t(z) = Lambda(z)/(z - R_t), q_t(R_s) = 0 for s != t, so
    c_t = <q_t, v> / Lambda'(R_t). V: (T, L) -> returns (T, L)."""
    T = R.shape[0]
    if T == 0:
        return np.zeros((0, V.shape[1]), dtype=np.int64)
    a = _master_poly(R, p)
    Q = np.empty((T, T), dtype=np.int64)           # Q[:, j] = q_(.),j
    Q[:, T - 1] = 1
    for j in range(T - 1, 0, -1):
        Q[:, j - 1] = (int(a[j]) + R * Q[:, j]) % p
    d = Q[:, T - 1].copy()                          # q_t(R_t) by Horner
    for j in range(T - 2, -1, -1):
        d = (d * R + Q[:, j]) % p
    dinv = np.array([pow(int(x), -1, p) for x in d], dtype=np.int64)
    out = np.empty((T, V.shape[1]), dtype=np.int64)
    for l in range(V.shape[1]):
        W = Q * V[:T, l][None, :] % p               # products reduced first
        out[:, l] = W.sum(axis=1) % p * dinv % p    # partial sums < T*p
    return out


# -------------------------------------------------------------- evaluation
def _svinv_res(svinv, p: int) -> list[list[int]]:
    return [[int(v.numerator) % p * pow(int(v.denominator), -1, p) % p
             for v in row] for row in svinv]


def _eval_points(pay_den: dict, pay_num: dict, s_res: np.ndarray,
                 svinv_res: list[list[int]], kept_vals: list[np.ndarray],
                 p: int, slab: int = 16384):
    """Per-s-power values of den and num at arbitrary kept-var points.
    kept_vals: one int64 array (J,) per kept generator. -> (vd, vn) (J, L)."""
    J = kept_vals[0].shape[0]
    L = s_res.shape[0]
    B_tot = J * L
    dd = np.empty(B_tot, dtype=np.int64)
    dn = np.empty(B_tot, dtype=np.int64)
    for start in range(0, B_tot, slab):
        stop = min(start + slab, B_tot)
        fl = np.arange(start, stop)
        s_ix = fl % L
        pt = fl // L
        gv = [s_res[s_ix]] + [kv[pt] for kv in kept_vals]
        dd[start:stop] = _batch_det_modp(_eval_dets_slab(pay_den, gv, p), p)
        dn[start:stop] = _batch_det_modp(_eval_dets_slab(pay_num, gv, p), p)
    vd = _transform_modp(dd.reshape(J, L), svinv_res, 1, p)
    vn = _transform_modp(dn.reshape(J, L), svinv_res, 1, p)
    return vd, vn


def _s_residues(s_pairs: list, p: int) -> np.ndarray:
    return np.array([a % p * pow(b, -1, p) % p for a, b in s_pairs],
                    dtype=np.int64)


# ------------------------------------------------------- support discovery
def _discover(pay_den, pay_num, s_pairs, svinv, lens, radK, p,
              progress=None):
    """Probe at omega^j in doubling batches until Berlekamp-Massey
    terminates early for BOTH polynomials. Returns
    (Es_den, Es_num, vd, vn) with probe values for j = 0..J-1."""
    K = 1
    for ln in lens:
        K *= ln
    if K > 5_000_000:
        raise BotError(f"candidate box K={K} too large for the "
                       f"brute-force exponent scan (needs discrete logs)")
    g = _proot(p)
    bases = [pow(g, Ka % (p - 1), p) for Ka in radK]
    s_res = _s_residues(s_pairs, p)
    sv_res = _svinv_res(svinv, p)
    L_s = s_res.shape[0]
    rng = random.Random(_SEED)
    wd = np.array([rng.randrange(1, p) for _ in range(L_s)], dtype=np.int64)
    wn = np.array([rng.randrange(1, p) for _ in range(L_s)], dtype=np.int64)

    Jmax = 2 * K + 4 * _PAD
    J = 0
    Jnext = min(max(64, 4 * _PAD), Jmax)
    vd = np.empty((0, L_s), dtype=np.int64)
    vn = np.empty((0, L_s), dtype=np.int64)
    while True:
        kept_vals = [_pow_seq(b, J, Jnext, p) for b in bases]
        nd, nn = _eval_points(pay_den, pay_num, s_res, sv_res, kept_vals, p)
        vd = np.concatenate([vd, nd])
        vn = np.concatenate([vn, nn])
        J = Jnext
        seq_d = (vd * wd[None, :] % p).sum(axis=1) % p
        seq_n = (vn * wn[None, :] % p).sum(axis=1) % p
        Cd, Ld = _bm(seq_d, p)
        Cn, Ln = _bm(seq_n, p)
        if progress is not None:
            # REAL movement against the worst-case probe budget -- the
            # old report was progress(0, J), a bar frozen at zero while
            # J doubled underneath it. Early BM termination just ends
            # the phase before the bar fills; that is honest too.
            progress(J, Jmax)
        if 2 * Ld + _PAD <= J and 2 * Ln + _PAD <= J:
            break
        if J >= Jmax:
            raise BotError(
                f"support did not stabilise within {J} probes "
                f"(dense grid size K={K}: the dense path is cheaper)")
        Jnext = min(2 * J, Jmax)

    cand = _candidate_values(lens, radK, g, p)
    Es_d = _roots_to_exponents(Cd, Ld, cand, p)
    Es_n = _roots_to_exponents(Cn, Ln, cand, p)
    return Es_d, Es_n, vd, vn


# --------------------------------------------------------- per-prime solve
def _coeffs_for_prime(pay_den, pay_num, s_pairs, svinv, radK, Es_d, Es_n,
                      p: int):
    """Residue coefficient arrays for one prime, reusing the known support:
    T probes, transposed-Vandermonde solves. -> (cd (Td, L), cn (Tn, L))."""
    g = _proot(p)
    bases = [pow(g, Ka % (p - 1), p) for Ka in radK]
    s_res = _s_residues(s_pairs, p)
    sv_res = _svinv_res(svinv, p)
    T = max(Es_d.shape[0], Es_n.shape[0])
    kept_vals = [_pow_seq(b, 0, T, p) for b in bases]
    vd, vn = _eval_points(pay_den, pay_num, s_res, sv_res, kept_vals, p)
    Rd = np.array([pow(g, int(E) % (p - 1), p) for E in Es_d],
                  dtype=np.int64)
    Rn = np.array([pow(g, int(E) % (p - 1), p) for E in Es_n],
                  dtype=np.int64)
    cd = _tvand_solve(Rd, vd[:Rd.shape[0]], p)
    cn = _tvand_solve(Rn, vn[:Rn.shape[0]], p)
    return cd, cn


def _prime_worker(args):
    """Picklable pool task: one extra prime's coefficient residues."""
    pay_den, pay_num, s_pairs, svinv, radK, Es_d, Es_n, p = args
    try:
        cd, cn = _coeffs_for_prime(pay_den, pay_num, s_pairs, svinv, radK,
                                   Es_d, Es_n, p)
        return p, cd.reshape(-1), cn.reshape(-1)
    except (ZeroDivisionError, BotError):          # unlucky prime
        return p, None, None


# -------------------------------------------------------------- assembling
def _to_dict(lift: dict, Es: np.ndarray, lens: list[int], L: int) -> dict:
    """{flat t*L+l: (a,b)} -> the dense-path tensor dict {idx: [QQ]*L},
    zeros included, so the caller's _tensor_check/_tensor_to_expr see the
    exact same shape the dense backends produce."""
    out = {idx: [QQ.zero] * L
           for idx in itertools.product(*(range(m) for m in lens))}
    for flat, (a, b) in lift.items():
        t, l = divmod(int(flat), L)
        idx = tuple(int(i) for i in np.unravel_index(int(Es[t]), lens))
        out[idx][l] = QQ(a, b)
    return out


#: the prime ceiling and its justification live in zpbatch -- one
#: number, one story; a re-tune must not apply to one backend only
from .zpbatch import _MAX_PRIMES


# ------------------------------------------------------------- entry point
def solve_tensors_sparse(pay_den: dict, pay_num: dict, grids_pairs: list,
                         s_pairs: list, vinvs: list, progress=None,
                         map_impl=None, start_primes: int = 6,
                         step_primes: int = 4, max_primes: int = _MAX_PRIMES,
                         batch: int | None = None,
                         info: dict | None = None):
    """The BOT/Kronecker pipeline. Same contract as zpbatch.solve_tensors:
    returns (den_t, num_t) tensor dicts or raises; the caller's exact probe
    self-check decides acceptance."""
    lens = [len(g) for g in grids_pairs]
    if not lens:
        raise BotError("no kept variables: nothing to interpolate sparsely")
    radK = [1] * len(lens)
    for a in range(len(lens) - 2, -1, -1):
        radK[a] = radK[a + 1] * lens[a + 1]
    L = len(s_pairs)
    svinv = vinvs[0]

    run = map_impl if map_impl is not None else map
    if batch and batch > 1:
        start_primes = max(start_primes, batch)
        step_primes = max(step_primes, batch)

    # support discovery on the first non-unlucky prime
    stream = 0
    Es_d = Es_n = None
    while Es_d is None:
        stream += 1
        if stream > 4:
            raise BotError("no usable support prime in 4 attempts")
        p1 = _primes(stream)[-1]
        try:
            Es_d, Es_n, vd, vn = _discover(
                pay_den, pay_num, s_pairs, svinv, lens, radK, p1, progress)
        except ZeroDivisionError:
            continue
    g1 = _proot(p1)
    Rd = np.array([pow(g1, int(E) % (p1 - 1), p1) for E in Es_d],
                  dtype=np.int64)
    Rn = np.array([pow(g1, int(E) % (p1 - 1), p1) for E in Es_n],
                  dtype=np.int64)
    cd1 = _tvand_solve(Rd, vd[:Rd.shape[0]], p1)
    cn1 = _tvand_solve(Rn, vn[:Rn.shape[0]], p1)

    primes = [p1]
    stacks_d = [cd1.reshape(-1)]
    stacks_n = [cn1.reshape(-1)]
    hard_d: list[int] = []           # coefficients that needed the most primes
    hard_n: list[int] = []
    st_d: dict = {}                  # incremental CRT state, one per tensor
    st_n: dict = {}
    want = max(start_primes, stream)
    while True:
        new = [q for q in _primes(want) if q not in primes]
        results = list(run(_prime_worker,
                           [(pay_den, pay_num, s_pairs, svinv, radK,
                             Es_d, Es_n, q) for q in new]))
        for q, cd, cn in results:
            if cd is None:
                continue
            primes.append(q)
            stacks_d.append(cd)
            stacks_n.append(cn)
        if progress is not None:
            # never report done == total: the caller reads that as "the
            # evaluation is finished", and more prime rounds may follow
            progress(len(primes), want + step_primes)
        if len(primes) >= 2:
            lift_d = _lift(stacks_d, primes, hard=hard_d, state=st_d)
            lift_n = (_lift(stacks_n, primes, hard=hard_n, state=st_n)
                      if lift_d is not None else None)
            if lift_n is not None:
                want += 1
                (q, cd, cn), = list(run(
                    _prime_worker,
                    [(pay_den, pay_num, s_pairs, svinv, radK,
                      Es_d, Es_n, _primes(want)[-1])]))
                if cd is not None:
                    if (_confirm(lift_d, cd, q) and _confirm(lift_n, cn, q)):
                        if info is not None:
                            info.update(primes=len(primes) + 1,
                                        probes_j=int(vd.shape[0]),
                                        T_den=int(Es_d.shape[0]),
                                        T_num=int(Es_n.shape[0]))
                        return (_to_dict(lift_d, Es_d, lens, L),
                                _to_dict(lift_n, Es_n, lens, L))
                    primes.append(q)
                    stacks_d.append(cd)
                    stacks_n.append(cn)
        if want >= max_primes:
            raise BotError(
                f"no stable rational reconstruction after {want} primes")
        want = min(max_primes, want + step_primes)
