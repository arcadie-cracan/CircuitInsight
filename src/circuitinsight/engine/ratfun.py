"""S-C: reduced rational-function reconstruction (shifted Cuyt-Lee over Zp).

The S-A/S-B backends reconstruct the RAW Cramer polynomials det(A) and
det(Ak): ~930-bit coefficients (31+ CRT primes) and more terms than the
reduced transfer function. This backend interpolates the REDUCED pair
N/D directly:

  * The black box is the VALUE of H = det(Ak)/det(A) at a point mod p --
    the polynomial content shared by both dets cancels in the ratio, so
    the values belong to the reduced function even though we never
    compute a gcd.
  * s is folded into the variable box as generator 0, so the whole object
    is one multivariate rational function over (s, x1..xk) and the output
    is the same {kept idx: [QQ]*L} tensor pair the other backends produce
    -- for the reduced polynomials.
  * Homogenizing variable with a SHIFT (Cuyt-Lee as hardened by
    FireFly): probe g_j(t) = H(beta + t*z_j) along geometric points
    z_j = omega^(j*K_a). The shift is NOT optional here: measured on the
    ota5t bench, the reduced num and den both have zero constant AND
    linear parts, so unshifted lines all carry a common t^2 factor and
    the per-line normalization becomes z-dependent (the slot sequences
    turn into rational functions and Berlekamp-Massey never terminates).
    With the shift, B(t=0) = D(beta) is a CONSTANT, the normalization is
    global, and every t-degree slot is a polynomial sequence again.
  * The shifted expansion's slots are dense, but the TRUE homogeneous
    parts are recovered top-down: the top slot is unshifted; each level
    below subtracts the binomial contributions of all already-known
    higher-degree terms ("temporal elimination of the shift"), leaving
    exactly the sparse degree-d part. BM + candidate scan + transposed
    Vandermonde per level -- all S-B machinery, with candidates
    restricted to total degree d.
  * Actual reduced total degrees (dN, dB) come from ONE Cauchy/EEA
    rational prescan on a random shifted line.
  * Per j, the cross-multiplied t-sweep systems are solved for all j at
    once by batched Gauss-Jordan mod p, with one spare row per system as
    verification.
  * Extra primes reuse the supports (T probes each, no BM); the shift
    point beta is FIXED across primes so every prime sees residues of
    the same rational coefficients (normalized by the constant D(beta)).
    CRT + maximal-quotient rational reconstruction + one confirming
    prime, shared with zpbatch. The caller accepts only through a
    cross-multiplied ratio identity against its exact QQ[s] probe, with
    the raw dense grid as fallback.
"""
from __future__ import annotations

import itertools
import math
import random

import numpy as np
import sympy as sp
from sympy import QQ

from .botsparse import (BotError, _bm, _pow_seq, _proot, _tvand_solve)
from .zpbatch import (_batch_det_modp, _confirm, _eval_dets_slab, _lift,
                      _powmod_arr, _primes)

__all__ = ["solve_tensors_ratfun", "RatFunError"]


class RatFunError(RuntimeError):
    pass


_PAD = 8
_SEED = 20260727


# ------------------------------------------------- univariate mod-p helpers
def _poly_eval(c: list[int], x: int, p: int) -> int:
    acc = 0
    for a in reversed(c):
        acc = (acc * x + a) % p
    return acc


def _poly_deg(c: list[int]) -> int:
    d = len(c) - 1
    while d > 0 and c[d] == 0:
        d -= 1
    return d if any(c) else -1


def _lagrange(ts: list[int], gs: list[int], p: int) -> list[int]:
    """Dense interpolating polynomial through (ts, gs), degree < len(ts)."""
    n = len(ts)
    P = [0] * n
    for i in range(n):
        num = [1]
        denom = 1
        for j in range(n):
            if j == i:
                continue
            num = [(a * (-ts[j]) + (num[m - 1] if m else 0)) % p
                   for m, a in enumerate(num)] + [num[-1]]
            denom = denom * (ts[i] - ts[j]) % p
        scale = gs[i] * pow(denom, -1, p) % p
        for m in range(len(num)):
            P[m] = (P[m] + scale * num[m]) % p
    return P


def _cauchy_minimal(ts: list[int], gs: list[int], p: int):
    """Minimal-degree rational interpolant A/B via the extended Euclidean
    scheme on (prod(t - ti), Lagrange(P)). Returns (dN, dB, B0_nonzero)."""
    M = [1]
    for t in ts:
        M = [(-t * a + (M[m - 1] if m else 0)) % p
             for m, a in enumerate(M)] + [M[-1]]
    P = _lagrange(ts, gs, p)
    r0, r1 = M, P
    v0, v1 = [0], [1]
    best = None
    cand = [(r1, v1)]
    while _poly_deg(r1) >= 0:
        q = [0] * max(1, _poly_deg(r0) - _poly_deg(r1) + 1)
        r2 = list(r0)
        d1 = _poly_deg(r1)
        inv = pow(r1[d1], -1, p)
        for d in range(_poly_deg(r2), d1 - 1, -1):
            c = r2[d] * inv % p
            if c:
                q[d - d1] = c
                for m in range(d1 + 1):
                    r2[d - d1 + m] = (r2[d - d1 + m] - c * r1[m]) % p
        v2 = list(v0) + [0] * max(0, len(q) + len(v1) - len(v0))
        for m, qa in enumerate(q):
            if qa:
                for mm, va in enumerate(v1):
                    v2[m + mm] = (v2[m + mm] - qa * va) % p
        r0, r1, v0, v1 = r1, r2, v1, v2
        if _poly_deg(r1) >= 0:
            cand.append((r1, v1))
    for A, B in cand:
        dA, dB = _poly_deg(A), _poly_deg(B)
        if dA < 0 or dB < 0:
            continue
        tot = dA + dB
        if best is None or tot < best[0]:
            best = (tot, dA, dB, B)
    if best is None:
        raise RatFunError("rational prescan found no interpolant")
    _, dN, dB, B = best
    return dN, dB, B[0] % p != 0


# ------------------------------------------------------ batched linear solve
def _batch_solve(M: np.ndarray, p: int):
    """Gauss-Jordan on a stack of augmented (u, u+1) systems mod p.
    Returns (solutions (B, u), bad mask for singular systems)."""
    M = M % p
    B, u, _ = M.shape
    bad = np.zeros(B, dtype=bool)
    for c in range(u):
        piv = M[:, c, c]
        z = np.nonzero((piv == 0) & ~bad)[0]
        if z.size:
            if c + 1 < u:
                sub = M[z, c + 1:, c] != 0
                has = sub.any(axis=1)
                src = c + 1 + np.argmax(sub, axis=1)
                swp, rs = z[has], src[has]
                if swp.size:
                    tmp = M[swp, c, :].copy()
                    M[swp, c, :] = M[swp, rs, :]
                    M[swp, rs, :] = tmp
                dead = z[~has]
            else:
                dead = z
            if dead.size:
                bad[dead] = True
                M[dead, c, c] = 1
        inv = _powmod_arr(M[:, c, c], p - 2, p)
        M[:, c, :] = M[:, c, :] * inv[:, None] % p
        col = M[:, :, c].copy()
        col[:, c] = 0
        M -= col[:, :, None] * M[:, c, :][:, None, :]
        M %= p
    return M[:, :, u], bad


# ------------------------------------------------------------- evaluation
def _hval_points(pay_den, pay_num, gen_vals, p, slab: int = 16384):
    """(den, num) det values at arbitrary generator points (arrays)."""
    B = gen_vals[0].shape[0]
    vd = np.empty(B, dtype=np.int64)
    vn = np.empty(B, dtype=np.int64)
    for start in range(0, B, slab):
        stop = min(start + slab, B)
        gv = [g[start:stop] for g in gen_vals]
        vd[start:stop] = _batch_det_modp(_eval_dets_slab(pay_den, gv, p), p)
        vn[start:stop] = _batch_det_modp(_eval_dets_slab(pay_num, gv, p), p)
    return vd, vn


def _t_systems(pay_den, pay_num, zvals, beta_res, tpows, dN, dB, p):
    """For each shifted line beta + t*z_j build and solve the
    cross-multiplied t-sweep system; returns (sols (J, u), ok).

    Equations at t_i:  sum_d a_d t^d * vD - sum_{e>=1} b_e t^e * vN = vN
    (B(0) = 1 normalization, valid because B(0) = D(beta)/D(beta))."""
    J = zvals[0].shape[0]
    M = dN + dB + 2
    u = dN + 1 + dB
    gv = [(b + z[:, None] * tpows[None, :]) % p
          for b, z in zip(beta_res, zvals)]
    gv = [g.reshape(-1) for g in gv]
    vD, vN = _hval_points(pay_den, pay_num, gv, p)
    vD = vD.reshape(J, M)
    vN = vN.reshape(J, M)
    tp = np.empty((M, max(dN, dB) + 1), dtype=np.int64)
    tp[:, 0] = 1
    for d in range(1, tp.shape[1]):
        tp[:, d] = tp[:, d - 1] * tpows % p
    A = np.empty((J, M, u + 1), dtype=np.int64)
    for d in range(dN + 1):
        A[:, :, d] = vD * tp[None, :, d] % p
    for e in range(1, dB + 1):
        A[:, :, dN + e] = (p - vN * tp[None, :, e] % p) % p
    A[:, :, u] = vN
    sols, bad = _batch_solve(A[:, :u, :].copy(), p)
    chk = (A[:, u, :u] * sols % p).sum(axis=1) % p
    ok = ~bad & (chk == A[:, u, u])
    return sols, ok


# ------------------------------------------------------ shift corrections
def _alpha_subsets(E: tuple, d: int):
    """All alpha <= E (componentwise) with |alpha| = d."""
    out = []

    def rec(i, left, acc):
        if i == len(E):
            if left == 0:
                out.append(tuple(acc))
            return
        for a in range(min(E[i], left) + 1):
            acc.append(a)
            rec(i + 1, left - a, acc)
            acc.pop()

    rec(0, d, [])
    return out


def _correction(known, d, beta, radK, bases, J, p, powcache):
    """sum over known higher-degree terms (E, c) of c * [t^d] of
    prod (beta_i + t z_i)^{E_i}, evaluated over the geometric points
    z_j = bases^j, as an array (J,)."""
    acc = np.zeros(J, dtype=np.int64)
    for E, c in known:
        for alpha in _alpha_subsets(E, d):
            coef = c
            for Ei, ai, bi in zip(E, alpha, beta):
                coef = coef * (math.comb(Ei, ai) % p) % p
                if Ei - ai:
                    coef = coef * pow(bi % p, Ei - ai, p) % p
            if coef == 0:
                continue
            aflat = sum(a * K for a, K in zip(alpha, radK))
            seq = powcache.get(aflat)
            if seq is None or seq.shape[0] < J:
                base = 1
                for b, a in zip(bases, alpha):
                    if a:
                        base = base * pow(b, a, p) % p
                seq = _pow_seq(base, 0, J, p)
                powcache[aflat] = seq
            acc = (acc + coef * seq[:J]) % p
    return acc


# ----------------------------------------------------------- degree buckets
def _degree_buckets(lens: list[int], radK: list[int]) -> np.ndarray:
    K = 1
    for ln in lens:
        K *= ln
    tot = np.zeros(K, dtype=np.int32)
    idx = np.arange(K)
    for ln, Ka in zip(lens, radK):
        tot += (idx // Ka) % ln
    return tot


def _unflat(E: int, lens: list[int]) -> tuple:
    return tuple(int(v) for v in np.unravel_index(int(E), lens))


# --------------------------------------------------------------- cascade
def _cascade(sols, dN, dB, lens, radK, beta, g, p, totdeg, J,
             supports=None):
    """Top-down shift elimination for both polynomials.

    Discovery mode (supports=None): BM per level with early termination;
    returns (supports, coeffs, stable). Fixed-support mode: tvand per
    level; returns coeffs (or raises on root trouble). supports/coeffs:
    {("den"|"num", d): np.ndarray}."""
    bases = [pow(g, Ka % (p - 1), p) for Ka in radK]
    powcache: dict = {}
    if supports is None:
        cand_full = None                            # built lazily
        sup_out: dict = {}
    coef_out: dict = {}
    for poly, dmax in (("den", dB), ("num", dN)):
        known: list = []
        for d in range(dmax, -1, -1):
            if poly == "den":
                seq = (np.ones(J, dtype=np.int64) if d == 0
                       else sols[:J, dN + d].astype(np.int64))
            else:
                seq = sols[:J, d].astype(np.int64)
            corr = _correction(known, d, beta, radK, bases, J, p, powcache)
            seq = (seq - corr) % p
            if supports is None:
                C, Lr = _bm(seq, p)
                if 2 * Lr + _PAD > J:
                    return None, None, False
                if Lr == 0:
                    sup_out[(poly, d)] = np.empty(0, dtype=np.int64)
                    coef_out[(poly, d)] = np.empty(0, dtype=np.int64)
                    continue
                if cand_full is None:
                    from .botsparse import _candidate_values
                    cand_full = _candidate_values(lens, radK, g, p)
                deg_ix = np.nonzero(totdeg == d)[0]
                cv = cand_full[deg_ix]
                acc = np.full(cv.shape[0], int(C[0]), dtype=np.int64)
                for i in range(1, Lr + 1):
                    acc = (acc * cv + int(C[i])) % p
                Es = deg_ix[np.nonzero(acc == 0)[0]]
                if Es.shape[0] != Lr:
                    # spurious BM stabilisation: ask for more probes
                    return None, None, False
                sup_out[(poly, d)] = Es
            else:
                Es = supports[(poly, d)]
                if Es.shape[0] == 0:
                    coef_out[(poly, d)] = np.empty(0, dtype=np.int64)
                    continue
            R = np.array([pow(g, int(E) % (p - 1), p) for E in Es],
                         dtype=np.int64)
            cs = _tvand_solve(R, seq[:R.shape[0]][:, None], p)[:, 0]
            coef_out[(poly, d)] = cs
            known.extend((_unflat(int(E), lens), int(c))
                         for E, c in zip(Es, cs) if c)
    if supports is None:
        return sup_out, coef_out, True
    return coef_out


def _flat_coeffs(slots, coeffs) -> np.ndarray:
    parts = [coeffs[sl] for sl in slots]
    return (np.concatenate(parts) if parts
            else np.empty(0, dtype=np.int64))


#: Rational reconstruction needs modulus proportional to the SIZE of the
#: exact coefficients -- sums of products of ~n matrix entries -- which a
#: 30x30 hybrid grid drives far past what a few dozen 62-bit primes span.
#: A real fc keep set failed at 64 ("no stable rational reconstruction
#: after 64 primes"), did all its probing for nothing, and re-ran the
#: whole grid densely; at 256 it CONVERGES and the solve is faster than
#: the fallback route (184 s vs 232 s, measured). Raising the ceiling is
#: free for easy cases -- the loop stops as soon as two lifts agree, so
#: nobody pays for primes they do not need.
_MAX_PRIMES = 256


# ------------------------------------------------------------- entry point
def solve_tensors_ratfun(pay_den: dict, pay_num: dict, grids_pairs: list,
                         s_pairs: list, vinvs: list, progress=None,
                         map_impl=None, start_primes: int = 4,
                         step_primes: int = 4, max_primes: int = _MAX_PRIMES,
                         batch: int | None = None,
                         info: dict | None = None):
    """Reduced-pair reconstruction. Returns (den_t, num_t) tensor dicts of
    the reduced transfer function normalized by the constant D(beta) (the
    caller must accept via a ratio check, not the raw tensor check)."""
    L = len(s_pairs)
    lens = [L] + [len(g) for g in grids_pairs]
    K = 1
    for ln in lens:
        K *= ln
    if K > 5_000_000:
        raise RatFunError(f"candidate box K={K} too large")
    radK = [1] * len(lens)
    for a in range(len(lens) - 2, -1, -1):
        radK[a] = radK[a + 1] * lens[a + 1]
    raw_tot = sum(ln - 1 for ln in lens)
    totdeg = _degree_buckets(lens, radK)

    run = map_impl if map_impl is not None else map
    if batch and batch > 1:
        start_primes = max(start_primes, batch)
        step_primes = max(step_primes, batch)

    rng = random.Random(_SEED)
    beta = [rng.randrange(1, 20) for _ in lens]     # FIXED across primes

    # ---- prescan on a random shifted line: reduced total degrees ---------
    p1 = _primes(1)[-1]
    for attempt in range(6):
        M0 = 2 * raw_tot + 2
        gam = rng.randrange(2, p1)
        ts = [pow(gam, i + 1, p1) for i in range(M0)]
        if len(set(ts)) < M0:
            continue
        z0 = [rng.randrange(1, p1) for _ in lens]
        gv = [np.array([(b + z * t) % p1 for t in ts], dtype=np.int64)
              for b, z in zip(beta, z0)]
        vD, vN = _hval_points(pay_den, pay_num, gv, p1)
        if (vD == 0).any():
            continue
        gs = [int(n) * pow(int(d), -1, p1) % p1 for n, d in zip(vN, vD)]
        dN, dB, b0_ok = _cauchy_minimal(ts, gs, p1)
        break
    else:
        raise RatFunError("rational prescan failed (poles at every trial)")
    if not b0_ok:
        raise RatFunError("D(beta) = 0 at the shift point (unlucky beta)")

    slots = [("den", d) for d in range(dB, -1, -1)] + \
            [("num", d) for d in range(dN, -1, -1)]   # cascade order

    # ---- support discovery on prime 1 ------------------------------------
    g1 = _proot(p1)
    bases1 = [pow(g1, Ka % (p1 - 1), p1) for Ka in radK]
    beta1 = [b % p1 for b in beta]
    tpows = np.array(ts[:dN + dB + 2], dtype=np.int64)
    Jmax = 2 * K + 4 * _PAD
    J = 0
    Jnext = min(max(64, 4 * _PAD), Jmax)
    sols = np.empty((0, dN + 1 + dB), dtype=np.int64)
    while True:
        zvals = [_pow_seq(b, J, Jnext, p1) for b in bases1]
        s_new, ok = _t_systems(pay_den, pay_num, zvals, beta1, tpows,
                               dN, dB, p1)
        if not ok.all():
            raise RatFunError(
                "t-sweep system singular or inconsistent at the support "
                "prime (unlucky points or degree misestimate)")
        sols = np.concatenate([sols, s_new])
        J = Jnext
        supports, coeffs1, stable = _cascade(
            sols, dN, dB, lens, radK, beta1, g1, p1, totdeg, J)
        if progress is not None:
            # real movement against the worst-case probe budget (the old
            # progress(0, J) froze the bar at zero while J doubled)
            progress(J, Jmax)
        if stable:
            break
        if J >= Jmax:
            raise RatFunError(
                f"support did not stabilise within {J} probes")
        Jnext = min(2 * J, Jmax)

    Tmax = max((supports[sl].shape[0] for sl in slots), default=0)
    if Tmax == 0 or all(supports[("num", d)].shape[0] == 0
                        for d in range(dN + 1)):
        raise RatFunError("empty support: degenerate transfer function")

    primes = [p1]
    stacks = [_flat_coeffs(slots, coeffs1)]
    sup_list = [supports[sl] for sl in slots]
    wargs = (pay_den, pay_num, lens, radK, beta, sup_list, dN, dB,
             max(Tmax, dN + dB + 2))

    want = start_primes
    while True:
        new = [q for q in _primes(want) if q not in primes]
        results = list(run(_ratfun_prime_worker,
                           [wargs + (q,) for q in new]))
        for q, arr in results:
            if arr is None:
                continue
            primes.append(q)
            stacks.append(arr)
        if progress is not None:
            progress(len(primes), want)
        if len(primes) >= 2:
            lift = _lift(stacks, primes)
            if lift is not None:
                want += 1
                (q, arr), = list(run(_ratfun_prime_worker,
                                     [wargs + (_primes(want)[-1],)]))
                if arr is not None:
                    if _confirm(lift, arr, q):
                        if info is not None:
                            info.update(primes=len(primes) + 1,
                                        probes_j=J, deg_num=dN, deg_den=dB)
                        return _assemble(lift, sup_list, slots, lens, L)
                    primes.append(q)
                    stacks.append(arr)
        if want >= max_primes:
            raise RatFunError(
                f"no stable rational reconstruction after {want} primes")
        want = min(max_primes, want + step_primes)


def _ratfun_prime_worker(args):
    """One prime's coefficient residues on the fixed supports (picklable)."""
    (pay_den, pay_num, lens, radK, beta, sup_list, dN, dB, Tmax, q) = args
    try:
        slots = [("den", d) for d in range(dB, -1, -1)] + \
                [("num", d) for d in range(dN, -1, -1)]
        supports = dict(zip(slots, sup_list))
        g = _proot(q)
        bases = [pow(g, Ka % (q - 1), q) for Ka in radK]
        beta_q = [b % q for b in beta]
        rng = random.Random(_SEED + q)
        for _ in range(4):
            gam = rng.randrange(2, q)
            tpows = np.array([pow(gam, i + 1, q)
                              for i in range(dN + dB + 2)], dtype=np.int64)
            if len(set(tpows.tolist())) == tpows.shape[0]:
                break
        zvals = [_pow_seq(b, 0, Tmax, q) for b in bases]
        sols, ok = _t_systems(pay_den, pay_num, zvals, beta_q, tpows,
                              dN, dB, q)
        if not ok.all():
            return q, None
        totdeg = _degree_buckets(lens, radK)
        coeffs = _cascade(sols, dN, dB, lens, radK, beta_q, g, q, totdeg,
                          Tmax, supports=supports)
        return q, _flat_coeffs(slots, coeffs)
    except (ZeroDivisionError, BotError, RatFunError):
        return q, None


def _assemble(lift: dict, sup_list: list, slots: list, lens: list[int],
              L: int):
    """Lifted flat coefficients -> (den_t, num_t) reduced tensor dicts."""
    kept_lens = lens[1:]
    den_t = {idx: [QQ.zero] * L
             for idx in itertools.product(*(range(m) for m in kept_lens))}
    num_t = {idx: [QQ.zero] * L
             for idx in itertools.product(*(range(m) for m in kept_lens))}
    offsets = []
    off = 0
    for Es in sup_list:
        offsets.append(off)
        off += Es.shape[0]
    for flat, (a, b) in lift.items():
        flat = int(flat)
        si = max(i for i, o in enumerate(offsets) if o <= flat)
        E = int(sup_list[si][flat - offsets[si]])
        full = np.unravel_index(E, lens)
        es, idx = int(full[0]), tuple(int(v) for v in full[1:])
        tgt = den_t if slots[si][0] == "den" else num_t
        tgt[idx][es] = QQ(a, b)
    return den_t, num_t
