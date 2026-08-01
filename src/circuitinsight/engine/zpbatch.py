"""Batched mod-p grid evaluation — the S-A `zp-batch` backend.

Replaces the per-point exact-QQ determinants of the dense multilinear grid
(engine/interp.py) with, per 31-bit prime p:

  1. vectorized evaluation of the decomposed matrix entries on the WHOLE
     (kept-vars x s) tensor grid (numpy int64; products of two residues fit:
     p < 2^31 so a*b < 2^62 < 2^63);
  2. batched DIVISION-FREE Gaussian elimination over the stacked matrices --
     no per-pivot modular inverses (the spike measured those at ~90% of the
     naive cost); one batched inverse per matrix at the end;
  3. mod-p inverse-Vandermonde transforms to coefficient tensors.

(A float64 kernel with 26-bit primes and reciprocal-multiplication
reduction was tried and measured SLOWER: its 8 array passes per reduction
are memory-bound, while the int64 percent-op is one pass. Keep int64.)

Residue tensors from several primes are lifted to exact rationals by CRT +
maximal-quotient rational reconstruction, then verified against one extra
confirming prime. The caller's off-grid exact-QQ probe self-check remains
the final authority: an unlucky prime or a spurious reconstruction fails
it and the caller falls back to the exact-QQ grid, so this backend can be
fast without ever being trusted.

Why mod p at all: the spike (docs/interp-speedup-plan.md S-A) showed the
win is NOT per-probe -- exact-QQ dets are already ~0.4 ms at dim 17 -- but
batchability: mod-p arithmetic vectorizes over the whole grid in numpy,
which rational arithmetic never can.
"""
from __future__ import annotations

import itertools
import warnings

import numpy as np
import sympy as sp
from sympy import QQ

__all__ = ["solve_tensors", "ZpBatchError"]


class ZpBatchError(RuntimeError):
    pass


# ------------------------------------------------------------------ primes
_PRIMES: list[int] = []


def _primes(n: int) -> list[int]:
    """The first n 31-bit primes, descending from 2^31 (int64-safe:
    residue products stay below 2^62)."""
    while len(_PRIMES) < n:
        _PRIMES.append(int(sp.prevprime(_PRIMES[-1] if _PRIMES else 2**31)))
    return _PRIMES[:n]


# ---------------------------------------------------------- mod-p helpers
def _powmod_arr(base: np.ndarray, exp: int, p: int) -> np.ndarray:
    """base**exp mod p, elementwise, by binary exponentiation (array ops
    only -- a Python pow() per element was ~90% of the spike's cost)."""
    result = np.ones_like(base)
    b = base % p
    e = exp
    while e:
        if e & 1:
            result = result * b % p
        b = b * b % p
        e >>= 1
    return result


def _batch_det_modp(A: np.ndarray, p: int) -> np.ndarray:
    """Determinants of a stack of matrices mod p, division-free.

    Row update is cross-multiplication (new_i = piv*row_i - a_ik*row_k),
    which multiplies each matrix's determinant by piv^(rows below); the
    final diagonal product is corrected by ONE batched inverse per matrix:

        det = v_{n-1} * inv( prod_k v_k^{n-k-2} ) * sign

    Zero pivots are repaired by a vectorized row swap from below; matrices
    with a fully zero pivot column have det = 0 mod p -- a correct VALUE,
    not an error (the numerator determinant legitimately vanishes at some
    grid points)."""
    A = np.asarray(A) % p
    B, n, _ = A.shape
    if n == 1:
        return A[:, 0, 0].copy()
    sign = np.ones(B, dtype=np.int64)
    alive = np.ones(B, dtype=bool)
    scale = np.ones(B, dtype=np.int64)          # prod v_k^(n-k-2)
    for k in range(n):
        piv = A[:, k, k]
        bad = np.nonzero(piv == 0)[0]
        if bad.size:
            if k + 1 < n:
                sub = A[bad, k + 1:, k] != 0
                has = sub.any(axis=1)
                src = k + 1 + np.argmax(sub, axis=1)
                swp, rsrc = bad[has], src[has]
                if swp.size:
                    tmp = A[swp, k, :].copy()
                    A[swp, k, :] = A[swp, rsrc, :]
                    A[swp, rsrc, :] = tmp
                    sign[swp] = -sign[swp]
                dead = bad[~has]
            else:
                dead = bad                       # zero last pivot: det = 0
            if dead.size:
                alive[dead] = False
                A[dead, k, k] = 1                # keep later arithmetic sane
            piv = A[:, k, k]
        if k + 1 < n:
            A[:, k + 1:, k:] = (
                piv[:, None, None] * A[:, k + 1:, k:]
                - A[:, k + 1:, k][:, :, None] * A[:, k, k:][:, None, :]
            ) % p
        e = n - k - 2
        if e > 0:
            scale = scale * _powmod_arr(piv, e, p) % p
    det = A[:, n - 1, n - 1] * _powmod_arr(scale, p - 2, p) % p
    det = np.where(sign > 0, det, (p - det) % p)
    return np.where(alive, det, 0)


def _transform_modp(T: np.ndarray, M: list[list[int]], axis: int,
                    p: int) -> np.ndarray:
    """Apply the (small) matrix M along `axis` of T, mod p. Contraction is
    an explicit loop with a reduction per step: np.tensordot would sum
    int64 products WITHOUT intermediate reduction and overflow for
    contraction lengths > 2."""
    T = np.moveaxis(T, axis, 0)
    m = len(M)
    out = np.zeros_like(T)
    for j in range(m):
        acc = np.zeros_like(T[0])
        row = M[j]
        for i in range(m):
            c = row[i]
            if c:
                acc = (acc + c * T[i]) % p
        out[j] = acc
    return np.moveaxis(out, 0, axis)


# ------------------------------------------------------------- evaluation
def _eval_dets_slab(pay: dict, gen_vals: list[np.ndarray],
                    p: int) -> np.ndarray:
    """Dets of the decomposed matrix over one slab of grid points.
    gen_vals: per generator, the residue value array for the slab."""
    n = pay["n"]
    B = gen_vals[0].shape[0]
    pows: dict[tuple[int, int], np.ndarray] = {}

    def pw(a: int, d: int) -> np.ndarray:
        key = (a, d)
        arr = pows.get(key)
        if arr is None:
            arr = _powmod_arr(gen_vals[a], d, p) if d > 1 else gen_vals[a]
            pows[key] = arr
        return arr

    A = np.zeros((B, n, n), dtype=np.int64)
    for (i, j), terms in pay["entries"].items():
        acc = np.zeros(B, dtype=np.int64)
        for exps, (cp, cq) in terms:
            c = cp % p * pow(cq, -1, p) % p      # raises if p | cq -> unlucky
            t = None
            for a, e in enumerate(exps):
                if e:
                    t = pw(a, e) if t is None else t * pw(a, e) % p
            acc = (acc + (c if t is None else c * t % p)) % p
        A[:, i, j] = acc
    return A


def _prime_tensors(pay_den: dict, pay_num: dict, axes_num: list[list[int]],
                   vinvs: list[list[list]], p: int,
                   slab: int = 16384) -> tuple[np.ndarray, np.ndarray]:
    """Coefficient tensors of den and num mod p over the full grid.

    axes_num: per generator (s FIRST, then kept axes), the integer-pair
    grid points as (num, den) rationals; vinvs: the matching QQ inverse
    Vandermonde rows. Grid layout: kept axes in C order, s fastest."""
    # residues of the axis points
    ax_res = [np.array([a % p * pow(b, -1, p) % p for a, b in ax],
                       dtype=np.int64) for ax in axes_num]
    L = len(ax_res[0])
    kept_lens = [len(a) for a in ax_res[1:]]
    B_kept = int(np.prod(kept_lens)) if kept_lens else 1
    B_tot = B_kept * L

    dets_d = np.empty(B_tot, dtype=np.int64)
    dets_n = np.empty(B_tot, dtype=np.int64)
    for start in range(0, B_tot, slab):
        stop = min(start + slab, B_tot)
        flat = np.arange(start, stop, dtype=np.int64)
        s_ix = flat % L
        rest = flat // L
        gen_vals = [ax_res[0][s_ix]]
        kept_ix = []
        for ln in reversed(kept_lens):
            kept_ix.append(rest % ln)
            rest = rest // ln
        kept_ix.reverse()
        for a, ix in enumerate(kept_ix):
            gen_vals.append(ax_res[1 + a][ix])
        dets_d[start:stop] = _batch_det_modp(
            _eval_dets_slab(pay_den, gen_vals, p), p)
        dets_n[start:stop] = _batch_det_modp(
            _eval_dets_slab(pay_num, gen_vals, p), p)

    shape = tuple(kept_lens) + (L,)
    vin_res = [[[int(v.numerator) % p * pow(int(v.denominator), -1, p) % p
                 for v in row] for row in rows] for rows in vinvs]
    out = []
    for dets in (dets_d, dets_n):
        T = dets.reshape(shape)
        T = _transform_modp(T, vin_res[0], len(shape) - 1, p)   # s axis
        for a in range(len(kept_lens)):
            T = _transform_modp(T, vin_res[1 + a], a, p)
        out.append(T)
    return out[0], out[1]


# --------------------------------------------------------------- CRT lift
_MQ_GAP = 1 << 20                        # quotient-gap threshold (Monagan)


def _ratrec(u: int, m: int):
    """Maximal-quotient rational reconstruction (Monagan, ISSAC 2004):
    (a, b) with a/b == u (mod m), recognised by a single huge quotient in
    the Euclidean sequence. Unlike Wang's symmetric sqrt(m/2) bounds this
    recovers UNBALANCED fractions (our tensor coefficients are ~480-bit
    integers over tiny denominators) as soon as m exceeds |a|*b by the gap
    threshold -- roughly HALF the primes Wang needs. Returns None when no
    quotient stands out (need more primes)."""
    import gmpy2
    u %= m
    if u == 0:
        return (0, 1)
    r0, r1 = gmpy2.mpz(m), gmpy2.mpz(u)
    t0, t1 = gmpy2.mpz(0), gmpy2.mpz(1)
    best, gap = None, _MQ_GAP
    while r1:
        q = r0 // r1
        if q >= gap:
            best, gap = (r1, t1), q
        r0, r1 = r1, r0 - q * r1
        t0, t1 = t1, t0 - q * t1
    if best is None:
        return None
    a, b = best
    if b < 0:
        a, b = -a, -b
    if b == 0 or gmpy2.gcd(a, b) != 1 or gmpy2.gcd(b, m) != 1:
        return None
    return int(a), int(b)


def _lift(residue_stacks: list[np.ndarray], primes: list[int]):
    """CRT-combine per-prime residue tensors and rationally reconstruct
    every nonzero coefficient. Returns {flat_index: (a, b)} or None when any
    coefficient fails to reconstruct (need more primes)."""
    import gmpy2
    flats = [r.reshape(-1) for r in residue_stacks]
    nz = np.nonzero(np.any(np.stack(flats) != 0, axis=0))[0]
    M = gmpy2.mpz(1)
    for p in primes:
        M *= p
    # incremental CRT per coefficient
    out = {}
    for ix in nz:
        val, mod = gmpy2.mpz(int(flats[0][ix])), gmpy2.mpz(primes[0])
        for r, p in zip(flats[1:], primes[1:]):
            d = (int(r[ix]) - val) * gmpy2.invert(mod, p) % p
            val, mod = val + mod * d, mod * p
        rec = _ratrec(int(val), int(M))
        if rec is None:
            return None
        out[int(ix)] = rec
    return out


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
def solve_tensors(pay_den: dict, pay_num: dict, grids_pairs: list,
                  s_pairs: list, vinvs: list, progress=None, map_impl=None,
                  start_primes: int = 6, step_primes: int = 4,
                  max_primes: int = _MAX_PRIMES, batch: int | None = None,
                  info: dict | None = None):
    """The zp-batch pipeline. Returns (den_t, num_t) as {idx: [QQ]*L} dicts
    shaped exactly like the exact-QQ path's tensors, or raises ZpBatchError.

    Primes are added in rounds until BOTH lifts reconstruct; the result is
    then verified against ONE extra confirming prime (residues of the
    lifted rationals must match that prime's tensors) -- on mismatch the
    confirming prime simply joins the stack and the rounds continue. The
    caller's exact probe self-check remains the final authority.
    map_impl(func, iterable) lets the caller parallelise the per-prime
    work (one task per prime)."""
    axes_num = [list(s_pairs)] + [list(g) for g in grids_pairs]
    kept_lens = [len(g) for g in grids_pairs]
    L = len(s_pairs)
    shape = tuple(kept_lens) + (L,)

    run = map_impl if map_impl is not None else map
    if batch and batch > 1:
        # a round of W primes costs the same wall time as a round of 4 on a
        # W-worker pool, so never leave workers idle between lifts
        start_primes = max(start_primes, batch)
        step_primes = max(step_primes, batch)
    primes: list[int] = []
    stacks_d: list[np.ndarray] = []
    stacks_n: list[np.ndarray] = []
    want = start_primes
    while True:
        new = [q for q in _primes(want) if q not in primes]
        results = list(run(_prime_worker,
                           [(pay_den, pay_num, axes_num, vinvs, q)
                            for q in new]))
        for q, td, tn in results:
            if td is None:
                continue                          # unlucky prime, dropped
            primes.append(q)
            stacks_d.append(td)
            stacks_n.append(tn)
        if progress is not None:
            progress(len(primes), want)
        if len(primes) >= 2:
            lift_d = _lift(stacks_d, primes)
            lift_n = _lift(stacks_n, primes) if lift_d is not None else None
            if lift_n is not None:
                # one confirming prime instead of a whole agreeing round
                want += 1
                (q, td, tn), = list(run(
                    _prime_worker,
                    [(pay_den, pay_num, axes_num, vinvs,
                      _primes(want)[-1])]))
                if td is not None:
                    if (_confirm(lift_d, td, q)
                            and _confirm(lift_n, tn, q)):
                        if info is not None:
                            info.update(primes=len(primes) + 1)
                        return (_to_dict(lift_d, shape),
                                _to_dict(lift_n, shape))
                    primes.append(q)              # spurious lift: keep going
                    stacks_d.append(td)
                    stacks_n.append(tn)
        if want >= max_primes:
            raise ZpBatchError(
                f"no stable rational reconstruction after {want} primes")
        want = min(max_primes, want + step_primes)


def _confirm(lift: dict, tens: np.ndarray, q: int) -> bool:
    """Do the lifted rationals reduce to this prime's residues? Coefficients
    whose denominator vanishes mod q are inconclusive and skipped (the
    caller's exact probe check still guards them)."""
    flat = tens.reshape(-1)
    nz = set(np.nonzero(flat)[0].tolist())
    for ix, (a, b) in lift.items():
        if b % q == 0:
            nz.discard(ix)
            continue
        if a % q * pow(b % q, -1, q) % q != int(flat[ix]):
            return False
        nz.discard(ix)
    return not nz                # residue nonzero where the lift has no term


def _prime_worker(args):
    """One prime's residue tensors; picklable for pool dispatch."""
    pay_den, pay_num, axes_num, vinvs, p = args
    try:
        td, tn = _prime_tensors(pay_den, pay_num, axes_num, vinvs, p)
        return p, td, tn
    except ZeroDivisionError:                     # p divides a denominator
        return p, None, None


def _to_dict(lift: dict, shape: tuple):
    """{flat_index: (a, b)} -> {kept idx tuple: [QQ]*L} covering every idx."""
    kept_shape, L = shape[:-1], shape[-1]
    out = {}
    for idx in itertools.product(*(range(m) for m in kept_shape)):
        out[idx] = [QQ.zero] * L
    for flat, (a, b) in lift.items():
        full = np.unravel_index(flat, shape)
        idx, s_ix = tuple(int(v) for v in full[:-1]), int(full[-1])
        out[idx][s_ix] = QQ(a, b)
    return out
