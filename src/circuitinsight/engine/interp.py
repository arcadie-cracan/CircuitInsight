"""Multilinear-interpolation transfer-function solver.

Design: docs/multilinear-solver-plan.md. Every MNA stamp is rank-1 in its
element symbol, so det A has degree <= (stamp count) in each kept symbol.
Instead of one multivariate determinant over QQ[s, x1..xk], evaluate the
determinant on a small tensor grid of exact rational points and reconstruct
the exact polynomial by Lagrange interpolation. Exact arithmetic throughout —
reconstruction, not approximation.

Speed structure (profiled on tb_ota2s, 17x17):
- the s-degree is probed once so grid evaluations are pure-QQ determinants;
- matrix entries are decomposed ONCE into monomial form, so each evaluation
  builds a DomainMatrix over QQ directly (no per-point sympy xreplace or
  Matrix->DomainMatrix conversion);
- grid points are independent: evaluated in parallel worker processes when
  the grid is large enough to amortize process startup.

Resistors stamp 1/R, so the determinant is polynomial in the *conductance*:
kept 'r' symbols are interpolated in an auxiliary variable u = 1/R and mapped
back at assembly time.

A mandatory probabilistic self-check compares the reconstruction against
direct QQ[s] determinants at an off-grid probe point; on failure the solver
retries with exact QQ[s] evaluations before raising. It cannot return
silently wrong symbolics.
"""
from __future__ import annotations

import itertools
import os
import random
import warnings

import sympy as sp
from sympy import QQ
from sympy.polys.matrices import DomainMatrix

from .mna import MnaError, MnaSystem, S, TransferFunction, _det, hybrid_split

# cancel() cost guard: skip multivariate GCD on huge operands — downstream
# (validation, simplification, num_den) is correct on the uncancelled ratio
_CANCEL_OPS_LIMIT = 20000

# parallelize the grid when there are at least this many QQ determinants
_PARALLEL_MIN_DETS = 600
_MAX_WORKERS = 10


def _spawn_would_deadlock() -> bool:
    """True when a process-pool spawn would deadlock: a Qt binding is already
    imported (GUI/tests), so worker re-import hangs under 'spawn' (Windows/macOS)."""
    import multiprocessing as mp
    import sys

    if mp.get_start_method(allow_none=True) == "fork":
        return False                          # fork inherits state; no re-import
    return any(m in sys.modules for m in
               ("PySide6", "PySide2", "PyQt5", "PyQt6"))


def _lagrange(points: list, values: list, var: sp.Symbol) -> sp.Expr:
    total = sp.Integer(0)
    for i, (pi, vi) in enumerate(zip(points, values)):
        li = sp.Integer(1)
        for j, pj in enumerate(points):
            if j != i:
                li *= (var - pj) / (pi - pj)
        total += vi * li
    return sp.expand(total)


def _tensor_interpolate(grids, ivars, values, prefix=()):
    """Recursive tensor-product Lagrange over exact rational grids.
    values: dict[index tuple -> Expr]. (Fallback path only — the fast path
    reconstructs coefficient tensors with inverse-Vandermonde transforms.)"""
    axis = len(prefix)
    if axis == len(grids):
        return values[prefix]
    vals = [
        _tensor_interpolate(grids, ivars, values, prefix + (i,))
        for i in range(len(grids[axis]))
    ]
    return _lagrange(grids[axis], vals, ivars[axis])


# --------------------------- coefficient-tensor reconstruction (fast path)

def _vinv_rows(points: list) -> list[list]:
    """Rows of the inverse Vandermonde for the given rational points, as QQ
    elements: coeffs_j = sum_i rows[j][i] * value_i."""
    n = len(points)
    V = sp.Matrix([[pt**j for j in range(n)] for pt in points])
    Vinv = V.inv()
    return [[QQ.from_sympy(Vinv[j, i]) for i in range(n)] for j in range(n)]


def _matvec(rows, vals):
    out = []
    for row in rows:
        acc = QQ.zero
        for c, v in zip(row, vals):
            if c and v:
                acc = acc + c * v
        out.append(acc)
    return out


def _transform_axis(data: dict, rows: list, axis: int) -> dict:
    """Apply an inverse-Vandermonde transform along one tensor axis.
    data: dict[index tuple -> list[QQ]] (s-coefficient vectors)."""
    from collections import defaultdict

    groups: dict = defaultdict(dict)
    for key, vec in data.items():
        groups[key[:axis] + key[axis + 1:]][key[axis]] = vec
    L = len(next(iter(data.values())))
    out: dict = {}
    for rkey, sub in groups.items():
        for j, row in enumerate(rows):
            acc = [QQ.zero] * L
            nonzero = False
            for i, vec in sub.items():
                c = row[i]
                if not c:
                    continue
                for t in range(L):
                    v = vec[t]
                    if v:
                        acc[t] = acc[t] + c * v
                        nonzero = True
            if nonzero:
                out[rkey[:axis] + (j,) + rkey[axis:]] = acc
    return out


def _tensor_to_expr(tensor: dict, gens: list[sp.Symbol]) -> sp.Expr:
    """tensor: dict[kept-var exponent tuple -> s-coefficient vector]."""
    rep = {}
    for exps, vec in tensor.items():
        for es, c in enumerate(vec):
            if c:
                rep[(es,) + tuple(exps)] = sp.Rational(
                    int(c.numerator), int(c.denominator))
    if not rep:
        return sp.Integer(0)
    return sp.Poly(rep, *gens).as_expr()


def _tensor_check(tensor: dict, probe_qq: list, ref_expr: sp.Expr,
                  n_scoeffs: int) -> bool:
    """Evaluate the coefficient tensor at the probe point (kept vars only)
    and compare with the direct QQ[s] determinant's coefficients."""
    acc = [QQ.zero] * n_scoeffs
    for exps, vec in tensor.items():
        m = QQ.one
        for v, e in zip(probe_qq, exps):
            for _ in range(e):
                m = m * v
        for t in range(n_scoeffs):
            if vec[t]:
                acc[t] = acc[t] + m * vec[t]
    ref = [QQ.from_sympy(sp.Rational(c))
           for c in reversed(sp.Poly(ref_expr, S).all_coeffs())]
    ref += [QQ.zero] * (n_scoeffs - len(ref))
    return acc == ref


# ---------------------------------------------------------------- fast eval

def _decompose(M: sp.Matrix, gens: list[sp.Symbol]) -> dict:
    """Entries -> {(i, j): [(exps, (p, q)), ...]} monomial form over gens,
    with rational coefficients as plain int pairs (picklable)."""
    n = M.shape[0]
    entries = {}
    maxdeg = [0] * len(gens)
    for i in range(n):
        for j in range(n):
            e = M[i, j]
            if e == 0:
                continue
            terms = []
            for exps, c in sp.Poly(e, *gens, domain="QQ").as_dict().items():
                c = sp.Rational(c)
                terms.append((tuple(int(x) for x in exps),
                              (int(c.p), int(c.q))))
                for a, x in enumerate(exps):
                    maxdeg[a] = max(maxdeg[a], int(x))
            entries[(i, j)] = terms
    return {"n": n, "entries": entries, "maxdeg": maxdeg}


def _qq_det(payload: dict, vals: list) -> object:
    """Evaluate the decomposed matrix at QQ values (aligned with gens) and
    take its determinant over QQ."""
    n = payload["n"]
    # power cache per generator
    pows = []
    for a, v in enumerate(vals):
        row = [QQ.one]
        for _ in range(payload["maxdeg"][a]):
            row.append(row[-1] * v)
        pows.append(row)
    zero = QQ.zero
    rows = [[zero] * n for _ in range(n)]
    for (i, j), terms in payload["entries"].items():
        acc = zero
        for exps, (p, q) in terms:
            t = QQ(p, q)
            for a, x in enumerate(exps):
                if x:
                    t = t * pows[a][x]
            acc = acc + t
        rows[i][j] = acc
    return DomainMatrix(rows, (n, n), QQ).det()


# ------------------------------------------------------- worker process API

def _worker_chunk(args):
    """args: (pay_den, pay_num, s int-pairs, chunk of (idx, point int-pairs)).
    Returns per idx the det values across the s sweep for den and num, as
    int pairs. Payloads travel with each chunk so a persistent pool needs no
    per-solve initializer."""
    pay_den, pay_num, s_pairs, chunk = args
    s_vals = [QQ(p, q) for p, q in s_pairs]
    out = []
    for idx, pt_pairs in chunk:
        vals = [QQ(p, q) for p, q in pt_pairs]
        dd, nn = [], []
        for sv in s_vals:
            allv = [sv] + vals
            d = _qq_det(pay_den, allv)
            n_ = _qq_det(pay_num, allv)
            dd.append((int(d.numerator), int(d.denominator)))
            nn.append((int(n_.numerator), int(n_.denominator)))
        out.append((idx, dd, nn))
    return out


_POOL = None


def _get_pool(workers: int):
    """Lazy persistent process pool: worker startup (spawn + sympy import)
    is paid once per session, not once per solve."""
    global _POOL
    if _POOL is None:
        import atexit
        from concurrent.futures import ProcessPoolExecutor

        _POOL = ProcessPoolExecutor(max_workers=workers)
        atexit.register(_POOL.shutdown)
    return _POOL


# ---------------------------------------------------------------- solver

def solve_tf_interp(
    system: MnaSystem, output: str, keep: list[str], progress=None
) -> TransferFunction:
    """progress: optional callable(done, total) over grid points, so a long
    hybrid solve can report real progress instead of freezing a UI."""
    if isinstance(output, int):                 # raw unknown index: node OR
        col = output                            # branch current (loop gain)
        if not 0 <= col < system.A.rows:
            raise MnaError(f"output index {col} out of range")
    elif output not in system.node_index:
        raise MnaError(f"output node {output!r} not found (or it is ground)")
    else:
        col = system.node_index[output]

    subs, kept = hybrid_split(system, keep)
    if not kept:
        # all-numeric: a single direct QQ[s] determinant is already optimal
        from .mna import solve_tf

        return solve_tf(system, output, keep, method="direct")

    A = system.A.xreplace(subs)
    Ak = A.copy()
    Ak[:, col] = system.z

    syms = [system.symbols[n] for n in kept]
    degs = [max(1, system.stamp_counts.get(n, 1)) for n in kept]
    recip = [n in system.reciprocal for n in kept]
    # interpolation variables: the symbol itself, or u = 1/R for resistors
    ivars = [
        sp.Dummy(f"u_{n}") if r else system.symbols[n]
        for n, r in zip(kept, recip)
    ]
    # grid: distinct small rationals; reciprocal axes avoid u = 0
    grids = [
        [sp.Rational(v) for v in (range(1, d + 2) if r else range(0, d + 1))]
        for d, r in zip(degs, recip)
    ]

    def matrix_subs(point) -> dict:
        # point: tuple of grid values in interpolation-variable space
        m = {}
        for s_, r, v in zip(syms, recip, point):
            m[s_] = sp.Rational(1) / v if r else v
        return m

    # ------- probe the exact s-degree once, so every grid evaluation can be
    # a pure-QQ determinant (s on its own interpolation axis) --------------
    rng = random.Random(0xC1AC)
    probe = tuple(
        sp.Rational(int(g[-1]) + rng.randint(2, 17)) for g in grids
    )
    m_probe = matrix_subs(probe)
    d_probe = _det(A.xreplace(m_probe))
    n_probe = _det(Ak.xreplace(m_probe))
    s_deg = max(sp.degree(d_probe, S), sp.degree(n_probe, S))
    s_pts = [sp.Rational(v) for v in range(int(s_deg) + 2)]  # +1 margin

    index_ranges = [range(len(g)) for g in grids]
    all_idx = list(itertools.product(*index_ranges))

    # entries in monomial form over (u-mapped) generators, computed once
    u_map = {s_: 1 / iv for s_, iv, r in zip(syms, ivars, recip) if r}
    gens = [S] + list(ivars)
    pay_den = _decompose(A.xreplace(u_map) if u_map else A, gens)
    pay_num = _decompose(Ak.xreplace(u_map) if u_map else Ak, gens)
    s_pairs = [(int(v.p), int(v.q)) for v in s_pts]

    def point_pairs(idx):
        return tuple(
            (int(grids[a][i].p), int(grids[a][i].q))
            for a, i in enumerate(idx)
        )

    def gather() -> list:
        tasks = [(idx, point_pairs(idx)) for idx in all_idx]
        n_dets = len(tasks) * len(s_pts) * 2
        workers = min(_MAX_WORKERS, max(1, (os.cpu_count() or 2) - 1))
        # A ProcessPoolExecutor started from a process that has already imported
        # a Qt binding deadlocks under the 'spawn' start method (the GUI/tests
        # path): the worker re-import hangs, and pool.map() blocks forever rather
        # than raising, so the serial fallback below never fires. Run serially in
        # that case -- scripts (no Qt) keep the parallel speedup.
        if _spawn_would_deadlock():
            workers = 1

        # Grid evaluation is the whole cost of a hybrid solve and its size is
        # known up front, so progress here is real, not a spinner. Reported per
        # chunk (both paths chunk the same way), which keeps the callback cheap.
        done = 0

        def tick(n):
            nonlocal done
            done += n
            if progress is not None:
                progress(done, len(tasks))

        if n_dets >= _PARALLEL_MIN_DETS and workers > 1:
            try:
                chunk_sz = max(1, len(tasks) // (workers * 4))
                args = [
                    (pay_den, pay_num, s_pairs, tasks[i:i + chunk_sz])
                    for i in range(0, len(tasks), chunk_sz)
                ]
                results = []
                for part in _get_pool(workers).map(_worker_chunk, args):
                    results.extend(part)
                    tick(len(part))
                return results
            except Exception as exc:            # pragma: no cover
                warnings.warn(f"parallel evaluation unavailable ({exc}); "
                              f"running sequentially")
        # serial: chunk it too, so the caller still sees movement
        results = []
        chunk_sz = max(1, len(tasks) // 20)
        for i in range(0, len(tasks), chunk_sz):
            part = _worker_chunk(
                (pay_den, pay_num, s_pairs, tasks[i:i + chunk_sz]))
            results.extend(part)
            tick(len(part))
        return results

    def evaluate_qqs() -> tuple[sp.Expr, sp.Expr]:
        # exact QQ[s] fallback (no s-grid), sympy path
        n_vals, d_vals = {}, {}
        for idx in all_idx:
            m = matrix_subs(tuple(grids[a][i] for a, i in enumerate(idx)))
            d_vals[idx] = _det(A.xreplace(m))
            n_vals[idx] = _det(Ak.xreplace(m))
        return (_tensor_interpolate(grids, ivars, n_vals),
                _tensor_interpolate(grids, ivars, d_vals))

    def self_check(num, den) -> bool:
        assign = dict(zip(ivars, probe))
        return (sp.expand(num.xreplace(assign) - n_probe) == 0
                and sp.expand(den.xreplace(assign) - d_probe) == 0)

    # ---- fast path: coefficient tensors via inverse-Vandermonde transforms
    raw = gather()
    L = len(s_pts)
    svinv = _vinv_rows(s_pts)
    den_t = {idx: _matvec(svinv, [QQ(p, q) for p, q in dd])
             for idx, dd, _ in raw}
    num_t = {idx: _matvec(svinv, [QQ(p, q) for p, q in nn])
             for idx, _, nn in raw}
    for a, g in enumerate(grids):
        rows = _vinv_rows(g)
        den_t = _transform_axis(den_t, rows, a)
        num_t = _transform_axis(num_t, rows, a)

    probe_qq = [QQ.from_sympy(v) for v in probe]
    if (_tensor_check(den_t, probe_qq, d_probe, L)
            and _tensor_check(num_t, probe_qq, n_probe, L)):
        num = _tensor_to_expr(num_t, gens)
        den = _tensor_to_expr(den_t, gens)
    else:
        # s-degree probe was unlucky (coefficient vanished at the probe
        # point): redo with exact QQ[s] determinants per grid point
        warnings.warn("multilinear s-grid self-check failed; retrying with "
                      "direct QQ[s] evaluations")
        num, den = evaluate_qqs()
        if not self_check(num, den):
            raise MnaError(
                "multilinear self-check failed: degree bookkeeping bug — "
                "please report (method='direct' will work)"
            )
    if den == 0:
        raise MnaError("singular MNA matrix: floating node or short loop?")

    # map reciprocal interpolation variables back: u -> 1/R
    back = {
        iv: sp.Integer(1) / system.symbols[n]
        for iv, n, r in zip(ivars, kept, recip)
        if r
    }
    if back:
        num = sp.together(num.xreplace(back))
        den = sp.together(den.xreplace(back))

    expr = num / den
    if sp.count_ops(num) + sp.count_ops(den) <= _CANCEL_OPS_LIMIT:
        expr = sp.cancel(expr)
    return TransferFunction(
        expr=expr, values=dict(system.values), symbols=dict(system.symbols)
    )
