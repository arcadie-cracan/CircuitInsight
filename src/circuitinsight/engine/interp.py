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
from collections import OrderedDict, deque
import random
import time
import warnings

import sympy as sp
from sympy import QQ
from sympy.polys.matrices import DomainMatrix

from .mna import MnaError, MnaSystem, S, TransferFunction, _det, hybrid_split

# cancel() cost guard: skip multivariate GCD on huge operands — downstream
# (validation, simplification, num_den) is correct on the uncancelled ratio
_CANCEL_OPS_LIMIT = 20000

# probe backend: "auto" (the S-D selector, see resolve_backend: the
# exact-QQ grid below the det-count crossover, the S-B sparse backend
# at or above it; the zp-batch DENSE grid stays out of auto -- it was
# MEASURED SLOWER at every k, ~930-bit tensor coefficients need 31+
# CRT primes and 31 x 47 us/det loses to one 0.5 ms gmpy2 det; see
# docs/interp-speedup-plan.md S-A results), "zp" (force the zp-batch dense
# mod-p backend), "bot" (force the S-B Ben-Or/Tiwari + Kronecker SPARSE
# backend: probes scale with the true support T instead of the grid size),
# "ratfun" (force the S-C REDUCED-pair reconstruction: interpolates the
# cancelled transfer function directly from det-ratio values -- fewer
# terms, roughly half the CRT primes; accepted through a ratio identity
# at the exact probe), or "qq" (exact-QQ grid only). The mod-p backends
# are correctness-guarded with the QQ grid as fallback. Overridable via
# the environment; tests set the module attribute directly.
PROBE_BACKEND = None


def _probe_backend() -> str:
    if PROBE_BACKEND is not None:
        return PROBE_BACKEND
    return os.environ.get("CIN_PROBE_BACKEND", "auto")


# auto-selector crossover (S-D), in DENSE determinant count K*L*2: below
# it the exact-QQ grid wins (bot is only at parity there and carries
# failure modes); above it bot's support-scaled probing pulls ahead
# steeply (measured: parity at ~26k dets, 7x at ~1.5M). ~50k dets is
# roughly a 7 s QQ solve on the calibration machine.
_AUTO_BOT_MIN_DETS = 50_000


def default_workers() -> int:
    """Pool size for the dense-grid paths: all cores but one, capped.
    (The numeric-s path's _worker_cap reserves TWO cores for the GUI;
    these grid pools measured best leaving one.)"""
    return min(_MAX_WORKERS, max(1, (os.cpu_count() or 2) - 1))


def resolve_backend(n_dense_dets: int) -> str:
    """The S-D auto selector, THE single source of truth also mirrored by
    analysis.estimate: explicit setting wins; 'auto' picks the exact-QQ
    grid below the crossover and the S-B sparse backend above it.

    Deliberately NOT an input: coefficient spread. A spread gate was
    tried against a real fallback ("no stable rational reconstruction
    after 64 primes") and MEASURED WRONG -- the failing call's spread
    was 11.0, below the 16.1 of a bench that reconstructs fine.
    Reconstruction size tracks the determinant's coefficients (products
    of ~n entries), not the entries' dynamic range, so the predictor
    has to come from somewhere else."""
    b = _probe_backend()
    if b != "auto":
        return b
    return "bot" if n_dense_dets >= _AUTO_BOT_MIN_DETS else "qq"


# telemetry of the most recent solve_tf_interp call (S-D): backend
# decisions, sizes, wall time, whether a mod-p backend fell back. Read by
# tests, the GUI, and future selector re-tuning; never consumed by the
# solver itself.
LAST_SOLVE: dict | None = None


def _record_solve(*, backend, backend_requested, fell_back, reduced,
                  batch, grid_K, L, n_dense_dets, wall_s,
                  fallback_reason=None, **binfo) -> dict:
    """Build LAST_SOLVE with ONE schema. The two solve paths used to
    assemble it by hand with different key sets (the numeric-s path
    omitted fallback_reason and the backend info), so every consumer
    had to read it defensively; now absent values are present-but-None
    and backend extras ride along explicitly."""
    global LAST_SOLVE
    LAST_SOLVE = {
        "backend": backend, "backend_requested": backend_requested,
        "fell_back": fell_back, "fallback_reason": fallback_reason,
        "reduced": reduced, "batch": batch, "grid_K": grid_K, "L": L,
        "n_dense_dets": n_dense_dets, "wall_s": wall_s, **binfo,
    }
    return LAST_SOLVE


# parallelize the grid when there are at least this many QQ determinants
_PARALLEL_MIN_DETS = 600


def _worker_cap() -> int:
    """How many processes the mod-p prime rounds may use.

    MEASURED (fc, 8 kept symbols, bot backend, 101 primes): 5039 s of worker
    CPU against 211 s in the parent -- the solve is ~96% parallel, so this
    cap, not the serial reconstruction, sets the wall time. The former fixed
    10 left 22 of a 32-thread host idle. Scale with the machine instead,
    keeping two cores for the GUI; CIRCUITINSIGHT_WORKERS overrides."""
    env = os.environ.get("CIRCUITINSIGHT_WORKERS", "").strip()
    if env.isdigit() and int(env) > 0:
        return int(env)
    return max(1, (os.cpu_count() or 4) - 2)


#: resolved once at import; the call sites still clamp to the local host, so
#: tests may monkeypatch this to force a serial or narrow pool.
_MAX_WORKERS = _worker_cap()


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

    if not data:
        return {}                # zero polynomial (e.g. a nulled numerator)
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


def _tensor_at_probe(tensor: dict, probe_qq: list,
                     n_scoeffs: int) -> list:
    """Evaluate the coefficient tensor at the probe point (kept vars only):
    the s-coefficient vector of the tensor's polynomial there."""
    acc = [QQ.zero] * n_scoeffs
    for exps, vec in tensor.items():
        m = QQ.one
        for v, e in zip(probe_qq, exps):
            for _ in range(e):
                m = m * v
        for t in range(n_scoeffs):
            if vec[t]:
                acc[t] = acc[t] + m * vec[t]
    return acc


def _scoeffs(ref_expr: sp.Expr, n_scoeffs: int) -> list:
    ref = [QQ.from_sympy(sp.Rational(c))
           for c in reversed(sp.Poly(ref_expr, S).all_coeffs())]
    return ref + [QQ.zero] * (n_scoeffs - len(ref))


def _tensor_check(tensor: dict, probe_qq: list, ref_expr: sp.Expr,
                  n_scoeffs: int) -> bool:
    """Evaluate the coefficient tensor at the probe point (kept vars only)
    and compare with the direct QQ[s] determinant's coefficients."""
    return _tensor_at_probe(tensor, probe_qq, n_scoeffs) == \
        _scoeffs(ref_expr, n_scoeffs)


def _ratio_check(num_t: dict, den_t: dict, probe_qq: list,
                 n_probe: sp.Expr, d_probe: sp.Expr,
                 n_scoeffs: int) -> bool:
    """Acceptance for REDUCED tensors (S-C): the raw tensor check cannot
    apply (content differs), but the cross-multiplied ratio identity in s
    at the exact probe point can: N_red(probe) * d_probe must equal
    D_red(probe) * n_probe as polynomials in s."""
    nv = _tensor_at_probe(num_t, probe_qq, n_scoeffs)
    dv = _tensor_at_probe(den_t, probe_qq, n_scoeffs)
    if not any(dv):
        return False
    nref = _scoeffs(n_probe, n_scoeffs)
    dref = _scoeffs(d_probe, n_scoeffs)

    def conv(a, b):
        out = [QQ.zero] * (2 * n_scoeffs - 1)
        for i, x in enumerate(a):
            if x:
                for j, y in enumerate(b):
                    if y:
                        out[i + j] = out[i + j] + x * y
        return out

    return conv(nv, dref) == conv(dv, nref)


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
    """args: (pay_den, [pay_num...], s int-pairs, chunk of (idx, point
    int-pairs)). Returns per idx the det values across the s sweep for den
    and EACH numerator, as int pairs — the multi-output (Tan-Shi cofactor)
    form: one denominator evaluation is shared by every output. Payloads
    travel with each chunk so a persistent pool needs no per-solve
    initializer."""
    pay_den, pay_nums, s_pairs, chunk = args
    s_vals = [QQ(p, q) for p, q in s_pairs]
    out = []
    for idx, pt_pairs in chunk:
        vals = [QQ(p, q) for p, q in pt_pairs]
        dd = []
        nns = [[] for _ in pay_nums]
        for sv in s_vals:
            allv = [sv] + vals
            d = _qq_det(pay_den, allv)
            dd.append((int(d.numerator), int(d.denominator)))
            for nn, pay in zip(nns, pay_nums):
                n_ = _qq_det(pay, allv)
                nn.append((int(n_.numerator), int(n_.denominator)))
        out.append((idx, dd, nns))
    return out


def _s_degree_bound(M: sp.Matrix) -> int:
    """An upper bound on the s-degree of det(M), computed structurally.

    Every MNA stamp is either a constant or s*symbol (conductances,
    transconductances and source rows are constant; capacitors,
    trans-capacitances and inductors contribute exactly one factor of s),
    so each entry is AFFINE in s. In the Leibniz expansion every term
    takes one entry per row, so the degree cannot exceed the number of
    rows that carry an s term at all. Rows for nodes with no capacitance
    -- and every source/branch constraint row -- contribute nothing.

    This is what lets an all-numeric solve skip the probe determinant: the
    probe exists to learn the s-degree, and it costs exactly the symbolic
    determinant we are trying to avoid."""
    rows = 0
    for i in range(M.shape[0]):
        if any(M[i, j].has(S) for j in range(M.shape[1])):
            rows += 1
    return rows


#: measured all-numeric solves, newest last: {"wall_s", "det_s"} where
#: det_s = t_det x total determinant count. The estimator reads this to
#: self-calibrate its overhead ratio instead of trusting two frozen
#: constants (the "empirical, re-check if the reconstruction changes"
#: debt, now self-servicing in-process).
NUMERIC_S_HISTORY: deque = deque(maxlen=8)


#: dets (den + numerators) above which the s-sweep fans out to the worker
#: pool. Deliberately below _PARALLEL_MIN_DETS: an s-sweep determinant is
#: a whole matrix at one point (ms), not one grid entry (us).
_S_SWEEP_PARALLEL_MIN = 200


def _worker_s_chunk(args):
    """args: (pay_den, [pay_num...], chunk of s int-pairs). Returns
    (den values, per-task numerator values) across the chunk, as int
    pairs. Payload-carrying, so the persistent pool needs no per-solve
    initializer."""
    pay_den, pay_nums, s_pairs = args
    dd, nns = [], [[] for _ in pay_nums]
    for pq in s_pairs:
        sv = [QQ(pq[0], pq[1])]
        d = _qq_det(pay_den, sv)
        dd.append((int(d.numerator), int(d.denominator)))
        for nn, pay in zip(nns, pay_nums):
            n_ = _qq_det(pay, sv)
            nn.append((int(n_.numerator), int(n_.denominator)))
    return dd, nns


#: det A(s) samples, keyed by the matrix itself. The ESTIMATOR and the
#: SOLVER want the same numbers: estimate._probe evaluates det A at integer
#: s to read the s-degree off finite differences and to time one
#: determinant, and the all-numeric solve evaluates det A at integer s to
#: reconstruct D(s). Without this, asking "how long will it take?" costs
#: about as much as answering it by solving. Bounded: these are lists of
#: multi-hundred-digit rationals.
_S_CACHE: "OrderedDict[sp.ImmutableMatrix, tuple]" = OrderedDict()
_S_CACHE_MAX = 4


def s_sweep_det(A: sp.Matrix, n: int, tick=None) -> tuple[list, float]:
    """det A(s) at s = 0..n-1 over QQ, and the median seconds per
    determinant. `tick()` is called once per FRESHLY computed sample (never
    for a cache hit), so a caller can report progress and cancel.

    A longer cached sweep satisfies a shorter request: the estimator probes
    dim+2 points and the solve needs only (degree bound + 1) <= dim+1, so
    an estimate immediately followed by a solve pays the denominator once."""
    key = sp.ImmutableMatrix(A)
    hit = _S_CACHE.get(key)
    if hit is not None and len(hit[0]) >= n:
        _S_CACHE.move_to_end(key)
        return hit[0][:n], hit[1]

    vals = list(hit[0]) if hit is not None else []
    times = []
    payload = _decompose(A, [S])
    for k in range(len(vals), n):
        t0 = time.perf_counter()
        vals.append(_qq_det(payload, [QQ(k, 1)]))
        times.append(time.perf_counter() - t0)
        if tick is not None:
            tick()
    times.sort()
    t_det = times[len(times) // 2] if times else (hit[1] if hit else 0.0)
    _S_CACHE[key] = (vals, t_det)
    _S_CACHE.move_to_end(key)
    while len(_S_CACHE) > _S_CACHE_MAX:
        _S_CACHE.popitem(last=False)
    return vals[:n], t_det


def _num_sweep(payload, n: int, tick=None) -> list:
    """A numerator's determinant at the same s = 0..n-1. Not cached: the
    matrix differs per output, and unlike the denominator it is never
    wanted twice."""
    out = []
    for k in range(n):
        out.append(_qq_det(payload, [QQ(k, 1)]))
        if tick is not None:
            tick()
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


def _solve_numeric_s(tasks, cols, A, progress=None) -> list:
    """All-numeric solve: sweep s, reconstruct N(s) and D(s) exactly.

    The alternative -- one symbolic determinant over QQ[s] -- was taken
    here for years on the claim that it "is already optimal". It is not:
    a QQ[s] determinant carries polynomial arithmetic through every
    elimination step, while a determinant at a NUMERIC s is plain rational
    arithmetic. Measured: 23.3 s -> 0.69 s on the uA741 (30x30), 12.5 s ->
    0.81 s on a 36-device folded cascode, both reconstructing bit-identical
    polynomials.

    The reconstruction is exact rather than approximate: with each entry
    affine in s (see _s_degree_bound), deg det <= B, and a polynomial of
    degree <= B is determined by B+1 samples. No probe, no self-check --
    the bound is structural, so there is no unlucky case to guard against.

    And it is what makes the solve REPORTABLE: the sweep is B+1 units of
    known size, so a caller sees real progress and can cancel between
    determinants, neither of which is possible inside sympy's det()."""
    t0 = time.perf_counter()
    sys0 = tasks[0][0]
    Aks = []
    for (system, _), col in zip(tasks, cols):
        Ak = A.copy()
        Ak[:, col] = system.z
        Aks.append(Ak)

    # B+1 samples pin a degree-<=B polynomial; the numerators share the
    # bound (replacing a column by z cannot add an s row).
    L = _s_degree_bound(A) + 1
    s_pts = [sp.Rational(v) for v in range(L)]

    total = L * (1 + len(tasks))
    done = 0

    def tick():
        nonlocal done
        done += 1
        if progress is not None:
            progress(min(done, total), total)

    key = sp.ImmutableMatrix(A)
    cached = _S_CACHE.get(key)
    workers = default_workers()
    if (total >= _S_SWEEP_PARALLEL_MIN and workers > 1
            and not _spawn_would_deadlock()
            and (cached is None or len(cached[0]) < L)):
        # big sweep, nothing cached: fan the s points across the pool.
        # Chunk-level ticks keep progress and cancellation real.
        t_par = time.perf_counter()
        pay_den = _decompose(A, [S])
        pay_nums = [_decompose(Ak, [S]) for Ak in Aks]
        s_pairs = [(int(v.p), int(v.q)) for v in s_pts]
        chunk_sz = max(1, L // (workers * 4))
        argsets = [(pay_den, pay_nums, s_pairs[i:i + chunk_sz])
                   for i in range(0, L, chunk_sz)]
        dvals, nvals = [], [[] for _ in Aks]
        for dd, nns in _get_pool(workers).map(_worker_s_chunk, argsets):
            dvals.extend(QQ(p_, q_) for p_, q_ in dd)
            for acc, nn in zip(nvals, nns):
                acc.extend(QQ(p_, q_) for p_, q_ in nn)
            done += len(dd) * (1 + len(Aks))
            if progress is not None:
                progress(min(done, total), total)
        t_det = (time.perf_counter() - t_par) / max(1, total)
        _S_CACHE[key] = (dvals, t_det)          # estimator reuse
        _S_CACHE.move_to_end(key)
        while len(_S_CACHE) > _S_CACHE_MAX:
            _S_CACHE.popitem(last=False)
    else:
        # the denominator comes from the shared cache: if the caller asked
        # for an estimate first, its probe already computed these
        dvals, _t = s_sweep_det(A, L, tick=tick)
        if done < L:                      # cache hit: account for it at once
            done = L
            if progress is not None:
                progress(done, total)
        nvals = [_num_sweep(_decompose(Ak, [S]), L, tick=tick)
                 for Ak in Aks]

    svinv = _vinv_rows(s_pts)
    den_c = _matvec(svinv, dvals)
    num_cs = [_matvec(svinv, nv) for nv in nvals]
    den = _tensor_to_expr({(): den_c}, [S])
    if den == 0:
        raise MnaError("singular MNA matrix: floating node or short loop?")

    wall = time.perf_counter() - t0
    _record_solve(backend="qq-s", backend_requested="qq-s",
                  fell_back=False, reduced=False, batch=len(tasks),
                  grid_K=1, L=L, n_dense_dets=total, wall_s=wall)
    t_det_now = _S_CACHE.get(key, (None, 0.0))[1]
    if t_det_now:
        NUMERIC_S_HISTORY.append({"wall_s": wall,
                                  "det_s": t_det_now * total})
    out = []
    for (system, _), num_c in zip(tasks, num_cs):
        num = _tensor_to_expr({(): num_c}, [S])
        expr = num / den
        if sp.count_ops(num) + sp.count_ops(den) <= _CANCEL_OPS_LIMIT:
            expr = sp.cancel(expr)
        out.append(TransferFunction(
            expr=expr, values=dict(system.values),
            symbols=dict(system.symbols)))
    return out


# ---------------------------------------------------------------- solver

def solve_tf_interp(
    system: MnaSystem, output: str, keep: list[str], progress=None
) -> TransferFunction:
    """progress: optional callable(done, total) over grid points, so a long
    hybrid solve can report real progress instead of freezing a UI."""
    return solve_tf_interp_batch([(system, output)], keep,
                                 progress=progress)[0]


def solve_tf_interp_batch(tasks, keep: list[str],
                          progress=None) -> list[TransferFunction]:
    """Multi-output interpolation solve: `tasks` is a list of
    (system, output) pairs whose systems all share the SAME matrix A
    (only the excitation vector z and the output unknown differ — the
    Tan-Shi hierarchical-suppression shape: several cofactors over one
    determinant). The grid is walked ONCE, the denominator is evaluated
    ONCE per point, and each task adds only its numerator determinant —
    so m outputs cost (1+m)/2m of m separate solves, plus a single s-probe
    and one tensor-transform pass per output.

    The caller (mna.solve_tf_batch) validates the shared-A precondition.
    With more than one task the mod-p backends are bypassed (the shared
    QQ grid is the point; batch users sit below the bot crossover), so
    a forced CIN_PROBE_BACKEND applies to single solves only."""
    sys0 = tasks[0][0]
    cols = []
    for system, output in tasks:
        if isinstance(output, int):             # raw unknown index: node OR
            col = output                        # branch current (loop gain)
            if not 0 <= col < system.A.rows:
                raise MnaError(f"output index {col} out of range")
        elif output not in system.node_index:
            raise MnaError(
                f"output node {output!r} not found (or it is ground)")
        else:
            col = system.node_index[output]
        cols.append(col)

    subs, kept = hybrid_split(sys0, keep)
    A = sys0.A.xreplace(subs)
    if not kept:
        return _solve_numeric_s(tasks, cols, A, progress)

    Aks = []
    for (system, _), col in zip(tasks, cols):
        Ak = A.copy()
        Ak[:, col] = system.z
        Aks.append(Ak)

    syms = [sys0.symbols[n] for n in kept]
    degs = [max(1, sys0.stamp_counts.get(n, 1)) for n in kept]
    recip = [n in sys0.reciprocal for n in kept]
    # interpolation variables: the symbol itself, or u = 1/R for resistors
    ivars = [
        sp.Dummy(f"u_{n}") if r else sys0.symbols[n]
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
    n_probes = [_det(Ak.xreplace(m_probe)) for Ak in Aks]
    s_deg = max([sp.degree(d_probe, S)]
                + [sp.degree(n_p, S) for n_p in n_probes])
    s_pts = [sp.Rational(v) for v in range(int(s_deg) + 2)]  # +1 margin

    index_ranges = [range(len(g)) for g in grids]
    all_idx = list(itertools.product(*index_ranges))

    # entries in monomial form over (u-mapped) generators, computed once
    u_map = {s_: 1 / iv for s_, iv, r in zip(syms, ivars, recip) if r}
    gens = [S] + list(ivars)
    pay_den = _decompose(A.xreplace(u_map) if u_map else A, gens)
    pay_nums = [_decompose(Ak.xreplace(u_map) if u_map else Ak, gens)
                for Ak in Aks]
    s_pairs = [(int(v.p), int(v.q)) for v in s_pts]

    def point_pairs(idx):
        return tuple(
            (int(grids[a][i].p), int(grids[a][i].q))
            for a, i in enumerate(idx)
        )

    def gather() -> list:
        gtasks = [(idx, point_pairs(idx)) for idx in all_idx]
        n_dets = len(gtasks) * len(s_pts) * (1 + len(tasks))
        workers = default_workers()
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
                progress(done, len(gtasks))

        if n_dets >= _PARALLEL_MIN_DETS and workers > 1:
            try:
                chunk_sz = max(1, len(gtasks) // (workers * 4))
                args = [
                    (pay_den, pay_nums, s_pairs, gtasks[i:i + chunk_sz])
                    for i in range(0, len(gtasks), chunk_sz)
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
        chunk_sz = max(1, len(gtasks) // 20)
        for i in range(0, len(gtasks), chunk_sz):
            part = _worker_chunk(
                (pay_den, pay_nums, s_pairs, gtasks[i:i + chunk_sz]))
            results.extend(part)
            tick(len(part))
        return results

    def evaluate_qqs(Ak) -> tuple[sp.Expr, sp.Expr]:
        # exact QQ[s] fallback (no s-grid), sympy path
        n_vals, d_vals = {}, {}
        for idx in all_idx:
            m = matrix_subs(tuple(grids[a][i] for a, i in enumerate(idx)))
            d_vals[idx] = _det(A.xreplace(m))
            n_vals[idx] = _det(Ak.xreplace(m))
        return (_tensor_interpolate(grids, ivars, n_vals),
                _tensor_interpolate(grids, ivars, d_vals))

    def self_check(num, den, n_probe) -> bool:
        assign = dict(zip(ivars, probe))
        return (sp.expand(num.xreplace(assign) - n_probe) == 0
                and sp.expand(den.xreplace(assign) - d_probe) == 0)

    # ---- fast path: coefficient tensors via inverse-Vandermonde transforms
    L = len(s_pts)
    svinv = _vinv_rows(s_pts)
    probe_qq = [QQ.from_sympy(v) for v in probe]

    # zp-batch backend (S-A): batched mod-p grids + CRT lift. Its result is
    # only ever ACCEPTED through the same exact probe self-check the QQ path
    # uses, so an unlucky prime or a spurious rational reconstruction cannot
    # produce a wrong answer -- it produces a fallback to the QQ grid.
    # Single-output solves only ("zp"/"bot"/"ratfun"): a batch's whole point
    # is the shared QQ-grid walk.
    t_solve0 = time.perf_counter()
    grid_K = 1
    for g in grids:
        grid_K *= len(g)
    n_dense = grid_K * L * 2
    den_t = None
    num_ts = [None] * len(tasks)
    reduced_ok = False
    fell_back = False
    fb_reason = None          # WHY, not just that
    backend = resolve_backend(n_dense) if len(tasks) == 1 else "qq"
    binfo: dict = {}
    if backend in ("zp", "bot", "ratfun") and grids:
        pay_num = pay_nums[0]
        try:
            grid_pairs = [[(int(v.p), int(v.q)) for v in g] for g in grids]
            vinvs = [svinv] + [_vinv_rows(g) for g in grids]
            workers = default_workers()
            zmap = None
            if workers > 1 and not _spawn_would_deadlock():
                zmap = _get_pool(workers).map
            if backend == "ratfun":
                from . import ratfun
                den_t, num_t = ratfun.solve_tensors_ratfun(
                    pay_den, pay_num, grid_pairs, s_pairs, vinvs,
                    progress=progress, map_impl=zmap,
                    batch=workers if zmap is not None else None,
                    info=binfo)
            elif backend == "bot":
                from . import botsparse
                den_t, num_t = botsparse.solve_tensors_sparse(
                    pay_den, pay_num, grid_pairs, s_pairs, vinvs,
                    progress=progress, map_impl=zmap,
                    batch=workers if zmap is not None else None,
                    info=binfo)
            else:
                from . import zpbatch
                den_t, num_t = zpbatch.solve_tensors(
                    pay_den, pay_num, grid_pairs, s_pairs, vinvs,
                    progress=progress, map_impl=zmap,
                    batch=workers if zmap is not None else None,
                    info=binfo)
            num_ts = [num_t]
        except Exception as exc:
            warnings.warn(f"{backend} backend unavailable ({exc}); "
                          f"using the exact-QQ grid")
            den_t = None
            num_ts = [None]
            fell_back = True
            fb_reason = f"{backend}: {type(exc).__name__}: {exc}"
        if den_t is not None and backend == "ratfun":
            # reduced tensors: the raw check cannot apply -- accept through
            # the cross-multiplied ratio identity at the exact probe
            if _ratio_check(num_ts[0], den_t, probe_qq, n_probes[0],
                            d_probe, L):
                reduced_ok = True
            else:
                warnings.warn(f"{backend} reduced result failed the ratio "
                              f"check; using the exact-QQ grid")
                den_t = None
                num_ts = [None]
                fell_back = True
                fb_reason = f"{backend}: reduced ratio check failed"
        elif den_t is not None and not (
                _tensor_check(den_t, probe_qq, d_probe, L)
                and _tensor_check(num_ts[0], probe_qq, n_probes[0], L)):
            warnings.warn(f"{backend} result failed the exact probe "
                          f"self-check; using the exact-QQ grid")
            den_t = None
            num_ts = [None]
            fell_back = True
            fb_reason = f"{backend}: probe self-check failed"

    if den_t is None:
        # gather()'s last tick is already done == total: the caller's one
        # evaluation-complete signal on this path.
        raw = gather()
        den_t = {idx: _matvec(svinv, [QQ(p, q) for p, q in dd])
                 for idx, dd, _ in raw}
        num_ts = [{idx: _matvec(svinv, [QQ(p, q) for p, q in nns[i]])
                   for idx, _, nns in raw}
                  for i in range(len(tasks))]
        for a, g in enumerate(grids):
            rows = _vinv_rows(g)
            den_t = _transform_axis(den_t, rows, a)
            num_ts = [_transform_axis(nt, rows, a) for nt in num_ts]
    elif progress is not None:
        # the mod-p backends never report done == total themselves (their
        # prime rounds continue past any one round), so THIS is the caller's
        # one evaluation-complete signal on the backend path
        progress(1, 1)

    den_ok = reduced_ok or _tensor_check(den_t, probe_qq, d_probe, L)
    den = _tensor_to_expr(den_t, gens) if den_ok else None
    nums = []
    retried_den = None
    for i, (num_t, n_probe) in enumerate(zip(num_ts, n_probes)):
        if den_ok and (reduced_ok
                       or _tensor_check(num_t, probe_qq, n_probe, L)):
            nums.append(_tensor_to_expr(num_t, gens))
            continue
        # s-degree probe was unlucky (coefficient vanished at the probe
        # point): redo with exact QQ[s] determinants per grid point
        warnings.warn("multilinear s-grid self-check failed; retrying with "
                      "direct QQ[s] evaluations")
        num_i, den_i = evaluate_qqs(Aks[i])
        if not self_check(num_i, den_i, n_probe):
            raise MnaError(
                "multilinear self-check failed: degree bookkeeping bug — "
                "please report (method='direct' will work)"
            )
        nums.append(num_i)
        retried_den = den_i
    if den is None:
        den = retried_den
    if den == 0:
        raise MnaError("singular MNA matrix: floating node or short loop?")

    # map reciprocal interpolation variables back: u -> 1/R
    back = {
        iv: sp.Integer(1) / sys0.symbols[n]
        for iv, n, r in zip(ivars, kept, recip)
        if r
    }
    if back:
        den = sp.together(den.xreplace(back))
        nums = [sp.together(n_.xreplace(back)) for n_ in nums]

    _record_solve(backend=backend if not fell_back else "qq",
                  backend_requested=backend, fell_back=fell_back,
                  fallback_reason=fb_reason, reduced=reduced_ok,
                  batch=len(tasks), grid_K=grid_K, L=L,
                  n_dense_dets=n_dense,
                  wall_s=time.perf_counter() - t_solve0, **binfo)
    out = []
    for (system, _), num in zip(tasks, nums):
        expr = num / den
        if sp.count_ops(num) + sp.count_ops(den) <= _CANCEL_OPS_LIMIT:
            expr = sp.cancel(expr)
        out.append(TransferFunction(
            expr=expr, values=dict(system.values),
            symbols=dict(system.symbols)))
    return out
