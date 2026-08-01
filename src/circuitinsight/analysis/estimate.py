"""Symbolic-solve time estimation and budget-driven keep-set planning.

The interpolation solver evaluates the network determinant on a tensor
grid: its work is

    n_dets = 2 * (s_degree + 2) * PROD_(kept symbols) (stamp_count + 1)

fully-numeric determinants (engine/interp.py). The grid product is the
dominant, keep-set-controlled factor and is known instantly from the
stamp counts; s_degree and the per-determinant time (t_det) are measured
cheaply with a handful of numeric determinants (never the symbolic
characteristic polynomial). Wall-clock is then

    seconds = alpha * (t_det * n_dets) + beta

with (alpha, beta) a machine calibration from calibrate() -- fitted
separately for the serial and parallel interp paths. t_det carries the
per-circuit determinant cost; alpha the reconstruction/cancellation tail.
The estimate is order-of-magnitude (circuit coefficient complexity adds
uncertainty; grid<=4 uses the cheaper direct path and is a lower bound),
which is what budget-driven keep-set *selection* needs -- pair with
band_sensitivities() to keep the most impactful symbols.
"""
from __future__ import annotations

import json
import math
import os
import platform
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import sympy as sp

from ..engine.interp import _MAX_WORKERS, _PARALLEL_MIN_DETS, resolve_backend
from ..engine.mna import S, MnaError, _det, hybrid_split

# lower bound on the spread-extrapolated alpha (best-case parallel amortization
# still can't beat ~1/cores of the serial determinant work)
_ALPHA_FLOOR = 0.1

# ---- S-B sparse ("bot") cost model (S-D). Bot's work scales with the
# TRUE support T, unknown before solving; on the hybrid workload T tracks
# ~4*sqrt(K) (measured: k=6 T~80 at K=324, k=12 T=723 at K=46656). Total
# mod-p dets ~ (discovery 2T + ~32 primes x T) x L x 2 at ~1/10 the QQ
# det cost, pool-shared. Order-of-magnitude by design, like the QQ model.
_BOT_T_COEF = 4.0
_BOT_PRIMES = 35
_BOT_ALPHA = 0.03


def _bot_work(grid: int, s_deg: int, t_det: float) -> tuple[float, int]:
    T_hat = min(grid, _BOT_T_COEF * math.sqrt(grid))
    n = int(_BOT_PRIMES * T_hat * (s_deg + 2) * 2)
    return _BOT_ALPHA * t_det * n, n


@dataclass
class Calibration:
    """Machine-specific solve-time model:

        seconds = (a + b*spread) * raw + beta

    with raw = t_det * n_dets the machine-measured *serial* determinant
    work and `spread` the circuit's coefficient magnitude spread
    (log10 max/min |entry|) -- the driver of the big-integer
    reconstruction/cancellation tail. The (a, b, beta) triple is fitted
    separately for the serial and parallel interp paths (the parallel one
    absorbs whatever real pool speedup the machine achieves, rather than
    assuming a perfect workers-fold division). All features are
    permutation-invariant, so netlist ordering does not affect the
    estimate. Produced by calibrate(); the built-in default is a
    conservative, spread-independent fallback."""
    a_serial: float
    b_serial: float
    beta_serial: float
    a_parallel: float
    b_parallel: float
    beta_parallel: float
    platform: str
    n_samples: int
    # ---- learned from REAL solves, across sessions (see observe()).
    # A bounded multiplicative correction per path rather than a refit:
    # the synthetic ladders identify the model's SHAPE, while real work
    # on real circuits says how far off its scale is here. A factor
    # cannot distort the shape, needs no spread diversity to be
    # identifiable, and is meaningful from the very first observation.
    k_serial: float = 1.0
    k_parallel: float = 1.0
    n_obs_serial: int = 0
    n_obs_parallel: int = 0
    # the sparse path has its OWN cost model (_bot_work), which used to
    # bypass the calibration completely -- so no amount of calibrating or
    # learning could correct it. It learns on its own key now.
    k_bot: float = 1.0
    n_obs_bot: int = 0

    def predict(self, raw: float, spread: float, parallel: bool) -> float:
        if parallel:
            a, b, beta = self.a_parallel, self.b_parallel, self.beta_parallel
            k = self.k_parallel
        else:
            a, b, beta = self.a_serial, self.b_serial, self.beta_serial
            k = self.k_serial
        # floor the spread-extrapolated alpha: a negative b (parallelism helps
        # more at high spread) must not drive alpha <= 0 beyond the fitted range
        alpha = max(_ALPHA_FLOOR, a + b * spread)
        return max(0.0, k * (alpha * raw + beta))


# built-in fallback until calibrate() runs on this machine (conservative,
# spread-independent: over- rather than under-estimates)
_DEFAULT_CAL = Calibration(a_serial=3.0, b_serial=0.0, beta_serial=0.0,
                           a_parallel=3.0, b_parallel=0.0, beta_parallel=0.0,
                           platform="builtin-default", n_samples=0)
_CAL = _DEFAULT_CAL


def set_calibration(cal: Calibration) -> None:
    global _CAL
    _CAL = cal


def get_calibration() -> Calibration:
    return _CAL


def _platform_id() -> str:
    return f"{platform.system()}-{platform.machine()}-cpu{os.cpu_count()}"


def _cache_file(path: str | os.PathLike | None = None) -> Path:
    if path is not None:
        return Path(path)
    base = os.environ.get("XDG_CACHE_HOME") or (Path.home() / ".cache")
    return Path(base) / "circuitinsight" / "calibration.json"


def load_calibration(path=None) -> Calibration | None:
    """Load a cached calibration if present and matching this machine."""
    try:
        cal = Calibration(**json.loads(_cache_file(path).read_text()))
    except Exception:
        return None
    return cal if cal.platform == _platform_id() else None


#: an observation may move the correction by at most this factor either
#: way: one pathological solve (a machine under load, a fallback) must not
#: wreck the model
_OBS_CLAMP = 10.0
#: ...but the ACCUMULATED correction needs far more room than one sample.
#: The built-in default is deliberately pessimistic (alpha 3.0) and a real
#: machine with gmpy2 and a worker pool runs 20x faster, so a 10x band
#: would clamp short of the truth and leave estimates permanently wrong on
#: any machine where calibrate() has not run.
_K_CLAMP = 100.0
#: the learning rate floors here, so the model keeps tracking a machine
#: that changes (a laptop on battery, a busier server) instead of freezing
_OBS_MIN_WEIGHT = 0.15


def observe(predicted_s: float, actual_s: float, parallel: bool,
            path=None, key: str | None = None) -> Calibration:
    """Learn from a real solve: fold actual/predicted into this machine's
    persistent correction factor and save it.

    Geometric (log-space) averaging, because solve times are ratios, not
    offsets -- being 3x over and 3x under must cancel, not average to a
    bias. The weight decays as 1/(n+1) but never below _OBS_MIN_WEIGHT.
    Estimates therefore improve across sessions on the circuits you
    actually run, not only on the synthetic ladders calibrate() times."""
    if not (predicted_s > 0 and actual_s > 0):
        return _CAL                              # nothing learnable
    ratio = actual_s / predicted_s
    ratio = min(max(ratio, 1.0 / _OBS_CLAMP), _OBS_CLAMP)
    cal = replace(_CAL)
    key = key or ("parallel" if parallel else "serial")
    n = getattr(cal, "n_obs_" + key)
    k = getattr(cal, "k_" + key)
    w = max(_OBS_MIN_WEIGHT, 1.0 / (n + 1))
    k_new = math.exp((1 - w) * math.log(k) + w * math.log(k * ratio))
    k_new = min(max(k_new, 1.0 / _K_CLAMP), _K_CLAMP)
    setattr(cal, "k_" + key, k_new)
    setattr(cal, "n_obs_" + key, n + 1)
    set_calibration(cal)
    save_calibration(cal, path)
    return cal


def save_calibration(cal: Calibration, path=None) -> None:
    try:
        f = _cache_file(path)
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(asdict(cal)))
    except OSError:
        pass


#: The all-numeric solve is not only determinants: it also builds the
#: inverse-Vandermonde rows (O(L^2) exact rational operations), applies
#: them, assembles the two polynomials and cancels. Measured total against
#: determinant work alone -- cs_amp, miller_ota, ota5t (L=2..3), a 5T OTA
#: from silicon (L=8) and the uA741 (L=28) -- gives 3-5x over a fixed
#: ~15 ms of sympy setup that dominates the tiny cases. These two terms
#: keep all five within 1.4x; determinant count alone under-predicted the
#: uA741 by 2.7x.
_NUMERIC_S_OVER = 3.0
_NUMERIC_S_FLOOR_S = 0.015


@dataclass
class SolveEstimate:
    keep: list[str]              # the keep-set as requested
    kept_names: list[str]        # resolved symbol names kept symbolic
    grid_size: int               # PROD (stamp_count + 1)
    s_degree: int                # degree of det A in s
    n_dets: int                  # total determinant evaluations
    matrix_dim: int
    coeff_spread: float          # log10 max/min |entry|, the alpha driver
    parallel: bool               # will the interp solver parallelize?
    path: str                    # 'interp' | 'numeric-s' | 'direct'
    seconds: float               # estimated wall-clock — see __str__
    backend: str = "qq"          # interp backend the auto selector will pick

    #: paths whose cost `seconds` actually models. 'direct' is NOT one of
    #: them: a symbolic determinant carries polynomial arithmetic through
    #: every elimination step, and no count of numeric determinants
    #: describes it. Printing a number there was worse than printing none —
    #: it read as a prediction and was wrong by an order of magnitude.
    MODELLED = ("interp", "numeric-s")

    def __str__(self) -> str:
        bk = f", {self.backend}" if self.backend not in ("qq", "qq-s") else ""
        shape = (f"grid {self.grid_size}, s-deg {self.s_degree}, "
                 f"{self.n_dets} dets, "
                 f"{'parallel' if self.parallel else 'serial'}{bk}")
        if self.path not in self.MODELLED:
            # no determinant count describes this path, so don't print one
            return (f"symbolic determinant – not modelled "
                    f"(s-deg {self.s_degree}, {len(self.kept_names)} symbols, "
                    f"grid {float(self.grid_size):.3g} – the tractability "
                    f"signal)")
        return f"~{self.seconds:.3g}s  ({shape})"


@dataclass
class KeepPlan:
    keep: list[str]              # chosen symbols (band-ranked, budget-fit)
    dropped: list[str]           # ranked-but-excluded symbols
    estimate: SolveEstimate
    budget_s: float
    feasible: bool               # does even keep=[] fit the budget?

    def report(self) -> str:
        if not self.feasible:
            return (f"budget ~{self.budget_s:g}s is infeasible: the baseline "
                    f"symbolic solve alone is est {self.estimate.seconds:.3g}s "
                    f"(s-degree {self.estimate.s_degree}, "
                    f"{self.estimate.matrix_dim} nodes). Use "
                    f"frequency_response() for a numeric result, or raise the "
                    f"budget.")
        lines = [f"keep-set for ~{self.budget_s:g}s budget "
                 f"(est {self.estimate.seconds:.3g}s): {self.keep}"]
        if self.dropped:
            lines.append(f"  dropped (kept numeric): {self.dropped}")
        return "\n".join(lines)


def _numeric_A(system):
    subs = {system.symbols[n]: sp.Rational(repr(v))
            for n, v in system.values.items() if n in system.symbols}
    free = set(system.symbols.values()) - {S} - set(subs)
    if free:
        raise MnaError(
            f"estimate: all element values must be numeric; missing "
            f"{sorted(str(s) for s in free)}")
    return system.A.xreplace(subs)


def _coeff_spread(A_num) -> float:
    """log10(max/min) over the nonzero magnitudes of the numeric matrix's
    constant and s-linear entries -- a permutation-invariant measure of the
    coefficient dynamic range that drives big-integer arithmetic."""
    mags = []
    for M in (A_num.xreplace({S: sp.Integer(0)}), A_num.diff(S)):
        for e in M:
            if e != 0:
                mags.append(abs(complex(e)))
    if len(mags) < 2:
        return 0.0
    return math.log10(max(mags) / min(mags))


def _probe(system) -> tuple[int, float, float]:
    """Return (s_degree, per-determinant seconds, coeff_spread) from numeric
    determinants.

    Evaluates det A(s) at integer s = 0,1,2,... (all element values already
    numeric, so each is a fast scalar rational determinant), reads the
    s-degree exactly from finite differences, and times the evaluations."""
    from ..engine.interp import s_sweep_det

    A = _numeric_A(system)
    dim = A.shape[0]
    spread = _coeff_spread(A)
    # SHARED with the all-numeric solver: these are the same determinants it
    # reconstructs D(s) from, so probing here does not make the solve that
    # follows pay for them twice. hybrid_split and _numeric_A build the
    # matrix identically (sp.Rational(repr(v))), which is what lets the
    # cache key match.
    vals, t_det = s_sweep_det(A, dim + 2)     # s-degree <= dim
    return _degree_from_samples(vals, dim), t_det, spread


def _degree_from_samples(vals: list, bound: int) -> int:
    """Exact s-degree by finite differences over the rationals. Sound as
    long as len(vals) > the true degree, which `bound` guarantees."""
    diffs = list(vals)
    for d in range(bound + 1):
        diffs = [diffs[i + 1] - diffs[i] for i in range(len(diffs) - 1)]
        if all(x == 0 for x in diffs):
            return d
    return bound


def estimate_solve(analyzer, inp: str, out: str, keep,
                   _probe_cache: tuple | None = None) -> SolveEstimate:
    """Estimate the interpolation-solver wall-clock for tf(inp, out, keep).
    Pass _probe_cache=(s_deg, t_det, dim, spread) to reuse a probe.

    keep=ALL means every symbol is kept — the unbounded case, and the whole
    reason to ask. (It was previously coerced to [], costing the cheapest solve
    when the caller asked about the most expensive one.)"""
    from ..keep import is_all

    system = analyzer.system(inp)
    if out not in system.node_index:
        raise MnaError(f"output node {out!r} not found (or it is ground)")
    if is_all(keep):
        kept_names = list(system.symbols)
    else:
        _, kept_names = hybrid_split(system, list(keep))

    grid = 1
    for n in kept_names:
        grid *= max(1, system.stamp_counts.get(n, 1)) + 1

    if not kept_names and _probe_cache is None:
        # ALL-NUMERIC, and nothing else to estimate: sample EXACTLY what the
        # solve will sample -- the degree bound B means B+1 points both pin
        # the polynomial and determine its degree by finite differences, so
        # the general probe's dim+2 points are more than the question needs.
        # Those samples land in the shared cache, so the solve that follows
        # gets its whole denominator for free: asking the question no longer
        # costs extra over answering it.
        from ..engine.interp import (NUMERIC_S_HISTORY, _s_degree_bound,
                                     s_sweep_det)

        A_num = _numeric_A(system)
        L = _s_degree_bound(A_num) + 1
        vals, t_det = s_sweep_det(A_num, L)
        n_dets = 2 * L
        # self-calibrating overhead: every completed all-numeric solve
        # records (wall, det work); the median observed ratio replaces the
        # frozen constant, so "re-check the constants if the reconstruction
        # changes" services itself in-process
        over = _NUMERIC_S_OVER
        obs = sorted((h["wall_s"] - _NUMERIC_S_FLOOR_S) / h["det_s"]
                     for h in NUMERIC_S_HISTORY if h["det_s"] > 0)
        if obs:
            over = min(10.0, max(1.0, obs[len(obs) // 2]))
        return SolveEstimate(
            keep=[], kept_names=[], grid_size=1,
            s_degree=_degree_from_samples(vals, L - 1), n_dets=n_dets,
            matrix_dim=A_num.shape[0], coeff_spread=_coeff_spread(A_num),
            parallel=False, path="numeric-s",
            seconds=_NUMERIC_S_FLOOR_S + over * t_det * n_dets,
            backend="qq-s")

    if _probe_cache is None:
        s_deg, t_det, spread = _probe(system)
        dim = system.A.shape[0]
    else:
        s_deg, t_det, dim, spread = _probe_cache

    if not kept_names:
        # ALL-NUMERIC: the solver sweeps s and reconstructs (engine.interp.
        # _solve_numeric_s), which is exactly what _probe above just did --
        # so this is a measurement, not an extrapolation. L samples for the
        # denominator and L for the numerator, at the same structural degree
        # bound the solver uses.
        from ..engine.interp import _s_degree_bound

        L = _s_degree_bound(_numeric_A(system)) + 1
        n_dets = 2 * L
        return SolveEstimate(
            keep=[], kept_names=[], grid_size=1, s_degree=s_deg,
            n_dets=n_dets, matrix_dim=dim, coeff_spread=spread,
            parallel=False, path="numeric-s",
            seconds=_NUMERIC_S_FLOOR_S + _NUMERIC_S_OVER * t_det * n_dets,
            backend="qq-s")

    raw, n_dets, parallel = _raw_work(grid, s_deg, t_det)
    # mirror the solver's own auto selector (engine.interp.resolve_backend):
    # above the crossover the sparse bot backend runs instead of the dense
    # QQ grid, with support-scaled (not grid-scaled) work
    # the selector also weighs the coefficient spread: a wide one puts
    # exact reconstruction beyond the sparse path's prime budget
    backend = resolve_backend(n_dets, spread)
    if backend == "bot":
        seconds, n_dets = _bot_work(grid, s_deg, t_det)
        seconds *= _CAL.k_bot            # the sparse path learns too
    else:
        # QQ grid (explicit zp/ratfun estimates also fall back to this
        # model: both are opt-in expert modes and slower than it anyway)
        seconds = _CAL.predict(raw, spread, parallel)
    # Which path solve_tf will actually take. keep=ALL is solved by a DIRECT
    # symbolic determinant (mna.solve_tf only substitutes when keep is not None),
    # so the interp model above does not describe it: for ALL, `seconds` is an
    # extrapolation of the wrong path and only `grid_size` (the size of the
    # symbol space) is a sound tractability signal. grid<=4 likewise goes direct.
    path = "direct" if (is_all(keep) or grid <= 4) else "interp"
    # keep=ALL is a sentinel, not a list: record what it actually keeps (every
    # symbol), so SolveEstimate.keep stays a plain, iterable list of names.
    requested = list(kept_names) if is_all(keep) else list(keep)
    return SolveEstimate(keep=requested, kept_names=kept_names,
                         grid_size=grid, s_degree=s_deg, n_dets=n_dets,
                         matrix_dim=dim, coeff_spread=spread,
                         parallel=parallel, path=path, seconds=seconds,
                         backend=backend if path == "interp" else "qq")


def _raw_work(grid: int, s_deg: int, t_det: float) -> tuple[float, int, bool]:
    """Serial determinant work in seconds (n_dets numeric determinants at
    t_det each) plus whether the interp solver will parallelize. Any real
    parallel speedup is captured by the calibration's parallel alpha, not
    assumed here."""
    n_dets = 2 * (s_deg + 2) * grid
    workers = min(_MAX_WORKERS, max(1, (os.cpu_count() or 2) - 1))
    parallel = n_dets >= _PARALLEL_MIN_DETS and workers > 1
    return t_det * n_dets, n_dets, parallel


def _ladder_cin(n: int, r: str = "1k", c: str = "1p") -> dict:
    """A resistive-C ladder with n sections; keeping the first k resistors
    symbolic makes a grid of exactly 2**k. Varying the r/c values sweeps the
    coefficient spread, which calibration needs to fit the spread term."""
    inst = [{"name": "Vin", "device_type": "vsource",
             "terminals": {"p": "in", "n": "0"}}]
    prev = "in"
    for i in range(1, n + 1):
        node = f"n{i}"
        inst.append({"name": f"R{i}", "device_type": "resistor",
                     "terminals": {"p": prev, "n": node}, "params": {"r": r}})
        inst.append({"name": f"C{i}", "device_type": "capacitor",
                     "terminals": {"p": node, "n": "0"}, "params": {"c": c}})
        prev = node
    return {"cin_version": "0.1",
            "design": {"name": "cal_ladder", "source": {"kind": "hand"}},
            "top": "m", "ground": ["0"],
            "definitions": {"m": {"ports": [], "instances": inst}}}


def _fit3(samples: list[tuple[float, float, float]],
          fallback: tuple[float, float, float]) -> tuple[float, float, float]:
    """Fit actual = a*raw + b*(spread*raw) + beta over (raw, spread, actual);
    returns (a, b, beta). Falls back when data is too thin, and drops the
    spread term (b=0) if spread does not vary across the samples."""
    import numpy as np

    pts = [(r, s, a) for r, s, a in samples if r > 0]
    if len(pts) < 3 or len({round(s, 2) for _, s, _ in pts}) < 2:
        # not enough spread variation to identify b: fit a*raw + beta
        two = [(r, a) for r, s, a in pts]
        if len(two) < 2:
            if two:
                r, a = two[0]
                return a / r, 0.0, 0.0
            return fallback
        X = np.array([[r, 1.0] for r, _ in two])
        y = np.array([a for _, a in two])
        (a_, beta), *_ = np.linalg.lstsq(X, y, rcond=None)
        return (float(a_) if a_ > 0 else fallback[0]), 0.0, float(max(0.0, beta))
    X = np.array([[r, s * r, 1.0] for r, s, _ in pts])
    y = np.array([a for _, _, a in pts])
    (a_, b_, beta), *_ = np.linalg.lstsq(X, y, rcond=None)
    # guard: alpha(spread)=a+b*spread must stay positive over the sampled
    # spread range, else drop the spread term
    lo = min(s for _, s, _ in pts)
    hi = max(s for _, s, _ in pts)
    if a_ + b_ * lo <= 0 or a_ + b_ * hi <= 0:
        return _fit3([(r, lo, a) for r, _, a in pts], fallback)  # collapse b
    return float(a_), float(b_), float(max(0.0, beta))


def calibrate(force: bool = False, max_seconds: float = 3.0,
              sections: int = 12, cache_path=None,
              verbose: bool = False) -> Calibration:
    """Measure this machine's solve-time model and cache it. Runs real
    interpolation solves on synthetic ladders spanning a range of grid
    sizes AND coefficient spreads, and fits

        seconds = (a + b*spread) * raw + beta

    separately for the serial and parallel interp paths. Stores the result
    under the user cache dir (reused on later runs), installs it as the
    active model, and returns it.

    NOTE: because it exercises the parallel solver, run calibrate() from a
    ``__main__``-guarded script (Windows spawns worker processes); if the
    pool is unavailable it degrades to serial timings, which the model then
    reflects faithfully. force=True re-measures despite a cache;
    max_seconds caps the largest solve (bounds runtime)."""
    if not force:
        cached = load_calibration(cache_path)
        if cached is not None:
            set_calibration(cached)
            return cached

    from ..analyzer import Analyzer

    # ladders at three coefficient spreads (low / mid / high) so the spread
    # term is identifiable
    rc_configs = [("100", "10m"), ("1k", "1p"), ("1meg", "1f")]
    inp, out = "Vin", f"n{sections}"

    serial: list[tuple[float, float, float]] = []
    parallel: list[tuple[float, float, float]] = []
    warmed = False
    for r, c in rc_configs:
        an = Analyzer.from_cin(_ladder_cin(sections, r, c))
        system = an.system(inp)
        s_deg, t_det, spread = _probe(system)
        if not warmed:                          # warm pool + JIT once
            an.tf(inp, out, keep=["R1", "R2", "R3"], method="interp")
            warmed = True
        for k in range(3, sections + 1):        # grids 8, 16, 32, ...
            raw, _, is_par = _raw_work(2 ** k, s_deg, t_det)
            keep = [f"R{i}" for i in range(1, k + 1)]
            t0 = time.perf_counter()
            an.tf(inp, out, keep=keep, method="interp")
            dt = time.perf_counter() - t0
            (parallel if is_par else serial).append((raw, spread, dt))
            if verbose:
                print(f"  spread {spread:4.1f} grid {2**k:>7} "
                      f"({'par' if is_par else 'ser'}): raw {raw:.3g}s "
                      f"-> actual {dt:.3g}s")
            if dt > max_seconds:
                break

    a_s, b_s, beta_s = _fit3(serial, (3.0, 0.0, 0.0))
    # parallel defaults to the serial model (conservative: no speedup)
    a_p, b_p, beta_p = _fit3(parallel, (a_s, b_s, beta_s))
    cal = Calibration(a_serial=a_s, b_serial=b_s, beta_serial=beta_s,
                      a_parallel=a_p, b_parallel=b_p, beta_parallel=beta_p,
                      platform=_platform_id(),
                      n_samples=len(serial) + len(parallel))
    save_calibration(cal, cache_path)
    set_calibration(cal)
    return cal


def plan_keep(analyzer, inp: str, out: str, budget_s: float,
              metric: str = "complex", fmin: float | None = None,
              fmax: float | None = None, max_keep: int = 16) -> KeepPlan:
    """Choose the largest band-ranked keep-set whose estimated symbolic
    solve fits `budget_s`. Ranks symbols by band_sensitivities(), then
    greedily adds them (most impactful first) while the estimate stays
    within budget."""
    ranked = [n for n, _ in
              analyzer.band_sensitivities(inp, out, metric, fmin, fmax).ranking]
    ranked = ranked[:max_keep]

    system = analyzer.system(inp)
    s_deg, t_det, spread = _probe(system)
    cache = (s_deg, t_det, system.A.shape[0], spread)   # estimate_solve order

    chosen: list[str] = []
    est = estimate_solve(analyzer, inp, out, chosen, _probe_cache=cache)
    for name in ranked:                          # greedily add while it fits
        e = estimate_solve(analyzer, inp, out, chosen + [name],
                           _probe_cache=cache)
        if e.seconds > budget_s:
            break
        chosen, est = chosen + [name], e
    dropped = [n for n in ranked if n not in chosen]
    return KeepPlan(keep=chosen, dropped=dropped, estimate=est,
                    budget_s=budget_s, feasible=est.seconds <= budget_s)
