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
from dataclasses import asdict, dataclass
from pathlib import Path

import sympy as sp

from ..engine.interp import _MAX_WORKERS, _PARALLEL_MIN_DETS
from ..engine.mna import S, MnaError, _det, hybrid_split

# lower bound on the spread-extrapolated alpha (best-case parallel amortization
# still can't beat ~1/cores of the serial determinant work)
_ALPHA_FLOOR = 0.1


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

    def predict(self, raw: float, spread: float, parallel: bool) -> float:
        if parallel:
            a, b, beta = self.a_parallel, self.b_parallel, self.beta_parallel
        else:
            a, b, beta = self.a_serial, self.b_serial, self.beta_serial
        # floor the spread-extrapolated alpha: a negative b (parallelism helps
        # more at high spread) must not drive alpha <= 0 beyond the fitted range
        alpha = max(_ALPHA_FLOOR, a + b * spread)
        return max(0.0, alpha * raw + beta)


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


def save_calibration(cal: Calibration, path=None) -> None:
    try:
        f = _cache_file(path)
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(asdict(cal)))
    except OSError:
        pass


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
    path: str                    # 'interp' (grid>4) or 'direct' (grid<=4)
    seconds: float               # estimated wall-clock

    def __str__(self) -> str:
        note = "" if self.path == "interp" else " (direct path; lower bound)"
        return (f"~{self.seconds:.3g}s  (grid {self.grid_size}, "
                f"s-deg {self.s_degree}, {self.n_dets} dets, "
                f"{'parallel' if self.parallel else 'serial'}){note}")


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
    A = _numeric_A(system)
    dim = A.shape[0]
    spread = _coeff_spread(A)
    vals, times = [], []
    for k in range(dim + 2):                 # s-degree <= dim
        t0 = time.perf_counter()
        vals.append(_det(A.xreplace({S: sp.Integer(k)})))
        times.append(time.perf_counter() - t0)
    # exact finite differences over the rationals -> degree
    s_deg, diffs = dim, list(vals)
    for d in range(dim + 1):
        diffs = [diffs[i + 1] - diffs[i] for i in range(len(diffs) - 1)]
        if all(x == 0 for x in diffs):
            s_deg = d
            break
    times.sort()
    t_det = times[len(times) // 2]            # median, robust to jitter
    return s_deg, t_det, spread


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

    if _probe_cache is None:
        s_deg, t_det, spread = _probe(system)
        dim = system.A.shape[0]
    else:
        s_deg, t_det, dim, spread = _probe_cache

    raw, n_dets, parallel = _raw_work(grid, s_deg, t_det)
    # apply the machine calibration (serial work + coeff spread -> wall clock)
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
                         parallel=parallel, path=path, seconds=seconds)


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
