"""Solve-time estimation, machine calibration, and keep-set planning."""
import os
import tempfile

import pytest
import sympy as sp

from circuitinsight import Analyzer
from circuitinsight.analysis.estimate import (Calibration, calibrate,
                                              get_calibration,
                                              load_calibration,
                                              save_calibration,
                                              set_calibration)

RC = {
    "cin_version": "0.1",
    "design": {"name": "rc", "source": {"kind": "hand"}},
    "top": "m", "ground": ["0"],
    "definitions": {"m": {"ports": [], "instances": [
        {"name": "Vin", "device_type": "vsource",
         "terminals": {"p": "in", "n": "0"}},
        {"name": "R", "device_type": "resistor",
         "terminals": {"p": "in", "n": "out"}, "params": {"r": "1k"}},
        {"name": "C", "device_type": "capacitor",
         "terminals": {"p": "out", "n": "0"}, "params": {"c": "159n"}}],
    }},
}


def test_grid_size_and_ndets_structure():
    an = Analyzer.from_cin(RC)
    e0 = an.estimate_solve_time("Vin", "out", [])
    assert e0.grid_size == 1                       # no kept symbols
    assert e0.s_degree >= 1                        # one reactive element
    # all-numeric goes through the s-sweep, not a symbolic determinant:
    # L samples for the denominator and L for the numerator, where L is the
    # structural degree bound + 1 (and the bound covers the true degree)
    assert e0.path == "numeric-s" and e0.backend == "qq-s"
    assert e0.n_dets % 2 == 0
    assert e0.n_dets // 2 >= e0.s_degree + 1
    assert e0.seconds > 0

    # each kept single-stamp symbol multiplies the grid by 2
    eR = an.estimate_solve_time("Vin", "out", ["R"])
    eRC = an.estimate_solve_time("Vin", "out", ["R", "C"])
    assert eR.grid_size == 2
    assert eRC.grid_size == 4
    assert eRC.n_dets == 2 * (eRC.s_degree + 2) * 4
    # determinant count is monotone in the grid (seconds tracks it, modulo
    # microsecond probe jitter on this tiny circuit)
    assert eRC.n_dets > eR.n_dets > e0.n_dets
    assert eR.seconds > 0 and eRC.seconds > 0


def test_plan_keep_fits_budget_and_is_monotone():
    an = Analyzer.from_cin(RC)
    big = an.plan_keep("Vin", "out", budget_s=1e9, fmin=10, fmax=1e5)
    assert big.feasible
    assert set(big.keep) == {"R", "C"}             # everything fits
    assert big.estimate.seconds <= 1e9

    # a generous-but-finite budget keeps a subset of the unlimited plan
    mid = an.plan_keep("Vin", "out", budget_s=big.estimate.seconds * 0.6,
                       fmin=10, fmax=1e5)
    assert set(mid.keep) <= set(big.keep)
    assert mid.estimate.seconds <= big.estimate.seconds


def test_plan_keep_flags_infeasible_budget():
    an = Analyzer.from_cin(RC)
    tiny = an.plan_keep("Vin", "out", budget_s=1e-12, fmin=10, fmax=1e5)
    assert not tiny.feasible
    assert tiny.keep == []
    assert "infeasible" in tiny.report()
    assert "frequency_response" in tiny.report()


def test_calibration_predict_regimes_and_spread():
    # seconds = (a + b*spread)*raw + beta, per regime
    cal = Calibration(a_serial=2.0, b_serial=0.1, beta_serial=0.5,
                      a_parallel=1.0, b_parallel=0.0, beta_parallel=0.0,
                      platform="x", n_samples=6)
    assert cal.predict(10.0, spread=0.0, parallel=False) == 20.5   # 2*10+0.5
    assert cal.predict(10.0, spread=5.0, parallel=False) == 25.5   # 2.5*10+0.5
    assert cal.predict(10.0, spread=3.0, parallel=True) == 10.0    # b=0
    # a large negative b must not drive the estimate negative (alpha floored)
    neg = Calibration(1.0, -0.5, 0.0, 1.0, -0.5, 0.0, "x", 6)
    assert neg.predict(10.0, spread=20.0, parallel=False) > 0.0


def test_calibration_cache_roundtrip():
    cal = Calibration(3.3, 0.02, 0.1, 1.2, -0.01, 0.0, "unit-test-platform", 5)
    path = os.path.join(tempfile.gettempdir(), "ci_cal_roundtrip.json")
    save_calibration(cal, path)
    # platform mismatch -> load returns None (won't apply another machine's)
    assert load_calibration(path) is None
    os.remove(path)


def test_calibrate_runs_and_installs_model():
    # a small ladder keeps calibration fast; verify it produces a usable,
    # platform-stamped model and installs it as active
    path = os.path.join(tempfile.gettempdir(), "ci_cal_unit.json")
    if os.path.exists(path):
        os.remove(path)
    prev = get_calibration()
    try:
        cal = calibrate(force=True, sections=5, max_seconds=2.0,
                        cache_path=path)
        assert cal.n_samples >= 1
        assert cal.a_serial > 0 and cal.a_parallel > 0
        assert cal.platform == get_calibration().platform
        # a cached reload matches this machine and returns the same model
        assert load_calibration(path).a_serial == cal.a_serial
    finally:
        set_calibration(prev)
        if os.path.exists(path):
            os.remove(path)


def test_estimate_and_solve_share_the_s_samples():
    """The probe used to be pure overhead: it evaluated det A(s) at integer
    s to read the degree, then the all-numeric solve evaluated det A(s) at
    integer s again to reconstruct D(s). Same determinants, computed twice,
    so asking how long a solve would take cost about as much as solving.
    They now come from one cache -- and the estimate samples exactly the
    B+1 points the solve needs, not the general probe's dim+2."""
    from circuitinsight.engine import interp as I
    from circuitinsight.engine.interp import _s_degree_bound
    from circuitinsight.engine.mna import hybrid_split

    an = Analyzer.from_cin(RC)
    system = an.system("Vin")
    subs, kept = hybrid_split(system, [])
    assert not kept
    A = system.A.xreplace(subs)
    L = _s_degree_bound(A) + 1

    I._S_CACHE.clear()
    an.estimate_solve_time("Vin", "out", [])
    key = sp.ImmutableMatrix(A)
    assert key in I._S_CACHE, "the estimate must populate the shared cache"
    assert len(I._S_CACHE[key][0]) == L      # exactly what the solve needs

    # the solve then adds no fresh denominator samples: every progress tick
    # it reports belongs to the numerator
    ticks = []
    an.tf("Vin", "out", keep=[], progress=lambda d, t: ticks.append(d))
    assert len(I._S_CACHE[key][0]) == L      # unchanged: nothing recomputed
    assert ticks and ticks[-1] == 2 * L      # den (cached) + num, all reported


def test_numeric_s_estimate_self_calibrates_from_solve_history():
    """The overhead ratio was a frozen constant with a "re-check if the
    reconstruction changes" note. Now every completed all-numeric solve
    records (wall, det work) and the estimator uses the median observed
    ratio — the constant is only the cold-start default."""
    from circuitinsight.analysis.estimate import _NUMERIC_S_FLOOR_S
    from circuitinsight.engine import interp as I

    an = Analyzer.from_cin(RC)
    I.NUMERIC_S_HISTORY.clear()
    e_cold = an.estimate_solve_time("Vin", "out", [])

    # observed solves running at exactly twice the cold-start ratio
    I.NUMERIC_S_HISTORY.append(
        {"wall_s": _NUMERIC_S_FLOOR_S + 6.0 * 0.001, "det_s": 0.001})
    try:
        e_warm = an.estimate_solve_time("Vin", "out", [])
        assert (e_warm.seconds - _NUMERIC_S_FLOOR_S) == pytest.approx(
            2.0 * (e_cold.seconds - _NUMERIC_S_FLOOR_S), rel=1e-6)
    finally:
        I.NUMERIC_S_HISTORY.clear()


def test_observations_teach_the_model_across_sessions(tmp_path):
    """The calibration learns from REAL solves, not only from
    calibrate()'s synthetic ladders: a bounded multiplicative
    correction per path, averaged geometrically (3x over and 3x under
    must cancel, not leave a bias), persisted so the next session
    starts where this one ended."""
    from dataclasses import replace as _replace

    from circuitinsight.analysis import estimate as est

    cache = tmp_path / "cal.json"
    base = _replace(est.get_calibration(), platform=est._platform_id(),
                    k_serial=1.0, k_parallel=1.0,
                    n_obs_serial=0, n_obs_parallel=0)
    est.set_calibration(base)
    try:
        # a machine where solves really take 3x the UNCORRECTED model:
        # each round predicts with the current factor, so the observed
        # ratio shrinks as the factor learns -- it converges on 3, it
        # does not compound
        for _ in range(8):
            k = est.get_calibration().k_parallel
            est.observe(10.0 * k, 30.0, parallel=True, path=cache)
        cal = est.get_calibration()
        assert cal.k_parallel == pytest.approx(3.0, rel=0.2)
        assert cal.n_obs_parallel == 8
        assert cal.k_serial == 1.0               # per-path, untouched

        # it PERSISTS: a fresh session loads where this one ended
        loaded = est.load_calibration(cache)
        assert loaded is not None
        assert loaded.k_parallel == pytest.approx(cal.k_parallel)

        # and the prediction moves with it
        hi = cal.predict(1.0, 3.0, parallel=True)
        est.set_calibration(_replace(cal, k_parallel=1.0))
        assert hi > est.get_calibration().predict(1.0, 3.0, parallel=True)

        # symmetric errors cancel (geometric mean), and one absurd
        # sample cannot run away with the model
        est.set_calibration(_replace(base))
        for _ in range(10):
            k = est.get_calibration().k_serial
            est.observe(10.0 * k, 30.0, parallel=False, path=cache)
            k = est.get_calibration().k_serial
            est.observe(10.0 * k, 10.0 / 3.0, parallel=False, path=cache)
        assert 0.6 < est.get_calibration().k_serial < 1.7
        est.set_calibration(_replace(base))
        est.observe(1.0, 1e9, parallel=False, path=cache)
        # one absurd sample is bounded by the PER-SAMPLE clamp, even
        # though the accumulated band (_K_CLAMP) is wider
        assert est.get_calibration().k_serial == pytest.approx(
            est._OBS_CLAMP, rel=1e-9)
    finally:
        est.set_calibration(est._DEFAULT_CAL)


def test_every_path_learns_on_its_own_key(tmp_path):
    """Trustworthy estimates need each cost model corrected separately.
    The sparse path has its OWN model (_bot_work) that bypassed the
    calibration entirely -- no calibrating or learning could touch it
    (measured: 45.6 s predicted, 184 s actual). It now carries its own
    factor, and dense samples must not leak into it or vice versa."""
    from dataclasses import replace as _replace

    from circuitinsight.analysis import estimate as est

    cache = tmp_path / "cal.json"
    base = _replace(est.get_calibration(), platform=est._platform_id(),
                    k_serial=1.0, k_parallel=1.0, k_bot=1.0,
                    n_obs_serial=0, n_obs_parallel=0, n_obs_bot=0)
    est.set_calibration(base)
    try:
        for _ in range(8):                       # the measured sparse case
            k = est.get_calibration().k_bot
            est.observe(45.6 * k, 184.0, parallel=True, key="bot",
                        path=cache)
        cal = est.get_calibration()
        assert cal.k_bot == pytest.approx(4.0, rel=0.2)
        assert cal.k_parallel == 1.0 and cal.k_serial == 1.0

        # the band must reach a 20x-wrong default, which 10x could not
        assert est._K_CLAMP >= 100.0
        est.set_calibration(_replace(base))
        for _ in range(30):
            k = est.get_calibration().k_parallel
            est.observe(4920.0 * k, 232.0, parallel=True, path=cache)
        assert est.get_calibration().k_parallel < 0.1
    finally:
        est.set_calibration(est._DEFAULT_CAL)


def test_flush_calibration_forgets_file_and_process(tmp_path):
    """The Options verb's primitive: the persistent file goes, the
    in-process model returns to the built-in default -- ready for a
    fresh measurement."""
    from dataclasses import replace as _replace

    from circuitinsight.analysis import estimate as est

    cache = tmp_path / "calibration.json"
    try:
        cal = _replace(est._DEFAULT_CAL, platform=est._platform_id(),
                       k_parallel=7.0, n_obs_parallel=5)
        est.set_calibration(cal)
        est.save_calibration(cal, cache)
        assert cache.exists()
        est.flush_calibration(cache)
        assert not cache.exists()
        fresh = est.get_calibration()
        assert fresh.platform == "builtin-default"
        assert fresh.k_parallel == 1.0 and fresh.n_obs_parallel == 0
        est.flush_calibration(cache)          # idempotent: no file, no raise
    finally:
        est.set_calibration(est._DEFAULT_CAL)


def test_the_all_numeric_path_does_not_claim_to_be_modelled():
    """Measured 2026-08-31 against real loop-gain solves, the numeric-s
    estimate read 0.1 s for 1.5 s and 0.9 s for 34.1 s -- 15x and 38x
    low, the error GROWING with size. `_numeric_s_over()` self-
    calibrates, but those two need ratios of 37.8 and 115.5, so no
    single scalar fits both and the clamp at 10 reaches neither: the
    cost is exact-rational coefficient growth, which t_det * n_dets
    does not carry.

    estimate.py already learned this for the symbolic path -- "printing
    a number there was worse than printing none, it read as a
    prediction and was wrong by an order of magnitude". The same rule,
    applied to the same failure.

    `seconds` stays POPULATED: the keep planner budgets against the
    numeric baseline and would break on None. It is a planning
    heuristic, and callers that SHOW it gate on MODELLED."""
    from circuitinsight.analysis.estimate import SolveEstimate

    assert "numeric-s" not in SolveEstimate.MODELLED
    assert "direct" not in SolveEstimate.MODELLED
    assert "interp" in SolveEstimate.MODELLED, "the fitted path must stay"


def test_an_unmodelled_estimate_names_its_own_reason():
    """The two unmodelled paths fail for different reasons and must not
    share a sentence: one is a symbolic determinant, the other exact
    rationals. A numeric solve described as 'symbolic determinant'
    would send a reader after the wrong cost."""

    from circuitinsight.analysis.estimate import SolveEstimate

    base = dict(grid_size=1, s_degree=5, n_dets=72, matrix_dim=17,
                kept_names=(), keep=(), coeff_spread=1.0,
                parallel=False, seconds=0.139)
    num = SolveEstimate(path="numeric-s", **base)
    sym = SolveEstimate(path="direct", **base)
    assert "all-numeric" in str(num) and "not modelled" in str(num)
    assert "symbolic" not in str(num), "wrong cost named"
    assert "coefficient growth" in str(num)
    assert "symbolic determinant" in str(sym)
    # and a modelled one still prints its number
    fit = SolveEstimate(path="interp", **base)
    assert "~0.139s" in str(fit)


def test_a_plain_script_prices_with_the_cached_calibration(tmp_path,
                                                           monkeypatch):
    """Only the GUI and Session used to load the machine's calibration:
    a script building an Analyzer priced with the factory default while
    its solves' lessons sat unread in the cache. The first use now loads
    it; an explicit set_calibration() still wins."""
    from dataclasses import replace as _replace

    from circuitinsight.analysis import estimate as est

    monkeypatch.setenv("CIRCUITINSIGHT_CACHE_DIR", str(tmp_path))
    cached = _replace(est._DEFAULT_CAL, platform=est._platform_id(),
                      k_serial=7.0, k_parallel=7.0, n_obs_parallel=3,
                      k_bot=4.4, n_obs_bot=1)
    est.save_calibration(cached)

    # a fresh process: nothing set yet
    monkeypatch.setattr(est, "_CAL", est._DEFAULT_CAL)
    monkeypatch.setattr(est, "_CAL_SETTLED", False)
    assert est.get_calibration() == cached
    # ...and that is what prices an estimate (fixed probe: no jitter)
    monkeypatch.setattr(est, "_CAL", est._DEFAULT_CAL)
    monkeypatch.setattr(est, "_CAL_SETTLED", False)
    an = Analyzer.from_cin(RC)
    e = est.estimate_solve(an, "Vin", "out", ["R", "C"],
                           _probe_cache=(1, 1e-4, 3, 0.0))
    assert e.seconds == pytest.approx(
        cached.predict(1e-4 * e.n_dets, 0.0, e.parallel))

    # observe() before any estimate learns ON TOP of the cache instead of
    # overwriting it with the factory default plus one sample
    monkeypatch.setattr(est, "_CAL", est._DEFAULT_CAL)
    monkeypatch.setattr(est, "_CAL_SETTLED", False)
    est.observe(1.0, 1.0, parallel=True)
    after = est.load_calibration()
    assert after.n_obs_parallel == 4 and after.k_bot == pytest.approx(4.4)

    # an explicit calibration is never replaced by the cache
    monkeypatch.setattr(est, "_CAL_SETTLED", False)
    mine = _replace(est._DEFAULT_CAL, a_serial=0.5)
    est.set_calibration(mine)
    assert est.get_calibration() is mine


def test_the_sparse_path_learns_from_script_solves(tmp_path):
    """k_bot never moved outside the GUI (n_obs_bot stayed 0): nothing
    but the GUI fed solves back. learn_from_solve() does, on the key that
    PRICED the solve, with the pool startup learned apart -- and a mod-p
    fallback to the QQ grid teaches nothing to the sparse model."""
    from dataclasses import replace as _replace

    from circuitinsight.analysis import estimate as est

    cache = tmp_path / "cal.json"
    est.set_calibration(_replace(est._DEFAULT_CAL,
                                 platform=est._platform_id()))
    e = est.SolveEstimate(keep=["R"], kept_names=["R"], grid_size=8192,
                          s_degree=7, n_dets=228084, matrix_dim=12,
                          coeff_spread=10.0, parallel=True, path="interp",
                          seconds=2.4, backend="bot", startup_s=3.0)
    solve = {"backend": "bot", "fell_back": False, "pool_startup_s": 3.6}
    assert est.learn_from_solve(e, 3.6 + 10.6, solve=solve, path=cache)
    cal = est.get_calibration()
    assert cal.n_obs_bot == 1
    assert cal.k_bot == pytest.approx(10.6 / 2.4)    # first sample: full
    assert cal.n_obs_parallel == 0                   # dense untouched
    assert cal.pool_startup_s == pytest.approx(3.6)  # learned apart
    assert est.load_calibration(cache).k_bot == pytest.approx(cal.k_bot)
    # the telemetry is marked: a second learner (the GUI's) skips it
    assert solve["observed"]
    assert not est.learn_from_solve(e, 99.0, solve=solve, path=cache)
    assert est.get_calibration().n_obs_bot == 1

    fell = {"backend": "qq", "fell_back": True, "pool_startup_s": 0.0}
    assert not est.learn_from_solve(e, 50.0, solve=fell, path=cache)
    assert est.get_calibration().n_obs_bot == 1


def test_estimate_then_tf_teaches_the_calibration(tmp_path, monkeypatch):
    """The Analyzer closes the loop for scripts: tf() of the solve just
    estimated feeds its wall clock back; a tf() of a DIFFERENT solve does
    not (its estimate priced something else)."""
    from dataclasses import replace as _replace

    from circuitinsight.analysis import estimate as est

    monkeypatch.setenv("CIRCUITINSIGHT_CACHE_DIR", str(tmp_path))
    est.set_calibration(_replace(est._DEFAULT_CAL,
                                 platform=est._platform_id()))
    an = Analyzer.from_cin(RC)
    e = an.estimate_solve_time("Vin", "out", ["R", "C"])
    assert e.path == "interp"
    key = est.solve_key(e)
    an.tf("Vin", "out", keep=["R"])                  # not what was priced
    assert getattr(est.get_calibration(), "n_obs_" + key) == 0
    an.estimate_solve_time("Vin", "out", ["R", "C"])
    an.tf("Vin", "out", keep=["R", "C"])
    assert getattr(est.get_calibration(), "n_obs_" + key) == 1
    assert getattr(est.load_calibration(), "n_obs_" + key) == 1
    an.tf("Vin", "out", keep=["R", "C"])             # the estimate is spent
    assert getattr(est.get_calibration(), "n_obs_" + key) == 1


def test_pool_startup_is_priced_once_per_process(monkeypatch):
    """The first parallel solve of a process starts the worker pool
    (measured 3.6 s on 16-core Windows): a per-process term, reported
    beside `seconds`, never inside it, and gone once the pool is up."""
    from dataclasses import replace as _replace

    from circuitinsight.analysis import estimate as est
    from circuitinsight.engine import interp

    monkeypatch.setattr(est, "default_workers", lambda: 15)
    monkeypatch.setattr(interp, "default_workers", lambda: 15)
    monkeypatch.setattr(interp, "_spawn_would_deadlock", lambda: False)
    monkeypatch.setattr(interp, "pool_running", lambda: False)
    est.set_calibration(_replace(est._DEFAULT_CAL, pool_startup_s=0.0))
    assert est.pool_startup_estimate() == pytest.approx(
        15 * est._POOL_STARTUP_PER_WORKER_S)         # unmeasured: a guess
    est.set_calibration(_replace(est._DEFAULT_CAL, pool_startup_s=3.6))
    assert est.pool_startup_estimate() == pytest.approx(3.6)
    e = est.SolveEstimate(keep=["R"], kept_names=["R"], grid_size=8,
                          s_degree=7, n_dets=1000, matrix_dim=12,
                          coeff_spread=1.0, parallel=True, path="interp",
                          seconds=10.0, startup_s=3.6)
    assert "pool startup" in str(e) and e.total_s == pytest.approx(13.6)
    monkeypatch.setattr(interp, "pool_running", lambda: True)
    assert est.pool_startup_estimate() == 0.0
    # the measurement folds in geometrically, like every other lesson
    est.set_calibration(_replace(est._DEFAULT_CAL, pool_startup_s=4.0,
                                 n_obs_startup=1))
    monkeypatch.setenv("CIRCUITINSIGHT_CACHE_DIR",
                       tempfile.mkdtemp(prefix="ci-cal-"))
    cal = est.observe_pool_startup(1.0)
    assert cal.pool_startup_s == pytest.approx(2.0) and cal.n_obs_startup == 2
