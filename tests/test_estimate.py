"""Solve-time estimation, machine calibration, and keep-set planning."""
import os
import tempfile

import pytest

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
    assert e0.n_dets == 2 * (e0.s_degree + 2) * 1
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
