"""S-D: auto backend selector, solve telemetry, estimator backend model.

The selector lives in ONE place (engine.interp.resolve_backend) and the
estimator mirrors it, so these tests pin both the decision rule and the
mirroring."""
import warnings
from pathlib import Path

import pytest
import sympy as sp

from circuitinsight import Analyzer
from circuitinsight.analysis import estimate as est
from circuitinsight.engine import interp

FIX = Path(__file__).resolve().parent / "fixtures" / "spectre"


@pytest.fixture(autouse=True)
def _reset_backend():
    old = interp.PROBE_BACKEND
    yield
    interp.PROBE_BACKEND = old


# ------------------------------------------------------------ the selector
def test_auto_picks_qq_below_and_bot_above_the_crossover(monkeypatch):
    monkeypatch.delenv("CIN_PROBE_BACKEND", raising=False)
    interp.PROBE_BACKEND = None
    assert interp.resolve_backend(interp._AUTO_BOT_MIN_DETS - 1) == "qq"
    assert interp.resolve_backend(interp._AUTO_BOT_MIN_DETS) == "bot"


def test_explicit_backend_overrides_auto(monkeypatch):
    monkeypatch.delenv("CIN_PROBE_BACKEND", raising=False)
    for forced in ("qq", "zp", "bot", "ratfun"):
        interp.PROBE_BACKEND = forced
        assert interp.resolve_backend(10**9) == forced
        assert interp.resolve_backend(1) == forced


def test_env_variable_is_honoured(monkeypatch):
    interp.PROBE_BACKEND = None
    monkeypatch.setenv("CIN_PROBE_BACKEND", "qq")
    assert interp.resolve_backend(10**9) == "qq"


# ------------------------------------------------------------- telemetry
@pytest.fixture(scope="module")
def _system():
    from circuitinsight.adapters.spectre import SpectreRun
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run = SpectreRun(FIX / "ota5t" / "tb_ota5t.cin.json",
                         FIX / "ota5t" / "psf")
        an = run.analyzer(cap_model="matrix")
        an.match("I0.MN0", "I0.MN1")
        an.match("I0.MP0", "I0.MP1")
    return an


KEEP = ["gm_I0_MN0", "gds_I0_MN0", "gm_I0_MP0"]


def _solve(an):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return an.tf("VIND", "vout", keep=KEEP, method="interp")


def test_telemetry_small_auto_solve_uses_qq(_system, monkeypatch):
    monkeypatch.delenv("CIN_PROBE_BACKEND", raising=False)
    interp.PROBE_BACKEND = None
    _solve(_system)
    tl = interp.LAST_SOLVE
    assert tl is not None
    assert tl["backend"] == "qq" and not tl["fell_back"]
    assert tl["n_dense_dets"] == tl["grid_K"] * tl["L"] * 2
    assert tl["n_dense_dets"] < interp._AUTO_BOT_MIN_DETS
    assert tl["wall_s"] > 0


def test_auto_bot_end_to_end_when_crossover_lowered(_system, monkeypatch):
    """Force the crossover below this small solve: auto must route through
    bot, record it in telemetry, and produce the identical expression."""
    monkeypatch.delenv("CIN_PROBE_BACKEND", raising=False)
    interp.PROBE_BACKEND = None
    h_qq = _solve(_system)                        # crossover still high: qq
    monkeypatch.setattr(interp, "_AUTO_BOT_MIN_DETS", 1)
    h_bot = _solve(_system)
    tl = interp.LAST_SOLVE
    assert tl["backend"] == "bot" and not tl["fell_back"]
    assert tl["T_den"] > 0 and tl["primes"] > 1
    assert sp.simplify(h_bot.expr - h_qq.expr) == 0


# ------------------------------------------------------------- estimator
def test_estimate_mirrors_the_selector():
    an = Analyzer.from_cin(est._ladder_cin(20))
    small = est.estimate_solve(an, "Vin", "n20", ["R1", "R2"])
    assert small.backend == "qq"
    big = est.estimate_solve(an, "Vin", "n20",
                             [f"R{i}" for i in range(1, 21)])
    assert big.backend == "bot"
    assert big.seconds > 0
    # the bot model must undercut the dense-QQ projection for huge grids
    raw, n_dets, par = est._raw_work(big.grid_size, big.s_degree, 1e-4)
    assert big.seconds < est._CAL.predict(raw, big.coeff_spread, par)


def test_estimate_backend_respects_forced_qq(monkeypatch):
    interp.PROBE_BACKEND = "qq"
    try:
        an = Analyzer.from_cin(est._ladder_cin(20))
        big = est.estimate_solve(an, "Vin", "n20",
                                 [f"R{i}" for i in range(1, 21)])
        assert big.backend == "qq"
    finally:
        interp.PROBE_BACKEND = None
