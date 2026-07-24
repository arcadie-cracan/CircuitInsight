"""OP-point sensitivity ranking and keep-set suggestion."""
import warnings
from pathlib import Path

import pytest

from circuitinsight import Analyzer
from circuitinsight.adapters.spectre import SpectreRun

GOLDEN = Path(__file__).resolve().parent / "golden" / "circuits"
FIX = Path(__file__).resolve().parent / "fixtures" / "spectre"


def test_cs_amp_sensitivities_match_hand_analysis():
    an = Analyzer.from_cin(GOLDEN / "cs_amp.cin.json")
    rep = an.sensitivities("V1", "vout", n_poles=1)

    dc = dict(rep.dc_gain)
    gds, gl = 8.5e-6, 1e-4                       # gds_M1, 1/RL
    assert dc["gm_M1"] == pytest.approx(1.0, abs=1e-9)
    assert dc["RL"] == pytest.approx(gl / (gds + gl), rel=1e-6)
    assert dc["gds_M1"] == pytest.approx(-gds / (gds + gl), rel=1e-6)
    assert dc["cgs_M1"] == pytest.approx(0.0, abs=1e-9)   # caps: no dc role

    p1 = dict(rep.pole_sens[0])
    cout = 100e-15 + 0.75e-15 + 1.7e-15
    assert p1["CL"] == pytest.approx(-100e-15 / cout, rel=1e-6)
    assert p1["RL"] == pytest.approx(-gl / (gds + gl), rel=1e-6)
    assert abs(p1.get("gm_M1", 0.0)) < 1e-9              # gm: no pole role


def test_5t_discovers_dc_gain_keep_set():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        run = SpectreRun(FIX / "ota5t" / "tb_ota5t.cin.json",
                         FIX / "ota5t" / "psf")
        an = run.analyzer()
        an.match("I0.MN0", "I0.MN1")
        an.match("I0.MP0", "I0.MP1")
    rep = an.sensitivities("VIND", "vout", n_poles=1)
    # A0 = gm/(gdsN + gdsP): the tool must find its participants unaided
    assert set(rep.suggest_keep("dc_gain", 3)) == {
        "gm_I0_MN0", "gds_I0_MN0", "gds_I0_MP0"}


def test_two_stage_discovers_pole_splitting_set():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        run = SpectreRun(FIX / "miller" / "tb_ota2s.cin.json",
                         FIX / "miller" / "psf")
        an = run.analyzer()
        an.match("I0.MN0", "I0.MN1")
        an.match("I0.MP0", "I0.MP1")
    rep = an.sensitivities("VIND", "vout", n_poles=1)
    # p1 = (g1+g3)(g5+g6)/(Cc*gm5): Miller cap and gm5 must rank on top
    top = rep.suggest_keep("p1", 5)
    assert top[0] == "I0_Cc"
    assert "gm_I0_MP2" in top[:2]
    # dc-gain top-6 reproduces the hand-picked A0 keep-set
    assert set(rep.suggest_keep("dc_gain", 6)) == {
        "gm_I0_MN0", "gds_I0_MN0", "gds_I0_MP0",
        "gm_I0_MP2", "gds_I0_MP2", "gds_I0_MN3"}


def test_bad_target_rejected():
    an = Analyzer.from_cin(GOLDEN / "cs_amp.cin.json")
    rep = an.sensitivities("V1", "vout", n_poles=1)
    with pytest.raises(ValueError, match="target"):
        rep.suggest_keep("q7")
