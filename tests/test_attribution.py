"""Per-pole attribution: which element establishes which pole — computed
from exact null-vector sensitivities and then VERIFIED by nudging the top
owner and re-rooting, instead of assumed the textbook way."""
import warnings
from pathlib import Path

import pytest

from circuitinsight import Analyzer
from circuitinsight.analysis.attribution import pole_attribution

FIX = Path(__file__).resolve().parent / "fixtures" / "spectre"


def test_rc_pole_belongs_to_r_and_c_equally():
    """tau = R*C: the one place the answer is beyond argument."""
    an = Analyzer.from_cin({
        "cin_version": "0.1", "top": "m", "ground": ["0"],
        "definitions": {"m": {"ports": [], "instances": [
            {"name": "V1", "device_type": "vsource",
             "terminals": {"p": "in", "n": "0"}},
            {"name": "R1", "device_type": "resistor",
             "terminals": {"p": "in", "n": "out"}, "params": {"r": "1k"}},
            {"name": "C2", "device_type": "capacitor",
             "terminals": {"p": "out", "n": "0"}, "params": {"c": "1n"}},
        ]}}})
    atts = pole_attribution(an.system("V1"))
    assert len(atts) == 1
    a = atts[0]
    assert a.f_hz == pytest.approx(1 / (2 * 3.141592653589793 * 1e-6),
                                   rel=1e-6)
    shares = {o.symbol: o.share for o in a.owners}
    assert shares["R1"] == pytest.approx(0.5, abs=0.01)
    assert shares["C2"] == pytest.approx(0.5, abs=0.01)
    assert a.verified


def test_folded_cascode_dominant_pole_is_owned_by_the_load(fc_system=None):
    """The OTA fact: the dominant pole is Sum(gds_out)/CL — so CL leads
    the attribution and the output-branch gds's follow. Every reported
    pole must pass the nudge-and-re-root verification."""
    from circuitinsight.adapters.spectre import SpectreRun

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run = SpectreRun(FIX / "fc" / "tb_fc.cin.json", FIX / "fc" / "psf")
        an = run.analyzer(cap_model="matrix")
    atts = pole_attribution(an.system("VIND"), n_poles=3)
    dom = atts[0]
    assert dom.f_hz == pytest.approx(20460, rel=0.02)
    assert dom.owners[0].symbol == "CL"
    follow = {o.symbol for o in dom.owners[1:]}
    assert any(s.startswith("gds_") for s in follow)
    for a in atts:
        assert a.verified, a.describe()
        # the verification is quantitative: first-order within 2x of exact
        assert a.actual_rel == pytest.approx(a.predicted_rel, rel=1.0)
