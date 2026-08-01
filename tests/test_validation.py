"""M3 validation harness: symbolic TF vs Spectre's own AC analysis.

The tolerances are tight (0.05-0.1 dB) because measured errors with the exact
charge-matrix model are ~0.001-0.01 dB. On SKY130 the *lumped* 5-cap model is
NOT enough -- the trans-caps are strongly non-reciprocal, so the lumped model
drifts to ~0.1-1 dB by 10 GHz; the matrix model captures the full dQ_i/dV_j
and stays essentially exact. The match tests therefore validate against
`cap_model="matrix"`. If these start failing after a model/stamp change,
believe the test.
"""
from pathlib import Path

import numpy as np
import pytest

from circuitinsight.adapters.spectre import SpectreRun
from circuitinsight.adapters.spectre.acdata import load_ac
from circuitinsight.analysis import assert_tf_matches, compare_tf
from circuitinsight.models import small_signal

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "spectre" / "m0"


# ------------------------------------------------------------------ AC parse

def test_parse_swept_ac_psfascii():
    ac = load_ac(FIXTURES / "psf")
    assert len(ac.freq) == 141                      # 1k..10G, 20/dec
    assert ac.freq[0] == pytest.approx(1e3)
    assert ac.freq[-1] == pytest.approx(1e10)
    assert np.all(ac.wave("vin") == 1.0)            # the driven source
    v0 = ac.wave("vout")[0]
    assert v0.real == pytest.approx(-3.654213458907056)
    assert "VDD:p" in ac.waves                      # branch currents too


def test_missing_wave_lists_alternatives():
    ac = load_ac(FIXTURES / "psf")
    with pytest.raises(KeyError, match="available"):
        ac.wave("nonexistent")


# ------------------------------------------------------- TF vs simulator AC

def test_nmos_stage_matches_spectre_ac():
    run = SpectreRun(FIXTURES / "tb_m0.cin.json", FIXTURES / "psf")
    H = run.analyzer(cap_model="matrix").tf("VIN", "vout", keep=[])
    ac = run.ac()
    report = assert_tf_matches(
        H, ac.freq, ac.wave("vout"), ac.wave("vin"),
        mag_tol_db=0.05, phase_tol_deg=0.5,          # full band to 10 GHz
    )
    assert report.worst_mag_db < 0.05


def test_pmos_stage_matches_spectre_ac():
    # separate run with VINP AC-driven (psf_acp)
    run = SpectreRun(FIXTURES / "tb_m0.cin.json", FIXTURES / "psf_acp")
    H = run.analyzer(cap_model="matrix").tf("VINP", "voutp", keep=[])
    ac = run.ac()
    assert_tf_matches(
        H, ac.freq, ac.wave("voutp"), ac.wave("vinp"),
        mag_tol_db=0.1, phase_tol_deg=1.0,
    )


def test_hierarchical_stage_matches_spectre_ac():
    run = SpectreRun(FIXTURES / "tb_hier.cin.json", FIXTURES / "psf_tbm2")
    H = run.analyzer(cap_model="matrix").tf("VIN", "net1", keep=[])
    ac = run.ac()
    assert_tf_matches(
        H, ac.freq, ac.wave("net1"), ac.wave("vin"),
        mag_tol_db=0.05, phase_tol_deg=0.5,
    )


# ------------------------------------------------------------------ canaries

def test_gm_sign_flip_is_caught(monkeypatch):
    # a reversed gm stamp produces a plausible-looking TF with the wrong
    # sign — exactly the class of bug this harness exists to catch
    flipped = [
        (p, k, (("s", "d", "g", "s") if p == "gm" else t))
        for (p, k, t) in small_signal._MOSFET
    ]
    monkeypatch.setattr(small_signal, "_MOSFET", flipped)

    run = SpectreRun(FIXTURES / "tb_m0.cin.json", FIXTURES / "psf")
    H = run.analyzer().tf("VIN", "vout", keep=[])
    ac = run.ac()
    report = compare_tf(H, ac.freq, ac.wave("vout"), ac.wave("vin"))
    assert not report.ok()                           # ~180 deg phase error
    assert report.worst_phase_deg > 90
    with pytest.raises(AssertionError, match="disagrees"):
        assert_tf_matches(H, ac.freq, ac.wave("vout"), ac.wave("vin"))


def test_zero_input_wave_diagnostic():
    # tb_m0's AC drives only VIN; using vinp as reference must be diagnosed,
    # not produce nan comparisons
    run = SpectreRun(FIXTURES / "tb_m0.cin.json", FIXTURES / "psf")
    H = run.analyzer().tf("VINP", "voutp", keep=[])
    ac = run.ac()
    with pytest.raises(ValueError, match="AC-driven"):
        compare_tf(H, ac.freq, ac.wave("voutp"), ac.wave("vinp"))
