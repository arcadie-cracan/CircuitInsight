"""Full-chain regression: real two-stage Miller OTA (SKY130).

Fixtures from worklib/tb_ota2s on the open-source SKY130 PDK (plain Miller
C_c=16 pF, no R_z -> A0=85.5 dB, PM=60 deg): CInExport CIN + psfascii of the
identical ADE netlist. Pole splitting and the RHP zero on a real process.

The tail node swings with the input-pair sources, so the 5-cap LUMPED model
carries large charge-non-reciprocity error (~20 dB at 10 GHz on SKY130); the
exact charge-matrix model stays below 0.1 dB, so the AC match uses
`cap_model="matrix"`.
"""
import warnings
from pathlib import Path

import numpy as np
import pytest
import sympy as sp

from circuitinsight.adapters.spectre import SpectreRun
from circuitinsight.analysis import assert_tf_matches

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "spectre" / "miller"


@pytest.fixture(scope="module")
def run():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)   # pair in moderate inv.
        return SpectreRun(FIXTURES / "tb_ota2s.cin.json", FIXTURES / "psf")


def test_join_two_stage(run):
    assert len(run.op_data) == 15
    assert run.op_data["I0.Cc"]["c"] == pytest.approx(16e-12)
    assert run.op_data["CL"]["c"] == pytest.approx(5e-12)
    assert run.op_data["I0.MP2"]["gm"] > 1e-3          # ~85 uA second stage
    assert run.op_data["I0.MP2"]["region"] == 2
    assert run.op_data["I0.MN3"]["region"] == 2


def test_two_stage_tf_matches_spectre_ac(run):
    H = run.analyzer(cap_model="matrix").tf("VIND", "vout", keep=[])
    ac = run.ac()
    assert_tf_matches(
        H, ac.freq, ac.wave("vout"), ac.wave("vin_dm"),
        mag_tol_db=0.1, phase_tol_deg=1.5,             # matrix: 0.050 / 1.18
    )
    a0 = complex(H.numeric([1e2])[0])          # Re[H] at 100 Hz, a band point
    assert a0.real == pytest.approx(14308, rel=2e-3)  # (true dc gain H(0)=18786)

    (p1, *_) = H.poles()
    assert abs(p1) == pytest.approx(178.8, rel=0.03)    # dominant pole
    # the Miller RHP zero (below 100 MHz; the matrix model also resolves a
    # higher-order RHP zero near 380 MHz, out of band)
    rhp = [z for z in H.zeros() if z.real > 0 and abs(z) < 1e8]
    assert len(rhp) == 1
    assert abs(rhp[0]) == pytest.approx(1.053e7, rel=0.05)
    # textbook check: z ~ gm5 / (2 pi Cc)
    gm5 = run.op_data["I0.MP2"]["gm"]
    assert abs(rhp[0]) == pytest.approx(gm5 / (2 * np.pi * 16e-12), rel=0.05)


def test_pole_splitting_symbolic(run):
    an = run.analyzer()
    an.match("I0.MN0", "I0.MN1")
    an.match("I0.MP0", "I0.MP1")
    H = an.tf("VIND", "vout", keep=["gm_I0_MP2", "I0.Cc", "CL"])
    Hs = H.simplify(mag_tol_db=1.0, phase_tol_deg=5, fmin=1e3, fmax=5e8)

    # dominant pole must carry the Miller product Cc*gm5
    p1 = Hs.dominant_pole_expr()
    assert {"I0_Cc", "gm_I0_MP2"} <= {str(x) for x in p1.free_symbols}
    subs = Hs._subs_map()
    f_p1 = abs(complex(p1.xreplace(subs))) / (2 * np.pi)
    assert f_p1 == pytest.approx(181, rel=0.15)
    assert Hs.pole_separation() > 100

    a0 = complex(Hs.numeric([1e2])[0])
    assert a0.real == pytest.approx(14399, rel=0.02)


def _pm_of(psf_dir):
    """Phase margin from a fixture's AC waves (vout/vin_dm)."""
    from circuitinsight.adapters.spectre.acdata import load_ac

    a = load_ac(psf_dir)
    f = np.asarray(a.freq)
    H = np.asarray(a.wave("vout")) / np.asarray(a.wave("vin_dm"))
    m = 20 * np.log10(np.abs(H))
    ph = np.degrees(np.unwrap(np.angle(H))); ph -= ph[0]
    k = np.where(np.diff(np.sign(m)))[0][0]
    x = np.log10(f)
    xu = np.interp(0, [m[k + 1], m[k]], [x[k + 1], x[k]])
    return 180 + np.interp(xu, [x[k], x[k + 1]], [ph[k], ph[k + 1]])


def test_textbook_cc_sizing_falls_short_of_60deg():
    """The paper's Sec. VI closed-form check (tb_ota2s_cc*.scs): the classical
    two-real-pole 60-deg sizing under-sizes Cc on this amplifier because p2 is
    a complex pair. Spectre-measured PMs for the textbook-p2 (7.32 pF) and
    exact-|p2| (8.62 pF) closed forms, vs 60.2 deg at the installed 16 pF."""
    assert _pm_of(FIXTURES / "psf_cc732") == pytest.approx(48.1, abs=0.3)
    assert _pm_of(FIXTURES / "psf_cc862") == pytest.approx(51.3, abs=0.3)
    assert _pm_of(FIXTURES / "psf") == pytest.approx(60.2, abs=0.3)
