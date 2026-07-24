"""r2r_bias_pos_loop: over-unity positive feedback via impact ionization.

The fixture is the floating-pair isolation bench of the rail-to-rail
class-AB investigation (SKY130 nfet/pfet_g5v0d10v5, <sim-host> run of
2026-07-19; regenerated on the open PDK from the original
foundry-PDK bench): two
common-gate devices in a ring closed by the Tian probe VPRB, sources tied
to fixed bias in DC and to the sense resistors in AC, plus a closed-loop
node-resistance ac sweep (unit AC current from IT into vsn).

Two CIN netlists share one psf: `r2r_bias_pos_loop.cin.json` is the plain
first-order model ({gm, gmbs, gds} from dcOpInfo), `r2r_ii.cin.json` adds
the identified impact-ionization stamps as gii_d/gii_m MOSFET params —
the conductances Spectre linearizes in stb/ac but never prints in
dcOpInfo. The pair pins down the session's central results:

  * with gii, the reconstruction reproduces Spectre's stb loopGain over
    the whole sweep and T(0) = +1.0041 > 1 (regenerative, latching);
  * without gii, T(0) = 0.99994 < 1 -- the first-order model of this
    bench is provably sub-unity (1 - T0 = L/(L+Q), all-positive), so it
    cannot show the effect at all;
  * the measured closed-loop node resistance at vsn is negative
    (R = R0/(1 - T0) exactly), and the first-order model gets the sign
    wrong.
"""
import warnings
from pathlib import Path

import numpy as np
import pytest
import sympy as sp

from circuitinsight.adapters.spectre import SpectreRun
from circuitinsight.adapters.spectre.acdata import load_ac
from circuitinsight.adapters.spectre.stbdata import load_stb

FIX = Path(__file__).resolve().parent / "fixtures" / "spectre" / "r2r"


def _analyzer(cin):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SpectreRun(FIX / cin, FIX / "psf").analyzer(cap_model="matrix")


@pytest.fixture(scope="module")
def stb():
    return load_stb(FIX / "psf")


@pytest.fixture(scope="module")
def T_ii(stb):
    return _analyzer("r2r_ii.cin.json").loop_gain("VPRB")


def test_gii_model_matches_stb_and_exceeds_unity(T_ii, stb):
    """With the impact-ionization stamps the reconstruction tracks Spectre's
    stb loopGain across the full sweep, and T(0) is positive real, over
    unity by +0.41 % -- the whole question at stake on this bench."""
    Tm = T_ii.numeric(stb.freq)
    err_db = 20 * np.log10(np.abs(Tm) / np.abs(stb.loop_gain))
    err_ph = np.degrees(np.angle(Tm / stb.loop_gain))
    # SKY130 matrix-cap residual peaks at 6e-3 dB at 10 GHz (cf. the m0
    # bench); below 1 GHz the match is < 5e-4 dB.
    assert np.max(np.abs(err_db)) < 1e-2
    assert np.max(np.abs(err_db[stb.freq <= 1e9])) < 1e-3
    assert np.max(np.abs(err_ph)) < 0.05
    t0 = complex(Tm[0])
    assert t0.real > 1.0 and abs(t0.imag) < 1e-6
    assert t0.real == pytest.approx(1.0040840, abs=2e-5)
    # the reference itself: Spectre says 1.0040976729
    assert stb.loop_gain[0].real == pytest.approx(1.0040976729, abs=1e-9)


def test_first_order_model_is_sub_unity():
    """{gm, gmbs, gds} alone lands on the WRONG side of unity: the exact
    symbolic 1 - T0 of this bench is a ratio of all-positive sums, so no
    first-order parameter values can reproduce the over-unity loop."""
    T = _analyzer("r2r_bias_pos_loop.cin.json").loop_gain("VPRB")
    t0 = complex(T.numeric([1e-3])[0])
    assert t0.real < 1.0
    assert t0.real == pytest.approx(0.9999366, abs=1e-6)


def test_symbolic_T0_over_unity_is_carried_by_gii(T_ii):
    """Hybrid solve with the 12 conductance symbols kept: substituting the
    fixture values reproduces T(0); zeroing only the gii symbols drops it
    below unity -- the regeneration is carried entirely by the
    impact-ionization terms."""
    keep = ["gm_MN0", "gmbs_MN0", "gds_MN0", "gm_MP0", "gmbs_MP0", "gds_MP0",
            "RSN", "RSP", "gii_d_MN0", "gii_m_MN0", "gii_d_MP0", "gii_m_MP0"]
    T = _analyzer("r2r_ii.cin.json").loop_gain("VPRB", keep=keep)
    s = sp.Symbol("s")
    T0 = sp.cancel(sp.together(T.expr.subs(s, 0)))
    subs = {T.symbols[k]: sp.Rational(repr(T.values[k]))
            for k in T.symbols if k in T.values}
    assert float(T0.xreplace(subs)) == pytest.approx(1.0040840, abs=1e-6)
    gii_zero = {T.symbols[k]: 0 for k in
                ("gii_d_MN0", "gii_m_MN0", "gii_d_MP0", "gii_m_MP0")}
    t0_noii = float(T0.xreplace(gii_zero).xreplace(subs))
    assert t0_noii < 1.0
    assert t0_noii == pytest.approx(0.9999366, abs=1e-6)


def test_negative_node_resistance_measured_and_reconstructed():
    """The ac sweep (IT injects 1 A into vsn, loop closed) measures the
    driving-point resistance directly: negative at low frequency, and
    R = R0/(1 - T0) with R0 > 0 makes the sign equivalent to T0 > 1.
    The gii reconstruction tracks the measurement; the first-order model
    predicts the wrong sign."""
    ac = load_ac(FIX / "psf" / "ac.ac")
    f = ac.freq
    z_sp = ac.wave("vsn")
    assert z_sp[0].real == pytest.approx(-3.335357e5, rel=1e-5)

    z_ii = _analyzer("r2r_ii.cin.json").impedance(node="vsn", keep=[]).numeric(f)
    rel = np.abs(z_ii - z_sp) / np.abs(z_sp)
    assert np.max(rel) < 5e-3
    assert complex(z_ii[0]).real < 0

    z0 = _analyzer("r2r_bias_pos_loop.cin.json").impedance(node="vsn", keep=[])
    assert complex(z0.numeric([1e-3])[0]).real > 0    # wrong sign without gii


def test_impact_ionization_flag():
    """The pure-OP advisory (session.impact_ionization_devices): the
    plain model flags MN0 (isub/ids ~ 1%, no gii modeled); adding the gii
    params silences it -- the r2r over-unity lesson turned into an
    automatic incompleteness warning."""
    import warnings

    from circuitinsight.session import SessionController

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plain = SessionController.open(FIX / "r2r_bias_pos_loop.cin.json",
                                       FIX / "psf")
        withgii = SessionController.open(FIX / "r2r_ii.cin.json", FIX / "psf")
    flagged = dict(plain.impact_ionization_devices())
    assert "MN0" in flagged and flagged["MN0"] > 0.005
    assert withgii.impact_ionization_devices() == []
