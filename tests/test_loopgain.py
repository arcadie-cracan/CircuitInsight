"""Tian loop gain from the reconstruction vs Spectre stb (M-B).

The probe bench (tb_ota2s_stb.cin.json + psf_stb) closes the two-stage in
unity feedback through IPRB0, a 0-V vsource standing in for the schematic
iprobe. `analyzer.loop_gain("IPRB0")` must reproduce Spectre's stb loopGain
-- sign convention included (arg T = +180 deg at DC) -- essentially exactly
over the whole band where the loop gain matters, and re-derive the
simulator's phase/gain margins.
"""
import warnings
from pathlib import Path

import numpy as np
import pytest

from circuitinsight.adapters.spectre import SpectreRun
from circuitinsight.adapters.spectre.stbdata import load_stb

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "spectre" / "miller"


@pytest.fixture(scope="module")
def T_and_ref():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run = SpectreRun(FIXTURES / "tb_ota2s_stb.cin.json", FIXTURES / "psf_stb")
        T = run.analyzer(cap_model="matrix").loop_gain("IPRB0")
    return T, load_stb(FIXTURES / "psf_stb")


def test_matches_stb_over_the_full_sweep(T_and_ref):
    """Eq. 30 (the bilateral return-loop model, reverse transmission
    included) matches Spectre's loopGain over the ENTIRE sweep to 10 GHz:
    measured worst 0.020 dB / 0.92 deg (the residual quasi-static model
    error). The earlier unilateral 2-readout combination erred ~1 dB in the
    deep-cancellation tail -- that was the reverse-transmission term the
    paper's Eq. 31 'normal return ratio' ignores, not model error."""
    T, stb = T_and_ref
    Tm = T.numeric(stb.freq)
    err_db = 20 * np.log10(np.abs(Tm) / np.abs(stb.loop_gain))
    err_ph = np.degrees(np.angle(Tm / stb.loop_gain))
    assert np.max(np.abs(err_db)) < 0.05
    assert np.max(np.abs(err_ph)) < 1.5
    # and essentially exact through the stability-relevant band
    band = 20 * np.log10(np.abs(stb.loop_gain)) >= -20.0
    assert np.max(np.abs(err_db[band])) < 0.01
    assert np.max(np.abs(err_ph[band])) < 0.05


def test_sign_convention_and_dc_value(T_and_ref):
    """Spectre's convention: T(DC) is negative real (arg = 180 deg) with
    |T(0)| = A0 of the amplifier in unity feedback."""
    T, stb = T_and_ref
    t0 = complex(T.numeric([1e-3])[0])
    assert t0.real < 0 and abs(t0.imag) < abs(t0.real) * 1e-3
    assert 20 * np.log10(abs(t0)) == pytest.approx(85.48, abs=0.05)
    assert abs(t0 - stb.loop_gain[0]) / abs(stb.loop_gain[0]) < 2e-3


def test_margins_match_spectre(T_and_ref):
    """PM and GM re-derived from the reconstructed T must land on the
    numbers in stb.margin.stb."""
    T, stb = T_and_ref
    f = np.logspace(np.log10(3e5), np.log10(3e7), 4001)
    Tm = T.numeric(f)
    m = 20 * np.log10(np.abs(Tm))
    ph = np.degrees(np.unwrap(np.angle(Tm)))

    k = np.where(np.diff(np.sign(m)))[0][0]
    x = np.log10(f)
    xu = np.interp(0, [m[k + 1], m[k]], [x[k + 1], x[k]])
    pm = np.interp(xu, [x[k], x[k + 1]], [ph[k], ph[k + 1]])
    assert pm == pytest.approx(stb.phase_margin_deg, abs=0.1)
    assert 10 ** xu == pytest.approx(stb.phase_margin_freq_hz, rel=2e-3)

    j = np.where(np.diff(np.sign(ph)))[0][0]
    xj = np.interp(0, [ph[j + 1], ph[j]], [x[j + 1], x[j]])
    gm = -np.interp(xj, [x[j], x[j + 1]], [m[j], m[j + 1]])
    assert gm == pytest.approx(stb.gain_margin_db, abs=0.05)
    assert 10 ** xj == pytest.approx(stb.gain_margin_freq_hz, rel=2e-3)


def test_probe_must_be_a_branch(T_and_ref):
    from circuitinsight.engine.mna import MnaError

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run = SpectreRun(FIXTURES / "tb_ota2s_stb.cin.json", FIXTURES / "psf_stb")
        an = run.analyzer(cap_model="matrix")
        with pytest.raises((MnaError, KeyError)):
            an.loop_gain("IB")           # an isource: no branch row


# ------------------------------------------------------------- M-C: symbolic


@pytest.fixture(scope="module")
def T_cc():
    """T(s) with the compensation capacitor kept symbolic."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run = SpectreRun(FIXTURES / "tb_ota2s_stb.cin.json", FIXTURES / "psf_stb")
        return run.analyzer(cap_model="matrix").loop_gain("IPRB0", keep=["I0.Cc"])


def test_symbolic_cc_reduces_to_numeric(T_and_ref, T_cc):
    """Substituting the fixture's Cc into the symbolic T must reproduce the
    numeric T exactly (the hybrid solve is exact, not approximate)."""
    import sympy as sp

    T_num, stb = T_and_ref
    ccsym = next(s for n, s in T_cc.symbols.items() if n == "I0_Cc")
    e = T_cc.expr.xreplace({ccsym: sp.Rational(repr(T_cc.values["I0_Cc"]))})
    fixed = type(T_cc)(expr=e, values=T_cc.values, symbols=T_cc.symbols)
    f = np.logspace(0, 9, 40)
    a, b = fixed.numeric(f), T_num.numeric(f)
    assert np.max(np.abs(a - b) / np.abs(b)) < 1e-9


def _pm_of_T(T, cc=None):
    import sympy as sp

    if cc is not None:
        ccsym = next(s for n, s in T.symbols.items() if n == "I0_Cc")
        T = type(T)(expr=T.expr.xreplace({ccsym: sp.Float(cc)}),
                    values=T.values, symbols=T.symbols)
    f = np.logspace(5.5, 7.5, 4001)
    Tm = T.numeric(f)
    m = 20 * np.log10(np.abs(Tm))
    ph = np.degrees(np.unwrap(np.angle(Tm)))
    k = np.where(np.diff(np.sign(m)))[0][0]
    x = np.log10(f)
    xu = np.interp(0, [m[k + 1], m[k]], [x[k + 1], x[k]])
    return np.interp(xu, [x[k], x[k + 1]], [ph[k], ph[k + 1]])


def test_symbolic_cc_predicts_pm_of_every_compensation(T_cc, T_and_ref):
    """The design-aid payoff: ONE symbolic T(s; Cc) from ONE DC solve
    predicts the phase margin of every compensation state. Reference points
    are the Spectre-measured PMs of the closed-form-sizing benches
    (psf_cc732/psf_cc862, measured 48.1/51.3 deg as forward-gain PM; the
    loop-gain PM differs from those by the ~0.3 deg loading term, cf.
    60.2 ac vs 59.90 stb at 16 pF)."""
    _, stb = T_and_ref
    assert _pm_of_T(T_cc, cc=16e-12) == pytest.approx(
        stb.phase_margin_deg, abs=0.1)                     # stb: same quantity
    assert _pm_of_T(T_cc, cc=7.32e-12) == pytest.approx(48.1, abs=0.7)
    assert _pm_of_T(T_cc, cc=8.62e-12) == pytest.approx(51.3, abs=0.7)


def test_two_symbol_T_carries_the_miller_structure():
    """Keeping {gm5, Cc}: the dominant (RHP) zero of T must be the paper's
    z = gm5/Cc in symbolic form -- the phase-margin ceiling's origin."""
    import sympy as sp

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run = SpectreRun(FIXTURES / "tb_ota2s_stb.cin.json", FIXTURES / "psf_stb")
        T = run.analyzer(cap_model="matrix").loop_gain(
            "IPRB0", keep=["gm_I0_MP2", "I0.Cc"])
    syms = {str(s) for s in T.expr.free_symbols}
    assert {"gm_I0_MP2", "I0_Cc"} <= syms
    # numeric anchor: at the fixture values it still reproduces the stb PM
    sub = {s: sp.Float(T.values[n]) for n, s in T.symbols.items()
           if n in ("gm_I0_MP2", "I0_Cc")}
    fixed = type(T)(expr=T.expr.xreplace(sub), values=T.values, symbols=T.symbols)
    assert _pm_of_T(fixed) == pytest.approx(59.898, abs=0.1)


# ---------------------------------------------------------- M-D: return ratio


def test_return_ratio_equals_global_loop_at_dc(T_and_ref):
    """Bode RR of the second-stage gm, via the return-difference determinant
    identity. At DC the Miller capacitor is open, so gm5 sits in exactly one
    loop -- the global unity-feedback loop -- and RR(0) must equal |T(0)|."""
    T, stb = T_and_ref
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run = SpectreRun(FIXTURES / "tb_ota2s_stb.cin.json", FIXTURES / "psf_stb")
        RR = run.analyzer(cap_model="matrix").return_ratio("gm_I0.MP2")
    r0 = complex(RR.numeric([1e-3])[0])
    assert r0.real > 0 and abs(r0.imag) < abs(r0.real) * 1e-3   # Bode sign
    assert abs(r0) == pytest.approx(abs(stb.loop_gain[0]), rel=1e-4)


def test_return_ratio_exposes_the_local_miller_loop(T_and_ref):
    """In-band, RR(gm5) is NOT the probe loop gain: gm5 also closes the
    LOCAL Miller loop through C_C, so its return ratio exceeds the global
    T by ~9-16 dB (measured +11.8 dB at the global PM crossing) and its
    'phase margin' lands elsewhere (68 deg @ 21.7 MHz vs 59.9 @ 3.5 MHz).
    Element return ratios and probe loop gains answer different questions
    on multi-loop amplifiers -- exactly the distinction worth teaching."""
    T, stb = T_and_ref
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run = SpectreRun(FIXTURES / "tb_ota2s_stb.cin.json", FIXTURES / "psf_stb")
        RR = run.analyzer(cap_model="matrix").return_ratio("gm_I0.MP2")
    k = np.argmin(np.abs(stb.freq - stb.phase_margin_freq_hz))
    ratio_db = 20 * np.log10(abs(complex(RR.numeric([stb.freq[k]])[0]))
                             / abs(stb.loop_gain[k]))
    assert 9 < ratio_db < 15


def test_return_ratio_hybrid_reduces_to_numeric():
    import sympy as sp

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run = SpectreRun(FIXTURES / "tb_ota2s_stb.cin.json", FIXTURES / "psf_stb")
        an = run.analyzer(cap_model="matrix")
        RR = an.return_ratio("gm_I0.MP2")
        RRc = an.return_ratio("gm_I0.MP2", keep=["I0.Cc"])
    cc = next(s for n, s in RRc.symbols.items() if n == "I0_Cc")
    fixed = type(RRc)(
        expr=RRc.expr.xreplace({cc: sp.Rational(repr(RRc.values["I0_Cc"]))}),
        values=RRc.values, symbols=RRc.symbols)
    f = np.logspace(0, 9, 30)
    a, b = fixed.numeric(f), RR.numeric(f)
    assert np.max(np.abs(a - b) / np.abs(b)) < 1e-9


def test_return_ratio_of_a_matched_pair_is_joint():
    """A shared (matched) symbol returns the pair's JOINT return ratio --
    all stamps carrying the symbol are zeroed together."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run = SpectreRun(FIXTURES / "tb_ota2s_stb.cin.json", FIXTURES / "psf_stb")
        an = run.analyzer(cap_model="matrix")
        an.match("I0.MN0", "I0.MN1")
        RR = an.return_ratio("gm_I0.MN0")
    r0 = complex(RR.numeric([1e-3])[0])
    assert np.isfinite(r0) and abs(r0) == pytest.approx(9.94e4, rel=0.01)


def test_return_ratio_unknown_source_lists_candidates():
    from circuitinsight.engine.mna import MnaError

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run = SpectreRun(FIXTURES / "tb_ota2s_stb.cin.json", FIXTURES / "psf_stb")
        with pytest.raises(MnaError, match="close matches"):
            run.analyzer(cap_model="matrix").return_ratio("gm_I0.MP9")


# ------------------------------------------------- M-E: SessionController


def test_session_loop_gain_result():
    """The front-end surface: probes discovered from the schematic meta, T
    packaged as a Result with the stb overlay and margins, bode_figure-ready."""
    from circuitinsight import SessionController

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c = SessionController.open(FIXTURES / "tb_ota2s_stb.cin.json",
                                   FIXTURES / "psf_stb", cap_model="matrix")
        assert c.probes[0] == "IPRB0"   # iprobe first; any vsource qualifies
        r = c.loop_gain("IPRB0")

    assert r.out == "T@IPRB0"
    assert r.pm_deg == pytest.approx(59.898, abs=0.1)
    assert r.pm_freq_hz == pytest.approx(3.489e6, rel=2e-3)
    assert r.gm_db == pytest.approx(9.015, abs=0.05)
    assert r.gm_freq_hz == pytest.approx(1.255e7, rel=2e-3)
    # overlay is the run's own stb sweep, point-aligned
    assert r.h_ref is not None and "stb" in r.ref_label
    assert len(r.h) == len(r.h_ref) == len(r.freqs)
    err = 20 * np.log10(np.abs(r.h) / np.abs(r.h_ref))
    band = 20 * np.log10(np.abs(r.h_ref)) >= -20
    assert np.max(np.abs(err[band])) < 0.1
    # stable, healthy margins: no advisories beyond none expected
    assert not any("UNSTABLE" in w or "low phase" in w for w in r.warnings)
    # cached
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert c.loop_gain("IPRB0") is r


def test_session_loop_gain_renders_in_bode_figure():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    from circuitinsight import SessionController
    from circuitinsight.gui import view

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c = SessionController.open(FIXTURES / "tb_ota2s_stb.cin.json",
                                   FIXTURES / "psf_stb", cap_model="matrix")
        r = c.loop_gain("IPRB0")
    fig = view.bode_figure(r)
    FigureCanvasAgg(fig).draw()
    assert len(fig.axes) == 2


def test_session_loop_gain_without_stb_reference():
    """On a run without stb results the Result degrades gracefully: model-only
    sweep, margins still computed, an advisory notes the missing overlay."""
    from circuitinsight import SessionController

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c = SessionController.open(FIXTURES / "tb_ota2s_stb.cin.json",
                                   FIXTURES / "psf_stb", cap_model="matrix")
        # point the adapter's stb at a missing file via a bogus name
        r = c.loop_gain("IPRB0", keep=(), reference=False)
    assert r.h_ref is None and r.pm_deg is not None
