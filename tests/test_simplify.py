"""M4 simplification engine: error-budgeted pruning to textbook forms."""
import warnings
from pathlib import Path

import numpy as np
import pytest
import sympy as sp

from circuitinsight import Analyzer
from circuitinsight.adapters.spectre import SpectreRun
from circuitinsight.engine.mna import MnaError

GOLDEN = Path(__file__).resolve().parent / "golden" / "circuits"
OTA = Path(__file__).resolve().parent / "fixtures" / "spectre" / "ota5t"
s = sp.Symbol("s")


def test_cs_amp_prunes_to_textbook_form():
    H = Analyzer.from_cin(GOLDEN / "cs_amp.cin.json").tf("V1", "vout")
    Hs = H.simplify(mag_tol_db=0.5, phase_tol_deg=3, fmin=1e3, fmax=1e9)
    y = Hs.symbols
    gm, gds, cgd = y["gm_M1"], y["gds_M1"], y["cgd_M1"]
    RL, CL = y["RL"], y["CL"]

    expected = RL * (cgd * s - gm) / (CL * RL * s + RL * gds + 1)
    assert sp.simplify(Hs.expr - expected) == 0
    # cdb (1.7f vs CL=100f) pruned; gds (8.5% of 1/RL) must survive at 0.5 dB
    free = {str(x) for x in Hs.expr.free_symbols}
    assert "cdb_M1" not in free and "gds_M1" in free
    assert Hs.achieved_mag_err_db <= 0.5
    assert Hs.achieved_phase_err_deg <= 3.0

    assert sp.simplify(Hs.dc_gain_expr() + RL * gm / (RL * gds + 1)) == 0
    assert sp.simplify(Hs.dominant_zero_expr() + gm / cgd) == 0
    assert sp.simplify(
        Hs.dominant_pole_expr() - (RL * gds + 1) / (CL * RL)) == 0


def test_ota5t_golden_prunes_to_single_pole():
    an = Analyzer.from_cin(GOLDEN / "ota5t.cin.json")
    an.match("M1", "M2")
    an.match("M3", "M4")
    H = an.tf("V1", "vout")
    Hs = H.simplify(mag_tol_db=0.5, phase_tol_deg=3, fmin=1e2, fmax=1e8)
    y = Hs.symbols
    gm, g1, g3 = y["gm_M1"], y["gds_M1"], y["gds_M3"]
    EP, EN, CL = y["EP"], y["EN"], y["CL"]

    expected = -gm * (EN - EP) / (CL * s + g1 + g3)
    assert sp.simplify(Hs.expr - expected) == 0


def test_real_ota_dc_gain_is_textbook():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        run = SpectreRun(OTA / "tb_ota5t.cin.json", OTA / "psf")
        an = run.analyzer(cap_model="matrix")      # SKY130 lumped errs ~28 dB
        an.match("I0.MN0", "I0.MN1")
        an.match("I0.MP0", "I0.MP1")
        H = an.tf("VIND", "vout",
                  keep=["gm_I0_MN0", "gds_I0_MN0", "gm_I0_MP0", "gds_I0_MP0"])
    Hs = H.simplify(mag_tol_db=1.0, phase_tol_deg=5, fmin=1e3, fmax=1e9)
    y = Hs.symbols
    gm, gn, gp = y["gm_I0_MN0"], y["gds_I0_MN0"], y["gds_I0_MP0"]

    # the textbook result gm/(gds,N+gds,P), extracted from the full bench (balun,
    # switches...). On SKY130 the exact DC gain carries a ~3% input-pair
    # body-effect residual (bulk to ground) that the textbook form drops, so the
    # match is numeric-to-3%, not exact-symbolic.
    subs = Hs._subs_map()
    textbook = complex((gm / (gn + gp)).xreplace(subs)).real
    assert complex(Hs.dc_gain_expr().xreplace(subs)).real == \
        pytest.approx(textbook, rel=0.03)
    assert complex(Hs.numeric([1e3])[0]).real == pytest.approx(197.17, rel=1e-3)
    assert Hs.achieved_mag_err_db <= 1.0

    # pruning must actually shrink the expression (terms, not symbols)
    def n_terms(tf):
        num, den = tf.num_den
        return sum(len(sp.Add.make_args(sp.expand(c)))
                   for poly in (num, den) for _, c in poly.terms())

    assert n_terms(Hs) < n_terms(H)


def test_miller_ota_m4_exit_criterion():
    """PLAN.md M4 exit: the two-stage OTA produces the recognizable Av0,
    dominant-pole, and RHP-zero expressions automatically."""
    an = Analyzer.from_cin(GOLDEN / "miller_ota.cin.json")
    an.match("M1", "M2")
    an.match("M3", "M4")
    H = an.tf("V1", "vout")
    Hs = H.simplify(mag_tol_db=1.0, phase_tol_deg=5, fmin=10, fmax=1e9)
    y = Hs.symbols
    gm1, g1, g3 = y["gm_M1"], y["gds_M1"], y["gds_M3"]
    gm5, g5, g6 = y["gm_M5"], y["gds_M5"], y["gds_M6"]
    EP, EN, CC, CL = y["EP"], y["EN"], y["CC"], y["CL"]

    expected = (-gm1 * (EN - EP) * (CC * s - gm5)
                / (CC * CL * s**2 + CC * gm5 * s
                   + (g1 + g3) * (g5 + g6)))
    assert sp.simplify(Hs.expr - expected) == 0
    assert sp.simplify(
        Hs.dc_gain_expr() - gm1 * gm5 * (EN - EP) / ((g1 + g3) * (g5 + g6))) == 0
    assert sp.simplify(
        Hs.dominant_pole_expr() - (g1 + g3) * (g5 + g6) / (CC * gm5)) == 0
    assert sp.simplify(Hs.dominant_zero_expr() + gm5 / CC) == 0
    assert Hs.pole_separation() > 1e5              # deep pole splitting
    assert Hs.achieved_mag_err_db <= 1.0


def test_tiny_budget_keeps_small_terms():
    H = Analyzer.from_cin(GOLDEN / "cs_amp.cin.json").tf("V1", "vout")
    Hs = H.simplify(mag_tol_db=0.01, phase_tol_deg=0.1, fmin=1e3, fmax=1e9)
    free = {str(x) for x in Hs.expr.free_symbols}
    assert "cdb_M1" in free                      # too tight to drop anything
    assert Hs.achieved_mag_err_db <= 0.01


def test_missing_values_raise():
    an = Analyzer.from_cin({
        "cin_version": "0.1", "top": "main", "ground": ["0"],
        "definitions": {"main": {"ports": [], "instances": [
            {"name": "V1", "device_type": "vsource",
             "terminals": {"p": "vin", "n": "0"}},
            {"name": "R1", "device_type": "resistor",
             "terminals": {"p": "vin", "n": "vout"}},
            {"name": "R2", "device_type": "resistor",
             "terminals": {"p": "vout", "n": "0"}},
        ]}}})
    H = an.tf("V1", "vout")
    with pytest.raises(MnaError, match="R1"):
        H.simplify()


def test_huge_exact_coefficients_do_not_overflow():
    """Regression, from a 36-device folded-cascode OTA solved with keep=[]:
    the exact coefficients of N(s) and D(s) are integers with hundreds of
    digits, so converting one to float64 raises "int too large to convert
    to float". The RATIO is O(gain) and fits in float64 even when neither
    part does, so the evaluator works in extended precision and the pruner
    compares terms only after scaling by the largest of them."""
    from circuitinsight.engine.mna import TransferFunction, numeric_eval

    big = sp.Integer(10) ** 320
    num = big * (1 + s / sp.Integer(10) ** 6)
    den = big * sp.Integer(10) ** 12 * (1 + s / sp.Integer(10) ** 3)
    with pytest.raises(OverflowError):               # the failure it guards
        sp.lambdify(s, sp.expand(num), "numpy")(np.array([2j * np.pi]))

    f = np.array([1.0, 1e3, 1e9])
    got = numeric_eval(num / den, s, f)
    ref = (1 + 2j * np.pi * f / 1e6) / (1e12 * (1 + 2j * np.pi * f / 1e3))
    assert np.allclose(got, ref, rtol=1e-9)

    tf = TransferFunction(expr=num / den)
    Hs = tf.simplify(mag_tol_db=1.0, phase_tol_deg=5, fmin=1.0, fmax=1e9)
    assert Hs.achieved_mag_err_db <= 1.0


def test_pruner_scales_before_comparing_terms():
    """Same overflow, one level down: a kept-symbol coefficient is a SUM of
    huge terms, and the pruning decision (which terms are negligible) is
    scale-invariant -- so it must be taken on the scaled ratios."""
    from circuitinsight.analysis.simplify import _prune_poly

    a, b = sp.symbols("a b", positive=True)
    big = sp.Integer(10) ** 320
    poly = sp.Poly(big * a * s + (big * b + big * a / 10 ** 6), s)
    subs = {a: sp.Float(1.0), b: sp.Float(1.0)}
    kept = _prune_poly(poly, 1e-3, subs)
    assert kept[1] == big * a                        # lone term, untouched
    assert kept[0] == big * b                        # a/1e6 pruned away


def test_degenerate_prune_candidate_is_rejected_not_a_crash():
    """Found live on the folded cascode at a 10 dB budget: a prune
    candidate whose denominator collapsed evaluates to ComplexInfinity,
    and lambdify died with KeyError('ComplexInfinity') deep in its
    printer, killing the whole solve. A degenerate candidate is a bad
    VALUE: numeric_eval returns NaNs, the error metric goes infinite,
    and the pruner backs off."""
    from circuitinsight.engine.mna import numeric_eval

    bad = sp.zoo * s / (s + 1)
    out = numeric_eval(bad, s, np.array([1.0, 1e3]))
    assert out.shape == (2,)
    assert np.all(np.isnan(out.real))
