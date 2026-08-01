"""Standard multistage form + error-budgeted composition.

The template is a VIEW of an exact solve: every formula it prints is
checked against the exact numeric root, so the tests assert both that
the textbook formulas come back AND that unreliable ones are labelled
rather than presented."""
import warnings
from pathlib import Path

import pytest
import sympy as sp

from circuitinsight import Analyzer
from circuitinsight.analysis import tearing
from circuitinsight.analysis.template import template_form

FIX = Path(__file__).resolve().parent / "fixtures" / "spectre"


def _rc_cascade(c2="1p"):
    """Two RC sections, decades apart: p1 = 1/(R1 C1), p2 = 1/(R2 C2)."""
    return {"cin_version": "0.1", "top": "main", "ground": ["0"],
            "definitions": {"main": {"ports": [], "instances": [
                {"name": "V1", "device_type": "vsource",
                 "terminals": {"p": "in", "n": "0"}},
                {"name": "R1", "device_type": "resistor",
                 "terminals": {"p": "in", "n": "m"}, "params": {"r": "1k"}},
                {"name": "C1", "device_type": "capacitor",
                 "terminals": {"p": "m", "n": "0"}, "params": {"c": "1u"}},
                {"name": "R2", "device_type": "resistor",
                 "terminals": {"p": "m", "n": "out"}, "params": {"r": "1k"}},
                {"name": "C2", "device_type": "capacitor",
                 "terminals": {"p": "out", "n": "0"}, "params": {"c": c2}},
            ]}}}


def test_template_recovers_the_textbook_rc_poles():
    an = Analyzer.from_cin(_rc_cascade())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        h = an.tf("V1", "out", keep=["R1", "C1", "R2", "C2"],
                  method="interp")
    tpl = template_form(h)
    assert len(tpl.poles) == 2
    # the solver's symbols carry assumptions (positive=True); plain
    # sp.symbols() would be DIFFERENT objects and never compare equal
    R1, C1 = h.symbols["R1"], h.symbols["C1"]
    R2, C2 = h.symbols["R2"], h.symbols["C2"]
    p0, p1 = tpl.poles[0], tpl.poles[1]
    assert p0.reliable and p1.reliable
    assert sp.simplify(p0.expr - 1 / (R1 * C1)) == 0
    assert sp.simplify(p1.expr - 1 / (R2 * C2)) == 0
    assert p0.f_hz < p1.f_hz
    assert "1/(C1*R1)" in tpl.describe().replace(" ", "")


def test_template_recovers_the_5t_ota_dc_gain():
    """A0 = gm/gds, pruned out of the exact hybrid rational."""
    from circuitinsight.adapters.spectre import SpectreRun

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run = SpectreRun(FIX / "ota5t" / "tb_ota5t.cin.json",
                         FIX / "ota5t" / "psf")
        an = run.analyzer(cap_model="matrix")
        an.match("I0.MN0", "I0.MN1")
        an.match("I0.MP0", "I0.MP1")
        h = an.tf("VIND", "vout",
                  keep=["gm_I0_MN0", "gds_I0_MN0", "gds_I0_MP0"],
                  method="interp")
    tpl = template_form(h, budget=0.25)
    gm, gds = h.symbols["gm_I0_MN0"], h.symbols["gds_I0_MN0"]
    assert sp.simplify(tpl.dc_gain - gm / gds) == 0
    assert 150 < tpl.dc_gain_value < 300
    assert tpl.gbw_hz and tpl.gbw_hz > 1e6


def test_template_labels_unreliable_formulas_instead_of_presenting_them():
    """Clustered roots break root splitting: the formula must be marked,
    not quietly printed as if it held."""
    an = Analyzer.from_cin(_rc_cascade(c2="0.9u"))   # both poles ~ same
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        h = an.tf("V1", "out", keep=["R1", "C1", "R2", "C2"],
                  method="interp")
    tpl = template_form(h, budget=0.01)
    assert any((not r.reliable) or r.expr is None for r in tpl.poles)
    text = tpl.describe()
    assert "UNRELIABLE" in text or "no closed form" in text


def test_template_needs_numeric_values():
    from circuitinsight.engine.mna import MnaError, TransferFunction

    tf = TransferFunction(expr=sp.Symbol("x") / sp.Symbol("s"))
    with pytest.raises(MnaError):
        template_form(tf)


# ------------------------------------------------- budgeted composition
def _two_stage_cin():
    inst = [
        {"name": "V1", "device_type": "vsource",
         "terminals": {"p": "in", "n": "0"}},
        {"name": "G1", "device_type": "vccs",
         "terminals": {"p": "0", "n": "m", "cp": "in", "cn": "0"},
         "params": {"gm": "1m"}},
        {"name": "RM", "device_type": "resistor",
         "terminals": {"p": "m", "n": "0"}, "params": {"r": "10k"}},
        {"name": "CM", "device_type": "capacitor",
         "terminals": {"p": "m", "n": "0"}, "params": {"c": "1p"}},
        {"name": "RS", "device_type": "resistor",
         "terminals": {"p": "m", "n": "0"}, "params": {"r": "900k"}},
        {"name": "G2", "device_type": "vccs",
         "terminals": {"p": "0", "n": "out", "cp": "m", "cn": "0"},
         "params": {"gm": "2m"}},
        {"name": "RL", "device_type": "resistor",
         "terminals": {"p": "out", "n": "0"}, "params": {"r": "5k"}},
        {"name": "CL", "device_type": "capacitor",
         "terminals": {"p": "out", "n": "0"}, "params": {"c": "2p"}},
    ]
    return {"cin_version": "0.1", "top": "main", "ground": ["0"],
            "definitions": {"main": {"ports": [], "instances": inst}}}


def test_budgeted_composition_stays_inside_its_budget():
    """Guerra's per-level error control: shrink each interface block under
    a share of the budget, compose, and MEASURE what it cost."""
    an = Analyzer.from_cin(_two_stage_cin())
    keep = ["G1", "G2", "RM", "RS", "RL", "CL"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        exact = tearing.split_tf(an.primitives, an.flat.ground, "V1",
                                 "out", "m", keep=keep)
        approx = tearing.split_tf(an.primitives, an.flat.ground, "V1",
                                  "out", "m", keep=keep, budget_db=1.0,
                                  fmin=1e3, fmax=1e10)
    assert isinstance(approx, tearing.ComposedTF)
    assert approx.blocks_total > 0
    assert approx.achieved_mag_err_db <= 1.0 + 1e-9
    assert "budgeted composition" in approx.report()
    # and the exact path is untouched by the budget machinery
    assert sp.simplify(sp.together(exact.expr - approx.exact.expr)) == 0


def test_budgeted_composition_is_exact_when_no_budget_is_given():
    an = Analyzer.from_cin(_two_stage_cin())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        h = tearing.split_tf(an.primitives, an.flat.ground, "V1", "out",
                             "m", keep=["G1", "G2"])
        mono = an.tf("V1", "out", keep=["G1", "G2"], method="interp")
    assert not isinstance(h, tearing.ComposedTF)
    assert sp.simplify(sp.together(h.expr - mono.expr)) == 0


def test_template_finds_roots_when_coefficients_overflow_float64():
    """Companion to the folded-cascode overflow: root-finding must scale the
    coefficient vector first. Polynomial roots are invariant to an overall
    factor, so without the scaling the finite-check below rejects the whole
    vector and the template reports a circuit with no poles at all."""
    from circuitinsight.engine.mna import TransferFunction

    s = sp.Symbol("s")
    big = sp.Integer(10) ** 320
    num = big * (1 + s / sp.Integer(10) ** 6)
    den = (big * sp.Integer(10) ** 12 * (1 + s / sp.Integer(10) ** 3)
           * (1 + s / sp.Integer(10) ** 9))
    t = template_form(TransferFunction(expr=num / den))

    assert t.dc_gain_value == pytest.approx(1e-12, rel=1e-9)
    assert [p.f_hz for p in t.poles] == pytest.approx(
        [1e3 / (2 * 3.141592653589793), 1e9 / (2 * 3.141592653589793)],
        rel=1e-9)
    assert t.zeros[0].f_hz == pytest.approx(1e6 / (2 * 3.141592653589793),
                                            rel=1e-9)


def test_template_reports_the_LOWEST_poles_not_an_arbitrary_subset():
    """A 17-pole folded cascode exposed this: np.roots returns companion
    eigenvalues in no useful order, and the template keeps only the first
    max_roots of them. Unsorted, it reported five poles starting at 328 MHz
    for an amplifier whose dominant pole is at 20 kHz, and GBW = A0 * p1
    came out 204 GHz. The reported poles must be the LOWEST ones."""
    from circuitinsight.engine.mna import TransferFunction

    s = sp.Symbol("s")
    # decades apart, deliberately built so the leading coefficient order
    # does not coincide with the frequency order
    fs = [1e2, 1e4, 1e6, 1e8, 1e10]
    den = sp.prod([1 + s / sp.Integer(int(f)) for f in fs])
    t = template_form(TransferFunction(expr=100 / den), max_roots=3)

    got = [p.f_hz * 2 * 3.141592653589793 for p in t.poles]
    assert got == pytest.approx(fs[:3], rel=1e-6)
    assert t.gbw_hz == pytest.approx(100 * fs[0] / (2 * 3.141592653589793),
                                     rel=1e-6)
