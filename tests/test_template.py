"""Standard multistage form + error-budgeted composition.

The template is a VIEW of an exact solve: every formula it prints is
checked against the exact numeric root, so the tests assert both that
the textbook formulas come back AND that unreliable ones are labelled
rather than presented."""
import json
import warnings
from pathlib import Path

import numpy as np
import pytest
import sympy as sp

from circuitinsight import Analyzer
from circuitinsight.analysis import tearing
from circuitinsight.analysis import template as template_mod
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
    not quietly printed as if it held. Two clustered roots are a
    quadratic's (see the pair tests below); three identical RC sections
    are a cluster no quadratic over adjacent coefficients holds."""
    cin = _rc_cascade()
    inst = cin["definitions"]["main"]["instances"]
    inst[4]["params"]["c"] = "1u"
    inst[3]["terminals"]["p"], inst[3]["terminals"]["n"] = "m", "m2"
    inst[4]["terminals"]["p"] = "m2"
    inst += [{"name": "R3", "device_type": "resistor",
              "terminals": {"p": "m2", "n": "out"}, "params": {"r": "1k"}},
             {"name": "C3", "device_type": "capacitor",
              "terminals": {"p": "out", "n": "0"}, "params": {"c": "1u"}}]
    an = Analyzer.from_cin(cin)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        h = an.tf("V1", "out",
                  keep=["R1", "C1", "R2", "C2", "R3", "C3"],
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


# ------------------------------------------------ root pairs (quadratics)
def _tf(expr, **vals):
    from circuitinsight.engine.mna import TransferFunction

    syms = {str(x): x for x in expr.free_symbols if str(x) != "s"}
    return TransferFunction(expr=expr, values=vals, symbols=syms)


def _pos(names):
    return sp.symbols(names, positive=True)


def test_template_complex_pair_is_w0_and_zeta_from_the_quadratic():
    """A dominant pole times a complex pair (Hurst et al. 2004, eqs. (29)
    and (58)): D = a0 + a1 s + a2 s^2 + a3 s^3 does not split into single
    ratios, but the pair is the root of a1 + a2 s + a3 s^2, so
    w0 = sqrt(a1/a3) and 2 zeta w0 = a2/a3."""
    s = sp.Symbol("s")
    A, B, C = _pos("A B C")
    # p1 = 1e3 rad/s, the pair at w0 = 1e8 rad/s with zeta = 0.4
    tau, b, c = 1e-3, 0.8e-8, 1e-16
    t = template_form(_tf(1 / (1 + A * s + B * s ** 2 + C * s ** 3),
                          A=tau + b, B=tau * b + c, C=tau * c))
    assert len(t.poles) == 2
    p = t.poles[1]
    assert p.damping == pytest.approx(0.4, rel=1e-3)
    assert p.reliable
    assert sp.simplify(p.expr ** 2 - A / C) == 0
    assert sp.simplify(p.zeta_w0 - B / C) == 0
    assert p.displacement < 1e-3 and p.zeta_displacement < 1e-3
    assert "w0 = sqrt(A/C)" in t.describe()
    assert "no closed form" not in t.describe()


def test_template_complex_pair_prunes_inside_its_coefficients():
    """The pair's coefficients are pruned like a single root's ratio:
    a term that cannot move the pair is dropped from w0."""
    s = sp.Symbol("s")
    A, B, C, D = _pos("A B C D")
    tau, b, c = 1e-3, 0.8e-8, 1e-16
    t = template_form(_tf(1 / (1 + A * s + B * s ** 2 + (C + D) * s ** 3),
                          A=tau + b, B=tau * b + c, C=tau * c,
                          D=tau * c * 1e-4))
    p = t.poles[1]
    assert p.reliable
    assert sp.simplify(p.expr ** 2 - A / C) == 0


def test_template_even_zero_pair_prints_plus_minus_sqrt():
    """A LHP/RHP zero pair from n0 + n1 s + n2 s^2 with n1 small
    (Hurst et al. 2004, eqs. (56)/(57)): +/- sqrt(-n0/n2)."""
    s = sp.Symbol("s")
    G5, G7, CA, Cc, e = _pos("G5 G7 CA Cc e")
    num = G5 * G7 - e * s - CA * Cc * s ** 2
    t = template_form(_tf(num / (1 + s * 1e-3),
                          G5=1e-3, G7=1e-3, CA=1e-12, Cc=1e-12, e=1e-17))
    z0, z1 = t.zeros
    assert z0.plusminus and z1.plusminus
    assert z0.reliable and z1.reliable
    assert sp.simplify(z0.expr ** 2 - G5 * G7 / (CA * Cc)) == 0
    assert z0.rhp != z1.rhp
    assert z1.partner == z0.index
    text = t.describe()
    assert "+/- sqrt(" in text and "no closed form" not in text
    assert "RHP" in text


def test_template_real_pair_keeps_n1_when_it_matters():
    """A LHP/RHP pair with a middle term that cannot go: each root gets
    its own magnitude from the quadratic formula."""
    s = sp.Symbol("s")
    a, b, c = _pos("a b c")
    t = template_form(_tf((a - b * s - c * s ** 2) / (1 + s * 1e-3),
                          a=1e18, b=5e8, c=1.0))
    exact = sorted(abs(x) for x in
                   [(-5e8 + (25e16 + 4e18) ** 0.5) / 2,
                    (-5e8 - (25e16 + 4e18) ** 0.5) / 2])
    assert len(t.zeros) == 2
    for z, w in zip(t.zeros, exact):
        assert z.reliable and not z.plusminus
        v = float(z.expr.xreplace({a: 1e18, b: 5e8, c: 1.0}))
        assert v == pytest.approx(w, rel=1e-9)


def test_template_reads_exact_factors_before_matching_ratios():
    """Miller OTA with a mirror doublet (Palmisano & Palumbo 1999): the
    exact (CA s + gm3) factor sits between the Miller stage's poles, so
    ratios of the expanded coefficients miss it and the pole above it;
    read as a factor, it is gm3/CA exactly."""
    s = sp.Symbol("s")
    CA, gm3, T1, T2 = _pos("CA gm3 T1 T2")
    t = template_form(_tf(1 / ((CA * s + gm3) * (1 + T1 * s + T2 * s ** 2)),
                          CA=1e-12, gm3=2e-6, T1=1e-3, T2=1e-9))
    assert len(t.poles) == 3
    assert [p.f_hz * 2 * 3.141592653589793 for p in t.poles] == \
        pytest.approx([1e3, 1e6, 2e6], rel=2e-3)
    assert all(p.reliable for p in t.poles)
    assert sp.simplify(t.poles[2].expr - gm3 / CA) == 0
    assert sp.simplify(t.poles[0].expr - 1 / T1) == 0
    assert sp.simplify(t.poles[1].expr - T1 / T2) == 0


def test_template_splits_a_quadratic_that_factors():
    """The simplified 1999 OTA: a1 + a2 s + a3 s^2 is
    (CA s + gm3)(CL s + gm5) times a constant although the cubic does not
    factor; the two closely spaced poles are its linear factors."""
    s = sp.Symbol("s")
    CA, CL, Cc, R, gm3, gm5 = _pos("CA CL Cc R gm3 gm5")
    den = (gm3 + Cc * R * gm3 * gm5 * s
           + (CA * Cc * R * gm5 + CL * Cc * R * gm3) * s ** 2
           + CA * CL * Cc * R * s ** 3)
    t = template_form(_tf(1 / den, CA=0.3e-12, CL=1e-12, Cc=1e-12, R=1e9,
                          gm3=75e-6, gm5=420e-6))
    p = t.poles
    assert len(p) == 3 and all(x.reliable for x in p)
    assert sp.simplify(p[1].expr - gm3 / CA) == 0
    assert sp.simplify(p[2].expr - gm5 / CL) == 0


def test_template_closely_spaced_real_pair_gets_the_quadratic():
    """Two RC sections with nearly equal time constants: the single
    ratios fail, the quadratic over the whole denominator holds."""
    an = Analyzer.from_cin(_rc_cascade(c2="0.9u"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        h = an.tf("V1", "out", keep=["R1", "C1", "R2", "C2"],
                  method="interp")
    tpl = template_form(h, budget=0.01)
    assert len(tpl.poles) == 2
    assert all(p.reliable and p.expr is not None for p in tpl.poles)
    assert all(p.displacement < 1e-9 for p in tpl.poles)


def test_template_rhp_real_root_gets_its_ratio():
    """The plain Miller RHP zero gm/Cc: the ratio n0/n1 is negative, so
    its magnitude is -n0/n1."""
    s = sp.Symbol("s")
    gm, Cc, R, CL = _pos("gm Cc R CL")
    t = template_form(_tf((gm - Cc * s) / (1 + R * CL * s),
                          gm=1e-3, Cc=1e-12, R=1e3, CL=1e-12))
    (z,) = t.zeros
    assert z.rhp and z.reliable
    assert sp.simplify(z.expr - gm / Cc) == 0
    assert "RHP ~ gm/Cc" in t.describe()


# ---------------------------------------------------- coupled roots: the LDO
LDO = FIX.parent / "ldo_ol.cin.json"   # Milliken 2007, Fig. 8 macromodel


@pytest.fixture(scope="module")
def ldo():
    """Milliken's loop, fully symbolic: the pair 1.7x below the next pole.
    The determinants without division, then each resistor back: `Analyzer.tf`
    gives the same polynomials after 3 minutes in the field of fractions."""
    from circuitinsight.engine.mna import TransferFunction

    an = Analyzer.from_cin(json.loads(LDO.read_text(encoding="utf-8")))
    sysm = an.system("V1")
    g = {sp.Symbol(n, positive=True): sysm.symbols[n]
         for n in sysm.reciprocal}
    A = sysm.A.xreplace({v: 1 / k for k, v in g.items()})
    Ak = A.copy()
    Ak[:, sysm.node_index["vfb"]] = sysm.z
    back = {k: 1 / v for k, v in g.items()}

    def det(M):
        return sp.fraction(sp.together(
            sp.expand(M.det(method="berkowitz")).xreplace(back)))

    (n, dn), (d, dd) = det(Ak), det(A)
    H = TransferFunction(expr=n * dd / (dn * d), values=dict(sysm.values),
                         symbols=dict(sysm.symbols))
    exact = sorted(an.tf("V1", "vfb", keep=[]).poles(), key=abs)
    return H, template_form(H), exact


def test_coupled_node_poles_are_milliken_10(ldo):
    """P4 and P6 of Milliken's (10), against the full model: each belongs
    to one capacitor's node, so each is 1/(R_node*C)."""
    H, tpl, exact = ldo
    v = H.symbols
    p4, p6 = tpl.poles[2], tpl.poles[3]
    assert p4.f_hz == pytest.approx(2.305e6, rel=1e-3)
    assert sp.simplify(p4.expr - (v["Rf1"] + v["Rf2"])
                       / (v["Cf2"] * v["Rf1"] * v["Rf2"])) == 0
    assert p4.displacement < 1e-3 and p4.reliable and p4.full_model
    assert p6.f_hz == pytest.approx(3.979e6, rel=1e-3)
    assert sp.simplify(p6.expr - 1 / (v["Ce"] * v["Re"])) == 0
    assert p6.displacement < 1e-9 and p6.reliable and p6.full_model
    text = tpl.describe()
    assert "R_node = Rf1*Rf2/(Rf1 + Rf2), node pole of Cf2" in text
    assert "1/(Ce*Re) rad/s" in text and "off the full model" in text
    # the statement is about the exact numbers, not the tool's own reading
    subs = H._subs_map()
    for p, f in ((p4, abs(exact[3])), (p6, abs(exact[4]))):
        w = abs(complex(sp.N(p.expr.xreplace(subs))))
        assert abs(w / (2 * np.pi) / f - 1) == pytest.approx(p.displacement,
                                                             abs=1e-6)


def test_coupled_pair_reads_on_a_reduced_model(ldo):
    """The pair has no quadratic in the full model. Without the parasitics
    it does, and the error stated is against the full model's pair; w0 has
    the shape of Milliken's (9), sqrt(wPD1*wP2)."""
    H, tpl, exact = ldo
    p = tpl.poles[1]
    assert p.expr is not None and p.reliable and p.full_model
    assert set(p.without) == {"C1", "C2", "Ce", "Cf2"}
    assert p.f_hz == pytest.approx(1.3698e6, rel=1e-3)
    assert p.damping == pytest.approx(0.289, rel=1e-2)
    subs = H._subs_map()
    w0 = abs(complex(sp.N(p.expr.xreplace(subs))))
    full = abs(exact[1]) * 2 * np.pi
    assert abs(w0 / full - 1) == pytest.approx(p.displacement, abs=1e-6)
    assert p.displacement < 0.05
    b = complex(sp.N(p.zeta_w0.xreplace(subs))).real
    zeta = -exact[1].real / abs(exact[1])
    assert abs(b / (2 * w0) / zeta - 1) == pytest.approx(p.zeta_displacement,
                                                         abs=1e-6)
    assert p.zeta_displacement < 0.15
    text = p.describe()
    assert "reduced model, without C1, C2, Ce, Cf2" in text
    assert "off the full model" in text
    # compact: the feedback divider and the gate's R1 have cancelled
    w0sq = sp.simplify(p.expr ** 2)
    assert not w0sq.has(H.symbols["Rf1"], H.symbols["Rf2"], H.symbols["R1"])
    assert w0sq.has(H.symbols["Gf1"], H.symbols["Gf2"], H.symbols["GP"])


def test_coupled_lines_leave_nothing_empty_or_unreadable(ldo):
    _, tpl, _ = ldo
    assert "no closed form" not in tpl.describe()
    assert all(len(g.describe()) <= 2 * template_mod._LONG_TEXT
               for g in tpl.poles + tpl.zeros)


def test_simplified_tf_states_the_full_models_roots(ldo):
    """A simplified TF moves the coupled roots (2.81 and 3.11 MHz here, for
    2.305 and 3.979): each line goes back to the root of the full model,
    while a line the coefficients could read is left as it was."""
    H, _, _ = ldo
    Hs = H.simplify(mag_tol_db=0.5, phase_tol_deg=3.0, fmin=1e2, fmax=1e7)
    tpl = template_form(Hs)
    plain = {g.index: g for g in tpl.poles if not g.full_model}
    assert 0 in plain and plain[0].expr is not None
    assert tpl.poles[2].f_hz == pytest.approx(2.305e6, rel=1e-3)
    assert tpl.poles[3].f_hz == pytest.approx(3.979e6, rel=1e-3)
    assert tpl.poles[1].f_hz == pytest.approx(1.3698e6, rel=1e-3)
    assert "no closed form" not in tpl.describe()


def _rc_ladder(n, *symbols):
    """n RC sections driving one another: real roots no ratio separates."""
    inst = [{"name": "V1", "device_type": "vsource",
             "terminals": {"p": "n0", "n": "0"}}]
    for i in range(1, n + 1):
        inst += [{"name": f"R{i}", "device_type": "resistor",
                  "terminals": {"p": f"n{i - 1}", "n": f"n{i}"},
                  "params": {"r": "1k"}},
                 {"name": f"C{i}", "device_type": "capacitor",
                  "terminals": {"p": f"n{i}", "n": "0"},
                  "params": {"c": "1u"}}]
    an = Analyzer.from_cin({"cin_version": "0.1", "top": "main",
                            "ground": ["0"], "definitions": {"main": {
                                "ports": [], "instances": inst}}})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return an.tf("V1", f"n{n}", keep=list(symbols), method="interp")


def test_coupled_cluster_prints_its_polynomial():
    """Identical sections stay coupled whatever is removed: what is left to
    say is the polynomial their roots come from. It is not a formula for one
    root, so expr stays None, and the numbers are still checked."""
    h = _rc_ladder(3, "R1", "C1", "R2", "C2", "R3", "C3")
    tpl = template_form(h, budget=0.01)
    clustered = [g for g in tpl.poles if g.cluster is not None]
    assert clustered
    text = tpl.describe()
    assert "*s^" in text and "no closed form of its own" in text
    for g in clustered:
        assert g.expr is None and not g.reliable and g.full_model
        assert g.displacement <= 0.01
    # a later root of the same cluster points at the first
    assert sum("root of (" in g.describe() for g in clustered) == 1


def test_coupled_needs_a_symbol_to_show():
    """A fully numeric TF has nothing symbolic to print: the line stays."""
    h = _rc_ladder(3)
    tpl = template_form(h, budget=0.01)
    assert all(g.cluster is None and not g.full_model for g in tpl.poles)


def test_coupled_hybrid_solve_reduces_only_what_it_kept():
    """Capacitors that are numbers cannot be removed from the TF: it reads
    or falls back with the symbols it has, and does not fail."""
    h = _rc_ladder(3, "R1", "R2", "R3")
    tpl = template_form(h, budget=0.01)
    assert len(tpl.poles) == 3
    assert not any(g.without for g in tpl.poles)
    assert tpl.describe()


def test_template_reactances_go_by_name_and_structure():
    """A symbol named like a capacitor that shows up at DC is not one, and
    a capacitor is multilinear."""
    s = sp.Symbol("s")
    Cx, Cy, Rz, Lm = _pos("Cx Cy Rz Lm")
    poly = sp.Poly(Cx + Rz + (Cy + Lm + Rz * Cy) * s + Cy * Lm * s ** 2, s)
    table = template_mod._TermTable(poly, {Cx: 1.0, Cy: 2.0, Rz: 3.0, Lm: 4.0})
    assert set(template_mod._reactances(table)) == {Cy, Lm}


def test_term_table_matches_the_exact_polynomial():
    """Roots, a removal and the sensitivities against known numbers."""
    s = sp.Symbol("s")
    C1, C2, R = _pos("C1 C2 R")
    subs = {C1: 1e-9, C2: 1e-12, R: 1e3}
    poly = sp.Poly(sp.expand((1 + R * C1 * s) * (1 + R * C2 * s)), s)
    t = template_mod._TermTable(poly, subs)
    assert np.sort(np.abs(t.roots(t.all()))) == pytest.approx([1e6, 1e9])
    assert np.abs(t.roots(t.all() & ~t.has(C2))) == pytest.approx([1e6])
    # r = -1/(R C1): d ln r / d ln C1 = -1, and C2 does not enter
    sens = t.sensitivities(complex(-1e6))
    assert sens[C1] == pytest.approx(-1, abs=2e-3)
    assert sens[C2] == pytest.approx(0, abs=2e-3)
    assert sens[R] == pytest.approx(-1, abs=2e-3)


def test_too_long_a_form_is_reported_by_size():
    """A form that no pruning shortens is not printed as if it were useful;
    the expression stays for whoever wants it."""
    s = sp.Symbol("s")
    a = _pos(" ".join(f"a{i}" for i in range(50)))
    b = _pos(" ".join(f"b{i}" for i in range(50)))
    vals = {str(x): 1.0 for x in (*a, *b)}
    t = template_form(_tf(1 / (sum(a) + sum(b) * s), **vals), budget=0.001)
    line = t.poles[0]
    assert line.expr is not None
    assert line.long_terms is not None and line.long_terms > 20
    assert "too long to print" in line.describe()
    assert len(line.describe()) < template_mod._LONG_TEXT
