"""Explain the numerals: shares are exact for linear stamps, by hand.

The common-source stage is small enough to check on paper: with
vout/vin = -gm/(gA + gB + s*CL) and gA = 3*gB, the denominator DC
coefficient is carried 75% by gA and 25% by gB, the s^1 coefficient
entirely by CL, and the numerator entirely by gm. A resistor stamps
1/R, so its share comes back sign-flipped with the same magnitude.
"""
from pathlib import Path

import pytest

from circuitinsight.analysis.explain import explain_coefficients
from circuitinsight.engine.mna import MnaError, build_mna
from circuitinsight.engine.primitives import Primitive

FIX = Path(__file__).resolve().parent / "fixtures" / "spectre"


def _cs_prims(load="g"):
    val = 3e-5 if load == "g" else 1 / 3e-5
    return [
        Primitive("V1", "", "vsrc", ("in", "0"), 1.0),
        Primitive("M1", "gm", "vccs", ("out", "0", "in", "0"), 2e-4),
        Primitive("RA", "", load, ("out", "0"), val),
        Primitive("RB", "", "g", ("out", "0"), 1e-5),
        Primitive("CL", "", "c", ("out", "0"), 1e-12),
    ]


def _by(stories, part, k):
    for st in stories:
        if st.part == part and st.k == k:
            return dict(st.contributors)
    raise AssertionError(f"no story for {part} s^{k}")


def test_shares_match_the_hand_calculation():
    system = build_mna(_cs_prims(), ("0",), "V1")
    stories = explain_coefficients(system, "out")

    den0 = _by(stories, "den", 0)
    assert den0["RA"] == pytest.approx(0.75, abs=1e-9)
    assert den0["RB"] == pytest.approx(0.25, abs=1e-9)
    assert "gm_M1" not in den0, "gm does not touch the denominator here"
    assert _by(stories, "den", 1)["CL"] == pytest.approx(1.0, abs=1e-9)
    assert _by(stories, "num", 0)["gm_M1"] == pytest.approx(1.0, abs=1e-9)


def test_a_resistor_share_flips_sign_only():
    """RA stamped as 1/R: the fraction of the coefficient flowing through
    it is unchanged, but the log-sensitivity to R is negative."""
    stories = explain_coefficients(build_mna(_cs_prims("r"), ("0",), "V1"),
                                   "out")
    den0 = _by(stories, "den", 0)
    assert den0["RA"] == pytest.approx(-0.75, abs=1e-9)
    assert den0["RB"] == pytest.approx(0.25, abs=1e-9)


def test_excluded_symbols_stay_out_and_describe_reads():
    system = build_mna(_cs_prims(), ("0",), "V1")
    stories = explain_coefficients(system, "out", exclude={"RA"})
    den0 = _by(stories, "den", 0)
    assert "RA" not in den0 and "RB" in den0
    line = [st for st in stories if st.part == "den" and st.k == 1][0]
    assert "CL +100%" in line.describe()

    with pytest.raises(MnaError):
        explain_coefficients(system, "nosuchnode")


def test_ratio_attribution_cancels_the_common_chain():
    """A displayed numeral is a RATIO of coefficients, and shares
    subtract: for the CS stage, pole scale 0 = den s^0/den s^1 is
    carried +75% by gA, +25% by gB and -100% by CL, while A0 =
    num s^0/den s^0 is +100% gm against the conductances."""
    from circuitinsight.analysis.explain import (ratio_contributors,
                                                 ratio_lines)

    system = build_mna(_cs_prims(), ("0",), "V1")
    stories = explain_coefficients(system, "out")

    pole = dict(ratio_contributors(stories, ("den", 0), ("den", 1)))
    assert pole["RA"] == pytest.approx(0.75, abs=1e-9)
    assert pole["RB"] == pytest.approx(0.25, abs=1e-9)
    assert pole["CL"] == pytest.approx(-1.0, abs=1e-9)

    a0 = dict(ratio_contributors(stories, ("num", 0), ("den", 0)))
    assert a0["gm_M1"] == pytest.approx(1.0, abs=1e-9)
    assert a0["RA"] == pytest.approx(-0.75, abs=1e-9)

    assert ratio_contributors(stories, ("den", 7), ("den", 8)) is None

    lines = ratio_lines(stories)
    assert any(ln.startswith("A0") for ln in lines)
    assert any("pole scale 0" in ln for ln in lines)

    from circuitinsight.gui.view import numeral_tips
    tips = numeral_tips(stories)
    assert "den:0" in tips and "A0" in tips and "p1" in tips
    assert "CL -100%" in tips["p1"]


def test_per_numeral_pass_separates_the_monomials():
    """What the OP-point sweep cannot do: with RB and CL kept, den s^0
    splits into its monomials — the '1' monomial is gA (carried 100% by
    RA), the RB monomial has numeral 1 with NO collapsed contributor —
    and the values sit in the exact 3e-5 : 1 ratio of the hand
    calculation. mono_key is the canonical bridge the GUI tags use."""
    from circuitinsight.analysis.explain import (explain_per_numeral,
                                                 mono_key)

    assert mono_key({"CL": 1, "gm_M1": 2}) == "CL·gm_M1^2"
    assert mono_key({}) == "1"

    system = build_mna(_cs_prims(), ("0",), "V1")
    stories = explain_per_numeral(system, "out", ["RB", "CL"])
    by = {(st.part, st.k, st.mono): st for st in stories}

    one = by[("den", 0, "1")]
    assert dict(one.contributors)["RA"] == pytest.approx(1.0, abs=1e-9)
    rb = by[("den", 0, "RB")]
    assert not rb.contributors
    assert one.value / rb.value == pytest.approx(3e-5, rel=1e-9)
    assert ("den", 1, "CL") in by
    gm = by[("num", 0, "1")]
    assert dict(gm.contributors)["gm_M1"] == pytest.approx(1.0, abs=1e-9)

    with pytest.raises(MnaError):
        explain_per_numeral(system, "out", [])      # needs kept letters


def test_session_surface_is_cached_and_excludes_the_keep():
    import warnings

    from circuitinsight.session import SessionController

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        c = SessionController.open(FIX / "ota5t" / "tb_ota5t.cin.json",
                                   FIX / "ota5t" / "psf")
    keep = ["gm_I0.MN1"]
    assert c.cached_numerals("VIND", "vout", keep=keep) is None
    stories = c.explain_numerals("VIND", "vout", keep=keep)
    assert stories, "an OTA has explainable coefficients"
    assert all("gm_I0_MN1" not in dict(st.contributors) for st in stories)
    assert c.explain_numerals("VIND", "vout", keep=keep) is stories
    # the Expression hover asks without computing
    assert c.cached_numerals("VIND", "vout", keep=keep) is stories

    # the deep pass on a real OTA: the kept letter appears as a
    # monomial, never as a contributor, and the cache answers
    assert c.cached_per_numeral("VIND", "vout", keep=keep) is None
    deep = c.explain_per_numeral("VIND", "vout", keep=keep)
    assert any("gm_I0_MN1" in st.mono for st in deep)
    assert all("gm_I0_MN1" not in dict(st.contributors) for st in deep)
    assert c.cached_per_numeral("VIND", "vout", keep=keep) is deep

    # a progress callback reports and does NOT bypass the cache: the
    # cached call answers instantly, a fresh call reports and stores
    assert c.explain_per_numeral("VIND", "vout", keep=keep,
                                 progress=lambda a, b: 1 / 0) is deep
    seen = []
    keep2 = ["gds_I0.MN1"]
    d2 = c.explain_per_numeral("VIND", "vout", keep=keep2,
                               progress=lambda a, b: seen.append((a, b)))
    assert seen and seen[-1][1] == seen[0][1]        # (done, total) shape
    assert c.cached_per_numeral("VIND", "vout", keep=keep2) is d2


def test_fast_pass_matches_exact_on_the_synthetic_circuit():
    """The float64 circle kernel against the exact mpfr sweep on the
    hand-calculable circuit: same slots (they come from the solved
    expression), same values, rankings and shares to well inside the
    ranking tolerance, nothing flagged approx."""
    from circuitinsight.analysis.explain import (explain_per_numeral,
                                                 explain_per_numeral_fast)
    from circuitinsight.engine.mna import solve_tf

    system = build_mna(_cs_prims(), ("0",), "V1")
    tf = solve_tf(system, "out", keep=["RB", "CL"])
    exact = explain_per_numeral(system, "out", ["RB", "CL"])
    fast = explain_per_numeral_fast(system, "out", ["RB", "CL"], tf.expr)

    ex = {(st.part, st.k, st.mono): st for st in exact}
    fa = {(st.part, st.k, st.mono): st for st in fast}
    assert set(ex) == set(fa)
    for key, st in fa.items():
        assert not st.approx, f"{key} unconfirmed on a trivial circuit"
        es, fs = dict(ex[key].contributors), dict(st.contributors)
        for p in set(es) | set(fs):
            assert fs.get(p, 0.0) == pytest.approx(
                es.get(p, 0.0), abs=5e-3), (key, p)


def test_fast_pass_matches_exact_on_the_ota():
    """On ota5t with two kept letters: every trusted story's top-3
    contributor ranking equals the exact pass's, and shares agree to
    the display tolerance. Trusted must be the common case, not the
    exception."""
    import warnings

    from circuitinsight.session import SessionController

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        c = SessionController.open(FIX / "ota5t" / "tb_ota5t.cin.json",
                                   FIX / "ota5t" / "psf")
    keep = ["gm_I0.MN1", "cdb_MN2"]
    exact = c.explain_per_numeral("VIND", "vout", keep=keep)
    fast = c.explain_per_numeral("VIND", "vout", keep=keep, fast=True)

    ex = {(st.part, st.k, st.mono): st for st in exact}
    fa = {(st.part, st.k, st.mono): st for st in fast}
    # the expression keeps every true coefficient; the exact pass floors
    # numerals below 1e-18 of scale — so fast covers exact, not equals
    assert set(ex) <= set(fa), sorted(set(ex) - set(fa))
    trusted = [k for k in ex if not fa[k].approx]
    assert len(trusted) >= len(ex) * 2 // 3, (
        f"only {len(trusted)}/{len(ex)} exact slots confirmed")
    for key in trusted:
        e3 = [p for p, _ in ex[key].contributors[:3]]
        f3 = [p for p, _ in fa[key].contributors[:3]]
        assert f3 == e3, (key, e3, f3)
        es, fs = dict(ex[key].contributors), dict(fa[key].contributors)
        # the value cross-check bounds C, not the derivative tensors:
        # shares on trusted slots are ranking-grade (a few percent),
        # not display-grade
        for p in set(es) & set(fs):
            assert fs[p] == pytest.approx(es[p], abs=1e-1), (key, p)

    # the session caches fast and exact separately; hovers prefer exact
    assert c.cached_per_numeral("VIND", "vout", keep=keep) is exact
