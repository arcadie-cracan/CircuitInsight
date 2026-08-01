"""The folded-cascode OTA — the first large public fixture.

36 SKY130 instances (nfet/pfet_01v8_lvt), 413 primitives, a degree-17
transfer function: the circuit that shook out three real bugs in one day
(the huge-coefficient OverflowError, the unsorted template poles, and the
"single QQ[s] determinant is optimal" myth), and the reference workload
for the AC-ground -> deactivate -> lump reduction chain. Captured with
psfascii + preset=mx so the numbers match the interactive ADE session
that first exposed them.
"""
import warnings
from pathlib import Path

import numpy as np
import pytest

from circuitinsight.session import SessionController

FIX = Path(__file__).resolve().parent / "fixtures" / "spectre" / "fc"


@pytest.fixture(scope="module")
def fc():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SessionController.open(FIX / "tb_fc.cin.json", FIX / "psf",
                                      cap_model="matrix")


def test_reconstruction_shape(fc):
    an = fc._analyzer_ready()
    assert len({p.inst for p in an.primitives}) == 36
    assert len(an.primitives) == 413


def test_numeric_solve_and_the_overflow_regression(fc):
    """keep=[] on a 30-node circuit: exact integer coefficients with
    hundreds of digits. This is the circuit whose Simplify raised
    "int too large to convert to float"."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = fc.solve("VIND", "vout", keep=[])
    assert r.dc_gain_db == pytest.approx(55.854, abs=0.01)
    n, d = r.tf.num_den
    assert n.degree() == 17 and d.degree() == 17
    # simulator truth rides along, and the model tracks it
    assert r.h_ref is not None
    err = np.nanmax(np.abs(20 * np.log10(np.abs(r.h / r.h_ref))))
    assert err < 0.5

    s = r.tf.simplify(mag_tol_db=1.0)        # used to raise OverflowError
    assert s.achieved_mag_err_db <= 1.0


def test_template_reports_the_lowest_poles(fc):
    """The 17-pole circuit that exposed the unsorted-roots bug: the
    template once reported five poles starting at 328 MHz and a 204 GHz
    GBW for this amplifier. Dominant pole 20.5 kHz, GBW ~12.7 MHz."""
    tpl = fc.template_form("VIND", "vout", keep=[])
    assert tpl.poles[0].f_hz == pytest.approx(20460, rel=0.02)
    freqs = [p.f_hz for p in tpl.poles]
    assert freqs == sorted(freqs)
    assert tpl.gbw_hz == pytest.approx(1.27e7, rel=0.05)
    assert 20 * np.log10(abs(tpl.dc_gain_value)) == pytest.approx(55.85,
                                                                  abs=0.02)


def test_acground_scan_separates_bias_from_signal(fc):
    """Five PMOS mirror references are groundable inside 0.1 dB; the NMOS
    cascode gate (I0.net1) and the other signal-carrying bias nodes are
    refused. Only measurement separates these — every candidate is a
    diode-connected mirror gate structurally."""
    rep = fc.scan_ac_grounds("VIND", "vout", budget_db=0.1)
    assert set(rep.recommended) == {"I0.net19", "I0.net21", "I0.net22",
                                    "I0.net23", "I0.net2"}
    assert rep.joint_db == pytest.approx(0.0959, abs=0.005)
    refused = {c.node for c in rep.candidates if not c.within_budget}
    assert "I0.net1" in refused


def test_reduction_chain_pays_and_reverts(fc):
    """The flagship chain: ground the five mirrors, 84 controlled sources
    die (exact), passives lump (exact) — 413 -> 262 primitives at a
    MEASURED cost under 0.1 dB, and the reduced solve's DC gain moves by
    less than 0.001 dB. Revert restores everything."""
    rep = fc.scan_ac_grounds("VIND", "vout", budget_db=0.1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        summ = fc.apply_reduction(rep.recommended, inp="VIND", out="vout")
    try:
        assert summ["prims_before"] == 413
        assert summ["prims_after"] == 262
        assert len(summ["dead_sources"]) == 84
        assert summ["worst_db"] < 0.15
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = fc.solve("VIND", "vout", keep=[])
        assert r.circuit_state == "reduced"
        assert r.dc_gain_db == pytest.approx(55.854, abs=0.01)
    finally:
        fc.revert_reduction()
    assert len(fc._analyzer_ready().primitives) == 413


def test_suggester_does_not_fuse_the_cascode_strangers(fc):
    """The regression behind the 52.22 dB screenshot: structural identity
    plus gm proximity fused eight NMOS whose gds differs 3.2x and whose
    junction caps differ 49x, and matching shares VALUES. The suggester
    now checks every significant parameter, so no suggested group may
    carry a significant conflict beyond its own tolerances."""
    groups = fc.suggest_matches()
    assert groups                                # still suggests real pairs
    fc.set_matches(*groups)
    try:
        conf = fc.match_conflicts()
        worst = conf[0][3] if conf else 1.0
        assert worst < 1.5, f"suggested matches conflict {worst:.1f}x: {conf[:3]}"
        # and the model they produce stays within 2 dB of the honest one
        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter("ignore")
            r = fc.solve("VIND", "vout", keep=[])
        assert abs(r.dc_gain_db - 55.854) < 2.0
    finally:
        fc.set_matches()                         # leave the fixture clean


def test_match_conflicts_reports_significant_parameters_only(fc):
    """A deliberately bad match must be reported by its SIGNIFICANT
    conflicts — gds among them — while ratios where BOTH sides are
    negligible on their device stay out. A huge ratio with one side
    significant is a genuine conflict (a real capacitance overwritten by
    ~nothing) and must stay in."""
    from circuitinsight.session import _significance_floors

    fc.set_match_value_policy("representative")  # test the reporter's floor
    fc.set_matches(("I0.NM1", "I0.M19"))         # cascode vs mirror rail
    try:
        conf = fc.match_conflicts()
        assert conf, "a 3x gds overwrite must be reported"
        params = [c[0] for c in conf]
        assert any(p.startswith("g") for p in params)

        by = {}
        for p in fc._analyzer_ready().primitives:
            if p.param and p.value is not None:
                by.setdefault(p.inst, {})[p.param] = p.value
        sig = _significance_floors(by)
        for param, kept, other, ratio in conf:
            kind = "g" if param.startswith("g") else "c"
            floor = min(sig[kept][kind], sig[other][kind])
            assert (abs(by[kept][param]) >= floor
                    or abs(by[other][param]) >= floor), \
                f"{param} reported though negligible on both devices"
    finally:
        fc.set_matches()


def test_match_value_policies(fc):
    """Three ways to give a match group its shared values, measured here
    on the circuit that motivated them. Weighted — each member's values
    pulled by its band sensitivity — finds the load-bearing member and
    beats both alternatives on BOTH benches (fc 0.35 dB vs-sim against
    1.80 first-wins; ota5t 0.04 against 0.08), but representative stays
    the default because with no representative chosen it is bit-exactly
    the historic behavior."""
    groups = fc.suggest_matches()
    fc.set_matches(*groups)
    try:
        fc.set_match_value_policy("representative")   # historic baseline
        assert fc.match_value_policy == "representative"
        r_rep = fc.solve("VIND", "vout", keep=[])

        fc.set_match_value_policy("mean")
        r_mean = fc.solve("VIND", "vout", keep=[])
        fc.set_match_value_policy("weighted", inp="VIND", out="vout")
        r_w = fc.solve("VIND", "vout", keep=[])

        def fid(r):
            return float(np.nanmax(np.abs(
                20 * np.log10(np.abs(r.h / r.h_ref)))))

        assert fid(r_w) < fid(r_mean) < fid(r_rep)
        assert r_w.dc_gain_db == pytest.approx(55.85, abs=0.3)

        # an explicit representative redirects the whole group's values
        fc.set_match_value_policy("representative")
        dcs = set()
        for rep in groups[0]:
            fc.set_match_representative(rep)
            dcs.add(round(fc.solve("VIND", "vout", keep=[]).dc_gain_db, 3))
        assert len(dcs) > 1, "changing the representative must change values"

        with pytest.raises(ValueError):
            fc.set_match_representative("I0.NOSUCH")
        with pytest.raises(ValueError):
            fc.set_match_value_policy("median")
    finally:
        fc._match_reps.clear()
        fc.set_matches()


def test_story_keep_names_the_letters_of_the_lowest_order_form(fc):
    """Form-aware Suggest: a lowest-order model is A0 and a dominant pole,
    so the right keeps are the symbols that will appear in those two
    expressions — the pursuit's surviving reactances (kept = protected)
    plus the top band-ranked conductances — never parasitic reactances,
    which would force the order back up."""
    fc.set_matches(*fc.suggest_matches())
    try:
        keep = fc.suggest_story_keep("VIND", "vout", fmin=1.0, fmax=386e3,
                                     tol_db=10.0)
        assert "CL" in keep, "the pursuit's surviving reactance is kept"
        assert len(keep) <= 5
        assert all(not k.startswith(("c", "k")) or k == "CL" for k in keep), \
            "no parasitic reactances in the story"

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = fc.reduce_solve("VIND", "vout", keep=keep, tol_db=10.0,
                                mag_db=10.0, phase_deg=20.0,
                                fmin=1.0, fmax=386e3)
        assert len(r.poles_hz) == 1                  # still the whiteboard form
        letters = {str(x) for x in r.tf.expr.free_symbols} - {"s"}
        assert "CL" in letters and any(x.startswith("gm_") for x in letters)
        # and the display carries no unwieldy exact rationals
        import sympy as sp
        for x in sp.preorder_traversal(r.tf.expr):
            if x.is_Rational and not x.is_Integer:
                assert x.q <= 10 ** 12
    finally:
        fc.set_matches()
        fc.set_match_value_policy("weighted")
