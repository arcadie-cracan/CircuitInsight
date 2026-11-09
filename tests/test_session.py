"""SessionController: the headless view-model both front ends drive.

Exercises the GUI's logic without a GUI — open a CIN+psf fixture, introspect,
solve, and check it reproduces the known circuit numbers. Also guards the
independence contract: the core/session must never import Qt or the Cadence
integration layer (docs/gui-virtuoso-integration-plan.md).
"""
import warnings
from pathlib import Path

import numpy as np
import pytest

from circuitinsight import SessionController

FIX = Path(__file__).resolve().parent / "fixtures" / "spectre"


def _open(name):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)   # moderate-inversion notes
        return SessionController.open(FIX / name / f"tb_{'ota5t' if name=='ota5t' else 'ota2s'}.cin.json",
                                      FIX / name / "psf")


@pytest.fixture(scope="module")
def ota5t():
    return _open("ota5t")


@pytest.fixture(scope="module")
def miller():
    return _open("miller")


# -------------------------------------------------------------- introspection
def test_introspection(ota5t):
    names = {d.name for d in ota5t.devices}
    assert {"I0.MN0", "I0.MN1", "I0.MP0", "I0.MP1", "I0.MN2"} <= names
    assert "VIND" in ota5t.sources()
    assert "VIND" in ota5t.input_ports()
    assert "vout" in ota5t.output_nets()
    for g in ota5t.ground:                              # ground excluded from nets
        assert g not in ota5t.nets


# ------------------------------------------------------------------ 5T solve
def test_solve_ota5t(ota5t):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        r = ota5t.solve("VIND", "vout", keep=[])
    assert r.dc_gain.real == pytest.approx(202.45, rel=1e-3)   # +46.1 dB
    assert r.dc_gain_db == pytest.approx(46.13, abs=0.1)
    assert r.poles_hz.size >= 1 and r.zeros_hz.size >= 0
    assert r.h.shape == r.freqs.shape
    assert r.h_ref is not None and r.h_ref.shape == r.freqs.shape   # AC overlay
    assert r.tf is not None


def test_cache_returns_same_object(ota5t):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        a = ota5t.solve("VIND", "vout", keep=[])
        b = ota5t.solve("VIND", "vout", keep=[])
    assert a is b


def test_solve_cache_keys_on_the_band(ota5t):
    """Regression: the solve cache key ignored fmin/fmax/points, so the
    certificate's wide sweep and the plot solve over the user's own band
    returned whichever Result happened to be computed first."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        r1 = ota5t.solve("VIND", "vout", [], fmin=1e3, fmax=1e9,
                         points=100)
        r2 = ota5t.solve("VIND", "vout", [], fmin=1e0, fmax=1e10,
                         points=200)
        r1b = ota5t.solve("VIND", "vout", [], fmin=1e3, fmax=1e9,
                          points=100)
    # distinct Result objects per band — before the fix the second call
    # returned the first cached object verbatim. The grid itself may
    # coincide (the AC reference overlay pins it to the simulator's own
    # sweep), so object identity is the contract, not grid shape.
    assert r1 is not r2
    assert r1b is r1                      # same band still hits the cache


def test_declaring_the_loop_reference_invalidates_the_cache(ota5t):
    """Regression: declare_ac_loop_gain changed what the loop family
    computes but left previously cached results in place."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        r1 = ota5t.solve("VIND", "vout", [], fmin=1e3, fmax=1e9,
                         points=100)
        ota5t.declare_ac_loop_gain("vout")
        try:
            r2 = ota5t.solve("VIND", "vout", [], fmin=1e3, fmax=1e9,
                             points=100)
            assert r2 is not r1
        finally:
            ota5t.declare_ac_loop_gain(None)


# --------------------------------------------------------- two-stage numbers
def test_solve_miller_dc_and_rhp_zero(miller):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        r = miller.solve("VIND", "vout", keep=[])
    assert r.dc_gain.real == pytest.approx(18786, rel=1e-3)     # 85.5 dB
    assert r.dc_gain_db == pytest.approx(85.48, abs=0.05)
    assert abs(r.poles_hz[0]) == pytest.approx(178.8, rel=0.03)     # dominant
    rhp = [z for z in r.zeros_hz if z.real > 0 and abs(z) < 1e9]
    assert len(rhp) == 1
    assert any("right-half-plane" in w for w in r.warnings)


def test_hybrid_keep_with_matches(miller):
    miller.set_matches(("I0.MN0", "I0.MN1"), ("I0.MP0", "I0.MP1"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        r = miller.solve("VIND", "vout", keep=["gm_I0_MP2", "I0.Cc", "CL"])
    assert r.n_terms > 3                       # genuinely symbolic, not collapsed
    assert r.dc_gain.real == pytest.approx(18786, rel=2e-2)
    assert "C" in r.tf_latex or "g" in r.tf_latex   # symbols survived to LaTeX
    miller.set_matches()                       # reset for other tests


# ------------------------------------------------------------------ planning
def test_estimate_and_suggest_keep(ota5t):
    est = ota5t.estimate("VIND", "vout", keep=[])
    assert est is not None
    plan = ota5t.suggest_keep("VIND", "vout", budget_s=5.0)
    assert hasattr(plan, "keep") and isinstance(list(plan.keep), list)


def test_suggest_matches(ota5t):
    groups = ota5t.suggest_matches()
    as_sets = {frozenset(g) for g in groups}
    assert frozenset({"I0.MN0", "I0.MN1"}) in as_sets   # pair (n, m=1)
    assert frozenset({"I0.MP0", "I0.MP1"}) in as_sets   # mirror (p, m=2)
    assert all("I0.MN2" not in g for g in groups)        # tail (n, m=2): alone


# ------------------------------------------------ keep-set ranking + simplify
def test_rank_symbols(miller):
    ranking = miller.rank_symbols("VIND", "vout")
    assert ranking and len(ranking[0]) == 3          # (name, score, peak_Hz)
    scores = [s for _, s, _ in ranking]
    assert scores == sorted(scores, reverse=True)     # descending by influence
    assert any("gm" in n for n, _, _ in ranking)


def test_simplify_stays_in_budget(miller):
    miller.set_matches(("I0.MN0", "I0.MN1"), ("I0.MP0", "I0.MP1"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        r = miller.simplify("VIND", "vout", keep=["gm_I0_MP2", "I0.Cc", "CL"],
                            mag_db=1.0, phase_deg=5.0)
    miller.set_matches()                              # reset shared fixture
    assert r.simplified
    assert r.n_terms_full is not None and r.n_terms <= r.n_terms_full
    assert r.mag_err_db is not None and r.mag_err_db <= 1.0 + 1e-6
    assert r.phase_err_deg <= 5.0 + 1e-6
    assert r.dc_gain.real == pytest.approx(18786, rel=5e-2)   # gain preserved


# independence contract: moved to test_import_guard.py (fixture-free, so it
# also runs in the public snapshot, which withholds the PDK-derived fixtures)


def test_solve_reports_progress():
    """A hybrid solve must report real progress: the grid size is known up front.

    Locks the whole chain (session -> analyzer -> mna -> interp). A progress
    callback that is accepted and never called is the natural failure here, and
    it would leave a GUI bar sitting at 0% through a 4-minute solve.
    """
    import warnings

    from circuitinsight import SessionController

    ota5t = FIX / "ota5t"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c = SessionController.open(str(ota5t / "tb_ota5t.cin.json"),
                                   str(ota5t / "psf"))
        c.set_matches(*c.suggest_matches())
        keep = [n for n, _, _ in c.rank_symbols("VIND", "vout")[:3]]

        ticks = []
        c.solve("VIND", "vout", keep, progress=lambda d, t: ticks.append((d, t)))

    assert ticks, "solver never reported progress"
    dones = [d for d, _ in ticks]
    assert dones == sorted(dones)                    # monotonic
    assert ticks[-1][0] == ticks[-1][1]              # reaches 100%
    assert all(t == ticks[0][1] for _, t in ticks)   # total is stable


def test_reduction_is_a_named_revertible_session_state():
    """gui-ux-plan.md U-C, session side: the working circuit has two named
    states — "as imported" and "reduced" — and the reduction (ground the
    ticked bias nodes, drop the sources that kills, lump) is applied
    per-session, measured at apply time, survives an analyzer rebuild, and
    reverts completely."""
    import warnings


    from circuitinsight import SessionController

    miller = FIX / "miller"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c = SessionController.open(str(miller / "tb_ota2s.cin.json"),
                                   str(miller / "psf"))
        rep = c.scan_ac_grounds("VIND", "vout", budget_db=0.2)
        assert rep.recommended, "vbn should be groundable on the two-stage"
        nodes = list(rep.recommended)

        # pricing an arbitrary ticked set matches the scan's joint pricing
        # (a dict now, so the GUI reads dB, degrees and the contract
        # score through one shape)
        jm = c.acground_joint("VIND", "vout", nodes)
        assert jm["worst_db"] == pytest.approx(rep.joint_db, rel=1e-6)
        assert jm["score"] is None            # no contract given here

        pv = c.preview_reduction(nodes)
        assert pv["prims_after"] < pv["prims_before"]
        assert pv["dead_sources"], "grounding vbn must kill bias sources"
        assert c.circuit_state == "as imported"      # preview changed nothing

        r_full = c.solve("VIND", "vout", keep=[])
        summ = c.apply_reduction(nodes, inp="VIND", out="vout")
        assert c.circuit_state == "reduced"
        assert summ["worst_db"] < 0.5                # measured, and small
        assert summ["prims_after"] == pv["prims_after"]

        r_red = c.solve("VIND", "vout", keep=[])
        assert r_red.circuit_state == "reduced"
        err = np.max(np.abs(
            20 * np.log10(np.abs(r_red.h / r_full.h))))
        assert err <= summ["worst_db"] + 0.05        # banner tells the truth

        # the reduction is SESSION state: a matches-triggered analyzer
        # rebuild must re-derive the same working circuit, not lose it
        c.set_matches(("I0.MN0", "I0.MN1"))
        assert c.circuit_state == "reduced"
        assert len(c._analyzer_ready().primitives) == summ["prims_after"]

        c.revert_reduction()
        assert c.circuit_state == "as imported"
        r_back = c.solve("VIND", "vout", keep=[])
        assert r_back.circuit_state == "as imported"
        assert len(c._analyzer_ready().primitives) == summ["prims_before"]


def test_reduce_enforcement_window_is_honest_and_adjustable(ota5t):
    """The budget-violation bug: the pursuit measured its error only
    where |H| stays within 60 dB of peak, then CLAIMED the user's full
    band -- a 1-pole model got certified at 0.6 dB over decades it
    visibly left by 25 dB. The window is now a parameter and the claim
    names the band actually enforced; a strict window (200 dB) honors
    the requested band and keeps more reactances for it."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        r60 = ota5t.reduce_solve("VIND", "vout", keep=[], tol_db=1.0,
                                 fmin=1.0, fmax=1e10)
        r200 = ota5t.reduce_solve("VIND", "vout", keep=[], tol_db=1.0,
                                  fmin=1.0, fmax=1e10, floor_db=200.0)
    assert "within 60 dB of peak" in r60.warnings[0]   # the relative form
    assert "lower the enforcement floor" in r60.warnings[0]
    assert "10 GHz" in r200.warnings[0]        # full band, no caveat
    assert "part of the band" not in r200.warnings[0]
    n60 = int(r60.warnings[0].split(" reactance")[0].split()[-1])
    n200 = int(r200.warnings[0].split(" reactance")[0].split()[-1])
    assert n200 >= n60                              # strictness costs order


def test_enforcement_floor_is_an_absolute_level(ota5t):
    """The window's reference was wrong for the question designers ask.
    "Within N dB of peak" hides the level that matters: with A0 = 56 dB
    a 60 dB window stops at -4 dB, barely reaching unity gain, so the
    crossover region a phase margin lives in went unchecked. The floor
    is an ABSOLUTE level -- 0 dB enforces down to unity."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        r0 = ota5t.reduce_solve("VIND", "vout", keep=[], tol_db=1.0,
                                fmin=1.0, fmax=1e10, floor_abs_db=0.0)
        rlo = ota5t.reduce_solve("VIND", "vout", keep=[], tol_db=1.0,
                                 fmin=1.0, fmax=1e10, floor_abs_db=-40.0)
    assert "widened by the" in r0.warnings[0]     # the tolerance is in it
    # the boundary IS the unity-gain region: |H| at the edge of the
    # enforced band sits at 0 dB, within the pursuit's sampling grid
    f = np.asarray(r0.freqs, dtype=float)
    mag_db = 20 * np.log10(np.abs(np.asarray(r0.h)))
    edge_db = float(np.interp(np.log10(r0.enforced_fmax),
                              np.log10(f), mag_db))
    assert abs(edge_db) < 8.0, edge_db
    # a lower floor certifies further into the rolloff
    assert rlo.enforced_fmax > r0.enforced_fmax


def test_enforcement_accounts_for_the_tolerances(ota5t):
    """A floor stated without the tolerance is not a guarantee: if the
    budget allows +/-10 dB, the true unity crossing can sit anywhere
    within 10 dB of the stated floor, in a region the pursuit never
    checked. And the gain-margin point lives at the +/-180 deg phase
    CROSSING, well below unity. Enforcement reaches below the floor by
    the magnitude budget and includes a bounded neighbourhood of the
    phase crossing."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        tight = ota5t.reduce_solve("VIND", "vout", keep=[], tol_db=1.0,
                                   phase_deg=20.0, fmin=1.0, fmax=1e10,
                                   floor_abs_db=0.0)
        loose = ota5t.reduce_solve("VIND", "vout", keep=[], tol_db=10.0,
                                   phase_deg=20.0, fmin=1.0, fmax=1e10,
                                   floor_abs_db=0.0)
    # a wider budget must enforce FURTHER, not less: the crossing it
    # could move to is further out
    assert loose.enforced_fmax > tight.enforced_fmax
    assert "widened by the 10 dB budget" in loose.warnings[0]
    assert "180" in loose.warnings[0]            # the phase crossing too


def test_reduce_solve_anchored_contract():
    """The one-knob path: eps sets the criterion, the band edges set the
    anchor, the note translates eps to dB/deg, and the certificate rides
    along with its order verdict."""
    import warnings

    from circuitinsight.session import SessionController

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        c = SessionController.open(FIX / "ota5t" / "tb_ota5t.cin.json",
                                   FIX / "ota5t" / "psf")
        r = c.reduce_solve("VIND", "vout", [], eps=0.10,
                           fmin=1e3, fmax=3e8)
    assert r.eps == 0.10 and r.anchor > 0
    # the strip line stays SHORT and actionable; the criterion, element
    # names, certificate verdict and doublet caveat live in r.details
    head = r.warnings[0]
    assert "reduced to" in head and len(head) < 160
    assert r.enforced_fmin == 1e3 and r.enforced_fmax == 3e8
    assert hasattr(r, "certificate")
    det = r.details
    assert any("criterion" in d and "anchor" in d for d in det)
    assert any("kept reactances" in d for d in det)
    assert any("order" in d for d in det)

    cert = c.order_certificate("VIND", "vout", 1e3, 3e8)
    assert cert.order_at(0.10) >= 1
    assert cert.shape in ("lowpass", "bandpass", "highpass", "flat")


def test_reduce_strategies_speak_the_designer_language():
    """fc, the user's own gesture (band a bit past crossover): stability
    yields ONE reactance with PM preserved within the budget; rejection
    tracks the dB curve; plain enforces both axes. Every strategy caps
    the order and never dumps an unreadable model."""
    import warnings

    from circuitinsight.session import SessionController

    fcdir = FIX / "fc"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        c = SessionController.open(fcdir / "tb_fc.cin.json",
                                   fcdir / "psf", cap_model="matrix")
        c.set_matches(*c.suggest_matches())

        r = c.reduce_solve("VIND", "vout", [], strategy="stability",
                           strategy_opts={"pm_deg": 5.0, "gm_db": 2.0},
                           fmin=151.0, fmax=76.8e6)
    assert r.strategy == "stability" and r.warnings[0].startswith("reduced")
    assert "PM" in r.warnings[0] and "Δ" in r.warnings[0]
    assert len(r.warnings[0]) < 220              # a strip line, not a dump
    mt = {d for d in r.details if "margins:" in d}
    assert mt, "the Summary carries the margin readouts"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        rj = c.reduce_solve("VIND", "vout", [], strategy="rejection",
                            strategy_opts={"rej_db": 3.0},
                            fmin=151.0, fmax=76.8e6)
        pl = c.reduce_solve("VIND", "vout", [], strategy="plain",
                            strategy_opts={"gain_db": 3.0,
                                           "phase_deg": 20.0},
                            fmin=151.0, fmax=76.8e6)
    assert "tracks within" in rj.warnings[0]
    assert "dB" in pl.warnings[0] and "°" in pl.warnings[0]
    # the readability cap: no strategy ever exceeds it
    for res in (r, rj, pl):
        n = int(res.warnings[0].split(" reactance")[0].split()[-1])
        assert n <= 6
    # units regression: band_score is the normalized budget fraction and
    # says so; mag_err_db is ALWAYS dB — it used to inherit whichever
    # raw unit the branch's error carried (fraction, dB, or budget
    # multiple), so downstream displays mixed units silently
    for res in (r, rj, pl):
        assert res.band_score is not None
        assert res.band_score_unit == "x budget"
    assert rj.mag_err_db == pytest.approx(rj.band_score * 3.0)
    assert pl.mag_err_db == pytest.approx(pl.band_score * 3.0)


def test_approximation_ledger_measures_not_sums(miller):
    """One contract prices every circuit-level approximation, and the
    total is MEASURED end to end -- never the sum of the entries."""

    miller.set_matches(*miller.suggest_matches())
    try:
        rep = miller.scan_ac_grounds(
            "VIND", "vout", strategy="plain",
            strategy_opts={"gain_db": 1.0, "phase_deg": 5.0},
            fmin=1e3, fmax=1e8)
        assert rep.criterion_label == "plain"
        assert all(c.score is not None for c in rep.candidates)
        assert rep.recommended and rep.joint_score <= 1.0
        miller.apply_reduction(list(rep.recommended), inp="VIND",
                               out="vout", strategy="plain",
                               strategy_opts={"gain_db": 1.0,
                                              "phase_deg": 5.0},
                               fmin=1e3, fmax=1e8)
        led = miller.approximation_report(
            "VIND", "vout", strategy="plain",
            strategy_opts={"gain_db": 1.0, "phase_deg": 5.0},
            fmin=1e3, fmax=1e8)
        steps = {e["step"].split(" ")[0] for e in led["entries"]}
        assert "matches" in steps and "reduction" in steps
        red = next(e["step"] for e in led["entries"]
                   if e["step"].startswith("reduction"))
        assert "grounded" in red
        assert all(e["score"] <= 1.0 for e in led["entries"]
                   if not e["exact"])
        # the total is a measurement of the composed circuit, within
        # the contract here, and NOT the arithmetic sum of the parts
        assert 0.0 < led["circuit_score"] <= 1.0
        s = sum(e["score"] for e in led["entries"] if not e["exact"])
        assert led["circuit_score"] != s
    finally:
        miller.revert_reduction()
        miller.set_matches()                      # reset shared fixture


def test_equivalent_elements_are_spelled_out():
    """A reduction's lumped Geq/Ceq symbol is a definition: the session
    lists its members (as H(s) join keys) and value; the formatter
    renders it G_{eq,node}, never the bare node name."""
    import warnings

    from circuitinsight.gui.view.format import symbol_tex
    from circuitinsight.session import SessionController

    fc = FIX / "fc"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c = SessionController.open(fc / "tb_fc.cin.json", fc / "psf")
        assert c.equivalent_elements() == []
        rep = c.scan_ac_grounds("VIND", "vout")
        nodes = [x.node for x in rep.candidates if x.within_budget][:2]
        c.apply_reduction(nodes, inp="VIND", out="vout")
    eqs = c.equivalent_elements()
    assert eqs, "grounding mirror nodes lumps parallel elements"
    g = next(e for e in eqs if e["name"].startswith("Geq_"))
    assert all("." not in m and m.startswith("g") for m in g["members"])
    assert g["value"] > 0
    assert symbol_tex(g["name"]) == (
        r"G_{eq,\mathrm{" + g["node"].split(".")[-1] + "}}")
    assert "net" not in symbol_tex("Geq_I0_net8").replace("net8", "")
    c.revert_reduction()
    assert c.equivalent_elements() == []


def test_matched_symbols_are_named_after_the_member_on_the_signal_path():
    """A match group's shared symbols carry ONE member's name. It used
    to be whichever came first in the group tuple, so the cascode
    NM5's transconductance showed as g_m,NM3 -- its mirror twin, off
    the signal path. The name bearer is now the member whose
    parameters shape the in->out response most (band sensitivity)."""
    import warnings

    from circuitinsight.session import SessionController

    fc = FIX / "fc"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c = SessionController.open(fc / "tb_fc.cin.json", fc / "psf")
        groups = [tuple(g) for g in c.suggest_matches()]
        c.set_matches(*groups)
        g = next(g for g in groups if "I0.NM5" in g)
        assert "I0.NM3" in g and g[0] != "I0.NM5"     # tuple order alone
        assert c.name_bearer(g) == "I0.NM5"
        r = c.solve("VIND", "vout", ["gm_I0_NM5"])
    names = {str(s) for s in r.tf.expr.free_symbols}
    assert "gm_I0_NM5" in names and "gm_I0_NM3" not in names
    # a user-chosen representative overrides the heuristic
    c.set_match_representative("I0.NM3")
    assert c.name_bearer(g) == "I0.NM3"


def test_snap_unity_divides_out_a_covered_leading_numeral():
    """The DC-gain coefficient snap, tested at the unit: a lowest-order
    form with a clean leading numeral (a collapsed ratio of unkept
    parameters) is divided down to 1 when the contract covers the
    shift, left alone when it does not, and left alone when the
    numeral sits inside a sum (no single coefficient to divide out).
    This used to be an fc end-to-end test; the common-node and
    running-mean match commits changed fc's suggested groups, and the
    surviving lowest-order numerator is now gm*(0.999*gm + 6e-5) --
    a sum, nothing to snap -- so the fixture no longer exercises the
    feature."""
    import sympy as sp

    from circuitinsight.analysis.criteria import PlainCriterion
    from circuitinsight.analysis.simplify import (SimplifiedTF,
                                                  leading_coefficient,
                                                  snap_unity)

    a, c, s = sp.symbols("a c s", positive=True)
    freqs = np.logspace(1, 6, 40)

    def tf_with(k):
        return SimplifiedTF(expr=sp.Float(k, 17) * a / (1 + s * c / a),
                            values={"a": 1e-3, "c": 1e-9},
                            symbols={"a": a, "c": c}, s=s)

    # 0.98 costs 0.18 dB: a 1 dB contract covers it, the form snaps
    tf = tf_with(0.98)
    h_full = tf.numeric(freqs)
    snapped, k, err = snap_unity(tf, PlainCriterion(gain_db=1.0),
                                 freqs, h_full)
    assert abs(k - 0.98) < 1e-12
    assert abs(leading_coefficient(snapped) - 1.0) < 1e-9
    # the recorded shift is measured: 20 log10(0.98)
    assert abs(err - abs(20 * np.log10(0.98))) < 1e-9

    # 0.5 costs 6 dB: the same contract does not cover it, no snap
    tf = tf_with(0.5)
    kept, k, err = snap_unity(tf, PlainCriterion(gain_db=1.0),
                              freqs, tf.numeric(freqs))
    assert k == 1.0 and err is None
    assert abs(leading_coefficient(kept) - 0.5) < 1e-12

    # a numeral inside a sum is not a leading coefficient
    b = sp.Symbol("b", positive=True)
    tf = SimplifiedTF(expr=(sp.Float(0.98) * a * b + sp.Float(6e-5) * b)
                      / (1 + s * c / a),
                      values={"a": 1e-3, "b": 1e-3, "c": 1e-9},
                      symbols={"a": a, "b": b, "c": c}, s=s)
    kept, k, err = snap_unity(tf, PlainCriterion(gain_db=10.0),
                              freqs, tf.numeric(freqs))
    assert k == 1.0 and err is None


def test_suggest_matches_requires_a_common_node(tmp_path):
    """One more structural criterion: a suggested set shares at least
    one NET (gate, source, drain -- a mirror's common gate, a pair's
    common source). Ground and global supplies do not count (every
    bulk ties to gnd!). A device identical in type and parameters but
    living on unrelated nets is not suggested."""
    import json
    import warnings

    fx = FIX / "ota5t"
    raw = json.loads((fx / "tb_ota5t.cin.json").read_text(encoding="utf-8"))
    ota = raw["definitions"]["ota5t"]
    mn0 = next(i for i in ota["instances"] if i["name"] == "MN0")
    clone = dict(mn0)
    clone["name"] = "MN9"
    clone["sim_name"] = mn0.get("sim_name", "MN0")   # same OP record
    clone["terminals"] = {"d": "zz1", "g": "zz2", "s": "zz3", "b": "vss"}
    ota["instances"] = list(ota["instances"]) + [clone]
    work = tmp_path / "cin.json"
    work.write_text(json.dumps(raw), encoding="utf-8")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from circuitinsight.session import SessionController
        c = SessionController.open(work, fx / "psf")
        groups = [tuple(g) for g in c.suggest_matches()]
    members = {m for g in groups for m in g}
    assert "I0.MN9" not in members, groups
    # the true pair is still found
    assert any({"I0.MN0", "I0.MN1"} <= set(g) for g in groups)


def test_suggest_refines_against_the_running_mean():
    """Membership must not depend on netlist order: fc's cascode NM4
    differed 2.8% in gm from the fixed head NM3 (out, at a 2%
    tolerance) but 0.9% from NM5. Against the subgroup's running mean
    all four cascodes group; NM2 (gm 4x off) stays out."""
    import warnings

    from circuitinsight.session import SessionController

    fc = FIX / "fc"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c = SessionController.open(fc / "tb_fc.cin.json", fc / "psf")
        groups = [set(g) for g in c.suggest_matches()]
    assert {"I0.NM1", "I0.NM3", "I0.NM4", "I0.NM5"} in groups
    assert not any("I0.NM2" in g for g in groups)


def test_mirror_gm_budget_admits_the_third_leg():
    """A 1:1:1 mirror (gates tied, V_GS and V_TH identical, currents
    equal to within the budget) earns a gm tolerance of 2% plus its
    own measured lambda term (dVds*gds/Id, capped at 10%): fc's PM3
    sat 2.1% from the (M24, PM4) mean with a 2.4% predicted Vds
    effect, so all three legs group. The strict 2% stays for
    everything else: no other bench's suggestions moved, and the
    cascodes (common gate but 20 mV of V_GS spread, gds apart by 2x)
    never see the budget."""
    import warnings

    from circuitinsight.session import SessionController

    fc = FIX / "fc"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c = SessionController.open(fc / "tb_fc.cin.json", fc / "psf")
        groups = [set(g) for g in c.suggest_matches()]
    assert {"I0.M24", "I0.PM3", "I0.PM4"} in groups
    # the cascodes (gate-tied but sources apart: V_GS differs by mV)
    # still needed the running mean, not the mirror budget -- and NM2
    # (4x off) stays out of everything
    assert {"I0.NM1", "I0.NM3", "I0.NM4", "I0.NM5"} in groups
    assert not any("I0.NM2" in g for g in groups)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c5 = SessionController.open(FIX / "ota5t" / "tb_ota5t.cin.json",
                                    FIX / "ota5t" / "psf")
        g5 = [set(g) for g in c5.suggest_matches()]
    assert g5 == [{"I0.MN0", "I0.MN1"}, {"I0.MP0", "I0.MP1"}]


def test_apply_reduction_deletes_ticked_removals():
    """Option 1 of the removal story: the scan's price becomes an
    action. apply_reduction takes `removals` -- instance names from the
    removal scan -- deletes them from the working circuit, records what
    was actually removed, and the banner figure is measured on the
    rewritten circuit (it agrees with the scan's rank-one price). The
    rewrite replays deterministically on an analyzer rebuild, and a
    name absent from the circuit deletes nothing."""
    import warnings

    from circuitinsight.session import SessionController

    d = FIX / "miller"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c = SessionController.open(d / "tb_ota2s.cin.json", d / "psf")
        rep = c.scan_removals("VIND", "vout")
        assert [x.inst for x in rep.candidates]
        summ = c.apply_reduction([], inp="VIND", out="vout",
                                 removals=["CL"])
    assert summ["removed"] == ["CL"]
    assert summ["prims_after"] == summ["prims_before"] - 1
    cl = next(x for x in rep.candidates if x.inst == "CL")
    assert abs(summ["worst_db"] - cl.worst_db) < 0.1
    # deterministic replay through a rebuilt analyzer
    c._analyzer = None
    assert all(p.inst != "CL" for p in c._analyzer_ready().primitives)
    # the ledger names the deletion
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        led = c.approximation_report("VIND", "vout")
    assert any("removed CL" in e["step"] for e in led["entries"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c.revert_reduction()
        s2 = c.apply_reduction([], inp="VIND", out="vout",
                               removals=["nope"])
    assert s2["removed"] == [] and \
        s2["prims_after"] == s2["prims_before"]


def test_removals_compose_with_the_grounding_chain():
    """The GUI flow on fc: ground the recommended bias nets, scan the
    REDUCED circuit (the candidates now include the lumped Geq/Ceq
    equivalents), then apply grounds and deletions together -- the
    replay chains ground -> lump -> delete, and the joint price is
    measured on exactly that circuit."""
    import warnings

    from circuitinsight.session import SessionController

    fc = FIX / "fc"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c = SessionController.open(fc / "tb_fc.cin.json", fc / "psf")
        c.set_matches(*[tuple(g) for g in c.suggest_matches()])
        rep = c.scan_ac_grounds("VIND", "vout")
        nodes = [x.node for x in rep.candidates if x.within_budget]
        assert nodes
        c.apply_reduction(nodes, inp="VIND", out="vout")
        rep2 = c.scan_removals("VIND", "vout", budget_db=10.0)
        rms = [x.inst for x in rep2.candidates if x.within_budget][:3]
        assert rms
        assert any(x.startswith(("Geq_", "Ceq_")) for x in rms), \
            "the reduced circuit's equivalents are candidates"
        summ = c.apply_reduction(nodes, inp="VIND", out="vout",
                                 removals=rms)
    assert set(summ["removed"]) == set(rms)
    an = c._analyzer_ready()
    assert not ({p.inst for p in an.primitives} & set(rms))
    jm = c.reduction_joint("VIND", "vout", nodes, rms)
    assert jm["worst_db"] >= 0 and jm["worst_deg"] >= 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = c.solve("VIND", "vout", [])
    assert r.h is not None


def test_impedance_specs_nets_and_seen_by():
    """impedance_result takes more than a marker: {"node"} is Z to
    ground, {"node","ref"} between two nets, {"seen_by"} what a
    two-terminal element sees (element removed, sources zeroed) --
    at high f, CL removed reads a different impedance than Z(vout)
    with CL in circuit. A net the applied reduction AC-grounds is
    refused with the reason, not 'node not found'."""
    import warnings

    import pytest

    from circuitinsight.engine.mna import MnaError
    from circuitinsight.session import SessionController

    d = FIX / "miller"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c = SessionController.open(d / "tb_ota2s.cin.json", d / "psf")
        r_node = c.impedance_result({"node": "vout"})
        r_seen = c.impedance_result({"seen_by": "CL"})
        r_pair = c.impedance_result({"node": "vout", "ref": "vin_p"})
        r_mark = c.impedance_result("VIND")          # legacy marker
    assert r_node.out == "Z(vout)"
    assert r_seen.out == "Z seen by CL"
    assert r_pair.out == "Z(vout, vin_p)"
    assert r_mark.out == "Z(VIND)"
    # Thevenin looking outward: with CL removed the high-f impedance
    # is the amplifier's own, several times the CL-shunted value
    assert abs(r_seen.h[-1]) > 3 * abs(r_node.h[-1])
    # and at low f, where CL is invisible, the two agree
    assert abs(abs(r_seen.h[0]) / abs(r_node.h[0]) - 1) < 0.01
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c.apply_reduction(["vin_p"], inp="VIND", out="vout")
    with pytest.raises(MnaError, match="AC-grounded by the applied"):
        c.impedance_result({"node": "vin_p"})
    with pytest.raises(MnaError, match="AC-grounded by the applied"):
        c.impedance_result({"node": "vout", "ref": "vin_p"})


def test_impedance_forms_have_transfer_parity():
    """The Impedance tool climbs the same contract ladder as Transfer:
    exact, simplified (full order, term pruning within a budget), and
    lowest order (a unit probe isource makes Z an ordinary transfer
    function; the reactance pursuit, collapse, snap and certificate
    run unchanged). Miller's Z(vout) reduces to the textbook Cc + CL
    second order; a between-two-nets spec refuses the lowest form with
    the reason; a series marker that cannot meet the contract at the
    order cap says NOT MET instead of pretending."""
    import warnings

    import pytest
    import sympy as sp

    from circuitinsight.engine.mna import MnaError
    from circuitinsight.session import SessionController

    d = FIX / "miller"
    plain = {"gain_db": 1.0, "phase_deg": 5.0}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c = SessionController.open(d / "tb_ota2s.cin.json", d / "psf")
        ex = c.impedance_result({"node": "vout"})
        fo = c.impedance_result({"node": "vout"}, form="simplified")
        lo = c.impedance_result({"node": "vout"}, form="lowest",
                                strategy="plain", strategy_opts=plain,
                                fmin=1e3, fmax=1e7)
    assert not getattr(ex, "simplified", False)
    assert fo.simplified and not getattr(fo, "reduced_order", False)
    assert abs(abs(fo.h[0]) / abs(ex.h[0]) - 1) < 0.05
    assert lo.simplified and lo.reduced_order
    # any simplified Z carries the exact solution as its reference
    for r in (fo, lo):
        assert r.ref_label == "exact Z" and r.h_ref is not None
        dev = np.abs(np.asarray(r.h)) / np.abs(np.asarray(r.h_ref))
        assert np.nanmax(np.abs(20 * np.log10(dev))) < 1.5
    s = sp.Symbol("s")
    assert sp.degree(sp.fraction(sp.together(lo.tf.expr))[1], s) == 2
    assert lo.warnings and "reduced to 2 reactance" in lo.warnings[0]
    assert getattr(lo, "band_score", None) is not None
    assert lo.band_score <= 1.0
    assert abs(abs(lo.h[0]) / abs(ex.h[0]) - 1) < 0.15
    with pytest.raises(MnaError, match="ground-referenced"):
        c.impedance_result({"node": "vout", "ref": "vin_p"},
                           form="lowest")
    # honesty at the cap: the series input port's Z needs more order
    # than the contract allows, and the headline says so
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lp = c.impedance_result("VIND", form="lowest",
                                strategy="plain", strategy_opts=plain,
                                fmin=1e3, fmax=1e7)
    assert "NOT MET" in lp.warnings[0]


def test_loop_gain_forms_have_transfer_parity():
    """The Loop-gain tool climbs the same contract ladder: exact,
    simplified (term pruning on T), lowest order. The lowest form uses
    BACKWARD elimination -- an AC-coupled loop (nmc3's stb rig) is
    identically zero with every reactance removed, so the transfer
    tool's from-zero add-back can never climb; starting full, each
    removal is priced against the margins contract. Under a 15-degree
    PM budget the loop keeps exactly the textbook set: CL and the two
    Miller caps."""
    import glob
    import warnings

    from circuitinsight.session import SessionController

    d = FIX / "nmc3"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c = SessionController.open(glob.glob(str(d / "*.cin.json"))[0],
                                   d / "psf")
        probe = c.probes[0]
        ex = c.loop_gain(probe)
        fo = c.loop_gain(probe, form="simplified", band_fmin=1e3,
                         band_fmax=1e8)
        lo = c.loop_gain(probe, form="lowest", strategy="stability",
                         strategy_opts={"pm_deg": 15.0, "gm_db": 6.0,
                                        "fc_rel": 0.3},
                         band_fmin=1e4, band_fmax=1e8)
    assert not getattr(ex, "simplified", False)
    assert fo.simplified and not getattr(fo, "reduced_order", False)
    assert fo.n_terms < ex.n_terms
    assert abs(fo.pm_deg - ex.pm_deg) < 0.5
    assert lo.simplified and lo.reduced_order
    assert "reduced to 3 reactance(s)" in lo.warnings[0]
    assert "[CL, CM1, CM2]" in lo.warnings[0]
    assert "PM" in lo.warnings[0]           # the stability grammar
    # the margins on screen are the simplified T's own, and the cached
    # exact is untouched
    assert c.loop_gain(probe).n_terms == ex.n_terms


def test_impedance_ranking_reflects_z_contributors():
    """The keep table's contributors must belong to the response on
    the bench: Z at fc's bias node I0.net3 is built on NM9's gm (the
    diode branch), while the transfer's ranking leads with the signal
    path's gm_PM13 -- the field report showed the latter offered for
    the former. rank_symbols_z ranks on the probe-injected system."""
    import warnings

    from circuitinsight.session import SessionController

    fc = FIX / "fc"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c = SessionController.open(fc / "tb_fc.cin.json", fc / "psf")
        c.set_matches(*[tuple(g) for g in c.suggest_matches()])
        tr = [n for n, _, _ in c.rank_symbols("VIND", "vout")]
        zr = [n for n, _, _ in c.rank_symbols_z({"node": "I0.net3"})]
    assert tr[0] == "gm_I0_PM13"          # the transfer's lead
    assert zr[0] == "gm_I0_NM9"           # Z's own lead
    assert set(zr[:5]) != set(tr[:5])


def test_rail_hung_port_and_seen_by_rank_as_z():
    """The diode_bias field case: an isource from the vdd! rail into
    a diode-connected bias node is a ground-referenced port in AC
    (the rail is held by the ideal supply), so BOTH the port form and
    'Z seen by' it rank Z at the bias node -- before, they raised and
    the keep table silently fell back to a transfer ranking."""
    import warnings

    from circuitinsight.session import SessionController

    d = FIX / "miller"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c = SessionController.open(d / "tb_ota2s.cin.json", d / "psf")
        zn = [n for n, _, _ in c.rank_symbols_z({"node": "vbn"})]
        zp = [n for n, _, _ in c.rank_symbols_z("IB")]
        zs = [n for n, _, _ in c.rank_symbols_z({"seen_by": "IB"})]
    # IB is an ideal isource (open in AC): removing it for seen_by
    # changes nothing, so all three are THE SAME ranking
    assert zp == zn and zs == zn
    assert zn[0] == "gm_MN2"              # the diode device leads


def test_loop_gain_cost_gate():
    """The keep plan budgets the TRANSFER solve; the Tian hybrid costs
    ~4x with the same ticks and nothing gated it -- on a fast machine
    the auto plan's set froze the bench for minutes (field report,
    2026-08-23). loop_gain now takes max_seconds and refuses over
    budget with the estimate and the way out, instantly; numeric T is
    the fast path and is never gated."""
    import time
    import warnings

    import pytest

    from circuitinsight.session import SessionController, SolveTooLarge

    d = FIX / "miller"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c = SessionController.open(d / "tb_ota2s_stb.cin.json",
                                   d / "psf_stb")
        big = [n for n, _, _ in c.rank_symbols("VIND", "vout")][:11]
        est = c.estimate_loop("IPRB0", big)
    assert est.seconds is not None
    assert est.n_dets % 4 == 0            # four rational solves
    # the budget is derived from the estimate itself: absolute
    # thresholds broke when an earlier test left a different machine
    # calibration in the process (estimates are calibration-scaled)
    budget = est.seconds / 2.0
    t0 = time.time()
    with pytest.raises(SolveTooLarge, match="Tian hybrid"):
        c.loop_gain("IPRB0", keep=big, max_seconds=budget)
    assert time.time() - t0 < 10.0        # a gate, not a hang
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = c.loop_gain("IPRB0", max_seconds=budget)
    assert r.pm_deg is not None           # numeric: never gated


def test_impedance_forwards_progress_to_the_solver(monkeypatch):
    """The 91.6-vs-11 s field report ended with 'phases: preparing
    (100%)': impedance_result never forwarded the progress callback,
    so the solver's evaluation units could not drive the phase ledger
    or refine the live estimate -- the whole solve filed under
    'preparing' and the promise had nothing to correct it. The
    callback now reaches solve_tf."""
    import warnings

    import circuitinsight.analyzer as anmod
    from circuitinsight.session import SessionController

    d = FIX / "miller"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c = SessionController.open(d / "tb_ota2s.cin.json", d / "psf")
    got = {}
    real = anmod.solve_tf

    def spy(system, output, keep=None, method="auto", progress=None):
        got["progress"] = progress
        return real(system, output, [], method=method)

    monkeypatch.setattr(anmod, "solve_tf", spy)

    def cb(done, total):
        pass

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c.impedance_result({"node": "vout"}, keep=[], progress=cb)
    assert got["progress"] is cb


def test_snap_unity_is_judged_against_the_achieved_score():
    """Under an ORDER CAP the budget can be missed before the snap; the
    snap is then judged against the score the form already carries, not
    against 1, so a leading numeral still divides out when the snap does
    not worsen the miss. A snap that would worsen it is still refused."""
    import sympy as sp

    from circuitinsight.analysis.criteria import PlainCriterion
    from circuitinsight.analysis.simplify import (SimplifiedTF,
                                                  leading_coefficient,
                                                  snap_unity)

    a, c, s = sp.symbols("a c s", positive=True)
    freqs = np.logspace(1, 6, 40)
    exact = SimplifiedTF(expr=a / (1 + s * c / a),
                         values={"a": 1e-3, "c": 1e-9},
                         symbols={"a": a, "c": c}, s=s)
    # the "full" response is 3 dB above the form everywhere: the form is
    # already outside a 1 dB contract whatever its leading numeral
    h_full = exact.numeric(freqs) * 10 ** (3 / 20)
    form = SimplifiedTF(expr=sp.Float(0.98, 17) * a / (1 + s * c / a),
                        values={"a": 1e-3, "c": 1e-9},
                        symbols={"a": a, "c": c}, s=s)
    crit = PlainCriterion(gain_db=1.0)
    assert crit.score(freqs, h_full, form.numeric(freqs)) > 1.0
    # 0.98 -> 1 moves the form TOWARD the full response: accepted
    snapped, k, err = snap_unity(form, crit, freqs, h_full)
    assert abs(k - 0.98) < 1e-12
    assert abs(leading_coefficient(snapped) - 1.0) < 1e-9
    # a numeral whose removal worsens the miss by more than a few percent
    # of itself stays: 1.6 -> 1 moves the form 4 dB AWAY from the full
    far = SimplifiedTF(expr=sp.Float(1.6, 17) * a / (1 + s * c / a),
                       values={"a": 1e-3, "c": 1e-9},
                       symbols={"a": a, "c": c}, s=s)
    kept, k, err = snap_unity(far, crit, freqs, h_full)
    assert k == 1.0 and err is None


def test_certificate_details_name_a_cap_below_the_bands_demand():
    """The Summary's certificate line distinguishes a cap the band fits
    under (order-optimal, or the named-components cost) from a cap BELOW
    the order the band demands: the capped model is the designer's
    approximation, and the line says so instead of calling it optimal.
    Fixture-free: a synthetic certificate and a stand-in result."""
    from types import SimpleNamespace

    from circuitinsight.analysis.certificate import order_certificate
    from circuitinsight.session import SessionController

    f = np.logspace(1, 7, 400)
    s = 2j * np.pi * f
    H = 1e3 / ((1 + s / (2 * np.pi * 1e3)) * (1 + s / (2 * np.pi * 1e5)))
    cert = order_certificate(f, H, 1e1, 1e7)
    k = cert.order_at(0.01)
    assert k >= 2
    det = SessionController._certificate_details(
        None, SimpleNamespace(), None, None, 1e1, 1e7, 0.01,
        n_sel=1, cert=cert, cap=1)
    assert det and det[0].startswith("capped at 1 below the order")
    assert "designer's approximation" in det[0]
    # the same selection without a cap reads as the named-components cost
    det = SessionController._certificate_details(
        None, SimpleNamespace(), None, None, 1e1, 1e7, 0.01,
        n_sel=k, cert=cert, cap=None)
    assert det and "order-optimal" in det[0]


def test_story_keep_follows_the_capped_pursuit():
    """The letters suggested for a lowest-order solve are the letters
    THAT solve keeps: under the plain contract with the cap at 2 the
    Miller story keep names Cc and CL and the three conductances of the
    textbook form, where the uncapped pursuit at the same budget also
    admits parasitic transcapacitances."""
    import warnings

    from circuitinsight.session import SessionController

    d = FIX / "miller"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c = SessionController.open(d / "tb_ota2s.cin.json", d / "psf")
        c.set_matches(*c.suggest_matches())
        capped = c.suggest_story_keep(
            "VIND", "vout", fmin=1e3, fmax=1e8, tol_db=3.0,
            strategy="plain",
            strategy_opts={"gain_db": 3.0, "phase_deg": 15.0, "cap": 2})
        wide = c.suggest_story_keep(
            "VIND", "vout", fmin=1e3, fmax=1e8, tol_db=3.0,
            strategy="plain",
            strategy_opts={"gain_db": 3.0, "phase_deg": 15.0, "cap": 6})
    def parasitic(n):
        return n not in ("I0_Cc", "CL") and n[0] in "ck"

    assert capped[:2] == ["I0_Cc", "CL"]
    assert not any(parasitic(n) for n in capped), capped
    assert any(parasitic(n) for n in wide), wide
    # the keep is the one the capped solve then holds: the two-pole form
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = c.reduce_solve("VIND", "vout", capped, strategy="plain",
                           strategy_opts={"gain_db": 3.0, "phase_deg": 15.0,
                                          "cap": 2}, fmin=1e3, fmax=1e8)
    assert r.reduced_order
    assert "reduced to 2 reactance(s)" in r.warnings[0]
    assert "raise the cap" in r.warnings[0]
