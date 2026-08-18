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

    import numpy as np

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
    import numpy as np

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
    import numpy as np

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
        assert "matches" in steps and "AC-ground" in steps
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
