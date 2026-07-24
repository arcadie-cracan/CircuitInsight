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
