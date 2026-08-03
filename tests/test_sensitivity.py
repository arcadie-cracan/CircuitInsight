"""OP-point sensitivity ranking and keep-set suggestion."""
import warnings
from pathlib import Path

import pytest

from circuitinsight import Analyzer
from circuitinsight.adapters.spectre import SpectreRun

GOLDEN = Path(__file__).resolve().parent / "golden" / "circuits"
FIX = Path(__file__).resolve().parent / "fixtures" / "spectre"


def test_cs_amp_sensitivities_match_hand_analysis():
    an = Analyzer.from_cin(GOLDEN / "cs_amp.cin.json")
    rep = an.sensitivities("V1", "vout", n_poles=1)

    dc = dict(rep.dc_gain)
    gds, gl = 8.5e-6, 1e-4                       # gds_M1, 1/RL
    assert dc["gm_M1"] == pytest.approx(1.0, abs=1e-9)
    assert dc["RL"] == pytest.approx(gl / (gds + gl), rel=1e-6)
    assert dc["gds_M1"] == pytest.approx(-gds / (gds + gl), rel=1e-6)
    assert dc["cgs_M1"] == pytest.approx(0.0, abs=1e-9)   # caps: no dc role

    p1 = dict(rep.pole_sens[0])
    cout = 100e-15 + 0.75e-15 + 1.7e-15
    assert p1["CL"] == pytest.approx(-100e-15 / cout, rel=1e-6)
    assert p1["RL"] == pytest.approx(-gl / (gds + gl), rel=1e-6)
    assert abs(p1.get("gm_M1", 0.0)) < 1e-9              # gm: no pole role


def test_5t_discovers_dc_gain_keep_set():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        run = SpectreRun(FIX / "ota5t" / "tb_ota5t.cin.json",
                         FIX / "ota5t" / "psf")
        an = run.analyzer()
        an.match("I0.MN0", "I0.MN1")
        an.match("I0.MP0", "I0.MP1")
    rep = an.sensitivities("VIND", "vout", n_poles=1)
    # A0 = gm/(gdsN + gdsP): the tool must find its participants unaided
    assert set(rep.suggest_keep("dc_gain", 3)) == {
        "gm_I0_MN0", "gds_I0_MN0", "gds_I0_MP0"}


def test_two_stage_discovers_pole_splitting_set():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        run = SpectreRun(FIX / "miller" / "tb_ota2s.cin.json",
                         FIX / "miller" / "psf")
        an = run.analyzer()
        an.match("I0.MN0", "I0.MN1")
        an.match("I0.MP0", "I0.MP1")
    rep = an.sensitivities("VIND", "vout", n_poles=1)
    # p1 = (g1+g3)(g5+g6)/(Cc*gm5): Miller cap and gm5 must rank on top
    top = rep.suggest_keep("p1", 5)
    assert top[0] == "I0_Cc"
    assert "gm_I0_MP2" in top[:2]
    # dc-gain top-6 reproduces the hand-picked A0 keep-set
    assert set(rep.suggest_keep("dc_gain", 6)) == {
        "gm_I0_MN0", "gds_I0_MN0", "gds_I0_MP0",
        "gm_I0_MP2", "gds_I0_MP2", "gds_I0_MN3"}


def test_bad_target_rejected():
    an = Analyzer.from_cin(GOLDEN / "cs_amp.cin.json")
    rep = an.sensitivities("V1", "vout", n_poles=1)
    with pytest.raises(ValueError, match="target"):
        rep.suggest_keep("q7")


def test_pursuit_reports_per_candidate_progress():
    """The exact-greedy rounds before a lowest-order solve were ~10 s of
    silence. Each candidate evaluation now reports, with the total a
    growing lower bound (rounds are unknown up front)."""
    import warnings as _w
    from pathlib import Path

    from circuitinsight.adapters.spectre import SpectreRun

    fix = Path(__file__).resolve().parent / "fixtures" / "spectre" / "ota5t"
    with _w.catch_warnings():
        _w.simplefilter("ignore")
        an = SpectreRun(fix / "tb_ota5t.cin.json",
                        fix / "psf").analyzer(cap_model="matrix")
        calls = []
        red = an.dominant_reactances("VIND", "vout", tol_db=1.0,
                                     fmin=1.0, fmax=1e9,
                                     progress=lambda d, t: calls.append((d, t)))
    assert red.selected                       # something was pursued
    assert calls
    dones = [d for d, _t in calls]
    assert dones == sorted(dones) and dones[-1] >= len(calls)  # monotone
    assert all(t >= d for d, t in calls)      # total is a live lower bound


def test_anchored_criterion_reads_the_band_edges():
    """The eps mode: |dH| <= eps*(|H| + anchor) with the anchor at the
    smaller band-edge |H|. On ota5t a band ending just past crossover
    keeps a small set at 10%; the achieved error is within eps; and the
    anchor equals the edge level, not a floor knob."""
    import numpy as np

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        run = SpectreRun(FIX / "ota5t" / "tb_ota5t.cin.json",
                         FIX / "ota5t" / "psf")
        an = run.analyzer(cap_model="matrix")

    red = an.dominant_reactances("VIND", "vout", fmin=1e3, fmax=3e8,
                                 eps=0.10)
    assert red.eps == pytest.approx(0.10)
    assert red.anchor is not None and red.anchor > 0
    achieved = red.errors_db[-1] if red.errors_db else red.baseline_db
    assert achieved <= 0.10 + 1e-9
    assert red.selected, "a band to past crossover needs reactances"
    assert "anchored" in red.report()

    # tighter eps can only grow the selection (same band)
    red2 = an.dominant_reactances("VIND", "vout", fmin=1e3, fmax=3e8,
                                  eps=0.02)
    assert len(red2.selected) >= len(red.selected)
    # the old floor path is untouched
    old = an.dominant_reactances("VIND", "vout", fmin=1e3, fmax=3e8,
                                 tol_db=1.0, floor_abs_db=0.0)
    assert old.eps is None and old.floor_is_abs


def test_pursuit_narrates_its_rounds():
    """The note channel: baseline, then one line per accepted reactance
    naming the element, the new error, and whether the pursuit stops or
    re-tries — the Log's answer to 'why did 70/74 grow'."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        run = SpectreRun(FIX / "ota5t" / "tb_ota5t.cin.json",
                         FIX / "ota5t" / "psf")
        an = run.analyzer(cap_model="matrix")

    notes = []
    red = an.dominant_reactances("VIND", "vout", fmin=1e3, fmax=3e8,
                                 eps=0.10, note=notes.append)
    assert notes and "baseline" in notes[0] and "candidates" in notes[0]
    rounds = [n for n in notes if n.startswith("round ")]
    assert len(rounds) == len(red.selected)
    assert red.selected[0] in rounds[0]
    assert "stopping" in rounds[-1] or "re-trying" in rounds[-1]
