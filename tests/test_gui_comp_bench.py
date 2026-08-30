"""The compensation bench's full surface: the spec goal, multi-branch (NMC)
synthesis, mirrored fully-differential pairs, and the re-compensate strip.

The engine for all of this is tested elsewhere; what matters here is that the
GUI reaches it and that what it PREVIEWS is what it REPORTED -- the preview
updater must run on the same system the search ran on, exclusions included,
or the plot would show a margin nobody designed.
"""
import os
import warnings
from pathlib import Path

import numpy as np
import pytest

# PySide6 the package can import while PySide6.QtWidgets fails to LOAD (no Qt
# platform libraries in a headless environment, e.g. the public snapshot's
# fresh-venv gate), so both need guarding -- and the offscreen platform must
# be set here, not left to the caller's environment.
pytest.importorskip("PySide6")
try:
    import PySide6.QtWidgets  # noqa: F401
except Exception as exc:                                  # pragma: no cover
    pytest.skip(f"PySide6.QtWidgets not loadable: {exc}",
                allow_module_level=True)
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from circuitinsight.analysis.compensate import Candidate  # noqa: E402
from circuitinsight.gui import view                       # noqa: E402

FIX = Path(__file__).resolve().parent / "fixtures" / "spectre"
FD_STRIP = ("CM1p", "CM1n", "CM2p", "CM2n",
            "IPRB1p", "IPRB1n", "IPRB2p", "IPRB2n")


@pytest.fixture(scope="module")
def qapp():
    import circuitinsight.gui.exprweb  # noqa: F401
    from PySide6.QtWidgets import QApplication
    yield QApplication.instance() or QApplication([])


def _win(tmp_path, cin, psf):
    from circuitinsight.gui.app import MainWindow
    w = MainWindow()
    type(w).settings_path = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        w.open_session(cin, psf, None)
    return w


# ------------------------------------------------------ goal / target UI
def test_target_spin_follows_the_goal(qapp, tmp_path):
    """Each goal exposes ITS target: flatness budget (mfm), PM floor
    (pm), peak sensitivity (spec) -- alternatives, one at a time."""
    w = _win(tmp_path, FIX / "miller" / "tb_ota2s_stb.cin.json",
             FIX / "miller" / "psf_stb")
    try:
        # isHidden() is the explicit hide state the goal handler sets;
        # isVisibleTo() would also depend on which tab is current
        w.goal_combo.setCurrentText("pm")
        assert not w.comp_pm_spin.isHidden()
        assert w.ms_spin.isHidden() and w.flat_spin.isHidden()
        w.goal_combo.setCurrentText("spec")
        assert not w.ms_spin.isHidden()
        assert w.comp_pm_spin.isHidden() and w.flat_spin.isHidden()
        w.goal_combo.setCurrentText("mfm")
        assert not w.flat_spin.isHidden()
        assert w.comp_pm_spin.isHidden() and w.ms_spin.isHidden()
        # only the active target reaches the search
        w.goal_combo.setCurrentText("spec")
        w.ms_spin.setValue(1.25)
        assert w._comp_kw() == {"goal": "spec", "ms_target": 1.25}
        w.goal_combo.setCurrentText("mfm")
        w.flat_spin.setValue(0.5)
        assert w._comp_kw() == {"goal": "mfm", "flat_tol_db": 0.5}
        # a single-psf session offers no worst-corner sizing
        assert w.robust_chk.isHidden()
    finally:
        w.close()
        type(w).settings_path = None


def test_spec_goal_runs_and_reports_ms(qapp, tmp_path):
    """goal='spec' reaches the peak-sensitivity search and the table fills."""
    w = _win(tmp_path, FIX / "miller" / "tb_ota2s_stb.cin.json",
             FIX / "miller" / "psf_stb")
    try:
        w.mode_combo.setCurrentText("Compensate")
        w.probe_combo.setCurrentText("IPRB0")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sugg = w.suggest_sync(
                "IPRB0", goal="spec", ms_target=1.5,
                candidates=[Candidate("miller", "I0.net1", "vout",
                                      "Miller port", 1.0)])
        assert sugg and w.comp_tbl.rowCount() == len(sugg)
        assert any(s.achieved and s.spec_dev is not None for s in sugg)
        # the verdict is the row's COLOR, not an ok column
        from circuitinsight.gui import theme
        assert w.comp_tbl.columnCount() == 10
        # a single-point session shows no worst-case columns
        assert w.comp_tbl.isColumnHidden(8) and w.comp_tbl.isColumnHidden(9)
        # spec rows report the measured criterion in the peak column
        assert w.comp_tbl.item(0, 7).text().startswith("Ms ")
        for i, s in enumerate(sugg):
            key = "ok" if s.achieved else "error"
            assert (w.comp_tbl.item(i, 0).background().color().name()
                    == theme.SEVERITY[key][0])
        # selecting a row proposes its branch to the Schematic pane
        # (this run ships no hints, so nothing renders -- the state
        # still sets, and it clears with fresh results)
        w.comp_tbl.selectRow(0)
        assert w._sch_proposed
        na, nb, lbl, _wr = w._sch_proposed[0]
        assert (na, nb) == (sugg[0].candidate.node_a,
                            sugg[0].candidate.node_b)
        assert "F" in lbl
        w._on_comp_done((w._comp_baseline, sugg))
        assert w._sch_proposed == ()
    finally:
        w.close()
        type(w).settings_path = None


def test_worst_case_columns_span_the_point_set(qapp, tmp_path):
    """A session over an operating-point SET re-measures every sized
    suggestion on every point: the worst-case columns show, carry the
    setting point in their tooltips, and the verdict tint gates on the
    worst point, not the nominal one. The synthetic tree copies one
    psf, so worst must equal nominal."""
    import shutil

    root = tmp_path / "ExplorerRun.0"
    for i, corner in ((1, "tt 27"), (2, "ss -40")):
        psf = root / str(i) / "work_tb" / "psf"
        shutil.copytree(FIX / "miller" / "psf_stb", psf)
        (psf / "runObjFile").write_text(
            'HEADER\n'
            f'"#maeCorner" "{corner}"\n'
            f' "#maeDesignPoint" {i}\n')
    w = _win(tmp_path, FIX / "miller" / "tb_ota2s_stb.cin.json", root)
    try:
        w.mode_combo.setCurrentText("Compensate")
        w.probe_combo.setCurrentText("IPRB0")
        # a point-set session offers worst-corner sizing, and defaults
        # to it -- the corners were run to be held
        assert not w.robust_chk.isHidden() and w.robust_chk.isChecked()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sugg = w.suggest_sync(
                "IPRB0", goal="mfm", robust=True, exclude=("I0.Cc",),
                candidates=[Candidate("miller", "I0.net1", "vout",
                                      "Miller port", 1.0)])
        assert sugg and w._comp_robust is not None
        assert not w.comp_tbl.isColumnHidden(8)
        assert not w.comp_tbl.isColumnHidden(9)
        assert w.comp_tbl.item(0, 9).text().endswith("dB")
        assert w.comp_tbl.item(0, 9).toolTip().startswith("at ")
        # identical copied points: worst equals nominal
        assert w.comp_tbl.item(0, 9).text() == w.comp_tbl.item(0, 7).text()
        rb = w._comp_robust[0]
        assert rb["labels"] == ["tt 27", "ss -40"]
        assert rb["worst_flat_db"] == pytest.approx(sugg[0].flat_db,
                                                    abs=1e-6)
        assert w._comp_worst_ok(rb) == (rb["worst_flat_db"] <= 1.0)
    finally:
        w.close()
        type(w).settings_path = None


# --------------------------------------------------- multi-branch (NMC)
@pytest.fixture(scope="module")
def _fd_cands():
    return [Candidate("miller", "outp", "n1p", "outer Miller", 50.0),
            Candidate("miller", "outp", "n2p", "inner Miller", 10.0)]


def test_multibranch_mirrored_network_and_preview(qapp, tmp_path, _fd_cands):
    """The payoff: on the fully-differential three-stage amplifier stripped of
    both Miller pairs, the bench grows a TWO-branch mirrored network, lists a
    row per branch showing each mirror twin, and -- the invariant that matters
    -- previews the margin it reported, over all four physical branches."""
    w = _win(tmp_path, FIX / "nmc3d" / "tb_nmc3d.cin.json",
             FIX / "nmc3d" / "psf")
    try:
        w.mode_combo.setCurrentText("Compensate")
        w.probe_combo.setCurrentText("IPRB_DM")
        w._comp_probe = "IPRB_DM"
        w.mirror_chk.setChecked(True)
        mirror = w._comp_mirror()
        assert mirror and mirror["outp"] == "outn"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = w.suggest_sync(
                "IPRB_DM", k_max=2, goal="spec", ms_target=1.1,
                exclude=FD_STRIP, candidates=_fd_cands, mirror=mirror,
                c_grid=np.geomspace(0.5e-12, 20e-12, 18))
        assert res.achieved and len(res.branches) == 2
        assert w.comp_tbl.rowCount() == 2
        # every branch is a mirrored pair, and the table says so
        assert all(b.twin is not None for b in res.branches)
        assert "outn" in w.comp_tbl.item(0, 0).text()
        assert "goal met" in w._comp_hint.text()
        assert "step 1" in w._comp_steps.text()

        w.comp_tbl.selectRow(0)
        qapp.processEvents()
        note = w._comp_hint.text()
        assert "4 branches installed" in note     # 2 pairs, both sides
        assert f"{res.pm_deg:.1f}" in note        # preview == reported
        labels = [ln.get_label() for ln in w.canvas.figure.axes[0].get_lines()]
        assert "preview" in labels
    finally:
        w.close()
        type(w).settings_path = None


def test_preview_honours_the_strip(qapp, tmp_path, _fd_cands):
    """Regression: the preview updater must be built on the SAME system the
    search ran on. With the original Miller pairs stripped, previewing on the
    unstripped system would stack the suggestion on top of the very branches
    the search removed, and report a margin nobody designed."""
    w = _win(tmp_path, FIX / "nmc3d" / "tb_nmc3d.cin.json",
             FIX / "nmc3d" / "psf")
    try:
        w.probe_combo.setCurrentText("IPRB_DM")
        w._comp_probe = "IPRB_DM"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = w.suggest_sync(
                "IPRB_DM", k_max=2, goal="spec", ms_target=1.1,
                exclude=FD_STRIP, candidates=_fd_cands,
                mirror=view.mirror_map(
                    w.controller._analyzer_ready()
                    .system("IPRB_DM").node_index),
                c_grid=np.geomspace(0.5e-12, 20e-12, 18))
        assert w._comp_exclude == FD_STRIP
        w.comp_tbl.selectRow(0)
        qapp.processEvents()
        # the previewed PM matches the reported one to the printed digit
        assert f"preview PM {res.pm_deg:.1f}" in w._comp_hint.text()
    finally:
        w.close()
        type(w).settings_path = None


# ------------------------------------------------- one-column bench
def test_compensate_presents_its_own_column(qapp, tmp_path):
    """Compensate is a suggestion tool: the keep set never enters the
    search, so the symbols column and Solve hide, the bench's controls
    take the column, the probe lives in the panel (mirroring the
    toolbar state), and the crumb's last step reads Suggest."""
    w = _win(tmp_path, FIX / "miller" / "tb_ota2s_stb.cin.json",
             FIX / "miller" / "psf_stb")
    try:
        w.mode_combo.setCurrentText("Compensate")
        # toolbar widgets never report visible on an unshown window:
        # the ACTIONS carry the state the mode logic controls
        assert not w._keep_col.isVisibleTo(w)
        assert not w._solve_act.isVisible()
        assert not w._probe_act.isVisible()
        assert w.comp_col.isVisibleTo(w)
        assert w.comp_probe_combo.currentText() == \
            w.probe_combo.currentText()
        # several Tian candidates in this run (iprobe + vsources):
        # the loop stays choosable, in the panel
        assert w.comp_probe_combo.isEnabled()
        assert w.comp_probe_combo.count() == w.probe_combo.count()
        assert "Suggest" in w.crumb.text()
        assert "Choose symbols" not in w.crumb.text()
        idx = w.tabs.indexOf(w._comp_tab)
        assert w.tabs.tabText(idx) == "Suggestions"
        # Summary/Expression/Error describe a SOLVED response; the
        # bench has no Solve, so they hide with it
        for view_tab in (w.summary, w._expr_tab, w.err_canvas):
            assert not w.tabs.isTabVisible(w.tabs.indexOf(view_tab))

        w.mode_combo.setCurrentText("Loop gain")
        assert w._keep_col.isVisibleTo(w)
        assert w._solve_act.isVisible()
        assert w._probe_act.isVisible()
        assert not w.comp_col.isVisibleTo(w)
        assert "Choose symbols" in w.crumb.text()
    finally:
        w.close()


def test_search_narrates_through_the_log(qapp, tmp_path):
    """The suggester's progress callback lands in the Log: a
    nine-minute screen used to show nothing after the start line."""
    w = _win(tmp_path, FIX / "miller" / "tb_ota2s_stb.cin.json",
             FIX / "miller" / "psf_stb")
    try:
        events = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            w.controller.suggest_compensation(
                "IPRB0", goal="pm", pm_target=55.0, exclude=("I0.Cc",),
                c_grid=np.array([1e-12, 5e-12]),
                r_grid=np.array([0.0, 2e3]),
                progress=lambda msg, done=None, total=None:
                    events.append((msg, done, total)))
        text = "  ".join(m for m, _d, _t in events)
        assert "T0" in text and "candidates" in text
        assert any("screening" in m and d is not None
                   for m, d, _t in events)
        assert "screen done" in text
    finally:
        w.close()
