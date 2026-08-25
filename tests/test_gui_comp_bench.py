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
        assert w.comp_tbl.columnCount() == 8
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


# ------------------------------------------------------- cross-probe (M10)
def test_cross_probe_resolves_and_sends(qapp, tmp_path):
    """Clicking a symbol or selecting a device row sends that instance to
    Virtuoso; with probing off both are silent no-ops."""
    from circuitinsight.virtuoso import xprobe

    class FakeWs:
        def __init__(self):
            self.calls = []

        def __getitem__(self, name):
            def call(*args):
                self.calls.append((name, args))
                return True
            return call

        def close(self):
            pass

    w = _win(tmp_path, FIX / "miller" / "tb_ota2s_stb.cin.json",
             FIX / "miller" / "psf_stb")
    try:
        assert w._xprobe is None
        assert w.xprobe_symbol("gm_I0_MN0") is False      # off: no-op

        ws = FakeWs()
        w._xprobe = xprobe.CrossProbe(ws)
        assert w.xprobe_symbol("gm_I0_MN0")
        assert ws.calls[-1] == ("CInHighlight", (["I0.MN0"],))
        assert w.xprobe_symbol("I0_Cc")                   # passive: own path
        assert ws.calls[-1] == ("CInHighlight", (["I0.Cc"],))
        assert w.xprobe_symbol("gm_NOTADEVICE") is False  # unknown: no send

        n = len(ws.calls)
        w.devices.setCurrentItem(w.devices.leaf_items()[0])
        qapp.processEvents()
        assert len(ws.calls) > n                          # selection probes
    finally:
        w.close()
        type(w).settings_path = None


def test_cross_probe_toggle_unchecks_itself_when_unavailable(qapp, tmp_path):
    """A failed connection must not leave the menu claiming cross-probe is on;
    the reason goes to the message strip."""
    w = _win(tmp_path, FIX / "miller" / "tb_ota2s_stb.cin.json",
             FIX / "miller" / "psf_stb")
    try:
        w.a_xprobe.setChecked(True)          # no Virtuoso in a test run
        qapp.processEvents()
        assert not w.a_xprobe.isChecked()
        assert w._xprobe is None
        assert "cross-probe unavailable" in w.msg_strip.text()
    finally:
        w.close()
        type(w).settings_path = None


def test_cross_probe_toggle_is_remembered(qapp, tmp_path):
    """The toggle persists like the cap model and the last mode. Restoring
    "on" with no Virtuoso running must re-attempt, fail gracefully and end up
    unchecked -- never a checked toggle with no connection behind it."""
    from circuitinsight.gui.app import MainWindow

    ini = str(tmp_path / "gui.ini")
    w = _win(tmp_path, FIX / "miller" / "tb_ota2s_stb.cin.json",
             FIX / "miller" / "psf_stb")
    try:
        MainWindow.settings_path = ini
        w.a_xprobe.blockSignals(True)      # pretend a session had it on
        w.a_xprobe.setChecked(True)
        w.a_xprobe.blockSignals(False)
        w.close()                          # closeEvent persists it
    finally:
        type(w).settings_path = None

    MainWindow.settings_path = ini
    try:
        w2 = MainWindow()                  # restores; no Virtuoso here
        try:
            assert not w2.a_xprobe.isChecked()   # honest about the failure
            assert w2._xprobe is None
        finally:
            w2.close()
    finally:
        MainWindow.settings_path = None


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
