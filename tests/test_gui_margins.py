"""The Margins readout: L-D surface of the margin ledger in the
Loop-gain bench -- the tab shows in its bench only, the tables carry
the folded contributor story and the element rows, and an OP-point
switch clears a readout that described the other point."""
import os
import warnings
from pathlib import Path

import pytest

pytest.importorskip("PySide6")
try:
    import PySide6.QtWidgets  # noqa: F401
except Exception as exc:                                  # pragma: no cover
    pytest.skip(f"PySide6.QtWidgets not loadable: {exc}",
                allow_module_level=True)
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

FIX = Path(__file__).resolve().parent / "fixtures" / "spectre"


@pytest.fixture(scope="module")
def qapp():
    import circuitinsight.gui.exprweb  # noqa: F401
    from PySide6.QtWidgets import QApplication
    yield QApplication.instance() or QApplication([])


def test_margins_tab_fills_and_follows_the_bench(qapp, tmp_path):
    from circuitinsight.gui.app import MainWindow

    w = MainWindow()
    type(w).settings_path = None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            w.open_session(FIX / "miller" / "tb_ota2s_stb.cin.json",
                           FIX / "miller" / "psf_stb", None)
        # the stb run opens the Loop-gain bench: Margins and the
        # split-out Sensitivities are its tabs, and the bench's verbs
        # sit in the MAIN toolbar (no point set: no Compare corners)
        assert w.mode_combo.currentText() == "Loop gain"
        assert w.tabs.isTabVisible(w.tabs.indexOf(w._ml_tab))
        assert w.tabs.isTabVisible(w.tabs.indexOf(w._ms_tab))
        assert w._ml_btn_act.isVisible() and w._ml_cplx_act.isVisible()
        assert not w._ml_cmp_act.isVisible()
        w.mode_combo.setCurrentText("Transfer")
        assert not w.tabs.isTabVisible(w.tabs.indexOf(w._ml_tab))
        assert not w.tabs.isTabVisible(w.tabs.indexOf(w._ms_tab))
        assert not w._ml_btn_act.isVisible()
        w.mode_combo.setCurrentText("Loop gain")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = w.margins_sync("IPRB0")
        led = out["ledger"]
        assert led.pm_deg is not None
        assert "PM" in w._ml_head.text()
        assert "certificate" in w._ml_head.text()
        # the folded contributor story: singularities only (the T0
        # anchor lives in the equation), ascending frequency, rest last
        assert w.ml_tbl.rowCount() >= 4
        assert w.ml_tbl.item(0, 0).text().startswith("p1")
        col0 = [w.ml_tbl.item(i, 0).text()
                for i in range(w.ml_tbl.rowCount())]
        assert any("pole" in t for t in col0)
        assert any("RHP zero" in t for t in col0)
        assert "smaller terms" in col0[-1]
        # the table wears the EQUATION's term names, and rows the
        # equation folded into its constant say so
        assert any(t.startswith("p1  ·") for t in col0)
        assert any(t.startswith("z1  ·") for t in col0)
        assert "(in remainder)" in col0[-1]
        # ...and runs in the same order: anchor, then ascending
        # frequency, remainder rows last
        named = [i for i, t in enumerate(col0)
                 if t.split("  ·")[0] in ("p1", "p2", "p3",
                                          "z1", "z2", "z3")]
        f0s = [w._ml_rows[i].f0_hz for i in named]
        assert f0s == sorted(f0s)
        assert any(ln.startswith("T0 = ")
                   for ln in w._ml_eq_lines)     # the anchor line
        # the margin equation renders (text fallback offscreen)
        assert w._ml_eq_lines
        assert w._ml_eq_lines[0].startswith("PM ~ 180")
        assert not w._ml_eq.isHidden()
        # the root-sensitivity column carries the L-G attribution
        moved = [w.ml_tbl.item(r, 4).text()
                 for r in range(w.ml_tbl.rowCount())
                 if w.ml_tbl.item(r, 4) is not None]
        assert any("gm_" in t for t in moved)
        # L-J: the element rows grouped per device, ranked, capped
        tree = w.ms_tree
        assert 1 <= tree.topLevelItemCount() <= 14
        top = tree.topLevelItem(0)
        assert top.text(1) == "device"            # a device group leads
        assert top.childCount() >= 1
        assert top.child(0).text(0).startswith("gm_")
        assert top.child(0).text(1) == "re-bias"
        # a netlist component stands alone with the exact-advice class
        comps = [tree.topLevelItem(i)
                 for i in range(tree.topLevelItemCount())
                 if tree.topLevelItem(i).text(1) == "component"]
        assert comps and any("C" in c.text(0) for c in comps)
        # a second call serves from the session cache; over the engine
        # process a reply is a fresh unpickle, so identity holds only
        # in-process -- the cached VALUES hold in both modes
        again = w.controller.margin_ledger("IPRB0")
        if type(w.controller).__name__ == "RemoteSession":
            assert again["ledger"].pm_deg == out["ledger"].pm_deg
        else:
            assert again is out
        # L-K: the What-if bridge ticks the strongest movers into the
        # keep set (solve deferred here); What-if shows in this bench
        assert w.ml_whatif_btn.isEnabled()
        found = w.margins_to_whatif(solve=False)
        assert found and set(found) == set(w.checked_keep())
        assert found[0] in {r["symbol"]
                            for r in w._ms_sens[:3]}
        assert w.tabs.isTabVisible(w.tabs.indexOf(w._whatif_tab))
        # cross-light: selecting rows never crashes; without hints the
        # schematic stays untouched (guarded no-op). Selecting a
        # singularity turns the right panel into ITS contributor list
        # (root sensitivities per device); deselection restores the
        # margin knobs
        w.ml_tbl.selectRow(0)                    # p1
        assert w.ms_tree.headerItem().text(3).startswith("df0")
        assert "contributors to p1" in w.ms_tree.headerItem().text(0)
        assert w.ms_tree.topLevelItemCount() >= 1
        w.ml_tbl.clearSelection()
        assert w.ms_tree.headerItem().text(3).startswith("dPM")
        # ...and clicking the selected row AGAIN releases it too
        w.ml_tbl.selectRow(0)
        w._ml_cell_clicked(0, 0)                 # the arming click
        assert w.ms_tree.headerItem().text(3).startswith("df0")
        w._ml_cell_clicked(0, 0)                 # the releasing click
        assert not w.ml_tbl.selectedItems()
        assert w.ms_tree.headerItem().text(3).startswith("dPM")
        w.ml_tbl.selectRow(1)
        w.ms_tree.setCurrentItem(w.ms_tree.topLevelItem(0))
        assert w.sch_view._hl_set == []
        # a contributor row cross-selects its device in the
        # Instances tree (the standard highlight and OP card follow)
        picked = w.ms_tree.topLevelItem(0).text(0)
        cur = w.devices.currentItem()
        assert cur is not None
        assert w.devices.device_name(cur) == picked
        assert w.dev_tabs.currentWidget() is w.devices
        # a matched group's shared symbol is RE-SPELLED onto the
        # device it acts through in the sensitivity views (no
        # match-group indirection there; formulas keep the shared
        # symbol, the tooltip carries it)
        assert w._ms_local_name("gm_I0_NM14", "I0.NM11") == "gm_I0_NM11"
        assert w._ms_local_name("gm_I0_NM14", "I0.NM14") == "gm_I0_NM14"
        assert w._ms_local_name("CL", "I0.NM11") == "CL"
        assert w._ms_local_name("gmbs_I0_NM14", "I0.NM2") ==             "gmbs_I0_NM2"
        # a theta link in the equation selects its root's row (the
        # web shell scrolls to the definition line; here the handler
        # is driven directly -- offscreen has no WebEngine)
        w.ml_tbl.clearSelection()
        w._ml_term_clicked("p1")
        assert w.ml_tbl.currentRow() == 0
        assert w.ms_tree.headerItem().text(3).startswith("df0")
        # the selection's THIRD view: a vertical cursor at the exact
        # root frequency on every Bode axis; deselection clears it
        # one line per axis plus the f_p1 label at the frequency axis
        assert len(w._ml_fmarks) == len(w.canvas.figure.axes) + 1 > 1
        f0 = w._ml_rows[0].f0_hz
        assert w._ml_fmarks[0].get_xdata()[0] == pytest.approx(f0)
        assert w._ml_fmarks[-1].get_text() == "$f_{p1}$"
        w.ml_tbl.clearSelection()
        assert w._ml_fmarks == []
        # leaving the bench releases the selection: the sensitivity
        # map (and the contributor view) go with it
        w.ml_tbl.selectRow(0)
        assert w.ms_tree.headerItem().text(3).startswith("df0")
        w.mode_combo.setCurrentText("Transfer")
        assert not w.ml_tbl.selectedItems()
        assert w.ms_tree.headerItem().text(3).startswith("dPM")
        w.mode_combo.setCurrentText("Loop gain")
        # a single-psf session offers no corner comparison
        assert not w._ml_cmp_act.isVisible()
        # clearing hook (the OP-switch path)
        w._ml_clear()
        assert w.ml_tbl.rowCount() == 0 and w._ml_head.text() == ""
    finally:
        w.close()
        type(w).settings_path = None


def test_corner_comparison_view(qapp, tmp_path):
    """A point-set session compares the ledger across its corners:
    rows are the bound corner's terms with one share column per
    corner plus the drift, headed by PM / f_u / GM rows; identical
    copied corners drift nowhere."""
    import shutil

    from circuitinsight.gui.app import MainWindow

    root = tmp_path / "ExplorerRun.0"
    for i, corner in ((1, "tt 27"), (2, "ss -40")):
        psf = root / str(i) / "work_tb" / "psf"
        shutil.copytree(FIX / "miller" / "psf_stb", psf)
        (psf / "runObjFile").write_text(
            'HEADER\n'
            f'"#maeCorner" "{corner}"\n'
            f' "#maeDesignPoint" {i}\n')
    w = MainWindow()
    type(w).settings_path = None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            w.open_session(FIX / "miller" / "tb_ota2s_stb.cin.json",
                           root, None)
        assert w._ml_cmp_act.isVisible()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cmp_ = w.margins_cmp_sync("IPRB0")
        tbl = w.mlc_tbl
        assert tbl.isVisibleTo(w._ml_tab)
        assert tbl.columnCount() == 4          # term, 2 corners, drift
        heads = [tbl.horizontalHeaderItem(k).text()
                 for k in range(tbl.columnCount())]
        assert heads == ["term", "tt 27", "ss -40", "Δ"]
        col0 = [tbl.item(r, 0).text() for r in range(tbl.rowCount())]
        assert col0[0] == "PM (°)" and "T0" in col0
        assert col0[-1] == "other terms"
        # identical copied corners: every drift cell is zero
        assert all(r["delta"] == pytest.approx(0.0, abs=1e-9)
                   for r in cmp_["rows"])
        # the OP switch clears the stale view
        w._ml_clear()
        assert not tbl.isVisibleTo(w._ml_tab)
    finally:
        w.close()
        type(w).settings_path = None


def test_bottom_tabs_south_and_max_toggle(qapp):
    """The result strip's tab bar sits at the BOTTOM edge, and each
    pane's corner toggle maximizes it -- the other pane collapses,
    a second press restores the default proportions, and the truth is
    the sizes, so an external restore never strands the button."""
    from PySide6.QtWidgets import QTabWidget

    from circuitinsight.gui.app import MainWindow

    w = MainWindow()
    type(w).settings_path = None
    try:
        assert w.tabs.tabPosition() == QTabWidget.South
        w.resize(1200, 800)
        w.show()
        w._toggle_right_max("bottom")
        s = w.right_split.sizes()
        assert s[0] == 0 and s[1] > 0
        w._toggle_right_max("bottom")
        s = w.right_split.sizes()
        assert s[0] > 0 and s[1] > 0
        w._toggle_right_max("top")
        s = w.right_split.sizes()
        assert s[1] == 0 and s[0] > 0
        # an external restore (the double-click path) then a press
        # must MAXIMIZE again, not "restore" a second time
        w.right_split.setSizes([500, 240])
        w._toggle_right_max("top")
        assert w.right_split.sizes()[1] == 0
    finally:
        w.close()
        type(w).settings_path = None


def test_root_tooltip_carries_the_verdict_and_its_residuals(qapp):
    """The f0 cell is where a designer looks at ONE root, so the jet's
    reading belongs there: the expression when one certifies, and
    otherwise why it does not. A partial reading must show what it did
    read -- naming half a pole's structure beats naming none of it."""
    from circuitinsight.gui.app import MainWindow

    w = MainWindow()
    type(w).settings_path = None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            w.open_session(FIX / "miller" / "tb_ota2s_stb.cin.json",
                           FIX / "miller" / "psf_stb", None)
            out = w.margins_sync("IPRB0")
        led = out["ledger"]
        tbl = w.ml_tbl
        tips = {}
        for r in range(tbl.rowCount()):
            it = tbl.item(r, 1)
            if it is not None and it.toolTip():
                tips[tbl.item(r, 0).text()] = it.toolTip()
        assert tips, "no root carried a reading"
        certified = [e for e in led.entries
                     if getattr(e, "reconstruction", None)
                     and e.reconstruction.verdict == "formula"]
        assert certified, "the bench must certify something"
        joined = "\n".join(tips.values())
        assert "f₀ ≈" in joined                 # a formula was shown
        assert "coverage:" in joined            # with its residuals
        assert "the named structure gives" in joined
        # and an unread root explains itself rather than going blank
        unread = [e for e in led.entries
                  if getattr(e, "reconstruction", None)
                  and e.reconstruction.verdict != "formula"]
        if unread:
            assert ("SHARED" in joined) or ("partly read:" in joined)
    finally:
        w.close()
