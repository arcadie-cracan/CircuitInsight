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


def test_on_demand_readings_reach_the_row(qapp):
    """The two DEEP readings are the ones the ledger raises but does
    not settle: what an aggregate G_eff actually is, and why a root
    reads as SHARED. Both were reachable only from Python until the
    bench grew a right-click for them, so what matters here is the
    whole path -- dispatch, attach to the entry, and the answer
    landing in the tooltip a designer actually reads."""
    from circuitinsight.gui.app import MainWindow

    w = MainWindow()
    type(w).settings_path = None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            w.open_session(FIX / "miller" / "tb_ota2s_stb.cin.json",
                           FIX / "miller" / "psf_stb", None)
            w.margins_sync("IPRB0")
            row = next(r for r, e in enumerate(w._ml_rows)
                       if e.kind == "pole" and e.f0_hz > 0 and e.roots)
            got = w.ml_reading_sync(row, "ray")
        assert got is not None and got["what"] == "ray"
        e = w._ml_rows[row]
        v = e.ray_reading
        assert v.kind in ("localized", "collapsible", "distributed")
        # the verdict must reach the cell the designer reads, not just
        # the log -- that was the whole gap
        tip = w.ml_tbl.item(row, 1).toolTip()
        assert v.describe() in tip
    finally:
        w.close()


def test_the_port_action_is_gated_on_a_conductive_aggregate():
    """A C_eff names a capacitance and the driving-point route reads
    conductances only, so the menu must say so rather than offer a
    computation that returns nothing. Gating it here keeps an expected
    answer from arriving as a job failure."""
    from circuitinsight.gui.app import MainWindow

    gate = MainWindow._ml_port_gate

    class E:
        def __init__(self, agg):
            self.reconstruction = (
                None if agg is None else type("R", (), {"aggregate": agg}))

    ok, why = gate(E({"symbol": "G_eff", "unit": "S"}))
    assert ok and not why
    ok, why = gate(E({"symbol": "C_eff", "unit": "F"}))
    assert not ok and "conductances only" in why
    ok, why = gate(E(None))
    assert not ok and "names no aggregate" in why
    ok, why = gate(object())              # a row with no reading at all
    assert not ok and why


def test_reading_renderers_speak_both_verdicts():
    """The port text must not dress a root in algebra that does not
    apply: when the port conductance disagrees with the value the
    identity fixed, it has to SAY so."""
    from circuitinsight.gui.app import MainWindow

    base = {"port": "cl", "keep": ("gds_M1", "gds_M2"), "num": "N",
            "den": "D", "value": 2e-5, "seconds": 1.2,
            "aggregate": {"symbol": "G_eff", "value": 2e-5},
            "branches": [{"gate": "gds_M2", "conductance": 1.2e-5,
                          "text": "gds_M2"},
                         {"gate": "gds_M1", "conductance": 8e-6,
                          "text": None}]}
    ok = MainWindow._ml_port_text({**base, "matches": True,
                                   "ratio": 1.0})
    assert "G_eff at the cl port" in ok
    assert "IS this node's time constant" in ok
    # the READABLE decomposition, and an unnamed branch owned up to --
    # never the raw num/den, which are uncancelled hundred-digit
    # rationals and unreadable on a row
    assert "2 parallel branches" in ok
    assert "gds_M2" in ok and "<not named>" in ok
    bad = MainWindow._ml_port_text({**base, "matches": False,
                                    "ratio": 5.2, "value": 1e-4})
    assert "does NOT match" in bad
    assert "stands unexpanded" in bad
    assert "IS this node's time constant" not in bad


def test_the_expansion_is_typeset_like_the_rest_of_the_equation():
    """The port's answer goes INTO the equation, where the bare G_eff
    is printed, so it has to be typeset the same way: real fractions,
    and every identifier through the session's own texer rather than
    pasted in raw beside the aliased names. The numeric residual a
    branch may carry must survive that pass -- the 'e' of 1.54e-05 is
    not an identifier."""
    from circuitinsight.gui.app import MainWindow

    def tx(n):
        return r"\mathrm{%s}" % n.replace("_", "")

    out = {"matches": True, "branches": [
        {"text": "gds_NM14*(gds_PM11+gds_NM10)/(gm_NM14+gmbs_NM14)",
         "conductance": 3.9e-7},
        {"text": "gds_PM12*gds_PM13/(gm_PM12+gmbs_PM12+1.54e-05)",
         "conductance": 2.6e-7}]}
    e = MainWindow._ml_expansion(out, tx)
    assert e["tex"].count(r"\frac") == 2      # one per branch
    assert r"\cdot" in e["tex"]
    assert "1.54e-05" in e["tex"], "the residual was mangled"
    assert r"\mathrm{gdsNM14}" in e["tex"], "texer not applied"
    assert "gds_NM14" not in e["tex"], "a raw name leaked through"
    # the plain-text form stays the readable one
    assert e["text"].startswith("gds_NM14*(")

    # a branch nobody could name must NOT reach the equation: a sum
    # with a hole in it is not the expression
    half = {"matches": True, "branches": [
        {"text": "gds_PM12", "conductance": 2.6e-7},
        {"text": None, "conductance": 3.9e-7}]}
    assert MainWindow._ml_expansion(half, tx) is None
    # nor does one whose port did not certify
    nomatch = {"matches": False, "branches": out["branches"]}
    assert MainWindow._ml_expansion(nomatch, tx) is None


def test_the_equation_drops_the_definition_it_no_longer_needs():
    """`reads G_eff/(2pi CL) with G_eff = 634 nS` loses that trailing
    clause once a port expansion writes the algebra in the symbol's
    place. The value belongs to the SYMBOL; with nothing named for it
    to attach to, trailing it anyway states that a frequency equals a
    conductance."""
    from circuitinsight.analysis.reconstruct import Block, Reconstruction

    blk = Block(members=("CL",), exponent=-1.0, exponent_raw=-1.0,
                exponent_from_rho=None)
    rec = Reconstruction(
        verdict="partial", text="G_eff/(2pi*CL)", blocks=(blk,),
        aggregate={"symbol": "G_eff", "unit": "S", "value": 6.34e-7,
                   "exponent": 1.0, "blocks": (blk,)})
    assert rec.display_text() == "G_eff/(2pi*CL)"
    rec.expansion = {"text": "gds_A+gds_B", "tex": r"g_A+g_B"}
    assert rec.display_text() == "[gds_A+gds_B]/(2pi*CL)"
    assert "G_eff" not in rec.display_text()
    # and the LaTeX picks it up without being asked again
    assert r"g_A+g_B" in rec.tex()
    assert r"\mathrm{eff}" not in rec.tex()


def test_the_aggregate_value_never_trails_an_expanded_reading(qapp):
    """A UNITS guard, driven through the real renderer.

    `reads <frequency expr> = <conductance>` is nonsense, and it is
    exactly what substituting the algebra in and keeping the trailing
    value produces. The value belongs to the SYMBOL: either that is
    named and carries it, or the algebra stands alone."""
    import warnings as _w

    from circuitinsight.analysis.marginledger import (
        margin_equation, margin_equation_text)
    from circuitinsight.session import SessionController

    with _w.catch_warnings():
        _w.simplefilter("ignore")
        s = SessionController.open(
            str(FIX / "miller" / "tb_ota2s_stb.cin.json"),
            str(FIX / "miller" / "psf_stb"))
        s.set_matches(*s.suggest_matches())
        probe = s.stb_probe() or s.probes[0]
        led = s.margin_ledger(probe)["ledger"]

    def reads_lines(l):
        return [x for x in margin_equation_text(margin_equation(l))
                if "reads" in x]

    e = next((x for x in led.entries
              if getattr(x, "reconstruction", None)
              and x.reconstruction.aggregate), None)
    if e is None:
        pytest.skip("this fixture names no aggregate at this pass")
    agg = e.reconstruction.aggregate
    before = reads_lines(led)
    assert any(f"with {agg['symbol']} =" in x for x in before), before

    e.reconstruction.expansion = {"text": "gds_A+gds_B",
                                  "tex": r"g_A+g_B"}
    after = reads_lines(led)
    joined = " | ".join(after)
    assert "[gds_A+gds_B]" in joined, joined
    assert agg["symbol"] not in joined
    # the conductance must not trail what is now a frequency
    assert f"{agg['value']:.4g}" not in joined, (
        "a conductance trailed a frequency expression")


def test_a_port_reading_rewrites_the_equation_pane(qapp):
    """The whole point of the on-demand port reading: the answer has
    to reach the EQUATION, where the bare aggregate symbol is printed.
    It reached the log first and that was not enough -- the log
    scrolls away, and the pane went on showing G_eff.

    The subtlety worth pinning is that the bench's rows come from ONE
    fold of the ledger while margin_equation folds it AGAIN. Setting
    the expansion on a row's reconstruction only shows up because a
    plain entry passes through folding by reference. If that ever
    stops being true, the pane silently stops updating while the log
    keeps working -- exactly the shape of the bug this guards."""
    from circuitinsight.gui.app import MainWindow

    w = MainWindow()
    type(w).settings_path = None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            w.open_session(FIX / "miller" / "tb_ota2s_stb.cin.json",
                           FIX / "miller" / "psf_stb", None)
            hit = []
            for cplx in (1, 2, 3):
                w.margins_sync("IPRB0", complexity=cplx)
                hit = [(r, e) for r, e in enumerate(w._ml_rows)
                       if getattr(e, "reconstruction", None)
                       and e.reconstruction.aggregate]
                if hit:
                    break
        if not hit:
            pytest.skip("no aggregate to expand in this fixture")
        row, e = hit[0]
        agg = e.reconstruction.aggregate
        assert any(agg["symbol"] in ln for ln in w._ml_eq_lines)

        out = {"port": "CL", "keep": ("gA", "gB"), "num": "N",
               "den": "D", "value": agg["value"], "seconds": 1.0,
               "matches": True, "ratio": 1.0, "aggregate": agg,
               "branches": [
                   {"gate": "gA", "conductance": 1e-6,
                    "text": "gds_A*(gds_C+gds_D)/(gm_A+gmbs_A)"},
                   {"gate": "gB", "conductance": 2e-6,
                    "text": "gds_B*gds_E/(gm_B+gmbs_B)"}]}
        w._on_ml_reading({"row": row, "what": "port", "out": out})

        joined = " | ".join(w._ml_eq_lines)
        assert "gds_A*(gds_C+gds_D)" in joined, "the pane did not update"
        assert agg["symbol"] not in joined, "the bare symbol survived"
        # and the row's tooltip carries it too
        assert w.ml_tbl.item(row, 1).toolTip()
        # the job left the Log forward; a reading that REWROTE the
        # equation has to hand the bench back, on the row it answered
        assert w.tabs.currentWidget() is w._ml_tab
        assert w.ml_tbl.currentRow() == row

        # ...but a reading whose branches could not all be named
        # changes nothing in the pane, so it stays where its answer is
        e.reconstruction.expansion = None
        w.tabs.setCurrentWidget(w.logview)
        half = dict(out, branches=[
            {"gate": "gA", "conductance": 1e-6, "text": None},
            out["branches"][1]])
        w._on_ml_reading({"row": row, "what": "port", "out": half})
        assert w.tabs.currentWidget() is w.logview
    finally:
        w.close()
