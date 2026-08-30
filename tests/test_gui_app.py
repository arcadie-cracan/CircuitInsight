"""Headless smoke test of the PySide6 desktop app (offscreen platform).

Skipped when PySide6 isn't installed. Exercises the window wiring — open a
fixture, combos populate, a synchronous solve updates the summary + canvas —
without a display and without the worker thread.
"""
import os
import time
import warnings
from pathlib import Path

import pytest

# On a clean install PySide6 imports but PySide6.QtWidgets can still fail to
# load its Qt DLLs (missing native runtime on CI runners -- a *broken-import*
# ImportError, which pytest.importorskip re-raises rather than skips). Catch it
# ourselves and skip the whole module so these GUI smoke tests skip, not error.
pytest.importorskip("PySide6")
try:
    import PySide6.QtWidgets  # noqa: F401
except Exception as exc:       # ImportError / native DLL-load failure
    pytest.skip(f"PySide6.QtWidgets not loadable: {exc}",
                allow_module_level=True)
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

FIX = Path(__file__).resolve().parent / "fixtures" / "spectre" / "ota5t"
MILLER = Path(__file__).resolve().parent / "fixtures" / "spectre" / "miller"


def _wait_bg(qapp, cond, timeout=20.0):
    """Process events until a background advisory lands (estimate, rank,
    certificate all run off the GUI thread now)."""
    import time as _t

    t0 = _t.monotonic()
    while _t.monotonic() - t0 < timeout:
        qapp.processEvents()
        if cond():
            return True
        _t.sleep(0.02)
    return False


def _have_symlib() -> bool:
    from circuitinsight.schematic.compose import BUNDLED_SYMLIB
    return bool(os.environ.get("CIN_SYMLIB"))         or (BUNDLED_SYMLIB / "terminals.csv").exists()


@pytest.fixture(scope="module")
def qapp():
    # QtWebEngine must be imported BEFORE the QApplication exists (it sets
    # AA_ShareOpenGLContexts); importing the feature-detected module first keeps
    # the ordering right on machines that have the addon, and is a no-op here.
    import circuitinsight.gui.exprweb  # noqa: F401
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


def test_progress_bar_is_time_driven_with_live_estimate(qapp, tmp_path):
    """The bar moves with the clock against a LIVE estimate: before any
    unit reports it advances on the pre-solve estimate alone, and when
    units arrive the estimate refines (elapsed/fraction blended with the
    prior by observed coverage) — the bar re-derives from the better
    number, backward if honesty requires."""
    import time as _t

    from circuitinsight.gui.app import MainWindow

    MainWindow.settings_path = str(tmp_path / "gui.ini")
    try:
        win = MainWindow()
    finally:
        MainWindow.settings_path = None
    try:
        win._est_s = 10.0
        win._run_est = 10.0          # what _launch records
        win._live_est = None
        win._t0 = _t.monotonic() - 2.0          # 2 s elapsed
        win._phase = "evaluating"
        win._phase_units = (None, None)
        win._refresh_progress()
        assert win.progress.maximum() == 1000   # time-driven, not units
        v_prior = win.progress.value()
        assert 150 <= v_prior <= 260            # ~2 s of ~10 s

        # observation disagrees: 10% done at 2 s projects 20 s total
        win._phase_units = (100, 1000)
        win._refresh_progress()
        assert win._live_est > 10.0             # estimate refined upward
        assert win.progress.value() < v_prior   # bar re-derived, honestly
        assert "~" in win.progress.format()
    finally:
        win._t0 = None
        win.close()


def test_rank_keeps_the_ticks_and_a_running_solve_keeps_its_estimate(
        qapp, tmp_path):
    """Field report: mid-solve the keep ticks vanished and a 772 s solve
    read '~0s'. Two causes, both here — re-ranking refilled the table
    with no checked set (dropping the user's selection), and the
    estimate then re-costed the now-EMPTY keep set over the running
    solve's own estimate."""
    from PySide6.QtCore import Qt
    from circuitinsight.gui.app import MainWindow

    MainWindow.settings_path = str(tmp_path / "gui.ini")
    try:
        win = MainWindow()
    finally:
        MainWindow.settings_path = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        win.open_session(str(FIX / "tb_ota5t.cin.json"), str(FIX / "psf"))
        win.in_combo.setCurrentText("VIND")
        win.out_combo.setCurrentText("vout")
        win._rank()
        win.keep_tbl.item(0, 0).setCheckState(Qt.Checked)
        picked = win.checked_keep()
        assert picked
        win._rank()                              # re-rank must not clear
    try:
        assert win.checked_keep() == picked

        # while a solve is in flight the estimate is frozen
        import time as _t
        win._est_s = 300.0
        win._run_est = 300.0          # what _launch records
        win._t0 = _t.monotonic()
        for i in range(win.keep_tbl.rowCount()):  # empty the table
            win.keep_tbl.item(i, 0).setCheckState(Qt.Unchecked)
        win._update_estimate()
        assert win._est_s == 300.0               # the running solve's own
        win._t0 = None
        win._update_estimate()                   # idle: re-costs freely
        assert _wait_bg(qapp, lambda: win._est_s != 300.0),             "the async re-cost must land"
    finally:
        win._t0 = None
        win.close()


def test_backend_selector_autoselects_and_overrides(qapp, tmp_path):
    """The crossover made visible: 'auto' names what the CURRENT keep
    set resolves to (one more symbol can flip the dense grid to the
    sparse path and run faster), and an explicit choice forces the
    engine global that BOTH the estimator and the solve consult."""
    from PySide6.QtCore import Qt
    from circuitinsight.engine import interp
    from circuitinsight.gui.app import MainWindow

    MainWindow.settings_path = str(tmp_path / "gui.ini")
    try:
        win = MainWindow()
    finally:
        MainWindow.settings_path = None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            win.open_session(str(FIX / "tb_ota5t.cin.json"), str(FIX / "psf"))
            win.in_combo.setCurrentText("VIND")
            win.out_combo.setCurrentText("vout")
            win._rank()
            win.keep_tbl.item(0, 0).setCheckState(Qt.Checked)
        assert interp.PROBE_BACKEND is None          # auto on open
        assert win.backend_combo.currentIndex() == 0
        assert win.backend_combo.itemText(0).startswith("auto")
        # the resolved name arrives with the async estimate
        assert _wait_bg(qapp,
                        lambda: "→" in win.backend_combo.itemText(0)),             "auto must name its resolved backend"

        win.backend_combo.setCurrentIndex(2)         # force bot
        # read THROUGH the session: in engine-process mode the global
        # lives in the child, where a GUI-side module read is blind
        assert win.controller.probe_backend() == "bot"
        assert _wait_bg(qapp, lambda: "bot" in win.estimate_lbl.text()),             "the async estimate must name the forced backend"
        win.backend_combo.setCurrentIndex(0)         # back to auto
        assert win.controller.probe_backend() is None
    finally:
        interp.PROBE_BACKEND = None
        win.close()


def test_cancel_is_a_base_exception():
    """The worker's cancel must not subclass Exception: the engine's
    backend fallbacks (except Exception: run the slower path) swallowed
    an Exception-cancel and silently RE-RAN the abandoned solve. The
    engine-side propagation half lives in test_zpbatch."""
    # _Cancelled lives in gui.jobs; app.py merely re-exported it,
    # and naming its real home keeps that re-export from being the
    # only reason the import exists
    from circuitinsight.gui.jobs import _Cancelled

    assert issubclass(_Cancelled, BaseException)
    assert not issubclass(_Cancelled, Exception)


def test_launcher_imports_nothing_heavy():
    """The loading banner exists because the imports are the slow part:
    gui.launch must import NOTHING at module level (not even Qt) so the
    console script reaches main() and shows the banner before paying
    for matplotlib, sympy and the session layer. Checked in a clean
    subprocess — this test process has everything imported already."""
    import subprocess
    import sys

    src = str(Path(__file__).resolve().parents[1] / "src")
    code = ("import sys; sys.path.insert(0, %r); "
            "import circuitinsight.gui.launch; "
            "heavy = [m for m in ('PySide6', 'matplotlib', 'sympy', "
            "'numpy', 'circuitinsight.session') if m in sys.modules]; "
            "assert not heavy, heavy; print('lazy')" % src)
    out = subprocess.run([sys.executable, "-c", code],
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert "lazy" in out.stdout


def test_splash_banner_builds(qapp):
    """The banner itself: paints, carries the red rule, names the
    VERSION (the first question any bug report needs), and stages
    narrate without error."""
    import circuitinsight
    from circuitinsight.gui import launch

    ver = launch._version()
    assert ver.startswith("v") and any(c.isdigit() for c in ver)
    assert circuitinsight.__version__ in ver or ver[1:].count(".") >= 1

    sp = launch._splash()
    assert not sp.pixmap().isNull()
    launch._stage(qapp, sp, "loading the analysis engine …")
    sp.close()


def test_expr_web_assets_ship():
    """The KaTeX shell and its assets live inside the package (hatchling ships
    the package dir wholesale), or the web view would come up blank after a pip
    install. Runs regardless of whether QtWebEngine itself is installed."""
    from circuitinsight.gui import exprweb
    assets = Path(exprweb.__file__).resolve().parent / "assets"
    assert (assets / "expr.html").is_file()
    assert (assets / "katex" / "katex.min.js").is_file()
    assert (assets / "katex" / "katex.min.css").is_file()
    assert list((assets / "katex" / "fonts").glob("*.woff2"))
    assert (assets / "katex" / "LICENSE").is_file()       # MIT, vendored


def test_open_populates_and_solves(qapp):
    from circuitinsight.gui.app import MainWindow

    win = MainWindow()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        win.open_session(str(FIX / "tb_ota5t.cin.json"), str(FIX / "psf"))

    inputs = [win.in_combo.itemText(i) for i in range(win.in_combo.count())]
    outputs = [win.out_combo.itemText(i) for i in range(win.out_combo.count())]
    assert "VIND" in inputs and "vout" in outputs
    assert len(win.devices.leaf_items()) > 5
    assert win.solve_btn.isEnabled()

    win.in_combo.setCurrentText("VIND")
    win.out_combo.setCurrentText("vout")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        r = win.solve_sync()

    assert r is not None
    assert r.dc_gain_db == pytest.approx(46.13, abs=0.1)
    assert "DC gain" in win.summary.toPlainText()
    assert len(win.canvas.figure.axes) == 2
    win.close()


def test_keepset_rank_estimate_and_simplify(qapp):
    from PySide6.QtCore import Qt

    from circuitinsight.gui.app import MainWindow

    win = MainWindow()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        win.open_session(str(FIX / "tb_ota5t.cin.json"), str(FIX / "psf"))
    win.in_combo.setCurrentText("VIND")
    win.out_combo.setCurrentText("vout")

    win._rank()
    assert win.keep_tbl.rowCount() > 0
    # a re-rank REORDERS; it preserves the selection (auto-setup's keep
    # set survives it) -- see test_rank_keeps_the_ticks_...
    before = set(win.checked_keep())
    top = win.keep_tbl.item(0, 0)
    top.setCheckState(Qt.Checked)                    # triggers estimate update
    assert set(win.checked_keep()) == before | {top.text()}
    assert win.estimate_lbl.text().startswith("estimate:") \
        and "—" not in win.estimate_lbl.text().split("estimate:")[1][:3]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        r = win.simplify_sync()
    assert r.simplified and r.mag_err_db is not None
    assert "pruned within" in win.summary.toPlainText()
    win.close()


def test_build_window_preloaded(qapp):
    """The --cin/--psf launch path (build_window) opens preloaded."""
    from circuitinsight.gui.app import build_window

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        win = build_window(str(FIX / "tb_ota5t.cin.json"), str(FIX / "psf"))
    assert win.controller is not None
    assert win.in_combo.count() > 0 and win.solve_btn.isEnabled()
    win.close()


def test_matches_and_export(qapp, tmp_path):
    from circuitinsight.gui.app import MainWindow

    win = MainWindow()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        win.open_session(str(FIX / "tb_ota5t.cin.json"), str(FIX / "psf"))

    win.suggest_matches()                            # (MN0,MN1) and (MP0,MP1)
    assert len(win._match_groups) == 2
    # the groups live on the tree as 🔗n decorations, tinted per group
    from circuitinsight.gui.devtree import LINK
    decorated = {win.devices.device_name(it)
                 for it in win.devices.leaf_items()
                 if LINK in it.text(0)}
    assert decorated == {n for g in win._match_groups for n in g}
    assert any(it.text(0).endswith(f"{LINK}2")
               for it in win.devices.leaf_items())

    win.in_combo.setCurrentText("VIND")
    win.out_combo.setCurrentText("vout")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        win.solve_sync()

    md = win._write_report(tmp_path / "rep.md")
    assert md.exists() and md.with_suffix(".png").exists()
    assert "# CircuitInsight" in md.read_text(encoding="utf-8")
    win.close()


def test_cadence_theme_applies(qapp):
    """The theme must apply cleanly and actually recolour the chrome — the app
    should sit beside Virtuoso's windows, not glow white next to them."""
    from circuitinsight.gui import theme

    theme.apply(qapp)
    assert qapp.palette().window().color().name() == theme.BG
    assert qapp.styleSheet()                      # widget rules installed

    from matplotlib.figure import Figure
    fig = Figure()
    fig.add_subplot(1, 1, 1)
    theme.style_figure(fig)
    # figure surround matches the chrome; the plot area stays a white data surface
    assert fig.patch.get_facecolor()[:3] == pytest.approx((0.851, 0.851, 0.851),
                                                          abs=0.01)
    assert fig.axes[0].get_facecolor()[:3] == pytest.approx((1.0, 1.0, 1.0))


def test_symbolic_by_default_on_open(qapp):
    """Opening a session must pre-select a keep set, so the FIRST solve is
    symbolic. The old default (empty keep table -> keep=[]) made a symbolic
    analyzer's first result show no symbol but `s` -- the top user complaint."""
    import warnings

    from circuitinsight.gui.app import build_window

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        win = build_window(str(FIX / "tb_ota5t.cin.json"), str(FIX / "psf"))

    assert win.out_combo.currentText() == "vout"       # not the first net (vbn)
    assert win.checked_keep(), "keep table opened empty -> numeric by default"
    assert len(win._match_groups) == 2                 # matched pairs auto-applied

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = win.solve_sync()
    # symbols actually survive into H(s)
    syms = {str(x) for x in r.tf.expr.free_symbols}
    assert syms - {"s"}, f"first solve was numeric: only {syms}"
    assert r.dc_gain_db == pytest.approx(46.13, abs=0.1)
    win.close()


def test_full_names_toggle_rerenders_expression(qapp):
    """The Expression tab's 'Full names' checkbox switches leaf device names
    (g_{m,MN1}) for the full instance hierarchy (g_{m,I0.MN1}) without a
    re-solve."""
    from circuitinsight.gui import view
    from circuitinsight.gui.app import build_window

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        win = build_window(str(MILLER / "tb_ota2s.cin.json"), str(MILLER / "psf"))
        win.solve_sync()

    assert not win.fullnames_chk.isChecked()               # base (leaf) by default
    base = view._expr_lines(win.result, base=True)
    full = view._expr_lines(win.result, base=False)
    assert base != full                                    # toggle changes the text
    assert any("I0." in tex for _, tex in full)            # full carries hierarchy
    assert not any("I0." in tex for _, tex in base)        # base drops it
    win.fullnames_chk.setChecked(True)                     # drives _render_expr
    win.close()


def test_expression_tab_scrolls_on_wheel(qapp):
    """A matplotlib canvas swallows wheel events, so the Expression tab (a canvas
    in a QScrollArea) wouldn't scroll. The event filter must forward the wheel to
    the scrollbar -- and only for that canvas."""
    from PySide6.QtCore import QPoint, QPointF, Qt
    from PySide6.QtGui import QWheelEvent

    from circuitinsight.gui.app import MainWindow

    win = MainWindow()
    if win.expr_canvas is None:
        # QtWebEngine present -> the Expression tab is the KaTeX web view, which
        # scrolls natively; there is no canvas/scrollarea to forward wheels to.
        # (On CI the addon loads, so this path is the common one.)
        win.close()
        pytest.skip("Expression tab uses the QtWebEngine view, not the mpl canvas")
    sb = win.expr_scroll.verticalScrollBar()
    sb.setRange(0, 480)
    sb.setValue(200)

    def wheel(dy):
        return QWheelEvent(QPointF(10, 10), QPointF(10, 10), QPoint(0, 0),
                           QPoint(0, dy), Qt.NoButton, Qt.NoModifier,
                           Qt.NoScrollPhase, False)

    assert win.eventFilter(win.expr_canvas, wheel(-120)) is True
    assert sb.value() == 320                         # scrolled down
    assert win.eventFilter(win.expr_canvas, wheel(+120)) is True
    assert sb.value() == 200                         # and back up
    # a wheel on another widget is left alone
    assert win.eventFilter(win.summary, wheel(-120)) is False
    win.close()


def test_status_bar_names_the_backend_that_ran(qapp):
    """S-D telemetry surfaced: after a solve the status bar says which
    backend actually ran, so the auto-selector is visible."""
    from circuitinsight.gui.app import MainWindow

    win = MainWindow()

    class _Session:                  # the note reads THROUGH the session
        tl = None                    # (an engine process has its own
                                     # interp module -- a GUI-side
                                     # global write never lands there)
        def last_solve(self):
            return self.tl

    win.controller = _Session()
    win.controller.tl = {"backend": "bot", "wall_s": 12.5,
                         "fell_back": False}
    note = win._backend_note()
    assert "bot" in note and "12.5" in note
    win.controller.tl = {"backend": "qq", "wall_s": 1.0, "fell_back": True}
    assert "fell back" in win._backend_note()
    win.controller.tl = None
    assert win._backend_note() == ""


def test_split_advice_button_renders_a_verdict(qapp):
    """The split advisory reaches the GUI: pressing the button fills the
    label with the one-line verdict (and never raises on a circuit where
    tearing does not pay)."""
    from circuitinsight.gui.app import MainWindow

    win = MainWindow()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        win.open_session(str(FIX / "tb_ota5t.cin.json"), str(FIX / "psf"))
        win._split_advice()
    text = win.split_lbl.text()
    assert text.startswith("split: ")
    assert "tear" in text or "cut" in text


def test_solve_form_selector_dispatches_and_shows_budget(qapp, tmp_path):
    """Phase-1 UX: ONE Solve button; the form selector states the contract.
    The error-budget spins exist only for the budgeted forms — and only in
    the Transfer mode — and the menu entries keep their shortcuts by
    selecting the form they mean."""
    from circuitinsight.gui.app import MainWindow

    MainWindow.settings_path = str(tmp_path / "gui.ini")   # not the registry
    try:
        win = MainWindow()
    finally:
        MainWindow.settings_path = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        win.open_session(str(FIX / "tb_ota5t.cin.json"), str(FIX / "psf"))
    try:
        win.mode_combo.setCurrentText("Transfer")
        assert not hasattr(win, "simplify_btn")     # the old trio is gone
        assert not hasattr(win, "reduce_btn")
        assert win.form_combo.currentText() == "Exact"
        assert not win._mag_act.isVisible()

        win.form_combo.setCurrentText("Simplified · full order")
        assert win._mag_act.isVisible() and win._phase_act.isVisible()
        win.form_combo.setCurrentText("Exact")
        assert not win._mag_act.isVisible()

        # the menu route selects the form it means
        win.controller = None                       # dispatch stops at launch
        win.simplify()
        assert win.form_combo.currentText() == "Simplified · full order"
        win.reduce()
        assert win.form_combo.currentText() == "Simplified · lowest order"

        # the selector shows wherever a response is solved (Transfer,
        # Impedance, Loop gain) and hides elsewhere
        win.mode_combo.setCurrentText("Loop gain")
        assert win._form_act.isVisible()
        win.mode_combo.setCurrentText("Modes")
        assert not win._form_act.isVisible()
        assert not win._mag_act.isVisible()
        win.mode_combo.setCurrentText("Transfer")
        assert win._form_act.isVisible()
    finally:
        win.close()


def test_summary_leads_with_the_template(qapp):
    """U-D: the designer numbers — A0, GBW, per-root formulas — head the
    Summary, coming from template_form on the result's own tf."""
    from circuitinsight.gui.app import MainWindow

    win = MainWindow()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        win.open_session(str(FIX / "tb_ota5t.cin.json"), str(FIX / "psf"))
        win.in_combo.setCurrentText("VIND")
        win.out_combo.setCurrentText("vout")
        r = win.solve_sync()
    try:
        assert r.template_text                      # attached by solve_sync
        assert "A0" in r.template_text
        assert "pole" in r.template_text
        txt = win.summary.toPlainText()
        # the template block precedes the exact record
        assert txt.index("A0") < txt.index("DC gain")
    finally:
        win.close()


def test_device_double_click_opens_op_inspector(qapp):
    """U-F: the table's gm/gds columns are a summary; double-click gives the
    device's FULL OP record, formatted with units."""
    from circuitinsight.gui.app import MainWindow

    win = MainWindow()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        win.open_session(str(FIX / "tb_ota5t.cin.json"), str(FIX / "psf"))
    try:
        from PySide6.QtWidgets import QTableWidget

        item = next(it for it in win.devices.leaf_items()
                    if win.devices.info(it)["type"] == "mosfet")
        win._show_device_op(item)
        dlg = win._op_dialog
        tbl = dlg.findChild(QTableWidget)
        assert tbl.rowCount() > 5                   # the full record, not 2 cols
        keys = {tbl.item(i, 0).text() for i in range(tbl.rowCount())}
        assert "gm" in keys and "region" in keys
        dlg.close()
    finally:
        win.close()


def test_reduce_bench_scan_apply_revert(qapp):
    """gui-ux-plan.md U-C: the Reduce-circuit bench walks the checklist —
    scan prices each bias node, ticking prices the SET and previews the
    exact follow-on, Apply rewrites the working circuit and banners the
    measured cost, and every later result carries its circuit state."""
    from circuitinsight.gui.app import MainWindow

    win = MainWindow()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        win.open_session(str(MILLER / "tb_ota2s.cin.json"),
                         str(MILLER / "psf"))
        win.in_combo.setCurrentText("VIND")
        win.out_combo.setCurrentText("vout")
    try:
        win.mode_combo.setCurrentText("Reduce circuit")
        assert win.reduce_col.isVisibleTo(win)

        # no local budget knob: the scan gates under the toolbar
        # contract (strategy + budgets + band), like every approximation
        assert not hasattr(win, "acg_budget")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            rep = win.controller.scan_ac_grounds(
                "VIND", "vout", **win._contract_kw())
            win._on_acg_scan_done(rep)
        assert rep.criterion_label            # the contract gated it
        assert all(c.score is not None for c in rep.candidates)
        assert win.acg_tbl.rowCount() > 0
        assert win.checked_acg_nodes() == list(rep.recommended)
        assert win.acg_apply.isEnabled()
        # the one grammar: x budget first, native units in parentheses
        assert "× budget" in win.acg_joint_lbl.text()
        assert "together" in win.acg_joint_lbl.text()
        pv = win.acg_preview.toPlainText()
        assert "removed (exact)" in pv and "primitives" in pv

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            summ = win.controller.apply_reduction(
                win.checked_acg_nodes(), inp="VIND", out="vout")
            win._on_reduction_applied(summ)
        # the headline carries the applied state and its cost, once
        assert win.red_banner.text().startswith("applied: 2 net(s)")
        assert "measured end to end" in win.red_banner.text()
        assert win.acg_revert.isEnabled()
        assert "circuit reduced" in win.logview.toPlainText()
        # the rewrite invalidates the old ranking (lumped/removed
        # symbols made a stale-tick solve fail), so the table re-ranks
        # the reduced circuit by itself: the reduced symbol set, with
        # the lumped equivalents in it and no vanished symbol
        assert "ranking the reduced" in win.estimate_lbl.text()
        assert _wait_bg(qapp, lambda: win.keep_tbl.rowCount() > 0,
                        timeout=60)
        names = {win.keep_tbl.item(i, 0).text()
                 for i in range(win.keep_tbl.rowCount())}
        eq_names = {e["name"] for e in summ.get("equivalents", [])}
        assert eq_names & names, "lumped equivalents are rankable symbols"
        absorbed = {m.replace(".", "_") for e in summ.get("equivalents", [])
                    for m in e["members"]}
        assert not (absorbed & names), "absorbed members are gone"
        # the Nets tree tells the reduction truth: grounded nets wear ⏚
        from circuitinsight.gui.devtree import EARTH
        for node in summ["nodes"]:
            it = win.nets_tree.item_for(node)
            assert it is not None and EARTH in it.text(0)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            win.mode_combo.setCurrentText("Transfer")
            r = win.solve_sync()
        assert r.circuit_state == "reduced"
        assert "REDUCED" in win.summary.toPlainText()
        assert "[reduced]" in win.history.item(
            win.history.count() - 1).text()

        win.revert_reduction()
        assert win.red_banner.text().startswith("applied: none")
        assert not win.acg_revert.isEnabled()
        assert not any(EARTH in win.nets_tree.item_for(n).text(0)
                       for n in summ["nodes"])
    finally:
        win.close()


def test_tool_dropdown_is_the_one_navigation_axis(qapp, tmp_path):
    """U-A revised: the Tool dropdown selects the analysis; the hidden
    mode combo is the state object it fronts, and the two never disagree
    — whichever side is driven. It shows ONLY tools the loaded run can
    ground in simulator data: ota5t carries no stb results, so the
    loop-analysis family is not offered."""
    from circuitinsight.gui.app import MainWindow

    MainWindow.settings_path = str(tmp_path / "gui.ini")
    try:
        win = MainWindow()
    finally:
        MainWindow.settings_path = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        win.open_session(str(FIX / "tb_ota5t.cin.json"), str(FIX / "psf"))
    try:
        names = [win.tool_combo.itemText(i)
                 for i in range(win.tool_combo.count())]
        assert names == ["Transfer", "Impedance", "Reduce circuit"]

        # tool drives mode
        win.tool_combo.setCurrentText("Impedance")
        assert win.mode_combo.currentText() == "Impedance"
        # mode drives tool (the programmatic path every test uses)
        win.mode_combo.setCurrentText("Transfer")
        assert win.tool_combo.currentText() == "Transfer"
        # a programmatic mode outside the offering is reinstated, not lied
        # about — the dropdown always names the active analysis
        win.mode_combo.setCurrentText("Loop gain")
        assert win.tool_combo.currentText() == "Loop gain"
    finally:
        win.close()


def test_tool_dropdown_offers_the_loop_family_only_with_stb_truth(
        qapp, tmp_path):
    """The loop family is gated on SIMULATOR GROUND TRUTH: stb results
    in the run (miller stb bench: Loop gain/Compensate/GFT but not the
    two-iprobe Modes; nmc3 with three iprobes: everything), or the
    user's explicit declaration that the AC data is a return-ratio
    capture (plain miller: absent, declared: present, withdrawn:
    absent). A vsource alone opens nothing."""
    from circuitinsight.gui.app import MainWindow

    spectre = FIX.parent
    MainWindow.settings_path = str(tmp_path / "gui.ini")
    try:
        win = MainWindow()
    finally:
        MainWindow.settings_path = None

    def names():
        return [win.tool_combo.itemText(i)
                for i in range(win.tool_combo.count())]

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            win.open_session(
                str(spectre / "miller" / "tb_ota2s_stb.cin.json"),
                str(spectre / "miller" / "psf_stb"))
        got = names()
        assert "Loop gain" in got and "GFT" in got \
            and "Compensate" in got
        assert "Modes" not in got

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            win.open_session(str(spectre / "nmc3" / "tb_nmc3.cin.json"),
                             str(spectre / "nmc3" / "psf"))
        assert names() == ["Transfer", "Loop gain", "Compensate", "Modes",
                           "GFT", "Impedance", "Reduce circuit"]

        # plain miller: AC data but no stb — probes exist, truth doesn't
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            win.open_session(str(spectre / "miller" / "tb_ota2s.cin.json"),
                             str(spectre / "miller" / "psf"))
        assert "Loop gain" not in names()
        assert win.controller.probes, "vsources alone must not gate it"

        # the explicit declaration is the one other key that opens them
        win.controller.declare_ac_loop_gain("vout")
        win._refresh_tools()
        got = names()
        assert "Loop gain" in got and "Modes" not in got
        win.controller.declare_ac_loop_gain(None)
        win._refresh_tools()
        assert "Loop gain" not in names()
    finally:
        win.close()


def test_loop_bench_first_light_plots_the_stb_truth(qapp, tmp_path):
    """An stb run opens into Loop gain -- and the plot must not be
    empty: the stb loopGain is already in the run, so it draws as the
    bench's first light (dashed truth, margins annotated), no model
    solve needed. The declared-AC curve serves the same way."""
    from circuitinsight.gui.app import MainWindow

    spectre = FIX.parent
    MainWindow.settings_path = str(tmp_path / "gui.ini")
    try:
        win = MainWindow()
    finally:
        MainWindow.settings_path = None

    def ref_labels():
        return [ln.get_label()
                for ax in win.canvas.figure.axes
                for ln in ax.get_lines()]

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            win.open_session(
                str(spectre / "miller" / "tb_ota2s_stb.cin.json"),
                str(spectre / "miller" / "psf_stb"))
        assert win.mode_combo.currentText() == "Loop gain"
        assert "Spectre stb loopGain" in ref_labels()
        fr, h, label, margins = win.controller.loop_reference()
        assert label == "Spectre stb loopGain"
        assert margins.pm_deg is not None
        # the reference BORROWED the plot from the transfer first
        # light; re-entering Transfer gives it back
        if win.result is not None:
            win.mode_combo.setCurrentText("Transfer")
            assert "Spectre stb loopGain" not in ref_labels()
            win.mode_combo.setCurrentText("Loop gain")
            assert "Spectre stb loopGain" in ref_labels()

        # declared-AC run: the declared curve is the first light
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            win.open_session(str(spectre / "miller" / "tb_ota2s.cin.json"),
                             str(spectre / "miller" / "psf"))
        win.result = None
        win.controller.declare_ac_loop_gain("vout")
        win._refresh_tools()
        win.mode_combo.setCurrentText("Loop gain")
        assert any("AC declared as T" in l for l in ref_labels())
    finally:
        win.close()


def test_declared_return_ratio_is_the_loop_reference():
    """Session side of the declaration: with no stb in the run, the
    declared net's AC trace becomes the loop-gain reference — labeled as
    a declaration, margins computed from the trace itself."""
    from circuitinsight.session import SessionController

    spectre = FIX.parent
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        c = SessionController.open(
            spectre / "miller" / "tb_ota2s.cin.json",
            spectre / "miller" / "psf")
    assert not c.has_stb
    fr, lg, obj, label = c._stb_reference(None)
    assert fr is None, "no truth, no reference"

    c.declare_ac_loop_gain("vout")
    fr, lg, obj, label = c._stb_reference(None)
    assert fr is not None and len(fr) == len(lg)
    assert "declared" in label and "vout" in label
    assert hasattr(obj, "phase_margin_deg")


def test_bench_tabs_show_only_their_own_views(qapp, tmp_path):
    """U-A: Summary/Expression/Error are views of any result and stay;
    What-if/Compensation/GFT/Reduce exist only in their bench."""
    from circuitinsight.gui.app import MainWindow

    MainWindow.settings_path = str(tmp_path / "gui.ini")
    try:
        win = MainWindow()
    finally:
        MainWindow.settings_path = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        win.open_session(str(FIX / "tb_ota5t.cin.json"), str(FIX / "psf"))
    try:
        def visible(w):
            return win.tabs.isTabVisible(win.tabs.indexOf(w))

        win.mode_combo.setCurrentText("Transfer")
        assert visible(win._whatif_tab)
        assert not visible(win._comp_tab) and not visible(win._gft_tab)
        assert not win.reduce_col.isVisibleTo(win)

        win.mode_combo.setCurrentText("Compensate")
        assert visible(win._comp_tab) and not visible(win._whatif_tab)
        assert win.tabs.currentWidget() is win._comp_tab

        win.mode_combo.setCurrentText("Reduce circuit")
        assert win.reduce_col.isVisibleTo(win)
        assert not visible(win._comp_tab)
        # Reduce is a state tool: Summary/Expression (views of a solve)
        # step aside there; Error and Log stay
        assert not win.tabs.isTabVisible(0)      # Summary
        assert visible(win.logview)
        win.mode_combo.setCurrentText("Transfer")
        assert win.tabs.isTabVisible(0)          # and come back
    finally:
        win.close()


def test_left_groups_collapse_and_history_starts_collapsed(qapp, tmp_path):
    """U-G: the three left groups collapse from their title checkbox, and
    History starts collapsed — it earns its space once there is history."""
    from circuitinsight.gui.app import MainWindow

    MainWindow.settings_path = str(tmp_path / "gui.ini")
    try:
        win = MainWindow()
    finally:
        MainWindow.settings_path = None
    try:
        win.show()                               # visibility is only real
        qapp.processEvents()                     # on a shown window
        from PySide6.QtWidgets import QGroupBox

        boxes = {b.title(): b for b in win.left_split.findChildren(QGroupBox)
                 if b.title()}
        hist = next(b for t, b in boxes.items() if "History" in t)
        keep = next(b for t, b in boxes.items() if "Keep" in t)
        assert hist.isCheckable() and not hist.isChecked()
        assert keep.isCheckable() and keep.isChecked()
        assert not win.history.isVisible()       # body hidden while collapsed
        assert win.keep_tbl.isVisible()
        hist.setChecked(True)
        qapp.processEvents()
        assert win.history.isVisible()
        keep.setChecked(False)
        qapp.processEvents()
        assert not win.keep_tbl.isVisible()
        keep.setChecked(True)
    finally:
        win.close()


def test_first_light_solves_on_open_without_touching_history(qapp):
    """U-E: the plot pane is never empty — opening a session runs the
    numeric solve immediately (sub-second since the s-sweep). It is
    scaffolding, not a user action, so History stays empty and the first
    user solve is still the first history entry."""
    from circuitinsight.gui.app import build_window

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        win = build_window(str(FIX / "tb_ota5t.cin.json"), str(FIX / "psf"))
    try:
        assert win.result is not None                # first light shown
        assert win.result.keep == []                 # numeric, by design
        assert win.result.template_text              # designer numbers too
        assert len(win.canvas.figure.axes) == 2      # Bode drawn
        assert win.history.count() == 0              # scaffolding, not history
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            win.solve_sync()
        assert win.history.count() == 1              # the USER's first solve
    finally:
        win.close()


def test_breadcrumb_tracks_the_workflow(qapp, tmp_path):
    """U-E: Open → Match → Choose symbols → Solve, first incomplete step
    bold, done steps ticked. Judged from what exists, not bookkeeping."""
    from circuitinsight.gui.app import MainWindow, build_window

    MainWindow.settings_path = str(tmp_path / "gui.ini")
    try:
        bare = MainWindow()
    finally:
        MainWindow.settings_path = None
    try:
        txt = bare.crumb.text()
        assert "1 Open" in txt and "✓" not in txt    # nothing done yet
        assert 'font-weight:bold' in txt.split("2 Match")[0]
    finally:
        bare.close()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        win = build_window(str(FIX / "tb_ota5t.cin.json"), str(FIX / "psf"))
    try:
        txt = win.crumb.text()
        # auto-setup opened, matched, and picked symbols: 1-3 ticked,
        # Solve is the one bold step left (first light is not a user solve)
        assert txt.count("✓") == 3
        assert 'font-weight:bold' in txt.split("4 Solve")[0].rsplit(
            "Choose symbols", 1)[1]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            win.solve_sync()
        win._update_crumb()
        assert win.crumb.text().count("✓") == 4      # journey complete
    finally:
        win.close()


def test_auto_matches_surface_their_measured_cost(qapp):
    """The 52.22 dB screenshot, closed end to end: first light runs BEFORE
    auto-matches so the honest model shows once, and if the applied
    matches move the model, the strip says so with the MEASURED dB and
    the worst parameter overwrite named."""
    from circuitinsight.gui.app import build_window

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        win = build_window(str(MILLER / "tb_ota2s.cin.json"),
                           str(MILLER / "psf"))
    try:
        # miller's suggested pairs are true pairs: no conflict strip
        assert "matches" not in win.msg_strip.text()
        # a deliberately bad group must be priced and announced
        win._match_groups = [("I0.MN3", "I0.MN0")]     # load vs input pair
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            win._apply_matches()
        txt = win.msg_strip.text()
        assert "differs" in txt and "the model follows" in txt
    finally:
        win.close()


def test_match_value_policy_controls(qapp, monkeypatch):
    """The values combo drives the session policy, and double-clicking a
    match group picks its representative (snapping the combo back to
    representative)."""
    from circuitinsight.gui.app import build_window

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        win = build_window(str(FIX / "tb_ota5t.cin.json"), str(FIX / "psf"))
    try:
        # weighted is the DEFAULT: it found the load-bearing member on
        # both benches (0.039 / 0.347 dB vs-sim)
        assert win._policy_label() == "weighted"
        assert win.controller.match_value_policy == "weighted"

        # the policy is a radio group in the Devices menu now
        win._policy_actions["weighted"].trigger()
        assert win.controller.match_value_policy == "weighted"
        win._policy_actions["average"].trigger()
        assert win.controller.match_value_policy == "mean"

        group = win._match_groups[0]
        from PySide6.QtWidgets import QInputDialog
        monkeypatch.setattr(QInputDialog, "getItem",
                            staticmethod(lambda *a, **k: (group[1], True)))
        win._pick_representative(0)
        assert win.controller.match_representative(group) == group[1]
        assert win.controller.match_value_policy == "representative"
        assert win._policy_label() == "representative"
    finally:
        win.close()


def test_advisors_surface_in_the_gui(qapp):
    """S-E tail, GUI side: the removal scan lands in the Reduce bench's
    tick table; Attribute poles (Analysis menu) appends the verified
    attribution to the Summary."""
    from circuitinsight.gui.app import build_window

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        win = build_window(str(MILLER / "tb_ota2s.cin.json"),
                           str(MILLER / "psf"))
        win.in_combo.setCurrentText("VIND")
        win.out_combo.setCurrentText("vout")
    try:
        rep = win.controller.scan_removals("VIND", "vout", budget_db=0.1)
        win._on_removal_scan_done(rep)
        assert win.rm_tbl.rowCount() == len(rep.candidates)
        assert "removal scan" in win.statusBar().currentMessage()

        atts = win.controller.pole_attribution("VIND", n_poles=2)
        before = win.summary.toPlainText()
        win._on_attribution_done(atts)
        txt = win.summary.toPlainText()
        assert txt.startswith(before)
        assert "pole attribution" in txt and "set by" in txt

        stories = win.controller.explain_numerals("VIND", "vout")
        # the handler now receives (aligned result, stories) so the
        # display can be refreshed when the configuration drifted
        win._on_explain_done((win.result, stories))
        txt2 = win.summary.toPlainText()
        assert txt2.startswith(txt)
        assert "the numbers, explained" in txt2 and "den s^0" in txt2
        assert "ratio attribution" in txt2       # the displayed numerals

        # the Expression payload carries the numeral hover data
        from circuitinsight.gui import view as _view
        payload = _view.expr_katex(win.result,
                                   numerals=_view.numeral_tips(stories),
                                   numhint="run Explain")
        assert payload["numhint"] == "run Explain"
        assert any(k.startswith("den:") for k in payload["numerals"])

        # the deep pass: per-numeral stories land in the Summary and as
        # finest-granularity hover tips (part:k:monomial)
        deep = win.controller.explain_per_numeral("VIND", "vout",
                                                  keep=["I0.Cc"])
        assert any(st.mono != "1" for st in deep)
        win._on_explain_deep_done(deep)
        assert "per-numeral attribution" in win.summary.toPlainText()
        tips = _view.numeral_tips(stories, deep=deep)
        assert any(k.count(":") == 2 for k in tips)
    finally:
        win.close()


def test_help_menu_opens_the_shipped_user_guide(qapp, tmp_path):
    """The user guide ships inside the package (gui/assets, like the KaTeX
    shell) and opens from Help -> User guide (F1). The content check is
    structural: every bench must have a section, because a guide that
    silently omits a feature teaches users the feature does not exist."""
    from circuitinsight.gui import exprweb
    from circuitinsight.gui.app import MainWindow

    manual = (Path(exprweb.__file__).resolve().parent / "assets"
              / "manual.html")
    assert manual.is_file()
    html = manual.read_text(encoding="utf-8")
    for bench in ("Transfer", "Loop gain", "Compensate", "Modes", "GFT",
                  "Impedance", "Reduce circuit"):
        assert bench in html, f"the guide never mentions the {bench} bench"
    for feature in ("First light", "weighted", "Attribute poles",
                    "cin_init.il", "keep set", "lowest order"):
        assert feature in html

    MainWindow.settings_path = str(tmp_path / "gui.ini")
    try:
        win = MainWindow()
    finally:
        MainWindow.settings_path = None
    try:
        assert win.a_manual.shortcut().toString() == "F1"
        win.show_manual()
        assert "user guide" in win._manual_browser.toPlainText().lower()
        assert "exactness contract" in \
            win._manual_browser.toPlainText().lower()
        first = win._manual_dlg
        win.show_manual()                        # second call raises, not dupes
        assert win._manual_dlg is first
        win._manual_dlg.close()
    finally:
        win.close()


def test_band_slider_certifies_the_budgeted_solve(qapp, tmp_path):
    """The certification band, in the user's hands: a two-cursor slider
    above the Bode, visible exactly when a budgeted form is chosen, its
    span mirrored on the plot and carried into the solve — the result's
    recorded band is whatever the cursors said."""
    from circuitinsight.gui.app import MainWindow

    MainWindow.settings_path = str(tmp_path / "gui.ini")
    try:
        win = MainWindow()
    finally:
        MainWindow.settings_path = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        win.open_session(str(FIX / "tb_ota5t.cin.json"), str(FIX / "psf"))
        win.in_combo.setCurrentText("VIND")
        win.out_combo.setCurrentText("vout")
    try:
        win.mode_combo.setCurrentText("Transfer")
        assert not win.band_row.isVisibleTo(win)      # Exact: no band, no tube
        assert not win._tol_bands
        win.form_combo.setCurrentText("Simplified · full order")
        assert not win.band_row.isVisibleTo(win)      # band = lowest order only
        assert win._tol_bands                         # but the tube appears
        win.form_combo.setCurrentText("Simplified · lowest order")
        assert win.band_row.isVisibleTo(win)
        assert win._band_spans                        # mirrored on the plot
        assert win._tol_bands                         # tube inside the band
        la, lb = win.band_slider.labels()             # cursors print their f
        assert "Hz" in la and "Hz" in lb

        # the slider covers exactly what the simulation covered: its
        # range follows the result's frequency grid (the AC sweep)
        f = [float(x) for x in win.result.freqs]
        rlo, rhi = win.band_slider.range()
        assert rlo == pytest.approx(min(f), rel=1e-9)
        assert rhi == pytest.approx(max(f), rel=1e-9)

        win.band_slider.setValues(1e4, 1e8)
        lo, hi = win.band_slider.values()
        assert lo == pytest.approx(1e4, rel=1e-6)
        assert hi == pytest.approx(1e8, rel=1e-6)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            r = win.reduce_sync()                     # lowest order: user band
        assert r.band_fmin == pytest.approx(1e4, rel=1e-6)
        assert r.band_fmax == pytest.approx(1e8, rel=1e-6)

        # cursors cannot cross
        win.band_slider.setValues(1e6, 1e3)
        lo, hi = win.band_slider.values()
        assert lo <= hi
    finally:
        win.close()


def test_instance_and_net_trees(qapp, tmp_path):
    """The circuit as two trees. Instances: devices grouped under their
    subcircuit path, an OP glance on hover. Nets: every net with its
    connections, in/out arrows live with the combos, double-click sets
    the output, and an AC-ground wish routes through the measured
    Reduce flow instead of grounding silently."""
    from circuitinsight.gui.app import MainWindow
    from circuitinsight.gui.devtree import IN_MARK, OUT_MARK

    MainWindow.settings_path = str(tmp_path / "gui.ini")
    try:
        win = MainWindow()
    finally:
        MainWindow.settings_path = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        win.open_session(str(FIX / "tb_ota5t.cin.json"), str(FIX / "psf"))
        win.in_combo.setCurrentText("VIND")
        win.out_combo.setCurrentText("vout")
    try:
        # hierarchy: I0.MN0 is a leaf under the I0 container
        it = win.devices.item_for("I0.MN0")
        assert it is not None and it.parent() is not None
        assert it.parent().text(0) == "I0"
        assert win.devices.device_name(it) == "I0.MN0"

        # the OP glance arrives on first hover, cached on the item --
        # bias and small-signal values only, no connectivity lines
        # (field report: the terminal→net rows were noise)
        win.devices._hover(it, 0)
        assert "gm" in it.toolTip(0) and "vds" in it.toolTip(0)
        assert "→" not in it.toolTip(0)

        # nets: vout exists, has connections, and wears the out arrow;
        # the input source's net wears the in arrow; ground is grayed
        vout = win.nets_tree.item_for("vout")
        assert vout is not None and vout.childCount() >= 2
        assert OUT_MARK in vout.text(0)
        innet = win._input_net()
        assert innet and IN_MARK in win.nets_tree.item_for(innet).text(0)

        # the nets tree is hierarchical: testbench nets at the root
        # first, then each subcircuit as a container holding its nets
        assert vout.parent() is None
        dotted = [n for n in win.controller.nets if "." in n]
        if dotted:
            inner = win.nets_tree.item_for(dotted[0])
            assert inner.parent() is not None
            assert inner.parent().text(0) == dotted[0].rsplit(".", 1)[0]
            first = [(win.nets_tree.topLevelItem(i).data(0, 257),
                      win.nets_tree.topLevelItem(i).data(0, 256))
                     for i in range(win.nets_tree.topLevelItemCount())]
            gnd = set(win.controller.ground)
            net_idx = [i for i, (k, n) in enumerate(first)
                       if k == "net" and n not in gnd]
            sub_idx = [i for i, (k, _n) in enumerate(first) if k == "sub"]
            if sub_idx and net_idx:
                assert max(net_idx) < min(sub_idx)

        # double-click semantics: a net becomes the output
        other = next(win.out_combo.itemText(i)
                     for i in range(win.out_combo.count())
                     if win.out_combo.itemText(i) != "vout")
        win.out_combo.setCurrentText(other)
        assert OUT_MARK not in vout.text(0)
        win.nets_tree._dclick(vout, 0)
        assert win.out_combo.currentText() == "vout"
        assert OUT_MARK in vout.text(0)

        # a connection child jumps to its instance in the Instances tree
        child = vout.child(0)
        win.nets_tree._dclick(child, 0)
        assert win.dev_tabs.currentWidget() is win.devices
        assert win.devices.currentItem() is not None

        # an AC-ground wish switches to the Reduce bench and remembers
        # the net until the scan has priced it — never grounds silently
        # (the scan itself is stubbed: threads outlive test teardown)
        scans = []
        win.run_acg_scan = lambda: scans.append(True)
        win._acg_from_net("vout")
        assert win.mode_combo.currentText() == "Reduce circuit"
        assert "vout" in win._acg_pending and scans
    finally:
        win.close()


def test_plot_toolbar_slider_alignment_and_typeset_summary(qapp, tmp_path):
    """The plot chrome refit: interaction buttons live in a vertical
    toolbar beside the plots; the band slider's groove is margin-synced
    to the Bode's frequency axis on every draw; the Summary typesets
    through the web view (buffer text unchanged) and no longer carries
    the LaTeX H(s) tail."""
    from PySide6.QtCore import Qt
    from circuitinsight.gui import summaryweb
    from circuitinsight.gui.app import MainWindow
    from circuitinsight.gui.rangeslider import HANDLE_R

    MainWindow.settings_path = str(tmp_path / "gui.ini")
    try:
        win = MainWindow()
    finally:
        MainWindow.settings_path = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        win.open_session(str(FIX / "tb_ota5t.cin.json"), str(FIX / "psf"))
    try:
        assert win.nav.orientation() == Qt.Vertical
        assert win.nav.parent() is not None      # in the layout, not floating

        pos = win.canvas.figure.axes[0].get_position()
        w = win.canvas.width()
        win._sync_band_row()
        m = win.band_row.layout().contentsMargins()
        assert m.left() == max(0, round(pos.x0 * w) - HANDLE_R)
        assert m.right() == max(0, round((1.0 - pos.x1) * w) - HANDLE_R)

        txt = win.summary.toPlainText()
        assert "DC gain" in txt and "H(s):" not in txt

        html = summaryweb.summary_html(
            "VIND → vout\nDC gain : 1\n⚠ bad <tag>\nsection lead:")
        assert 'class="warn"' in html and "&lt;tag&gt;" in html
        assert 'class="sec"' in html
    finally:
        win.close()


def test_whatif_refuses_lowest_order_and_status_states_the_consequence(
        qapp, tmp_path):
    """The two companions of the form rename. What-if on a lowest-order
    result is a correctness trap — slider excursions leave the band and
    operating point the reduction was certified for — so it refuses with
    the reason and points at full order. And the status line states the
    order consequence on every budgeted solve, because the degree change
    is the fact that makes the two forms click."""
    from PySide6.QtCore import Qt
    from circuitinsight.gui.app import MainWindow

    MainWindow.settings_path = str(tmp_path / "gui.ini")
    try:
        win = MainWindow()
    finally:
        MainWindow.settings_path = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        win.open_session(str(FIX / "tb_ota5t.cin.json"), str(FIX / "psf"))
        win.in_combo.setCurrentText("VIND")
        win.out_combo.setCurrentText("vout")
        win._rank()
        win.keep_tbl.item(0, 0).setCheckState(Qt.Checked)

        r_full = win.simplify_sync()
        assert not getattr(r_full, "reduced_order", False)
        s_full = win._result_status(r_full)
        assert "poles kept" in s_full or "pole kept" in s_full
        win._rebuild_whatif(r_full)
        assert win._wf_sliders, "full order must keep its sliders"

        r_low = win.reduce_sync()
    try:
        assert r_low.reduced_order
        s_low = win._result_status(r_low)
        assert "LOWERED to" in s_low and "over the band" in s_low

        win._rebuild_whatif(r_low)
        assert not win._wf_sliders, "lowest order must refuse sliders"
        assert "disabled for lowest-order" in win._wf_hint.text()
        assert not win._wf_hint.isHidden()       # explicitly shown (the tab
                                                 # page itself may be inactive)

        # and a later full-order rebuild restores the ordinary hint
        win._rebuild_whatif(r_full)
        assert win._wf_sliders
        assert "disabled" not in win._wf_hint.text()
    finally:
        win.close()


def test_suggest_is_form_aware(qapp, tmp_path):
    """With lowest order selected, Suggest proposes the STORY keep — the
    pursuit's reactances plus the A0/pole conductances over the band the
    user chose — instead of the time-budget plan."""
    from circuitinsight.gui.app import MainWindow

    MainWindow.settings_path = str(tmp_path / "gui.ini")
    try:
        win = MainWindow()
    finally:
        MainWindow.settings_path = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        win.open_session(str(FIX / "tb_ota5t.cin.json"), str(FIX / "psf"))
        win.in_combo.setCurrentText("VIND")
        win.out_combo.setCurrentText("vout")
    try:
        win.mode_combo.setCurrentText("Transfer")
        win.form_combo.setCurrentText("Simplified · lowest order")
        win.band_slider.setValues(1e2, 1e6)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            win._suggest_keep()
        keep = win.checked_keep()
        assert keep and len(keep) <= 5
        assert any(k.startswith(("gm_", "gds_")) or k == "CL" for k in keep)
        assert "story keep" in win.statusBar().currentMessage()
    finally:
        win.close()


def test_log_tab_records_the_run(qapp, tmp_path):
    """A live operations log: the status bar shows only the current
    line, so a finished run's history — phases, timings, estimate
    accuracy, which backend ran — was gone exactly when a bug report
    needed it. The Log tab keeps it, timestamped and copyable."""
    from circuitinsight.gui.app import MainWindow

    MainWindow.settings_path = str(tmp_path / "gui.ini")
    try:
        win = MainWindow()
    finally:
        MainWindow.settings_path = None
    try:
        assert win.tabs.indexOf(win.logview) >= 0        # it is a tab
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            win.open_session(str(FIX / "tb_ota5t.cin.json"), str(FIX / "psf"))
            win.in_combo.setCurrentText("VIND")
            win.out_combo.setCurrentText("vout")

        import time as _t
        win._est_s = 4.0
        win._run_est = 4.0          # what _launch records
        win._t0 = _t.monotonic() - 8.0
        win._set_phase("preparing")      # the FIRST phase logs too
        win._set_phase("evaluating", 5, 10)              # (field request:
        win._log_finish("DONE")          # preparing only appeared post
        txt = win.logview.toPlainText()  # mortem in the took-suffix)
        assert "phase: preparing" in txt
        assert "phase: evaluating" in txt
        assert "DONE after 8" in txt and "2.0x" in txt   # vs the estimate
        assert txt.count("[") >= 2                       # timestamped lines

        for i in range(win._LOG_MAX_LINES + 50):         # bounded
            win.log(f"line {i}")
        assert win.logview.document().blockCount() <= win._LOG_MAX_LINES + 5
    finally:
        win._t0 = None
        win.close()


def test_log_reports_phase_durations_and_a_breakdown(qapp, tmp_path):
    """Field report: the log gave phase ENTRY timestamps, so the reader
    had to subtract to see where the time went — and a solve that
    entered 'evaluating' twice (a backend fallback re-running the grid)
    hid that fact entirely. Each transition now states the finished
    phase's duration, and the closing lines give the breakdown with
    repeat counts."""
    import time as _t

    from circuitinsight.gui.app import MainWindow

    MainWindow.settings_path = str(tmp_path / "gui.ini")
    try:
        win = MainWindow()
    finally:
        MainWindow.settings_path = None
    try:
        win._est_s = 10.0
        win._run_est = 10.0          # what _launch records
        win._t0 = _t.monotonic()
        win._phase_totals, win._phase_runs = {}, {}
        win._phase, win._phase_t0 = None, None

        win._set_phase("evaluating", 1, 10)
        win._phase_t0 -= 4.0                      # pretend 4 s elapsed
        win._set_phase("reconstructing")
        win._phase_t0 -= 2.0
        win._set_phase("evaluating", 1, 10)       # the fallback re-run
        win._phase_t0 -= 6.0
        win._log_finish("DONE")

        txt = win.logview.toPlainText()
        assert "[evaluating took 4.0s]" in txt
        assert "[reconstructing took 2.0s]" in txt
        assert "phases:" in txt
        assert "x2" in txt                        # evaluating ran twice
        assert win._phase_totals["evaluating"] == pytest.approx(10.0, abs=0.5)
        assert "%" in win._phase_breakdown()
    finally:
        win._t0 = None
        win.close()


def test_error_plots_align_with_the_bode(qapp, tmp_path):
    """The residual must sit under the main plots, frequency axis for
    frequency axis. Both figures run tight_layout, but the Error tab's
    y labels are wider ('Δ|H| (dB)'), so equal fractions are not equal
    pixels -- and the two canvases have different widths. The alignment
    is computed in WINDOW coordinates."""
    from PySide6.QtCore import QPoint

    from circuitinsight.gui.app import MainWindow

    MainWindow.settings_path = str(tmp_path / "gui.ini")
    try:
        win = MainWindow()
    finally:
        MainWindow.settings_path = None
    try:
        win.resize(1200, 800)
        win.show()
        qapp.processEvents()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            win.open_session(str(FIX / "tb_ota5t.cin.json"), str(FIX / "psf"))
            win.in_combo.setCurrentText("VIND")
            win.out_combo.setCurrentText("vout")
            win.solve_sync()
        # the Error page must be SHOWN to be laid out (a hidden tab
        # page keeps a stale width, and the aligner correctly declines
        # rather than computing from it)
        win.tabs.setCurrentWidget(win.err_canvas)
        qapp.processEvents()
        win.err_canvas.draw()
        win._align_error_axes()
        qapp.processEvents()

        def span(canvas):
            ax = canvas.figure.axes[0].get_position()
            x0 = canvas.mapTo(win, QPoint(0, 0)).x()
            w = canvas.width()
            return x0 + ax.x0 * w, x0 + ax.x1 * w

        ml, mr = span(win.canvas)
        el, er = span(win.err_canvas)
        assert abs(ml - el) <= 3 and abs(mr - er) <= 3   # within 3 px
    finally:
        win.close()


def test_strategy_dropdown_gates_the_reduction(qapp, tmp_path):
    """The lowest-order contract is a TOLERANCE STRATEGY: per-strategy
    spins appear with their strategy, the tube renders each strategy's
    own promise (plain: both budgets over the band; stability: phase
    around the crossover only; rejection: magnitude only), and the
    stability solve reports margins in the strip."""

    from circuitinsight.gui.app import MainWindow

    MainWindow.settings_path = str(tmp_path / "gui.ini")
    try:
        win = MainWindow()
    finally:
        MainWindow.settings_path = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        win.open_session(str(FIX / "tb_ota5t.cin.json"), str(FIX / "psf"))
        win.in_combo.setCurrentText("VIND")
        win.out_combo.setCurrentText("vout")
    try:
        win.mode_combo.setCurrentText("Transfer")
        win.form_combo.setCurrentText("Simplified · lowest order")
        assert win._strategy_act.isVisible()

        # plain: the dB/deg pair serves as the criterion spins
        win.strategy_combo.setCurrentText("Gain & phase")
        assert win._mag_act.isVisible() and win._phase_act.isVisible()
        assert not win._pm_act.isVisible() and not win._rej_act.isVisible()
        win.band_slider.setValues(1e4, 1e8)
        win._update_tol_bands()
        assert len(win._tol_bands) >= 2          # mag AND phase tubes

        # rejection: one dB knob, magnitude tube only
        win.strategy_combo.setCurrentText("Rejection (dB)")
        assert win._rej_act.isVisible() and not win._mag_act.isVisible()
        win._update_tol_bands()
        fills = [a for a in win._tol_bands if hasattr(a, "get_paths")]
        assert len(fills) == 1

        # stability: PM/GM knobs; the tube lives around the crossover
        win.strategy_combo.setCurrentText("Stability (margins)")
        assert win._pm_act.isVisible() and win._gm_act.isVisible()
        win.band_slider.setValues(1e3, 1e9)
        win._update_tol_bands()
        fills = [a for a in win._tol_bands if hasattr(a, "get_paths")]
        if fills:                                # crossover in band
            verts = fills[0].get_paths()[0].vertices
            span = verts[:, 0].max() / max(verts[:, 0].min(), 1e-30)
            assert span < 10.0, "stability tube hugs the crossover"

        # the solve carries the strategy end to end. ota5t's |H| plateaus
        # ABOVE unity (feedthrough), so there is no crossover in band —
        # and the honest stability verdict says exactly that instead of
        # inventing margins to preserve.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            r = win.reduce_sync()
        assert getattr(r, "strategy", None) == "stability"
        assert "PM" in r.warnings[0] or "unity crossing" in r.warnings[0]
        assert any("margins" in d or "criterion" in d for d in r.details)
    finally:
        win.close()


def test_advisory_passes_do_not_inherit_the_solve_estimate(qapp, tmp_path):
    """Caught in the Log: 'START explaining the numbers ... [estimate
    ~537s]' -- the explain passes had silently inherited the SOLVE's
    estimate, which prices a completely different analysis. A job the
    keep-set estimator does not describe must say so, and must not feed
    its own wall time back into the solve-time model."""
    from circuitinsight.gui.app import MainWindow

    MainWindow.settings_path = str(tmp_path / "gui.ini")
    try:
        win = MainWindow()
    finally:
        MainWindow.settings_path = None
    try:
        win._est_s = 537.0
        win._run_est = 537.0          # what _launch records                       # a solve's estimate
        launched = []
        win._thread = None
        orig = win._launch

        def spy(fn, label, on_done=None, est_s=win._KEEP_EST):
            launched.append((label, est_s))
            win._run_est = (win._est_s if est_s is win._KEEP_EST else est_s)

        win._launch = spy
        # explain now keys off the DISPLAYED result (the field report:
        # right after opening, panel and display disagree by design and
        # stories computed for the panel could never attach) — a stub
        # result stands in; only the launch pricing is exercised
        from types import SimpleNamespace

        win.controller = object()
        win.result = SimpleNamespace(inp="VIND", out="vout", keep=[])
        win.explain_numbers()
        assert launched and launched[-1][1] is None
        assert win._run_est is None              # no inherited promise

        win._launch = orig
        win._est_s = 537.0
        win._run_est = 537.0          # what _launch records
        win._run_est = win._est_s                # a real solve keeps its own
        assert win._run_est == 537.0
    finally:
        win.controller = None
        win.close()


def test_log_names_the_reason_when_a_phase_total_grows(qapp, tmp_path):
    """70/74 was a mystery: the total grew silently when the pursuit
    queued another round. The Log now prints old -> new with the
    launcher's reason."""
    from circuitinsight.gui.app import MainWindow

    MainWindow.settings_path = str(tmp_path / "gui.ini")
    try:
        win = MainWindow()
    finally:
        MainWindow.settings_path = None
    try:
        import time as _t

        win._t0 = _t.monotonic()
        win._phase_totals, win._phase_runs = {}, {}
        win._phase, win._phase_t0 = None, None
        win._growth_reason = "the pursuit accepted a reactance"
        win._set_phase("evaluating", 5, 70)
        win._set_phase("evaluating", 6, 74)
        text = win.logview.toPlainText()
        assert "70 -> 74" in text
        assert "accepted a reactance" in text
    finally:
        win._t0 = None
        win.close()


def test_cap_toggle_keeps_the_result_and_ticks_correlated(qapp, tmp_path):
    """Field report: toggling Charge-matrix caps replaced the hybrid
    result with the re-open's numeric first light while the keep panel
    kept its ticks — display and panel disagreed. The toggle must keep
    the previous result on screen, keep the ticks untouched, re-launch
    the solve WITH those ticks, and say what state the screen is in."""
    from PySide6.QtCore import Qt
    from circuitinsight.gui.app import MainWindow

    MainWindow.settings_path = str(tmp_path / "gui.ini")
    try:
        win = MainWindow()
    finally:
        MainWindow.settings_path = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        win.open_session(str(FIX / "tb_ota5t.cin.json"), str(FIX / "psf"))
        win.in_combo.setCurrentText("VIND")
        win.out_combo.setCurrentText("vout")
        win._rank()
        win.keep_tbl.item(0, 0).setCheckState(Qt.Checked)
        ticks = win.checked_keep()
        r_hybrid = win.solve_sync()
    try:
        assert r_hybrid.keep == ticks

        launches = []
        win.solve = lambda: launches.append(True)   # spy the re-launch
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            win.a_matrix_caps.setChecked(False)     # matrix -> lumped

        # the displayed result is NOT replaced by the numeric first
        # light, the re-solve launched, and the strip says what state
        # the screen is in
        assert win.result is r_hybrid
        assert launches, "the re-solve with the keep set must launch"
        assert "previous" in win.msg_strip.text()
        # ticks: every symbol that EXISTS under the new cap model stays
        # ticked, and nothing the user did not tick gets added — the
        # lumped model's auto-plan must not pollute a curated set
        new_names = {n for n, _, _ in
                     win.controller.rank_symbols("VIND", "vout")}
        survivors = [t for t in ticks if t in new_names]
        assert win.checked_keep() == survivors

        # the previous result stays on screen until the re-solve lands,
        # and it IS from the other cap model: the star says so, by name
        win._refresh_solve_hint()
        assert win.solve_btn.text() == "Solve *"
        assert "cap model" in win.solve_btn.toolTip()
    finally:
        win.close()


def test_strategy_pm_spin_is_the_strategys_own(qapp, tmp_path):
    """Regression: self.pm_spin was bound twice — first as the stability
    strategy's PM budget (5°), then rebound as Compensate's PM target
    (60°, clamped 30–85). The later binding won, so the toolbar spin the
    user turned did nothing and every GUI stability reduction silently
    ran with a 60° budget. The two families now carry distinct names and
    a stale bare pm_spin reference fails loudly."""
    from circuitinsight.gui.app import MainWindow

    MainWindow.settings_path = str(tmp_path / "gui.ini")
    try:
        win = MainWindow()
    finally:
        MainWindow.settings_path = None
    try:
        assert win.strat_pm_spin is not win.comp_pm_spin
        assert not hasattr(win, "pm_spin")
        win.strategy_combo.setCurrentText("Stability (margins)")
        win.strat_pm_spin.setValue(7.0)
        win.comp_pm_spin.setValue(60.0)     # must NOT leak into the opts
        opts = win._strategy_opts()
        assert opts["pm_deg"] == 7.0
        assert opts["gm_db"] == win.strat_gm_spin.value()
    finally:
        win.close()


def test_job_finished_reenables_the_whole_action_family(qapp, tmp_path):
    """Regression: four completion handlers each re-enabled their own
    subset of the launch-disabled actions, so a Compensate/Modes/GFT run
    left Simplify and Reduce greyed until an unrelated mode change
    happened to fix them. One helper re-enables ALL of them plus the
    caller's extras."""
    from circuitinsight.gui.app import MainWindow

    MainWindow.settings_path = str(tmp_path / "gui.ini")
    try:
        win = MainWindow()
    finally:
        MainWindow.settings_path = None
    try:
        family = (win.solve_btn, win.a_solve, win.a_simplify,
                  win.a_reduce, win.suggest_btn)
        for b in family:
            b.setEnabled(False)
        win._job_finished(win.suggest_btn)
        for b in family:
            assert b.isEnabled()
    finally:
        win.close()


def test_strategy_controls_persist_across_restarts(qapp, tmp_path):
    """Regression: the newer toolbar controls (solve form, tolerance
    strategy, PM/GM/rejection budgets) were never saved on close while
    the older mag/phase pair was — half the toolbar forgot its settings
    between runs."""
    from circuitinsight.gui.app import MainWindow

    MainWindow.settings_path = str(tmp_path / "gui.ini")
    try:
        win = MainWindow()
        win.form_combo.setCurrentText("Simplified · lowest order")
        win.strategy_combo.setCurrentText("Rejection (dB)")
        win.strat_pm_spin.setValue(11.0)
        win.strat_gm_spin.setValue(4.0)
        win.strat_rej_spin.setValue(2.5)
        win.close()

        win2 = MainWindow()
        try:
            assert (win2.form_combo.currentText()
                    == "Simplified · lowest order")
            assert win2.strategy_combo.currentText() == "Rejection (dB)"
            assert win2.strat_pm_spin.value() == 11.0
            assert win2.strat_gm_spin.value() == 4.0
            assert win2.strat_rej_spin.value() == 2.5
        finally:
            win2.close()
    finally:
        MainWindow.settings_path = None


def test_autosave_failure_is_reported_once(qapp, tmp_path, monkeypatch):
    """Regression: _autosave_state swallowed every exception, so a
    broken autosave path silently disabled crash recovery for the whole
    session. The first failure now lands in the Log and the strip, and
    does not repeat on every shown result."""
    from circuitinsight.gui import state as st
    from circuitinsight.gui.app import MainWindow

    MainWindow.settings_path = str(tmp_path / "gui.ini")
    try:
        win = MainWindow()
    finally:
        MainWindow.settings_path = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        win.open_session(str(FIX / "tb_ota5t.cin.json"), str(FIX / "psf"))
    try:
        def boom(*_a, **_k):
            raise OSError("disk full")

        monkeypatch.setattr(st, "save_state", boom)
        win._autosave_state()
        txt = win.logview.toPlainText()
        assert "autosave FAILED" in txt and "disk full" in txt
        assert "autosave failed" in win.msg_strip.text()
        win._autosave_state()               # said once, not per result
        assert win.logview.toPlainText().count("autosave FAILED") == 1
    finally:
        win.close()


def test_gmb_lump_toggle_reopens_with_the_model(qapp, tmp_path):
    """The Model-menu ĝm toggle mirrors the cap-model flow: it flips
    mos_model, persists it, re-opens the run with the model baked in,
    and reports per device what the exact criterion actually did."""
    from circuitinsight.gui.app import MainWindow

    MainWindow.settings_path = str(tmp_path / "gui.ini")
    try:
        win = MainWindow()
    finally:
        MainWindow.settings_path = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        win.open_session(str(FIX / "tb_ota5t.cin.json"), str(FIX / "psf"))
        win.in_combo.setCurrentText("VIND")
        win.out_combo.setCurrentText("vout")
    try:
        assert win.mos_model == "separate"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            win.a_lump_gmb.setChecked(True)
        assert win.mos_model == "lumped-gmb"
        assert win.controller.mos_model == "lumped-gmb"
        # the report says what happened -- lumped, dropped, or nothing
        assert "ĝm" in win.logview.toPlainText().lower()             or "gm lumping" in win.logview.toPlainText().lower()
        # persisted beside the cap model
        assert str(win._settings().value("mos_model")) == "lumped-gmb"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            win.a_lump_gmb.setChecked(False)
        assert win.controller.mos_model == "separate"
    finally:
        win.close()


def test_summary_grows_the_approximation_ledger(qapp, tmp_path):
    """After a solve, a background advisor appends the ledger to the
    Summary: every circuit-level approximation priced under the toolbar
    contract, with the measured totals — one unit everywhere."""
    from circuitinsight.gui.app import MainWindow

    MainWindow.settings_path = str(tmp_path / "gui.ini")
    try:
        win = MainWindow()
    finally:
        MainWindow.settings_path = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        win.open_session(str(FIX / "tb_ota5t.cin.json"), str(FIX / "psf"))
        win.in_combo.setCurrentText("VIND")
        win.out_combo.setCurrentText("vout")
        win.solve_sync()
    try:
        assert _wait_bg(qapp, lambda: "approximation ledger"
                        in win.summary.toPlainText()),             "the ledger advisor must land in the Summary"
        txt = win.summary.toPlainText()
        assert "× budget" in txt
        assert "measured end to end" in txt
    finally:
        win.close()


def test_async_open_shows_the_window_first(qapp, tmp_path):
    """The launcher's path: open_session(async_open=True) returns after
    the cheap populate (combos filled, no result yet), first light lands
    from the worker, and the auto keep plan fills the table from the
    advisors — the splash no longer hides 8+ seconds of session work."""
    from circuitinsight.gui.app import MainWindow

    MainWindow.settings_path = str(tmp_path / "gui.ini")
    try:
        win = MainWindow()
    finally:
        MainWindow.settings_path = None
    import time as _t
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        t0 = _t.monotonic()
        win.open_session(str(FIX / "tb_ota5t.cin.json"), str(FIX / "psf"),
                         async_open=True)
        open_s = _t.monotonic() - t0
    try:
        # the synchronous slice is the cheap populate only
        assert win.in_combo.count() > 0
        assert win.result is None, "first light must NOT be synchronous"
        assert _wait_bg(qapp, lambda: win.result is not None),             "first light lands from the worker"
        assert _wait_bg(qapp, lambda: win.keep_tbl.rowCount() > 0),             "the auto keep plan fills the table from the advisors"
        assert _wait_bg(qapp,
                        lambda: "auto keep-set"
                        in win.statusBar().currentMessage())
        assert win._match_groups, "auto matches applied in the chain"
        # the guard: a user tick set is never clobbered by a late plan —
        # simulated by the rowCount>0 gate the chain checks
        assert open_s < 10.0        # sanity, not a benchmark
    finally:
        win.close()


def test_explain_right_after_open_attaches_to_the_display(qapp, tmp_path):
    """Field report: right after opening, the shown result is the
    pre-match first light with keep=[], while the panel already carries
    the auto plan's ticks — Explain the numbers computed stories for
    the PANEL's keep and they could never attach to the displayed
    numerals; the hover kept its run-me prompt. Explain now keys off
    the displayed result and aligns it to the current configuration
    first."""
    from circuitinsight.gui.app import MainWindow

    MainWindow.settings_path = str(tmp_path / "gui.ini")
    try:
        win = MainWindow()
    finally:
        MainWindow.settings_path = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        win.open_session(str(FIX / "tb_ota5t.cin.json"), str(FIX / "psf"))
        win.in_combo.setCurrentText("VIND")
        win.out_combo.setCurrentText("vout")
    try:
        # the field state: display keep=[] (first light), panel ticked
        assert win.result is not None and win.result.keep == []
        assert win.checked_keep(), "the auto plan pre-ticks the panel"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            win.explain_numbers()
        assert _wait_bg(qapp, lambda: win._thread is None
                        or not win._thread.isRunning())
        qapp.processEvents()
        # the stories are cached under the DISPLAYED result's key, so
        # the hover lookup in _render_expr finds them
        r = win.result
        rkeep = r.keep if isinstance(r.keep, list) else []
        assert win.controller.cached_numerals(r.inp, r.out,
                                              keep=rkeep),             "stories must attach to the displayed expression"
        assert "the numbers, explained" in win.summary.toPlainText()
    finally:
        win.close()


def test_reduce_tool_shows_the_contract_it_gates_with(qapp, tmp_path):
    """Field report: the Reduce-circuit tool hid the band slider and
    the strategy dropdown while its scans priced and gated under
    exactly those controls — a criterion set by invisible knobs. The
    contract surface now shows wherever the contract governs."""
    from circuitinsight.gui.app import MainWindow

    MainWindow.settings_path = str(tmp_path / "gui.ini")
    try:
        win = MainWindow()
    finally:
        MainWindow.settings_path = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        win.open_session(str(FIX / "tb_ota5t.cin.json"), str(FIX / "psf"))
        win.in_combo.setCurrentText("VIND")
        win.out_combo.setCurrentText("vout")
    try:
        win.mode_combo.setCurrentText("Reduce circuit")
        assert win._strategy_act.isVisible()
        assert win.band_row.isVisibleTo(win)
        # the strategy's own spins follow it, exactly as in Transfer
        win.strategy_combo.setCurrentText("Stability (margins)")
        assert win._pm_act.isVisible() and win._gm_act.isVisible()
        win.strategy_combo.setCurrentText("Gain & phase")
        assert win._mag_act.isVisible() and win._phase_act.isVisible()
        # the result-form selector stays a Transfer concept
        assert not win._form_act.isVisible()
        # and Transfer/Exact still hides the whole surface
        win.mode_combo.setCurrentText("Transfer")
        win.form_combo.setCurrentText("Exact")
        assert not win._strategy_act.isVisible()
        assert not win.band_row.isVisibleTo(win)
    finally:
        win.close()


def test_declare_return_ratio_offers_the_current_transfer(qapp, monkeypatch):
    """The expected gesture is 'my current transfer IS the loop gain':
    the dialog preselects the Transfer out net and names the run's AC
    input in the prompt, instead of an anonymous v(input) and a hunt
    from '(no declaration)'."""
    from PySide6.QtWidgets import QInputDialog

    from circuitinsight.gui.app import build_window

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        win = build_window(str(FIX / "tb_ota5t.cin.json"), str(FIX / "psf"))
    try:
        win.out_combo.setCurrentText("vout")
        seen = {}

        def fake(parent, title, label, items, cur, editable):
            seen.update(label=label, items=items, cur=cur)
            return items[cur], True
        monkeypatch.setattr(QInputDialog, "getItem", staticmethod(fake))
        win.declare_return_ratio()
        assert seen["items"][seen["cur"]] == "vout"     # preselected
        inp = win.controller.suggested_input()
        assert f"v({inp})" in seen["label"]             # concrete input
        assert win.controller.ac_loop_gain == "vout"
        # already-declared wins over the current selection
        win.out_combo.setCurrentText("net1")
        win.declare_return_ratio()
        assert seen["items"][seen["cur"]] == "vout"
    finally:
        win.close()


def test_op_point_selector_binds_the_session(qapp, tmp_path):
    """A maestro corner tree opens as ONE session over its points: the
    OP combo appears with the corner labels, switching rebinds the
    controller, and the circuit key carries the point so a shown
    result goes stale honestly. A single-psf run shows no combo."""
    import shutil

    from circuitinsight.gui.app import MainWindow

    spectre = FIX.parent
    root = tmp_path / "ExplorerRun.0"
    for i, corner in ((1, "tt 27"), (2, "ss -40")):
        psf = root / str(i) / "work_tb" / "psf"
        shutil.copytree(spectre / "miller" / "psf_stb", psf)
        (psf / "runObjFile").write_text(
            f'HEADER\n"#maeCorner" "{corner}"\n'
            f' "#maeDesignPoint" {i}\n')
    MainWindow.settings_path = str(tmp_path / "s.ini")
    win = MainWindow()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            win.open_session(
                str(spectre / "miller" / "tb_ota2s_stb.cin.json"),
                str(root))
        assert win._op_act.isVisible()
        assert [win.op_combo.itemText(i)
                for i in range(win.op_combo.count())] == ["tt 27",
                                                          "ss -40"]
        k1 = win._circuit_key()["op"]
        win.op_combo.setCurrentIndex(1)
        assert win.controller.session_snapshot()["op_index"] == 1
        assert win._circuit_key()["op"] != k1
        assert "ss -40" in win.strip_lbl.text() \
            if hasattr(win, "strip_lbl") else True

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            win.open_session(str(FIX / "tb_ota5t.cin.json"),
                             str(FIX / "psf"))
        assert not win._op_act.isVisible()
    finally:
        win.close()


def test_reduce_prices_against_the_chosen_response(qapp, tmp_path):
    """The Reduce bench's contract response is a choice: Transfer for
    an ac bench, the Tian loop gain at the stb probe for a stability
    bench. The selector threads the probe through the one contract
    (_contract_kw), clears the other contract's prices on switch, and
    refuses the loop choice when the run has no probe."""
    from circuitinsight.gui.app import MainWindow

    spectre = FIX.parent
    MainWindow.settings_path = str(tmp_path / "s.ini")
    win = MainWindow()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            win.open_session(
                str(spectre / "miller" / "tb_ota2s_stb.cin.json"),
                str(spectre / "miller" / "psf_stb"))
        win.mode_combo.setCurrentText("Reduce circuit")
        assert "loop_probe" not in win._contract_kw()   # default: transfer
        win.acg_tbl.setRowCount(3)                      # stale prices
        win.reduce_against.setCurrentIndex(1)
        assert win._contract_kw()["loop_probe"] == "IPRB0"
        assert win.acg_tbl.rowCount() == 0              # cleared on switch
        win.reduce_against.setCurrentIndex(0)
        assert "loop_probe" not in win._contract_kw()
    finally:
        win.close()


def test_log_holds_the_stage_and_results_return(qapp, tmp_path):
    """A launch brings the Log tab forward (the narration lives
    there); a finished computation surfaces the tool's result tab --
    Expression for the solving benches, Suggestions for Compensate,
    the Log itself where the result tab is hidden (Reduce). The
    bottom tabs are SHARED under the Analysis/Schematic pages."""
    from circuitinsight.gui.app import MainWindow

    MainWindow.settings_path = str(tmp_path / "s.ini")
    win = MainWindow()
    try:
        # one strip of tabs under both pages
        assert win.right_split.widget(0) is win.main_tabs
        assert win.right_split.widget(1) is win.tabs
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            win.open_session(str(FIX / "tb_ota5t.cin.json"),
                             str(FIX / "psf"))
        win.mode_combo.setCurrentText("Transfer")
        win._show_result_tab()
        assert win.tabs.currentWidget() is win._expr_tab
        win.mode_combo.setCurrentText("Reduce circuit")
        win.tabs.setCurrentWidget(win.logview)
        win._show_result_tab()                # target hidden: Log stays
        assert win.tabs.currentWidget() is win.logview
    finally:
        win.close()


def test_symbol_variant_resolves_switch_state():
    """The devtypes join the pane hands the composer: switches carry
    their resolved AC state (the solver's own rule) in the variant
    slot, everything else its polarity."""
    from circuitinsight.gui.benches.schematic import _symbol_variant

    assert _symbol_variant("switch",
                           {"ac_position": "0", "position": "1"}) == "open"
    assert _symbol_variant("switch", {"position": "1"}) == "closed"
    assert _symbol_variant("switch", {}) is None    # symmap: closed
    assert _symbol_variant("mosfet", {"polarity": "n"}) == "n"


@pytest.mark.skipif(not _have_symlib(),
                    reason="no symbol library (CIN_SYMLIB unset, symbols "
                           "submodule not checked out)")
def test_proposed_branch_localizes_and_draws(qapp, tmp_path):
    """The compensation overlay through the pane: session net names
    localize per page (prefix strip on the block's page, terminal
    translation for a parent net entering it, globals pass), and a
    set _sch_proposed reaches the composed SVG on the page that owns
    the nets."""
    from circuitinsight.gui.app import MainWindow
    from circuitinsight.schematic import compose

    FC = Path(__file__).resolve().parent / "fixtures" / "spectre" / "fc"
    MainWindow.settings_path = str(tmp_path / "s.ini")
    win = MainWindow()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        win.open_session(str(FC / "tb_fc.cin.json"), str(FC / "psf"))
    try:
        # top page: own nets pass, block-internal names have no name
        assert win._sch_localize_net("vout") == "vout"
        assert win._sch_localize_net("I0.net1") is None
        win._sch_proposed = (("vin_dm", "vout", "1.1 pF", False),)
        win._schematic_render()
        assert compose.last_proposed_drawn == 1
        # descend into the OTA: prefixed nets localize, the parent
        # net entering through the port takes the port's name, and
        # the overlay follows the branch to ITS page
        win._sch_path.append(("I0", "ota_folded_cascode"))
        assert win._sch_localize_net("I0.net1") == "net1"
        assert win._sch_localize_net("vout") == "vout"   # port name
        win._sch_proposed = (("I0.net1", None, "2.2 pF", False),)
        win._schematic_render()
        assert compose.last_proposed_drawn == 1
        # a net with no name on this page drops the entry
        win._sch_proposed = (("vin_dm", "I0.net1", "3 pF", False),)
        win._schematic_render()
        assert compose.last_proposed_drawn == 0
    finally:
        win.close()


def test_schematic_pane_reports_its_missing_pieces(qapp, tmp_path):
    """Without hints beside the CIN the pane says so; without a library
    it says so; never an exception in the open path."""
    from circuitinsight.gui.app import MainWindow

    MainWindow.settings_path = str(tmp_path / "s.ini")
    win = MainWindow()
    assert win.main_tabs.indexOf(win._schematic_tab) >= 0
    assert win.tabs.indexOf(win._schematic_tab) < 0   # not a strip tab
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        win.open_session(str(FIX / "tb_ota5t.cin.json"), str(FIX / "psf"))
    assert "no layout hints" in win._sch_status.text()
    win.close()


@pytest.mark.skipif(not _have_symlib(),
                    reason="no symbol library (CIN_SYMLIB unset, symbols "
                           "submodule not checked out)")
def test_schematic_pane_draws_pages_and_cross_probes(qapp, tmp_path):
    """fc ships its hints + confirmed block symbol: the tb page draws,
    a double-click on the OTA descends, a click on a device selects it
    in the Instances tree by its dot-joined path, the crumb climbs
    back."""
    from PySide6.QtCore import QPoint

    from circuitinsight.gui.app import MainWindow

    FC = Path(__file__).resolve().parent / "fixtures" / "spectre" / "fc"
    MainWindow.settings_path = str(tmp_path / "s.ini")
    win = MainWindow()
    win.resize(1400, 900)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        win.open_session(str(FC / "tb_fc.cin.json"), str(FC / "psf"))
    st = win._sch_status.text()
    assert st.startswith("tb_ota_folded_cascode:"), st
    # the counts blurb is gone by request -- the label keeps only the
    # analysis-state summary
    assert "instances" not in st and "block(s)" not in st
    assert win.sch_view.bounds_of("I0") is not None
    # descend into the block
    win._schematic_activated("I0", "block")
    assert win._sch_status.text().startswith("ota_folded_cascode:")
    assert win.sch_view.bounds_of("PM13") is not None
    assert [d for _, d in win._sch_path] == ["tb_ota_folded_cascode",
                                              "ota_folded_cascode"]
    # click a device: the tree selects I0.PM13
    win._schematic_clicked("PM13", "device")
    assert win.devices.selected_names() == ["I0.PM13"]
    # tree selection highlights on the page
    win._schematic_follow_tree()
    assert win.sch_view._hl is not None
    # hit-testing maps a view point back to the instance
    r = win.sch_view.bounds_of("PM13")
    sp = win.sch_view._item.mapToScene(r.center())
    vp = win.sch_view.mapFromScene(sp)
    assert win.sch_view.hit(QPoint(vp.x(), vp.y()))[0] == "PM13"
    # a click on the background clears both selections
    win._schematic_deselect()
    assert win.sch_view._hl is None
    assert win.devices.selected_names() == []
    # crumb back to the top
    win._schematic_up(0)
    assert win._sch_status.text().startswith("tb_ota_folded_cascode:")
    win.close()


@pytest.mark.skipif(not _have_symlib(),
                    reason="no symbol library (CIN_SYMLIB unset, symbols "
                           "submodule not checked out)")
def test_schematic_pane_draws_the_analysis_state(qapp, tmp_path,
                                                 monkeypatch):
    """The simplified view on the same layout: after first light the
    devices whose symbols are in the formula are emphasized; after an
    AC-ground reduction the grounded net is gray-dashed with an earth
    and a dead source fades; the toggle returns the plain drawing."""
    from circuitinsight.gui.app import MainWindow
    from circuitinsight.gui.benches import schematic as schmod

    HINTS = Path(__file__).resolve().parent / "fixtures" / "hints" \
        / "tb_ota5t.hints.json"
    monkeypatch.setattr(schmod, "hints_path", lambda _cin: HINTS)
    MainWindow.settings_path = str(tmp_path / "s.ini")
    win = MainWindow()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        win.open_session(str(FIX / "tb_ota5t.cin.json"), str(FIX / "psf"))
        win.in_combo.setCurrentText("VIND")
        win.out_combo.setCurrentText("vout")
        win.solve_sync()
    # first light / solve: symbols in the formula -> emphasized labels
    win._schematic_activated("I0", "block")
    styles, nets = win._schematic_styles()
    assert any(v == "symbolic" for v in styles.values())
    svg = win.sch_view._renderer            # rendered without error
    assert svg.isValid()
    # ground the mirror node inside the OTA; the page shows it
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        win.controller.apply_reduction(["I0.net1"], inp="VIND", out="vout")
    win._refresh_reduction_banner()
    styles, nets = win._schematic_styles()
    assert nets == {"net1": "acground"}
    # nothing is faded as removed (gds and caps always remain); a device
    # whose gm died is a load, one that lost only gmbs keeps its look
    assert "removed" not in styles.values()
    summ = win.controller.reduction_summary()
    assert summ["devices_silenced"] == []
    for prim in summ["dead_primitives"]:
        kind, dev = prim.split("_", 1)
        if not dev.startswith("I0."):
            continue                      # not on the page shown
        local = dev.split(".", 1)[1]
        if kind == "gm":
            assert styles.get(local) == "passive"
        elif kind == "gmbs":
            assert styles.get(local) != "passive" or any(
                p2 == "gm_" + dev for p2 in summ["dead_primitives"])
    assert "acground" in win._sch_status.text() or \
        "AC-grounded" in win._sch_status.text()
    # the toggle gives the plain drawing back
    win._sch_state_chk.setChecked(False)
    assert "AC-grounded" not in win._sch_status.text()
    win.controller.revert_reduction()
    win.close()


@pytest.mark.skipif(not _have_symlib(),
                    reason="no symbol library (CIN_SYMLIB unset, symbols "
                           "submodule not checked out)")
def test_block_symbol_dialog_round_trip(qapp, tmp_path):
    """A block with an unconfirmed proposal wears a ? badge; the dialog
    opens prefilled from the proposal, validates, and OK writes the
    sidecar confirmed so the page redraws the symbol; 'Draw as box'
    takes it back."""
    import json
    import shutil

    from circuitinsight.gui.app import MainWindow
    from circuitinsight.gui.blockdialog import BlockSymbolDialog

    FC = Path(__file__).resolve().parent / "fixtures" / "spectre" / "fc"
    work = tmp_path / "fc"
    shutil.copytree(FC, work)
    (work / "tb_fc.symbols.json").unlink()      # no mapping at all
    MainWindow.settings_path = str(tmp_path / "s.ini")
    win = MainWindow()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        win.open_session(str(work / "tb_fc.cin.json"), str(work / "psf"))
    assert "ota_folded_cascode" not in win._sch_blocks
    # a fresh proposal exists, so the box is badged
    assert win._schematic_proposal("ota_folded_cascode") is not None
    svg_boxed = win.sch_view._renderer
    assert svg_boxed.isValid()

    cdef = win._sch_cin["definitions"]["ota_folded_cascode"]
    prop = win._schematic_proposal("ota_folded_cascode")
    dlg = BlockSymbolDialog("ota_folded_cascode", cdef["ports"],
                            win._sch_lib, prop)
    assert dlg.sym_combo.currentText() == "gm.svg"
    assert dlg._combos["vin_p"].currentText() == "in+"
    assert dlg._combos["ib"].currentText() == "stub: auto"
    assert dlg.ok_btn.isEnabled()
    # a duplicate anchor is refused with the reason
    dlg._combos["vin_n"].setCurrentText("in+")
    assert not dlg.ok_btn.isEnabled() and "in+" in dlg.verdict.text()
    dlg._combos["vin_n"].setCurrentText("in-")
    assert dlg.ok_btn.isEnabled()
    # the op-amp symbol is offered too, and re-prefills the table
    assert dlg.sym_combo.findText("oa.svg") >= 0
    bs = dlg.current()
    assert bs.confirmed and bs.pins["vout"] == "out"

    # the pane's OK path, with the dialog replaced by a double that
    # behaves as the real one after OK (result_mapping set): the real
    # handler once compared exec()'s int with a DialogCode enum and
    # read every OK as Cancel
    class _Accepting:
        def __init__(self, *a, **k):
            self.result_mapping = bs
            self.draw_as_box = False

        def exec(self):
            return 1

    win._schematic_symbol_dialog("I0", dialog_cls=_Accepting)
    side = json.loads((work / "tb_fc.symbols.json").read_text())
    assert side["cells"]["ota_folded_cascode"]["confirmed"] is True
    assert "ota_folded_cascode" in win._sch_blocks
    # the counts blurb is gone by request; the confirmed block shows by
    # drawing (bounds exist), not by a header count
    assert win.sch_view.bounds_of("I0") is not None

    class _Boxing(_Accepting):
        def __init__(self, *a, **k):
            self.result_mapping = None
            self.draw_as_box = True

    win._schematic_symbol_dialog("I0", dialog_cls=_Boxing)
    side = json.loads((work / "tb_fc.symbols.json").read_text())
    assert side["cells"]["ota_folded_cascode"]["confirmed"] is False
    assert "ota_folded_cascode" not in win._sch_blocks
    win.close()


def test_launcher_defers_the_open_chain_past_the_first_paint(qapp, tmp_path):
    """With _auto_deferred set (the launcher's path) the async open
    queues the first-light start instead of launching it, so the worker
    cannot hold the GIL while the window paints for the first time;
    firing the thunk launches it. A programmatic open still starts at
    once."""
    from circuitinsight.gui.app import MainWindow

    MainWindow.settings_path = str(tmp_path / "s.ini")
    win = MainWindow()
    win._auto_deferred = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        win.open_session(str(FIX / "tb_ota5t.cin.json"), str(FIX / "psf"),
                         async_open=True)
    assert len(win._auto_deferred) == 1
    assert win._thread is None or not win._thread.isRunning()
    win._auto_deferred.pop()()
    assert win._thread is not None
    assert _wait_bg(qapp, lambda: win.result is not None, timeout=60)
    win.close()


def test_expression_defines_and_expands_the_equivalents(qapp, tmp_path):
    """After an AC-ground reduction the lumped equivalents are DEFINED
    under the expressions (G_eq,net = sum of members = value); ticking
    Expand equivalents substitutes them into H(s) so A0 reads in the
    original device symbols, and the member values join the hover map."""
    from circuitinsight.gui.app import MainWindow

    MainWindow.settings_path = str(tmp_path / "s.ini")
    win = MainWindow()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        win.open_session(str(FIX / "tb_ota5t.cin.json"), str(FIX / "psf"))
        win.in_combo.setCurrentText("VIND")
        win.out_combo.setCurrentText("vout")
        assert not win.expand_eq_chk.isEnabled()      # nothing to expand
        # grounding the mirror node lumps Geq_I0_net3 = gds_MN0 + gds_MN2
        win.controller.apply_reduction(["I0.net1"], inp="VIND", out="vout")
        r = win.controller.attach_template(win.controller.solve(
            "VIND", "vout", ["gm_I0_MN1", "Geq_I0_net3"]))
    win._show(r, push_history=False)
    assert win.expand_eq_chk.isEnabled()
    from circuitinsight.gui import view
    defs = win.controller.equivalent_definitions()
    assert any(n == "Geq_I0_net3" for n, _, _ in defs)
    # the view lists only the equivalents the shown H(s) mentions
    win._render_expr()
    present = {str(s) for s in r.tf.expr.free_symbols}
    listed = [n for n, _, _ in defs if n in present]
    assert listed == ["Geq_I0_net3"] and len(defs) > 1
    payload = view.expr_katex(r, equivalents=defs)
    joined = " ".join(payload["lines"])
    assert "G_{eq," in joined and r"\mathrm{uS}" in joined
    # expanded: the equivalent is gone from H(s), its members are in
    x = win.controller.expand_equivalents(r)
    names = {str(s) for s in x.tf.expr.free_symbols}
    assert "Geq_I0_net3" not in names
    members = {m for n, e, _ in defs if n == "Geq_I0_net3"
               for m in map(str, e.free_symbols)}
    assert members and members <= names
    assert members <= set(x.tf.values)             # hover values present
    win.expand_eq_chk.setChecked(True)             # re-renders, no error
    win.close()


def test_solve_button_flags_a_result_from_a_previous_state(qapp, tmp_path):
    """The result on screen describes the circuit it was solved on. When
    the working circuit changes under it -- a reduction applied, the
    matches changed, the in/out pair moved, keep ticks edited -- the
    Solve button turns amber and names what changed; a fresh solve
    clears it."""
    from circuitinsight.gui.app import MainWindow

    MainWindow.settings_path = str(tmp_path / "s.ini")
    win = MainWindow()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        win.open_session(str(FIX / "tb_ota5t.cin.json"), str(FIX / "psf"))
        win.in_combo.setCurrentText("VIND")
        win.out_combo.setCurrentText("vout")
        win.solve_sync()
    # a solve with the current ticks: current. And at start-up the
    # numeric first light under an auto-ticked plan is NOT stale: the
    # ticks are options, the plot does describe the circuit
    win._refresh_solve_hint()
    assert win.solve_btn.text() == "Solve"
    assert not win.solve_btn.styleSheet()
    # a reduction applied under it: stale, and the reason is named
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        win.controller.apply_reduction(["I0.net1"], inp="VIND", out="vout")
    win._refresh_reduction_banner()
    assert win.solve_btn.text() == "Solve *"
    assert "reduction" in win.solve_btn.toolTip()
    assert win.solve_btn.styleSheet()
    # a new solve on the reduced circuit: current again. The bench's
    # own Apply re-ranks the table and carries ticks onto survivors;
    # this test drove the controller directly, so the auto plan's
    # ticks may name symbols the rewrite lumped away -- untick them:
    # the subject here is the button, not tick survival
    from PySide6.QtCore import Qt
    for i in range(win.keep_tbl.rowCount()):
        it = win.keep_tbl.item(i, 0)
        if it is not None:
            it.setCheckState(Qt.Unchecked)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        win.solve_sync()
    win._refresh_solve_hint()
    assert win.solve_btn.text() == "Solve"
    # the in/out pair moved: stale
    win.in_combo.setCurrentText("VINC")
    assert win.solve_btn.text() == "Solve *"
    assert "in/out" in win.solve_btn.toolTip()
    win.controller.revert_reduction()
    win.close()


@pytest.mark.skipif(not _have_symlib(),
                    reason="no symbol library (CIN_SYMLIB unset, symbols "
                           "submodule not checked out)")
def test_schematic_tints_matched_devices_like_the_tree(qapp, tmp_path):
    """A match group wears the same tint and number in the schematic as
    in the Instances tree: a plate behind each member's symbol and a
    link badge after its name."""
    from circuitinsight.gui.app import MainWindow
    from circuitinsight.gui.theme import GROUP_TINTS

    FC = Path(__file__).resolve().parent / "fixtures" / "spectre" / "fc"
    MainWindow.settings_path = str(tmp_path / "s.ini")
    win = MainWindow()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        win.open_session(str(FC / "tb_fc.cin.json"), str(FC / "psf"))
    win._match_groups = [("I0.PM12", "I0.PM13")]
    win.controller.set_matches(("I0.PM12", "I0.PM13"))
    win._refresh_matches_label()
    win._schematic_activated("I0", "block")
    tints = win._schematic_tints()
    assert tints == {"PM12": (GROUP_TINTS[0], "🔗1"),
                     "PM13": (GROUP_TINTS[0], "🔗1")}
    # the sensitivity map's wire gradients SURVIVE a scene rebuild:
    # they are overlay items, and the matches toggle re-composes the
    # page (field report: turning the tints off vanished them)
    win._ml_light({"I0.PM12": 1.0, "I0.PM13": 0.4})
    assert win.sch_view._hl_wires        # two contributors share a net
    win._sch_match_chk.setChecked(False)
    win._schematic_render()
    assert win.sch_view._hl_wires        # re-applied after the rebuild
    # ...and NAVIGATION is a rebuild too (field report: descending
    # into the OTA lost them): up to the testbench the contributors
    # collapse into the one block (no joining wires there), and the
    # descent brings the gradients back
    win._sch_path = win._sch_path[:1]
    win._schematic_render()
    assert not win.sch_view._hl_wires
    win._schematic_activated("I0", "block")
    assert win.sch_view._hl_wires
    win._ml_light({})
    assert not win.sch_view._hl_wires
    # the gradient covers only the JOINING run: a T-shaped net whose
    # branch leads to a bystander keeps that branch unlit
    from circuitinsight.gui.benches.schematic import SchematicView
    polys = [[(0.0, 0.0), (5.0, 0.0), (10.0, 0.0)],
             [(5.0, 0.0), (5.0, 5.0)]]
    segs = SchematicView._net_join_segments(
        polys, [(0.0, 0.1), (10.0, 0.1)])
    flat_pts = {pt for seg in segs for pt in seg}
    assert (5.0, 5.0) not in flat_pts        # the branch tip stays out
    assert (0.0, 0.0) in flat_pts and (10.0, 0.0) in flat_pts
    # a branch attaching MID-SEGMENT (the junction-dot case) must not
    # disconnect the graph: the through-segment splits at the
    # junction, and the joining run is CONTINUOUS through it (field
    # report: gaps between top-level contributors)
    polys2 = [[(0.0, 0.0), (10.0, 0.0)],       # one long bar
              [(5.0, 0.0), (5.0, 5.0)]]        # branch taps its middle
    segs2 = SchematicView._net_join_segments(
        polys2, [(0.0, 0.1), (5.0, 4.9)])
    pts2 = {pt for seg in segs2 for pt in seg}
    assert (0.0, 0.0) in pts2 and (5.0, 5.0) in pts2
    assert (5.0, 0.0) in pts2                  # split at the junction
    assert (10.0, 0.0) not in pts2             # the far bar stays out
    win.close()


def test_async_open_ends_with_a_current_result(qapp, tmp_path):
    """The launcher's chain: first light, then the suggested matches.
    The plot must end up describing the MATCHED circuit (the matched
    numeric first light replaces the pre-match one), so the Solve
    button is plain at the end of start-up -- a starred Solve on a
    screen the user has not touched reads as a bug."""
    from circuitinsight.gui.app import MainWindow

    MainWindow.settings_path = str(tmp_path / "s.ini")
    win = MainWindow()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        win.open_session(str(FIX / "tb_ota5t.cin.json"), str(FIX / "psf"),
                         async_open=True)
        assert _wait_bg(qapp, lambda: win.keep_tbl.rowCount() > 0
                        and win.result is not None
                        and getattr(win.result, "_circuit_key", {}).get(
                            "matches") == tuple(sorted(
                                tuple(g) for g in win._match_groups)),
                        timeout=120), "the matched first light lands"
    assert win._match_groups                       # ota5t has suggestions
    win._refresh_solve_hint()
    assert win.solve_btn.text() == "Solve"
    assert not win.solve_btn.styleSheet()
    win.close()


def test_reduce_bench_shows_only_what_it_uses(qapp, tmp_path):
    """Reduce circuit is a state tool: the keep panel, History, the
    match controls, Summary/Expression and the toolbar's Solve are
    irrelevant to picking and applying a node set and are hidden; the
    Nets tree comes to the front; the plot is the as-imported vs
    ticked comparison and returns to the result on leaving."""
    from circuitinsight.gui.app import MainWindow

    MainWindow.settings_path = str(tmp_path / "s.ini")
    win = MainWindow()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        win.open_session(str(FIX / "tb_ota5t.cin.json"), str(FIX / "psf"))
        win.in_combo.setCurrentText("VIND")
        win.out_combo.setCurrentText("vout")
        win.solve_sync()
    win.show()
    qapp.processEvents()
    assert win._solve_act.isVisible() and win._keep_box.isVisible()
    win.mode_combo.setCurrentText("Reduce circuit")
    qapp.processEvents()
    assert not win._solve_act.isVisible()
    assert not win._keep_col.isVisible()           # the symbols column
    assert win.reduce_col.isVisible()              # the candidates column
    assert not hasattr(win, "_match_bar")          # no button row anywhere
    assert win.dev_tabs.currentWidget() is win.nets_tree
    for w in (win.summary, win._expr_tab):
        assert not win.tabs.isTabVisible(win.tabs.indexOf(w))
    assert win.tabs.isTabVisible(win.tabs.indexOf(win.logview))
    assert win.band_row.isVisibleTo(win)          # the contract stays
    # the comparison plot lands: as imported, no overlay yet
    assert _wait_bg(qapp, lambda: win._reduce_plot_base is not None,
                    timeout=60)
    labels = [l.get_label() for ax in win.canvas.figure.axes
              for l in ax.get_lines()]
    assert "as imported" in labels
    win.mode_combo.setCurrentText("Transfer")
    qapp.processEvents()
    assert win._solve_act.isVisible() and win._keep_box.isVisible()
    assert not win.reduce_col.isVisible()
    assert win._reduce_plot_base is None
    labels = [l.get_label() for ax in win.canvas.figure.axes
              for l in ax.get_lines()]
    assert "model" in labels                       # the result's plot is back
    win.close()


def test_op_card_follows_the_selection(qapp, tmp_path):
    """The operating-point card under the trees shows the selected
    device the way a designer reads it -- region with its margins,
    bias, small-signal with the derived figures, and what the working
    model uses per symbol -- and follows the selection from the tree
    and from a keep-table row."""
    from circuitinsight.gui.app import MainWindow
    from circuitinsight.gui.opcard import op_card_rows

    MainWindow.settings_path = str(tmp_path / "s.ini")
    win = MainWindow()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        win.open_session(str(FIX / "tb_ota5t.cin.json"), str(FIX / "psf"))
        win.in_combo.setCurrentText("VIND")
        win.out_combo.setCurrentText("vout")
    assert win.op_card.name is None and not win.op_card.copy_btn.isEnabled()
    assert not win.op_card.isChecked()             # collapsed by default
    it = win.devices.item_for("I0.MN1")
    win.devices.setCurrentItem(it)
    assert win.op_card.name == "I0.MN1"
    assert win.op_card.title().startswith("Operating point — I0.MN1")
    html = win.op_card.body.text()
    for frag in ("region", "V_DS − V_DSAT", "g_m / I_D",
                 "r_o = 1/g_ds", "f_T", "as the model sees it", "gm_I0_MN1"):
        assert frag in html, frag
    win.op_card.setChecked(True)                   # open it: buttons live
    assert win.op_card.copy_btn.isEnabled()
    # the pure data: derived figures are there and sensible
    rows = dict(op_card_rows(win.controller, "I0.MN1", win.result,
                             win._match_groups, "weighted"))
    ss = {lab: val for lab, val, _ in rows["small-signal"]}
    assert float(ss["g_m / g_ds"].split()[0]) > 1.0
    assert "Hz" in ss["f_T ≈ g_m/2π(c_gs+c_gd)"]
    # a passive shows its own value, not "no live primitive" (ota5t has
    # no explicit passive: the Miller bench's CL)
    from circuitinsight.session import SessionController
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        cm = SessionController.open(MILLER / "tb_ota2s.cin.json",
                                    MILLER / "psf")
    crow = dict(op_card_rows(cm, "CL", None, (), ""))
    assert crow["as the model sees it"][0][0] == "CL"
    assert "5" in crow["as the model sees it"][0][1]
    assert not any("no live" in r[0] for r in crow["as the model sees it"])
    # a keep row selects its device on the card (the open chain ranked
    # the table; no symbolic solve needed)
    assert win.keep_tbl.rowCount() > 0
    sym = win.keep_tbl.item(0, 0).text()
    win.keep_tbl.selectRow(0)
    assert win.op_card.name and sym.endswith(
        "_" + win.op_card.name.replace(".", "_"))
    win.close()


def test_collapsed_left_panes_come_back(qapp, tmp_path):
    """Panes may be dragged to zero width (a wide plot is worth it),
    but there must be a way home: Toggle left panes (Ctrl+Shift+L)
    restores them, Reset layout restores the default proportions."""
    from circuitinsight.gui.app import MainWindow

    MainWindow.settings_path = str(tmp_path / "s.ini")
    win = MainWindow()
    win.resize(1500, 900)
    win.show()
    qapp.processEvents()
    total = sum(win.h_split.sizes())
    win.h_split.setSizes([0, total])              # the drag that hid them
    qapp.processEvents()
    assert win.h_split.sizes()[0] == 0
    win._toggle_left_panes()
    assert win.h_split.sizes()[0] > 300
    win._toggle_left_panes()
    assert win.h_split.sizes()[0] == 0
    win._toggle_left_panes()
    # the keep column collapsed INSIDE the left side comes back too
    tot = sum(win.left_split.sizes())
    win.left_split.setSizes([tot, 0])
    qapp.processEvents()
    assert win.left_split.sizes()[1] == 0
    win._toggle_left_panes()
    assert win.left_split.sizes()[1] > 100
    # and a double-click on the handle does the same
    from PySide6.QtCore import QEvent, QPointF, Qt
    from PySide6.QtGui import QMouseEvent
    win.left_split.setSizes([tot, 0])
    qapp.processEvents()
    assert win.left_split.sizes()[1] == 0
    handle = win.left_split.handle(1)
    ev = QMouseEvent(QEvent.MouseButtonDblClick, QPointF(2, 2),
                     Qt.LeftButton, Qt.LeftButton, Qt.NoModifier)
    handle.mouseDoubleClickEvent(ev)
    assert win.left_split.sizes()[1] > 100
    win._reset_layout()
    assert win.h_split.sizes()[0] >= 600
    # every VISIBLE child has room (the bench-swapped candidates
    # column is hidden outside Reduce and rightly reads 0)
    sizes = win.left_split.sizes()
    assert all(sizes[i] > 0 for i in range(win.left_split.count())
               if not win.left_split.widget(i).isHidden())
    win.close()


def test_keep_column_can_go_narrow(qapp, tmp_path):
    """The estimate line's text width was the keep column's minimum
    width; the column must shrink well below it (the label wraps)."""
    from circuitinsight.gui.app import MainWindow

    MainWindow.settings_path = str(tmp_path / "s.ini")
    win = MainWindow()
    win.resize(1500, 900)
    win.show()
    qapp.processEvents()
    win.estimate_lbl.setText("estimate: ~43.4s (grid 864, s-deg 18, "
                             "34560 dets, parallel)")
    qapp.processEvents()
    assert win._keep_col.minimumSizeHint().width() < 200
    win.close()


def test_instances_tree_is_alphabetical_with_grouped_blocks(qapp, tmp_path):
    """Within each level the tree is alphabetical (natural: NM5 before
    NM10), device instances first and compound instances (subcircuits)
    after them, and a match group travels as a block placed at its
    representative's alphabetical spot, members adjacent behind it."""
    from circuitinsight.gui.app import MainWindow

    FC = Path(__file__).resolve().parent / "fixtures" / "spectre" / "fc"
    MainWindow.settings_path = str(tmp_path / "s.ini")
    win = MainWindow()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        win.open_session(str(FC / "tb_fc.cin.json"), str(FC / "psf"))
        win.in_combo.setCurrentText("VIND")
        win.out_combo.setCurrentText("vout")

    def kids(it):
        return [it.child(i) for i in range(it.childCount())]

    root = win.devices.invisibleRootItem()
    top_items = kids(root)
    leaf = [c.text(0) for c in top_items if c.childCount() == 0]
    subs = [c.text(0) for c in top_items if c.childCount() > 0]
    # devices first, compound instances after, each run alphabetical
    assert [c.text(0) for c in top_items] == leaf + subs
    assert leaf == sorted(leaf, key=win.devices._natkey)
    assert subs == sorted(subs, key=win.devices._natkey)
    assert "I0" in subs
    i0 = next(c for c in kids(root) if c.text(0) == "I0")
    names = [c.text(0).split(" ")[0] for c in kids(i0)]
    # a group sits as one block anchored at its alphabetically-first
    # member, purely alphabetical inside (NM1..NM5 before NM2's slot)
    grp = sorted((m.split(".")[-1] for m in next(
        g for g in win._match_groups if "I0.NM5" in g)),
        key=win.devices._natkey)
    i = names.index(grp[0])
    assert names[i:i + len(grp)] == grp
    assert names.index("NM0") < i               # anchored at NM1's slot
    # ungrouped neighbours are alphabetical around the blocks
    assert names.index("M26") < names.index("NM0") < names.index("NM2")
    win.close()


def test_removal_scan_fills_a_tick_table_and_apply_deletes(qapp):
    """Scan removals is no longer advisory: it fills its own tick
    table (the left column's Device-removals tab), ticking prices the
    joint rewrite and previews the deletions, Apply carries them into
    the working circuit -- the banner then names the removed elements,
    and the applied rows wear the green 'applied' mark."""
    from circuitinsight.gui.app import MainWindow

    win = MainWindow()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        win.open_session(str(MILLER / "tb_ota2s.cin.json"),
                         str(MILLER / "psf"))
        win.in_combo.setCurrentText("VIND")
        win.out_combo.setCurrentText("vout")
    try:
        win.mode_combo.setCurrentText("Reduce circuit")
        # the candidates live in the left column, two labelled tabs
        assert win.reduce_col.isVisibleTo(win)
        assert not win._keep_col.isVisibleTo(win)
        assert [win.reduce_tabs.tabText(i)
                for i in range(win.reduce_tabs.count())] ==             ["AC grounds", "Device removals"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            rep = win.controller.scan_removals(
                "VIND", "vout", **win._contract_kw())
            win._on_removal_scan_done(rep)
        assert win.rm_tbl.rowCount() == len(rep.candidates)
        # miller's CL and Cc are both load-bearing: nothing pre-ticked,
        # Apply stays disabled until the user decides
        assert win.checked_removals() == list(rep.recommended)
        # the user forces one: the joint price and the preview follow
        from PySide6.QtCore import Qt
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            win.rm_tbl.item(0, 0).setCheckState(Qt.Checked)
        picked = win.rm_tbl.item(0, 0).text()
        assert win.checked_removals() == [picked]
        assert win.acg_apply.isEnabled()
        assert "together" in win.acg_joint_lbl.text()
        assert "deleted outright" in win.acg_preview.toPlainText()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            summ = win.controller.apply_reduction(
                [], inp="VIND", out="vout", removals=[picked])
            win._on_reduction_applied(summ)
        assert f"1 element(s) removed ({picked})" in win.red_banner.text()
        assert "measured end to end" in win.red_banner.text()
        assert win.acg_revert.isEnabled()
        # the row's tick is now in force: it says so instead of
        # reading like a pending suggestion
        from circuitinsight.gui import theme as _theme
        it = win.rm_tbl.item(0, 0)
        assert "applied" in it.toolTip()
        assert it.foreground().color().name() == _theme.GOOD.lower()
        # ticked = applied: the joint line stops double-quoting the cost
        win._refresh_acg_choice()
        assert win.acg_joint_lbl.text().startswith("ticked = applied")
        # revert clears the marks
        win.revert_reduction()
        assert win.rm_tbl.item(0, 0).toolTip() == ""
    finally:
        win.close()


def test_impedance_target_label_and_solve_on_demand(qapp):
    """The Impedance tool has NO port dropdown: a tree- or wire-picked
    target -- one net, two nets, or 'seen by' a two-terminal
    instance -- switches to the tool, the toolbar LABEL says what
    Solve will compute, the ranking re-ranks immediately, and the
    impedance itself is computed only when Solve is pressed."""
    from circuitinsight.gui.app import MainWindow

    win = MainWindow()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        win.open_session(str(MILLER / "tb_ota2s.cin.json"),
                         str(MILLER / "psf"))
        win.in_combo.setCurrentText("VIND")
        win.out_combo.setCurrentText("vout")
    try:
        # the keep ranking follows the response: choosing a target
        # re-ranks for Z (the transfer ranking comes back on the way
        # out); choosing NEVER solves
        assert _wait_bg(qapp, lambda: win.keep_tbl.rowCount() > 0,
                        timeout=60)
        launched = []
        real_launch = win._launch
        win._launch = lambda fn, *a, **k: launched.append(a)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            win._impedance_at_nets("vout", "")
        assert win.mode_combo.currentText() == "Impedance"
        assert win._imp_spec == {"node": "vout"}
        # the label is the target's surface; the probe combo is
        # loop-gain vocabulary and stays hidden here
        assert win.imp_target_lbl.text() == "from node vout to gnd"
        assert win._imp_target_act.isVisible()
        assert not win._probe_act.isVisible()
        assert launched == []                  # a selection is not a command
        assert "Solve to compute" in win.statusBar().currentMessage()
        assert _wait_bg(qapp, lambda: win.keep_tbl.rowCount() > 0,
                        timeout=60)
        z_top = win.keep_tbl.item(0, 0).text()
        # the GUI ranks over the band the plot shows (the slider),
        # so the reference ranking carries the same bound
        lo, hi = win.band_slider.values()
        assert z_top == [n for n, _, _ in
                         win.controller.rank_symbols_z(
                             {"node": "vout"}, fmin=lo, fmax=hi)][0]
        # Solve computes THE TARGET
        win._launch = real_launch
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            win.solve()
        assert _wait_bg(qapp, lambda: win.result is not None
                        and win.result.out == "Z(vout)", timeout=60)
        # seen-by, via the Instances-tree signal: target only, then
        # Solve
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            win._impedance_seen_by("CL")
        assert win.imp_target_lbl.text() == "seen by CL"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            win.solve()
        assert _wait_bg(qapp, lambda: win.result is not None
                        and win.result.out == "Z seen by CL", timeout=60)
        # the Instances tree offers the verb only on two-terminal
        # elements: CL qualifies, a mosfet does not
        it = win.devices.item_for("CL")
        assert it is not None
        assert len(dict.fromkeys(
            (win.devices.info(it).get("terminals") or {}).values())) == 2
        # merely SELECTING a net in the tool chooses the target and
        # re-ranks -- with NO solve until Solve (field request)
        launched = []
        win._launch = lambda fn, *a, **k: launched.append(a)
        win.nets_tree.setCurrentItem(win.nets_tree.item_for("vin_p"))
        assert win._imp_spec == {"node": "vin_p"}
        assert win.imp_target_lbl.text() == "from node vin_p to gnd"
        assert launched == []                  # nothing solved
        assert "Solve to compute" in win.statusBar().currentMessage()
        # the flow crumb carries the tool's own step, checked once a
        # target is set; its link fronts the Nets tree
        steps = win._crumb_steps()
        assert [s[0] for s in steps] == ["Open", "Match", "Set target",
                                         "Choose symbols", "Solve"]
        assert steps[2][2]                     # target chosen: done
        win._on_crumb_clicked("ztarget")
        assert win.dev_tabs.currentWidget() is win.nets_tree
        win._imp_spec = None
        assert not win._crumb_steps()[2][2]
        win.mode_combo.setCurrentText("Transfer")
        assert [s[0] for s in win._crumb_steps()] == [
            "Open", "Match", "Choose symbols", "Solve"]
        # a target BETWEEN TWO INTERNAL NETS cannot rank as Z: the
        # transfer fallback now says so (field case: it showed
        # silently under the Impedance toolbar and was misread as
        # Z's contributors, twice)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            win._impedance_user_spec({"node": "I0.net1", "ref": "vout"},
                                     solve=False)
        assert _wait_bg(qapp, lambda: win.keep_tbl.rowCount() > 0,
                        timeout=60)
        assert "TRANSFER" in win.statusBar().currentMessage()
        # the ranking follows the band: a settled band drag re-ranks
        # (the slider parameterizes the scores), debounced through
        # _rank_band_timer -> _rerank_on_band
        win.keep_tbl.setRowCount(0)
        win._rerank_on_band()
        assert _wait_bg(qapp, lambda: win.keep_tbl.rowCount() > 0,
                        timeout=60)
        assert win._rank_band_timer.isSingleShot()
    finally:
        win.close()


@pytest.mark.skipif(not _have_symlib(),
                    reason="no symbol library (CIN_SYMLIB unset, symbols "
                           "submodule not checked out)")
def test_schematic_highlights_a_selected_net(qapp, tmp_path):
    """Selecting a net in the Nets tree lights its wire run on the
    schematic -- the same cross-probe an instance gets -- and the two
    highlights displace each other; background deselect clears both
    trees and both marks."""
    from circuitinsight.gui.app import MainWindow

    FC = Path(__file__).resolve().parent / "fixtures" / "spectre" / "fc"
    MainWindow.settings_path = str(tmp_path / "s.ini")
    win = MainWindow()
    win.resize(1400, 900)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        win.open_session(str(FC / "tb_fc.cin.json"), str(FC / "psf"))
    try:
        it = win.nets_tree.item_for("vout")
        assert it is not None
        win.nets_tree.setCurrentItem(it)
        assert win.sch_view._net_hl is not None
        assert win.sch_view._hl is None
        # a device selection displaces the net highlight
        win._schematic_clicked("I0", "block")
        assert win.sch_view._hl is not None
        assert win.sch_view._net_hl is None
        # descend: the page's local nets highlight by their full name
        win._schematic_activated("I0", "block")
        win.nets_tree.setCurrentItem(win.nets_tree.item_for("I0.net8"))
        assert win.sch_view._net_hl is not None
        # a supply net is never prefixed and matches on any page
        win.nets_tree.setCurrentItem(win.nets_tree.item_for("vdd!"))
        assert win.sch_view._net_hl is not None
        # background deselect clears both trees and both marks
        win._schematic_deselect()
        assert win.sch_view._net_hl is None and win.sch_view._hl is None
        assert win.nets_tree.selected_nets() == []
        # a re-render with UNCHANGED style inputs is skipped outright
        # -- the scene object survives untouched
        win.nets_tree.setCurrentItem(win.nets_tree.item_for("I0.net8"))
        assert win.sch_view._net_hl is not None
        item0 = win.sch_view._item
        win._schematic_render()
        assert win.sch_view._item is item0        # skipped
        assert win.sch_view._net_hl is not None
        # a FORCED same-page re-render (changed inputs) rebuilds the
        # scene but re-applies the selection and keeps zoom and pan
        win.sch_view.scale(2, 2)
        m11 = win.sch_view.transform().m11()
        win._sch_last_render_key = None
        win._schematic_render()
        assert win.sch_view._item is not item0    # truly re-rendered
        assert win.sch_view._net_hl is not None
        assert abs(win.sch_view.transform().m11() - m11) < 1e-9
        # and a click on the ALREADY-selected net re-lights it too
        # (setCurrentItem on the current item fires no signal)
        win.sch_view.highlight_net(None)
        win._schematic_net_clicked("net8")
        assert win.sch_view._net_hl is not None
        win._schematic_deselect()
        # the reverse probe: a click on a wire finds its net and
        # selects it in the Nets tree, which lights the run
        from PySide6.QtCore import QPoint
        polys = win.sch_view._net_wires["net8"]
        ax, ay = polys[0][0]
        bx, by = polys[0][1]
        mx, my = ((ax + bx) / 2, (ay + by) / 2)
        vb = win.sch_view._renderer.viewBoxF()
        ip = ((mx - vb.x()) * win.sch_view._sx,
              (my - vb.y()) * win.sch_view._sy)
        sp = win.sch_view._item.mapToScene(*ip)
        vp = win.sch_view.mapFromScene(sp)
        assert win.sch_view.hit_net(QPoint(vp.x(), vp.y())) == "net8"
        win._schematic_net_clicked("net8")
        assert win.nets_tree.selected_nets() == ["I0.net8"]
        assert win.sch_view._net_hl is not None
        # the wire carries the Nets tree's own verbs; ground gets none
        labels = [l for l, _ in
                  win._schematic_net_actions("I0.net8", "net8")]
        assert labels == ["Set as output", "AC ground (measure cost)…",
                          "Set as impedance target (net8 to gnd)"]
        assert win._schematic_net_actions("gnd!", "gnd!") == []
        # the target verb sets the spec and the toolbar label -- it
        # never solves (the launcher is stubbed to prove it)
        launched = []
        win._launch = lambda *a, **k: launched.append(a)
        dict(win._schematic_net_actions("I0.net8", "net8"))[
            "Set as impedance target (net8 to gnd)"]()
        assert launched == []
        assert win._imp_spec == {"node": "I0.net8"}
        assert win.imp_target_lbl.text() == "from node I0.net8 to gnd"
        # a net selected beforehand adds the two-net verb: Z between
        # the pair (click one wire, right-click the other)
        acts = dict(win._schematic_net_actions("I0.net8", "net8",
                                               others=["vout"]))
        two = "Set as impedance target (between net8 and vout)"
        assert two in acts
        acts[two]()
        assert win._imp_spec == {"node": "I0.net8", "ref": "vout"}
        assert win.imp_target_lbl.text() == \
            "between nodes I0.net8 and vout"
        # ground never joins a pair
        labels = [l for l, _ in win._schematic_net_actions(
            "I0.net8", "net8", others=["gnd!"])]
        assert not any("between" in l for l in labels)
        # the middle button pans; the left button never does (it is
        # selection only -- a left-drag pan stole moving clicks)
        from PySide6.QtCore import QPointF, Qt as _Qt
        from PySide6.QtGui import QMouseEvent
        from PySide6.QtWidgets import QGraphicsView
        sv = win.sch_view
        assert sv.dragMode() == QGraphicsView.NoDrag
        sv.resize(300, 200)
        sv.scale(4, 4)
        hb = sv.horizontalScrollBar()
        hb.setValue((hb.minimum() + hb.maximum()) // 2)
        v0 = hb.value()
        press = QMouseEvent(QMouseEvent.Type.MouseButtonPress,
                            QPointF(50, 50), _Qt.MiddleButton,
                            _Qt.MiddleButton, _Qt.NoModifier)
        move = QMouseEvent(QMouseEvent.Type.MouseMove,
                           QPointF(90, 50), _Qt.NoButton,
                           _Qt.MiddleButton, _Qt.NoModifier)
        rel = QMouseEvent(QMouseEvent.Type.MouseButtonRelease,
                          QPointF(90, 50), _Qt.MiddleButton,
                          _Qt.NoButton, _Qt.NoModifier)
        sv.mousePressEvent(press)
        sv.mouseMoveEvent(move)
        sv.mouseReleaseEvent(rel)
        assert hb.value() == v0 - 40
        assert sv._pan_from is None
        # pan headroom: the scene rect carries a page on every side,
        # so the scroll range never collapses -- the pan keeps working
        # even when the WHOLE schematic fits in the viewport
        sv.fit()
        br = sv._item.boundingRect()
        assert sv._scene.sceneRect().width() >= 2.5 * br.width()
        assert hb.maximum() > hb.minimum()
        # the "f" key snaps back to the fitted view
        fitted = sv.transform().m11()
        sv.scale(4, 4)
        assert abs(sv.transform().m11() - fitted) > 1e-9
        from PySide6.QtCore import QEvent
        from PySide6.QtGui import QKeyEvent
        sv.keyPressEvent(QKeyEvent(QEvent.KeyPress, _Qt.Key_F,
                                   _Qt.NoModifier))
        assert abs(sv.transform().m11() - fitted) < 1e-9
    finally:
        win.close()


def test_impedance_tool_offers_the_form_ladder(qapp):
    """Parity with Transfer: the result-form selector shows in the
    Impedance tool, the two simplified forms map to the session's
    contract kwargs, and a lowest-order Z solve lands with the
    reduced-order flags."""
    from circuitinsight.gui.app import MainWindow

    win = MainWindow()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        win.open_session(str(MILLER / "tb_ota2s.cin.json"),
                         str(MILLER / "psf"))
        win.in_combo.setCurrentText("VIND")
        win.out_combo.setCurrentText("vout")
    try:
        win.mode_combo.setCurrentText("Impedance")
        assert win._form_act.isVisible()
        # the in/out pair is Transfer/GFT/Reduce vocabulary: hidden
        # here, not grayed (field report)
        assert not win._in_act.isVisible()
        assert not win._out_act.isVisible()
        win.mode_combo.setCurrentText("Transfer")
        assert win._in_act.isVisible() and win._out_act.isVisible()
        win.mode_combo.setCurrentText("Impedance")
        win.form_combo.setCurrentText("Simplified · full order")
        kw, _ = win._impedance_form_kw()
        assert kw["form"] == "simplified" and "mag_db" in kw
        win.form_combo.setCurrentText("Simplified · lowest order")
        kw, _ = win._impedance_form_kw()
        assert kw["form"] == "lowest" and kw["strategy"] == "plain"
        # the contract surface follows: band slider visible on lowest
        assert win.band_row.isVisibleTo(win)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            r = win.impedance_sync({"node": "vout"}, **kw)
        assert r.reduced_order and r.out == "Z(vout)"
        assert "reduced to" in r.warnings[0]
        win.form_combo.setCurrentText("Exact")
        kw, phrase = win._impedance_form_kw()
        assert kw == {} and phrase == ""
    finally:
        win.close()


def test_loop_gain_tool_offers_the_form_ladder(qapp):
    """Parity: the result-form selector shows in the Loop-gain tool,
    maps to the session contract, and a lowest-order T lands with the
    reduced flags and the stability headline."""
    from circuitinsight.gui.app import MainWindow

    NMC = Path(__file__).resolve().parent / "fixtures" / "spectre" / "nmc3"
    win = MainWindow()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        win.open_session(str(next(NMC.glob("*.cin.json"))),
                         str(NMC / "psf"))
    try:
        win.mode_combo.setCurrentText("Loop gain")
        assert win._form_act.isVisible()
        win.form_combo.setCurrentText("Simplified · lowest order")
        win.strategy_combo.setCurrentText("Stability (margins)")
        kw, phrase = win._loopgain_form_kw()
        assert kw["form"] == "lowest" and kw["strategy"] == "stability"
        assert "lowest order" in phrase
        kw["strategy_opts"] = {"pm_deg": 15.0, "gm_db": 6.0,
                               "fc_rel": 0.3}
        kw["band_fmin"], kw["band_fmax"] = 1e4, 1e8
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            r = win.loop_gain_sync(win.probe_combo.currentText(), **kw)
        assert r.reduced_order and r.out.startswith("T@")
        assert "reduced to" in r.warnings[0]
    finally:
        win.close()


def test_loop_gain_dispatch_hands_over_the_budget(qapp):
    """The Solve dispatch passes the keep pane's budget into
    loop_gain as max_seconds -- the session's gate then refuses a
    runaway Tian hybrid instead of freezing the bench -- and the
    estimate label costs the loop solve, not the transfer."""
    from circuitinsight.gui.app import MainWindow

    win = MainWindow()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        win.open_session(str(MILLER / "tb_ota2s_stb.cin.json"),
                         str(MILLER / "psf_stb"))
        win.in_combo.setCurrentText("VIND")
        win.out_combo.setCurrentText("vout")
    try:
        win.mode_combo.setCurrentText("Loop gain")
        rec = {}

        def fake_loop_gain(probe, **kw):
            rec.update(kw, probe=probe)
            raise RuntimeError("stop here")

        win.controller.loop_gain = fake_loop_gain
        # run the launched fn synchronously: the wiring is the point
        win._launch = lambda fn, *a, **k: fn(lambda *aa: None)
        try:
            win.solve()
        except RuntimeError:
            pass
        assert rec["probe"] == "IPRB0"
        assert rec["max_seconds"] == win.budget_spin.value()
        # the label costs the loop solve
        est_rec = {}
        real = win.controller.estimate_loop

        def fake_est(probe, keep=()):
            est_rec["probe"] = probe
            return real(probe, keep)

        win.controller.estimate_loop = fake_est
        win._update_estimate()
        assert _wait_bg(qapp, lambda: est_rec.get("probe") == "IPRB0",
                        timeout=30)
    finally:
        win.close()


def test_surviving_ticks_stay_visible_and_clearable(qapp, tmp_path):
    """Ticks survive reranks by design, but a rerank could sink them
    below the fold -- the field report solved with 5 symbols while
    believing in 2. On refill the ticked rows lead the list; the
    'ticked only' filter shows exactly them; the count sits on the
    checkbox and on the estimate line; clear ticks starts over."""
    from PySide6.QtCore import Qt

    from circuitinsight.gui.app import MainWindow

    MainWindow.settings_path = str(tmp_path / "s.ini")
    win = MainWindow()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        win.open_session(str(FIX / "tb_ota5t.cin.json"), str(FIX / "psf"))
        win.in_combo.setCurrentText("VIND")
        win.out_combo.setCurrentText("vout")
    try:
        assert _wait_bg(qapp, lambda: win.keep_tbl.rowCount() > 0,
                        timeout=60)
        # tick two rows from the BOTTOM half, then refill with the
        # same ranking: the order stays PURE rank order -- ticked
        # rows floated to the top misread as importance (field
        # report) -- and the count still tells the truth
        n = win.keep_tbl.rowCount()
        picked = [win.keep_tbl.item(i, 0).text() for i in (n - 1, n - 2)]
        ranking = list(win._last_ranking)
        win._fill_keep_table(ranking, checked=picked)
        top2 = {win.keep_tbl.item(i, 0).text() for i in (0, 1)}
        assert top2 != set(picked)              # rank order preserved
        assert win.keep_tbl.item(n - 1, 0).checkState() == Qt.Checked
        assert "ticked only (2)" in win.ticked_chk.text()
        # 'ticked only' hides everything else
        win.ticked_chk.setChecked(True)
        visible = [win.keep_tbl.item(i, 0).text()
                   for i in range(win.keep_tbl.rowCount())
                   if not win.keep_tbl.isRowHidden(i)]
        assert set(visible) == set(picked)
        # unticking under the filter removes the row from view
        # (rows keep rank order, so the ticked one sits at n-1)
        win.keep_tbl.item(n - 1, 0).setCheckState(Qt.Unchecked)
        visible = [i for i in range(win.keep_tbl.rowCount())
                   if not win.keep_tbl.isRowHidden(i)]
        assert len(visible) == 1
        win.ticked_chk.setChecked(False)
        # the estimate line carries the count
        win._update_estimate()
        assert _wait_bg(qapp, lambda: "1 ticked"
                        in win.estimate_lbl.text(), timeout=60)
        # clear ticks starts over
        win._clear_keep_ticks()
        assert win.checked_keep() == []
        assert "ticked only (0)" in win.ticked_chk.text()
    finally:
        win.close()


def test_first_light_stays_off_screen_outside_transfer(qapp, tmp_path):
    """A restored session that opens INTO the Impedance tool once
    showed the transfer Bode it never asked for (field report). The
    first light is still computed -- it is the pre-match baseline and
    the Transfer tool's opening view -- but other tools start with an
    empty plot; switching to Transfer with nothing solved shows it."""
    from circuitinsight.gui.app import MainWindow

    MainWindow.settings_path = str(tmp_path / "s.ini")
    win = MainWindow()
    win.mode_combo.setCurrentText("Impedance")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        win.open_session(str(MILLER / "tb_ota2s.cin.json"),
                         str(MILLER / "psf"))
    try:
        assert win.result is None                 # nothing displayed
        assert win._first_light_result is not None
        assert not any(ax.get_lines()
                       for ax in win.canvas.figure.axes)
        # entering Transfer with nothing solved shows the first light
        win.mode_combo.setCurrentText("Transfer")
        assert win.result is not None
        assert win.result.out == "vout"
    finally:
        win.close()


def test_engine_death_offers_restart_and_replays(qapp, monkeypatch):
    """Plan E-H: an uninvited engine death -- noticed idle or via a
    failed worker -- puts up ONE dialog; Yes runs the same respawn-and-
    replay as the hard cancel, No leaves the results readable with a
    warning strip."""
    from PySide6.QtWidgets import QMessageBox

    from circuitinsight.gui.app import MainWindow

    win = MainWindow()

    class _Engine:
        restarted = 0

        def engine_alive(self):
            return False

        def engine_restart(self):
            self.restarted += 1

    win.controller = _Engine()
    win._engine_log = "no-such-file.log"
    monkeypatch.setattr(QMessageBox, "question",
                        staticmethod(lambda *a, **k: QMessageBox.Yes))
    win._tail_engine_log()
    assert win.controller.restarted == 1
    assert not win._engine_down_handled     # recovered: armed again

    # declining leaves the strip warning and does NOT restart
    monkeypatch.setattr(QMessageBox, "question",
                        staticmethod(lambda *a, **k: QMessageBox.No))
    win._tail_engine_log()
    assert win.controller.restarted == 1
    assert win._engine_down_handled         # handled: no dialog storm
    win._tail_engine_log()                  # timer fires again: silent
    assert win.controller.restarted == 1
    win.close()


def test_an_advisory_phase_does_not_quote_the_solve_estimate(qapp):
    """The keep-set label prices the SOLVE the ticks describe. A margin
    ledger, a suggestion, a what-if is a different computation, and
    _launch leaves it deliberately unpriced -- the START line says so.
    The phase lines must not then quote the solve's number on the very
    next line, which is what a field reader caught: a ledger promising
    the exact solve's 321 s."""
    from circuitinsight.gui.app import MainWindow

    w = MainWindow()
    type(w).settings_path = None
    try:
        w._est_s = 321.0                # the ticked keep-set estimate
        w._run_est = None               # this pass is unpriced
        w._live_est = None
        w._t0 = time.monotonic()
        w._phase, w._phase_t0 = None, None
        w._phase_totals, w._phase_runs = {}, {}
        w.logview.clear()
        w._set_phase("preparing")
        line = w.logview.toPlainText()
        assert "preparing" in line
        assert "321" not in line, line
        assert "est ~" not in line, line
        # a pass that IS priced still says so
        w._phase = None
        w._run_est = 12.0
        w.logview.clear()
        w._set_phase("preparing")
        assert "est ~12s" in w.logview.toPlainText()
    finally:
        w.close()
