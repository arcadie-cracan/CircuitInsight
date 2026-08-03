"""Headless smoke test of the PySide6 desktop app (offscreen platform).

Skipped when PySide6 isn't installed. Exercises the window wiring — open a
fixture, combos populate, a synchronous solve updates the summary + canvas —
without a display and without the worker thread.
"""
import os
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
        assert win._est_s != 300.0
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
        assert "→" in win.backend_combo.itemText(0)  # names its choice

        win.backend_combo.setCurrentIndex(2)         # force bot
        assert interp.PROBE_BACKEND == "bot"
        assert "bot" in win.estimate_lbl.text()      # estimate re-costed
        win.backend_combo.setCurrentIndex(0)         # back to auto
        assert interp.PROBE_BACKEND is None
    finally:
        interp.PROBE_BACKEND = None
        win.close()


def test_cancel_is_a_base_exception():
    """The worker's cancel must not subclass Exception: the engine's
    backend fallbacks (except Exception: run the slower path) swallowed
    an Exception-cancel and silently RE-RAN the abandoned solve. The
    engine-side propagation half lives in test_zpbatch."""
    from circuitinsight.gui.app import _Cancelled

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
    from circuitinsight.engine import interp
    from circuitinsight.gui.app import MainWindow

    win = MainWindow()
    interp.LAST_SOLVE = {"backend": "bot", "wall_s": 12.5,
                         "fell_back": False}
    note = win._backend_note()
    assert "bot" in note and "12.5" in note
    interp.LAST_SOLVE = {"backend": "qq", "wall_s": 1.0, "fell_back": True}
    assert "fell back" in win._backend_note()
    interp.LAST_SOLVE = None
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

        # outside Transfer the selector hides entirely
        win.mode_combo.setCurrentText("Loop gain")
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
        assert win.tabs.currentWidget() is win._reduce_tab

        win.acg_budget.setValue(0.2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            rep = win.controller.scan_ac_grounds("VIND", "vout",
                                                 budget_db=0.2)
            win._on_acg_scan_done(rep)
        assert win.acg_tbl.rowCount() > 0
        assert win.checked_acg_nodes() == list(rep.recommended)
        assert win.acg_apply.isEnabled()
        assert "dB together" in win.acg_joint_lbl.text()
        pv = win.acg_preview.toPlainText()
        assert "removed (exact)" in pv and "primitives" in pv

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            summ = win.controller.apply_reduction(
                win.checked_acg_nodes(), inp="VIND", out="vout")
            win._on_reduction_applied(summ)
        assert "REDUCED" in win.red_banner.text()
        assert win.acg_revert.isEnabled()
        assert "measured" in win.msg_strip.text()
        # the rewrite invalidates the ranking: lumped/removed symbols
        # made a stale-tick solve fail with "matched no symbol"
        assert win.keep_tbl.rowCount() == 0
        assert "re-Rank" in win.estimate_lbl.text()
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
        assert win.red_banner.text() == "circuit: as imported"
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
        assert not visible(win._reduce_tab)

        win.mode_combo.setCurrentText("Compensate")
        assert visible(win._comp_tab) and not visible(win._whatif_tab)
        assert win.tabs.currentWidget() is win._comp_tab

        win.mode_combo.setCurrentText("Reduce circuit")
        assert visible(win._reduce_tab) and not visible(win._comp_tab)
        # the shared views never disappear
        assert win.tabs.isTabVisible(0)          # Summary
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
        assert win.matchval_combo.currentText() == "weighted"
        assert win.controller.match_value_policy == "weighted"

        win.matchval_combo.setCurrentText("weighted")
        assert win.controller.match_value_policy == "weighted"
        win.matchval_combo.setCurrentText("average")
        assert win.controller.match_value_policy == "mean"

        group = win._match_groups[0]
        from PySide6.QtWidgets import QInputDialog
        monkeypatch.setattr(QInputDialog, "getItem",
                            staticmethod(lambda *a, **k: (group[1], True)))
        win._pick_representative(0)
        assert win.controller.match_representative(group) == group[1]
        assert win.controller.match_value_policy == "representative"
        assert win.matchval_combo.currentText() == "representative"
    finally:
        win.close()


def test_advisors_surface_in_the_gui(qapp):
    """S-E tail, GUI side: the removal scan lands in the Reduce bench's
    preview pane; Attribute poles (Analysis menu) appends the verified
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
        assert "removal scan" in win.acg_preview.toPlainText()

        atts = win.controller.pole_attribution("VIND", n_poles=2)
        before = win.summary.toPlainText()
        win._on_attribution_done(atts)
        txt = win.summary.toPlainText()
        assert txt.startswith(before)
        assert "pole attribution" in txt and "set by" in txt

        stories = win.controller.explain_numerals("VIND", "vout")
        win._on_explain_done(stories)
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

        # the OP glance arrives on first hover, cached on the item
        win.devices._hover(it, 0)
        assert "gm" in it.toolTip(0) and "→" in it.toolTip(0)

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
        win._set_phase("evaluating", 5, 10)              # a phase change
        win._log_finish("DONE")
        txt = win.logview.toPlainText()
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
    import numpy as np

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
            f = np.asarray(win.result.freqs, dtype=float)
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
        win.controller = object()                # only the launch is exercised
        win.result = None
        win.explain_numbers()                    # needs no result
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
