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
    assert win.devices.rowCount() > 5
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
    top = win.keep_tbl.item(0, 0)
    top.setCheckState(Qt.Checked)                    # triggers estimate update
    assert win.checked_keep() == [top.text()]
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
    assert win.matches_list.count() == 2             # groups listed, not a label

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
