"""GUI P0 (docs/gui-ux-review.md): splitter sizing, persistence +
recents, cancellable worker, toolbar shortcuts, copy-LaTeX. Headless.
"""
import os
import warnings
from pathlib import Path

import pytest

pytest.importorskip("PySide6")
try:
    import PySide6.QtWidgets  # noqa: F401
except Exception as exc:
    pytest.skip(f"PySide6.QtWidgets not loadable: {exc}",
                allow_module_level=True)
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

FIX = Path(__file__).resolve().parent / "fixtures" / "spectre" / "ota5t"


@pytest.fixture(scope="module")
def qapp():
    import circuitinsight.gui.exprweb  # noqa: F401  (import order for WebEngine)
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture()
def win(qapp, tmp_path):
    from circuitinsight.gui.app import MainWindow

    MainWindow.settings_path = str(tmp_path / "gui.ini")
    w = MainWindow()
    yield w
    w.close()
    MainWindow.settings_path = None


def test_splitter_gives_plots_the_width(qapp, win):
    """The P0-a bug: setSizes before show() was clobbered and the plots
    collapsed to a sliver. With stretch factors + showEvent re-apply the
    right pane must end up wider than the left."""
    win.resize(1400, 860)
    win.show()
    qapp.processEvents()
    left, right = win.h_split.sizes()
    assert right > left, f"plots pane squeezed: left={left}, right={right}"


def test_settings_roundtrip_and_recents(qapp, tmp_path, win):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        win.open_session(str(FIX / "tb_ota5t.cin.json"), str(FIX / "psf"))
    win.mag_spin.setValue(2.5)
    win.close()                                   # persists via closeEvent

    from circuitinsight.gui.app import MainWindow
    w2 = MainWindow()
    try:
        assert w2.mag_spin.value() == pytest.approx(2.5)
        pairs = w2.recents()
        assert pairs and pairs[0][0].endswith("tb_ota5t.cin.json")
        # the File menu shows it
        acts = [a.text() for a in w2.m_recent.actions()]
        assert any("tb_ota5t" in a for a in acts)
    finally:
        w2.close()


def test_worker_cancel_lands_between_grid_points(qapp):
    from circuitinsight.gui.app import _Worker

    state = {"cancelled": False, "done": False, "steps": 0}

    def slow(progress):
        for i in range(10_000):
            state["steps"] = i
            progress(i, 10_000)                   # cancellation checkpoint
        return "finished"

    w = _Worker(slow)
    w.cancelled.connect(lambda: state.__setitem__("cancelled", True))
    w.done.connect(lambda _: state.__setitem__("done", True))
    w.cancel()                                    # pre-cancelled: first cb raises
    w.start()
    assert w.wait(5000)
    qapp.processEvents()
    assert state["cancelled"] and not state["done"]
    assert state["steps"] == 0


def test_shortcuts_and_toolbar(qapp, win):
    from PySide6.QtGui import QKeySequence

    assert win.toolbar is not None
    assert win.a_solve.shortcut() == QKeySequence("Ctrl+Return")
    assert win.a_export.shortcut() == QKeySequence("Ctrl+E")
    assert win.a_copy_tex.shortcut() == QKeySequence("Ctrl+L")
    # the two toolbars are real QToolBars on the window (shared row)
    from PySide6.QtWidgets import QToolBar
    assert len(win.findChildren(QToolBar)) >= 2


def test_copy_latex_puts_tf_on_clipboard(qapp, win):
    from PySide6.QtWidgets import QApplication

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        win.open_session(str(FIX / "tb_ota5t.cin.json"), str(FIX / "psf"))
        win.solve_sync()
    assert win.a_copy_tex.isEnabled()
    win.copy_latex()
    text = QApplication.clipboard().text()
    assert text.startswith("H(s) = ")
    assert "s" in text and len(text) > 20


def test_progress_text_names_the_phase_and_ticks_elapsed(qapp, win):
    """A solve that cannot report units still has to look alive: without an
    elapsed clock an indeterminate bar makes "working" and "wedged" look
    identical. The estimate rides along so a bad one is visible while you
    wait, not only afterwards."""
    import time as _t

    win._est_s = 4.0

    win._run_est = 4.0          # what _launch records
    win._live_est = None
    win._t0 = _t.monotonic() - 1.0
    win._set_phase("preparing")
    txt = win.progress.format()
    assert txt.startswith("preparing")
    assert "1s" in txt and "~4s" in txt           # estimate rides along

    # once elapsed OVERTAKES it, the estimate grows rather than
    # asserting a finish time that has already passed
    win._t0 = _t.monotonic() - 7.0
    win._refresh_progress()
    txt = win.progress.format()
    assert "7s" in txt and "(over)" in txt
    assert win._live_est > 7.0

    win._on_progress(3, 28)                       # mid-sweep
    txt = win.progress.format()
    assert "evaluating" in txt and "3/28" in txt
    # the TEXT carries the units; the BAR is time-driven (elapsed over
    # the live estimate) so it keeps moving between unit reports
    assert win.progress.maximum() == 1000
    assert 0 < win.progress.value() <= 1000

    win._on_progress(28, 28)                      # sweep done, rebuild left
    assert "reconstructing" in win.progress.format()
    assert win.progress.maximum() == 1000         # still time-driven


def test_progress_flags_an_estimate_that_has_been_blown(qapp, win):
    import time as _t

    win._est_s = 2.0

    win._run_est = 2.0          # what _launch records
    win._live_est = None
    win._t0 = _t.monotonic() - 30.0
    win._set_phase("evaluating", 5, 40)
    # the estimate self-corrects (30 s at 1/8 done projects ~240 s), so
    # elapsed no longer overtakes it -- what must stay visible is that
    # the PROMISE was blown: the live number ran away from the
    # pre-solve one
    txt = win.progress.format()
    assert "(over)" in txt
    assert win._live_est > 1.5 * win._est_s
