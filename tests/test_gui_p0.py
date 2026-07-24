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
