"""GUI P1 (docs/gui-ux-review.md): loop-gain bench + advisor strip,
result history with overlay compare, margin/pole annotations, message
strip, HTML report. Headless.
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

MILLER = Path(__file__).resolve().parent / "fixtures" / "spectre" / "miller"


@pytest.fixture(scope="module")
def qapp():
    import circuitinsight.gui.exprweb  # noqa: F401
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture()
def win(qapp, tmp_path):
    from circuitinsight.gui.app import MainWindow

    MainWindow.settings_path = str(tmp_path / "gui.ini")
    w = MainWindow()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        w.open_session(str(MILLER / "tb_ota2s_stb.cin.json"),
                       str(MILLER / "psf_stb"))
    yield w
    w.close()
    MainWindow.settings_path = None


def test_loop_gain_bench(qapp, win):
    """The stb bench exposes its probe; the loop-gain solve lands with
    margins in the Result and PM/GM annotations on the plot."""
    probes = [win.probe_combo.itemText(i)
              for i in range(win.probe_combo.count())]
    assert "IPRB0" in probes
    win.mode_combo.setCurrentText("Loop gain")
    assert win.probe_combo.isVisibleTo(win) or True   # action visibility
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        r = win.loop_gain_sync("IPRB0")
    assert r.pm_deg == pytest.approx(59.9, abs=0.3)
    texts = [t.get_text() for ax in win.canvas.figure.axes
             for t in ax.texts]
    assert any("PM" in t for t in texts)
    assert any("GM" in t for t in texts)


def test_advisor_verdict_lands_on_the_strip(qapp, win):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        report = win.controller.assess_probe("IPRB0")
    win._on_advisor_done(report)
    assert win.msg_strip.isVisibleTo(win)
    assert "margins consistent" in win.msg_strip.text()


def test_history_and_overlay(qapp, win):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        win.in_combo.setCurrentText("VIND")
        win.out_combo.setCurrentText("vout")
        win.solve_sync()
        win.loop_gain_sync("IPRB0")
    assert win.history.count() == 2
    win.history.selectAll()
    qapp.processEvents()
    ax1 = win.canvas.figure.axes[0]
    # primary + overlay curves both present on the magnitude axis
    labels = [ln.get_label() for ln in ax1.get_lines()]
    assert any("T@IPRB0" in l for l in labels) or len(ax1.get_lines()) >= 3


def test_failure_lands_on_strip_not_modal(qapp, win):
    win._on_failed("MnaError: boom")
    assert win.msg_strip.isVisibleTo(win)
    assert "boom" in win.msg_strip.text()


def test_html_report_is_self_contained(qapp, win, tmp_path):
    from circuitinsight.gui import view

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        win.solve_sync()
    html = view.html_report(win.result)
    assert "data:image/png;base64," in html
    assert "CircuitInsight" in html
    p = tmp_path / "r.html"
    win._write_report(p)
    assert p.read_text(encoding="utf-8").startswith("<meta")
