"""GUI P2: what-if sliders on kept symbols, Compensate bench with live
preview, devices-table OP columns. Headless.
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

FIX = Path(__file__).resolve().parent / "fixtures" / "spectre"


@pytest.fixture(scope="module")
def qapp():
    import circuitinsight.gui.exprweb  # noqa: F401
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


def _win(tmp_path, cin, psf):
    from circuitinsight.gui.app import MainWindow

    MainWindow.settings_path = str(tmp_path / "gui.ini")
    w = MainWindow()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        w.open_session(str(cin), str(psf))
    return w


def test_whatif_sliders_drive_the_bode(qapp, tmp_path):
    w = _win(tmp_path, FIX / "ota5t" / "tb_ota5t.cin.json",
             FIX / "ota5t" / "psf")
    try:
        w.in_combo.setCurrentText("VIND")
        w.out_combo.setCurrentText("vout")
        for i in range(w.keep_tbl.rowCount()):     # tick exactly gm_I0_MN1
            it = w.keep_tbl.item(i, 0)
            it.setCheckState(
                __import__("PySide6.QtCore", fromlist=["Qt"]).Qt.Checked
                if it.text() == "gm_I0_MN1" else
                __import__("PySide6.QtCore", fromlist=["Qt"]).Qt.Unchecked)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            w.solve_sync()
        assert "gm_I0_MN1" in w._wf_sliders
        sl, _ = w._wf_sliders["gm_I0_MN1"]
        sl.setValue(100)                            # x4
        qapp.processEvents()
        labels = [ln.get_label() for ln in w.canvas.figure.axes[0].get_lines()]
        assert "what-if" in labels
        assert w.whatif_factors()["gm_I0_MN1"] == pytest.approx(4.0)
    finally:
        w.close()
        type(w).settings_path = None


def test_compensate_bench_preview(qapp, tmp_path):
    from circuitinsight.analysis.compensate import Candidate

    w = _win(tmp_path, FIX / "miller" / "tb_ota2s_stb.cin.json",
             FIX / "miller" / "psf_stb")
    try:
        w.mode_combo.setCurrentText("Compensate")
        w.probe_combo.setCurrentText("IPRB0")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            sugg = w.suggest_sync(
                "IPRB0", goal="pm", pm_target=65.0,
                candidates=[Candidate("miller", "I0.net1", "vout",
                                      "existing Miller port", 1.0)])
        assert sugg and w.comp_tbl.rowCount() == len(sugg)
        w.comp_tbl.selectRow(0)
        qapp.processEvents()
        assert "preview PM" in w._comp_hint.text()
        labels = [ln.get_label() for ln in w.canvas.figure.axes[0].get_lines()]
        assert "preview" in labels
    finally:
        w.close()
        type(w).settings_path = None


def test_devices_table_carries_op(qapp, tmp_path):
    w = _win(tmp_path, FIX / "ota5t" / "tb_ota5t.cin.json",
             FIX / "ota5t" / "psf")
    try:
        gm_cells = [w.devices.item(i, 3).text()      # col 2 is now region
                    for i in range(w.devices.rowCount())
                    if w.devices.item(i, 1).text() == "mosfet"]
        assert gm_cells and any("S" in c for c in gm_cells)
    finally:
        w.close()
        type(w).settings_path = None
