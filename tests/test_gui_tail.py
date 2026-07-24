"""GUI tail: Modes and GFT benches, matches-list UX, keep filter.
Headless."""
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


def test_modes_bench(qapp, tmp_path):
    w = _win(tmp_path, FIX / "fd" / "tb_fdota_stb.cin.json",
             FIX / "fd" / "psf_dm")
    try:
        w.mode_combo.setCurrentText("Modes")
        assert w.probe2_combo.count() >= 2
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            rep = w.modes_sync("FDPRB.IPRB_DM", "FDPRB.IPRB_CM")
        assert rep.margins[0][0] == pytest.approx(64.9, abs=0.3)
        # the CM margin is the sensitive one: auto-matching once fused the
        # input pair with the CMFB devices and distorted it to 55.3
        assert rep.margins[1][0] == pytest.approx(61.5, abs=0.3)
        labels = [ln.get_label() for ln in w.canvas.figure.axes[0].get_lines()]
        assert any("PM 64.9" in ln for ln in labels)
        assert "modes:" in w.msg_strip.text()
        assert "Schur certificate" in w.msg_strip.text()
    finally:
        w.close()
        type(w).settings_path = None


def test_gft_bench_exact_badge(qapp, tmp_path):
    w = _win(tmp_path, FIX / "miller" / "tb_ota2s_stb.cin.json",
             FIX / "miller" / "psf_stb")
    try:
        w.mode_combo.setCurrentText("GFT")
        w.in_combo.setCurrentText("VIND")
        w.out_combo.setCurrentText("vout")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            payload = w.gft_sync("IPRB0", "VIND", "vout", "vin_p", -1)
        assert payload["residual"] == 0.0
        assert "EXACT" in w.msg_strip.text()
        labels = [ln.get_label() for ln in w.canvas.figure.axes[0].get_lines()]
        assert "H" in labels and "feedthrough" in labels
    finally:
        w.close()
        type(w).settings_path = None


def test_matches_list_unmatch_and_tint(qapp, tmp_path):
    w = _win(tmp_path, FIX / "ota5t" / "tb_ota5t.cin.json",
             FIX / "ota5t" / "psf")
    try:
        assert w.matches_list.count() >= 1        # auto-setup suggested pairs
        tinted = sum(
            1 for i in range(w.devices.rowCount())
            if w.devices.item(i, 0).background().color().name() != "#000000")
        assert tinted >= 2                        # matched rows carry a tint
        n0 = w.matches_list.count()
        w.matches_list.setCurrentRow(0)
        w.unmatch_selected()
        assert w.matches_list.count() == n0 - 1
    finally:
        w.close()
        type(w).settings_path = None


def test_keep_filter_hides_rows(qapp, tmp_path):
    w = _win(tmp_path, FIX / "ota5t" / "tb_ota5t.cin.json",
             FIX / "ota5t" / "psf")
    try:
        assert w.keep_tbl.rowCount() > 4
        w.keep_filter.setText("gm_")
        hidden = sum(w.keep_tbl.isRowHidden(i)
                     for i in range(w.keep_tbl.rowCount()))
        shown = w.keep_tbl.rowCount() - hidden
        assert hidden > 0 and shown > 0
        for i in range(w.keep_tbl.rowCount()):
            if not w.keep_tbl.isRowHidden(i):
                assert "gm_" in w.keep_tbl.item(i, 0).text()
        w.keep_filter.setText("")
        assert not any(w.keep_tbl.isRowHidden(i)
                       for i in range(w.keep_tbl.rowCount()))
    finally:
        w.close()
        type(w).settings_path = None


def test_session_report_accumulates(qapp, tmp_path):
    w = _win(tmp_path, FIX / "miller" / "tb_ota2s_stb.cin.json",
             FIX / "miller" / "psf_stb")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            w.in_combo.setCurrentText("VIND")
            w.out_combo.setCurrentText("vout")
            w.solve_sync()
            w.add_to_report()
            w.loop_gain_sync("IPRB0")
            w.add_to_report()
        assert len(w._report_sections) == 2
        assert w.a_export_session.isEnabled()
        from circuitinsight.gui import view
        html = view.session_report("t", w._report_sections)
        assert html.count("<h2>") == 2
        assert html.count("data:image/png;base64,") == 2
        assert "T@IPRB0" in html
    finally:
        w.close()
        type(w).settings_path = None


def test_csv_traces_and_region_column(qapp, tmp_path):
    from circuitinsight.gui import view

    w = _win(tmp_path, FIX / "ota5t" / "tb_ota5t.cin.json",
             FIX / "ota5t" / "psf")
    try:
        regions = [w.devices.item(i, 2).text()
                   for i in range(w.devices.rowCount())
                   if w.devices.item(i, 1).text() == "mosfet"]
        assert regions and all(r == "sat" for r in regions if r)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            w.solve_sync()
        csv = view.traces_csv(w.result)
        lines = csv.strip().splitlines()
        assert lines[0].startswith("freq_hz,model_db,model_deg")
        assert len(lines) > 100
    finally:
        w.close()
        type(w).settings_path = None


def test_group_toggle_preserves_checks(qapp, tmp_path):
    w = _win(tmp_path, FIX / "ota5t" / "tb_ota5t.cin.json",
             FIX / "ota5t" / "psf")
    try:
        before = set(w.checked_keep())
        assert before                              # auto-setup ticked some
        n = w.keep_tbl.rowCount()
        w.group_chk.setChecked(True)
        assert w.keep_tbl.rowCount() == n
        assert set(w.checked_keep()) == before
    finally:
        w.close()
        type(w).settings_path = None


def test_mode_and_band_shading_persist(qapp, tmp_path):
    from circuitinsight.gui.app import MainWindow

    w = _win(tmp_path, FIX / "ota5t" / "tb_ota5t.cin.json",
             FIX / "ota5t" / "psf")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            w.simplify_sync()
        assert w.result.band_fmin == 1e3           # carried on the Result
        # the shaded band renders as a patch on both axes
        assert len(w.canvas.figure.axes[0].patches) >= 1
        w.mode_combo.setCurrentText("Loop gain")
        w.close()
        w2 = MainWindow()
        try:
            assert w2.mode_combo.currentText() == "Loop gain"
        finally:
            w2.close()
    finally:
        type(w).settings_path = None


def test_impedance_bench_with_xf_truth(qapp, tmp_path):
    """The Impedance bench: any isource/vsource is a port candidate
    (IPORT first); the run's xf result is the overlay truth, and with the
    GUI's matrix cap model the reconstruction tracks it to ~0.01 dB."""
    import numpy as np

    w = _win(tmp_path, FIX / "follower3p" / "tb_follower3p.cin.json",
             FIX / "follower3p" / "psf_z")
    try:
        assert "xf" in w.controller.analyses()
        w.mode_combo.setCurrentText("Impedance")
        assert w.probe_combo.itemText(0) == "IPORT"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            r = w.impedance_sync("IPORT")
        assert abs(r.dc_gain) == pytest.approx(986.8, abs=1.0)
        assert r.h_ref is not None and "xf" in r.ref_label
        err = np.max(np.abs(20 * np.log10(np.abs(r.h) / np.abs(r.h_ref))))
        assert err < 0.05
        assert "Z(IPORT)" in w._history_label(r)
    finally:
        w.close()
        type(w).settings_path = None


def test_modes_bench_overlays_the_runs_stb(qapp, tmp_path):
    """The Modes plot carries the run's own stb curve on the matching
    locus -- simulator truth wherever it exists."""
    w = _win(tmp_path, FIX / "fd" / "tb_fdota_stb.cin.json",
             FIX / "fd" / "psf_dm")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            w.modes_sync("FDPRB.IPRB_DM", "FDPRB.IPRB_CM")
        labels = [ln.get_label() for ln in w.canvas.figure.axes[0].get_lines()]
        assert any("Spectre stb" in ln for ln in labels)
    finally:
        w.close()
        type(w).settings_path = None


def test_latex_aliases_edit_persist_and_render(qapp, tmp_path):
    """Editing the Devices-table LaTeX column aliases a device instance:
    every symbol of that device gets the new subscript, the alias
    persists per-CIN, and it reloads on reopen."""
    from circuitinsight.gui.app import MainWindow

    MainWindow.settings_path = str(tmp_path / "gui.ini")
    w = MainWindow()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            w.open_session(str(FIX / "ota5t" / "tb_ota5t.cin.json"),
                           str(FIX / "ota5t" / "psf"))
        # find MN0's row and set its alias via the table (fires itemChanged)
        row = next(i for i in range(w.devices.rowCount())
                   if w.devices.item(i, 0).text() == "I0.MN0")
        w.devices.item(row, 5).setText("M_1")
        assert w.controller.sym_aliases.get("I0.MN0") == "M_1"
        from circuitinsight.gui import view
        assert view.symbol_tex("gm_I0_MN0",
                               aliases=w.controller.sym_aliases) == "g_{m,M_1}"
        w.close()
        # reopen: the alias comes back from QSettings
        w2 = MainWindow()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            w2.open_session(str(FIX / "ota5t" / "tb_ota5t.cin.json"),
                            str(FIX / "ota5t" / "psf"))
        try:
            assert w2.controller.sym_aliases.get("I0.MN0") == "M_1"
        finally:
            w2.close()
    finally:
        MainWindow.settings_path = None


def test_latex_aliases_device_and_symbol(qapp, tmp_path):
    """Two alias surfaces sharing one store: the Devices 'LaTeX' column
    aliases a whole device (subscript across all its symbols), the keep
    table's 'LaTeX' column overrides ONE symbol and wins over the device
    alias. Both persist per-CIN and only their own column edits."""
    from PySide6.QtCore import Qt

    from circuitinsight.gui.app import MainWindow
    from circuitinsight.gui import view

    MainWindow.settings_path = str(tmp_path / "gui.ini")
    w = MainWindow()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            w.open_session(str(FIX / "ota5t" / "tb_ota5t.cin.json"),
                           str(FIX / "ota5t" / "psf"))
        # only the alias columns are editable
        assert not (w.devices.item(0, 0).flags() & Qt.ItemIsEditable)
        assert w.devices.item(0, 5).flags() & Qt.ItemIsEditable
        assert not (w.keep_tbl.item(0, 0).flags() & Qt.ItemIsEditable)
        assert w.keep_tbl.item(0, 4).flags() & Qt.ItemIsEditable

        # device alias -> remaps every symbol of I0.MN0
        drow = next(i for i in range(w.devices.rowCount())
                    if w.devices.item(i, 0).text() == "I0.MN0")
        w.devices.item(drow, 5).setText("M_1")
        al = w.controller.sym_aliases
        assert al["I0.MN0"] == "M_1"
        assert view.symbol_tex("gm_I0_MN0", aliases=al) == "g_{m,M_1}"
        assert view.symbol_tex("gds_I0_MN0", aliases=al) == "g_{ds,M_1}"

        # per-symbol override on any listed symbol is stored verbatim and
        # wins over a device alias for that exact symbol
        assert w.keep_tbl.rowCount() > 0
        sym = w.keep_tbl.item(0, 0).text()
        w.keep_tbl.item(0, 4).setText(r"\hat{g}")
        assert w.controller.sym_aliases[sym] == r"\hat{g}"
        assert view.symbol_tex(sym, aliases=w.controller.sym_aliases) == r"\hat{g}"

        # persistence across reopen
        w.close()
        w2 = MainWindow()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            w2.open_session(str(FIX / "ota5t" / "tb_ota5t.cin.json"),
                            str(FIX / "ota5t" / "psf"))
        try:
            assert w2.controller.sym_aliases.get("I0.MN0") == "M_1"
            assert w2.controller.sym_aliases.get(sym) == r"\hat{g}"
        finally:
            w2.close()
    finally:
        MainWindow.settings_path = None


def test_impact_ionization_advisory_in_gui(qapp, tmp_path):
    """Opening an II-active run without a gii model surfaces the warning:
    the flagged device is red in the table with a tooltip, and the
    message strip names it."""
    w = _win(tmp_path, FIX / "r2r" / "r2r_bias_pos_loop.cin.json",
             FIX / "r2r" / "psf")
    try:
        assert "MN0" in getattr(w, "_ii", {})
        row = next(i for i in range(w.devices.rowCount())
                   if w.devices.item(i, 0).text() == "MN0")
        assert "impact ionization" in w.devices.item(row, 0).toolTip()
        assert "impact ionization" in w.msg_strip.text()
        assert w.msg_strip.isVisibleTo(w)
    finally:
        w.close()
        type(w).settings_path = None


def test_cap_model_toggle_reopens_and_persists(qapp, tmp_path):
    """The Model-menu charge-matrix toggle: default matrix, flipping to
    lumped re-opens the run with the other model (visible on the follower
    Zout, where the two differ by ~0.7 dB), and the choice persists."""
    import numpy as np

    from circuitinsight.gui.app import MainWindow

    MainWindow.settings_path = str(tmp_path / "gui.ini")
    w = MainWindow()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            w.open_session(str(FIX / "follower3p" / "tb_follower3p.cin.json"),
                           str(FIX / "follower3p" / "psf_z"))
        assert w.controller.cap_model == "matrix"
        assert w.a_matrix_caps.isChecked()
        w.mode_combo.setCurrentText("Impedance")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            r_matrix = w.impedance_sync("IPORT")
        em = np.max(np.abs(20 * np.log10(np.abs(r_matrix.h)
                                         / np.abs(r_matrix.h_ref))))
        # flip to lumped -> re-opens, model changes
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            w.a_matrix_caps.setChecked(False)
        assert w.controller.cap_model == "lumped"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            r_lumped = w.impedance_sync("IPORT")
        el = np.max(np.abs(20 * np.log10(np.abs(r_lumped.h)
                                         / np.abs(r_lumped.h_ref))))
        assert el > em                       # lumped tracks xf worse
        assert em < 0.05 and el > 0.3
        w.close()
        # persistence
        w2 = MainWindow()
        try:
            assert w2.cap_model == "lumped" and not w2.a_matrix_caps.isChecked()
        finally:
            w2.close()
    finally:
        MainWindow.settings_path = None
