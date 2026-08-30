"""Session states: selections always restore; solutions only under a
matching fingerprint (gui/state.py + the MainWindow wiring)."""
import shutil
import warnings
from pathlib import Path

import pytest

pytest.importorskip("PySide6")
try:
    import PySide6.QtWidgets  # noqa: F401
except Exception as exc:
    pytest.skip(f"PySide6.QtWidgets not loadable: {exc}",
                allow_module_level=True)
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

FIX = Path(__file__).resolve().parent / "fixtures" / "spectre" / "ota5t"


@pytest.fixture(scope="module")
def qapp():
    import circuitinsight.gui.exprweb  # noqa: F401
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _copy_fixture(tmp_path):
    cin = tmp_path / "tb_ota5t.cin.json"
    shutil.copy(FIX / "tb_ota5t.cin.json", cin)
    shutil.copytree(FIX / "psf", tmp_path / "psf")
    return cin, tmp_path / "psf"


def test_state_file_round_trip_and_fingerprint_gate(tmp_path):
    """The two-layer contract at file level: the manifest always loads;
    the pickled payload only under the expected fingerprint."""
    from circuitinsight.gui import state as st

    cin, psf = _copy_fixture(tmp_path)
    fp = st.fingerprint(cin, psf, "matrix", [("a", "b")], "as imported")
    # deterministic for the same inputs, sensitive to each ingredient
    assert fp == st.fingerprint(cin, psf, "matrix", [("a", "b")],
                                "as imported")
    assert fp != st.fingerprint(cin, psf, "lumped", [("a", "b")],
                                "as imported")

    path = st.state_path(cin, "test")
    st.save_state(path, {"keep": ["gm_x"], "fingerprint": fp},
                  result={"payload": 42})
    man, res, stale = st.load_state(path, fp)
    assert man["keep"] == ["gm_x"] and res == {"payload": 42}
    assert not stale
    man2, res2, stale2 = st.load_state(path, "different")
    assert man2["keep"] == ["gm_x"] and res2 is None and stale2


def test_state_restores_selections_and_gated_solution(qapp, tmp_path):
    """End to end: solve with ticks (autosave fires), wreck the
    selections, restore — ticks, band and the SOLUTION come back
    because the fingerprint still matches; flip the cap model and the
    same file restores selections only."""
    from PySide6.QtCore import Qt
    from circuitinsight.gui import state as st
    from circuitinsight.gui.app import MainWindow

    cin, psf = _copy_fixture(tmp_path)
    MainWindow.settings_path = str(tmp_path / "gui.ini")
    try:
        win = MainWindow()
    finally:
        MainWindow.settings_path = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        win.open_session(str(cin), str(psf))
        win.in_combo.setCurrentText("VIND")
        win.out_combo.setCurrentText("vout")
        win._rank()
        win.keep_tbl.item(0, 0).setCheckState(Qt.Checked)
        ticks = win.checked_keep()
        r = win.solve_sync()
    try:
        assert st.state_path(str(cin)).exists(), "autosave after _show"

        # wreck the session state, then restore the rolling last-state
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            win._rank()
        for i in range(win.keep_tbl.rowCount()):
            win.keep_tbl.item(i, 0).setCheckState(Qt.Unchecked)
        win.band_slider.setValues(1e5, 1e6)
        win.result = None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            win._load_state_file(None)
        assert win.checked_keep() == ticks
        assert win.result is not None and win.result.keep == r.keep
        assert "solution" in win.msg_strip.text()

        # a named state survives; a cap-model change makes it stale
        named = st.state_path(str(cin), "chk")
        st.save_state(named, win._state_manifest(), win.result)
        win.cap_model = "lumped"          # fingerprint ingredient changes
        win.result = None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            win._load_state_file(named)
        assert win.checked_keep() == ticks     # selections still restore
        assert win.result is None              # solution refused as stale
        assert "stale" in win.msg_strip.text()
    finally:
        win.close()


def test_reopen_restores_the_ticks_by_default(qapp, tmp_path):
    """Policy (2026-08-29): the SELECTIONS come back on reopen without
    a click -- the auto plan fills only a first-ever open -- while the
    solution stays behind File -> Restore last state. A saved set that
    matches nothing in the model (a mangled state) yields to the auto
    plan."""
    from PySide6.QtCore import Qt

    from circuitinsight.gui import state as st
    from circuitinsight.gui.app import MainWindow

    cin, psf = _copy_fixture(tmp_path)
    MainWindow.settings_path = str(tmp_path / "gui.ini")
    try:
        win = MainWindow()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            win.open_session(str(cin), str(psf))
        auto = set(win.checked_keep())     # the plan (may be empty here)
        assert win.keep_tbl.rowCount() >= 2
        # curate: keep exactly two symbols of our own choosing
        mine = [win.keep_tbl.item(r, 0).text() for r in range(2)]
        for row in range(win.keep_tbl.rowCount()):
            it = win.keep_tbl.item(row, 0)
            it.setCheckState(Qt.Checked if it.text() in mine
                             else Qt.Unchecked)
        win._autosave_state()
        win.close()

        win2 = MainWindow()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            win2.open_session(str(cin), str(psf))
        assert set(win2.checked_keep()) == set(mine), \
            "the curated ticks return, not the auto plan"
        win2.close()

        # a mangled state (symbols from another model) yields to the plan
        p = st.state_path(str(cin))
        import json
        import zipfile
        with zipfile.ZipFile(p) as z:
            manifest = json.loads(z.read("manifest.json"))
        manifest["keep"] = ["gm_NOPE_1", "cgs_NOPE_2"]
        st.save_state(p, manifest)
        win3 = MainWindow()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            win3.open_session(str(cin), str(psf))
        assert set(win3.checked_keep()) == auto, \
            "a state matching nothing yields to the auto plan"
        win3.close()
    finally:
        MainWindow.settings_path = None
