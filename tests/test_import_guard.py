"""Independence contract, fixture-free — runs in every checkout including the
public snapshot (which withholds the PDK-derived fixtures and their tests)."""
import subprocess
import sys


def test_core_imports_no_gui_or_cadence():
    """A clean interpreter importing the core must not pull in Qt or the
    Virtuoso/Cadence integration layer."""
    code = (
        "import circuitinsight, circuitinsight.session, sys\n"
        "bad = [m for m in sys.modules if m.startswith(("
        "'PySide6','PyQt5','PyQt6','PyQt','shiboken','skillbridge','cadence'))]\n"
        "print(bad)\n"
        "sys.exit(1 if bad else 0)\n"
    )
    r = subprocess.run([sys.executable, "-c", code],
                       capture_output=True, text=True)
    assert r.returncode == 0, (
        f"core/session imported GUI/Cadence modules: {r.stdout.strip()} "
        f"{r.stderr.strip()}")
