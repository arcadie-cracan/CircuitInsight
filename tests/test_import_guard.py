"""Independence contract, fixture-free — runs in every checkout including the
public snapshot (which withholds the PDK-derived fixtures and their tests)."""
import os
import subprocess
import sys
from pathlib import Path


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
    # the subprocess does not inherit conftest's sys.path insert, so on
    # a checkout without a pip install the import fails before the
    # contract is even tested — hand it the same src/ the suite uses
    env = dict(os.environ)
    src = str(Path(__file__).resolve().parents[1] / "src")
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    r = subprocess.run([sys.executable, "-c", code],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 0, (
        f"core/session imported GUI/Cadence modules: {r.stdout.strip()} "
        f"{r.stderr.strip()}")


def test_non_gui_test_modules_never_touch_the_gui():
    """The gate venv has no Qt, and only the GUI test modules carry the
    importorskip header that turns that into a SKIP. A gui import that
    sneaks into any other test file passes every dev-machine run and
    then fails the fresh-venv gate half an hour in — this static check
    catches it in milliseconds, at authoring time. (Field-tested: a
    cancel-class assertion in an engine test did exactly that.)"""
    import re
    from pathlib import Path

    # gui.view is deliberately Qt-free ("No Qt" is its contract) and fair
    # game everywhere; every OTHER gui module pulls PySide6 on import
    bad = re.compile(
        r"circuitinsight\.gui\.(?!view\b)\w+"
        r"|from\s+circuitinsight\.gui\s+import\s+(?!view\b)")
    tests = Path(__file__).resolve().parent
    offenders = []
    for f in sorted(tests.glob("test_*.py")):
        if f.name == Path(__file__).name:
            continue                     # this file names the pattern
        src = f.read_text(encoding="utf-8")
        skips = ('importorskip("PySide6")' in src
                 or "importorskip('PySide6')" in src)
        if skips:
            continue
        if bad.search(src):
            offenders.append(f.name)
    assert not offenders, (
        f"non-GUI test modules import Qt-bearing gui modules (breaks "
        f"the Qt-less gate venv): {offenders} — move the assertion to a "
        f"GUI test file, use gui.view, or add the PySide6 importorskip "
        f"header")
