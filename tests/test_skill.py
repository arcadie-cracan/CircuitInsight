"""The Virtuoso SKILL (.il) files ship with the package and are locatable.

Locks the M9 packaging contract: `pip install circuitinsight` (even directly
from GitHub) must deliver cin_export.il / cin_launch.il so the Virtuoso side of
the flow needs no separate clone.
"""
import subprocess
import sys

import pytest

from circuitinsight import skill


@pytest.mark.parametrize("name", ["cin_export.il", "cin_launch.il", "cin_init.il"])
def test_skill_file_shipped(name):
    p = skill.path(name)
    assert p.exists() and p.suffix == ".il"
    assert p.parent == skill.skill_dir()
    assert p.read_text(encoding="utf-8").strip()          # non-empty


def test_missing_file_errors_clearly():
    with pytest.raises(FileNotFoundError) as ei:
        skill.path("does_not_exist.il")
    assert "cin_export.il" in str(ei.value)                # lists what IS available


def test_module_prints_dir():
    out = subprocess.run(
        [sys.executable, "-m", "circuitinsight.skill"],
        capture_output=True, text=True, check=True).stdout.strip()
    assert out == str(skill.skill_dir())


def test_skill_import_pulls_no_heavy_gui_deps():
    """Locating the .il files must not drag in Qt (core stays headless)."""
    code = (
        "import sys, circuitinsight.skill as s; s.skill_dir(); "
        "assert 'PySide6' not in sys.modules and 'PyQt5' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
