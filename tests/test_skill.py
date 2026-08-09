"""The Virtuoso SKILL (.il) files ship with the package and are locatable.

Locks the M9 packaging contract: `pip install circuitinsight` (even directly
from GitHub) must deliver cin_export.il / cin_launch.il so the Virtuoso side of
the flow needs no separate clone.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

from circuitinsight import skill

# the subprocess does not inherit conftest's sys.path insert, so on a
# checkout without a pip install the import fails before the packaging
# contract is even tested -- hand it the same src/ the suite uses
_ENV = dict(os.environ)
_ENV["PYTHONPATH"] = (str(Path(__file__).resolve().parents[1] / "src")
                      + os.pathsep + _ENV.get("PYTHONPATH", ""))


@pytest.mark.parametrize("name", ["cin_export.il", "cin_launch.il",
                                  "cin_xprobe.il", "cin_init.il"])
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
        capture_output=True, text=True, check=True, env=_ENV).stdout.strip()
    assert out == str(skill.skill_dir())


def test_skill_import_pulls_no_heavy_gui_deps():
    """Locating the .il files must not drag in Qt (core stays headless)."""
    code = (
        "import sys, circuitinsight.skill as s; s.skill_dir(); "
        "assert 'PySide6' not in sys.modules and 'PyQt5' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True, env=_ENV)


def test_bootstrap_loads_every_other_il_file():
    """cin_init.il is advertised as THE one line for .cdsinit, so it must load
    every sibling. It once omitted cin_xprobe.il, and the only symptom was the
    GUI reporting "cross-probe unavailable: connected, but CInHighlight is not
    defined" -- a silent feature loss, since the export and launch paths it
    did load worked fine."""
    init = skill.path("cin_init.il").read_text(encoding="utf-8")
    siblings = sorted(p.name for p in skill.skill_dir().glob("cin_*.il")
                      if p.name != "cin_init.il")
    assert siblings                                        # the glob works
    missing = [n for n in siblings if n not in init]
    assert not missing, f"cin_init.il never loads: {missing}"


def test_ade_menu_installs_by_scanning_windows_not_matching_names():
    """The Launch-menu regression: the session name handed to the creation
    callback never matches what axlGetWindowSession reports on that path,
    so a name-matched install retried into the void and the menu only
    appeared when the view was opened directly. The callback must route
    through the janitor (window scan), and both entry points must kick it
    -- session creation for new windows, load time for windows that
    already exist when the files are (re)loaded."""
    launch = skill.path("cin_launch.il").read_text(encoding="utf-8")
    init = skill.path("cin_init.il").read_text(encoding="utf-8")

    assert "defun CInMenuJanitor" in launch
    cb = launch.split("defun CInAdeSessionCb", 1)[1]
    assert "CInMenuJanitor" in cb.split("defun", 1)[0]
    assert "CInInstallAdeMenuFor(" not in cb.split("defun", 1)[0]
    assert "CInMenuJanitor()" in init                  # the load-time kick


def test_skill_files_have_balanced_parens():
    """No Virtuoso runs in CI, so at least guarantee the .il files parse at
    the paren level -- an unbalanced edit otherwise surfaces as a cryptic
    load error in the user's CIW."""
    for p in sorted(skill.skill_dir().glob("*.il")):
        depth = 0
        in_str = in_comment = False
        prev = ""
        for ch in p.read_text(encoding="utf-8"):
            if in_comment:
                in_comment = ch != "\n"
            elif in_str:
                if ch == '"' and prev != "\\":
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == ";":
                in_comment = True
            elif ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                assert depth >= 0, f"{p.name}: stray closing paren"
            prev = ch
        assert depth == 0, f"{p.name}: {depth} unclosed parens"
        assert not in_str, f"{p.name}: unterminated string"
