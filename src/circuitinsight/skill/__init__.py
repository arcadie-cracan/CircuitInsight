"""Cadence Virtuoso SKILL files shipped with CircuitInsight.

These ``.il`` files implement the schematic->CIN exporter (``cin_export.il``)
and the one-click GUI launcher (``cin_launch.il``). They install alongside the
package so ``pip install circuitinsight`` -- even directly from GitHub --
delivers them; no separate clone is needed for the Virtuoso integration.

Find them from a shell::

    circuitinsight-skill-path            # prints the directory holding the .il files

or from Python (e.g. to build the CIW ``load()`` lines)::

    from circuitinsight.skill import skill_dir, path
    skill_dir()                          # -> Path to the directory
    path("cin_export.il")                # -> Path to one file
"""
from __future__ import annotations

from pathlib import Path

__all__ = ["skill_dir", "path"]


def skill_dir() -> Path:
    """Absolute path to the directory containing the SKILL (.il) files.

    The files sit next to this module, so this resolves correctly for both
    editable installs (source tree) and wheel/``pip install`` layouts.
    """
    return Path(__file__).resolve().parent


def path(name: str) -> Path:
    """Absolute path to a named SKILL file, e.g. ``path("cin_export.il")``."""
    p = skill_dir() / name
    if not p.exists():
        have = ", ".join(sorted(q.name for q in skill_dir().glob("*.il"))) or "none"
        raise FileNotFoundError(
            f"no SKILL file {name!r} in {skill_dir()} (have: {have})")
    return p


def _main() -> None:
    """Console script ``circuitinsight-skill-path``: print the skill directory."""
    print(skill_dir())
