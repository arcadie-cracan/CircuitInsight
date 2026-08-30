"""Operating-point SET discovery: one topology, many OPs.

A maestro corner or parametric run (ADE Explorer/Assembler) writes one
results directory PER DESIGN POINT under `ExplorerRun.N/<point>/<test>/psf`,
each a complete psf directory -- dcOpInfo, dcOp, stb when enabled -- plus
`runObjFile`, whose ASCII header names the point: `"#maeCorner" "ff -40"`
and `"#maeDesignPoint" 1`. Swept design variables (a parametric run, or
maestro's temperature axis) sit in `variables_file` as name/value pairs.

`discover_op_points` turns any of these into a labeled point list of the
simulator-NEUTRAL adapters.points.OpPoint:
- a plain psf directory (dcOpInfo.info present)  -> one unlabeled point;
- a <point>/<test> tree or an ExplorerRun root   -> one point per design
  point, labeled by corner and non-default variables.

The maestro layout and runObjFile grammar are Cadence conventions and
stay HERE; the session sees only OpPoints.
"""
from __future__ import annotations

import re
from pathlib import Path

from ..points import OpPoint


def _read_runobj(psf: Path) -> tuple[str, int | None]:
    try:
        head = (psf / "runObjFile").read_text(errors="replace")[:4000]
    except OSError:
        return "", None
    corner = re.search(r'"#maeCorner"\s+"([^"]*)"', head)
    point = re.search(r'"#maeDesignPoint"\s+(\d+)', head)
    return (corner.group(1) if corner else "",
            int(point.group(1)) if point else None)


def _read_variables(psf: Path) -> dict:
    try:
        text = (psf / "variables_file").read_text(errors="replace")[:20000]
    except OSError:
        return {}
    out: dict = {}
    for m in re.finditer(r'"([^"]+)"\s+"variable"\s+\(\s*([-+0-9.eE]+)\s*\)',
                         text):
        try:
            out[m.group(1)] = float(m.group(2))
        except ValueError:
            pass
    return out


def _read_toolargs_corner(psf: Path) -> str:
    """The corner name as the USER defined it, from the netlister's
    toolArgs.ils record. The leaf runObjFile is not always faithful: a
    maestro corner run recorded "#maeCorner" "nom" for every 125C skew
    point (fc field case) while toolArgs.ils kept "ff 125"; when
    present, this name wins."""
    try:
        text = (psf.parent / "netlist" / "toolArgs.ils").read_text(
            errors="replace")[:20000]
    except OSError:
        return ""
    m = re.search(r'"CornerName"\s+"([^"]+)"', text)
    return m.group(1) if m else ""


def _read_sections(psf: Path) -> str:
    """The model-library section(s) the point simulates with. A corner
    named \"nom\" can hide a skewed process behind a nominal-looking name;
    the netlist's `section=` include is the ground truth. The netlist dir
    sits beside psf in the same test directory."""
    nl = psf.parent / "netlist"
    secs: list[str] = []
    for fname in (".modelFiles", "ade_e.scs"):
        try:
            text = (nl / fname).read_text(errors="replace")[:20000]
        except OSError:
            continue
        for m in re.finditer(r'section\s*=\s*"?(\w+)"?', text):
            if m.group(1) not in secs:
                secs.append(m.group(1))
        if secs:
            break
    return "/".join(secs)


def _point_psfs(child: Path):
    """The psf dirs under one design-point directory."""
    for psf in sorted(child.glob("*/psf")):
        if (psf / "dcOpInfo.info").exists():
            yield psf


def discover_op_points(root) -> list[OpPoint]:
    """The operating-point set at `root` (see module docstring). A plain
    psf directory yields exactly one point, so a caller can treat every
    open uniformly."""
    root = Path(root)
    if (root / "dcOpInfo.info").exists():          # a plain psf dir
        corner, idx = _read_runobj(root)
        return [OpPoint(index=idx or 1, psf=root, corner=corner,
                        process=_read_sections(root),
                        variables=_read_variables(root))]
    pts: list[OpPoint] = []
    numbered = [c for c in root.iterdir()
                if c.is_dir() and c.name.isdigit()] if root.is_dir() else []
    for child in sorted(numbered, key=lambda c: int(c.name)):
        for psf in _point_psfs(child):
            corner, idx = _read_runobj(psf)
            corner = _read_toolargs_corner(psf) or corner
            pts.append(OpPoint(index=idx if idx is not None
                               else int(child.name),
                               psf=psf, corner=corner,
                               process=_read_sections(psf),
                               variables=_read_variables(psf)))
    if not pts and root.is_dir():
        # a <test> directory holding one psf ("<root>/psf")
        for psf in _point_psfs(root):
            corner, idx = _read_runobj(psf)
            corner = _read_toolargs_corner(psf) or corner
            pts.append(OpPoint(index=idx or 1, psf=psf, corner=corner,
                               process=_read_sections(psf),
                               variables=_read_variables(psf)))
    return pts
