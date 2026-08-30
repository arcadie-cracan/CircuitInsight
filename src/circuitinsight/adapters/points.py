"""The simulator-NEUTRAL operating-point set.

One CIN topology, many operating points: PVT corners, parametric sweep
points, transient snapshots -- values change, connectivity never does.
`OpPoint` is the neutral currency: a label, an opaque results location
the owning adapter knows how to open, and whatever variables identify
the point. Each adapter contributes its own discovery (how a multi-point
run is laid out on disk is a simulator convention: maestro's
ExplorerRun tree for Spectre, a directory of raw files for ngspice, ...)
and returns these; the session and the GUI never see anything
simulator-shaped.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class OpPoint:
    """One simulated operating point of the shared topology."""
    index: int                       # the run's own point number
    psf: Path                        # results location, adapter-owned
    corner: str = ""                 # corner name when the run has one
    process: str = ""                # model-library section when known
    variables: dict = field(default_factory=dict)   # swept values

    @property
    def label(self) -> str:
        if self.corner:
            # a corner name sometimes embeds its process and temperature
            # ("ff -40") and sometimes neither ("nom", four times over,
            # each a different model section at 125C): append whichever
            # the name does not already carry
            parts = [self.corner]
            if self.process and self.process not in self.corner:
                parts.append(self.process)
            t = self.variables.get("temperature",
                                   self.variables.get("temp"))
            if t is not None and f"{t:g}" not in self.corner:
                parts.append(f"{t:g}")
            return " ".join(parts)
        vs = {k: v for k, v in self.variables.items()
              if k not in ("temp", "temperature")}
        if vs:
            return ", ".join(f"{k}={v:g}" for k, v in sorted(vs.items()))
        t = self.variables.get("temperature")
        return (f"point {self.index}" + (f" ({t:g}C)" if t is not None
                                         else ""))


def load_points_manifest(path) -> list[OpPoint]:
    """OpPoints from a POINTS MANIFEST -- a small JSON file a
    simulator-side launcher writes from the tool's OWN results database
    (for Cadence, axlReadHistoryResDB: the official corner names, which
    the on-disk runObjFile does not always carry faithfully):

        {"cin_points_version": "0.1", "history": "...",
         "points": [{"corner": "ff 125", "psf": "/abs/or/relative"}, ...]}

    `index`, `process` and `variables` are optional per entry. Relative
    psf paths resolve against the manifest's directory. Entries whose
    psf carries no OP data are skipped -- a launcher lists what the run
    attempted, the session needs what it can open. The format names no
    simulator, so any tool's flow can write one."""
    import json

    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    pts: list[OpPoint] = []
    for i, e in enumerate(data.get("points", []), start=1):
        psf = Path(e["psf"])
        if not psf.is_absolute():
            psf = path.parent / psf
        if not (psf / "dcOpInfo.info").exists():
            continue
        pts.append(OpPoint(index=int(e.get("index", i)), psf=psf,
                           corner=str(e.get("corner", "")),
                           process=str(e.get("process", "")),
                           variables=dict(e.get("variables", {}))))
    return pts
