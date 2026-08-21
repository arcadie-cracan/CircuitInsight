"""Layout hints harvested from the ORIGINAL Virtuoso schematic.

The human already placed and routed the circuit; cin_hints.il dumps
that geometry (instance origins, orients, bboxes, terminal positions,
wire polylines, labels) into a `.hints.json` sidecar next to the CIN.
This module loads it and FLATTENS the hierarchy by composing
transforms exactly the way the netlist flattener composes names
(I0.MN0) — so hint geometry joins CIN devices by name.

Coordinates stay in Virtuoso user units, y-up; the composer maps them
to the symbol library's mm grid.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


class HintsError(ValueError):
    pass


#: Virtuoso orient -> 2x2 matrix, y-up convention. R90 is CCW; MX
#: mirrors across the x-axis (y -> -y), MY across the y-axis; the
#: combined forms apply the mirror FIRST, then the rotation (the
#: Virtuoso convention — to be re-verified against the first real
#: harvest, see the plan's open questions).
_R0 = ((1, 0), (0, 1))
_R90 = ((0, -1), (1, 0))
_R180 = ((-1, 0), (0, -1))
_R270 = ((0, 1), (-1, 0))
_MX = ((1, 0), (0, -1))
_MY = ((-1, 0), (0, 1))


def _mul(a, b):
    return (
        (a[0][0] * b[0][0] + a[0][1] * b[1][0],
         a[0][0] * b[0][1] + a[0][1] * b[1][1]),
        (a[1][0] * b[0][0] + a[1][1] * b[1][0],
         a[1][0] * b[0][1] + a[1][1] * b[1][1]),
    )


ORIENTS = {
    "R0": _R0, "R90": _R90, "R180": _R180, "R270": _R270,
    "MX": _MX, "MY": _MY,
    "MXR90": _mul(_R90, _MX), "MYR90": _mul(_R90, _MY),
}


@dataclass
class Transform:
    """xy offset + 2x2 orient matrix; composes parent-first."""
    dx: float = 0.0
    dy: float = 0.0
    m: tuple = _R0

    def apply(self, x: float, y: float) -> tuple:
        return (self.dx + self.m[0][0] * x + self.m[0][1] * y,
                self.dy + self.m[1][0] * x + self.m[1][1] * y)

    def compose(self, child: "Transform") -> "Transform":
        dx, dy = self.apply(child.dx, child.dy)
        return Transform(dx, dy, _mul(self.m, child.m))

    @classmethod
    def of(cls, xy, orient: str) -> "Transform":
        try:
            m = ORIENTS[orient]
        except KeyError:
            raise HintsError(f"unknown orient {orient!r}") from None
        return cls(float(xy[0]), float(xy[1]), m)


@dataclass
class HintInstance:
    name: str
    lib: str
    cell: str
    xy: tuple
    orient: str
    bbox: tuple      # (x0, y0, x1, y1); cin_hints harvests inst~>bBox
    #                  already in the DEFINITION's coordinates
    terms: dict = field(default_factory=dict)   # term -> (x, y), same coords
    m: tuple = _R0   # WORLD orient matrix, filled by flatten()


@dataclass
class HintWire:
    net: str
    points: list


@dataclass
class HintDef:
    bbox: tuple | None
    instances: list
    wires: list
    labels: list                       # {"text", "xy"}
    pins: list                         # {"name", "xy"}


@dataclass
class Hints:
    source: dict
    top: str
    definitions: dict


@dataclass
class FlatHints:
    """The whole schematic in TOP coordinates, names dot-joined."""
    instances: list                    # HintInstance, world coords
    wires: list                        # HintWire, world coords
    labels: list
    pins: list = field(default_factory=list)   # {"name", "xy"}: the
                                       # page's own ports (top only)


def load_hints(path: str | Path) -> Hints:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    ver = d.get("hints_version")
    if ver != "0.1":
        raise HintsError(f"unsupported hints_version {ver!r}")
    defs = {}
    for name, dd in d.get("definitions", {}).items():
        defs[name] = HintDef(
            bbox=tuple(dd["bbox"]) if dd.get("bbox") else None,
            instances=[HintInstance(
                name=i["name"], lib=i.get("lib", ""), cell=i.get("cell", ""),
                xy=tuple(i["xy"]), orient=i.get("orient", "R0"),
                bbox=tuple(i["bbox"]),
                terms={k: tuple(v) for k, v in (i.get("terms") or {}).items()})
                for i in dd.get("instances", [])],
            wires=[HintWire(net=w.get("net", ""),
                            points=[tuple(p) for p in w["points"]])
                   for w in dd.get("wires", [])],
            labels=list(dd.get("labels", [])),
            pins=list(dd.get("pins", [])),
        )
    return Hints(source=d.get("source", {}), top=d["top"], definitions=defs)


def _xform_bbox(t: Transform, bb) -> tuple:
    (x0, y0), (x1, y1) = (bb[0], bb[1]), (bb[2], bb[3])
    pts = [t.apply(x, y) for x in (x0, x1) for y in (y0, y1)]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return (min(xs), min(ys), max(xs), max(ys))


def flatten(h: Hints, descend=None) -> FlatHints:
    """Walk from the top, descending into instances whose cell is a
    harvested definition; geometry lands in top coordinates, names
    join with '.' exactly like the netlist flattener.

    descend: cells to treat as subcircuits. Pass the CIN's definition
    names: the harvester also dumps DRAWING cells (analogLib vdd/gnd
    have schematic views!), and inlining those would replace the rail
    glyph with its internal strokes. None (the default) descends into
    every harvested definition — the pre-calibration behavior."""
    out = FlatHints(instances=[], wires=[], labels=[],
                    pins=[dict(pn) for pn in h.definitions[h.top].pins]
                    if h.top in h.definitions else [])

    def walk(defname: str, prefix: str, t: Transform):
        try:
            dd = h.definitions[defname]
        except KeyError:
            raise HintsError(f"definition {defname!r} missing") from None
        for w in dd.wires:
            out.wires.append(HintWire(
                net=w.net, points=[t.apply(*p) for p in w.points]))
        for lab in dd.labels:
            out.labels.append({"text": lab.get("text", ""),
                               "xy": t.apply(*lab["xy"])})
        for inst in dd.instances:
            name = prefix + inst.name
            if inst.cell in h.definitions and (
                    descend is None or inst.cell in descend):
                child = Transform.of(inst.xy, inst.orient)
                walk(inst.cell, name + ".", t.compose(child))
                continue
            # a mirrored PARENT mirrors its children too: the world
            # orientation is the composed matrix, not the local string
            leaf = t.compose(Transform.of(inst.xy, inst.orient))
            out.instances.append(HintInstance(
                name=name, lib=inst.lib, cell=inst.cell,
                xy=(leaf.dx, leaf.dy), orient=inst.orient,
                bbox=_xform_bbox(t, inst.bbox),
                terms={k: t.apply(*v) for k, v in inst.terms.items()},
                m=leaf.m))

    walk(h.top, "", Transform())
    return out


def page(h: Hints, defname: str) -> FlatHints:
    """ONE definition as a page, in its own coordinates — the primary
    view: the schematic is hierarchical like its source, subcircuit
    instances are drawn as blocks (their harvested symbol bbox and pin
    positions) and navigated into, never inlined. flatten() remains
    for the flat engine-side view."""
    try:
        dd = h.definitions[defname]
    except KeyError:
        raise HintsError(f"definition {defname!r} missing") from None
    insts = [HintInstance(name=i.name, lib=i.lib, cell=i.cell, xy=i.xy,
                          orient=i.orient, bbox=i.bbox,
                          terms=dict(i.terms),
                          m=Transform.of(i.xy, i.orient).m)
             for i in dd.instances]
    return FlatHints(instances=insts,
                     wires=[HintWire(w.net, list(w.points))
                            for w in dd.wires],
                     labels=[dict(lab) for lab in dd.labels],
                     pins=[dict(pn) for pn in dd.pins])
