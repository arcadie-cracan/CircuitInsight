"""Lump parallel ground-referred passives into single symbols.

Designers do not reason about six separate capacitances at a node; they
reason about C_node. Reaching that form automatically is the achievable
core of "name the demon" (Middlebrook's low-entropy analysis), and it
pairs with the AC-ground scan: GROUNDING is the approximation whose cost
we measure, and it is what CONVERTS cross-coupled elements into
ground-referred ones; LUMPING those is then EXACT -- parallel elements of
the same kind add, with no error at all.

Measured on the µA741: grounding the five recommended bias nodes
(0.018 dB) moves 23 passives from floating to ground-referred and takes
the lumpable-symbol saving from 14 to 32.

Why it matters for the solver: the interpolation grid is a PRODUCT over
kept symbols, so folding five capacitances at a node into one C_node
removes four axes -- an exponential saving -- while making the result
more readable, not less. Elements are lumped PER KIND (capacitances with
capacitances, conductances with conductances): a single primitive cannot
represent a mixed R+C admittance, and pretending otherwise would be the
one place this rewrite could stop being exact.
"""
from __future__ import annotations

import dataclasses
from collections import defaultdict
from dataclasses import dataclass, field

from ..engine.primitives import Primitive

__all__ = ["LumpGroup", "LumpReport", "deactivate", "lump_report",
           "lump_to_ground"]

# kinds whose parallel combination is a single element of the same kind
_LUMPABLE = {"c": "c", "g": "g", "r": "r"}


@dataclass
class LumpGroup:
    """Parallel same-kind elements at one node, and their equivalent."""
    node: str
    kind: str
    members: tuple                 # original instance names
    value: float | None            # equivalent value (None if unknown)
    name: str                      # synthetic instance name

    def describe(self) -> str:
        unit = {"c": "F", "r": "Ω", "g": "S"}.get(self.kind, "")
        val = f" = {self.value:.4g} {unit}" if self.value is not None else ""
        return (f"{self.name}{val}  <- {len(self.members)} elements: "
                f"{', '.join(self.members[:5])}"
                f"{'…' if len(self.members) > 5 else ''}")


@dataclass
class LumpReport:
    groups: list = field(default_factory=list)
    symbols_saved: int = 0

    def describe(self) -> str:
        if not self.groups:
            return "no parallel ground-referred groups to lump"
        lines = [f"lumping: {len(self.groups)} groups, "
                 f"{self.symbols_saved} symbols saved"]
        lines += ["  " + g.describe() for g in self.groups]
        return "\n".join(lines)


def deactivate(primitives, ground):
    """Drop controlled sources that can no longer drive anything, and turn
    self-controlled trans-capacitances into plain capacitors. EXACT.

    This is what makes AC-grounding pay twice. A current-source load whose
    gate is at AC ground has v_gs = 0, so its gm (and gmbs, and every
    trans-capacitance controlled by that gate) contributes exactly zero
    current: the device stops being active and what remains of it -- gds
    and its drain capacitances -- IS a load impedance. Measured on the
    two-stage OTA: grounding vbn kills the transconductances of the
    output-stage load and the tail source, and removing them changes the
    transfer function by 0 dB.

    Returns (primitives, removed)."""
    gnd = set(ground) | {"0"}
    out, removed = [], []
    for p in primitives:
        if p.kind in ("vccs", "cx") and len(p.nodes) == 4:
            d, s_, cp, cn = p.nodes
            dead = (cp in gnd and cn in gnd) or cp == cn
            if dead:
                removed.append(p)
                continue
            if p.kind == "cx" and (cp, cn) == (d, s_):
                # a trans-capacitance controlled by its OWN terminal pair
                # is just a capacitor between them
                out.append(dataclasses.replace(p, kind="c",
                                               nodes=(d, s_)))
                continue
        out.append(p)
    return out, removed


def _groups(primitives, ground, nodes=None, kinds=("c", "g", "r"),
            min_size: int = 2):
    gnd = set(ground) | {"0"}
    want = set(kinds)
    buckets: dict = defaultdict(list)
    for p in primitives:
        if p.kind not in _LUMPABLE or p.kind not in want:
            continue
        a, b = p.nodes
        ga, gb = a in gnd, b in gnd
        if ga == gb:                      # floating, or shorted to itself
            continue
        node = b if ga else a
        if nodes is not None and node not in nodes:
            continue
        buckets[(node, p.kind)].append(p)
    return {k: v for k, v in buckets.items() if len(v) >= min_size}


def _equivalent(kind: str, members) -> float | None:
    vals = [p.value for p in members]
    if any(v is None for v in vals):
        return None
    if kind in ("c", "g"):                # parallel: values add
        return float(sum(vals))
    total = 0.0                           # parallel resistors: conductances add
    for v in vals:
        if v == 0:
            return 0.0
        total += 1.0 / float(v)
    return 1.0 / total if total else None


def lump_report(primitives, ground, *, nodes=None, kinds=("c", "g", "r"),
                min_size: int = 2) -> LumpReport:
    """What would lumping fold together, and how many symbols would that
    save? Pure inspection -- nothing is rewritten."""
    groups = []
    for (node, kind), members in sorted(_groups(primitives, ground, nodes,
                                                kinds, min_size).items()):
        san = str(node).replace(".", "_").replace("!", "")
        groups.append(LumpGroup(
            node=node, kind=kind, members=tuple(sorted(
                f"{p.param or p.kind}_{p.inst}" for p in members)),
            value=_equivalent(kind, members),
            name=f"{kind.upper()}eq_{san}"))
    return LumpReport(groups=groups,
                      symbols_saved=sum(len(g.members) - 1 for g in groups))


def lump_to_ground(primitives, ground, *, nodes=None,
                   kinds=("c", "g", "r"), min_size: int = 2):
    """Replace each group of parallel same-kind ground-referred elements by
    ONE equivalent element, and return (primitives, report).

    EXACT: parallel elements of one kind combine into an element of that
    kind, so the circuit's behaviour is unchanged -- what changes is how
    many symbols describe it. Groups whose members lack numeric values are
    left alone (there would be nothing to put in the equivalent)."""
    rep = lump_report(primitives, ground, nodes=nodes, kinds=kinds,
                      min_size=min_size)
    usable = {(g.node, g.kind): g for g in rep.groups if g.value is not None}
    if not usable:
        return list(primitives), LumpReport(groups=[], symbols_saved=0)

    gnd = set(ground) | {"0"}
    gnd0 = ground[0] if ground else "0"
    absorbed = {m for g in usable.values() for m in g.members}
    out, emitted = [], set()
    for p in primitives:
        tag = f"{p.param or p.kind}_{p.inst}"
        if tag not in absorbed:
            out.append(p)
            continue
        a, b = p.nodes
        node = b if a in gnd else a
        key = (node, p.kind)
        if key in emitted:
            continue                       # already replaced by the equivalent
        emitted.add(key)
        g = usable[key]
        out.append(Primitive(inst=g.name, param="", kind=p.kind,
                             nodes=(node, gnd0), value=g.value))
    kept = LumpReport(groups=[g for g in usable.values()],
                      symbols_saved=sum(len(g.members) - 1
                                        for g in usable.values()))
    return out, kept
