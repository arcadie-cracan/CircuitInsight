"""Structure- and OP-driven netlist reduction.

A pipeline of topology transforms over a `FlatCircuit`, applied after the OP
join and before the small-signal build. Each transform returns a new (immutable)
`FlatCircuit` plus human-readable notes -- a component that silently vanished
would be worse than a crash, so every change is reported.

Implemented
-----------
`drop_unmatched_passives` -- remove passives the simulator pruned. A device with
no OP record is (in every case seen) a 0-valued passive the simulator optimized
away. Reflect that *electrically*, per type:

  * resistor / inductor  -> a 0 value is a SHORT: merge its two nets into one.
  * capacitor            -> a 0 value is an OPEN: delete it.

Active devices (mosfet/bjt/diode) are never dropped here -- a missing transistor
OP is a real error, left to the caller.

Planned (the pipeline is the point)
-----------------------------------
series/parallel merge of identical passives; parallel-device (bussed-transistor)
combining. Each is a transform of the same shape: `(flat, ...) -> (flat, notes)`.
"""
from __future__ import annotations

from dataclasses import replace

from .adapters.cin import FlatCircuit, FlatDevice

_SHORT_IF_MISSING = frozenset({"resistor", "inductor"})   # 0-value -> short
_OPEN_IF_MISSING = frozenset({"capacitor"})               # 0-value -> open
REDUCIBLE_IF_MISSING = _SHORT_IF_MISSING | _OPEN_IF_MISSING


def _value_note(d: FlatDevice) -> str:
    v = d.params.get("r", d.params.get("c", d.params.get("l")))
    return f" (value {v})" if v is not None else ""


def drop_unmatched_passives(
    flat: FlatCircuit, missing: list[FlatDevice],
    cause: str = "no OP data -- simulator pruned it",
) -> tuple[FlatCircuit, list[str]]:
    """Reduce zero-valued passives. Returns (new_flat, notes).

    A passive is 0-valued either because the simulator pruned it (no OP record,
    the default `cause`) or because its OP record carries value 0 (some
    simulators keep the device instead of pruning it -- pass that `cause`).
    Either way the reduction is the same: R/L merge their nets (short); C is
    deleted (open). Nets are merged with a union-find so a chain of shorts
    collapses correctly in one pass; ground always wins as the surviving net.
    """
    ground = set(flat.ground)
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        root = x
        while parent.get(root, root) != root:
            root = parent[root]
        while parent.get(x, x) != x:          # path compression
            parent[x], x = root, parent[x]
        return root

    def union(a: str, b: str) -> tuple[str, str]:
        ra, rb = find(a), find(b)
        if ra == rb:
            return ra, rb
        # survivor: ground first, then lexicographic (stable)
        if rb in ground or (ra not in ground and rb < ra):
            ra, rb = rb, ra
        parent[rb] = ra
        return ra, rb

    notes: list[str] = []
    dropped: set[str] = set()
    for d in missing:
        vtxt = _value_note(d)
        if d.device_type in _SHORT_IF_MISSING:
            p, n = d.terminals.get("p"), d.terminals.get("n")
            if p is not None and n is not None and find(p) != find(n):
                keep, gone = union(p, n)
                notes.append(f"{d.name}: {cause}{vtxt}; "
                             f"treating as a SHORT ({gone} = {keep}).")
            else:
                notes.append(f"{d.name}: {cause}{vtxt} -- treating as a short "
                             f"(terminals already one net).")
            dropped.add(d.name)
        elif d.device_type in _OPEN_IF_MISSING:
            notes.append(f"{d.name}: {cause}{vtxt}; "
                         f"treating as an OPEN (removed).")
            dropped.add(d.name)
        # anything else: not ours to drop; caller already errored

    if not dropped and not parent:
        return flat, notes

    devs = tuple(
        replace(d, terminals={t: find(net) for t, net in d.terminals.items()})
        for d in flat.devices if d.name not in dropped
    )
    nets = frozenset(find(x) for x in flat.nets)
    grd = tuple(dict.fromkeys(find(g) for g in flat.ground))   # dedup, keep order
    return FlatCircuit(devs, nets, grd), notes
