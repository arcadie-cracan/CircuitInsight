"""Passive networks as single equivalent admittances.

The "name the demon" step of design-oriented analysis: a passive
subnetwork is not k elements, it is ONE driving-point admittance
Y_eq(s), and calling it by one name is what keeps an answer readable
(Middlebrook's low-entropy form) as well as cheap -- the interpolation
grid is a product over kept symbols, so k axes become one.

What this module does:

* `passive_chunks` finds maximal connected groups of passive elements
  and classifies their nodes. A node is a TERMINAL when anything else
  touches it -- an active device, a source, the probe, the requested
  output -- and INTERNAL otherwise. Internal nodes are what a chunk can
  eliminate; terminals are what it must preserve.
* `chunk_admittance` computes the exact rational Y_eq(s) of a
  two-terminal chunk, symbolic in its members' own symbols, by solving
  the chunk ALONE with a unit current injected across its terminals.
  Internal nodes are eliminated by that solve, so the result is exact
  however tangled the interior is.

Measured reality check (2026-07-28): on the ota5t / miller / fd / µA741
fixtures NO node is touched only by passives -- every node has an active
device on it -- so there is nothing for internal-node elimination to
eliminate. The passive structures that actually occur are two-terminal
groups between active nodes, and the interesting ones are MIXED kinds
(fd's RCMA ∥ CCMA, a resistor and a capacitor across the same pair)
whose Y_eq = 1/R + sC is not expressible as any single primitive. That
is the honest scope of this module today: it names and prices those
networks exactly, and reports what collapsing them would save. Actually
CARRYING one opaque Y symbol through a solve needs an admittance
primitive in the engine (the "substitute late" step of
docs/interp-speedup-plan.md S-E) -- deliberately not faked here.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import sympy as sp

from ..engine.mna import build_mna, solve_tf
from ..engine.primitives import Primitive

__all__ = ["PassiveChunk", "passive_chunks", "chunk_admittance"]

_PASSIVE = {"r", "g", "c", "l"}
_PROBE = "__chunk_probe"


@dataclass
class PassiveChunk:
    """One connected passive subnetwork."""
    members: tuple                       # Primitive objects
    terminals: tuple                     # nodes the rest of the circuit sees
    internal: tuple = ()                 # nodes only this chunk touches
    kinds: tuple = ()

    @property
    def names(self) -> tuple:
        return tuple(sorted(p.inst for p in self.members))

    @property
    def symbols_saved(self) -> int:
        """Symbols removed by naming the chunk once (2-terminal only)."""
        return max(0, len(self.members) - 1) if len(self.terminals) == 2 else 0

    def describe(self) -> str:
        t = " - ".join(self.terminals) if self.terminals else "(floating)"
        inner = f", {len(self.internal)} internal" if self.internal else ""
        return (f"{t}: {len(self.members)} elements "
                f"{{{', '.join(self.names)}}} kinds={'/'.join(self.kinds)}"
                f"{inner}")


def passive_chunks(primitives, ground, *, keep_terminals=(),
                   min_size: int = 2, explicit_only: bool = True) -> list:
    """Maximal connected passive subnetworks, with terminals classified.

    `keep_terminals`: extra nodes that must stay visible (the requested
    output, a probe pair) even if only passives touch them -- an iprobe
    inside a network SPLITS it, and the output must remain observable.

    `explicit_only` (the default): consider only NETLIST passives, not the
    r/c elements of device models. A transistor's gds and cpi are passive
    by stamp but they are OP parameters -- folding them into an anonymous
    Y_eq would erase exactly the design information this tool exists to
    show. Measured consequence of not doing this: the chunker happily
    grouped six devices' substrate capacitances at vdd! into one
    "network"."""
    gnd = set(ground) | {"0"}
    active_nodes = set(keep_terminals)
    for p in primitives:
        if p.kind not in _PASSIVE:
            active_nodes |= {n for n in p.nodes if n not in gnd}

    # union-find over passive elements sharing a NON-terminal node
    parent: dict = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    passives = [p for p in primitives if p.kind in _PASSIVE
                and (not explicit_only or not p.param)]
    by_node: dict = defaultdict(list)
    for i, p in enumerate(passives):
        parent.setdefault(i, i)
        for n in p.nodes:
            if n not in gnd:
                by_node[n].append(i)
    for node, ids in by_node.items():
        # elements meeting at an INTERNAL node are one network; meeting at a
        # terminal they are merely both connected to the rest of the circuit
        if node in active_nodes:
            continue
        for j in ids[1:]:
            union(ids[0], j)
    # elements sharing BOTH terminals are parallel: one network too
    pairs: dict = defaultdict(list)
    for i, p in enumerate(passives):
        pairs[frozenset(p.nodes)].append(i)
    for ids in pairs.values():
        for j in ids[1:]:
            union(ids[0], j)

    groups: dict = defaultdict(list)
    for i in range(len(passives)):
        groups[find(i)].append(passives[i])

    out = []
    for members in groups.values():
        if len(members) < min_size:
            continue
        nodes = {n for p in members for n in p.nodes}
        term = tuple(sorted((nodes & active_nodes) | (nodes & gnd)))
        inner = tuple(sorted(nodes - set(term)))
        out.append(PassiveChunk(members=tuple(members), terminals=term,
                                internal=inner,
                                kinds=tuple(sorted({p.kind
                                                    for p in members}))))
    out.sort(key=lambda c: (-len(c.members), c.terminals))
    return out


def chunk_admittance(chunk: PassiveChunk, ground, *, keep=None):
    """Exact Y_eq(s) across a two-terminal chunk, symbolic in its own
    element symbols.

    Solves the chunk ALONE with a unit current injected between its
    terminals: Z_eq = v across the port, Y_eq = 1/Z_eq. Any internal
    nodes are eliminated by that solve, so the interior may be arbitrary.
    """
    if len(chunk.terminals) != 2:
        raise ValueError(
            f"chunk_admittance needs a 2-terminal chunk, got "
            f"{len(chunk.terminals)}: {chunk.terminals}")
    a, b = chunk.terminals
    gnd = set(ground) | {"0"}
    # probe from terminal b; when b is not a real ground it is grounded
    # for the measurement via `grounds` below
    ref = b
    probe = Primitive(inst=_PROBE, param="", kind="isrc", nodes=(ref, a))
    prims = list(chunk.members) + [probe]
    grounds = tuple(ground) if b in gnd else (b,) + tuple(ground)
    system = build_mna(prims, grounds, _PROBE)
    z = solve_tf(system, a, keep=(list(system.symbols) if keep is None
                                  else keep), method="direct")
    y = sp.cancel(1 / sp.cancel(z.expr))
    return sp.together(y), z
