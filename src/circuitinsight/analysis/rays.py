"""The ray test: proving what a SHARED root is, and what would fix it.

A root that no expression names is not merely unexplained. For a
dominant pole it is a design verdict -- a simple formula is what
CONDITIONING produces, so a response that will not read as one has not
been conditioned onto anything yet. But a verdict has to be earned,
and "no partition stood" is an absence, not evidence.

This supplies the evidence, in the form a designer can act on. Where
a root's attribution lives is aggregated onto circuit NODES, giving a
localization L in [0, 1]: one node owning the root reads near 1, a
root spread over several reads low. Then each candidate element is
walked along a ray -- a decade either way -- with the root FOLLOWED by
Newton rather than re-solved, and L watched along the path. Three
outcomes, and each says something different to a designer:

  localized      L is high and stays high. The root lives at one node.
                 If no formula named it, that is a gap in the reader,
                 not a property of the circuit -- so it is reported as
                 a defect rather than as a verdict about the design.

  collapsible    L is low here but RISES along some ray. The root is
                 mixed between locations, and that ray is the
                 direction which would condition it. This is the
                 useful case: the compass points somewhere.

  distributed    L is low and no ray raises it. No resizing conditions
                 this response; it needs topology. Saying so early is
                 worth more than three failed sizing iterations.

Rays are swept over EVERY element, knob or not. A gds cannot be tuned
directly, but a ray along one still names the mechanism -- and process
walks along exactly those axes, so a collapse that only happens at one
corner is a robustness statement. Actionability ranks the advice; it
does not restrict the proof.
"""
from __future__ import annotations

import dataclasses

import numpy as np

from .secondorder import follow_root, loop_pole_jet


def node_shares(g: dict, stamps: dict, node_index: dict, *,
                kinds=("c", "cx")) -> dict:
    """Each element's pull on the root, aggregated onto LOCATIONS.

    A location is a node for a grounded element and the node PAIR for
    a floating one, not half of each: a compensation capacitor
    localizes a pole onto a port, which is the whole point of Miller
    compensation, and splitting its weight between the two ends would
    make such a pole look distributed by construction.

    REACTANCES only, by default. "This pole lives there" is a
    statement about capacitance -- the conductances spread over
    branches by their nature and only blur it, and by Euler the cap
    shares already sum to one, so the normalization means something.
    Widen `kinds` to weigh the whole spectrum instead."""
    rev = {i: n for n, i in node_index.items()}
    out: dict = {}
    for name, w in g.items():
        w = abs(complex(w).real)
        if w <= 0.0:
            continue
        places: set = set()
        for st in stamps.get(name, ()):
            a = st[1]
            if a is None or (kinds and st[0] not in kinds):
                continue
            rows = tuple(sorted(rev[int(i)]
                                for i in np.nonzero(np.asarray(a))[0]
                                if int(i) in rev))
            if rows:
                places.add(rows)
        if not places:
            continue
        for rows in places:                 # a matched symbol may sit
            out[" ↔ ".join(rows)] = (  # at several places at once
                out.get(" ↔ ".join(rows), 0.0) + w / len(places))
    return out


def localization(shares: dict):
    """(L, place): the share of the largest location, and which."""
    if not shares:
        return 0.0, ""
    tot = sum(shares.values()) or 1.0
    node = max(shares, key=lambda n: shares[n])
    return shares[node] / tot, node


@dataclasses.dataclass
class RayPoint:
    scale: float
    f0_hz: float
    localization: float
    node: str


@dataclasses.dataclass
class RayVerdict:
    kind: str                      # localized | collapsible | distributed
    l_nominal: float = 0.0
    node_nominal: str = ""
    best_symbol: str = ""          # the ray that collapses it, if any
    best_scale: float = 1.0
    best_localization: float = 0.0
    best_node: str = ""
    path: tuple = ()               # the winning ray, for the record

    def describe(self) -> str:
        if self.kind == "localized":
            return (f"the root already lives at {self.node_nominal} "
                    f"({self.l_nominal:.0%}); no formula named it, "
                    f"which is a gap in the reader")
        if self.kind == "collapsible":
            way = "up" if self.best_scale > 1 else "down"
            return (f"shared ({self.l_nominal:.0%} at "
                    f"{self.node_nominal}) but COLLAPSIBLE: "
                    f"{self.best_symbol} {way} "
                    f"{max(self.best_scale, 1 / self.best_scale):.3g}x "
                    f"gathers it onto {self.best_node} "
                    f"({self.best_localization:.0%})")
        return (f"distributed ({self.l_nominal:.0%} at "
                f"{self.node_nominal}); no element ray gathers it, so "
                f"no resizing conditions this response")


def ray(system, probe, primitives, alias, symbol, s0, *,
        factors=(0.1, 0.2, 0.5, 2.0, 5.0, 10.0)):
    """Walk one element and watch where the root's attribution goes.

    The root is FOLLOWED from the previous point by Newton, never
    re-solved, so a whole ray costs a handful of inverses. Points
    where the root will not track are dropped rather than guessed at."""
    from .marginledger import _collect_stamps, _tian_context

    out = []
    base = float(system.values[symbol])
    for lam in sorted(factors, key=lambda v: abs(np.log(v))):
        sysl = dataclasses.replace(
            system, values={**system.values, symbol: base * lam})
        fn, Q, ib, ip = _tian_context(sysl, probe)
        s = follow_root(fn, Q, ib, ip, s0)
        if s is None:
            continue
        stamps, vals, _m = _collect_stamps(sysl, primitives, alias)
        g, _H = loop_pole_jet(fn, Q, ib, ip, stamps, vals, s)
        loc, node = localization(node_shares(g, stamps,
                                            sysl.node_index))
        out.append(RayPoint(scale=float(lam),
                            f0_hz=float(abs(s) / (2 * np.pi)),
                            localization=float(loc), node=node))
    return out


def assess(system, probe, primitives, alias, g, stamps, s0, *,
           top: int = 6, high: float = 0.60, gain: float = 0.15,
           factors=(0.1, 0.2, 0.5, 2.0, 5.0, 10.0)) -> RayVerdict:
    """The three-valued verdict for one root, with its evidence.

    Only the strongest movers are swept -- a ray along an element that
    barely touches the root cannot say anything about it -- but they
    are swept whether or not they are knobs."""
    shares = node_shares(g, stamps, system.node_index)
    l0, node0 = localization(shares)
    if l0 >= high:
        return RayVerdict(kind="localized", l_nominal=l0,
                          node_nominal=node0)
    order = sorted((n for n in g if n in system.values),
                   key=lambda n: -abs(complex(g[n]).real))[:top]
    best = None
    for name in order:
        try:
            path = ray(system, probe, primitives, alias, name, s0,
                       factors=factors)
        except Exception:                          # noqa: BLE001
            continue
        for pt in path:
            if pt.localization >= max(l0 + gain, high) and (
                    best is None or pt.localization > best[0].localization):
                best = (pt, name, path)
    if best is None:
        return RayVerdict(kind="distributed", l_nominal=l0,
                          node_nominal=node0)
    pt, name, path = best
    return RayVerdict(kind="collapsible", l_nominal=l0,
                      node_nominal=node0, best_symbol=name,
                      best_scale=pt.scale,
                      best_localization=pt.localization,
                      best_node=pt.node, path=tuple(path))
