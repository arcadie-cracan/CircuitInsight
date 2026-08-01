"""Which nodes are really AC grounds, and what does assuming so cost?

CMOS designs are full of mirror and bias nodes, and declaring them AC
grounds is the modelling step that makes a circuit tractable -- it is
what turns the uA741's hopeless 15/155 tear into a balanced 81/88 one,
and it is the "AC ground" column of Shi's stage-form panel (ACM TODAES
2019). The catch is that the schematic cannot tell you which ones are
safe: in a 5T OTA both `vbn` (the tail-source gate) and the active-load
mirror gate are diode-connected mirror gates, but grounding the first
costs nothing and grounding the second destroys the gain, because the
load mirror carries half the signal.

So this module proposes structurally and decides numerically.

STRUCTURE (free) nominates candidates: nodes that control transconductor
gates, flagged as mirror references when the node also drives the device
that feeds it (a diode connection, i.e. a 1/gm to ground).

COST is then EXACT, not estimated. Grounding node k is a rank-one
modification of the MNA matrix, so Sherman-Morrison gives the output
change in closed form:

    dv_out(k)  =  - v_k * Z[out, k] / Z[k, k]        (Z = A^-1)

with v the ordinary solution. One matrix inverse per frequency therefore
scores EVERY node exactly -- no per-candidate re-solve -- and the same
inverse gives the joint effect of grounding a SET K by the Woodbury form

    dv_out(K)  =  - Z[out, K] . inv(Z[K, K]) . v[K]

which is what makes greedy set growth affordable and exact. The reported
number is the worst |dB| deviation of the transfer function over the
frequency grid, i.e. the error the designer would actually see.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import sympy as sp

from ..engine.mna import MnaError, S, build_mna

__all__ = ["BiasCandidate", "AcGroundReport", "joint_cost",
           "scan_ac_grounds"]


@dataclass
class BiasCandidate:
    """One node, its structural story, and the measured cost of grounding."""
    node: str
    kind: str                     # "mirror reference" | "gate node" | "node"
    controls: tuple = ()          # devices whose gates it drives
    worst_db: float = 0.0
    worst_deg: float = 0.0
    f_worst_hz: float = 0.0
    within_budget: bool = False

    def describe(self) -> str:
        who = (f" (gates {', '.join(self.controls[:4])}"
               f"{'…' if len(self.controls) > 4 else ''})"
               if self.controls else "")
        verdict = "OK" if self.within_budget else "NOT an AC ground"
        return (f"{self.node}: {self.worst_db:.3g} dB / "
                f"{self.worst_deg:.3g}° -> {verdict}  [{self.kind}]{who}")


@dataclass
class AcGroundReport:
    candidates: list = field(default_factory=list)   # all, worst_db ascending
    recommended: list = field(default_factory=list)  # greedy set within budget
    joint_db: float = 0.0                            # measured cost of the set
    budget_db: float = 0.1

    def describe(self) -> str:
        lines = [f"AC-ground scan (budget {self.budget_db:g} dB):"]
        for c in self.candidates:
            lines.append("  " + c.describe())
        if self.recommended:
            lines.append(f"  -> ground {', '.join(self.recommended)} "
                         f"together: {self.joint_db:.3g} dB")
        else:
            lines.append("  -> no node can be grounded within the budget")
        return "\n".join(lines)


def _structural_candidates(primitives, ground) -> dict:
    """Nodes that control transconductors, labelled by their role. A node
    that controls a device AND is that device's own output node is a
    diode-connected mirror reference -- the classic bias node."""
    gnd = set(ground) | {"0"}
    controls: dict = defaultdict(list)
    outputs: dict = defaultdict(set)
    for p in primitives:
        if p.kind in ("vccs", "cx") and len(p.nodes) == 4:
            d, _s, cp, cn = p.nodes
            for c in (cp, cn):
                if c not in gnd:
                    controls[c].append(p.inst)
            if d not in gnd:
                outputs[d].add(p.inst)
    out = {}
    for node, insts in controls.items():
        diode = bool(set(insts) & outputs.get(node, set()))
        out[node] = ("mirror reference" if diode else "gate node",
                     tuple(sorted(set(insts))))
    return out


def scan_ac_grounds(primitives, ground, inp: str, out: str, *,
                    freqs=None, budget_db: float = 0.1, alias=None,
                    max_report: int = 12, structural_only: bool = True,
                    exclude=()) -> AcGroundReport:
    """Rank nodes by the EXACT error that declaring them AC grounds would
    introduce in tf(inp -> out), and recommend the largest set that fits
    `budget_db`.

    structural_only: score just the transconductor-control nodes (the
    mirror/bias candidates a designer would consider). Set False to score
    every node in the circuit -- the same single inverse per frequency
    covers them all, so it costs nothing extra."""
    system = build_mna(primitives, ground, inp, alias)
    subs = {system.symbols[n]: sp.Float(v)
            for n, v in system.values.items() if n in system.symbols}
    A = system.A.xreplace(subs)
    free = set(A.free_symbols) - {S}
    if free:
        raise MnaError("ac-ground scan needs numeric values; missing "
                       f"{sorted(map(str, free))}")
    if out not in system.node_index:
        raise MnaError(f"output node {out!r} not found (or it is ground)")

    struct = _structural_candidates(primitives, ground)
    gnd = set(ground) | {"0"}
    skip = set(exclude) | gnd | {out}
    # nodes held by an independent source are ALREADY ac grounds (or the
    # excitation itself): "grounding" them adds a second definition of the
    # same node and the modified matrix is singular, so they are not
    # candidates -- they are already true
    for p in primitives:
        if p.kind == "vsrc":
            skip |= set(p.nodes)
        elif p.kind == "balun":
            skip |= set(p.nodes)           # ideal 4-terminal driver
        elif p.kind == "vcvs":
            skip |= set(p.nodes[:2])       # its output is ideally driven
    # An ideally driven node cannot be "declared" an AC ground: the source
    # fixes its voltage no matter what admittance is hung on it, so the
    # scan would score a harmless-looking 0 dB for a change that simply
    # does not happen (measured on the 5T OTA: vin_p, held by the input
    # balun, reads 0 dB while obviously not being a bias node).
    names = [n for n in system.node_index
             if n not in skip and (not structural_only or n in struct)]
    if not names:
        return AcGroundReport(budget_db=budget_db)

    if freqs is None:
        freqs = np.geomspace(1.0, 1e10, 41)
    freqs = np.atleast_1d(np.asarray(freqs, dtype=float))
    fn = sp.lambdify(S, A, "numpy")
    z = np.array(system.z.tolist(), dtype=complex).ravel()
    k_out = system.node_index[out]
    idx = [system.node_index[n] for n in names]

    # per frequency: ONE inverse scores every candidate (and any set)
    worst = np.zeros(len(names))
    worst_deg = np.zeros(len(names))
    f_worst = np.zeros(len(names))
    ratios = []                                  # per-f dv_out/v_out rows
    for f in freqs:
        Z = np.linalg.inv(np.asarray(fn(2j * np.pi * f), dtype=complex))
        v = Z @ z
        v_out = v[k_out]
        if v_out == 0:
            ratios.append(np.zeros(len(names), dtype=complex))
            continue
        zkk = np.array([Z[k, k] for k in idx])
        zok = np.array([Z[k_out, k] for k in idx])
        vk = np.array([v[k] for k in idx])
        with np.errstate(divide="ignore", invalid="ignore"):
            dv = -vk * zok / zkk
        r = np.nan_to_num(dv / v_out, nan=0.0, posinf=1e12, neginf=1e12)
        ratios.append(r)
        db = np.abs(20 * np.log10(np.abs(1 + r)))
        deg = np.abs(np.degrees(np.angle(1 + r)))
        upd = db > worst
        worst = np.where(upd, db, worst)
        worst_deg = np.where(upd, deg, worst_deg)
        f_worst = np.where(upd, f, f_worst)

    cands = [BiasCandidate(node=n, kind=struct.get(n, ("node", ()))[0],
                           controls=struct.get(n, ("node", ()))[1],
                           worst_db=float(worst[i]),
                           worst_deg=float(worst_deg[i]),
                           f_worst_hz=float(f_worst[i]),
                           within_budget=bool(worst[i] <= budget_db))
             for i, n in enumerate(names)]
    cands.sort(key=lambda c: c.worst_db)

    # greedy set growth, joint error EXACT from the same inverses
    chosen: list[int] = []
    chosen_names: list[str] = []
    joint = 0.0
    for c in cands:
        if not c.within_budget:
            break
        trial = chosen + [names.index(c.node)]
        j = _joint_db(fn, z, k_out, [idx[t] for t in trial], freqs)
        if j <= budget_db:
            chosen, chosen_names, joint = trial, chosen_names + [c.node], j
    return AcGroundReport(candidates=cands[:max_report],
                          recommended=chosen_names, joint_db=float(joint),
                          budget_db=budget_db)


def _joint_db(fn, z, k_out, ks, freqs) -> float:
    """Worst |dB| from grounding the SET ks, by the Woodbury form of the
    same rank-one identity (exact, no re-solve of the modified circuit)."""
    worst = 0.0
    for f in freqs:
        Z = np.linalg.inv(np.asarray(fn(2j * np.pi * f), dtype=complex))
        v = Z @ z
        if v[k_out] == 0:
            continue
        Zkk = Z[np.ix_(ks, ks)]
        try:
            corr = Z[k_out, ks] @ np.linalg.solve(Zkk, v[ks])
        except np.linalg.LinAlgError:
            return float("inf")
        r = -corr / v[k_out]
        worst = max(worst, abs(20 * np.log10(abs(1 + r))))
    return worst


def joint_cost(primitives, ground, inp: str, out: str, nodes, *,
               alias=None, freqs=None) -> float:
    """Worst |dB| of grounding an ARBITRARY node set together — the GUI's
    recompute-on-toggle: the scan prices the recommended set, this prices
    whatever the user actually ticks, by the same Woodbury identity."""
    nodes = list(nodes)
    if not nodes:
        return 0.0
    system = build_mna(primitives, ground, inp, alias)
    subs = {system.symbols[n]: sp.Float(v)
            for n, v in system.values.items() if n in system.symbols}
    A = system.A.xreplace(subs)
    if set(A.free_symbols) - {S}:
        raise MnaError("joint_cost needs numeric values")
    missing = [n for n in nodes if n not in system.node_index]
    if out not in system.node_index or missing:
        raise MnaError(f"unknown node(s): {missing or [out]}")
    if freqs is None:
        freqs = np.geomspace(1.0, 1e10, 41)
    fn = sp.lambdify(S, A, "numpy")
    z = np.array(system.z.tolist(), dtype=complex).ravel()
    ks = [system.node_index[n] for n in nodes]
    return float(_joint_db(fn, z, system.node_index[out], ks,
                           np.atleast_1d(np.asarray(freqs, dtype=float))))
