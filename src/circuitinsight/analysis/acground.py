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
    score: float | None = None    # x budget under the active criterion

    def describe(self) -> str:
        who = (f" (gates {', '.join(self.controls[:4])}"
               f"{'…' if len(self.controls) > 4 else ''})"
               if self.controls else "")
        verdict = "OK" if self.within_budget else "NOT an AC ground"
        cost = (f"{self.score:.2f}x budget ({self.worst_db:.3g} dB / "
                f"{self.worst_deg:.3g}°)" if self.score is not None else
                f"{self.worst_db:.3g} dB / {self.worst_deg:.3g}°")
        return f"{self.node}: {cost} -> {verdict}  [{self.kind}]{who}"


@dataclass
class AcGroundReport:
    candidates: list = field(default_factory=list)   # cheapest first
    recommended: list = field(default_factory=list)  # greedy set within budget
    joint_db: float = 0.0                            # measured cost of the set
    budget_db: float = 0.1
    joint_score: float | None = None  # x budget under the active criterion
    criterion_label: str = ""         # names the contract that gated

    def describe(self) -> str:
        gate = (f"criterion: {self.criterion_label}" if self.criterion_label
                else f"budget {self.budget_db:g} dB")
        lines = [f"AC-ground scan ({gate}):"]
        for c in self.candidates:
            lines.append("  " + c.describe())
        if self.recommended:
            cost = (f"{self.joint_score:.2f}x budget"
                    if self.joint_score is not None
                    else f"{self.joint_db:.3g} dB")
            lines.append(f"  -> ground {', '.join(self.recommended)} "
                         f"together: {cost}")
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
                    exclude=(), criterion=None, include=(),
                    loop_probe: str | None = None) -> AcGroundReport:
    """Rank nodes by the EXACT error that declaring them AC grounds would
    introduce in tf(inp -> out), and recommend the largest set that fits
    `budget_db`.

    structural_only: score just the transconductor-control nodes (the
    mirror/bias candidates a designer would consider). Set False to score
    every node in the circuit -- the same single inverse per frequency
    covers them all, so it costs nothing extra.

    criterion: a BandCriterion. When given, every candidate (and the
    greedy set) is priced AND gated by criterion.score -- the same
    contract, unit and window as the order reduction -- instead of the
    magnitude-only budget_db; the dB/deg figures stay as secondary
    information. Pass `freqs` spanning the contract's band.

    include: nodes the USER asked about (a Nets-tree right-click).
    They are priced alongside the structural candidates -- same single
    inverse, zero extra cost -- and always survive into the report,
    past both the structural filter and the max_report cap: a request
    must come back with a price, never vanish.

    loop_probe: price against the TIAN LOOP GAIN at this probe instead
    of tf(inp -> out) -- the contract of an stb bench, where a
    reduction must preserve T, not a transfer nobody measured. `inp`
    and `out` are ignored; the same one-inverse-per-frequency
    machinery reads T through NumericTian (two solves, six scalars)."""
    system = build_mna(primitives, ground, loop_probe or inp, alias)
    tian = None
    if loop_probe is not None:
        from .loopgain import NumericTian

        tian = NumericTian(system, loop_probe)
    subs = {system.symbols[n]: sp.Float(v)
            for n, v in system.values.items() if n in system.symbols}
    A = system.A.xreplace(subs)
    free = set(A.free_symbols) - {S}
    if free:
        raise MnaError("ac-ground scan needs numeric values; missing "
                       f"{sorted(map(str, free))}")
    if tian is None and out not in system.node_index:
        raise MnaError(f"output node {out!r} not found (or it is ground)")

    struct = _structural_candidates(primitives, ground)
    gnd = set(ground) | {"0"}
    skip = set(exclude) | gnd
    if tian is None:
        skip |= {out}
    else:
        # grounding a probe terminal would cut the very loop being
        # priced: those two nodes are never candidates
        inv_nodes = {i: n for n, i in system.node_index.items()}
        skip |= {inv_nodes[tian.p], inv_nodes[tian.q]}
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
    include = [n for n in include if n in system.node_index
               and n not in skip]
    names = [n for n in system.node_index
             if n not in skip and (not structural_only or n in struct
                                   or n in include)]
    if not names:
        return AcGroundReport(budget_db=budget_db)

    if freqs is None:
        freqs = np.geomspace(1.0, 1e10, 41)
    freqs = np.atleast_1d(np.asarray(freqs, dtype=float))
    fn = sp.lambdify(S, A, "numpy")
    z = np.array(system.z.tolist(), dtype=complex).ravel()
    k_out = system.node_index[out] if tian is None else None
    idx = [system.node_index[n] for n in names]

    # per frequency: ONE inverse scores every candidate (and any set)
    worst = np.zeros(len(names))
    worst_deg = np.zeros(len(names))
    f_worst = np.zeros(len(names))
    ratios = []                                  # per-f dv_out/v_out rows
    vouts = []                                   # the full response H(f)
    for f in freqs:
        Z = np.linalg.inv(np.asarray(fn(2j * np.pi * f), dtype=complex))
        if tian is not None:
            resp, xv, xi = tian.response(Z, z)
        else:
            v = Z @ z
            resp = v[k_out]
        vouts.append(resp)
        if resp == 0:
            ratios.append(np.zeros(len(names), dtype=complex))
            continue
        zkk = np.array([Z[k, k] for k in idx])
        with np.errstate(divide="ignore", invalid="ignore"):
            if tian is not None:
                # grounding node k corrects BOTH Tian solves by the
                # same rank-one column: x' = x - x[k] Z[:,k]/Z[k,k];
                # only the six read rows are needed, per candidate
                rd = {}
                for tag, x in (("v", xv), ("i", xi)):
                    xk = x[idx]
                    for j in tian.rows:
                        rd[tag, j] = x[j] - xk * Z[j, idx] / zkk
                Tp = tian.combine(rd["v", tian.ib], rd["v", tian.p],
                                  rd["i", tian.ib], rd["i", tian.q])
                r = np.nan_to_num(Tp / resp - 1.0, nan=0.0,
                                  posinf=1e12, neginf=1e12)
            else:
                zok = np.array([Z[k_out, k] for k in idx])
                vk = np.array([v[k] for k in idx])
                dv = -vk * zok / zkk
                r = np.nan_to_num(dv / resp, nan=0.0,
                                  posinf=1e12, neginf=1e12)
        ratios.append(r)
        db = np.abs(20 * np.log10(np.abs(1 + r)))
        deg = np.abs(np.degrees(np.angle(1 + r)))
        upd = db > worst
        worst = np.where(upd, db, worst)
        worst_deg = np.where(upd, deg, worst_deg)
        f_worst = np.where(upd, f, f_worst)

    H_full = np.array(vouts, dtype=complex)
    R = np.array(ratios, dtype=complex)          # (n_freq, n_cand)
    scores = None
    if criterion is not None:
        scores = [float(criterion.score(freqs, H_full,
                                        H_full * (1.0 + R[:, i])))
                  for i in range(len(names))]

    def ok(i):
        return (scores[i] <= 1.0 if scores is not None
                else worst[i] <= budget_db)

    cands = [BiasCandidate(node=n, kind=struct.get(n, ("node", ()))[0],
                           controls=struct.get(n, ("node", ()))[1],
                           worst_db=float(worst[i]),
                           worst_deg=float(worst_deg[i]),
                           f_worst_hz=float(f_worst[i]),
                           within_budget=bool(ok(i)),
                           score=(None if scores is None else scores[i]))
             for i, n in enumerate(names)]
    cands.sort(key=lambda c: c.worst_db if c.score is None else c.score)
    # the requested nodes always make the report, cap or not
    shown = cands[:max_report]
    for c in cands[max_report:]:
        if c.node in include:
            shown.append(c)

    # greedy set growth, joint error EXACT from the same inverses -- and
    # gated by the SAME contract that gated the singles
    chosen: list[int] = []
    chosen_names: list[str] = []
    joint = 0.0
    joint_score = None
    for c in cands:
        if not c.within_budget:
            break
        trial = chosen + [names.index(c.node)]
        Hj = _joint_response(fn, z, k_out, [idx[t] for t in trial], freqs,
                             tian=tian)
        with np.errstate(divide="ignore", invalid="ignore"):
            jdb = np.abs(20 * np.log10(np.abs(Hj / H_full)))
        j = float(np.nanmax(jdb[np.isfinite(jdb)])) if np.isfinite(
            jdb).any() else float("inf")
        if criterion is not None:
            js = float(criterion.score(freqs, H_full, Hj))
            if js <= 1.0:
                chosen, chosen_names = trial, chosen_names + [c.node]
                joint, joint_score = j, js
        elif j <= budget_db:
            chosen, chosen_names, joint = trial, chosen_names + [c.node], j
    label = "" if criterion is None else (criterion.name or "dB band")
    return AcGroundReport(candidates=shown,
                          recommended=chosen_names, joint_db=float(joint),
                          budget_db=budget_db, joint_score=joint_score,
                          criterion_label=label)


def _joint_response(fn, z, k_out, ks, freqs, tian=None) -> np.ndarray:
    """The MODIFIED response from grounding the SET ks, by the Woodbury
    form of the same rank-one identity (exact, no re-solve). An array so
    any contract can price it. With `tian`, the response is the Tian
    loop gain: the Woodbury correction applies to BOTH probe solves and
    the six corrected reads combine into T."""
    out = np.empty(len(freqs), dtype=complex)
    for i, f in enumerate(freqs):
        Z = np.linalg.inv(np.asarray(fn(2j * np.pi * f), dtype=complex))
        Zkk = Z[np.ix_(ks, ks)]
        if tian is not None:
            _t, xv, xi = tian.response(Z, z)
            try:
                rd = {}
                for tag, x in (("v", xv), ("i", xi)):
                    cv = np.linalg.solve(Zkk, x[ks])
                    for j in tian.rows:
                        rd[tag, j] = x[j] - Z[j, ks] @ cv
            except np.linalg.LinAlgError:
                out[i] = np.inf
                continue
            out[i] = tian.combine(rd["v", tian.ib], rd["v", tian.p],
                                  rd["i", tian.ib], rd["i", tian.q])
            continue
        v = Z @ z
        if v[k_out] == 0:
            out[i] = 0.0
            continue
        try:
            corr = Z[k_out, ks] @ np.linalg.solve(Zkk, v[ks])
        except np.linalg.LinAlgError:
            out[i] = np.inf
            continue
        out[i] = v[k_out] - corr
    return out


def _joint_db(fn, z, k_out, ks, freqs) -> float:
    """Worst |dB| of grounding the SET ks (the pre-contract gate; kept
    for the criterion-less callers)."""
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


def joint_metrics(primitives, ground, inp: str, out: str, nodes, *,
                  alias=None, freqs=None, criterion=None) -> dict:
    """Price grounding an ARBITRARY node set under the active contract:
    {"worst_db", "worst_deg", "score" (None without a criterion)}. The
    GUI's recompute-on-toggle reads THIS, so the ticked set is priced in
    the same unit as every other approximation."""
    nodes = list(nodes)
    if not nodes:
        return {"worst_db": 0.0, "worst_deg": 0.0,
                "score": None if criterion is None else 0.0}
    system = build_mna(primitives, ground, inp, alias)
    subs = {system.symbols[n]: sp.Float(v)
            for n, v in system.values.items() if n in system.symbols}
    A = system.A.xreplace(subs)
    if set(A.free_symbols) - {S}:
        raise MnaError("joint_metrics needs numeric values")
    missing = [n for n in nodes if n not in system.node_index]
    if out not in system.node_index or missing:
        raise MnaError(f"unknown node(s): {missing or [out]}")
    if freqs is None:
        freqs = np.geomspace(1.0, 1e10, 41)
    freqs = np.atleast_1d(np.asarray(freqs, dtype=float))
    fn = sp.lambdify(S, A, "numpy")
    z = np.array(system.z.tolist(), dtype=complex).ravel()
    k_out = system.node_index[out]
    ks = [system.node_index[n] for n in nodes]
    H_full = np.empty(len(freqs), dtype=complex)
    for i, f in enumerate(freqs):
        Z = np.linalg.inv(np.asarray(fn(2j * np.pi * f), dtype=complex))
        H_full[i] = (Z @ z)[k_out]
    H_mod = _joint_response(fn, z, k_out, ks, freqs)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = H_mod / H_full
        db = np.abs(20 * np.log10(np.abs(ratio)))
        deg = np.abs(np.degrees(np.angle(ratio)))
    fin = np.isfinite(db)
    return {
        "worst_db": float(db[fin].max()) if fin.any() else float("inf"),
        "worst_deg": float(deg[np.isfinite(deg)].max())
                     if np.isfinite(deg).any() else float("inf"),
        "score": (None if criterion is None
                  else float(criterion.score(freqs, H_full, H_mod))),
    }
