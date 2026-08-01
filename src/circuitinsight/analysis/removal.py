"""Element-removal advisory: which elements can simply be DELETED.

The Lei & Wu setting-zero idea (IJCTA 2024) turned into a measured scan:
their designer-level analysis proves a compensation capacitor UNNECESSARY
by nulling quantities; here every explicit two-terminal element is priced
by the exact response shift its removal would cause, from ONE matrix
inverse per frequency — the dual of the candidate-INSTALLATION sweep the
compensation suggester runs, and the same rank-one identity the AC-ground
scan uses.

Removing an element of admittance Y across nodes (a, b) is adding −Y
there, which is rank-one in the incidence vector b_ab:

    H_removed / H = 1 + Y (Z v)_path / (1 − Y Z_bb)      (Sherman–Morrison)

with Z = A⁻¹, so a single inverse per frequency prices every element.
Scope mirrors the chunker's: EXPLICIT netlist elements only. A device
model's gds or cpi is passive by stamp but it is an OP parameter — the
scan must not advise deleting half a transistor.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import sympy as sp

from ..engine.mna import MnaError, S, build_mna

__all__ = ["RemovalCandidate", "RemovalReport", "scan_removals"]


@dataclass
class RemovalCandidate:
    """One removable element and the measured cost of deleting it."""
    inst: str
    kind: str                      # "c" | "r" | "g" | "l"
    value: float | None
    nodes: tuple
    worst_db: float = 0.0
    worst_deg: float = 0.0
    f_worst_hz: float = 0.0
    within_budget: bool = False

    def describe(self) -> str:
        unit = {"c": "F", "r": "Ω", "g": "S", "l": "H"}.get(self.kind, "")
        val = f" = {self.value:.4g} {unit}" if self.value is not None else ""
        verdict = ("REMOVABLE" if self.within_budget
                   else "needed")
        return (f"{self.inst}{val}: removing costs {self.worst_db:.3g} dB / "
                f"{self.worst_deg:.3g}° (worst at {self.f_worst_hz:.3g} Hz) "
                f"-> {verdict}")


@dataclass
class RemovalReport:
    candidates: list = field(default_factory=list)   # worst_db ascending
    recommended: list = field(default_factory=list)  # jointly inside budget
    joint_db: float = 0.0                            # measured for the set
    budget_db: float = 0.1

    def describe(self) -> str:
        lines = [f"element-removal scan (budget {self.budget_db:g} dB):"]
        for c in self.candidates:
            lines.append("  " + c.describe())
        if self.recommended:
            lines.append(f"  -> delete {', '.join(self.recommended)} "
                         f"together: {self.joint_db:.3g} dB")
        else:
            lines.append("  -> nothing is removable within the budget")
        return "\n".join(lines)


def _admittance(kind: str, value: float, w: complex):
    if kind == "c":
        return w * value
    if kind == "r":
        return 1.0 / value if value else None
    if kind == "g":
        return value
    if kind == "l":
        return 1.0 / (w * value) if value else None
    return None


def scan_removals(primitives, ground, inp: str, out: str, *,
                  freqs=None, budget_db: float = 0.1, alias=None,
                  max_report: int = 12, protect=()) -> RemovalReport:
    """Price the removal of every EXPLICIT two-terminal passive in
    tf(inp -> out), and recommend the largest jointly-removable set inside
    `budget_db`. `protect` names instances never to propose (e.g. the load
    the spec is defined against)."""
    explicit = [p for p in primitives
                if p.kind in ("c", "r", "g", "l") and not p.param
                and p.value is not None and p.inst not in protect
                and len(p.nodes) == 2]
    if not explicit:
        return RemovalReport(budget_db=budget_db)

    system = build_mna(primitives, ground, inp, alias)
    subs = {system.symbols[n]: sp.Float(v)
            for n, v in system.values.items() if n in system.symbols}
    A = system.A.xreplace(subs)
    if set(A.free_symbols) - {S}:
        raise MnaError("removal scan needs numeric values")
    if out not in system.node_index:
        raise MnaError(f"output node {out!r} not found (or it is ground)")
    if freqs is None:
        freqs = np.geomspace(1.0, 1e10, 41)
    freqs = np.atleast_1d(np.asarray(freqs, dtype=float))

    fn = sp.lambdify(S, A, "numpy")
    z = np.array(system.z.tolist(), dtype=complex).ravel()
    k_out = system.node_index[out]
    gnd = set(ground) | {"0"}

    def bvec(nodes):
        b = np.zeros(len(z), dtype=complex)
        for node, sign in zip(nodes, (1.0, -1.0)):
            if node not in gnd:
                b[system.node_index[node]] += sign
        return b

    bs = [bvec(p.nodes) for p in explicit]
    n = len(explicit)
    worst = np.zeros(n)
    worst_deg = np.zeros(n)
    f_worst = np.zeros(n)
    for f in freqs:
        w = 2j * np.pi * f
        Z = np.linalg.inv(np.asarray(fn(w), dtype=complex))
        v = Z @ z
        v_out = v[k_out]
        if v_out == 0:
            continue
        for i, (p, b) in enumerate(zip(explicit, bs)):
            Y = _admittance(p.kind, p.value, w)
            if Y is None:
                continue
            Zb = Z @ b
            # A' = A − Y bbᵀ  ->  v' = v + Y·Zb·(bᵀv)/(1 − Y·bᵀZb)
            den = 1.0 - Y * (b @ Zb)
            if den == 0:
                worst[i] = float("inf")
                continue
            dv_out = Y * Zb[k_out] * (b @ v) / den
            r = dv_out / v_out
            db = abs(20 * np.log10(abs(1 + r))) if 1 + r != 0 else np.inf
            if db > worst[i]:
                worst[i] = db
                worst_deg[i] = abs(np.degrees(np.angle(1 + r)))
                f_worst[i] = f

    cands = [RemovalCandidate(inst=p.inst, kind=p.kind, value=p.value,
                              nodes=tuple(p.nodes),
                              worst_db=float(worst[i]),
                              worst_deg=float(worst_deg[i]),
                              f_worst_hz=float(f_worst[i]),
                              within_budget=bool(worst[i] <= budget_db))
             for i, p in enumerate(explicit)]
    cands.sort(key=lambda c: c.worst_db)

    # greedy joint growth, VERIFIED numerically per step: unlike the
    # AC-ground case there is no cheap exact joint identity across mixed
    # frequency-dependent admittances, so the joint check is a re-solve of
    # the pruned circuit -- still numeric-only and cheap
    chosen: list[str] = []
    joint = 0.0
    for c in cands:
        if not c.within_budget:
            break
        trial = chosen + [c.inst]
        j = _joint_removal_db(primitives, ground, inp, out, trial,
                              freqs, alias)
        if j <= budget_db:
            chosen, joint = trial, j
    return RemovalReport(candidates=cands[:max_report], recommended=chosen,
                         joint_db=float(joint), budget_db=budget_db)


def _joint_removal_db(primitives, ground, inp, out, insts, freqs,
                      alias) -> float:
    """Worst |dB| of deleting the SET, by re-solving the pruned circuit."""
    from . import tearing

    keep = [p for p in primitives
            if not (p.inst in insts and p.kind in ("c", "r", "g", "l")
                    and not p.param)]
    a = tearing._numeric_response(primitives, ground, inp, out, freqs, alias)
    b = tearing._numeric_response(keep, ground, inp, out, freqs, alias)
    with np.errstate(divide="ignore", invalid="ignore"):
        db = np.abs(20 * np.log10(np.abs(b / a)))
    db = db[np.isfinite(db)]
    return float(db.max()) if db.size else float("inf")
