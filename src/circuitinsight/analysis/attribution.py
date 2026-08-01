"""Per-pole attribution: which element establishes which pole — verified.

The Manocha & Rincón-Mora recursive-shunt intuition (2025) says each
capacitance shunts its parallel resistance past its RC corner; textbooks
assign poles to nodes the same way, by inspection. This module computes
the assignment EXACTLY and then checks it, instead of assuming it:

    dp/dv_i = − w̄ (∂A/∂v_i)(p) v  /  w̄ Q v        [A = P + s Q]

with (w, v) the left/right null vectors of A(p). The attribution share of
element i at pole p is its LOG-sensitivity |v_i/p · dp/dv_i| — the
relative pole movement per relative parameter change — normalized over
all elements. ∂A/∂v_i is affine in s and computed ONCE per symbol as a
(P_i, Q_i) pattern pair, so poles add no symbolic work.

Verification is a first-order-vs-exact check: nudge the top owner by 5%,
re-root the (numeric, s-swept) denominator, and compare the actual pole
movement with the predicted one. A pole whose owner fails that check is
reported UNVERIFIED rather than silently mis-attributed — clustered or
near-cancelling roots earn that flag honestly.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import sympy as sp

from ..engine.mna import MnaError, S, MnaSystem

__all__ = ["PoleOwner", "PoleAttribution", "pole_attribution"]


@dataclass
class PoleOwner:
    symbol: str                    # model parameter (gm_M5, CL, ...)
    share: float                   # normalized log-sensitivity share

    def describe(self) -> str:
        return f"{self.symbol} ({self.share:.0%})"


@dataclass
class PoleAttribution:
    f_hz: float
    damping: float | None          # zeta for a complex pair, else None
    owners: list = field(default_factory=list)   # PoleOwner, share desc
    verified: bool = False         # first-order vs exact check on owner 1
    predicted_rel: float = 0.0     # predicted |dp/p| for the 5% nudge
    actual_rel: float = 0.0        # measured  |dp/p|

    def describe(self) -> str:
        from ..units import eng

        who = ", ".join(o.describe() for o in self.owners)
        tail = "" if self.verified else "  [UNVERIFIED]"
        zt = (f" (complex pair, zeta {self.damping:.3g})"
              if self.damping is not None else "")
        return f"{eng(self.f_hz, 'Hz')}{zt}: set by {who}{tail}"


def _den_roots(A_num: sp.Matrix) -> np.ndarray:
    """Denominator roots via the s-sweep (never the symbolic determinant:
    that is the very path the all-numeric solver replaced)."""
    from ..engine.interp import _s_degree_bound, _matvec, _vinv_rows, \
        s_sweep_det

    L = _s_degree_bound(A_num) + 1
    vals, _ = s_sweep_det(A_num, L)
    coeffs = _matvec(_vinv_rows([sp.Rational(v) for v in range(L)]), vals)
    c = np.array([float(sp.Float(sp.Rational(int(x.numerator),
                                             int(x.denominator)), 30))
                  if x else 0.0 for x in coeffs], dtype=float)
    nz = np.nonzero(c)[0]
    if nz.size == 0:
        raise MnaError("zero denominator")
    c = c[: nz[-1] + 1]
    scale = np.abs(c[np.nonzero(c)]).max()
    r = np.roots((c / scale)[::-1])
    r = r[np.abs(r) > 0]
    return r[np.argsort(np.abs(r))]


def pole_attribution(system: MnaSystem, n_poles: int = 4, top: int = 3,
                     verify: bool = True, eps: float = 0.05) -> list:
    """Attribute the lowest n_poles natural frequencies of `system` to the
    model parameters that establish them. Returns [PoleAttribution]."""
    subs = {system.symbols[n]: sp.Float(v)
            for n, v in system.values.items() if n in system.symbols}
    A = system.A.xreplace(subs)
    if set(A.free_symbols) - {S}:
        raise MnaError("pole attribution needs numeric values")
    P = np.array(A.xreplace({S: sp.Integer(0)}).tolist(), dtype=complex)
    Q = np.array(A.diff(S).tolist(), dtype=complex)

    # per-symbol stamp patterns, ONCE: dA/dv_i = P_i + s Q_i. Built by a
    # single pass over the nonzero ENTRIES (a per-symbol matrix diff on a
    # 30x30 sympy matrix costs ~90 ms x 250 symbols = the 20 s that made
    # the first cut unusable; per-entry scalar diffs are hundreds of
    # micro-ops)
    nrows = system.A.rows
    by_name = {s_: n_ for n_, s_ in system.symbols.items()}
    patterns = {}
    for i in range(nrows):
        for j in range(nrows):
            e = system.A[i, j]
            if e == 0 or getattr(e, "is_Number", False):
                continue
            for sym in e.free_symbols - {S}:
                name = by_name.get(sym)
                if name is None or name not in system.values:
                    continue
                d = sp.diff(e, sym).xreplace(subs)
                d0 = complex(d.xreplace({S: sp.Integer(0)}))
                d1 = complex(sp.diff(d, S))
                if name not in patterns:
                    patterns[name] = (
                        np.zeros((nrows, nrows), dtype=complex),
                        np.zeros((nrows, nrows), dtype=complex),
                        float(system.values[name]))
                Pi, Qi, _ = patterns[name]
                Pi[i, j] += d0
                Qi[i, j] += d1

    roots = _den_roots(A)[:n_poles]
    out = []
    used = set()
    for idx, p in enumerate(roots):
        if idx in used:
            continue
        zeta = None
        for j in range(idx + 1, len(roots)):
            if j not in used and abs(np.conj(roots[j]) - p) < 1e-9 * abs(p):
                zeta = float(-p.real / abs(p))
                used.add(j)
                break
        used.add(idx)

        M = P + p * Q
        U, sv, Vh = np.linalg.svd(M)
        v = Vh[-1].conj()
        w = U[:, -1]
        denom = np.conj(w) @ Q @ v
        shares = []
        if denom != 0:
            for name, (Pi, Qi, val) in patterns.items():
                num = np.conj(w) @ (Pi + p * Qi) @ v
                dp = -num / denom
                shares.append((name, abs(val * dp / p)))
        tot = sum(s for _, s in shares) or 1.0
        shares.sort(key=lambda t: -t[1])
        owners = [PoleOwner(symbol=n_, share=s_ / tot)
                  for n_, s_ in shares[:top]]

        att = PoleAttribution(f_hz=float(abs(p) / (2 * np.pi)),
                              damping=zeta, owners=owners)
        if verify and owners and denom != 0:
            name = owners[0].symbol
            Pi, Qi, val = patterns[name]
            dp = -(np.conj(w) @ (Pi + p * Qi) @ v) / denom
            att.predicted_rel = float(abs(dp * (eps * val) / p))
            A2 = system.A.xreplace(
                {**subs, system.symbols[name]:
                 sp.Float(val * (1 + eps))})
            try:
                moved = _den_roots(A2)
                near = moved[np.argmin(np.abs(moved - p))]
                att.actual_rel = float(abs((near - p) / p))
                lo, hi = 0.5 * att.predicted_rel, 2.0 * att.predicted_rel
                att.verified = (lo <= att.actual_rel <= hi
                                or (att.predicted_rel < 1e-9
                                    and att.actual_rel < 1e-9))
            except Exception:
                att.verified = False
        out.append(att)
    return out
