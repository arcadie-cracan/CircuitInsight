"""Per-capacitor bandwidth attribution via zero-value time constants (ZVTC).

For D(s) = a0 + a1*s + ..., the ratio a1/a0 equals the sum of zero-value
time constants sum_i tau_i (tau_i = Ri0 * Ci), and 1/(a1/a0) estimates the
dominant pole of a well-separated system. Because every reactive stamp is
rank-1, a1 is MULTILINEAR in the reactive element values, so each element's
tau_i is obtained exactly as a1(nominal) - a1(that element zeroed) — one
cheap numeric determinant per element, no symbolic solve.

Exactness self-check: sum_i tau_i must equal a1/a0 to the last rational
digit; a mismatch raises instead of reporting wrong attributions.

This answers the design/teaching question "which capacitance limits the
bandwidth, at this operating point?" (education scenario 2 in the paper).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import sympy as sp

from ..engine.mna import MnaError, S, _det, build_mna, symbol_name


@dataclass
class BandwidthRow:
    name: str          # symbol name (matched devices share one row)
    value: float       # element value (F or H)
    tau: float         # zero-value time constant contribution (s)
    share: float       # tau / tau_total


@dataclass
class BandwidthReport:
    rows: list[BandwidthRow]          # sorted by |tau|, descending
    tau_total: float                  # = a1/a0 exactly
    f_zvtc: float                     # 1 / (2 pi tau_total)
    f_dominant: float                 # |smallest denominator root| / 2 pi

    @property
    def validity(self) -> float:
        """f_zvtc / f_dominant: ~1 for a well-separated dominant pole."""
        return self.f_zvtc / self.f_dominant

    def report(self) -> str:
        lines = [
            f"ZVTC bandwidth attribution: sum(tau) = {self.tau_total:.4g} s "
            f"-> f_-3dB(ZVTC) ~ {self.f_zvtc:.4g} Hz "
            f"(true dominant pole {self.f_dominant:.4g} Hz, "
            f"estimate/true = {self.validity:.2f})",
        ]
        for r in self.rows:
            lines.append(
                f"  {r.name:24s} {r.value:10.3g}  tau = {r.tau:10.3g} s"
                f"  ({100 * r.share:5.1f} %)"
            )
        return "\n".join(lines)


def bandwidth_contributions(analyzer) -> BandwidthReport:
    system = build_mna(analyzer.primitives, analyzer.flat.ground,
                       None, analyzer._alias)

    reactive: dict[str, float] = {}
    for p in analyzer.primitives:
        if p.kind in ("c", "l", "cx"):
            reactive[symbol_name(p, analyzer._alias)] = None

    def den_poly(zeroed: str | None) -> sp.Poly:
        subs = {}
        for name, symb in system.symbols.items():
            if name == zeroed:
                subs[symb] = sp.Integer(0)
                continue
            v = system.values.get(name)
            if v is None:
                raise MnaError(
                    f"bandwidth report: no numeric value for {name}")
            subs[symb] = sp.Rational(repr(v))
        return sp.Poly(_det(system.A.xreplace(subs)), S)

    base = den_poly(None)
    coeffs = list(reversed(base.all_coeffs()))          # ascending in s
    if not coeffs or coeffs[0] == 0:
        raise MnaError("bandwidth report: singular network at DC")
    a0 = coeffs[0]
    a1 = coeffs[1] if len(coeffs) > 1 else sp.Integer(0)
    tau_total = a1 / a0

    rows = []
    tau_sum = sp.Integer(0)
    for name in reactive:
        c = list(reversed(den_poly(name).all_coeffs()))
        if c[0] != a0:
            raise MnaError(
                f"bandwidth report: a0 changed when zeroing {name} — "
                f"bookkeeping bug")
        a1v = c[1] if len(c) > 1 else sp.Integer(0)
        tau = (a1 - a1v) / a0
        tau_sum += tau
        rows.append((name, tau))

    if sp.simplify(tau_sum - tau_total) != 0:
        raise MnaError(
            "bandwidth report: sum of per-element time constants does not "
            "reproduce a1/a0 — bookkeeping bug")

    tt = float(tau_total)
    roots = np.roots([complex(x) for x in base.all_coeffs()])
    roots = roots[np.abs(roots) > 0]
    f_dom = float(np.min(np.abs(roots))) / (2 * np.pi)

    out = [
        BandwidthRow(
            name=n,
            value=float(system.values.get(n, float("nan"))),
            tau=float(t),
            share=float(t / tau_total) if tt else 0.0,
        )
        for n, t in rows
    ]
    out.sort(key=lambda r: -abs(r.tau))
    return BandwidthReport(
        rows=out,
        tau_total=tt,
        f_zvtc=1.0 / (2 * np.pi * tt) if tt else float("inf"),
        f_dominant=f_dom,
    )
