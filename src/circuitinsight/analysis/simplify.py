"""Numeric-guided simplification ("simplification after generation").

Every symbol carries its operating-point value, so each additive term in each
coefficient of N(s)/D(s) has a magnitude. Terms are pruned smallest-first,
per coefficient, and the pruned TF is verified against the original over a
log-frequency grid: the result is the shortest expression that stays inside
an explicit magnitude/phase error budget. What survives is factored and
reported as named results (A0, dominant pole/zero).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import sympy as sp

from ..engine.mna import MnaError, TransferFunction

S = sp.Symbol("s")


def _tidy(expr: sp.Expr, digits: int = 4) -> sp.Expr:
    """Replace unwieldy exact-Rational coefficients (hybrid-mode artifacts)
    with short Floats. Small numbers (exponents, 1/2 from baluns, unit
    coefficients) are left exact, so fully-symbolic expressions pass through
    unchanged."""
    return expr.replace(
        lambda x: x.is_Number and x.is_Rational
        and (abs(x) > 100 or (x != 0 and abs(x) < sp.Rational(1, 100))),
        lambda x: sp.Float(x, digits),
    )


@dataclass
class SimplifiedTF(TransferFunction):
    original: TransferFunction | None = None
    achieved_mag_err_db: float = 0.0
    achieved_phase_err_deg: float = 0.0
    band_hz: tuple[float, float] = (0.0, 0.0)

    # ------------------------------------------------------ named results
    def dc_gain_expr(self) -> sp.Expr:
        return _tidy(sp.factor(sp.cancel(self.dc_gain())))

    def _edge_ratio(self, poly: sp.Poly) -> sp.Expr | None:
        """a0/a1 of a polynomial in s (rad/s of the dominant root when the
        roots are well separated)."""
        a = list(reversed(poly.all_coeffs()))          # ascending powers
        if len(a) >= 2 and a[0] != 0 and a[1] != 0:
            return _tidy(sp.factor(sp.cancel(a[0] / a[1])))
        return None

    def dominant_pole_expr(self) -> sp.Expr | None:
        """Symbolic dominant-pole magnitude in rad/s (a0/a1 of the
        denominator). Valid when the pole separation ratio is large."""
        return self._edge_ratio(self.num_den[1])

    def dominant_zero_expr(self) -> sp.Expr | None:
        return self._edge_ratio(self.num_den[0])

    def pole_separation(self) -> float:
        p = self.poles()
        return float(abs(p[1]) / abs(p[0])) if len(p) >= 2 else float("inf")

    def report(self) -> str:
        subs = self._subs_map()

        def val(e):
            return complex(e.xreplace(subs))

        lines = []
        n_orig = len(self.original.expr.free_symbols) if self.original else 0
        lines.append(
            f"simplified within {self.achieved_mag_err_db:.3f} dB / "
            f"{self.achieved_phase_err_deg:.2f} deg over "
            f"{self.band_hz[0]:.3g}..{self.band_hz[1]:.3g} Hz"
            + (f"  (symbols: {n_orig} -> {len(self.expr.free_symbols) - 1})"
               if self.original else "")
        )
        a0 = self.dc_gain_expr()
        v = val(a0)
        lines.append(f"A0   = {a0}")
        lines.append(f"     = {v.real:.4g}  ({20 * np.log10(abs(v)):.2f} dB)")
        p1 = self.dominant_pole_expr()
        if p1 is not None:
            f1 = abs(val(p1)) / (2 * np.pi)
            lines.append(f"p1   = ({p1}) / 2pi")
            lines.append(f"     = {f1:.4g} Hz   (separation x{self.pole_separation():.1f})")
            lines.append(f"GBW  ~ {abs(v) * f1:.4g} Hz")
        z1 = self.dominant_zero_expr()
        if z1 is not None:
            lines.append(f"z1   = ({z1}) / 2pi = {abs(val(z1)) / (2 * np.pi):.4g} Hz")
        return "\n".join(lines)


def _prune_poly(poly: sp.Poly, eps: float, subs: dict) -> dict[int, sp.Expr]:
    """Drop the smallest additive terms of each coefficient, keeping the
    dropped total below eps * |coefficient value|."""
    out: dict[int, sp.Expr] = {}
    for monom, c in poly.terms():
        k = monom[0]
        terms = list(sp.Add.make_args(sp.expand(c)))
        if len(terms) <= 1:
            out[k] = c
            continue
        vals = [complex(t.xreplace(subs)) for t in terms]
        total = abs(sum(vals))
        if total == 0.0:
            out[k] = c
            continue
        order = sorted(range(len(terms)), key=lambda i: abs(vals[i]))
        keep = set(range(len(terms)))
        dropped = 0.0
        for i in order:
            if dropped + abs(vals[i]) <= eps * total:
                dropped += abs(vals[i])
                keep.discard(i)
            else:
                break
        out[k] = sp.Add(*[terms[i] for i in sorted(keep)])
    return out


def _to_expr(coeffs: dict[int, sp.Expr]) -> sp.Expr:
    return sp.Add(*[c * S**k for k, c in coeffs.items() if c != 0])


def simplify_tf(
    tf: TransferFunction,
    mag_tol_db: float = 1.0,
    phase_tol_deg: float = 5.0,
    fmin: float = 10.0,
    fmax: float = 1e10,
    npoints: int = 40,
) -> SimplifiedTF:
    subs = tf._subs_map()
    missing = sorted(str(x) for x in (tf.expr.free_symbols - {tf.s}) - set(subs))
    if missing:
        raise MnaError(f"simplify: no numeric value for symbols: {missing}")

    freq = np.logspace(np.log10(fmin), np.log10(fmax), npoints)
    w = 2j * np.pi * freq
    h0 = tf.numeric(freq)
    mask = np.abs(h0) > np.max(np.abs(h0)) * 1e-9

    num, den = tf.num_den

    def evaluate(expr: sp.Expr) -> np.ndarray:
        fn = sp.lambdify(tf.s, expr.xreplace(subs), "numpy")
        out = np.asarray(fn(w), dtype=complex)
        return np.full(w.shape, complex(out)) if out.shape != w.shape else out

    def verify(expr: sp.Expr) -> tuple[bool, float, float]:
        hp = evaluate(expr)
        m = mask & (np.abs(hp) > 0)
        if not m.any():
            return False, np.inf, np.inf
        mag = np.max(np.abs(20 * np.log10(np.abs(hp[m]) / np.abs(h0[m]))))
        ph = np.max(np.abs(np.degrees(np.angle(hp[m] * np.conj(h0[m])))))
        return (mag <= mag_tol_db and ph <= phase_tol_deg), float(mag), float(ph)

    tol_lin = 10 ** (mag_tol_db / 20) - 1
    result = None
    for eps in (2 * tol_lin, tol_lin, tol_lin / 2, tol_lin / 4, tol_lin / 10,
                tol_lin / 40, 0.0):
        nd = _prune_poly(num, eps, subs)
        dd = _prune_poly(den, eps, subs)
        expr = sp.cancel(_to_expr(nd) / _to_expr(dd))
        ok, mag, ph = verify(expr)
        if ok:
            result = (expr, mag, ph)
            break
    if result is None:                                  # eps=0 must pass
        raise MnaError("simplify: even the unpruned TF failed verification")

    expr, mag, ph = result
    n, d = sp.fraction(sp.together(expr))
    n, d = sp.expand(n), sp.expand(d)
    # hybrid-mode rationals carry a huge common scale; normalize it away
    cmax = max(
        (abs(t.as_coeff_Mul()[0]) for t in sp.Add.make_args(d)),
        default=sp.Integer(1),
    )
    if cmax not in (0, 1):
        n, d = sp.expand(n / cmax), sp.expand(d / cmax)
    expr = _tidy(sp.factor(n) / sp.factor(d))

    return SimplifiedTF(
        expr=expr,
        values=dict(tf.values),
        symbols=dict(tf.symbols),
        original=tf,
        achieved_mag_err_db=mag,
        achieved_phase_err_deg=ph,
        band_hz=(fmin, fmax),
    )
