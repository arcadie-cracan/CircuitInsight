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

from ..engine.mna import MnaError, TransferFunction, numeric_eval
from ..units import eng

S = sp.Symbol("s")


def _tidy(expr: sp.Expr, digits: int = 4) -> sp.Expr:
    """Replace unwieldy exact-Rational coefficients (hybrid-mode artifacts)
    with short Floats. Small numbers (exponents, 1/2 from baluns, unit
    coefficients) are left exact, so fully-symbolic expressions pass through
    unchanged."""
    return expr.replace(
        lambda x: x.is_Number and x.is_Rational
        and (abs(x) > 100 or (x != 0 and abs(x) < sp.Rational(1, 100))
             # a near-1 ratio of two 35-digit integers is just as unwieldy
             # as a large one -- magnitude alone cannot spot it
             or abs(x.p) > 10 ** 12 or x.q > 10 ** 12),
        lambda x: sp.Float(x, digits),
    )


@dataclass
class SimplifiedTF(TransferFunction):
    original: TransferFunction | None = None
    achieved_mag_err_db: float = 0.0
    achieved_phase_err_deg: float = 0.0
    band_hz: tuple[float, float] = (0.0, 0.0)

    def tolerance_margin(self, spread: float = 0.2, top: int = 6,
                         npoints: int = 40) -> float:
        """Worst simplification error (dB) over a PARAMETER REGION, not
        just the nominal point -- the Kolka (MATEC 2021) extension. The
        pruning decision was taken at the nominal OP; a dropped term that
        was marginal there can matter a spread away. Sampled exactly
        (both sides are numeric, so sampling beats their first-order
        bound): the `top` parameters the simplified form is most
        sensitive to are each pushed +/-spread, one at a time, and the
        worst |H_simplified/H_full| over the band is returned.

        One-at-a-time is the honest cheap probe (2*top re-evaluations);
        it bounds single-parameter drift, not worst-case corners."""
        if self.original is None:
            return self.achieved_mag_err_db
        fmin, fmax = self.band_hz
        freq = np.logspace(np.log10(fmin), np.log10(fmax), npoints)

        # rank parameters by how much a spread moves the SIMPLIFIED form:
        # cheap numeric probes on the small expression
        base_s = self.numeric(freq)
        names = [n for n in self.values
                 if self.symbols.get(n) is not None
                 and self.symbols[n] in self.expr.free_symbols]
        scores = []
        for n in names:
            vals = dict(self.values)
            vals[n] = vals[n] * (1 + spread)
            probe = SimplifiedTF(expr=self.expr, values=vals,
                                 symbols=dict(self.symbols))
            with np.errstate(divide="ignore", invalid="ignore"):
                d = np.nanmax(np.abs(
                    20 * np.log10(np.abs(probe.numeric(freq) / base_s))))
            scores.append((float(d), n))
        scores.sort(reverse=True)

        worst = self.achieved_mag_err_db
        for _, n in scores[:top]:
            for sgn in (1 + spread, 1 - spread):
                vals = dict(self.values)
                vals[n] = vals[n] * sgn
                hs = SimplifiedTF(expr=self.expr, values=vals,
                                  symbols=dict(self.symbols)).numeric(freq)
                hf = TransferFunction(
                    expr=self.original.expr, values=vals,
                    symbols=dict(self.original.symbols)).numeric(freq)
                with np.errstate(divide="ignore", invalid="ignore"):
                    db = np.abs(20 * np.log10(np.abs(hs / hf)))
                db = db[np.isfinite(db)]
                if db.size:
                    worst = max(worst, float(db.max()))
        return worst

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
            lines.append(f"     = {eng(f1, 'Hz')}   (separation x{self.pole_separation():.1f})")
            lines.append(f"GBW  ~ {eng(abs(v) * f1, 'Hz')}")
        z1 = self.dominant_zero_expr()
        if z1 is not None:
            lines.append(f"z1   = ({z1}) / 2pi = {eng(abs(val(z1)) / (2 * np.pi), 'Hz')}")
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
        # SCALE FIRST. On a fully numeric solve each term is an exact
        # integer with hundreds of digits, so complex(t) raises
        # "int too large to convert to float" -- but the pruning decision
        # is scale-invariant, so divide by the largest term (a sympy Float
        # keeps the exponent) and convert only the ratios.
        exact = [t.xreplace(subs) for t in terms]
        mags = [sp.Abs(sp.N(e, 30)) for e in exact]
        scale = max(mags)
        if scale == 0:
            out[k] = c
            continue
        vals = [complex(sp.N(e / scale, 30)) for e in exact]
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


#: symbolic cosmetics (cancel of the candidate ratio, factor of the final
#: result) are capped by operation count: on a 9-keep solve the final
#: sp.factor alone was 16 of 41 seconds, purely for display
_COSMETIC_OPS_LIMIT = 4000


def _term_values(poly: sp.Poly, subs: dict):
    """Value every additive term of every coefficient ONCE: per power k,
    (term expressions, complex values on one per-polynomial scale).

    This is what makes the eps search cheap: a prune candidate is a term
    SUBSET, so its response is a subset sum of these values times
    (jw)^k -- pure numpy, no sympy in the loop. The common scale keeps
    hybrid-mode giant integers inside float64; it divides out of every
    ratio the search ever takes."""
    exacts: dict[int, tuple] = {}
    mags_all = []
    for monom, c in poly.terms():
        k = monom[0]
        terms = list(sp.Add.make_args(sp.expand(c)))
        ex = [t.xreplace(subs) for t in terms]
        mags = [sp.Abs(sp.N(e, 30)) for e in ex]
        exacts[k] = (terms, ex)
        mags_all += mags
    scale = max(mags_all) if mags_all else sp.Integer(1)
    if scale == 0:
        scale = sp.Integer(1)
    return {k: (terms, np.array([complex(sp.N(e / scale, 30)) for e in ex],
                                dtype=complex))
            for k, (terms, ex) in exacts.items()}


def _select_terms(vals: np.ndarray, eps: float) -> np.ndarray:
    """The greedy rule of _prune_poly on precomputed values: drop the
    smallest terms while the dropped total stays under eps x |sum|."""
    n = len(vals)
    keep = np.ones(n, dtype=bool)
    if n <= 1 or eps <= 0.0:
        return keep
    total = abs(vals.sum())
    if total == 0.0:
        return keep
    dropped = 0.0
    for i in np.argsort(np.abs(vals)):
        v = abs(vals[i])
        if dropped + v <= eps * total:
            dropped += v
            keep[i] = False
        else:
            break
    return keep


def _subset_response(data: dict, sel: dict, W: np.ndarray) -> np.ndarray:
    """Response of a term subset: sum_k (sum of selected values at k) w^k.
    W[k] holds w**k rows for every power present."""
    out = np.zeros(W.shape[1], dtype=complex)
    for k, (_terms, vals) in data.items():
        c = vals[sel[k]].sum()
        if c != 0:
            out += c * W[k]
    return out


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
        # shared evaluator: it carries the mpmath fallback for coefficients
        # that overflow float64 (large circuits, fully numeric solves)
        return numeric_eval(expr.xreplace(subs), tf.s, freq)

    def verify(expr: sp.Expr) -> tuple[bool, float, float]:
        hp = evaluate(expr)
        m = mask & (np.abs(hp) > 0)
        if not m.any():
            return False, np.inf, np.inf
        mag = np.max(np.abs(20 * np.log10(np.abs(hp[m]) / np.abs(h0[m]))))
        ph = np.max(np.abs(np.degrees(np.angle(hp[m] * np.conj(h0[m])))))
        return (mag <= mag_tol_db and ph <= phase_tol_deg), float(mag), float(ph)

    # value every term ONCE; the eps search is then pure numpy subset
    # sums -- the old loop re-expanded, re-evaluated and re-lambdified the
    # whole rational at every level (41 s on a 4-keep folded cascode; this
    # search is milliseconds, and the symbolic assembly runs once)
    ndata = _term_values(num, subs)
    ddata = _term_values(den, subs)
    kmax = max([*ndata, *ddata], default=0)
    W = np.array([w ** k for k in range(kmax + 1)])
    h0n = _subset_response(ndata, {k: np.ones(len(v[1]), bool)
                                   for k, v in ndata.items()}, W)
    h0d = _subset_response(ddata, {k: np.ones(len(v[1]), bool)
                                   for k, v in ddata.items()}, W)
    ok0 = (h0d != 0)
    h0s = np.where(ok0, h0n / np.where(ok0, h0d, 1.0), np.nan)

    def numeric_check(nsel, dsel) -> bool:
        hn = _subset_response(ndata, nsel, W)
        hd = _subset_response(ddata, dsel, W)
        m = mask & ok0 & (hd != 0)
        if not m.any():
            return False
        hp = hn[m] / hd[m]
        r = hp / h0s[m]
        good = (r != 0) & np.isfinite(r)
        if not good.all():
            return False
        mag = np.max(np.abs(20 * np.log10(np.abs(r))))
        ph = np.max(np.abs(np.degrees(np.angle(r))))
        return mag <= mag_tol_db and ph <= phase_tol_deg

    def assemble(nsel, dsel) -> sp.Expr:
        nd = {k: sp.Add(*[t for t, keepit in zip(terms, nsel[k]) if keepit])
              for k, (terms, _v) in ndata.items()}
        dd = {k: sp.Add(*[t for t, keepit in zip(terms, dsel[k]) if keepit])
              for k, (terms, _v) in ddata.items()}
        e = _to_expr(nd) / _to_expr(dd)
        if sp.count_ops(e) <= _COSMETIC_OPS_LIMIT:
            e = sp.cancel(e)
        return e

    tol_lin = 10 ** (mag_tol_db / 20) - 1
    result = None
    for eps in (2 * tol_lin, tol_lin, tol_lin / 2, tol_lin / 4, tol_lin / 10,
                tol_lin / 40, 0.0):
        nsel = {k: _select_terms(v[1], eps) for k, v in ndata.items()}
        dsel = {k: _select_terms(v[1], eps) for k, v in ddata.items()}
        if eps > 0.0 and not numeric_check(nsel, dsel):
            continue                    # cheap rejection, no symbolic work
        expr = assemble(nsel, dsel)
        ok, mag, ph = verify(expr)      # the official numbers, on the real expr
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
    # factoring is DISPLAY only, and two things make it explode: sheer
    # size (ops), and hybrid-mode GIANT integer coefficients -- sp.factor
    # runs factorint on the polynomial's integer content, which for a
    # ~500-digit content means seconds of Miller-Rabin per call (measured:
    # 15 of 24 s, in sympy.external.ntheory). Neither makes the result
    # more readable, so both skip the factoring.
    huge = cmax > sp.Integer(10) ** 40 or max(
        (abs(t.as_coeff_Mul()[0]) for t in sp.Add.make_args(n)),
        default=sp.Integer(0)) > sp.Integer(10) ** 40
    if cmax not in (0, 1):
        n, d = sp.expand(n / cmax), sp.expand(d / cmax)
    if not huge and sp.count_ops(n) + sp.count_ops(d) <= _COSMETIC_OPS_LIMIT:
        expr = _tidy(sp.factor(n) / sp.factor(d))
    else:
        expr = _tidy(n / d)

    return SimplifiedTF(
        expr=expr,
        values=dict(tf.values),
        symbols=dict(tf.symbols),
        original=tf,
        achieved_mag_err_db=mag,
        achieved_phase_err_deg=ph,
        band_hz=(fmin, fmax),
    )
