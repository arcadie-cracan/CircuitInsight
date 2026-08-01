"""Standard multistage form: the transfer function as designers write it.

An exact rational H(s) = N(s)/D(s) is correct but not readable. The
literature on multistage amplifier design states results in a fixed
template -- DC gain, a dominant pole, a non-dominant cluster, zeros --
with a short symbolic formula per root (Aminzadeh, Grasso & Palumbo,
IEEE Access 2022, derive exactly these by hand for MCNR / NMC / RNMC).
This module produces that template automatically from an exact solve.

Two steps, both error-controlled:

1. ROOT SPLITTING (the classical widely-separated-roots approximation,
   as sharpened by Behmanesh-Fard et al., Mathematics 2023): with
   D(s) = a0 + a1 s + ... + an s^n and well-separated roots,

       p1 ~ -a0/a1 ,   p2 ~ -a1/a2 ,   ... ,   pk ~ -a_{k-1}/a_k

   Each formula is checked against the EXACT numeric root and reported
   with its displacement, so a formula that does not hold (clustered or
   complex roots) is marked unreliable instead of being presented.

2. PER-ROOT SIMPLIFICATION under a displacement budget: additive terms
   of each ratio are pruned smallest-first (magnitudes from the
   operating point, as in analysis.simplify), keeping only what the
   budget allows the root to move. Where the papers hand-simplify with
   design intuition, this prunes with the OP in hand and states the
   error it accepted.

The exact TF is never modified: the template is a VIEW of it, and every
number in it is checked against the exact roots.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import sympy as sp

from ..engine.mna import MnaError, TransferFunction
from .simplify import _tidy

S = sp.Symbol("s")


@dataclass
class RootFormula:
    """One pole or zero: where it is, and the short formula for it."""
    kind: str                     # "pole" | "zero"
    index: int                    # 0 = dominant (lowest |root|)
    f_hz: float                   # exact numeric magnitude, Hz
    damping: float | None         # zeta for a complex pair, else None
    expr: sp.Expr | None          # symbolic magnitude in rad/s, simplified
    displacement: float           # |expr/exact - 1| at the OP
    reliable: bool                # displacement within the budget

    def describe(self) -> str:
        from ..units import eng, eng_expr_str

        where = f"{self.kind} {self.index}: {eng(self.f_hz, 'Hz')}"
        if self.damping is not None:
            where += f" (complex pair, zeta {self.damping:.3g})"
        if self.expr is None:
            return where + "  [no closed form]"
        tail = "" if self.reliable else "  [UNRELIABLE]"
        return (f"{where} ~ {eng_expr_str(self.expr)} rad/s "
                f"({self.displacement * 100:.2g}% off){tail}")


@dataclass
class TemplateForm:
    """H(s) in the standard multistage form."""
    dc_gain: sp.Expr | None
    dc_gain_value: float
    poles: list[RootFormula] = field(default_factory=list)
    zeros: list[RootFormula] = field(default_factory=list)
    gbw_hz: float | None = None
    budget: float = 0.05

    def describe(self) -> str:
        from ..units import eng, eng_expr_str

        lines = []
        a0 = eng(self.dc_gain_value, sig=4)
        if self.dc_gain is not None:
            lines.append(f"A0 = {eng_expr_str(self.dc_gain)}  ({a0})")
        else:
            lines.append(f"A0 = {a0}")
        if self.gbw_hz is not None:
            lines.append(f"GBW ~ {eng(self.gbw_hz, 'Hz')} "
                         f"(A0 x dominant pole)")
        for r in self.poles:
            lines.append("  " + r.describe())
        for r in self.zeros:
            lines.append("  " + r.describe())
        return "\n".join(lines)


def _terms(expr: sp.Expr) -> list[sp.Expr]:
    return list(sp.Add.make_args(sp.expand(expr)))


def _prune_ratio(num: sp.Expr, den: sp.Expr, subs: dict, exact: complex,
                 budget: float) -> tuple[sp.Expr | None, float]:
    """Prune additive terms of num and den (smallest OP magnitude first)
    while |ratio/exact - 1| stays within `budget`. Returns the simplified
    ratio and its final displacement."""
    scale = None
    for _e in _terms(num) + _terms(den):
        m = sp.Abs(sp.N(_e.xreplace(subs), 30))
        if scale is None or m > scale:
            scale = m
    if scale is None or scale == 0:
        scale = sp.Integer(1)

    def value(e) -> complex:
        # scale-normalised: exact coefficients of a large circuit overflow
        # float64 term by term, while the ratios that decide pruning do not
        return complex(sp.N(e.xreplace(subs) / scale, 30))

    def disp(n_terms, d_terms) -> float:
        if not n_terms or not d_terms:
            return float("inf")
        try:
            nv = sum(value(x) for x in n_terms)
            dv = sum(value(x) for x in d_terms)
        except (TypeError, ValueError):
            return float("inf")
        if dv == 0 or exact == 0:
            return float("inf")
        return abs(nv / dv / exact - 1.0)   # the scale cancels in nv/dv

    nt, dt = _terms(num), _terms(den)
    if disp(nt, dt) > budget:            # the exact ratio itself is off
        return None, disp(nt, dt)

    for terms in (nt, dt):
        def mag(x):
            try:
                return abs(value(x))
            except Exception:
                return float("inf")

        # smallest-magnitude first, over the TERM OBJECTS (the working list
        # is mutated, so precomputed indices would go stale)
        for cand in sorted(list(terms), key=mag):
            if len(terms) <= 1:
                break
            terms.remove(cand)
            if disp(nt, dt) > budget:
                terms.append(cand)       # load-bearing: put it back
    return (_drop_unit_floats(_normalize(sp.cancel(
        sp.Add(*nt) / sp.Add(*dt)))), disp(nt, dt))


def _normalize(expr: sp.Expr) -> sp.Expr:
    """Pull the numeric scale out front so the shape is readable: hybrid
    coefficients span tens of orders of magnitude, and `cancel` leaves
    that spread inside the fraction where it hides the structure."""
    try:
        n, d = sp.fraction(sp.together(expr))

        def scale(e):
            mags = []
            for term in sp.Add.make_args(sp.expand(e)):
                c = term.as_coeff_Mul()[0]
                if c.is_number:
                    mags.append(abs(complex(c)))
            return max(mags) if mags else None

        sn, sd = scale(n), scale(d)
        if not sn or not sd:
            return _tidy(expr)
        k = sn / sd
        # when the scale is already ~1 leave the expression ALONE: dividing
        # through by a Python float would turn an exact textbook formula
        # (1/(C1*R1)) into a Float one (1.0/(C1*R1))
        if abs(k - 1.0) <= 1e-6:
            return _tidy(expr)
        body = sp.cancel(sp.expand(n / sn) / sp.expand(d / sd))
        return sp.Float(k, 4) * _tidy(body)
    except Exception:
        return _tidy(expr)


def _drop_unit_floats(expr: sp.Expr) -> sp.Expr:
    """1.0*x -> x: sympy keeps a Float(1.0) coefficient that makes an exact
    formula look approximate (and blocks exact comparison)."""
    return expr.xreplace({sp.Float(1.0): sp.Integer(1),
                          sp.Float(-1.0): sp.Integer(-1)})


def _safe(fn, x) -> bool:
    try:
        fn(x)
        return True
    except Exception:
        return False


def _root_formulas(poly: sp.Poly, roots: np.ndarray, kind: str,
                   subs: dict, budget: float) -> list[RootFormula]:
    a = list(reversed(poly.all_coeffs()))            # ascending powers
    order = np.argsort(np.abs(roots))
    out = []
    used = set()
    for idx, ri in enumerate(order):
        r = roots[ri]
        if ri in used:
            continue
        # complex pair: report once, with damping; no scalar ratio formula
        pair = None
        if abs(r.imag) > 1e-12 * abs(r):
            for rj in order:
                if rj != ri and rj not in used and abs(
                        np.conj(roots[rj]) - r) < 1e-9 * abs(r):
                    pair = rj
                    break
        f_hz = float(abs(r) / (2 * np.pi))
        zeta = None
        if pair is not None:
            zeta = float(-r.real / abs(r))
            used.add(pair)
        used.add(ri)
        expr = None
        disp = None
        if (pair is None and idx + 1 < len(a)
                and a[idx] != 0 and a[idx + 1] != 0):
            # |root| ~ |a_k / a_{k+1}| for well-separated roots
            expr, disp = _prune_ratio(a[idx], a[idx + 1], subs,
                                      complex(abs(r)), budget)
        out.append(RootFormula(
            kind=kind, index=len(out), f_hz=f_hz, damping=zeta,
            expr=expr, displacement=float(disp) if disp is not None else 0.0,
            reliable=bool(expr is not None and disp is not None
                          and disp <= budget)))
    return out


def template_form(tf: TransferFunction, *, budget: float = 0.05,
                  max_roots: int = 6) -> TemplateForm:
    """Render `tf` in the standard multistage form: DC gain, GBW, and a
    short symbolic formula per pole/zero, each checked against the exact
    numeric root and reported with its displacement.

    `budget` is the fractional root displacement allowed while pruning
    each formula (0.05 = the root may move 5%)."""
    subs = tf._subs_map()
    missing = sorted(str(x) for x in (tf.expr.free_symbols - {tf.s}) - set(subs))
    if missing:
        raise MnaError(f"template: no numeric value for symbols: {missing}")

    num, den = tf.num_den
    dc = None
    try:
        dc_expr = sp.cancel(tf.dc_gain())
    except Exception:
        dc_expr = None
    dc_val = float("nan")
    if dc_expr is not None:
        try:
            dc_val = float(sp.re(sp.N(dc_expr.xreplace(subs))))
        except Exception:
            pass
        if dc_expr.free_symbols - {tf.s}:
            # the raw A0 of a hybrid solve is a ratio of huge exact
            # rationals; prune it under the same budget so the template
            # states a formula a designer can read
            dn, dd = sp.fraction(sp.together(dc_expr))
            pruned, _ = _prune_ratio(dn, dd, subs, complex(dc_val), budget)
            dc = _drop_unit_floats(
                _tidy(sp.factor(pruned if pruned is not None else dc_expr)))

    def roots_of(poly):
        # normalise by the largest coefficient before converting: on a big
        # circuit the raw ones overflow float64, and without this the
        # isfinite guard below would quietly report a circuit with no poles
        exact = [sp.N(x.xreplace(subs), 30) for x in poly.all_coeffs()]
        mags = [sp.Abs(x) for x in exact if x != 0]
        scale = max(mags) if mags else sp.Integer(1)
        c = [complex(sp.N(x / scale, 30)) for x in exact]
        c = np.array(c, dtype=complex)
        if c.size < 2 or not np.isfinite(c).all():
            return np.array([], dtype=complex)
        r = np.roots(c)
        r = r[np.abs(r) > 0]
        # ASCENDING by frequency. np.roots returns companion-matrix
        # eigenvalues in no useful order, and the caller keeps only the
        # first max_roots -- so without this the template reports an
        # arbitrary handful of a 17-pole circuit's poles as though they
        # were the low-frequency ones, and GBW = A0 * poles[0] is
        # meaningless. It also aligns the roots with the coefficient
        # ratios |a_k / a_{k+1}| that _root_formulas fits by rank.
        return r[np.argsort(np.abs(r))]

    p_roots, z_roots = roots_of(den), roots_of(num)
    poles = _root_formulas(den, p_roots[:max_roots], "pole", subs, budget)
    zeros = _root_formulas(num, z_roots[:max_roots], "zero", subs, budget)

    gbw = None
    if poles and np.isfinite(dc_val) and dc_val != 0:
        gbw = float(abs(dc_val) * poles[0].f_hz)
    return TemplateForm(dc_gain=dc, dc_gain_value=dc_val, poles=poles,
                        zeros=zeros, gbw_hz=gbw, budget=budget)
