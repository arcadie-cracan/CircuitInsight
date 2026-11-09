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

   Exact polynomial factors are read first: a factor such as
   (CA s + gm3) is a root of its own, and ranks are counted inside each
   factor. Two roots a single ratio cannot reach -- a complex pair, or
   closely spaced real roots -- are the roots of the quadratic over the
   next three coefficients, a_k + a_{k+1} s + a_{k+2} s^2 (Hurst et al.,
   TCAS-I 2004, eqs. (29), (56)-(58)): a complex pair as
   w0 = sqrt(a_k/a_{k+2}) and 2 zeta w0 = a_{k+1}/a_{k+2}; a real pair
   from the quadratic's linear factors when it splits, as
   +/- sqrt(-a_k/a_{k+2}) when the middle term can go, else from the
   quadratic formula.

2. PER-ROOT SIMPLIFICATION under a displacement budget: additive terms
   of each ratio are pruned smallest-first (magnitudes from the
   operating point, as in analysis.simplify), keeping only what the
   budget allows the root to move. Where the papers hand-simplify with
   design intuition, this prunes with the OP in hand and states the
   error it accepted.

3. COUPLED ROOTS. A root that is not separated from its neighbours has no
   coefficient ratio or quadratic of its own, so the lines step 1 leaves
   empty (or too long to read) are taken again against the FULL model,
   and each result states its error against that model:

   - a real pole whose sensitivity to one capacitor is about -1 belongs
     to that capacitor's node: 1/(R_node*C), with R_node from the
     determinant's cofactor and the other capacitors open or shorted;
   - otherwise the reactances whose removal barely moves the root go,
     those owning another pole first, and the coefficient reading is
     repeated on the reduced model. C -> 0 in N and D IS the re-solve
     without C, at the cost of a mask over their terms;
   - roots that stay coupled get the polynomial of their cluster.

   A form that is still too long is pruned within its stated error, and
   one that stays long is reported by size instead of being printed.

The exact TF is never modified: the template is a VIEW of it, and every
number in it is checked against the exact roots.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np
import sympy as sp

from ..engine.mna import MnaError, TransferFunction
from .simplify import _tidy



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
    rhp: bool = False             # right-half-plane root
    # complex pair from a quadratic: expr is w0, this is 2*zeta*w0
    zeta_w0: sp.Expr | None = None
    zeta_displacement: float = 0.0
    plusminus: bool = False       # expr is the +/- sqrt of an even quadratic
    # the other root of the same quadratic (real pair): its index, once
    # the roots are numbered, and the object while they are not
    partner: int | None = None
    # how to print expr when sstr would not do: sympy re-evaluates a sqrt
    # on printing and splits it over positive symbols
    display: str | None = None
    legend: str = ""              # what the symbols in `display` stand for
    # errors are against the full model, not the (simplified) TF being read
    full_model: bool = False
    without: tuple[str, ...] = () # read on the model without these reactances
    # the polynomial of the coupled roots this one belongs to, with no
    # formula of its own; a later root of the cluster points at the first
    cluster: str | None = None
    cluster_of: int | None = None
    long_terms: int | None = None # expr is too long to print: its term count
    _partner_ref: object = field(default=None, repr=False, compare=False)

    def describe(self) -> str:
        from ..units import eng, eng_expr_str

        where = f"{self.kind} {self.index}: {eng(self.f_hz, 'Hz')}"
        if self.damping is not None:
            where += f" (complex pair, zeta {self.damping:.3g})"
        if self.rhp:
            where += " RHP"
        versus = " the full model" if self.full_model else ""
        off = f"({self.displacement * 100:.2g}% off{versus})"
        if self.cluster_of is not None:
            return (f"{where}  [coupled, no closed form of its own: a root "
                    f"of {self.kind} {self.cluster_of}'s polynomial]")
        reduced = (f"  [reduced model, without {', '.join(self.without)}]"
                   if self.without else "")
        if self.cluster is not None:
            return (f"{where}  [coupled, no closed form of its own] ~ root "
                    f"of {self.cluster} {off}{reduced}")
        if self.expr is None:
            return where + "  [no closed form]"
        if self.long_terms is not None:
            return (f"{where} ~ [form of {self.long_terms} terms, too long "
                    f"to print: see expr] {off}")
        tail = reduced + ("" if self.reliable else "  [UNRELIABLE]")
        if (self.plusminus and self.partner is not None
                and self.partner < self.index):
            return (f"{where} ~ the other root of {self.kind} "
                    f"{self.partner}'s +/- pair {off}{tail}")
        body = self.display or eng_expr_str(self.expr)
        if self.plusminus:
            body = "+/- " + body
        if self.zeta_w0 is not None:
            return (f"{where} ~ w0 = {body} rad/s {off}, 2*zeta*w0 = "
                    f"{eng_expr_str(self.zeta_w0)} rad/s "
                    f"(zeta {self.zeta_displacement * 100:.2g}% off){tail}")
        legend = self.legend
        if legend and self.partner is not None and self.partner < self.index:
            legend = f", b and c as for {self.kind} {self.partner}"
        return f"{where} ~ {body} rad/s {off}{legend}{tail}"


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


def _numeric_roots(poly: sp.Poly, subs: dict) -> np.ndarray:
    """Nonzero numeric roots of `poly` at the OP, ascending by |root|."""
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


_FACTOR_MAX_TERMS = 200      # sympy's multivariate factoring is not cheap


def _n_terms(exprs) -> int:
    return sum(len(sp.Add.make_args(sp.expand(e))) for e in exprs)


def _symbolic_factors(expr: sp.Expr, s: sp.Symbol) -> list[sp.Poly] | None:
    """The factors of `expr` in s (with multiplicity, constants and the
    content dropped), or None when factoring is not worth trying: Float
    coefficients (factoring over RR says nothing), no symbol besides s
    (a numeric factor has no formula to show), or too many terms."""
    if (expr.has(sp.Float) or not expr.free_symbols - {s}
            or _n_terms([expr]) > _FACTOR_MAX_TERMS):
        return None
    try:
        _, fl = sp.factor_list(sp.expand(expr))
    except Exception:
        return None
    out = []
    for f, m in fl:
        if f.has(s):
            p = sp.Poly(f, s)
            out += [p] * int(m)
    return out


def _exact_factors(poly: sp.Poly) -> list[sp.Poly]:
    """Split `poly` into the exact polynomial factors it carries, so that
    a factor such as (CA*s + gm3) is read as a root of its own instead of
    being matched against coefficient ratios of the expanded product.
    s^m factors (roots at the origin, which the root list excludes) are
    dropped."""
    s = poly.gen
    a = list(reversed(poly.all_coeffs()))
    m = next((k for k, c in enumerate(a) if c != 0), 0)
    base = sp.Poly(list(reversed(a[m:])), s) if m else poly
    if base.degree() < 2:
        return [base] if base.degree() >= 1 else []
    found = _symbolic_factors(base.as_expr(), s)
    if not found:
        return [base]
    found = [f for f in found if f.degree() >= 1 and f.as_expr() != s]
    return found or [base]


def _fit_values(exprs, subs):
    """A scale-normalised evaluator for terms of `exprs` (exact large
    coefficients overflow float64 term by term; ratios of sums do not)."""
    scale = None
    for e in exprs:
        for t in _terms(e):
            mg = sp.Abs(sp.N(t.xreplace(subs), 30))
            if scale is None or mg > scale:
                scale = mg
    if scale is None or scale == 0:
        scale = sp.Integer(1)

    def value(e) -> complex:
        return complex(sp.N(e.xreplace(subs) / scale, 30))
    return value


def _quad_roots(c0: complex, c1: complex, c2: complex):
    import cmath

    if c2 == 0:
        return None
    b, c = c1 / c2, c0 / c2
    d = cmath.sqrt(b * b / 4 - c)
    return (-b / 2 - d, -b / 2 + d)


def _pair_disp(q, r) -> float:
    """Worst displacement of the approximate roots q from the exact r,
    over the better of the two pairings, relative to each root's distance
    from the jw axis: near its peak a complex pair moves the response by
    about |dr| / |Re r|, so a lightly damped pair gets a tighter budget.
    For a real root this is the plain relative displacement."""
    if q is None:
        return float("inf")
    r0, r1 = complex(r[0]), complex(r[1])
    d0, d1 = abs(r0.real) or abs(r0), abs(r1.real) or abs(r1)
    return min(max(abs(q[0] - r0) / d0, abs(q[1] - r1) / d1),
               max(abs(q[1] - r0) / d0, abs(q[0] - r1) / d1))


def _single(num, den, r: complex, subs, budget):
    """|root| ~ |a_k / a_{k+1}|, pruned. A real RHP root has a negative
    ratio, so its magnitude is -a_k / a_{k+1}."""
    if r.real > 0 and abs(r.imag) <= 1e-12 * abs(r):
        num = -num
    return _prune_ratio(num, den, subs, complex(abs(r)), budget)


def _quadratic_pair(c, r, kind, subs, budget) -> list[RootFormula] | None:
    """Closed forms for two roots r = (r0, r1) of c0 + c1 s + c2 s^2: a
    complex pair as w0 = sqrt(c0/c2) and 2 zeta w0 = c1/c2; a real pair
    from the exact linear factors when the quadratic splits, else as
    +/- sqrt(-c0/c2) when c1 can be dropped, else from the quadratic
    formula. None when the quadratic does not hold these two roots
    within `budget` (relative root displacement)."""
    c0, c1, c2 = c
    s = sp.Symbol("s")
    value = _fit_values(c, subs)

    def roots_of(t0, t1, t2):
        if not t0 or not t2:
            return None
        try:
            v = [sum((value(x) for x in t), 0j) for t in (t0, t1, t2)]
        except (TypeError, ValueError):
            return None
        return _quad_roots(*v)

    t = [_terms(c0), _terms(c1), _terms(c2)]
    if _pair_disp(roots_of(*t), r) > budget:
        return None
    complex_pair = abs(r[0].imag) > 1e-12 * abs(r[0])

    def mk(**kw):
        kw.setdefault("damping", None)
        return RootFormula(kind=kind, index=0, **kw)

    def f_hz(x):
        return float(abs(x) / (2 * np.pi))

    def rhp(x):
        return bool(x.real > 0)

    if not complex_pair:
        # exact linear factors of the quadratic: each root is then a
        # single ratio, as for a well-separated root
        fs = _symbolic_factors(c0 + c1 * s + c2 * s ** 2, s)
        if fs and len(fs) == 2 and all(f.degree() == 1 for f in fs):
            out, left = [], [complex(r[0]), complex(r[1])]
            for f in fs:
                l1, l0 = f.all_coeffs()
                v = -complex(sp.N((l0 / l1).xreplace(subs), 30))
                rr = min(left, key=lambda x: abs(x - v))
                left.remove(rr)
                expr, disp = _single(l0, l1, rr, subs, budget)
                if expr is None:
                    break
                out.append(mk(f_hz=f_hz(rr), expr=expr,
                              displacement=float(disp),
                              reliable=disp <= budget, rhp=rhp(rr)))
            if len(out) == 2:
                return sorted(out, key=lambda x: x.f_hz)

    # prune: the middle coefficient may go entirely (an even quadratic),
    # the outer two keep at least one term; smallest OP magnitude first
    def disp():
        return _pair_disp(roots_of(*t), r)

    def mag(x):
        try:
            return abs(value(x))
        except Exception:
            return float("inf")

    for k in (1, 0, 2):
        keep = 0 if k == 1 else 1
        for cand in sorted(list(t[k]), key=mag):
            if len(t[k]) <= keep:
                break
            t[k].remove(cand)
            if disp() > budget:
                t[k].append(cand)
    d_all = disp()
    q = roots_of(*t)
    n0, n1, n2 = (sp.Add(*x) for x in t)

    def ratio(a, b):
        e = _normalize(sp.cancel(a / b))
        # a hybrid coefficient within rounding of 1 prints as "1.0*"
        return e.xreplace({x: sp.Integer(1 if x > 0 else -1)
                           for x in e.atoms(sp.Float)
                           if abs(abs(float(x)) - 1) < 5e-5})

    from ..units import eng_expr_str

    def sqrt_text(e):
        return f"sqrt({eng_expr_str(e)})"

    if complex_pair:
        w0sq = ratio(n0, n2)
        w0v = abs(q[0] * q[1]) ** 0.5
        bv = -(q[0] + q[1]).real
        w0_exact = abs(r[0])
        zeta_exact = -r[0].real / abs(r[0])
        zeta_v = bv / (2 * w0v)
        return [mk(f_hz=f_hz(r[0]), damping=float(zeta_exact),
                   expr=sp.sqrt(w0sq), display=sqrt_text(w0sq),
                   displacement=float(abs(w0v / w0_exact - 1)),
                   reliable=d_all <= budget, rhp=rhp(r[0]),
                   zeta_w0=ratio(n1, n2) if t[1] else sp.Integer(0),
                   zeta_displacement=float(
                       abs(zeta_v / zeta_exact - 1) if zeta_exact else 0.0))]

    # real pair: match the approximate roots to the exact ones
    ra, rb = complex(r[0]), complex(r[1])
    if (abs(q[0] - ra) / abs(ra) + abs(q[1] - rb) / abs(rb)
            > abs(q[1] - ra) / abs(ra) + abs(q[0] - rb) / abs(rb)):
        q = (q[1], q[0])
    if not t[1]:
        q_e = ratio(-n0, n2)
        out = [mk(f_hz=f_hz(x), expr=sp.sqrt(q_e), display=sqrt_text(q_e),
                  displacement=float(abs(abs(y) / abs(x) - 1)),
                  reliable=d_all <= budget, rhp=rhp(x), plusminus=True)
               for x, y in ((ra, q[0]), (rb, q[1]))]
    else:
        b, c = ratio(n1, n2), ratio(n0, n2)
        sq = sp.sqrt(b ** 2 / 4 - c)
        legend = (f", roots of s^2 + b*s + c with b = {eng_expr_str(b)}"
                  f" rad/s, c = {eng_expr_str(c)} (rad/s)^2")
        mid = ((q[0] + q[1]) / 2).real                # = -b/2
        out = []
        for x, y in ((ra, q[0]), (rb, q[1])):
            # the branch of -b/2 +/- sqrt(b^2/4 - c) this root is on, then
            # its magnitude (a LHP root is the negated value)
            branch = 1 if y.real >= mid else -1
            sgn = -1 if y.real < 0 else 1
            e = sgn * (-b / 2 + branch * sq)
            hb = "b/2" if sgn < 0 else "-b/2"
            op = "+" if sgn * branch > 0 else "-"
            text = (f"{hb} {op} sqrt(b^2/4 - c)" if sgn < 0
                    else ("sqrt(b^2/4 - c) - b/2" if op == "+"
                          else "-b/2 - sqrt(b^2/4 - c)"))
            out.append(mk(f_hz=f_hz(x), expr=e, display=text, legend=legend,
                          displacement=float(abs(abs(y) / abs(x) - 1)),
                          reliable=d_all <= budget, rhp=rhp(x)))
    out[0]._partner_ref, out[1]._partner_ref = out[1], out[0]
    return sorted(out, key=lambda x: x.f_hz)


def _factor_formulas(poly: sp.Poly, roots: np.ndarray, kind: str,
                     subs: dict, budget: float, cutoff: float,
                     only: int | None = None
                     ) -> list[tuple[float, RootFormula]]:
    """Root formulas for one exact factor, by rank: a real root from the
    ratio of adjacent coefficients, a complex pair -- or a real pair the
    ratio misses -- from the quadratic over the next three. `only` reads
    the root of that rank alone."""
    a = list(reversed(poly.all_coeffs()))            # ascending powers
    m = next((k for k, c in enumerate(a) if c != 0), 0)
    a = a[m:]                                        # roots at 0 excluded
    out = []
    used = set()
    n = len(roots)
    for idx in range(n):
        r = roots[idx]
        if idx in used or (only is not None and idx != only):
            continue
        if abs(r) > cutoff:
            break
        # complex pair: report once, with damping
        pair = None
        if abs(r.imag) > 1e-12 * abs(r):
            for j in range(n):
                if j != idx and j not in used and abs(
                        np.conj(roots[j]) - r) < 1e-9 * abs(r):
                    pair = j
                    break
        used.add(idx)
        if pair is not None:
            used.add(pair)
            got = None
            if idx + 2 < len(a) and a[idx] != 0 and a[idx + 2] != 0:
                got = _quadratic_pair((a[idx], a[idx + 1], a[idx + 2]),
                                      (r, roots[pair]), kind, subs, budget)
            if got:
                out += [(abs(r), g) for g in got]
            else:
                out.append((abs(r), RootFormula(
                    kind=kind, index=0, f_hz=float(abs(r) / (2 * np.pi)),
                    damping=float(-r.real / abs(r)), expr=None,
                    displacement=0.0, reliable=False, rhp=bool(r.real > 0))))
            continue
        expr = disp = None
        if idx + 1 < len(a) and a[idx] != 0 and a[idx + 1] != 0:
            # |root| ~ |a_k / a_{k+1}| for well-separated roots
            expr, disp = _single(a[idx], a[idx + 1], r, subs, budget)
        if (expr is None and idx + 1 < n and idx + 1 not in used
                and abs(roots[idx + 1].imag) <= 1e-12 * abs(roots[idx + 1])
                and idx + 2 < len(a) and a[idx] != 0 and a[idx + 2] != 0):
            # closely spaced real roots: the quadratic over the pair
            got = _quadratic_pair((a[idx], a[idx + 1], a[idx + 2]),
                                  (r, roots[idx + 1]), kind, subs, budget)
            if got:
                used.add(idx + 1)
                out += [(g.f_hz * 2 * np.pi, g) for g in got]
                continue
        out.append((abs(r), RootFormula(
            kind=kind, index=0, f_hz=float(abs(r) / (2 * np.pi)),
            damping=None, expr=expr,
            displacement=float(disp) if disp is not None else 0.0,
            reliable=bool(expr is not None and disp is not None
                          and disp <= budget),
            rhp=bool(r.real > 0))))
    return out


def _root_formulas(poly: sp.Poly, kind: str, subs: dict, budget: float,
                   max_roots: int) -> list[RootFormula]:
    """The lowest `max_roots` roots of `poly` with their formulas, read
    factor by factor from the exact factors the polynomial carries."""
    factors = _exact_factors(poly)
    per = [(f, _numeric_roots(f, subs)) for f in factors]
    allr = np.concatenate([r for _, r in per]) if per else np.array([])
    allr = allr[np.argsort(np.abs(allr))]
    if not allr.size:
        return []
    cutoff = abs(allr[:max_roots][-1]) * (1 + 1e-9)
    found = []
    for f, r in per:
        found += _factor_formulas(f, r, kind, subs, budget, cutoff)
    found.sort(key=lambda x: x[0])
    out = [g for _, g in found]
    for i, g in enumerate(out):
        g.index = i
    for g in out:
        if g._partner_ref is not None:
            g.partner = g._partner_ref.index
    for g in out:
        g._partner_ref = None          # the index is what callers keep
    return out


_LONG_TEXT = 400         # characters: a line longer than this is not read
_TABLE_MAX_TERMS = 50000 # beyond this the rescue reads the TF it was given
_NODE_SENS_TOL = 0.15    # |S + 1| under which a pole belongs to one capacitor
_MATCH_RATIO = 1.5       # a simplified root and its full-model counterpart
_ZETA_SLACK = 3.0        # a removal may move zeta this many budgets
_FACTOR_MAX_INT = 10 ** 40
_FREE_SENS = 0.02        # a reactance this insensitive goes without a reading
_NODE_MAX_CAPS = 8       # capacitors tried both open and short at a node pole
_COMPACT_MAX_TERMS = 40  # a form with more terms is pruned by size alone
_COMPACT_TRIES = 3       # terms compared by print length per step, per list


class _TermTable:
    """The additive terms of a polynomial in s as numpy arrays: s power,
    exponent per symbol, log magnitude and sign at the OP. Masking terms
    is removing a reactance (C -> 0), so trying a removal or taking every
    root's sensitivities costs no symbolic work. Logs keep the exact giant
    coefficients of a hybrid solve inside float64."""

    def __init__(self, poly: sp.Poly, subs: dict):
        self.s = poly.gen
        self.names = sorted({x for c in poly.all_coeffs()
                             for x in c.free_symbols}, key=str)
        vals = [sp.N(x.xreplace(subs), 30) for x in self.names]
        coeffs, rows, ks = [], [], []
        for k, c in enumerate(reversed(poly.all_coeffs())):
            if c == 0:
                continue
            for monom, co in sp.Poly(c, *self.names).terms():
                coeffs.append(co)
                rows.append(monom)
                ks.append(k)
        E = np.array(rows, dtype=int).reshape(len(rows), len(self.names))
        zero = np.array([v == 0 for v in vals], dtype=bool)
        keep = ~(E[:, zero] > 0).any(axis=1)   # a symbol at value zero
        lnx = np.array([math.log(abs(float(v))) if v != 0 else 0.0
                        for v in vals], dtype=float)
        neg = np.array([bool(v < 0) for v in vals], dtype=bool)
        lnc = np.array([_ln_abs(c) for c in coeffs], dtype=float)
        odd = np.array([bool(c < 0) for c in coeffs], dtype=bool) ^ (
            E[:, neg].sum(axis=1) % 2 == 1)
        self.coeffs = [c for c, m in zip(coeffs, keep) if m]
        self.k = np.array(ks, dtype=int)[keep]
        self.E = E[keep]
        self.lnv = (lnc + E @ lnx)[keep]
        self.sign = np.where(odd, -1.0, 1.0)[keep]

    def all(self) -> np.ndarray:
        return np.ones(len(self.coeffs), dtype=bool)

    def has(self, name: sp.Symbol) -> np.ndarray:
        return self.E[:, self.names.index(name)] > 0

    def at(self, r: complex) -> np.ndarray:
        """Each term's value at s = r, on one scale."""
        ln = self.lnv + self.k * math.log(abs(r))
        return (self.sign * np.exp(ln - ln.max())
                * np.exp(1j * self.k * np.angle(r)))

    def roots(self, mask: np.ndarray, lo: int | None = None,
              hi: int | None = None) -> np.ndarray:
        """Nonzero roots over the coefficients of s^lo .. s^hi (default:
        all of the masked polynomial), ascending by |root|."""
        k = self.k[mask]
        if not k.size:
            return np.array([], dtype=complex)
        lo = int(k.min()) if lo is None else lo
        hi = int(k.max()) if hi is None else hi
        pick = (k >= lo) & (k <= hi)
        k, lnv, sign = k[pick], self.lnv[mask][pick], self.sign[mask][pick]
        if hi == lo or not k.size or k.min() != lo or k.max() != hi:
            return np.array([], dtype=complex)
        # s = w0*x balances the ends of the ladder, as in scaled_roots
        lnw0 = (lnv[k == lo].max() - lnv[k == hi].max()) / (hi - lo)
        ln = lnv + (k - lo) * lnw0
        c = np.bincount(k - lo, weights=sign * np.exp(ln - ln.max()),
                        minlength=hi - lo + 1)
        r = np.roots(c[::-1]) * math.exp(lnw0)
        r = r[np.abs(r) > 0]
        return r[np.argsort(np.abs(r))]

    def sensitivities(self, r: complex) -> dict[sp.Symbol, complex]:
        """d ln(r) / d ln(x) per symbol, from D'(r) and x dD/dx at r."""
        v = self.at(r)
        with np.errstate(all="ignore"):  # a repeated root: D'(r) is 0
            return dict(zip(self.names, -(self.E.T @ v) / np.dot(self.k, v)))

    def sum_terms(self, mask: np.ndarray) -> sp.Expr:
        return sp.Add(*[self.coeffs[i] * sp.Mul(*[
            x ** int(e) for x, e in zip(self.names, self.E[i]) if e])
            for i in np.flatnonzero(mask)])

    def poly(self, mask: np.ndarray) -> sp.Poly:
        top = int(self.k[mask].max())
        return sp.Poly([self.sum_terms(mask & (self.k == i))
                        for i in range(top, -1, -1)], self.s)


def _ln_abs(c) -> float:
    """ln|c| of a sympy number, including rationals float64 cannot hold."""
    c = abs(c)
    if c == 0:
        return -math.inf
    if isinstance(c, sp.Rational):
        return math.log(int(c.p)) - math.log(int(c.q))
    return math.log(float(c))


def _reactances(table: _TermTable) -> list[sp.Symbol]:
    """The capacitors and inductors among the symbols. A TF carries no
    element kinds, so this goes by name and confirms by structure: a
    reactance comes with s, so it is multilinear and no term of s^k holds
    it below the k-th power above the polynomial's lowest."""
    if not table.coeffs:
        return []
    rank = table.k - table.k.min()
    return [x for j, x in enumerate(table.names)
            if str(x)[0] in "cClL"
            and (table.E[:, j] <= 1).all() and (table.E[:, j] <= rank).all()]


def _full_roots(poly: sp.Poly, subs: dict) -> list[tuple[complex, ...]]:
    """The polynomial's roots, a complex pair as one entry."""
    out, roots, used = [], _numeric_roots(poly, subs), set()
    for i, r in enumerate(roots):
        if i in used:
            continue
        used.add(i)
        if abs(r.imag) <= 1e-9 * abs(r):
            out.append((complex(r.real),))
            continue
        j = next((j for j in range(len(roots)) if j not in used
                  and abs(np.conj(roots[j]) - r) < 1e-6 * abs(r)), None)
        if j is not None:
            used.add(j)
        out.append((complex(r),) if j is None
                   else (complex(r), complex(roots[j])))
    return out


def _match_lines(lines: list[RootFormula], full: list[tuple[complex, ...]]
                 ) -> dict[int, tuple[complex, ...]]:
    """Each line's root in the full model: nearest of the same class, then
    for what is left of either class, since pruning can turn a real pair
    complex. A simplified TF's roots move within its error, so a line with
    nothing near it (a root the full model does not have) stays unmatched."""
    left = list(full)
    out = {}
    for same_class in (True, False):
        for g in lines:
            w = g.f_hz * 2 * np.pi
            near = [x for x in left if not same_class
                    or (len(x) == 2) == (g.damping is not None)]
            if id(g) in out or not near or w <= 0:
                continue
            best = min(near, key=lambda x: abs(math.log(abs(x[0]) / w)))
            if abs(math.log(abs(best[0]) / w)) <= math.log(_MATCH_RATIO):
                out[id(g)] = best
                left.remove(best)
    return out


def _value(expr, subs) -> complex:
    return complex(sp.N(expr.xreplace(subs), 30))


def _node_form(table: _TermTable, reactive, r: complex, subs, budget
               ) -> RootFormula | None:
    """A real pole set by one capacitor: 1/(R_node*C). With D = D0 + s C D1,
    R_node(s) = D1/D0 is what C sees. With every other capacitor open or
    short it is a resistive ratio of the terms that stay, so the assignment
    is the one whose pole lands nearest, the fewest terms among near ties
    (Rosenstark's time constants, the states chosen by the exact root)."""
    from ..units import eng_expr_str

    sens = table.sensitivities(r)
    cands = [x for x in reactive if abs(sens[x] + 1) <= _NODE_SENS_TOL]
    if not cands:
        return None
    cap = min(cands, key=lambda x: abs(sens[x] + 1))
    v = table.at(r)
    has = table.has(cap)

    def whole(mask) -> bool:             # both D0 and D1 still present
        return bool((mask & has).any() and (mask & ~has).any())

    # every assignment of the largest capacitors, keeping those that leave
    # both parts; the smaller ones then open. Ranked by the pole each gives.
    others = sorted((x for x in reactive if x != cap),
                    key=lambda x: -abs(_value(x, subs)))
    sels = [table.all()]
    for x in others[:_NODE_MAX_CAPS]:
        hx = table.has(x)
        sels = [m for b in sels for m in (b & hx, b & ~hx) if whole(m)]
    for x in others[_NODE_MAX_CAPS:]:
        sels = [b & ~table.has(x) for b in sels if whole(b & ~table.has(x))]
    scored = []
    for m in sels:
        k0, k1 = table.k[m & ~has], table.k[m & has]
        # one power of s each, one above the other, or the ratio has an s
        if k0.min() == k0.max() and k1.min() == k1.max() == k0.min() + 1:
            est = -v[m & ~has].sum() / v[m & has].sum() * r
            scored.append((abs(est / r - 1), int(m.sum()), m))
    if not scored:
        return None
    best = min(x[0] for x in scored)
    sel = min((x for x in scored if x[0] <= max(2 * best, budget / 5)),
              key=lambda x: x[1])[2]
    num, den = sp.fraction(sp.cancel(table.sum_terms(sel & ~has)
                                     / table.sum_terms(sel & has)))
    expr, disp = _single(num, den, complex(r), subs, budget)
    if expr is None:
        return None
    rnode = sp.cancel(1 / (cap * expr))
    display, legend = None, ""
    if rnode.has(sp.Add):
        display = f"1/({cap}*R_node)"
        legend = f", R_node = {eng_expr_str(rnode)}"
    legend += f", node pole of {cap} (sensitivity {sens[cap].real:.3g})"
    return RootFormula(
        kind="pole", index=0, f_hz=float(abs(r) / (2 * np.pi)), damping=None,
        expr=expr, displacement=float(disp), reliable=bool(disp <= budget),
        rhp=bool(r.real > 0), display=display, legend=legend, full_model=True)


def _readable(expr):
    """The factored form, when that is affordable and says something:
    factoring walks the integer content of a hybrid solve's coefficients,
    seconds per call once they have hundreds of digits."""
    if (expr.has(sp.Float) or _n_terms([expr]) > _FACTOR_MAX_TERMS
            or any(abs(x.p) > _FACTOR_MAX_INT or x.q > _FACTOR_MAX_INT
                   for x in expr.atoms(sp.Rational))):
        return expr
    try:
        return sp.factor(expr)
    except Exception:
        return expr


def _prune_within(expr, target: float, budget: float, subs):
    """expr with additive terms of its numerator and denominator gone while
    it stays within `budget` of `target`, choosing at each step the term
    whose loss prints shortest: a small term that blocks a common factor
    goes before one that merely is smallest. The budget opens in stages so
    that it is not spent on large terms while small ones still stand. As it
    was when already off."""
    if _n_terms(sp.fraction(sp.together(expr))) <= _FACTOR_MAX_TERMS:
        expr = sp.cancel(expr)
    num, den = sp.fraction(sp.together(expr))
    nt, dt = _terms(num), _terms(den)
    value = _fit_values([num, den], subs)
    vn, vd = {t: value(t) for t in nt}, {t: value(t) for t in dt}

    def ratio(n, d) -> complex:
        dv = sum(vd[t] for t in d)
        return sum(vn[t] for t in n) / dv if dv else complex(math.inf)

    def off(n, d) -> float:
        return abs(ratio(n, d) / target - 1)

    if off(nt, dt) > budget:
        return expr
    # too many terms to compare by print length: first what is negligible
    # against the form's own value, in the list that holds too many (the
    # other may be factored, and a lost term would break that)
    start = ratio(nt, dt)
    for stage in (budget / 8, budget / 4, budget / 2):
        if len(nt) + len(dt) <= _COMPACT_MAX_TERMS:
            break
        for terms, mags in ((nt, vn), (dt, vd)):
            if len(terms) <= _COMPACT_MAX_TERMS // 2:
                continue
            for t in sorted(terms, key=lambda x: abs(mags[x])):
                kept = [x for x in terms if x != t]
                n, d = (kept, dt) if terms is nt else (nt, kept)
                if abs(ratio(n, d) / start - 1) <= stage:
                    terms.remove(t)
    if len(nt) + len(dt) > _COMPACT_MAX_TERMS:
        return _readable(sp.cancel(sp.Add(*nt) / sp.Add(*dt)))

    seen: dict = {}

    def text(n, d):
        key = (frozenset(n), frozenset(d))
        if key not in seen:
            e = _readable(sp.cancel(sp.Add(*n) / sp.Add(*d)))
            seen[key] = len(sp.sstr(e)), e
        return seen[key]

    size, best = text(nt, dt)
    for stage in (budget / 8, budget / 4, budget / 2, budget):
        while True:
            trials = []
            for terms, pair, mags in ((nt, lambda x: (x, dt), vn),
                                      (dt, lambda x: (nt, x), vd)):
                if len(terms) < 2:
                    continue
                tried = 0
                for t in sorted(terms, key=lambda x: abs(mags[x])):
                    n, d = pair([x for x in terms if x != t])
                    if off(n, d) <= stage and tried < _COMPACT_TRIES:
                        trials.append((*text(n, d), n, d))
                        tried += 1
            trials = [x for x in trials if x[0] < size]
            if not trials:
                break
            size, best, nt, dt = min(trials, key=lambda x: x[0])
    return best


def _restate(g: RootFormula, roots, subs, budget: float) -> bool:
    """Point the formula g, read on another model, at the full-model roots:
    its numbers become the full model's and its errors are against them.
    Whether it holds within `budget`."""
    r = roots[0]
    w0 = abs(_value(g.expr, subs))
    g.f_hz = float(abs(r) / (2 * np.pi))
    g.rhp = bool(r.real > 0)
    g.full_model = True
    g.partner, g._partner_ref = None, None
    g.displacement = float(abs(w0 / abs(r) - 1))
    if len(roots) == 2:
        zeta = -r.real / abs(r)
        g.damping = float(zeta)
        b = _value(g.zeta_w0, subs).real if g.zeta_w0 is not None else 0.0
        g.zeta_displacement = float(abs(b / (2 * w0) / zeta - 1)) if zeta else 0.0
    g.reliable = bool(g.displacement <= budget)
    return g.reliable


def _compact(g: RootFormula, roots, subs, budget: float,
             raw: tuple | None = None) -> RootFormula:
    """Prune a form within the error it may state: a pair's w0^2 and
    2*zeta*w0 each on its own, as they are printed, since these budgets are
    on the printed quantity and not on the root's position. `raw` are the
    pair's quadratic coefficients before any pruning, whose factors pruning
    for position may have broken."""
    from ..units import eng_expr_str

    r = roots[0]
    if len(roots) == 2:
        c0, c1, c2 = raw if raw is not None else (g.expr ** 2, g.zeta_w0,
                                                  sp.Integer(1))
        w0sq = _prune_within(c0 / c2, abs(r) ** 2, 2 * budget, subs)
        g.expr = sp.sqrt(w0sq)
        g.display = f"sqrt({eng_expr_str(w0sq)})"
        if c1 is not None and c1 != 0:
            # the damping term starts from what the model it was read on
            # gives, which may already be off the full model
            start = c1 / c2
            g.zeta_w0 = _prune_within(start, _value(start, subs).real,
                                      _ZETA_SLACK * budget, subs)
    elif g.display is None:
        g.expr = _prune_within(g.expr, abs(r), budget, subs)
    _restate(g, roots, subs, budget)
    return g


def _too_long(g: RootFormula) -> bool:
    if g.expr is None:
        return False
    # a pair's line carries two formulas
    return len(g.describe()) > _LONG_TEXT * (1 if g.zeta_w0 is None else 2)


def _reduced_form(table, reactive, kind, roots, full, subs, budget):
    """Read the root on a model without the reactances that do not set it.
    Those owning another pole go first, since it is their root that crowds
    this one, then the least sensitive; a removal that moves the root out
    of the budget is refused. Stops at the first model whose reading holds
    against the full model, not reading before the reactances the root does
    not notice are gone. Returns that form (or None), and the mask and names
    of the furthest model reached, for the fallback."""
    r = roots[0]
    here = table.sensitivities(r)
    owners = set()
    for other in full:
        if other is not roots and len(other) == 1:
            so = table.sensitivities(other[0])
            owners |= {x for x in reactive
                       if abs(so[x] + 1) <= _NODE_SENS_TOL}
    w0 = abs(r)
    zeta = -r.real / w0
    pair = len(roots) == 2

    def holds(q) -> bool:
        cx = abs(q.imag) > 1e-9 * abs(q)
        if cx != pair or abs(abs(q) / w0 - 1) > budget:
            return False
        return not pair or (abs(-q.real / abs(q) / zeta - 1)
                            <= _ZETA_SLACK * budget)

    mask, removed = table.all(), []
    for x in sorted(reactive, key=lambda x: (abs(here[x]) > _FREE_SENS,
                                             x not in owners, abs(here[x]))):
        trial = mask & ~table.has(x)
        rs = table.roots(trial)
        if not rs.size or not any(holds(q) for q in rs):
            continue
        mask = trial
        removed.append(x)
        if int(mask.sum()) > _FACTOR_MAX_TERMS or any(
                abs(here[y]) <= _FREE_SENS and y not in removed
                for y in reactive):
            continue
        poly = table.poly(mask)
        a = list(reversed(poly.all_coeffs()))
        k0 = next(k for k, c in enumerate(a) if c != 0)
        exact = _numeric_roots(poly, subs)
        if not exact.size:
            continue
        cx = np.abs(exact.imag) > 1e-9 * np.abs(exact)
        rank = int(np.argmin(np.abs(np.log(np.abs(exact) / w0))
                             + 10 * (cx != pair)))
        if pair and rank > 0 and abs(exact[rank - 1] - np.conj(exact[rank])
                                     ) < 1e-9 * abs(exact[rank]):
            rank -= 1                    # the pair reads from its first root
        read = _factor_formulas(sp.Poly(a[k0:][::-1], poly.gen), exact, kind,
                                subs, budget, math.inf, only=rank)
        same = [g for _, g in read if g.expr is not None
                and (g.damping is not None) == pair]
        if not same:
            continue
        g = min(same, key=lambda g: abs(math.log(g.f_hz * 2 * np.pi / w0)))
        found = replace(g, _partner_ref=None,
                        without=tuple(sorted(map(str, removed))))
        if not _restate(found, roots, subs, budget):
            continue
        raw = tuple(a[k0 + rank:k0 + rank + 3]) if pair else None
        _compact(found, roots, subs, budget, raw)
        if not _too_long(found):
            return found, mask, removed
    return None, mask, removed


def _cluster_form(table, mask, removed, g, roots, subs, budget):
    """The polynomial of the coupled roots around `roots`, in s: the
    smallest slice of coefficients that holds them and whatever crowds
    them, its terms pruned while every root in it stays within `budget`.
    None when the slice carries no symbol to show."""
    from ..units import eng_expr_str

    rs = table.roots(mask)
    if not rs.size:
        return None
    ranks = [int(np.argmin(np.abs(np.log(np.abs(rs) / abs(x))))) for x in roots]
    lo, hi = min(ranks), max(ranks)
    k0 = int(table.k[mask].min())

    def miss(keep, lo, hi) -> float:
        q = table.roots(keep, k0 + lo, k0 + hi + 1)
        if q.size != hi - lo + 1:
            return math.inf
        # a pair's two members sort arbitrarily: |Re| and |Im| do not care
        d = rs[lo:hi + 1]
        return float(np.max(np.maximum(abs(abs(q.real) - abs(d.real)),
                                       abs(abs(q.imag) - abs(d.imag)))
                            / np.abs(d)))

    # at least two roots: alone, a root's slice was the ratio already tried
    while miss(mask, lo, hi) > budget or lo == hi:
        below = abs(rs[lo] / rs[lo - 1]) if lo > 0 else math.inf
        above = abs(rs[hi + 1] / rs[hi]) if hi + 1 < rs.size else math.inf
        if math.isinf(min(below, above)):
            break
        if below < above:
            lo -= 1
        else:
            hi += 1
    if miss(mask, lo, hi) > budget:
        return None
    pick = mask.copy()
    inside = (table.k >= k0 + lo) & (table.k <= k0 + hi + 1)
    for t in sorted(np.flatnonzero(mask & inside), key=lambda t: table.lnv[t]):
        pick[t] = False
        if miss(pick, lo, hi) > budget:
            pick[t] = True
    pick &= inside
    low = table.E[pick].min(axis=0)      # says nothing about the roots
    common = sp.Mul(*[x ** int(e) for x, e in zip(table.names, low) if e])
    if not (table.E[pick] > low).any():
        return None
    # the common scale of a hybrid solve's coefficients says nothing either
    scale = max(abs(table.coeffs[t]) for t in np.flatnonzero(pick))
    if not sp.Rational(1, 1000) <= scale <= 1000:
        common *= scale
    parts = []
    for k in range(k0 + hi + 1, k0 + lo - 1, -1):
        c = table.sum_terms(pick & (table.k == k)) / common
        if c != 0:
            p = k - k0 - lo
            power = "" if p == 0 else "*s" if p == 1 else f"*s^{p}"
            parts.append(f"({eng_expr_str(_readable(sp.expand(c)))}){power}")
    text = " + ".join(parts)
    q = table.roots(pick | (mask & ~inside), k0 + lo, k0 + hi + 1)
    mine = q[np.argmin(np.abs(np.log(np.abs(q) / abs(roots[0]))))]
    return replace(
        g, cluster=(text if len(text) <= 2 * _LONG_TEXT else
                    f"a polynomial of {int(pick.sum())} terms, too long to "
                    f"print"),
        without=tuple(sorted(map(str, removed))), full_model=True,
        f_hz=float(abs(roots[0]) / (2 * np.pi)), expr=None, reliable=False,
        damping=(float(-roots[0].real / abs(roots[0])) if len(roots) == 2
                 else None),
        rhp=bool(roots[0].real > 0),
        displacement=float(abs(abs(mine) / abs(roots[0]) - 1)),
        _partner_ref=None)


def _rescue(tf: TransferFunction, lines: list[RootFormula], kind: str,
            budget: float) -> list[RootFormula]:
    """Forms for the lines the coefficient reading left empty or too long,
    each stated against the full model: the node form, the reduced model,
    the long form pruned, the cluster's polynomial. Every other line is
    returned untouched."""
    needy = {id(g) for g in lines if g.expr is None or _too_long(g)}
    if not needy:
        return lines
    ref = getattr(tf, "original", None) or tf
    which = 1 if kind == "pole" else 0
    subs = ref._subs_map()
    poly = ref.num_den[which]
    source = poly                        # what removals act on
    if _n_terms(poly.all_coeffs()) > _TABLE_MAX_TERMS:
        source = tf.num_den[which]
        if _n_terms(source.all_coeffs()) > _TABLE_MAX_TERMS:
            return lines
    if not any(c.free_symbols for c in source.all_coeffs()):
        return lines                     # all numeric: nothing to show
    try:
        table = _TermTable(source, subs)
    except (TypeError, ValueError, sp.PolynomialError):
        return lines                     # not a polynomial in plain symbols
    full = _full_roots(poly, subs)
    matched = _match_lines(lines, full)
    reactive = _reactances(table)
    out, seen = [], {}
    for g in lines:
        roots = matched.get(id(g))
        if id(g) not in needy or roots is None:
            out.append(g)
            continue
        found, mask, removed = None, table.all(), []
        if kind == "pole" and len(roots) == 1:
            found = _node_form(table, reactive, roots[0], subs, budget)
        if found is None:
            found, mask, removed = _reduced_form(
                table, reactive, kind, roots, full, subs, budget)
        if found is None and g.expr is not None:
            found = _compact(replace(g, _partner_ref=None), roots, subs, budget)
            if _too_long(found):
                found.long_terms = _n_terms(sp.fraction(sp.together(found.expr)))
        if found is None:
            found = _cluster_form(table, mask, removed, g, roots, subs, budget)
            if found is not None and found.cluster in seen:
                found.cluster_of = seen[found.cluster]
            elif found is not None:
                seen[found.cluster] = g.index
        if found is not None:
            found.kind, found.index = kind, g.index
        out.append(found if found is not None else g)
    return out


def template_form(tf: TransferFunction, *, budget: float = 0.05,
                  max_roots: int = 6) -> TemplateForm:
    """Render `tf` in the standard multistage form: DC gain, GBW, and a
    short symbolic formula per pole/zero, each checked against the exact
    numeric root and reported with its displacement. A root the
    coefficients cannot reach is read against the full model instead (see
    the module docstring), and says so.

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

    poles = _root_formulas(den, "pole", subs, budget, max_roots)
    zeros = _root_formulas(num, "zero", subs, budget, max_roots)

    gbw = None
    if poles and np.isfinite(dc_val) and dc_val != 0:
        gbw = float(abs(dc_val) * poles[0].f_hz)
    poles = _rescue(tf, poles, "pole", budget)
    zeros = _rescue(tf, zeros, "zero", budget)
    return TemplateForm(dc_gain=dc, dc_gain_value=dc_val, poles=poles,
                        zeros=zeros, gbw_hz=gbw, budget=budget)
