"""Explain the numerals: which parameters a collapsed coefficient is made of.

A hybrid solve substitutes every unkept parameter before solving, so the
numbers in a simplified expression -- the 4.832e-5 multiplying gds in a
lowest-order denominator -- are sums of products of parameters whose
names are gone by the time anyone can ask. This module ranks the
contributors without re-solving anything symbolically.

The mathematics that makes it cheap and mostly exact: det A is
MULTILINEAR in the stamp values (every stamp is rank-one), so for a
parameter that stamps linearly the quantity

    share(p) = p * (d c_k / d p) / c_k

is not a heuristic: it is exactly the fraction of coefficient c_k
carried by the Leibniz terms that contain p. Shares of different
parameters overlap -- a term g1*g2*CL counts fully toward g1, g2 and CL
-- so they do not sum to one; each answers "how much of this number
flows through that parameter". Resistors stamp as 1/r and matched
groups stamp several times; their shares are first-order log-
sensitivities instead, which still rank correctly.

All derivatives for ALL parameters come from one inverse per s-sample:

    d det/d p (s_j) = det(s_j) * tr(A(s_j)^-1 dA/dp)

with dA/dp a sparse constant pattern (times s for reactances), and the
per-coefficient derivatives recovered through the same inverse
Vandermonde the all-numeric solver uses. That reconstruction is
hopeless in float64 -- at 18 integer points its condition number eats
all 16 digits -- so the whole pass runs in mpmath at `dps` digits, the
same reason the engine itself works in exact rationals.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import mpmath as mp
import sympy as sp

from ..engine.interp import _s_degree_bound, _vinv_rows
from ..engine.mna import S, MnaError, MnaSystem

__all__ = ["CoefficientStory", "NumeralStory", "explain_coefficients",
           "explain_per_numeral", "mono_key",
           "ratio_contributors", "ratio_lines"]


def mono_key(powers: dict) -> str:
    """Canonical name of a kept-monomial: 'CL·gm_I0_NM3', '1' for the
    constant term. Keys are symbol names (or Symbols), values integer
    exponents. The GUI derives the same key from the displayed
    expression's terms, so a numeral on screen and its NumeralStory
    meet on this string."""
    items = sorted((str(n), int(e)) for n, e in powers.items() if e)
    return "·".join(f"{n}^{e}" if e > 1 else n for n, e in items) or "1"


@dataclass
class NumeralStory:
    """One collapsed numeral of the hybrid expression: the coefficient
    of s^k · monomial(kept letters) in N or D, and who builds it."""
    part: str                      # "num" | "den"
    k: int                         # power of s
    mono: str                      # kept-monomial key (mono_key), '1' = none
    value: float                   # the numeral, on a shared N/D scale
    contributors: list = field(default_factory=list)   # (name, share) desc.

    def describe(self, top: int = 6) -> str:
        who = ", ".join(f"{n} {s:+.0%}"
                        for n, s in self.contributors[:top])
        m = "" if self.mono == "1" else f" · {self.mono}"
        return f"{self.part} s^{self.k}{m}: {who or '(no single contributor)'}"


@dataclass
class CoefficientStory:
    """One power of s in one polynomial: its value and who builds it."""
    part: str                      # "num" | "den"
    k: int                         # power of s
    value: float                   # coefficient, on a shared N/D scale
    contributors: list = field(default_factory=list)   # (name, share) desc.

    def describe(self, top: int = 3) -> str:
        """Grouped by parameter family (gm, gds, gmbs, caps ...): the gm
        chain is the shared scaffolding of every coefficient, while the
        families that DIFFER between coefficients -- the gds pair behind
        a DC term, the cap behind an s term -- are what the numeral is
        about. A share near 100% means the parameter multiplies
        essentially every term; above 100% means cancellation or a
        repeated appearance (matched groups stamp their symbol twice)."""
        fams: dict[str, list] = {}
        for n, s in self.contributors:
            fams.setdefault(n.split("_", 1)[0], []).append((n, s))
        bits = []
        for items in sorted(fams.values(), key=lambda it: -abs(it[0][1])):
            head = ", ".join(f"{n} {s:+.0%}" for n, s in items[:top])
            if len(items) > top:
                head += f" (+{len(items) - top} more)"
            bits.append(head)
        if not bits:
            bits = ["no single parameter reaches the share floor"]
        return f"{self.part} s^{self.k}: " + "; ".join(bits)


def _to_mpf(x) -> mp.mpf:
    r = sp.Rational(x)
    return mp.mpf(int(r.p)) / mp.mpf(int(r.q))


def _mp_matrix(M: sp.Matrix) -> mp.matrix:
    out = mp.matrix(M.rows, M.cols)
    for i in range(M.rows):
        for j in range(M.cols):
            e = M[i, j]
            if e != 0:
                out[i, j] = _to_mpf(e)
    return out


def ratio_contributors(stories, a, b, *, top: int = 8,
                       min_share: float = 0.02):
    """Attribution of a DISPLAYED numeral, which is never one coefficient
    but a ratio of two: log-sensitivities SUBTRACT in a ratio, so a
    parameter with the same share in both coefficients -- the gm chain
    that scales every term -- cancels and contributes nothing. What
    remains is what actually shapes the numeral. `a` and `b` are
    (part, k) pairs; returns [(name, diff_share)] ranked by magnitude,
    or None when either coefficient has no story."""
    def find(pk):
        for st in stories:
            if (st.part, st.k) == pk:
                return dict(st.contributors)
        return None

    A, B = find(a), find(b)
    if A is None or B is None:
        return None
    out = [(n, A.get(n, 0.0) - B.get(n, 0.0)) for n in set(A) | set(B)]
    out = [(n, d) for n, d in out if abs(d) >= min_share]
    out.sort(key=lambda t: -abs(t[1]))
    return out[:top]


def ratio_lines(stories, top: int = 6, shown=None) -> list[str]:
    """The displayed numerals, attributed: A0 (lowest num over lowest
    den coefficient) and each adjacent same-part pair -- the pole/zero
    scale ratios the factored form and the root formulas are built
    from (exact for well-separated real roots).

    Each line carries its NUMBERS: the operating-point value of the
    ratio (A0 as a gain, pole/zero scales as frequencies) and, via
    `shown` ((part, k) -> string), the collapsed numeral the expression
    displays for that coefficient -- the bridge from the number on
    screen to its contributors."""
    import math

    from ..units import eng

    shown = shown or {}
    vals = {(st.part, st.k): st.value for st in stories}
    ks = {p: sorted(st.k for st in stories if st.part == p)
          for p in ("num", "den")}

    def value(a, b, kind):
        va, vb = vals.get(a), vals.get(b)
        if not va or not vb:
            return ""
        r = va / vb
        if kind == "hz":
            return f" = {eng(abs(r) / (2 * math.pi), 'Hz')}"
        return f" = {eng(r, sig=4)}"

    def seen(a):
        return f" — the {shown[a]} numeral" if a in shown else ""

    entries = []
    if ks["num"] and ks["den"]:
        a, b = ("num", ks["num"][0]), ("den", ks["den"][0])
        entries.append((f"A0{value(a, b, 'gain')}  "
                        f"(num s^{a[1]} / den s^{b[1]}{seen(a)})", a, b))
    for part, label in (("den", "pole scale"), ("num", "zero scale")):
        kk = ks[part]
        for i in range(len(kk) - 1):
            a, b = (part, kk[i]), (part, kk[i + 1])
            entries.append(
                (f"{label} {i}{value(a, b, 'hz')}  "
                 f"({part} s^{kk[i]} / s^{kk[i + 1]}{seen(a)})", a, b))
    lines = []
    for label, a, b in entries:
        rc = ratio_contributors(stories, a, b, top=top)
        if rc:
            lines.append(label + ": " + ", ".join(
                f"{n} {d:+.0%}" for n, d in rc))
    return lines


def explain_coefficients(system: MnaSystem, out: str, *, exclude=(),
                         top: int = 24, min_share: float = 0.02,
                         dps: int = 60,
                         progress=None) -> list[CoefficientStory]:
    """Rank, for every coefficient of N(s) and D(s) at the operating
    point, the parameters that carry its value.

    `exclude` names the KEPT symbols -- they are already letters in the
    expression, so the question is only about what collapsed. Shares
    below `min_share` are dropped as noise."""
    if out not in system.node_index:
        raise MnaError(f"output node {out!r} not found (or it is ground)")
    col = system.node_index[out]
    subs = {system.symbols[n]: sp.Rational(repr(v))
            for n, v in system.values.items() if n in system.symbols}
    A_num = system.A.xreplace(subs)
    free = set(A_num.free_symbols) - {S}
    if free:
        raise MnaError("explain_coefficients needs numeric values; "
                       f"missing {sorted(map(str, free))}")

    # per-symbol stamp patterns from a single pass over the nonzero
    # ENTRIES (the attribution.py lesson: per-symbol matrix diffs on a
    # 30x30 sympy matrix cost 20 s; per-entry scalar diffs are micro-ops)
    by_name = {s_: n_ for n_, s_ in system.symbols.items()}
    excl = set(exclude)
    patterns: dict[str, list] = {}
    with mp.workdps(dps):
        for i in range(system.A.rows):
            for j in range(system.A.cols):
                e = system.A[i, j]
                if e == 0 or getattr(e, "is_Number", False):
                    continue
                for sym in e.free_symbols - {S}:
                    name = by_name.get(sym)
                    if (name is None or name in excl
                            or name not in system.values):
                        continue
                    d = sp.diff(e, sym).xreplace(subs)
                    d1 = d.coeff(S)
                    d0 = d - S * d1
                    patterns.setdefault(name, []).append(
                        (i, j, _to_mpf(d0), _to_mpf(d1)))

        P0 = _mp_matrix(A_num.xreplace({S: sp.Integer(0)}))
        P1 = _mp_matrix(A_num.diff(S))
        Ak = A_num.copy()
        Ak[:, col] = system.z
        K0 = _mp_matrix(Ak.xreplace({S: sp.Integer(0)}))
        K1 = _mp_matrix(Ak.diff(S))

        L = _s_degree_bound(A_num) + 1
        for base in (0, L, 7 * L):          # shift past any singular sample
            pts = list(range(base + 1, base + L + 1))
            try:
                stories = _sweep(pts, P0, P1, K0, K1, col, patterns,
                                 system.values, top, min_share,
                                 progress=progress)
                break
            except ZeroDivisionError:
                continue
        else:
            raise MnaError("explain_coefficients: singular sample matrices")
    return stories


def explain_per_numeral(system: MnaSystem, out: str, keep, *,
                        top: int = 8, min_share: float = 0.02,
                        dps: int = 60, progress=None) -> list[NumeralStory]:
    """Attribution of EVERY collapsed numeral of the hybrid expression:
    the coefficient n_{k,m} of s^k · monomial_m(kept letters), with the
    parameters that carry it.

    The operating-point sweep cannot separate the monomials -- keeps at
    their values arrive pre-summed -- so this pass varies the KEPT
    symbols over the same small integer grid the hybrid solver uses
    (degree = stamp count per axis) and reconstructs both the numeral
    tensor and its derivative tensors per unkept parameter through
    inverse-Vandermonde contractions along every axis. One matrix
    inverse per (grid point, s-sample) serves all parameters via
    tr(A^-1 dA/dp). Shares are exact for linear stamps, per numeral.
    Measured on the 413-primitive fc: 3.6 s at 2 keeps, 108 s for the
    5-keep story set -- on demand and cached."""
    import itertools

    import numpy as np

    from ..engine.mna import hybrid_split

    if out not in system.node_index:
        raise MnaError(f"output node {out!r} not found (or it is ground)")
    col = system.node_index[out]
    subs, kept = hybrid_split(system, list(keep))
    if not kept:
        raise MnaError("per-numeral attribution needs a symbolic keep set")
    rec = [n for n in kept if n in system.reciprocal]
    if rec:
        raise MnaError(f"kept resistances (1/r stamps) not supported "
                       f"yet: {rec}")
    ksyms = [system.symbols[n] for n in kept]
    A = system.A.xreplace(subs)
    free = set(A.free_symbols) - {S} - set(ksyms)
    if free:
        raise MnaError("explain_per_numeral needs numeric values; "
                       f"missing {sorted(map(str, free))}")
    Ak = A.copy()
    Ak[:, col] = system.z

    degs = [max(1, int(system.stamp_counts.get(n, 1))) for n in kept]
    L = _s_degree_bound(A) + 1

    # everything numeric runs on gmpy2 mpfr scalars in plain Python
    # lists: mpmath's dict-backed matrix paid ~2 s per grid point on a
    # 30x30 system (det and inverse each re-decomposing), which turned
    # the fc-scale pass into hours. C-speed scalars + one Gauss-Jordan
    # per matrix (det and inverse from the same elimination) bring the
    # same computation to tens of milliseconds per point.
    import gmpy2

    bits = int(dps * 3.4) + 34
    n = A.rows
    by_name = {s_: n_ for n_, s_ in system.symbols.items()}
    kept_set = set(kept)

    with gmpy2.context(precision=bits):
        g0 = gmpy2.mpfr(0)

        def desc(matrix):
            # per nonzero entry: (i, j, terms0, terms1) with each terms
            # list [(mpfr coeff, exps over kept)] -- EXACT conversion of
            # the substituted rationals (a lambdify float literal would
            # truncate them to 16 digits and poison the reconstruction)
            out = []
            for i in range(n):
                for j in range(n):
                    e = matrix[i, j]
                    if e == 0:
                        continue
                    e1 = e.coeff(S)
                    e0 = sp.expand(e - S * e1)
                    t0 = _g_terms(e0, ksyms)
                    t1 = _g_terms(e1, ksyms)
                    if t0 or t1:
                        out.append((i, j, t0, t1))
            return out

        adesc = desc(A)
        kdesc = desc(Ak)

        patterns: dict[str, list] = {}
        for i in range(system.A.rows):
            for j in range(system.A.cols):
                e = system.A[i, j]
                if e == 0 or getattr(e, "is_Number", False):
                    continue
                for sym in e.free_symbols - {S}:
                    name = by_name.get(sym)
                    if (name is None or name in kept_set
                            or name not in system.values):
                        continue
                    d = sp.diff(e, sym).xreplace(subs)
                    d1 = d.coeff(S)
                    d0 = sp.expand(d - S * d1)
                    patterns.setdefault(name, []).append(
                        (i, j, _g_terms(d0, ksyms), _g_terms(d1, ksyms)))

        axes_pts = [list(range(1, d + 2)) for d in degs]
        shape_k = tuple(d + 1 for d in degs)
        n_grid = 1
        for d in degs:
            n_grid *= d + 1

        for s_base in (0, L, 7 * L):        # shift past singular samples
            s_pts = list(range(s_base + 1, s_base + L + 1))
            try:
                T = {p: np.empty(shape_k + (L,), dtype=object)
                     for p in ("num", "den")}
                D = {part: {p: np.empty(shape_k + (L,), dtype=object)
                            for p in patterns} for part in ("num", "den")}
                done = 0
                for kidx in itertools.product(
                        *[range(d + 1) for d in degs]):
                    kv = [axes_pts[ax][kidx[ax]]
                          for ax in range(len(degs))]

                    def ev(terms):
                        acc = g0
                        for c, exps in terms:
                            m = c
                            for x, e in zip(kv, exps):
                                if e:
                                    m = m * x ** e     # int power, exact
                            acc = acc + m
                        return acc

                    def build(dsc):
                        M0 = [[g0] * n for _ in range(n)]
                        M1 = [[g0] * n for _ in range(n)]
                        for (i, j, t0, t1) in dsc:
                            if t0:
                                M0[i][j] = ev(t0)
                            if t1:
                                M1[i][j] = ev(t1)
                        return M0, M1

                    A0, A1 = build(adesc)
                    K0, K1 = build(kdesc)
                    pv = {p: [(i, j, ev(t0), ev(t1))
                              for (i, j, t0, t1) in entries]
                          for p, entries in patterns.items()}
                    for js, sv in enumerate(s_pts):
                        MA = [[A0[r][c2] + sv * A1[r][c2]
                               for c2 in range(n)] for r in range(n)]
                        MK = [[K0[r][c2] + sv * K1[r][c2]
                               for c2 in range(n)] for r in range(n)]
                        dA, Ainv = _gj_det_inv(MA, n, g0)
                        dK, Kinv = _gj_det_inv(MK, n, g0)
                        T["den"][kidx + (js,)] = dA
                        T["num"][kidx + (js,)] = dK
                        for p, entries in pv.items():
                            trA = g0
                            trK = g0
                            for (i, j, d0v, d1v) in entries:
                                dv = d0v + sv * d1v
                                trA = trA + dv * Ainv[j][i]
                                if j != col:
                                    trK = trK + dv * Kinv[j][i]
                            D["den"][p][kidx + (js,)] = dA * trA
                            D["num"][p][kidx + (js,)] = dK * trK
                    done += 1
                    if progress is not None:
                        progress(done, n_grid)
                break
            except ZeroDivisionError:
                continue
        else:
            raise MnaError("explain_per_numeral: singular sample matrices")

        rows_axes = [_rows_g(pts) for pts in axes_pts]
        rows_axes.append(_rows_g(s_pts))

        def recon(tensor):
            cur = tensor
            for ax, R in enumerate(rows_axes):
                cur = np.moveaxis(
                    np.tensordot(R, cur, axes=([1], [ax])), 0, ax)
            return cur

        C = {part: recon(T[part]) for part in ("num", "den")}
        DC = {part: {p: recon(D[part][p]) for p in patterns}
              for part in ("num", "den")}

        scale = max(abs(v) for part in ("num", "den")
                    for v in C[part].flat if v != 0)
        floor = scale * gmpy2.mpfr("1e-18")
        stories = []
        for part in ("num", "den"):
            for idx in np.ndindex(*C[part].shape):
                v = C[part][idx]
                if abs(v) <= floor:
                    continue
                mono = mono_key({kept[i]: idx[i]
                                 for i in range(len(kept))})
                contr = []
                for p in patterns:
                    share = float(gmpy2.mpfr(repr(system.values[p]))
                                  * DC[part][p][idx] / v)
                    if abs(share) >= min_share:
                        contr.append((p, share))
                contr.sort(key=lambda t: -abs(t[1]))
                stories.append(NumeralStory(
                    part=part, k=int(idx[-1]), mono=mono,
                    value=float(v / scale), contributors=contr[:top]))
    return stories


def _rows_g(pts):
    """Inverse-Vandermonde rows for integer points as mpfr (object
    ndarray) -- the exact QQ rows of the engine, converted once at the
    working precision for the tensor contractions."""
    import gmpy2
    import numpy as np

    rows = _vinv_rows([sp.Rational(v) for v in pts])
    m = len(rows)
    out = np.empty((m, m), dtype=object)
    for i in range(m):
        for j in range(m):
            q = rows[i][j]
            out[i, j] = gmpy2.mpfr(gmpy2.mpq(int(q.numerator),
                                             int(q.denominator)))
    return out


def _g_terms(expr, ksyms):
    """Polynomial expr in the kept symbols as [(mpfr coeff, exps)] --
    the exact-rational coefficients converted once at working
    precision."""
    import gmpy2

    if expr == 0:
        return []
    p = sp.Poly(expr, *ksyms, domain="QQ")
    out = []
    for exps, c in p.terms():
        out.append((gmpy2.mpfr(gmpy2.mpq(int(c.numerator),
                                         int(c.denominator))),
                    tuple(int(e) for e in exps)))
    return out


def _gj_det_inv(M, n, zero):
    """Gauss-Jordan with partial pivoting on [M | I]: determinant and
    inverse from ONE elimination. M is a list of row lists (consumed).
    Raises ZeroDivisionError on a singular pivot -- the caller shifts
    the sample points."""
    one = zero + 1
    for i in range(n):
        M[i] = M[i] + [one if j == i else zero for j in range(n)]
    det = one
    for cp in range(n):
        piv = max(range(cp, n), key=lambda r: abs(M[r][cp]))
        if M[piv][cp] == 0:
            raise ZeroDivisionError
        if piv != cp:
            M[cp], M[piv] = M[piv], M[cp]
            det = -det
        prow = M[cp]
        pval = prow[cp]
        det = det * pval
        inv_p = 1 / pval
        for j in range(cp, 2 * n):
            prow[j] = prow[j] * inv_p
        for r in range(n):
            if r == cp:
                continue
            f = M[r][cp]
            if f == 0:
                continue
            rr = M[r]
            for j in range(cp, 2 * n):
                rr[j] = rr[j] - f * prow[j]
    return det, [row[n:] for row in M]


def _sweep(pts, P0, P1, K0, K1, col, patterns, values, top, min_share,
           progress=None):
    n = P0.rows
    L = len(pts)
    dvals, nvals = [], []
    dder = {p: [] for p in patterns}
    nder = {p: [] for p in patterns}
    for js, s in enumerate(pts):
        if progress is not None:
            progress(js, L)
        Aj = P0 + s * P1
        Kj = K0 + s * K1
        dA, dK = mp.det(Aj), mp.det(Kj)
        if dA == 0 or dK == 0:
            raise ZeroDivisionError
        Ainv = Aj ** -1
        Kinv = Kj ** -1
        dvals.append(dA)
        nvals.append(dK)
        for p, entries in patterns.items():
            trA = mp.mpf(0)
            trK = mp.mpf(0)
            for (i, j, d0, d1) in entries:
                d = d0 + s * d1
                trA += d * Ainv[j, i]
                if j != col:                 # z replaced that column in Ak
                    trK += d * Kinv[j, i]
            dder[p].append(dA * trA)
            nder[p].append(dK * trK)

    rows = [[_to_mpf(sp.Rational(int(q.numerator), int(q.denominator)))
             for q in row]
            for row in _vinv_rows([sp.Rational(v) for v in pts])]

    def coeffs(vals):
        return [sum(rows[k][i] * vals[i] for i in range(L))
                for k in range(L)]

    d_c, n_c = coeffs(dvals), coeffs(nvals)
    cmax = max(abs(c) for c in d_c + n_c)
    if cmax == 0:
        raise MnaError("explain_coefficients: zero transfer function")

    # per-coefficient trust: the reconstruction sums terms of magnitude
    # |rows[k][i]*vals[i]|; a coefficient smaller than that magnitude
    # times the working roundoff is cancellation noise, not a value.
    # High-order coefficients of a deep polynomial (products of many
    # tiny caps) legitimately fall below it -- they are simply not
    # explainable at this dps, and are skipped rather than misreported.
    eps = mp.mpf(10) ** (12 - mp.mp.dps)     # ~12 digits lost in det/inv

    def trust(vals, k):
        return eps * sum(abs(rows[k][i] * vals[i]) for i in range(L))

    stories = []
    for part, base, vals_, ders in (("num", n_c, nvals, nder),
                                    ("den", d_c, dvals, dder)):
        for k in range(L):
            c = base[k]
            if abs(c) <= 10 ** 4 * trust(vals_, k):
                continue
            shares = []
            for p in patterns:
                dc_k = sum(rows[k][i] * ders[p][i] for i in range(L))
                share = float(_to_mpf(repr(values[p])) * dc_k / c)
                if abs(share) >= min_share:
                    shares.append((p, share))
            shares.sort(key=lambda t: -abs(t[1]))
            stories.append(CoefficientStory(
                part=part, k=k, value=float(c / cmax),
                contributors=shares[:top]))
    return stories
