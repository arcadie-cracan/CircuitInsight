"""Per-element bandwidth attribution via time and transfer constants.

Hajimiri's generalized time- and transfer-constant analysis (A. Hajimiri,
IEEE TCAS-I 57(6), 2010) writes H(s) = N(s)/D(s) with, for D(0) = 1,

    b1 = sum_i tau_i^0                    a1 = sum_i tau_i^0 H^i
    b2 = sum_i sum_{j>i} tau_i^0 tau_j^i

tau_i^0 is element i's zero-value time constant, tau_j^i the time constant of
j with i at its infinite value (capacitor shorted, inductor open), and H^i the
low-frequency gain with i at its infinite value. 1/b1 estimates a dominant
pole of a well-separated system (ZVTC); eqs. (41)-(42) correct it for zeros,
with tau-bar_i = tau_i^0 (1 - |H^i/H^0|), and Q = sqrt(b2)/b1, w_n = 1/sqrt(b2)
(eqs. (51), (53)) describe a dominant complex pair.

Because every reactive stamp is rank-1, the coefficients of D and N are
MULTILINEAR in the element values, so each of these is an exact rational. With
A(s) = G + s sum_i x_i C_i and M_i = G^-1 C_i, det(I + s sum_i x_i M_i) gives

    a1/a0 term of x_i        tr(M_i)
    a2/a0 term of x_i x_j    tr(M_i) tr(M_j) - tr(M_i M_j)
    N1/a0 term of x_i        tr(M_i) H^0 - e' M_i y0     (y0 = G^-1 z, H^0 = e' y0)

One exact inverse, no symbolic solve. A determinant per pair would give the
same numbers at far greater cost: 861 of them on the 42-element Miller OTA.

Exactness self-check: the per-element terms must reproduce a1/a0, a2/a0 and
N1/a0 of the true determinants to the last rational digit; a mismatch raises
instead of reporting wrong attributions.

This answers the design/teaching question "which capacitance limits the
bandwidth, at this operating point?" (education scenario 2 in the paper).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import sympy as sp
from sympy.polys.domains import QQ
from sympy.polys.matrices import DomainMatrix

from ..engine.mna import MnaError, S, _det, build_mna, symbol_name
from ..units import eng

# The dominant root counts as complex past this |Im p|/|p|. Coincident real
# poles (a matched pair's two halves) come back from the rooter split by about
# sqrt(eps), and a true pair this close to the axis acts like coincident poles.
_COMPLEX_TOL = 1e-4
_RHP_TOL = 1e-9
_REPORT_PAIRS = 6


@dataclass
class BandwidthRow:
    name: str          # symbol name (matched devices share one row)
    value: float       # element value (F or H)
    tau: float         # zero-value time constant contribution (s)
    share: float       # tau / tau_total
    h: float | None = None
    # H^i, the low-frequency gain with this element at its infinite value.
    # None without an input and output. +-inf if the gain grows without bound
    # (tau = 0 but the element still feeds the output, so tau*H^i is finite).
    tau_bar: float | None = None
    # tau (1 - |H^i/H^0|), eqs. (41)-(42); None if H^i is, or if H^0 = 0.


@dataclass
class BandwidthPair:
    """One term of b2. `term` = tau_i^0 tau_j^i = tau_j^0 tau_i^j is always
    finite. A factor whose zero-value constant is 0 is None: the paper's
    indeterminate 0 * inf product (Example 8, tau_C^0 tau_L^C). i == j is the
    term of a matched row with itself, which a single element never has."""
    i: str
    j: str
    term: float                # s^2
    tau_j_i: float | None      # tau_j^i, element j with i at infinity
    tau_i_j: float | None      # tau_i^j, element i with j at infinity


@dataclass
class BandwidthReport:
    rows: list[BandwidthRow]          # sorted by |tau|, descending
    tau_total: float                  # = a1/a0 exactly
    f_zvtc: float                     # 1 / (2 pi tau_total)
    f_dominant: float                 # |smallest denominator root| / 2 pi
    dominant_pole: complex = complex("nan")   # that root, rad/s
    b2: float = 0.0                   # = a2/a0 exactly
    pairs: list[BandwidthPair] = field(default_factory=list)  # by |term|, descending
    inp: str | None = None
    out: str | None = None
    h0: float | None = None           # H^0, None without an input and output
    tau_bar_total: float | None = None  # sum of tau_bar, None if H^0 = 0
    f_modified: float | None = None   # 1 / (2 pi tau_bar_total), None if <= 0

    @property
    def b1(self) -> float:
        return self.tau_total

    @property
    def q(self) -> float | None:
        """sqrt(b2)/b1, eq. (53). Negative when b1 is (a RHP pair). None
        unless b2 > 0, since otherwise there is no second-order pair."""
        if self.b2 <= 0:
            return None
        return math.sqrt(self.b2) / self.b1 if self.b1 else math.inf

    @property
    def w_n(self) -> float | None:
        """1/sqrt(b2) in rad/s, eq. (51). None unless b2 > 0."""
        return 1 / math.sqrt(self.b2) if self.b2 > 0 else None

    @property
    def f_n(self) -> float | None:
        return self.w_n / (2 * np.pi) if self.b2 > 0 else None

    def tau_second(self, j: str, i: str) -> float | None:
        """tau_j^i: element j with element i at its infinite value. A pair
        with no b2 term has tau_j^i = 0 if tau_i^0 is nonzero."""
        for p in self.pairs:
            if (p.i, p.j) == (i, j):
                return p.tau_j_i
            if (p.j, p.i) == (i, j):
                return p.tau_i_j
        taus = {r.name: r.tau for r in self.rows}
        if i not in taus or j not in taus:
            raise KeyError((j, i))
        return 0.0 if taus[i] else None

    @property
    def dominant_kind(self) -> str:
        """'real' (LHP), 'complex' (LHP pair), 'rhp', or 'none' if D(s) has
        no finite root."""
        p = self.dominant_pole
        if math.isnan(abs(p)):
            return "none"
        if p.real > _RHP_TOL * abs(p):
            return "rhp"
        return "complex" if abs(p.imag) > _COMPLEX_TOL * abs(p) else "real"

    @property
    def validity_applicable(self) -> bool:
        """The ZVT sum estimates one real LHP pole; nothing else compares."""
        return self.dominant_kind == "real" and self.tau_total > 0

    @property
    def validity(self) -> float:
        """f_zvtc / f_dominant: ~1 for a well-separated dominant pole. NaN,
        not a number to misread, when the dominant pole is complex or in the
        RHP or the ZVT sum is not positive; see validity_applicable."""
        if not self.validity_applicable:
            return float("nan")
        return self.f_zvtc / self.f_dominant

    def report(self) -> str:
        if self.validity_applicable:
            check = (f"true dominant pole {eng(self.f_dominant, 'Hz')}, "
                     f"estimate/true = {self.validity:.2f}")
        elif self.dominant_kind == "none":
            check = "no finite pole: estimate/true not applicable"
        else:
            kind = {"rhp": "in the RHP", "complex": "complex"}.get(
                self.dominant_kind, "real but the sum of tau is not positive")
            check = (f"dominant pole |p| = {eng(self.f_dominant, 'Hz')} is "
                     f"{kind}: estimate/true not applicable")
        lines = [
            f"ZVTC bandwidth attribution: sum(tau) = {self.tau_total:.4g} s "
            f"-> f_-3dB(ZVTC) ~ {eng(self.f_zvtc, 'Hz')} ({check})",
        ]
        for r in self.rows:
            line = (f"  {r.name:24s} {r.value:10.3g}  tau = {r.tau:10.3g} s"
                    f"  ({100 * r.share:5.1f} %)")
            if r.h is not None:
                line += f"  H^i = {r.h:10.4g}"
            if r.tau_bar is not None:
                line += f"  tau-bar = {r.tau_bar:10.3g} s"
            lines.append(line)
        if self.inp is not None:
            lines.append(self._modified_line())
        lines.append(self._second_order_line())
        for p in self.pairs[:_REPORT_PAIRS]:
            lines.append(
                f"  i = {p.i}, j = {p.j}: term = {p.term:.3g} s^2, "
                f"tau_j^i = {_opt(p.tau_j_i)}, tau_i^j = {_opt(p.tau_i_j)}")
        if len(self.pairs) > _REPORT_PAIRS:
            lines.append(f"  ... {len(self.pairs) - _REPORT_PAIRS} smaller terms")
        return "\n".join(lines)

    def _modified_line(self) -> str:
        head = f"Modified ZVT, {self.inp} -> {self.out}: H^0 = {self.h0:.4g}"
        if self.tau_bar_total is None:
            return f"{head}; not applicable, H^0 = 0"
        head += f", sum(tau-bar) = {self.tau_bar_total:.4g} s"
        if self.f_modified is None:
            return f"{head}; not applicable, the sum is not positive"
        return f"{head} -> f_-3dB ~ {eng(self.f_modified, 'Hz')}"

    def _second_order_line(self) -> str:
        head = f"Second order: b1 = {self.b1:.4g} s, b2 = {self.b2:.4g} s^2"
        if self.b2 <= 0:
            return f"{head}; Q and f_n not applicable"
        return f"{head}, Q = {self.q:.3g}, f_n = {eng(self.f_n, 'Hz')}"


def _opt(x: float | None) -> str:
    return "indeterminate" if x is None else f"{x:.3g} s"


def _float(q) -> float:
    return float(QQ.to_sympy(q))


def bandwidth_contributions(analyzer, inp: str | None = None,
                            out: str | None = None) -> BandwidthReport:
    if (inp is None) != (out is None):
        raise MnaError("bandwidth report: give both an input and an output, "
                       "or neither")
    if inp is None:
        system = build_mna(analyzer.primitives, analyzer.flat.ground,
                           None, analyzer._alias)
    else:
        system = analyzer.system(inp)
        if out not in system.node_index:
            raise MnaError(f"output node {out!r} not found (or it is ground)")

    reactive: dict[str, list] = {}                      # name -> stamps of C_i
    for p in analyzer.primitives:
        if p.kind in ("c", "l", "cx"):
            reactive[symbol_name(p, analyzer._alias)] = []

    subs = {}
    x = {}                                              # element -> value
    for name, symb in system.symbols.items():
        v = system.values.get(name)
        if v is None:
            raise MnaError(f"bandwidth report: no numeric value for {name}")
        if name in reactive:
            x[name] = QQ.from_sympy(sp.Rational(repr(v)))
        else:
            subs[symb] = sp.Rational(repr(v))
    A = system.A.xreplace(subs)                         # G + s sum_i x_i C_i
    subs.update({system.symbols[n]: QQ.to_sympy(v) for n, v in x.items()})
    A_num = system.A.xreplace(subs)

    base = sp.Poly(_det(A_num), S)
    coeffs = list(reversed(base.all_coeffs()))          # ascending in s
    if not coeffs or coeffs[0] == 0:
        raise MnaError("bandwidth report: singular network at DC")
    a = [QQ.from_sympy(c / coeffs[0]) for c in coeffs]  # a0 = 1
    a1 = a[1] if len(a) > 1 else QQ(0)
    a2 = a[2] if len(a) > 2 else QQ(0)

    G = A.xreplace({S: sp.Integer(0)})
    names_of = {system.symbols[n]: n for n in reactive}
    for (r, c), e in (A.xreplace({S: sp.Integer(1)}) - G).todok().items():
        for symb, k in e.as_coefficients_dict().items():
            reactive[names_of[symb]].append((r, c, QQ.from_sympy(k)))
    Gi = DomainMatrix.from_Matrix(G).convert_to(QQ).inv().to_list()

    def tr(i: str):                                     # tr(M_i)
        return sum((k * Gi[c][r] for r, c, k in reactive[i]), QQ(0))

    def tr2(i: str, j: str):                            # tr(M_i M_j)
        return sum((ki * kj * Gi[ci][rj] * Gi[cj][ri]
                    for ri, ci, ki in reactive[i]
                    for rj, cj, kj in reactive[j]), QQ(0))

    names = list(reactive)
    unit = {n: tr(n) for n in names}                    # tau_i per unit x_i
    tau = {n: unit[n] * x[n] for n in names}
    if sum(tau.values(), QQ(0)) != a1:
        raise MnaError(
            "bandwidth report: sum of per-element time constants does not "
            "reproduce a1/a0 — bookkeeping bug")

    terms = {}                                          # (i, j) in names order
    for m, i in enumerate(names):
        for j in names[m:]:
            if i == j:
                term = (unit[i] ** 2 - tr2(i, i)) / 2 * x[i] ** 2
            else:
                term = (unit[i] * unit[j] - tr2(i, j)) * x[i] * x[j]
            if term != 0:
                terms[i, j] = term
    if sum(terms.values(), QQ(0)) != a2:
        raise MnaError(
            "bandwidth report: per-pair terms do not reproduce a2/a0 — "
            "bookkeeping bug")

    h, tau_bar, h0 = {}, {}, None
    tau_bar_total = f_modified = None
    if inp is not None:
        h, tau_bar, h0 = _transfer_constants(
            system, A_num, coeffs[0], out, Gi, reactive, unit, tau, x)
        if h0 != 0:
            tau_bar_total = sum(tau_bar.values())
            if tau_bar_total > 0:
                f_modified = 1.0 / (2 * np.pi * tau_bar_total)

    tt = _float(a1)
    # frequency-scaled rooting: this site had NO normalization at all,
    # so a det ladder past float64 overflowed outright -- and f_dom is
    # the smallest root, exactly the one degradation eats first
    from ..engine.mna import scaled_roots

    roots = scaled_roots(base.all_coeffs())
    roots = roots[np.abs(roots) > 0]
    # no root at all: D(s) is constant, e.g. tau = 0 with a single element
    pole = complex(roots[np.argmin(np.abs(roots))]) if len(roots) else complex("nan")
    f_dom = abs(pole) / (2 * np.pi)

    out_rows = [
        BandwidthRow(
            name=n,
            value=float(system.values.get(n, float("nan"))),
            tau=_float(tau[n]),
            share=_float(tau[n] / a1) if a1 else 0.0,
            h=h.get(n),
            tau_bar=tau_bar.get(n),
        )
        for n in names
    ]
    out_rows.sort(key=lambda r: -abs(r.tau))

    def second(term, i):
        return _float(term / tau[i]) if tau[i] else None

    pairs = [BandwidthPair(i, j, _float(t), second(t, i), second(t, j))
             for (i, j), t in terms.items()]
    pairs.sort(key=lambda p: -abs(p.term))
    return BandwidthReport(
        rows=out_rows,
        tau_total=tt,
        f_zvtc=1.0 / (2 * np.pi * tt) if tt else float("inf"),
        f_dominant=f_dom,
        dominant_pole=pole,
        b2=_float(a2),
        pairs=pairs,
        inp=inp,
        out=out,
        h0=None if h0 is None else _float(h0),
        tau_bar_total=tau_bar_total,
        f_modified=f_modified,
    )


def _transfer_constants(system, A_num, a0, out, Gi, reactive, unit, tau, x):
    """H^0 (exact) and, per element, H^i and tau-bar_i, checked against N(s)."""
    k_out = system.node_index[out]
    n = len(Gi)
    z = [QQ.from_sympy(e) for e in system.z]
    y0 = [sum((Gi[r][c] * z[c] for c in range(n)), QQ(0)) for r in range(n)]
    h0 = y0[k_out]

    h, tau_bar = {}, {}
    n1 = QQ(0)
    for name, stamps in reactive.items():
        back = sum((k * Gi[k_out][r] * y0[c] for r, c, k in stamps), QQ(0))
        n1 += tau[name] * h0 - back * x[name]           # x_i term of N1/a0
        if unit[name]:
            hi = h0 - back / unit[name]
            h[name] = _float(hi)
            if h0:
                tau_bar[name] = _float(tau[name] * (1 - abs(hi / h0)))
        elif back == 0:
            # tau_i = 0 and no x_i term in N1 either: to this order the gain
            # does not depend on x_i at all, so its limit is H^0 (Example 8's
            # C, shorted by the zero-value L)
            h[name] = _float(h0)
            if h0:
                tau_bar[name] = 0.0
        else:
            # N1 grows with x_i, D(0) does not: the gain is unbounded, but
            # tau_i H^i = -back x_i is the finite product that (41) needs
            h[name] = -math.copysign(math.inf, _float(back))
            if h0:
                tau_bar[name] = -abs(_float(back * x[name] / h0))

    Ak = A_num.copy()
    Ak[:, k_out] = system.z
    num = list(reversed(sp.Poly(_det(Ak), S).all_coeffs()))
    n0_true = QQ.from_sympy(num[0] / a0) if num else QQ(0)
    n1_true = QQ.from_sympy(num[1] / a0) if len(num) > 1 else QQ(0)
    if n0_true != h0 or n1_true != n1:
        raise MnaError(
            "bandwidth report: transfer constants do not reproduce N(s) — "
            "bookkeeping bug")
    return h, tau_bar, h0
