"""Symbolic Modified Nodal Analysis over sympy.

Builds A(s)·x = z from primitives and extracts a transfer function via
Cramer's rule with fraction-free determinants (well-suited to symbolic
matrices). Hybrid numeric-symbolic mode substitutes exact Rationals for all
symbols not explicitly kept, before the determinants are taken.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
import sympy as sp

from ..keep import ALL, is_all

from .primitives import Primitive

S = sp.Symbol("s")


class MnaError(ValueError):
    pass


def sanitize(name: str) -> str:
    out = re.sub(r"[^0-9A-Za-z_]", "_", name)
    return out + "_" if out == "s" else out


def symbol_name(prim: Primitive, alias: dict[str, str] | None = None) -> str:
    inst = (alias or {}).get(prim.inst, prim.inst)
    return sanitize(f"{prim.param}_{inst}" if prim.param else inst)


@dataclass
class MnaSystem:
    A: sp.Matrix
    z: sp.Matrix
    unknowns: list[str]            # node names then branch labels, row order
    node_index: dict[str, int]
    branch_index: dict[str, int]   # instance name of V-defined element -> row
    symbols: dict[str, sp.Symbol]
    values: dict[str, float]       # symbol name -> numeric value (if known)
    stamp_counts: dict[str, int] = field(default_factory=dict)
    reciprocal: set[str] = field(default_factory=set)  # stamped as 1/sym ('r')


def build_mna(
    primitives: list[Primitive],
    ground: tuple[str, ...],
    input_name: str | None = None,
    alias: dict[str, str] | None = None,
) -> MnaSystem:
    """Assemble the MNA system. All independent sources are zeroed except
    `input_name`, whose value is set to exactly 1 (unit excitation)."""
    gnd = set(ground)
    nodes = sorted({n for p in primitives for n in p.nodes if n not in gnd})
    nidx = {n: i for i, n in enumerate(nodes)}
    n_nodes = len(nodes)

    # branch-current unknowns; balun consumes two rows
    bidx: dict[str, int] = {}
    branch_labels: list[str] = []
    nxt = n_nodes
    for p in primitives:
        if p.kind in ("vsrc", "vcvs", "l"):
            bidx[p.inst] = nxt
            branch_labels.append(p.inst)
            nxt += 1
        elif p.kind == "balun":
            bidx[p.inst] = nxt
            branch_labels.extend([p.inst, p.inst + "#2"])
            nxt += 2
    dim = nxt
    if dim == 0:
        raise MnaError("empty circuit")

    A = sp.zeros(dim, dim)
    z = sp.zeros(dim, 1)
    symbols: dict[str, sp.Symbol] = {}
    values: dict[str, float] = {}
    stamp_counts: dict[str, int] = {}
    reciprocal: set[str] = set()

    def sym(prim: Primitive) -> sp.Symbol:
        name = symbol_name(prim, alias)
        stamp_counts[name] = stamp_counts.get(name, 0) + 1
        if prim.kind == "r":
            reciprocal.add(name)
        if name not in symbols:
            symbols[name] = sp.Symbol(name, positive=True) if prim.is_positive \
                else sp.Symbol(name, real=True)
            if prim.value is not None:
                values[name] = prim.value
        elif prim.value is not None:
            prev = values.get(name)
            if prev is None:
                values[name] = prim.value
            elif prev != 0 and abs(prim.value - prev) > 0.05 * abs(prev):
                import warnings
                warnings.warn(
                    f"matched symbol {name}: values differ by >5% "
                    f"({prev:g} vs {prim.value:g}); keeping {prev:g}"
                )
        return symbols[name]

    def idx(node: str) -> int | None:
        return None if node in gnd else nidx[node]

    def add(i: int | None, j: int | None, val) -> None:
        if i is not None and j is not None:
            A[i, j] += val

    seen_input = False
    for p in primitives:
        ni = [idx(n) for n in p.nodes]
        if p.kind == "r":
            g = 1 / sym(p)
            a, b = ni
            add(a, a, g); add(b, b, g); add(a, b, -g); add(b, a, -g)
        elif p.kind == "g":
            g = sym(p)
            a, b = ni
            add(a, a, g); add(b, b, g); add(a, b, -g); add(b, a, -g)
        elif p.kind == "c":
            g = S * sym(p)
            a, b = ni
            add(a, a, g); add(b, b, g); add(a, b, -g); add(b, a, -g)
        elif p.kind == "vccs":
            gm = sym(p)
            a, b, c, d = ni
            add(a, c, gm); add(a, d, -gm); add(b, c, -gm); add(b, d, gm)
        elif p.kind == "cx":
            # trans-capacitance: a VCCS with gain s*C (C signed; used for the
            # exact charge-matrix MOS model)
            gm = S * sym(p)
            a, b, c, d = ni
            add(a, c, gm); add(a, d, -gm); add(b, c, -gm); add(b, d, gm)
        elif p.kind == "l":
            br = bidx[p.inst]
            a, b = ni
            add(a, br, 1); add(b, br, -1)
            add(br, a, 1); add(br, b, -1)
            A[br, br] -= S * sym(p)
        elif p.kind == "vcvs":
            br = bidx[p.inst]
            a, b, c, d = ni
            mu = sym(p)
            add(a, br, 1); add(b, br, -1)
            add(br, a, 1); add(br, b, -1)
            add(br, c, -mu); add(br, d, mu)
        elif p.kind == "balun":
            # ideal balun, nodes (d, c, p, n):
            #   v(p) = v(c) + v(d)/2 ; v(n) = v(c) - v(d)/2
            # symmetric (reciprocal) stamp: current incidence is the
            # transpose of the constraint rows -> power-conserving
            half = sp.Rational(1, 2)
            d, c, a, b = ni           # d, c, p, n
            r1, r2 = bidx[p.inst], bidx[p.inst] + 1
            for row, node, coef in (
                (r1, a, 1), (r1, c, -1), (r1, d, -half),
                (r2, b, 1), (r2, c, -1), (r2, d, half),
            ):
                add(row, node, coef)
                add(node, row, coef)
        elif p.kind == "vsrc":
            br = bidx[p.inst]
            a, b = ni
            add(a, br, 1); add(b, br, -1)
            add(br, a, 1); add(br, b, -1)
            if p.inst == input_name:
                z[br, 0] = 1
                seen_input = True
        elif p.kind == "isrc":
            a, b = ni
            if p.inst == input_name:
                # unit current flowing p->n inside the source: extracts from
                # p, injects into n
                if a is not None:
                    z[a, 0] += -1
                if b is not None:
                    z[b, 0] += 1
                seen_input = True

    if input_name is not None and not seen_input:
        raise MnaError(
            f"input {input_name!r} is not a vsource/isource instance in this circuit"
        )

    unknowns = nodes + branch_labels
    return MnaSystem(A, z, unknowns, nidx, bidx, symbols, values,
                     stamp_counts, reciprocal)


@dataclass
class TransferFunction:
    expr: sp.Expr                  # N(s)/D(s), cancelled
    values: dict[str, float] = field(default_factory=dict)
    symbols: dict[str, sp.Symbol] = field(default_factory=dict)
    s: sp.Symbol = S

    @cached_property
    def num_den(self) -> tuple[sp.Poly, sp.Poly]:
        n, d = sp.fraction(sp.together(self.expr))
        return sp.Poly(sp.expand(n), self.s), sp.Poly(sp.expand(d), self.s)

    def dc_gain(self) -> sp.Expr:
        n, d = self.num_den
        return sp.simplify(
            n.eval(0) / d.eval(0)) if d.eval(0) != 0 else sp.zoo

    def _subs_map(self) -> dict:
        return {
            self.symbols[name]: sp.Float(val)
            for name, val in self.values.items()
            if name in self.symbols
        }

    def numeric_expr(self) -> sp.Expr:
        """The TF with every known symbol replaced by its numeric value."""
        e = self.expr.xreplace(self._subs_map())
        free = e.free_symbols - {self.s}
        if free:
            raise MnaError(f"no numeric value for symbols: {sorted(map(str, free))}")
        return e

    def numeric(self, f):
        """Evaluate the TF at frequency array f (Hz) -> complex ndarray.

        Uses fast numpy evaluation; for large circuits whose exact integer
        coefficients overflow float64, falls back to arbitrary-precision
        mpmath per point (the ratio itself is O(gain) and fits)."""
        expr = self.numeric_expr()
        fvec = np.atleast_1d(np.asarray(f, dtype=float))
        w = 2j * np.pi * fvec

        def _shape(o):                           # broadcast s-independent TFs
            o = np.asarray(o, dtype=complex)
            return o if o.shape == w.shape else np.full(w.shape, complex(o))

        try:
            fn = sp.lambdify(self.s, expr, "numpy")
            out = _shape(fn(w))
            mask = fvec != 0
            if mask.any() and not np.all(np.isfinite(out[mask])):
                raise OverflowError
        except (OverflowError, TypeError, ValueError):
            import mpmath

            fn = sp.lambdify(self.s, expr, "mpmath")
            with mpmath.workdps(50):
                out = _shape([complex(fn(mpmath.mpc(0, wi.imag)))
                              for wi in w])
        return out

    def _float_coeffs(self, poly: sp.Poly) -> list[complex]:
        # normalize by the largest-magnitude coefficient so conversion to
        # float64 is safe even when the raw integers have hundreds of digits;
        # polynomial roots are invariant to an overall scale factor
        subs = self._subs_map()
        coeffs = [c.xreplace(subs) for c in poly.all_coeffs()]
        mags = [abs(c) for c in coeffs if c != 0]
        if not mags:
            return [complex(c) for c in coeffs]
        scale = max(mags)
        return [complex((c / scale).evalf()) for c in coeffs]

    def poles(self) -> np.ndarray:
        """Numeric poles in Hz (complex), sorted by magnitude."""
        r = np.roots(self._float_coeffs(self.num_den[1])) / (2 * np.pi)
        return r[np.argsort(np.abs(r))]

    def zeros(self) -> np.ndarray:
        r = np.roots(self._float_coeffs(self.num_den[0])) / (2 * np.pi)
        return r[np.argsort(np.abs(r))]

    def simplify(self, **kwargs):
        """Numeric-guided pruning within an error budget; see
        analysis/simplify.py. Returns a SimplifiedTF."""
        from ..analysis.simplify import simplify_tf

        return simplify_tf(self, **kwargs)


def _det(M: sp.Matrix) -> sp.Expr:
    """Determinant via DomainMatrix (fast structured-domain arithmetic, e.g.
    QQ[s, gm_M1, ...]); falls back to Berkowitz on exotic entries."""
    try:
        from sympy.polys.matrices import DomainMatrix

        dM = DomainMatrix.from_Matrix(M)
        if not dM.domain.is_Field:
            dM = dM.to_field()
        return dM.domain.to_sympy(dM.det())
    except Exception:
        return M.det(method="berkowitz")


def hybrid_split(
    system: MnaSystem, keep: list[str]
) -> tuple[dict, list[str]]:
    """Split symbols for hybrid mode: exact-Rational substitutions for
    non-kept symbols, plus the list of kept symbol names.

    Every `keep` entry (raw or sanitize()d) must match at least one symbol by
    exact name or owning-instance suffix; unmatched entries raise MnaError
    rather than being silently substituted away."""
    forms = {k: {k, sanitize(k)} for k in keep}
    subs: dict = {}
    kept_names: list[str] = []
    missing = []
    matched: set[str] = set()
    for name, symb in system.symbols.items():
        hit = False
        for k, fs in forms.items():
            if name in fs or any(name.endswith("_" + f) for f in fs):
                matched.add(k)
                hit = True
        if hit:
            kept_names.append(name)
            continue
        if name in system.values:
            subs[symb] = sp.Rational(repr(system.values[name]))
        else:
            missing.append(name)
    unmatched = [k for k in keep if k not in matched]
    if unmatched:
        parts = {p for k in unmatched for p in sanitize(k).split("_") if p}
        cands = sorted((n for n in system.symbols
                        if any(p in n for p in parts)),
                       key=lambda n: (-sum(p in n for p in parts), n))
        raise MnaError(
            f"keep entries matched no symbol: {unmatched}. Symbols are named "
            f"param_instance (e.g. 'gm_MN0', not 'MN0.gm'); an instance name "
            f"alone keeps all its symbols. Close matches: {cands[:6]}")
    if missing:
        raise MnaError(f"hybrid mode: no numeric value for {sorted(missing)}")
    return subs, kept_names


def solve_tf(
    system: MnaSystem,
    output: str,
    keep=ALL,
    method: str = "auto",
    progress=None,
) -> TransferFunction:
    """Cramer's-rule transfer function from the (unit-excited) system to the
    voltage at node `output`.

    keep=ALL (the default): fully symbolic. keep=[...] hybrid — only symbols
    whose name or owning-instance suffix matches an entry stay symbolic; the
    rest are replaced by exact Rationals of their numeric values. keep=None
    is an alias of [] (fully numeric).

    method: 'auto' uses the multilinear-interpolation solver for hybrid
    solves with kept symbols (docs/multilinear-solver-plan.md), the direct
    determinant otherwise; 'interp' / 'direct' force a path.
    """
    if method not in ("auto", "direct", "interp"):
        raise MnaError(f"unknown method {method!r}")
    if not is_all(keep):
        keep = [] if keep is None else list(keep)
    if method != "direct" and not is_all(keep):
        _, kept_names = hybrid_split(system, keep)
        if kept_names:
            # tiny grids (e.g. one unmatched kept symbol) don't amortize the
            # interpolation machinery — the direct det is faster there
            grid = 1
            for n in kept_names:
                grid *= max(1, system.stamp_counts.get(n, 1)) + 1
            if method == "interp" or grid > 4:
                from .interp import solve_tf_interp

                return solve_tf_interp(system, output, keep,
                                       progress=progress)

    if isinstance(output, int):                 # raw unknown index: node OR
        k = output                              # branch current (loop gain)
        if not 0 <= k < system.A.rows:
            raise MnaError(f"output index {k} out of range")
    elif output not in system.node_index:
        raise MnaError(f"output node {output!r} not found (or it is ground)")
    else:
        k = system.node_index[output]

    A, z = system.A, system.z
    if not is_all(keep):
        subs, _ = hybrid_split(system, keep)
        A = A.xreplace(subs)

    den = _det(A)
    if den == 0:
        raise MnaError("singular MNA matrix: floating node or short loop?")
    Ak = A.copy()
    Ak[:, k] = z
    num = _det(Ak)
    expr = sp.cancel(num / den)
    return TransferFunction(
        expr=expr, values=dict(system.values), symbols=dict(system.symbols)
    )
