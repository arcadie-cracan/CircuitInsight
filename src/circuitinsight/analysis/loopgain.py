"""Loop gain via Tian's loop-based two-port algorithm, on the MNA system.

Implements Eq. 30 of Tian, Visvanathan, Hantgan, Kundert, "Striving for
small-signal stability", IEEE Circuits & Devices Mag., Jan. 2001 -- the
BILATERAL return-loop model, which accounts for reverse transmission around
the loop (the unilateral 2-readout combination T = -(TvTi-1)/(Tv+Ti+2) is
the paper's *normal* return ratio and errs by the reverse-transmission term,
~1 dB in this fixture's deep-cancellation tail; see docs/loopgain-plan.md).

The probe is a designated 0-V source branch in the loop (schematic: analogLib
``iprobe``, mapped to a CIN vsource; the branch is a DC short, so the bias is
untouched). Two analyses on the SAME matrix A (the paper's double-injection,
Eqs. 22-23), four readouts:

- E1, series unit voltage on the probe's branch row (exactly what
  ``build_mna(input_name=probe)`` produces):   B = -i_b,  D = v_p ;
- E2, unit current into the p-side node:       A = -i_b,  C = v_q
  (with the 0-V branch present, v_q = v_p identically here).

    T = (2*(A*D - B*C) - A + D) / (2*(B*C - A*D) + A - D + 1)      [Eq. 30]

All readouts are Cramer ratios over the shared determinant Dt = det A, so T
assembles into an exact rational in s (numerators Nb_v, Np_v from E1 and
Nb_p, Nq_p from E2):

    T = [2*(Nb_v*Nq_p - Nb_p*Np_v) + Dt*(Nb_p + Np_v)]
        / [2*(Nb_p*Np_v - Nb_v*Nq_p) - Dt*(Nb_p + Np_v) + Dt**2]

Sign convention matches Spectre ``stb`` (which implements the same paper):
arg T = +180 deg at DC, PM = arg T at |T| = 1, GM at arg T = 0. Validated
against the stb fixture to <=0.02 dB over the FULL sweep to 10 GHz,
reverse-transmission tail included. Dead ends kept for the record: the
across-branch current injection (degenerate: everything flows through the
0-V short), and a branch-opening determinant ratio (a wire is not a Bode
element; the determinant identity lives in return_ratio() below, per the
paper's device-based gain-nulling algorithm).
"""
from __future__ import annotations

import dataclasses

import numpy as np

import sympy as sp

from ..engine.mna import (MnaError, MnaSystem, S, TransferFunction,
                          _det, hybrid_split, sanitize, solve_tf)
from ..keep import is_all

# the cancel-ops guard lives in engine.interp: one number for
# every "skip the final cancel() when the expression is huge"
# site
from ..engine.interp import _CANCEL_OPS_LIMIT


def _probe_nodes(system: MnaSystem, probe: str) -> tuple[int, int]:
    """Node-row indices (p, q) of the probe branch, from its incidence
    column: +1 on the p row, -1 on the q row."""
    ib = system.branch_index[probe]
    n_nodes = len(system.node_index)
    p = q = None
    for i in range(n_nodes):
        v = system.A[i, ib]
        if v == 1:
            p = i
        elif v == -1:
            q = i
    if p is None or q is None:
        raise MnaError(
            f"probe {probe!r}: could not identify both branch nodes "
            f"(is one side grounded? ground the loop elsewhere)")
    return p, q


class NumericTian:
    """The numeric Eq. 30 for SCAN pricing: per frequency, the Tian
    loop gain needs the series-voltage solve x_v = Z z (the system is
    built with input_name=probe, so z is the unit probe injection) and
    the p-node current solve x_i = Z[:, p] -- a COLUMN of the same
    inverse. Six scalars combine into T, so a candidate's rank-one
    update of Z updates T through the same six reads: the ac-ground
    and removal scans price against the loop gain with the identical
    one-inverse-per-frequency machinery they use for a transfer."""

    def __init__(self, system: MnaSystem, probe: str):
        if probe not in system.branch_index:
            raise MnaError(f"probe {probe!r} is not a voltage-defined "
                           f"branch")
        if system.z[system.branch_index[probe], 0] != 1:
            raise MnaError(f"system was not built with "
                           f"input_name={probe!r}")
        self.ib = system.branch_index[probe]
        self.p, self.q = _probe_nodes(system, probe)

    #: the rows a scan must read/update per solve
    @property
    def rows(self):
        return (self.ib, self.p, self.q)

    @staticmethod
    def combine(xv_ib, xv_p, xi_ib, xi_q):
        """T from the four Tian reads (stb sign convention)."""
        B = -xv_ib
        D = xv_p
        A = -xi_ib
        C = xi_q
        num = 2.0 * (A * D - B * C) - A + D
        den = 2.0 * (B * C - A * D) + A - D + 1.0
        return -num / den

    def response(self, Z, z):
        """T at one frequency from the full inverse Z and the probe
        rhs z, plus the two solve vectors for candidate updates."""
        xv = Z @ z
        xi = Z[:, self.p]
        return self.combine(xv[self.ib], xv[self.p],
                            xi[self.ib], xi[self.q]), xv, xi


def numeric_loop_response(primitives, ground, probe, freqs,
                          alias=None) -> np.ndarray:
    """T(f) of the circuit, numerically: one inverse per frequency
    through NumericTian. The scans' joint re-solve and the reduction
    bench's measured comparison, for the loop-gain contract."""
    from ..engine.mna import build_mna

    system = build_mna(primitives, ground, probe, alias)
    tian = NumericTian(system, probe)
    subs = {system.symbols[n]: sp.Float(v)
            for n, v in system.values.items() if n in system.symbols}
    A = system.A.xreplace(subs)
    if set(A.free_symbols) - {S}:
        raise MnaError("numeric loop response needs numeric values")
    fn = sp.lambdify(S, A, "numpy")
    z = np.array(system.z.tolist(), dtype=complex).ravel()
    out = np.empty(len(freqs), dtype=complex)
    for i, f in enumerate(np.atleast_1d(np.asarray(freqs, dtype=float))):
        Z = np.linalg.inv(np.asarray(fn(2j * np.pi * f), dtype=complex))
        out[i], _xv, _xi = tian.response(Z, z)
    return out


def tian_determinants(system: MnaSystem, probe: str) -> dict:
    """The five Cramer determinants the numeric Eq. 30 combines, as
    exact polynomials in s at the system's numeric values:

      Dt    the shared system determinant
      Nb_v  i_b under the series-voltage injection E1  (B = -Nb_v/Dt)
      Np_v  v_p under E1                               (D =  Np_v/Dt)
      Nb_p  i_b under the p-side current injection E2  (A = -Nb_p/Dt)
      Nq_p  v_q under E2                               (C =  Nq_p/Dt)

    Exposed because each determinant is MULTILINEAR in the element
    values -- affine in every element that stamps once -- while the
    Eq. 30 combination of them is quadratic (den = -num + Dt^2).
    A parametric study therefore needs only the four corner
    determinants of a pair to pin that pair's whole dependence
    EXACTLY, instead of one assembled solve per parameter point: with
    d_S the determinant with the elements of S removed,

      d(a, b) = d_ab + (d_b - d_ab) a + (d_a - d_ab) b
                     + (d_0 - d_a - d_b + d_ab) a b

    for scale factors a, b. One nominal, n single-removal and
    C(n,2) pair-removal evaluations thus fix every pair-plane of an
    n-element study, and every root locus and sensitivity jet on it
    follows analytically -- no finite differences, no step size.
    Multiplicity caveat: a symbol carrying k stamps (matched twins
    share one symbol) is degree k, not affine, and needs k+1 points
    per dimension."""
    if probe not in system.branch_index:
        raise MnaError(
            f"probe {probe!r} is not a voltage-defined branch; map the "
            f"schematic iprobe to a 0 V vsource in the CIN")
    ib = system.branch_index[probe]
    if system.z[ib, 0] != 1:
        raise MnaError(
            f"system was not built with input_name={probe!r}; the probe's "
            f"branch row must carry the unit series-voltage injection")
    p, q = _probe_nodes(system, probe)
    z_i = sp.zeros(system.A.rows, 1)
    z_i[p, 0] = 1                       # unit current into the p side (E2)
    subs, _ = hybrid_split(system, [])
    A_num = system.A.xreplace(subs)
    z_v = system.z

    def cramer(col: int, z: sp.Matrix) -> sp.Expr:
        Ak = A_num.copy()
        Ak[:, col] = z
        return _det(Ak)

    return {"Dt": _det(A_num),
            "Nb_v": cramer(ib, z_v), "Np_v": cramer(p, z_v),
            "Nb_p": cramer(ib, z_i), "Nq_p": cramer(q, z_i)}


def tian_combine(d: dict):
    """Eq. 30 from the five determinants: returns (num, den) with
    T = -num/den. The single place the combination is spelled, shared
    by loop_gain's numeric path and by parametric studies that
    reconstruct T from determinants they varied themselves."""
    Dt, Nb_v, Np_v = d["Dt"], d["Nb_v"], d["Np_v"]
    Nb_p, Nq_p = d["Nb_p"], d["Nq_p"]
    num = sp.expand(2 * (Nb_v * Nq_p - Nb_p * Np_v)
                    + Dt * (Nb_p + Np_v))
    den = sp.expand(2 * (Nb_p * Np_v - Nb_v * Nq_p)
                    - Dt * (Nb_p + Np_v) + Dt * Dt)
    return num, den


def loop_gain(system: MnaSystem, probe: str, keep=(),
              progress=None, note=None) -> TransferFunction:
    """Tian loop gain at `probe` (a 0-V vsource branch inside the loop),
    as an exact rational T(s) in the Spectre stb sign convention.

    `system` must be built with ``input_name=probe`` so the series-voltage
    injection RHS is in place. keep follows the tf() convention: [] (the
    default; None is its alias) fully numeric (fast direct-determinant
    path), [names] hybrid, ALL fully symbolic -- the injections route
    through the same hybrid machinery as tf(), so kept symbols stay
    symbolic in T."""
    if probe not in system.branch_index:
        raise MnaError(
            f"probe {probe!r} is not a voltage-defined branch; map the "
            f"schematic iprobe to a 0 V vsource in the CIN")
    ib = system.branch_index[probe]
    if system.z[ib, 0] != 1:
        raise MnaError(
            f"system was not built with input_name={probe!r}; the probe's "
            f"branch row must carry the unit series-voltage injection")

    p, q = _probe_nodes(system, probe)
    z_i = sp.zeros(system.A.rows, 1)
    z_i[p, 0] = 1                       # unit current into the p side (E2)

    if not is_all(keep):
        keep = [] if keep is None else list(keep)
    if not is_all(keep) and not keep:
        # fully numeric: share one determinant across the four Cramer
        # reads. Eq. 30 yields the classical return ratio (positive real
        # at DC under negative feedback); Spectre's loopGain displays
        # -T, and we follow the stb convention throughout
        num, den = tian_combine(tian_determinants(system, probe))
        expr = -num / den
    else:
        # hybrid / fully symbolic: four rational solves on the shared matrix
        # (the series-voltage system as built, plus a z-swapped copy for the
        # p-side current injection), combined per Eq. 30. Each solve is
        # exact, so the nested rational is exact; cancel only when cheap.
        # Each read narrates itself: the Log otherwise shows four
        # unexplained evaluate/reconstruct cycles (field question).
        sys_i = dataclasses.replace(system, z=z_i)

        def _say(k, what):
            if note is not None:
                # "solve", not the literature's "read(out)": the Log
                # narrates COST, and each of these is a full rational
                # solve with its own reconstruction, not an instant read
                note(f"Tian solve {k}/4: {what}")

        _say(1, "branch current under the voltage injection (B)")
        Bv = -solve_tf(system, ib, keep, progress=progress).expr
        _say(2, "p-node voltage under the voltage injection (D)")
        Dv = solve_tf(system, p, keep, progress=progress).expr
        _say(3, "branch current under the current injection (A)")
        Ai = -solve_tf(sys_i, ib, keep, progress=progress).expr
        _say(4, "q-node voltage under the current injection (C)")
        Ci = solve_tf(sys_i, q, keep, progress=progress).expr
        expr = -(2 * (Ai * Dv - Bv * Ci) - Ai + Dv) / \
                (2 * (Bv * Ci - Ai * Dv) + Ai - Dv + 1)
        if sp.count_ops(expr) <= _CANCEL_OPS_LIMIT:
            expr = sp.cancel(sp.together(expr))

    return TransferFunction(expr=expr, values=dict(system.values),
                            symbols=dict(system.symbols))


def reduced_loop_gain(analyzer, probe: str, keep=(), *,
                      tol_db: float = 0.5,
                      fmin: float = 1e3, fmax: float = 1e7,
                      max_elements=None, phase_tol_deg=None,
                      strategy=None, strategy_opts=None,
                      progress=None, note=None):
    """Reduced-ORDER Tian loop gain: keep only the reactances that
    shape T(s) over the band under the tolerance strategy, zero the
    rest, then solve T on the pruned system. The transfer pursuit's
    exact greedy runs unchanged -- only the response evaluator
    differs: T is the Eq.-30 combination of two per-frequency solves
    (series-voltage and p-side current injections), so removing a
    reactance is the same stamp subtraction from Q.

    Returns (TransferFunction, ReactanceReduction)."""
    import math

    from ..engine.mna import S
    from .criteria import make_criterion
    from .sensitivity import (ReactanceReduction, _greedy_elimination,
                              _reactive_symbols, _stamp_derivatives)

    system = analyzer.system(probe)
    if probe not in system.branch_index:
        raise MnaError(
            f"probe {probe!r} is not a voltage-defined branch")
    ib = system.branch_index[probe]
    p, q = _probe_nodes(system, probe)

    subs, missing = {}, []
    for name, symb in system.symbols.items():
        v = system.values.get(name)
        (missing.append(name) if v is None
         else subs.__setitem__(symb, sp.Rational(repr(v))))
    if missing:
        raise MnaError(
            f"reduced loop gain: no numeric value for {sorted(missing)}")
    A_v = system.A.xreplace(subs)
    P = np.array(A_v.xreplace({S: sp.Integer(0)}).tolist(), dtype=complex)
    Q = np.array(A_v.diff(S).tolist(), dtype=complex)
    z_v = np.array(system.z.T.tolist()[0], dtype=complex)
    z_i = np.zeros_like(z_v)
    z_i[p] = 1.0

    ndec = max(1.0, math.log10(fmax / fmin))
    freqs = np.logspace(math.log10(fmin), math.log10(fmax),
                        int(8 * ndec) + 1)
    w = 2j * np.pi * freqs

    def T_of(Q_eff: np.ndarray) -> np.ndarray:
        out = np.empty(len(freqs), dtype=complex)
        for j, wj in enumerate(w):
            An = P + wj * Q_eff
            try:
                xv = np.linalg.solve(An, z_v)
                yi = np.linalg.solve(An, z_i)
            except np.linalg.LinAlgError:
                out[j] = np.nan
                continue
            Bv, Dv = -xv[ib], xv[p]
            Ai, Ci = -yi[ib], yi[q]
            den = 2 * (Bv * Ci - Ai * Dv) + Ai - Dv + 1
            out[j] = (-(2 * (Ai * Dv - Bv * Ci) - Ai + Dv) / den
                      if den != 0 else np.nan)
        return out

    params = _stamp_derivatives(system, subs)
    stamps = {name: (x0, BP, BQ) for name, x0, BP, BQ in params}
    react = sorted(n for n in _reactive_symbols(analyzer) if n in stamps)
    contrib = {c: stamps[c][0] * stamps[c][2] for c in react}

    T_full = T_of(Q)
    m = np.isfinite(T_full) & (np.abs(T_full) > 0)
    mag_full = np.abs(T_full)
    crit = make_criterion(strategy=strategy,
                          strategy_opts=dict(strategy_opts or {}),
                          eps=None, tol_db=tol_db,
                          phase_tol_deg=phase_tol_deg)
    sig = crit.window(freqs, mag_full, m, T_full)
    if not sig.any():
        sig = m
    crit.prepare(freqs, T_full, sig)

    def err(Tr: np.ndarray) -> float:
        return crit.error(freqs, T_full, mag_full, Tr, sig)

    if crit.cap is not None and max_elements is None:
        max_elements = crit.cap
    # BACKWARD elimination, not the transfer tool's add-back: an
    # AC-coupled loop (the common DC-closed/AC-open stb rig) is
    # identically zero with every reactance removed, so a from-zero
    # greedy faces an infinite baseline no single restoration can
    # improve. Starting full, the contract holds by construction and
    # each removal is priced against it.
    selected, errors, Tr = _greedy_elimination(
        T_of, err, contrib, react, Q, crit.tol, crit.unit,
        note, progress)
    baseline = float(err(Tr))
    final = errors[-1] if errors else baseline
    met = bool(final <= crit.tol + 1e-12) and         (max_elements is None or len(selected) <= max_elements)
    if note is not None and not met and             max_elements is not None and len(selected) > max_elements:
        note(f"{len(selected)} reactance(s) remain above the order "
             f"cap {max_elements}: the loop needs them to hold the "
             f"contract")
    red = ReactanceReduction(
        selected=selected, errors_db=errors, baseline_db=baseline,
        tol_db=tol_db, freqs=freqs, metric="complex",
        sig_lo=float(freqs[sig].min()), sig_hi=float(freqs[sig].max()),
        strategy=strategy, met=met,
        metrics=crit.metrics(freqs, Tr, sig))

    from ..keep import norm_keep_list

    keepl = norm_keep_list(keep)
    keep_set = set(() if is_all(keepl) else keepl)
    vals = dict(system.values)
    for n in react:
        if n not in set(selected) | keep_set and n in vals:
            vals[n] = 0.0
    T = loop_gain(dataclasses.replace(system, values=vals), probe,
                  keepl, progress=progress)
    return T, red


def return_ratio(system: MnaSystem, source: str,
                 keep=()) -> TransferFunction:
    """Bode/Rosenstark return ratio of a designated controlled source, via
    the return-difference determinant identity

        F(s) = 1 + RR(s) = det A(k) / det A(k -> 0)

    which holds for a controlled-source gain k (each stamp is rank-one in
    k). One determinant with k kept symbolic yields both evaluations, so
    the hybrid machinery applies: extra `keep` names stay symbolic in RR.

    `source` is the symbol name as used in keep sets (e.g. ``gm_I0.MP2``).
    When matched devices share the symbol, RR is the joint return ratio of
    ALL stamps carrying it -- the pair acting together.

    Convention: classical Bode -- RR is positive real at DC for negative
    feedback. The Spectre-stb loop gain of a probe in the same loop is
    approximately ``-RR`` wherever the loop signal flows through the
    source (they differ by forward feedthrough that bypasses it, e.g. the
    C_C path at high frequency). `system` needs no excitation
    (``input_name=None``)."""
    name = sanitize(source)
    if name not in system.symbols:
        cands = sorted(n for n in system.symbols if name.split("_")[0] in n)
        raise MnaError(
            f"source {source!r}: no such symbol; close matches: {cands[:6]}")
    if name not in system.values:
        raise MnaError(f"source {source!r} has no numeric operating-point "
                       f"value to evaluate the return difference at")

    x = system.symbols[name]
    val = sp.Rational(repr(system.values[name]))
    if is_all(keep):                              # fully symbolic
        A_h = system.A
    else:
        subs, _ = hybrid_split(system,
                               [name] + list(() if keep is None else keep))
        A_h = system.A.xreplace(subs)

    D = _det(A_h)                                 # poly in s, x (, kept)
    num = sp.expand(D.xreplace({x: val}) - D.xreplace({x: sp.Integer(0)}))
    den = sp.expand(D.xreplace({x: sp.Integer(0)}))
    expr = num / den
    if sp.count_ops(expr) <= _CANCEL_OPS_LIMIT:
        expr = sp.cancel(expr)
    return TransferFunction(expr=expr, values=dict(system.values),
                            symbols=dict(system.symbols))
