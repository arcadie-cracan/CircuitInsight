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

from ..engine.mna import (MnaError, MnaSystem, TransferFunction, _det,
                          hybrid_split, sanitize, solve_tf)
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


def loop_gain(system: MnaSystem, probe: str, keep=(),
              progress=None) -> TransferFunction:
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
        # fully numeric: share one determinant across the four Cramer reads
        subs, _ = hybrid_split(system, [])
        A_num = system.A.xreplace(subs)
        z_v = system.z

        def cramer(col: int, z: sp.Matrix) -> sp.Expr:
            Ak = A_num.copy()
            Ak[:, col] = z
            return _det(Ak)

        Dt = _det(A_num)
        Nb_v = cramer(ib, z_v)          # i_b under E1  (B = -Nb_v/Dt)
        Np_v = cramer(p, z_v)           # v_p under E1  (D =  Np_v/Dt)
        Nb_p = cramer(ib, z_i)          # i_b under E2  (A = -Nb_p/Dt)
        Nq_p = cramer(q, z_i)           # v_q under E2  (C =  Nq_p/Dt)
        # Eq. 30 yields the classical return ratio (positive real at DC
        # under negative feedback); Spectre's loopGain displays -T, and we
        # follow the stb convention throughout
        num = sp.expand(2 * (Nb_v * Nq_p - Nb_p * Np_v)
                        + Dt * (Nb_p + Np_v))
        den = sp.expand(2 * (Nb_p * Np_v - Nb_v * Nq_p)
                        - Dt * (Nb_p + Np_v) + Dt * Dt)
        expr = -num / den
    else:
        # hybrid / fully symbolic: four rational solves on the shared matrix
        # (the series-voltage system as built, plus a z-swapped copy for the
        # p-side current injection), combined per Eq. 30. Each solve is
        # exact, so the nested rational is exact; cancel only when cheap.
        sys_i = dataclasses.replace(system, z=z_i)
        Bv = -solve_tf(system, ib, keep, progress=progress).expr
        Dv = solve_tf(system, p, keep, progress=progress).expr
        Ai = -solve_tf(sys_i, ib, keep, progress=progress).expr
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
