"""The General Feedback Theorem quartet (Middlebrook), exact on the MNA.

Dissects a closed-loop transfer H into second-level transfer functions

    H = Hinf * T/(1+T) + H0/(1+T) = Hinf * (1 + 1/Tn)/(1 + 1/T),
    H0/Hinf = T/Tn                                     (redundancy)

with Hinf the ideal gain (loop gain -> infinity), H0 the direct forward
transmission (loop gain -> 0), Tn the null loop gain, and D = T/(1+T) the
discrepancy factor. Reference: R. D. Middlebrook, GFT Template manual
(Intusoft); derived from the EET via null double injection.

The implementation uses ONE consistent test-signal configuration -- the
single voltage injection at the designated probe (the manual's GFTve,
appropriate for follower-type/direct feedback) with a DESIGNATED ERROR
signal u_y = v(error_ref) - v(probe p side):

    Hinf = H under the error null (replace the probe branch equation by
           u_y = 0; the branch current becomes the free e_z drive);
    T    = u_y/u_x with the input zeroed and unit e_z (u_x references the
           probe q side) -- the voltage return ratio of THIS configuration
           (not the Tian dual-injection T of loopgain.py, which belongs to
           the dual configuration and reports stability margins);
    Tn   = the same ratio with the OUTPUT nulled by a free input amplitude
           (a bordered system);
    H0   = Hinf * T/Tn by the redundancy relation.

Everything is an exact rational in s, so identity (A.2) is verified in
EXACT rational arithmetic at probe points -- a mandatory self-check: a
wrong sign or a mis-designated error cannot survive it.

The hard-won lessons encoded here (see docs/loopgain-plan.md Sec. 7): the
null must be the designated error signal, not the probe node to ground;
the injection configuration defines which Hinf you get; and reusing the
dual-injection Tian combination inside the output-nulled bordered system
does NOT give Tn.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np
import sympy as sp

from ..engine.mna import (MnaError, MnaSystem, S, TransferFunction, _det,
                          hybrid_split, solve_tf)
from ..keep import is_all

# skip the final rational cancel() when the combined expression is huge --
# the nested rational is exact either way (same guard as loopgain.py)
_CANCEL_OPS_LIMIT = 20000


def _probe_indices(system: MnaSystem, probe: str):
    if probe not in system.branch_index:
        raise MnaError(f"probe {probe!r} is not a voltage-defined branch")
    ib = system.branch_index[probe]
    n_nodes = len(system.node_index)
    ip = iq = None
    for i in range(n_nodes):
        e = system.A[i, ib]
        if e == 1:
            ip = i
        elif e == -1:
            iq = i
    if ip is None or iq is None:
        raise MnaError(f"probe {probe!r}: could not identify both nodes")
    return ip, iq, ib


@dataclass
class GftQuartet:
    """Exact GFT dissection. All members are TransferFunctions in s."""
    H: TransferFunction
    Hinf: TransferFunction
    T: TransferFunction
    Tn: TransferFunction
    H0: TransferFunction
    error_ref: str
    probe: str

    def discrepancy(self, freqs) -> np.ndarray:
        """D = T/(1+T) evaluated on freqs."""
        t = self.T.numeric(freqs)
        return t / (1 + t)

    def spec_deviation(self, freqs) -> np.ndarray:
        """|H/Hinf - 1| -- the deviation from the ideal gain, Middlebrook's
        spec-level quantity (the D-peaking requirement lives here)."""
        return np.abs(self.H.numeric(freqs) / self.Hinf.numeric(freqs) - 1)

    def feedthrough_crossover(self, freqs) -> float | None:
        """The frequency beyond which |H0/(1+T)| exceeds |Hinf*T/(1+T)| --
        past it the closed-loop response is direct-transmission dominated
        and loop-gain shaping cannot change H. None if never in band."""
        f = np.asarray(freqs, dtype=float)
        t = self.T.numeric(f)
        a = np.abs(self.Hinf.numeric(f) * t / (1 + t))
        b = np.abs(self.H0.numeric(f) / (1 + t))
        k = np.where(b > a)[0]
        return float(f[k[0]]) if k.size else None

    def identity_residual(self, s_points=(2, 3, 7)) -> float:
        """Verify H = Hinf*T/(1+T) + H0/(1+T) in EXACT rational arithmetic
        at rational s points; returns the worst |lhs/rhs - 1| as float
        (0.0 means exact). A configuration/sign error cannot pass this.

        For a hybrid quartet (kept symbols still present) the kept symbols
        are collapsed to their exact OP values first, so the check stays a
        fast scalar residual -- the GFT identity is an algebraic identity in
        the matrix entries, hence holds for any kept designation, and this
        pins it numerically at the OP just as the keep=[] case does."""
        num = {self.H.symbols[n]: sp.Rational(repr(v))
               for n, v in self.H.values.items() if n in self.H.symbols}
        worst = 0.0
        for sv in s_points:
            sv = {S: sp.Rational(sv), **num}
            H = self.H.expr.xreplace(sv)
            Hi = self.Hinf.expr.xreplace(sv)
            T = self.T.expr.xreplace(sv)
            H0 = self.H0.expr.xreplace(sv)
            rhs = sp.together(Hi * T / (1 + T) + H0 / (1 + T))
            r = sp.simplify(H - rhs)
            if r != 0:
                worst = max(worst, abs(float((r / H).evalf())))
        return worst


def gft(system_in: MnaSystem, system_probe: MnaSystem, probe: str,
        out: str, error_ref: str, keep=()) -> GftQuartet:
    """Compute the GFT quartet.

    system_in:    built with input_name = the input source (z = input);
    system_probe: the same circuit built with input_name = probe (z = the
                  unit series-voltage injection on the probe branch);
    out:          output net; error_ref: the node whose voltage referenced
                  against the probe's p side defines the error signal
                  u_y = v(error_ref) - v(p) (K-side error).

    keep follows the tf()/loop_gain() convention: [] (default; None is its
    alias) is fully numeric (the fast shared-determinant Cramer path);
    [names] keeps those symbols symbolic; ALL is fully symbolic. In the
    hybrid case every quartet member routes through the SAME multilinear
    solver tf() uses -- the null-row (Hinf) and output-bordered (Tn)
    surgeries are applied to the raw symbolic matrix and solved column by
    column, so the dissection stays symbolic in the kept parameters (the
    compensation-inversion story of the paper generalizes H(s) -> the whole
    quartet). The GFT identity, an algebraic identity in the matrix entries,
    holds for any kept designation and is still self-checked."""
    ip, iq, ib = _probe_indices(system_probe, probe)
    if out not in system_probe.node_index:
        raise MnaError(f"output node {out!r} not found")
    io = system_probe.node_index[out]
    if error_ref not in system_probe.node_index:
        raise MnaError(f"error_ref node {error_ref!r} not found")
    ie = system_probe.node_index[error_ref]

    vals = dict(system_probe.values)
    syms = dict(system_probe.symbols)
    z_in = system_in.z
    z_pr = system_probe.z
    dim = system_probe.A.rows

    if not is_all(keep):
        keep = [] if keep is None else list(keep)
    if not (is_all(keep) or keep):
        return _gft_numeric(system_probe, probe, ip, iq, ib, io, ie,
                            z_in, z_pr, dim, vals, syms, error_ref)

    return _gft_hybrid(system_probe, probe, ip, iq, ib, io, ie,
                       z_in, z_pr, dim, vals, syms, error_ref, keep)


def _gft_numeric(system_probe, probe, ip, iq, ib, io, ie,
                 z_in, z_pr, dim, vals, syms, error_ref) -> GftQuartet:
    """Fully numeric OP (keep=[]): one shared determinant, Cramer reads.
    Validated against the stb fixture; the identity is exact-rational."""
    subs, _ = hybrid_split(system_probe, [])
    A = system_probe.A.xreplace(subs)

    def cramer(M, col, z):
        Mk = M.copy()
        Mk[:, col] = z
        return _det(Mk)

    def tf(num, den):
        return TransferFunction(expr=num / den, values=vals, symbols=syms)

    # ---- H: plain closed-loop transfer
    D0 = _det(A)
    H = tf(cramer(A, io, z_in), D0)

    # ---- Hinf: error null replaces the probe branch equation
    Ai = A.copy()
    for j in range(dim):
        Ai[ib, j] = 0
    Ai[ib, ie] = 1
    Ai[ib, ip] = -1                      # u_y = v(ref) - v(p) = 0
    Di = _det(Ai)
    Hinf = tf(cramer(Ai, io, z_in), Di)

    # ---- T: u_y/u_x under unit e_z, input zero (this configuration's own
    # voltage return ratio; u_x = v(ref) - v(q))
    # sign per the manual (T = -u_y/u_x: the returned signal opposes the
    # injection) -- locked by the EXACT rational identity check, which is
    # zero only for this convention
    Ny = cramer(A, ie, z_pr) - cramer(A, ip, z_pr)
    Nx = cramer(A, ie, z_pr) - cramer(A, iq, z_pr)
    T = tf(-Ny, Nx)

    # ---- Tn: same ratio with the output nulled by a free input amplitude
    # (bordered system: extra column -z_in, extra row v(out) = 0)
    Ab = sp.zeros(dim + 1, dim + 1)
    Ab[:dim, :dim] = A
    for i in range(dim):
        Ab[i, dim] = -z_in[i, 0]
    Ab[dim, io] = 1
    zb = sp.zeros(dim + 1, 1)
    for i in range(dim):
        zb[i, 0] = z_pr[i, 0]
    Nyb = cramer(Ab, ie, zb) - cramer(Ab, ip, zb)
    Nxb = cramer(Ab, ie, zb) - cramer(Ab, iq, zb)
    Tn = tf(-Nyb, Nxb)

    # ---- H0 by the redundancy relation
    H0 = tf(sp.expand(Hinf.expr.as_numer_denom()[0]
                      * T.expr.as_numer_denom()[0]
                      * Tn.expr.as_numer_denom()[1]),
            sp.expand(Hinf.expr.as_numer_denom()[1]
                      * T.expr.as_numer_denom()[1]
                      * Tn.expr.as_numer_denom()[0]))

    return GftQuartet(H=H, Hinf=Hinf, T=T, Tn=Tn, H0=H0,
                      error_ref=error_ref, probe=probe)


def _gft_hybrid(system_probe, probe, ip, iq, ib, io, ie,
                z_in, z_pr, dim, vals, syms, error_ref, keep) -> GftQuartet:
    """Hybrid / fully-symbolic quartet: each member is a column solve on the
    raw symbolic matrix (with the null-row or output-border applied), routed
    through the same multilinear solver tf() uses, so the kept symbols
    survive. The T/Tn ratios cancel their shared determinant exactly, so no
    Cramer determinant is ever needed on its own."""
    A = system_probe.A                                  # raw, symbolic

    def col(system, k):
        return solve_tf(system, k, keep).expr           # cramer(k)/det

    def finish(expr):
        if sp.count_ops(expr) <= _CANCEL_OPS_LIMIT:
            expr = sp.cancel(sp.together(expr))
        return TransferFunction(expr=expr, values=vals, symbols=syms)

    # ---- H: plain closed-loop transfer (input RHS on the untouched matrix)
    sys_in = dataclasses.replace(system_probe, z=z_in)
    H = finish(col(sys_in, io))

    # ---- Hinf: error null v(ref) - v(p) = 0 replaces the probe branch row
    Ai = A.copy()
    for j in range(dim):
        Ai[ib, j] = 0
    Ai[ib, ie] = 1
    Ai[ib, ip] = -1
    sys_hinf = dataclasses.replace(system_probe, A=Ai, z=z_in)
    Hinf = finish(col(sys_hinf, io))

    # ---- T = -(v_ie - v_ip)/(v_ie - v_iq) under the probe injection; the
    # three solves share A's determinant, which cancels in the ratio
    vie = col(system_probe, ie)
    vip = col(system_probe, ip)
    viq = col(system_probe, iq)
    T = finish(-(vie - vip) / (vie - viq))

    # ---- Tn: the same ratio on the output-bordered system (extra column
    # -z_in, extra row v(out) = 0), input RHS z_pr padded with a 0
    Ab = sp.zeros(dim + 1, dim + 1)
    Ab[:dim, :dim] = A
    for i in range(dim):
        Ab[i, dim] = -z_in[i, 0]
    Ab[dim, io] = 1
    zb = sp.zeros(dim + 1, 1)
    for i in range(dim):
        zb[i, 0] = z_pr[i, 0]
    sys_tn = dataclasses.replace(system_probe, A=Ab, z=zb)
    vieb = col(sys_tn, ie)
    vipb = col(sys_tn, ip)
    viqb = col(sys_tn, iq)
    Tn = finish(-(vieb - vipb) / (vieb - viqb))

    # ---- H0 by the redundancy relation H0 = Hinf * T / Tn
    H0 = finish(Hinf.expr * T.expr / Tn.expr)

    return GftQuartet(H=H, Hinf=Hinf, T=T, Tn=Tn, H0=H0,
                      error_ref=error_ref, probe=probe)
