"""Nested (MIMO) GFT -- exact multi-loop quartet family (M-H + depth-3).

The scalar GFT dissects H against ONE designated error/injection. For a
circuit with several coupled loops (DM and CM of a fully-differential
amplifier; the three nested loops of a nested-Miller amplifier), the nested
dissection applies the scalar GFT once per loop, each on the system nulled by
the prior loops:

    level 1:  H       = Hinf1  * T1/(1+T1)   + H01/(1+T1)     (exact)
    level 2:  Hinf1   = Hinf12 * T2'/(1+T2') + H02'/(1+T2')   (exact)
    level 3:  Hinf12  = Hinf123* T3''/(1+T3'')+ H03''/(1+T3'')(exact)
    ...

where level k runs on the levels-1..k-1-NULLED system (each error null row
replaces its probe's branch equation), so T_k is loop k's gain with the
higher loops ideal. Every identity is exact for ANY designation -- nulls
compose in determinant algebra (docs/loopgain-plan.md Sec. 10.2) -- which is
precisely what the Tian convention cannot do (Sec. 10.1): here there is
no termination convention to betray the composition.

`NestedGft` is the two-level dissection (M-H); `DeepNestedGft` generalizes
it to arbitrary depth N.

Structure of the family:
  * the corner gain Hinf12 (both errors nulled) is ORDER-INVARIANT by
    construction (same doubly-constrained matrix);
  * the intermediate quantities are order-dependent: T2' (loop 2 under
    loop-1 null) vs the plain T2 (loop 2 with loop 1 merely closed)
    coincide exactly when the loops decouple, and their ratio is a
    coupling diagnostic with no error of its own -- both are exact.

Errors are designated PROBE-ALIGNED, as (ref, c) with

    u_y = v(ref) + c * v(probe p side),   u_x = v(ref) + c * v(q side)

-- the error must STRADDLE the injection (the M-F lesson, re-learned
here: a designation entirely on one side of the probe, like the
amp-input pair, makes u_x - u_y a frequency-dependent divider rather
than +-e_z, and no sign convention is exact). c = -1 is the follower
form of gft.py (null forces v(p) -> v(ref): the CM loop tracking
vcmref); c = +1 is the inverting form (null forces v(p) -> -v(ref):
the DM wire tracking -v(vin_dm) in the unity-inverting bench).

Cost model: quartets are evaluated NUMERICALLY on frequency grids
(lambdified A + row surgery + solves), while `identity_residual` runs
the oracle in EXACT rational arithmetic at fixed rational s points
(det of a QQ matrix is fast; the full symbolic dets of gft.py would
take minutes at fd size).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import sympy as sp

from ..engine.mna import MnaError, MnaSystem, S, _det, hybrid_split
from .gft import _probe_indices


def _exact_A(system: MnaSystem, scale=None) -> sp.Matrix:
    subs, _ = hybrid_split(system, [])
    if scale:
        for name, fac in scale.items():
            sym = system.symbols[name]
            if sym not in subs:
                raise MnaError(f"unknown symbol {name!r}")
            subs = {**subs, sym: subs[sym] * sp.Rational(str(fac))}
    return system.A.xreplace(subs)


def _node(system: MnaSystem, name: str) -> int:
    if name not in system.node_index:
        raise MnaError(f"unknown node {name!r}")
    return system.node_index[name]


def _null_row(A, ib, iref, ipn, c):
    """Base matrix with probe branch row ib replaced by the error null
    v(ref) + c*v(p) = 0 (the branch current stays as the free norator
    drive)."""
    Ai = A.copy()
    if isinstance(Ai, sp.MatrixBase):
        for j in range(Ai.cols):
            Ai[ib, j] = 0
        Ai[ib, iref] = 1
        Ai[ib, ipn] = Ai[ib, ipn] + c
    else:
        Ai[ib, :] = 0.0
        Ai[ib, iref] = 1.0
        Ai[ib, ipn] += c
    return Ai


# ---------------------------------------------------------------- exact
def _cramer_x(M: sp.Matrix, col: int, z) -> sp.Expr:
    Mk = M.copy()
    Mk[:, col] = z
    return _det(Mk)


def _point_quartet(A0, z_in, z_pr, io, probe, err):
    """All quartet scalars at one exact rational s point. A0 is the
    (possibly already level-1-constrained) base matrix at that point.
    Returns dict with H, Hinf, T, Tn, H0 as sympy Rationals/expr."""
    ip, iq, ib = probe
    iref, c = err
    D0 = _det(A0)
    H = _cramer_x(A0, io, z_in) / D0

    Ai = _null_row(A0, ib, iref, ip, c)
    Hinf = _cramer_x(Ai, io, z_in) / _det(Ai)

    cr = _cramer_x(A0, iref, z_pr)
    cp = _cramer_x(A0, ip, z_pr)
    cq = _cramer_x(A0, iq, z_pr)
    Ny = cr + c * cp
    Nx = cr + c * cq
    T = -Ny / Nx

    dim = A0.rows
    Ab = sp.zeros(dim + 1, dim + 1)
    Ab[:dim, :dim] = A0
    for i in range(dim):
        Ab[i, dim] = -z_in[i, 0]
    Ab[dim, io] = 1
    zb = sp.zeros(dim + 1, 1)
    for i in range(dim):
        zb[i, 0] = z_pr[i, 0]
    crb = _cramer_x(Ab, iref, zb)
    cpb = _cramer_x(Ab, ip, zb)
    cqb = _cramer_x(Ab, iq, zb)
    Nyb = crb + c * cpb
    Nxb = crb + c * cqb
    Tn = -Nyb / Nxb

    H0 = Hinf * T / Tn
    return {"H": H, "Hinf": Hinf, "T": T, "Tn": Tn, "H0": H0}


# --------------------------------------------------------------- numeric
def _num_quartet(M, z_in, z_pr, io, probe, err):
    """Same quartet at one complex frequency, by dense solves."""
    ip, iq, ib = probe
    iref, c = err
    x_in = np.linalg.solve(M, z_in)
    H = x_in[io]

    Mi = _null_row(M, ib, iref, ip, c)
    Hinf = np.linalg.solve(Mi, z_in)[io]

    x = np.linalg.solve(M, z_pr)
    Ny = x[iref] + c * x[ip]
    Nx = x[iref] + c * x[iq]
    T = -Ny / Nx

    dim = M.shape[0]
    Mb = np.zeros((dim + 1, dim + 1), complex)
    Mb[:dim, :dim] = M
    Mb[:dim, dim] = -z_in
    Mb[dim, io] = 1.0
    xb = np.linalg.solve(Mb, np.append(z_pr, 0.0))
    Nyb = xb[iref] + c * xb[ip]
    Nxb = xb[iref] + c * xb[iq]
    Tn = -Nyb / Nxb

    return {"H": H, "Hinf": Hinf, "T": T, "Tn": Tn,
            "H0": Hinf * T / Tn}


def _residual_of(q) -> sp.Expr:
    """H - [Hinf T/(1+T) + H0/(1+T)], exact."""
    return sp.simplify(q["H"] - sp.together(
        q["Hinf"] * q["T"] / (1 + q["T"]) + q["H0"] / (1 + q["T"])))


@dataclass
class NestedGft:
    """The two-level dissection. Level 1: H against (probe1, error1);
    level 2: Hinf1 against (probe2, error2) on the level-1-nulled
    system. All grid evaluations numeric; oracles exact."""
    probes: tuple
    errors: tuple
    out: str
    _sys_in: MnaSystem = field(repr=False)
    _A: sp.Matrix = field(repr=False)
    _z_in: sp.Matrix = field(repr=False)
    _z1: sp.Matrix = field(repr=False)
    _z2: sp.Matrix = field(repr=False)
    _idx: dict = field(repr=False)

    # ---- numeric grid evaluation
    def _grids(self, freqs):
        fn = sp.lambdify(S, self._A, "numpy")
        z_in = np.asarray(self._z_in, dtype=complex).ravel()
        z1 = np.asarray(self._z1, dtype=complex).ravel()
        z2 = np.asarray(self._z2, dtype=complex).ravel()
        io = self._idx["io"]
        p1, e1 = self._idx["p1"], self._idx["e1"]
        p2, e2 = self._idx["p2"], self._idx["e2"]
        lv1, lv2, plain2 = [], [], []
        # the plain-loop-2 quartet's Tn can pass through infinity at isolated
        # frequencies (benign; the diagnostics never read it) -- stay quiet
        with np.errstate(divide="ignore", invalid="ignore"):
            for f in np.atleast_1d(freqs):
                M = np.asarray(fn(2j * np.pi * f), complex)
                lv1.append(_num_quartet(M, z_in, z1, io, p1, e1))
                Mi = _null_row(M, p1[2], e1[0], p1[0], e1[1])
                lv2.append(_num_quartet(Mi, z_in, z2, io, p2, e2))
                plain2.append(_num_quartet(M, z_in, z2, io, p2, e2))
        pack = lambda qs: {k: np.array([q[k] for q in qs])
                           for k in qs[0]}
        return pack(lv1), pack(lv2), pack(plain2)

    def level1(self, freqs):
        return self._grids(freqs)[0]

    def level2(self, freqs):
        """Quartet of Hinf1: its 'H' is Hinf1, its 'Hinf' is the corner
        Hinf12, its 'T' is T2' (loop 2 under the loop-1 null)."""
        return self._grids(freqs)[1]

    def plain2(self, freqs):
        """Loop-2 quartet on the FULL system (loop 1 closed, not
        nulled): its 'T' is the plain T2."""
        return self._grids(freqs)[2]

    def coupling(self, freqs):
        """|T2'/T2 - 1|: how much loop 2 changes when loop 1 is
        idealized -- zero iff the loops decouple. Both T's are exact
        objects; this diagnostic has no approximation error."""
        g = self._grids(freqs)
        return np.abs(g[1]["T"] / g[2]["T"] - 1)

    # ---- exact oracle
    def identity_residual(self, level: int = 1,
                          s_points=(2, 3, 7)) -> float:
        """Worst |residual/H| of the GFT identity at exact rational s
        points; 0.0 means the dissection is exact."""
        io = self._idx["io"]
        worst = 0.0
        for sv in s_points:
            sv = sp.Rational(sv)
            A0 = self._A.xreplace({S: sv})
            if level == 1:
                q = _point_quartet(A0, self._z_in, self._z1, io,
                                   self._idx["p1"], self._idx["e1"])
            else:
                Ai = _null_row(A0, self._idx["p1"][2],
                               self._idx["e1"][0], self._idx["p1"][0],
                               self._idx["e1"][1])
                q = _point_quartet(Ai, self._z_in, self._z2, io,
                                   self._idx["p2"], self._idx["e2"])
            r = _residual_of(q)
            if r != 0:
                worst = max(worst, abs(float((r / q["H"]).evalf())))
        return worst


def nested_gft(system_in: MnaSystem, system_p1: MnaSystem,
               system_p2: MnaSystem, probe1: str, probe2: str,
               out: str, error1, error2, *, scale=None) -> NestedGft:
    """Build the nested dissection. error1/error2 are (ref_node, c)
    with u_y = v(ref) + c*v(probe p side), c = -1 follower / +1
    inverting; scale is the exact-rational mismatch knob
    ({"gm_...": factor})."""
    p1 = _probe_indices(system_p1, probe1)
    p2 = _probe_indices(system_p1, probe2)
    io = _node(system_p1, out)
    e1 = (_node(system_p1, error1[0]), int(error1[1]))
    e2 = (_node(system_p1, error2[0]), int(error2[1]))
    A = _exact_A(system_p1, scale)
    return NestedGft(
        probes=(probe1, probe2), errors=(error1, error2), out=out,
        _sys_in=system_in, _A=A, _z_in=system_in.z, _z1=system_p1.z,
        _z2=system_p2.z,
        _idx={"io": io, "p1": p1, "p2": p2, "e1": e1, "e2": e2})


# ------------------------------------------------- N-level generalization


@dataclass
class DeepNestedGft:
    """N-level nested GFT dissection (arbitrary depth; two-level NestedGft is
    the N=2 case). Level k dissects the running ideal gain Hinf_{1..k-1}
    against (probe_k, error_k) on the system nulled by errors 1..k-1:

        H       = Hinf_1     T1/(1+T1)     + H0_1/(1+T1)        (level 1)
        Hinf_1  = Hinf_12    T2'/(1+T2')   + H0_2'/(1+T2')      (level 2)
        Hinf_12 = Hinf_123   T3''/(1+T3'') + H0_3''/(1+T3'')    (level 3)
        ...

    Each identity is exact for ANY designation -- nulls compose in
    determinant algebra (docs/loopgain-plan.md Sec. 10.2), so there is no
    termination convention to betray the composition (contrast Sec. 10.1's
    MIMO-Tian obstruction). The corner gain Hinf_{1..N} (all N errors nulled)
    is order-invariant by construction; the intermediate T_k are
    order-dependent and their ratio to the plain T_k is a coupling
    diagnostic with no error of its own. All grid evaluations are numeric;
    the identity oracle runs in exact rational arithmetic at rational s
    points. Errors are probe-aligned (ref, c) exactly as for NestedGft."""
    probes: tuple
    errors: tuple
    out: str
    _sys_in: MnaSystem = field(repr=False)
    _A: sp.Matrix = field(repr=False)
    _z_in: sp.Matrix = field(repr=False)
    _zs: list = field(repr=False)          # per-level probe-injection RHS
    _idx: dict = field(repr=False)         # io; P, E lists of per-level tuples

    @property
    def depth(self) -> int:
        return len(self.probes)

    def _grids(self, freqs):
        fn = sp.lambdify(S, self._A, "numpy")
        z_in = np.asarray(self._z_in, dtype=complex).ravel()
        zs = [np.asarray(z, dtype=complex).ravel() for z in self._zs]
        io = self._idx["io"]
        P, E = self._idx["P"], self._idx["E"]
        N = len(P)
        nested = [[] for _ in range(N)]
        plain = [[] for _ in range(N)]
        # a plain quartet's Tn can pass through infinity at isolated
        # frequencies (its null denominator crosses zero) -- expected, and the
        # nested T / coupling diagnostics never read it; keep the sweep quiet
        with np.errstate(divide="ignore", invalid="ignore"):
            for f in np.atleast_1d(freqs):
                M = np.asarray(fn(2j * np.pi * f), complex)
                Mcur = M
                for k in range(N):
                    nested[k].append(_num_quartet(Mcur, z_in, zs[k], io,
                                                  P[k], E[k]))
                    plain[k].append(_num_quartet(M, z_in, zs[k], io,
                                                 P[k], E[k]))
                    Mcur = _null_row(Mcur, P[k][2], E[k][0], P[k][0], E[k][1])
        pack = lambda qs: {k: np.array([q[k] for q in qs]) for k in qs[0]}
        return [pack(x) for x in nested], [pack(x) for x in plain]

    def level(self, k: int, freqs):
        """Quartet at nesting level k (1-based): its 'H' is Hinf_{1..k-1},
        'Hinf' the corner Hinf_{1..k}, 'T' loop-k's gain under the higher
        nulls (T_k with loops 1..k-1 ideal)."""
        return self._grids(freqs)[0][k - 1]

    def plain(self, k: int, freqs):
        """Loop-k quartet on the FULLY CLOSED system (no nulls): its 'T' is
        the plain T_k."""
        return self._grids(freqs)[1][k - 1]

    def coupling(self, k: int, freqs):
        """|T_k(nested)/T_k(plain) - 1|: how much loop k changes under the
        higher-level nulls; zero iff loop k decouples from loops 1..k-1.
        Both T's are exact objects, so the diagnostic has no error of its
        own."""
        g = self._grids(freqs)
        return np.abs(g[0][k - 1]["T"] / g[1][k - 1]["T"] - 1)

    def identity_residual(self, level: int = 1, s_points=(2, 3, 7)) -> float:
        """Worst |residual/H| of the level-`level` GFT identity at exact
        rational s points (the higher nulls 1..level-1 applied exactly);
        0.0 means that level's dissection is exact."""
        io = self._idx["io"]
        P, E = self._idx["P"], self._idx["E"]
        worst = 0.0
        for sv in s_points:
            sv = sp.Rational(sv)
            A0 = self._A.xreplace({S: sv})
            for j in range(level - 1):
                A0 = _null_row(A0, P[j][2], E[j][0], P[j][0], E[j][1])
            q = _point_quartet(A0, self._z_in, self._zs[level - 1], io,
                               P[level - 1], E[level - 1])
            r = _residual_of(q)
            if r != 0:
                worst = max(worst, abs(float((r / q["H"]).evalf())))
        return worst


def nested_gft_deep(system_in: MnaSystem, systems_p: list, probes,
                    out: str, errors, *, scale=None) -> DeepNestedGft:
    """Build an N-level nested dissection. `systems_p[k]` is the circuit built
    with input_name=probes[k] (for the level-k series-voltage injection RHS);
    `errors[k]` is (ref_node, c) probe-aligned as for nested_gft. `scale` is
    the exact-rational mismatch knob ({"gm_...": factor})."""
    if not (len(systems_p) == len(probes) == len(errors)):
        raise MnaError("probes, systems_p and errors must have equal length")
    if len(probes) < 1:
        raise MnaError("need at least one nesting level")
    ref = systems_p[0]
    P = [_probe_indices(ref, pr) for pr in probes]
    io = _node(ref, out)
    E = [(_node(ref, e[0]), int(e[1])) for e in errors]
    A = _exact_A(ref, scale)
    return DeepNestedGft(
        probes=tuple(probes), errors=tuple(errors), out=out,
        _sys_in=system_in, _A=A, _z_in=system_in.z,
        _zs=[s.z for s in systems_p], _idx={"io": io, "P": P, "E": E})
