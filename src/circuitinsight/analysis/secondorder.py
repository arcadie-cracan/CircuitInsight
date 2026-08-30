"""Level-2 sensitivity: the second derivatives that read STRUCTURE.

First-order sensitivities say how much each element moves a
singularity; the second derivatives say how the elements are WIRED
TOGETHER in the expression that places it. For a low-entropy formula
f0 = prod_k S_k**a_k the normalized coupling

    rho_ij = H_ij / (g_i g_j),   g = dln f0/dln x, H = d2ln f0/dln x dln x

is QUANTIZED: rho = -1/a_k for two elements inside the same plain sum
S_k (so -1 in a numerator sum, +1 in a denominator sum, -+2 under a
square root), rho = 0 across factors and for a pure monomial, and
systematic non-quantized values for nested forms. Measured on the
folded cascode and the two-stage across two operating points and two
independent methods, that prediction held for every pair tried.

This module supplies the primitive the whole construction rests on.
Everything comes from ONE factorization: with the MNA matrix carrying
rank-one stamps A = A0 + sum_m Y_m(theta) a_m b_m^T, Jacobi's formula
gives

    dln det/dtheta_i     = sum_{m in i} Y'_m (b_m^T M a_m)
    d2ln det/dth_i dth_j = delta_ij sum_{m in i} Y''_m (b_m^T M a_m)
                           - sum_{m in i} sum_{n in j}
                             Y'_m Y'_n (b_m^T M a_n)(b_n^T M a_m)

with M = A^-1. Every term is an entry of the single K x K matrix
P = B^T M A_ (K the stamp count), so the FULL interaction matrix costs
one dim x dim x K product and one K x dim x K product -- no new solves,
no finite differences, and no step size to trade off.

The double sum over stamps is also exactly why a symbol SHARED by
matched devices behaves differently from a lone one: it carries one
stamp per member, so it enters the determinant with degree equal to
its stamp count rather than affinely.
"""
from __future__ import annotations

import numpy as np


def _dY(kind: str, s: complex, val: float) -> complex:
    """dY/dtheta of one stamp -- the same table the first-order root
    sensitivities use (marginledger._kfac), restated here so this
    module stands alone."""
    if kind in ("c", "cx"):
        return s
    if kind in ("g", "vccs"):
        return 1.0
    if kind == "r":
        return -1.0 / (val * val)
    if kind == "l":
        return -s
    return -1.0                                    # vcvs


def _d2Y(kind: str, s: complex, val: float) -> complex:
    """d2Y/dtheta2. Zero wherever the stamp is LINEAR in its parameter
    (a conductance, a capacitance, an inductance, a controlled
    source); nonzero only where the parameter is the reciprocal of the
    admittance, i.e. a resistance."""
    if kind == "r":
        return 2.0 / (val ** 3)
    return 0.0


def logdet_jet(M: np.ndarray, stamps: dict, vals: dict, s: complex):
    """(g, H) of ln det A(s) in LOG-LOG coordinates, keyed by symbol.

    `M` is A(s)^-1, `stamps` maps a symbol to its rank-one patterns as
    circuitinsight.analysis.marginledger._collect_stamps returns them
    ((kind, a, b, inst) tuples; a trailing instance label is ignored),
    and `vals` maps a symbol to its value. Returns

      g[name]        = dln det / dln theta
      H[(n1, n2)]    = d2ln det / dln theta_1 dln theta_2   (both orders
                       stored, so H[a, b] and H[b, a] both resolve)

    The log-log lift adds the first-order term on the diagonal:
    d2/dln x dln y = x y d2/dx dy + delta_xy x d/dx.
    """
    names = [n for n in stamps if n in vals]
    flat = []                                      # (name, k1, k2, a, b)
    for name in names:
        for st in stamps[name]:
            kind, a, b = st[0], st[1], st[2]
            flat.append((name, _dY(kind, s, vals[name]),
                         _d2Y(kind, s, vals[name]),
                         np.asarray(a, complex), np.asarray(b, complex)))
    if not flat:
        return {}, {}
    K = len(flat)
    Am = np.empty((M.shape[0], K), complex)
    Bm = np.empty((M.shape[0], K), complex)
    for m, (_n, _k1, _k2, a, b) in enumerate(flat):
        Am[:, m] = a
        Bm[:, m] = b
    # P[m, n] = b_m^T M a_n -- every term of both formulas lives here
    P = Bm.T @ (M @ Am)
    idx: dict = {}
    for m, (name, _k1, _k2, _a, _b) in enumerate(flat):
        idx.setdefault(name, []).append(m)

    g: dict = {}
    for name, ms in idx.items():
        L1 = sum(flat[m][1] * P[m, m] for m in ms)
        g[name] = complex(vals[name] * L1)

    H: dict = {}
    for i, name_i in enumerate(names):
        for name_j in names[i:]:
            L2 = -sum(flat[m][1] * flat[n][1] * P[m, n] * P[n, m]
                      for m in idx[name_i] for n in idx[name_j])
            if name_i == name_j:
                L2 += sum(flat[m][2] * P[m, m] for m in idx[name_i])
            v = complex(vals[name_i] * vals[name_j] * L2)
            if name_i == name_j:
                v += g[name_i]
            H[name_i, name_j] = v
            H[name_j, name_i] = v
    return g, H


def coupling_ratios(g: dict, H: dict, *, floor: float = 4e-3) -> dict:
    """rho_ij = H_ij/(g_i g_j) for every pair whose |g_i g_j| clears
    `floor` -- below it the ratio is noise over a vanishing
    denominator and the pair simply has no reading. Real parts only:
    the structural claim is about the real log-log spectrum, and an
    imaginary part means the reading was taken off the real axis."""
    out: dict = {}
    names = sorted(g)
    for i, a in enumerate(names):
        for b in names[i:]:
            if a == b:
                continue
            gg = (g[a] * g[b]).real
            if abs(gg) <= floor:
                continue
            out[a, b] = float(H[a, b].real / gg)
    return out


class _Bilinear:
    """Derivatives of R = u^T M w, where M = A(s)^-1 and A carries
    rank-one stamps. Everything is an inner product against the
    precomputed stamp columns -- no solve per parameter, and none of
    the finite differencing the measurement scripts had to use.

    Held to second order in the elements AND in s, because lifting a
    determinant identity to a ROOT needs the mixed and s terms too."""

    def __init__(self, ctx, u, w):
        M, Q = ctx.M, ctx.Q
        self.ctx = ctx
        Mu = M.T @ u
        Mw = M @ w
        self.R = complex(Mu @ w)
        self.al = ctx.Am.T @ Mu                    # alpha[m] = u^T M a_m
        self.be = ctx.Bm.T @ Mw                    # beta[m]  = b_m^T M w
        qu = Q.T @ Mu
        qw = Q @ Mw
        self.R_s = complex(-(Mu @ qw))
        self.al_s = -(ctx.Ua.T @ qu)               # d alpha / ds
        self.be_s = -(ctx.Wb.T @ qw)               # d beta  / ds
        self.R_ss = complex(2.0 * (Mu @ (Q @ (M @ qw))))

    def d_th(self, name):
        c = self.ctx
        return -sum(c.k1[m] * self.al[m] * self.be[m]
                    for m in c.idx[name])

    def d_th_s(self, name):
        c = self.ctx
        return -sum(c.k1s[m] * self.al[m] * self.be[m]
                    + c.k1[m] * self.al_s[m] * self.be[m]
                    + c.k1[m] * self.al[m] * self.be_s[m]
                    for m in c.idx[name])

    def d_th_th(self, n1, n2):
        c, P = self.ctx, self.ctx.P
        out = 0j
        for m in c.idx[n1]:
            for n in c.idx[n2]:
                out += c.k1[m] * c.k1[n] * (
                    self.al[n] * P[n, m] * self.be[m]
                    + self.al[m] * P[m, n] * self.be[n])
        if n1 == n2:
            out -= sum(c.k2[m] * self.al[m] * self.be[m]
                       for m in c.idx[n1])
        return out


class _Ctx:
    """Stamp columns and the one inverse everything is read from."""

    def __init__(self, A, Q, stamps, vals, s):
        self.M = np.linalg.inv(A)
        self.Q = np.asarray(Q, complex)
        self.names = [n for n in stamps if n in vals]
        flat, self.idx = [], {}
        for name in self.names:
            for st in stamps[name]:
                kind, a, b = st[0], st[1], st[2]
                self.idx.setdefault(name, []).append(len(flat))
                flat.append((kind, a, b, vals[name]))
        K = len(flat)
        dim = self.M.shape[0]
        self.Am = np.empty((dim, K), complex)
        self.Bm = np.empty((dim, K), complex)
        self.k1 = np.empty(K, complex)
        self.k2 = np.empty(K, complex)
        self.k1s = np.zeros(K, complex)            # d(dY/dtheta)/ds
        for m, (kind, a, b, val) in enumerate(flat):
            self.Am[:, m] = a
            self.Bm[:, m] = b
            self.k1[m] = _dY(kind, s, val)
            self.k2[m] = _d2Y(kind, s, val)
            if kind in ("c", "cx"):
                self.k1s[m] = 1.0
            elif kind == "l":
                self.k1s[m] = -1.0
        self.Ua = self.M @ self.Am                 # columns M a_m
        self.Wb = self.M.T @ self.Bm               # columns M^T b_m
        self.P = self.Bm.T @ self.Ua               # P[m,n] = b_m^T M a_n


def _root_derivs(fn, Q, ib, ip, stamps, vals, s0):
    """(names, ds/dtheta, d2s/dtheta2) of a Tian loop-gain pole.

    A pole of T sits where the Eq. 30 denominator
    Dn = 2(B C - A D) + A - D + 1 vanishes, B, D, A, C being the four
    probe reads. Each is a bilinear form in M = A(s)^-1, so Dn and all
    of its first and second derivatives in the elements AND in s come
    from one inverse plus inner products. The root then follows by
    implicit differentiation of Dn(theta, s) = 0:

        s_i  = -F_i / F_s
        s_ij = -(F_ij + F_si s_j + F_sj s_i + F_ss s_i s_j) / F_s

    and the log-log jet of f0 = |s|/2pi is Re(s_i/s) and
    Re(s_ij/s - s_i s_j/s^2), scaled by the values.

    Unlike the first-order path in marginledger, this evaluates AT the
    root: Dn is finite and vanishing there, where T itself diverges,
    so the deliberate 1e-6 off-root offset -- and the accuracy it
    costs -- is not needed."""
    A = np.asarray(fn(s0), complex)
    ctx = _Ctx(A, Q, stamps, vals, s0)
    dim = A.shape[0]
    e_b = np.zeros(dim, complex)
    e_b[ib] = 1.0
    e_p = np.zeros(dim, complex)
    e_p[ip] = 1.0
    # B = -e_b^T M z_v, D = e_p^T M z_v, A = -e_b^T M z_i, C = e_p^T M z_i
    # with z_v = e_b (series-voltage injection) and z_i = e_p
    RB = _Bilinear(ctx, e_b, e_b)
    RD = _Bilinear(ctx, e_p, e_b)
    RA = _Bilinear(ctx, e_b, e_p)
    RC = _Bilinear(ctx, e_p, e_p)
    names = ctx.names

    def parts(get):
        """(B, D, A, C) of one derivative order, signs applied."""
        return -get(RB), get(RD), -get(RA), get(RC)

    B, D, A_, C = -RB.R, RD.R, -RA.R, RC.R

    def dn(b, d, a, c, b1=None, d1=None, a1=None, c1=None,
           b2=None, d2=None, a2=None, c2=None):
        """Dn or one of its derivatives by the product rule."""
        if b1 is None:                              # first derivative
            return 2 * (b * C + B * c - a * D - A_ * d) + a - d
        # second derivative: b,d,a,c are the MIXED terms, (b1..) and
        # (b2..) the two first-derivative sets
        return (2 * (b * C + b1 * c2 + b2 * c1 + B * c
                     - a * D - a1 * d2 - a2 * d1 - A_ * d)
                + a - d)

    F_s = dn(*parts(lambda R: R.R_s))
    F_ss = dn(*parts(lambda R: R.R_ss),
              *parts(lambda R: R.R_s), *parts(lambda R: R.R_s))
    F_i, F_si, s_i = {}, {}, {}
    for n in names:
        F_i[n] = dn(*parts(lambda R, n=n: R.d_th(n)))
        F_si[n] = dn(*parts(lambda R, n=n: R.d_th_s(n)),
                     *parts(lambda R, n=n: R.d_th(n)),
                     *parts(lambda R: R.R_s))
        s_i[n] = -F_i[n] / F_s

    s_ij = {}
    for i, n1 in enumerate(names):
        for n2 in names[i:]:
            F_ij = dn(*parts(lambda R, a=n1, b=n2: R.d_th_th(a, b)),
                      *parts(lambda R, a=n1: R.d_th(a)),
                      *parts(lambda R, b=n2: R.d_th(b)))
            v = -(F_ij + F_si[n1] * s_i[n2] + F_si[n2] * s_i[n1]
                  + F_ss * s_i[n1] * s_i[n2]) / F_s
            s_ij[n1, n2] = s_ij[n2, n1] = v
    return names, s_i, s_ij


def _lift(names, s_i, s_ij, vals, s0, what: str):
    """Log-log jet of a scalar read off the root: its FREQUENCY
    omega0 = |s|, or the DAMPING zeta = -Re(s)/|s| of the conjugate
    pair it belongs to.

        d ln|s|      = Re(s_x / s)
        d2 ln|s|     = Re(s_xy/s - s_x s_y/s^2)
        d ln(-Re s)  = Re(s_x)/Re(s)
        d2 ln(-Re s) = Re(s_xy)/Re(s) - Re(s_x) Re(s_y)/Re(s)^2

    and ln zeta is the difference of the two. A conjugate pair needs
    no second root for either: s2 = conj(s1) makes omega0 and zeta
    both functions of s1 alone, which is also what keeps them well
    conditioned where the individual roots of a near-coalescent pair
    are not."""
    sr = s0.real
    if what == "zeta" and (sr == 0.0 or s0.imag == 0.0):
        return {}, {}

    def d1(x):
        if what == "omega":
            return (x / s0).real
        return (x.real / sr) - (x / s0).real

    def d2(x, xi, xj):
        w = (x / s0 - xi * xj / s0 ** 2).real
        if what == "omega":
            return w
        return (x.real / sr - xi.real * xj.real / sr ** 2) - w

    g = {n: complex(vals[n] * d1(s_i[n])) for n in names}
    H = {}
    for i, n1 in enumerate(names):
        for n2 in names[i:]:
            v = complex(vals[n1] * vals[n2]
                        * d2(s_ij[n1, n2], s_i[n1], s_i[n2]))
            if n1 == n2:
                v += g[n1]
            H[n1, n2] = H[n2, n1] = v
    return g, H


def loop_pole_jet(fn, Q, ib, ip, stamps, vals, s0):
    """(g, H) log-log jet of a Tian loop-gain pole's FREQUENCY."""
    names, s_i, s_ij = _root_derivs(fn, Q, ib, ip, stamps, vals, s0)
    return _lift(names, s_i, s_ij, vals, s0, "omega")


def loop_pair_jets(fn, Q, ib, ip, stamps, vals, s0):
    """((g, H) of omega0, (g, H) of zeta) for a COMPLEX pair.

    A pair carries two invariants and the frequency is only half of
    it: the damping sets the peaking and the settling, and a designer
    who cannot express zeta cannot trade it. Both come from the one
    set of root derivatives, so the second costs nothing. The zeta
    jets come back empty for a real root, which has no damping to
    speak of."""
    names, s_i, s_ij = _root_derivs(fn, Q, ib, ip, stamps, vals, s0)
    return (_lift(names, s_i, s_ij, vals, s0, "omega"),
            _lift(names, s_i, s_ij, vals, s0, "zeta"))


def loop_denominator(fn, Q, ib, ip, s):
    """(Dn, dDn/ds) where Dn = 2(BC - AD) + A - D + 1: the function
    whose roots ARE the Tian loop gain's poles.

    One inverse, four bilinear reads. Exposed so a root can be
    FOLLOWED by Newton as an element is walked along a ray, without
    re-solving the circuit at every step -- the loop gain's poles are
    what a parameter sweep has to track, and tracking them is what
    makes the collapse of a shared root observable."""
    A = np.asarray(fn(s), complex)
    M = np.linalg.inv(A)
    dim = A.shape[0]
    e_b = np.zeros(dim, complex)
    e_b[ib] = 1.0
    e_p = np.zeros(dim, complex)
    e_p[ip] = 1.0
    Qm = np.asarray(Q, complex)

    def read(u, w):
        Mu = M.T @ u
        Mw = M @ w
        return complex(Mu @ w), complex(-(Mu @ (Qm @ Mw)))

    (Bv, Bs) = read(e_b, e_b)
    (Dv, Ds) = read(e_p, e_b)
    (Av, As) = read(e_b, e_p)
    (Cv, Cs) = read(e_p, e_p)
    B, D, A_, C = -Bv, Dv, -Av, Cv
    B_s, D_s, A_s, C_s = -Bs, Ds, -As, Cs
    Dn = 2.0 * (B * C - A_ * D) + A_ - D + 1.0
    Dn_s = (2.0 * (B_s * C + B * C_s - A_s * D - A_ * D_s)
            + A_s - D_s)
    return Dn, Dn_s


def follow_root(fn, Q, ib, ip, s0, *, tol=1e-10, itmax=40):
    """Newton-track a loop-gain pole from `s0` under whatever values
    `fn` now carries. Returns the refined root, or None if it will not
    converge -- a root that runs away under a ray step is a root that
    left, and saying so beats reporting where it was."""
    s = complex(s0)
    for _ in range(itmax):
        Dn, Dn_s = loop_denominator(fn, Q, ib, ip, s)
        if Dn_s == 0 or not np.isfinite(Dn_s):
            return None
        step = Dn / Dn_s
        s = s - step
        if not np.isfinite(s):
            return None
        if abs(step) <= tol * max(abs(s), 1.0):
            return s
    return None
