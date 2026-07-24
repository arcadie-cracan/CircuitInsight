"""Candidate compensation-branch screening (milestone N-A of
docs/compensation-synthesis-plan.md).

Only OP-invariant candidates are considered: a capacitor or series-RC branch
across a node pair (or node to ground) carries no DC current, so the
operating point of the reconstruction is untouched -- one DC solve spans the
whole compensation design space. Adding admittance Y(s) across incidence
vector b is a rank-one MNA update,

    det(A + Y b b^T) = det(A) * (1 + Y(s) * Z_port(s)),
    Z_port = b^T A^{-1} b   (driving-point impedance across the candidate),

the one-port return-ratio identity (Tian Eqs. 4-8 / Bode). Everything here
follows from it:

- **screening**: sensitivities of every natural frequency p to a new cap at
  every pair, dp/dC = -p (w*.b)(b.v) / (w*. A' v), with w, v the left/right
  null vectors of A(p) -- ALL pairs at once from one SVD per pole;
- **exact loci**: Z_port as an exact rational Nz/D (two Cramer solves per
  pair); a cap C moves the natural frequencies to the roots of
  D(s) + s C Nz(s); a series RC to the roots of D (1 + sRC) + s C Nz;
- **the design goal**, stated the structured-design (Delft) way rather than
  as a bare phase-margin floor: the servo (closed-loop) response should be
  maximally flat (Butterworth poles), and the attainable bandwidth is
  budgeted by the midband loop gain and the n dominant loop poles,

      omega_h = |(1 - L_MB) * prod_{i=1..n} p_i|^{1/n}.

  Compensation places poles on that Butterworth circle; PM/GM stay derived,
  reported metrics. (Verhoeven et al., Structured Electronic Design; the
  TU Delft webbook formulates the goal exactly so.)

Second objective: **least area** -- among goal-achieving networks, minimize
the total added capacitance (primary; caps dominate area) with resistance
as a weighted secondary. The screening leverage |dp/dC| is the
area-efficiency proxy: Miller-multiplied bridges and phantom zeros move
poles per femtofarad, load caps per picofarad.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import sympy as sp

from ..engine.mna import MnaError, MnaSystem, S, _det, hybrid_split


def _numeric_A(system: MnaSystem) -> sp.Matrix:
    subs, _ = hybrid_split(system, [])
    return system.A.xreplace(subs)


def _pair_incidence(system: MnaSystem, node_a: str, node_b: str | None):
    """Incidence vector b for a branch node_a -> node_b (None = ground)."""
    n = system.A.rows
    b = np.zeros(n)
    if node_a not in system.node_index:
        raise MnaError(f"unknown node {node_a!r}")
    b[system.node_index[node_a]] = 1.0
    if node_b is not None:
        if node_b not in system.node_index:
            raise MnaError(f"unknown node {node_b!r}")
        b[system.node_index[node_b]] -= 1.0
    return b


@dataclass
class PoleCapScreen:
    """Sensitivity of one natural frequency to a unit cap at every candidate
    pair: dp/dC in (rad/s)/F, keyed by (node_a, node_b) with node_b=None for
    a shunt-to-ground candidate."""
    pole_hz: complex
    dpdc: dict[tuple[str, str | None], complex] = field(default_factory=dict)

    def ranked(self, top: int = 10):
        """Pairs by |dp/dC| descending -- where a small cap moves THIS pole
        the most (the direction of movement is the complex phase)."""
        return sorted(self.dpdc.items(), key=lambda kv: -abs(kv[1]))[:top]


def cap_pole_screen(system: MnaSystem, n_poles: int = 4,
                    pairs=None) -> list[PoleCapScreen]:
    """Screen every candidate pair against the first n_poles natural
    frequencies. One SVD per pole; O(N^2) pairs read off the null vectors.

    dp/dC = -p * (conj(w).b)(b.v) / (conj(w). A'(p) v)   [A = P + s Q]
    """
    A_v = _numeric_A(system)
    Q = np.array(A_v.diff(S).tolist(), dtype=complex)
    P = np.array(A_v.xreplace({S: sp.Integer(0)}).tolist(), dtype=complex)

    den = sp.Poly(_det(A_v), S)
    coeffs = [complex(c) for c in den.all_coeffs()]
    scale = max(abs(c) for c in coeffs if c != 0)
    roots = np.roots([c / scale for c in coeffs])
    roots = roots[np.argsort(np.abs(roots))]
    roots = roots[np.abs(roots) > 0][:n_poles]

    nodes = sorted(system.node_index, key=system.node_index.get)
    nn = len(nodes)
    if pairs is None:
        pairs = [(nodes[i], nodes[j]) for i in range(nn)
                 for j in range(i + 1, nn)]
        pairs += [(n_, None) for n_ in nodes]

    out = []
    for p in roots:
        M = P + p * Q
        U, sv, Vh = np.linalg.svd(M)
        v = Vh[-1].conj()
        w = U[:, -1]
        denom = np.vdot(w, Q @ v)
        wj = w.conj()
        scr = PoleCapScreen(pole_hz=complex(p / (2 * np.pi)))
        for (na, nb) in pairs:
            ia = system.node_index[na]
            wb = wj[ia]
            bv = v[ia]
            if nb is not None:
                ib = system.node_index[nb]
                wb = wb - wj[ib]
                bv = bv - v[ib]
            scr.dpdc[(na, nb)] = complex(-p * wb * bv / denom)
        out.append(scr)
    return out


def port_impedance_rational(system: MnaSystem, node_a: str,
                            node_b: str | None = None):
    """Exact rational Z_port = Nz/D across (node_a, node_b|ground):
    two Cramer numerators over the shared determinant. Returns
    (Nz, D) as sympy Polys in s."""
    A_v = _numeric_A(system)
    b = _pair_incidence(system, node_a, node_b)
    z = sp.Matrix([[sp.Integer(int(x))] for x in b])

    def cramer(col: int) -> sp.Expr:
        Ak = A_v.copy()
        Ak[:, col] = z
        return _det(Ak)

    ia = system.node_index[node_a]
    Nz = cramer(ia)
    if node_b is not None:
        Nz = Nz - cramer(system.node_index[node_b])
    D = _det(A_v)
    return sp.Poly(sp.expand(Nz), S), sp.Poly(sp.expand(D), S)


def _scaled_roots(poly: sp.Poly) -> np.ndarray:
    coeffs = [complex(c) for c in poly.all_coeffs()]
    scale = max((abs(c) for c in coeffs if c != 0), default=1.0)
    r = np.roots([c / scale for c in coeffs])
    return r[np.argsort(np.abs(r))]


def _is_rhp(poles, rel_tol: float = 1e-6) -> bool:
    """True if any natural frequency is genuinely in the right half plane,
    by a SCALE-INVARIANT damping test: a pole p counts as unstable only when
    Re(p) > rel_tol * |p| (its damping ratio is negative by more than the
    noise floor).

    An absolute threshold in Hz (the old `poles.real > 1e-3`) is fragile: a
    high-degree characteristic polynomial rooted numerically gives its large
    (GHz) roots an absolute real-part error of ~|p|*eps, which on a fast
    circuit is many orders above 1 mHz -- so a genuinely stable pole would be
    misread as unstable and its candidate spuriously dropped. The relative
    test tracks the pole magnitude, so the same numerical noise (Re/|p| ~
    1e-9..1e-12) never trips it while a real instability (Re/|p| from ~1e-4
    up to O(1)) always does. Poles negligible against the spectral scale
    (numerical zeros near the origin) are ignored -- their relative real part
    is meaningless."""
    p = np.asarray(poles, dtype=complex)
    if p.size == 0:
        return False
    mag = np.abs(p)
    scale = mag.max()
    if scale == 0:
        return False
    live = mag > 1e-9 * scale                    # ignore origin/zero roots
    return bool(np.any(p.real[live] > rel_tol * mag[live]))


def cap_locus(system: MnaSystem, node_a: str, node_b: str | None,
              C_values) -> list[np.ndarray]:
    """Natural frequencies (Hz, sorted by |.|) after adding a capacitor C
    across the pair, for each C: roots of D(s) + s*C*Nz(s). Exact rank-one
    update -- no re-solve of the network per value."""
    Nz, D = port_impedance_rational(system, node_a, node_b)
    s = S
    out = []
    for C in np.atleast_1d(C_values):
        poly = sp.Poly(D.as_expr() + s * sp.Rational(repr(float(C)))
                       * Nz.as_expr(), s)
        out.append(_scaled_roots(poly) / (2 * np.pi))
    return out


def rc_locus(system: MnaSystem, node_a: str, node_b: str | None,
             C: float, R_values) -> list[np.ndarray]:
    """Natural frequencies after adding a series R-C branch across the pair:
    roots of D(s)*(1 + sRC) + s*C*Nz(s), per R."""
    Nz, D = port_impedance_rational(system, node_a, node_b)
    s = S
    Cq = sp.Rational(repr(float(C)))
    out = []
    for R in np.atleast_1d(R_values):
        Rq = sp.Rational(repr(float(R)))
        poly = sp.Poly(D.as_expr() * (1 + s * Rq * Cq)
                       + s * Cq * Nz.as_expr(), s)
        out.append(_scaled_roots(poly) / (2 * np.pi))
    return out


# ------------------------------------------------- the Delft-style goal


def servo_bandwidth(T, n: int) -> float:
    """Attainable MFM (Butterworth) servo bandwidth in Hz from the loop
    gain: omega_h = |(1 - L_MB) * prod_{i=1..n} p_i|^{1/n} with L_MB the
    midband (DC here) loop gain and p_i the n dominant loop poles
    (structured-design bandwidth budget; homogeneous, so evaluated in Hz).
    `T` is the loop-gain TransferFunction in the stb convention (T(0)
    negative real under negative feedback)."""
    l0 = complex(T.numeric([1e-3])[0])
    poles = T.poles()[:n]
    if len(poles) < n:
        raise MnaError(f"loop gain has only {len(poles)} poles, need {n}")
    prod = np.prod(np.abs(poles))
    return float(abs(1 - l0) ** (1.0 / n) * prod ** (1.0 / n))


def butterworth_targets(f_h: float, n: int) -> np.ndarray:
    """Target closed-loop pole positions (Hz, complex, LHP) for an n-th
    order Butterworth (MFM) response with bandwidth f_h."""
    k = np.arange(n)
    ang = np.pi * (2 * k + n + 1) / (2 * n)
    return f_h * np.exp(1j * ang)


# ============================ N-B: semantic candidate generation =========


def dc_gain_matrix(system: MnaSystem):
    """Node-to-node DC voltage-gain proxies a[i, j] = Z_ij(0)/Z_jj(0): the
    voltage appearing at node i when node j is driven, sources zeroed --
    the detector behind Miller-bridge candidates (strongly inverting pairs
    get Miller multiplication). Returns (nodes, a)."""
    A_v = _numeric_A(system)
    P = np.array(A_v.xreplace({S: sp.Integer(0)}).tolist(), dtype=float)
    M = np.linalg.inv(P)
    nodes = sorted(system.node_index, key=system.node_index.get)
    n = len(nodes)
    Z = M[:n, :n]
    d = np.diag(Z).copy()
    d[np.abs(d) < 1e-300] = np.inf
    return nodes, Z / d[None, :]


def miller_candidates(system: MnaSystem, min_gain: float = 2.0):
    """Bridge positions across inverting gain >= min_gain: candidate
    pole-splitting (Miller) pairs, strongest inversion first. Returns
    [(node_in, node_out, gain)] with gain = a(out from in) < 0."""
    nodes, a = dc_gain_matrix(system)
    out = []
    n = len(nodes)
    for j in range(n):              # driven (stage input)
        for i in range(n):          # response (stage output)
            if i != j and a[i, j] < -min_gain:
                out.append((nodes[j], nodes[i], float(a[i, j])))
    out.sort(key=lambda t: t[2])
    return out


def pole_participation(system: MnaSystem, n_poles: int = 4):
    """Which nodes own each natural frequency: modal participation
    |w_k conj * v_k| per node from the left/right null vectors, normalized.
    Shunt/damping candidates go where the offending pole lives. Returns
    [(pole_hz, [(node, participation), ...] descending)]."""
    A_v = _numeric_A(system)
    Q = np.array(A_v.diff(S).tolist(), dtype=complex)
    P = np.array(A_v.xreplace({S: sp.Integer(0)}).tolist(), dtype=complex)
    den = sp.Poly(_det(A_v), S)
    coeffs = [complex(c) for c in den.all_coeffs()]
    scale = max(abs(c) for c in coeffs if c != 0)
    roots = np.roots([c / scale for c in coeffs])
    roots = roots[np.argsort(np.abs(roots))]
    roots = roots[np.abs(roots) > 0][:n_poles]

    nodes = sorted(system.node_index, key=system.node_index.get)
    n = len(nodes)
    out = []
    for p in roots:
        U, sv, Vh = np.linalg.svd(P + p * Q)
        v = Vh[-1].conj()
        w = U[:, -1]
        part = np.abs(w[:n].conj() * v[:n])
        tot = part.sum() or 1.0
        ranked = sorted(zip(nodes, part / tot), key=lambda t: -t[1])
        out.append((complex(p / (2 * np.pi)), ranked))
    return out


class LoopGainUpdater:
    """Sherman-Morrison fast path: the Eq.-30 loop gain at the probe,
    re-evaluated for ANY candidate branch (pair, Y(s)) on a frequency grid
    without re-factoring the network. Per frequency, A^{-1} and the two
    injection solutions are cached; a candidate updates each solution in
    closed form:

        x_new = x - Y (b.x) / (1 + Y b.A^{-1}b) * (A^{-1}b).

    Scores phantom-zero / damping candidates by their effect on the loop
    gain near the crossover; feeds the MFM cost in N-C."""

    def __init__(self, system: MnaSystem, probe: str, freqs):
        if probe not in system.branch_index:
            raise MnaError(f"probe {probe!r} is not a branch")
        self.system = system
        self.freqs = np.atleast_1d(np.asarray(freqs, dtype=float))
        self.ib = system.branch_index[probe]
        n_nodes = len(system.node_index)
        self.ip = self.iq = None
        for i in range(n_nodes):
            e = system.A[i, self.ib]
            if e == 1:
                self.ip = i
            elif e == -1:
                self.iq = i

        A_v = _numeric_A(system)
        fn = sp.lambdify(S, A_v, "numpy")
        dim = A_v.rows
        zv = np.zeros(dim, complex); zv[self.ib] = 1.0
        zi = np.zeros(dim, complex); zi[self.ip] = 1.0
        self.Minv, self.xv, self.xi = [], [], []
        for f in self.freqs:
            M = np.linalg.inv(np.asarray(fn(2j * np.pi * f), complex))
            self.Minv.append(M)
            self.xv.append(M @ zv)
            self.xi.append(M @ zi)

    @staticmethod
    def _tian(xv, xi, ip, ib):
        B = -xv[ib]; D = xv[ip]
        A_ = -xi[ib]; C = xi[ip]    # v_q == v_p under the closed 0-V branch
        return -(2 * (A_ * D - B * C) - A_ + D) / \
            (2 * (B * C - A_ * D) + A_ - D + 1)

    def baseline(self) -> np.ndarray:
        return np.array([self._tian(xv, xi, self.ip, self.ib)
                         for xv, xi in zip(self.xv, self.xi)])

    def with_branch(self, node_a: str, node_b: str | None,
                    Y_of_s) -> np.ndarray:
        """T(j 2 pi f) with admittance Y (callable of s) added across the
        pair, via Sherman-Morrison on the cached solutions."""
        b = _pair_incidence(self.system, node_a, node_b)
        out = np.empty(len(self.freqs), complex)
        for k, f in enumerate(self.freqs):
            Y = complex(Y_of_s(2j * np.pi * f))
            Minv = self.Minv[k]
            Mb = Minv @ b
            den = 1 + Y * (b @ Mb)
            xv = self.xv[k] - (Y * (b @ self.xv[k]) / den) * Mb
            xi = self.xi[k] - (Y * (b @ self.xi[k]) / den) * Mb
            out[k] = self._tian(xv, xi, self.ip, self.ib)
        return out

    def with_branches(self, branches) -> np.ndarray:
        """T(j 2 pi f) with SEVERAL admittance branches added at once
        (Woodbury update on the cached solutions). branches: iterable of
        (node_a, node_b|None, Y_of_s) -- e.g. a mirrored symmetric pair
        for fully-differential compensation."""
        branches = list(branches)
        Bcols = [_pair_incidence(self.system, na, nb)
                 for na, nb, _ in branches]
        B = np.column_stack(Bcols)
        out = np.empty(len(self.freqs), complex)
        for i, f in enumerate(self.freqs):
            s = 2j * np.pi * f
            Yd = np.array([complex(Y(s)) for _, _, Y in branches])
            Minv = self.Minv[i]
            W = Minv @ B                     # dim x k
            G = B.T @ W                      # k x k
            core = np.diag(1.0 / Yd) + G
            xv = self.xv[i] - W @ np.linalg.solve(core, B.T @ self.xv[i])
            xi = self.xi[i] - W @ np.linalg.solve(core, B.T @ self.xi[i])
            out[i] = self._tian(xv, xi, self.ip, self.ib)
        return out


def pair_port_rational(system: MnaSystem, pair_a, pair_b):
    """(N11, N12, D): self and cross Cramer numerators of the 2x2 port
    impedance matrix for two branch pairs (N22 = N11 under matched
    symmetry, which is the intended use)."""
    A_v = _numeric_A(system)
    b1 = _pair_incidence(system, *pair_a)
    b2 = _pair_incidence(system, *pair_b)
    z1 = sp.Matrix([[sp.Integer(int(x))] for x in b1])
    z2 = sp.Matrix([[sp.Integer(int(x))] for x in b2])

    def functional(zvec, pair):
        Ak = A_v.copy()
        ia = system.node_index[pair[0]]
        Ak[:, ia] = zvec
        n = _det(Ak)
        if pair[1] is not None:
            Ak = A_v.copy()
            Ak[:, system.node_index[pair[1]]] = zvec
            n = n - _det(Ak)
        return n

    N11 = functional(z1, pair_a)
    N12 = functional(z2, pair_a)
    D = _det(A_v)
    return (sp.Poly(sp.expand(N11), S), sp.Poly(sp.expand(N12), S),
            sp.Poly(sp.expand(D), S))


def pair_locus_family(system: MnaSystem, pair_a, pair_b):
    """roots(C, R) for a SYMMETRIC mirrored pair of series-RC branches
    (same Y on both). Under matched symmetry the rank-2 determinant
    factorizes by modes:

        det(I + Y G2) = (1 + Y (Z11+Z12)) (1 + Y (Z11-Z12)),

    so the natural frequencies are the union of two rank-1 loci (the
    even- and odd-mode port impedances) with one copy of the
    unperturbed poles (roots of D) removed -- exact pole/zero
    accounting of F+ F- / D. Assumes the circuit is matched-symmetric
    under the mirror that maps pair_a to pair_b."""
    N11, N12, D = pair_port_rational(system, pair_a, pair_b)
    s = S
    Dx = D.as_expr()
    Np = sp.expand(N11.as_expr() + N12.as_expr())
    Nm = sp.expand(N11.as_expr() - N12.as_expr())
    d_roots = _scaled_roots(D)

    def roots(C: float, R: float = 0.0) -> np.ndarray:
        Cq = sp.Rational(repr(float(C)))
        Rq = sp.Rational(repr(float(R)))
        rr = []
        for Nx in (Np, Nm):
            poly = sp.Poly(Dx * (1 + s * Rq * Cq) + s * Cq * Nx, s)
            rr.append(_scaled_roots(poly))
        allr = list(np.concatenate(rr))
        for dr in d_roots:
            j = min(range(len(allr)), key=lambda m: abs(allr[m] - dr))
            allr.pop(j)
        return np.asarray(allr) / (2 * np.pi)

    return roots


def multi_port_rational(system: MnaSystem, pairs):
    """The k x k matrix of port cofactor-numerators M[i][j] = b_i^T adj(A) b_j
    (so Z_ij = M[i][j]/D) for an arbitrary, not-necessarily-symmetric set of
    branch pairs, plus D = det A. Two Cramer numerators per ordered pair.
    Generalizes pair_port_rational() (which assumes a matched 2-pair)."""
    A_v = _numeric_A(system)
    D = _det(A_v)
    zs = [sp.Matrix([[sp.Integer(int(x))]
                     for x in _pair_incidence(system, *p)]) for p in pairs]

    def numer(pi, zj):
        ia = system.node_index[pi[0]]
        Ak = A_v.copy(); Ak[:, ia] = zj
        val = _det(Ak)
        if pi[1] is not None:
            Ak = A_v.copy(); Ak[:, system.node_index[pi[1]]] = zj
            val = val - _det(Ak)
        return sp.expand(val)

    M = [[sp.Poly(numer(pi, zs[j]), S) for j in range(len(pairs))]
         for pi in pairs]
    return M, sp.Poly(sp.expand(D), S)


def multi_locus(system: MnaSystem, pairs):
    """roots(values) for a SET of independently sized OP-invariant branches
    across `pairs` (values = [(C, R), ...] per branch, R=0 -> plain C):
    the exact closed-loop natural frequencies (Hz) with the k branches
    installed, with no re-solve of the network per value.

    The reconstructed system is linear in s -- A(s) = P + s Q (conductances
    and transconductors are constant, every capacitance stamps s C, no
    inductors) -- so a plain cap adds C into Q across its incidence and a
    series R-C adds an internal node (R into P, C into Q), keeping the
    augmented system linear and giving the branch its LHP zero exactly. The
    natural frequencies are the roots of the characteristic polynomial
    g(s) = det(P + s Q), of degree = the number of dynamic states.

    g is recovered by interpolation on a circle in the scaled variable
    u = s/rho (rho ~ the base pole scale, from the uncompensated finite
    eigenvalues): g(rho*w^m) = sum_j a_j (rho w^m)^j sampled at the M roots of
    unity w^m gives a_j rho^j = FFT(g)/M, so g's coefficients (and hence its
    roots) come out with uniform relative accuracy regardless of the
    coefficient dynamic range -- which a matrix-cap-model bench (every node
    already dynamic, huge-dynamic-range coefficients) has in abundance. This
    is numerically robust for ANY number of branches and any degree, unlike
    forming det(P)/D^(k-1) as a polynomial (whose degree-~k*deg D intermediate
    loses precision as k grows) or the generalized-eigenvalue pencil (blocked
    here by a broken LAPACK ggev, and index-2 anyway from the 0-V probe
    branches). Verified to ~1e-10 against an explicit internal-node stamp for
    plain, mixed and all-RC sets at k up to 4; k=1 matches locus_family, a
    matched 2-pair matches pair_locus_family."""
    A = _numeric_A(system)
    P0 = np.array(A.xreplace({S: sp.Integer(0)}).tolist(), dtype=complex)
    Q0 = np.array(A.diff(S).tolist(), dtype=complex)
    n = P0.shape[0]
    k = len(pairs)
    incs = []
    for na, nb in pairs:
        if na not in system.node_index:
            raise MnaError(f"unknown node {na!r}")
        if nb is not None and nb not in system.node_index:
            raise MnaError(f"unknown node {nb!r}")
        incs.append((system.node_index[na],
                     None if nb is None else system.node_index[nb]))

    # radius = geometric mean of the uncompensated finite pole magnitudes
    # (one symbolic determinant, once): the circle sits at the pole scale, so
    # coefficient recovery is well conditioned. Correctness does not depend on
    # rho (any M > deg samples fix the polynomial); only conditioning does.
    droots = _scaled_roots(sp.Poly(sp.expand(_det(A)), S))
    mag = np.abs(droots[np.abs(droots) > 0])
    rho = float(np.exp(np.mean(np.log(mag)))) if mag.size else 1.0

    def roots(values) -> np.ndarray:
        if len(values) != k:
            raise MnaError(f"expected {k} (C, R) values, got {len(values)}")
        extra = sum(1 for v in values if float(v[1]) != 0.0)
        N = n + extra
        P = np.zeros((N, N), complex); P[:n, :n] = P0
        Q = np.zeros((N, N), complex); Q[:n, :n] = Q0
        m = n
        for (ia, ib), (Cv, Rv) in zip(incs, values):
            Cv, Rv = float(Cv), float(Rv)
            if Rv == 0.0:                            # plain cap into Q
                Q[ia, ia] += Cv
                if ib is not None:
                    Q[ib, ib] += Cv; Q[ia, ib] -= Cv; Q[ib, ia] -= Cv
            else:                                    # series R-C via node m
                g = 1.0 / Rv
                P[ia, ia] += g; P[m, m] += g
                P[ia, m] -= g; P[m, ia] -= g
                Q[m, m] += Cv
                if ib is not None:
                    Q[ib, ib] += Cv; Q[m, ib] -= Cv; Q[ib, m] -= Cv
                m += 1
        # interpolate g(u) = det(P + rho*u*Q) on the unit circle, root in u
        M = 1 << max(4, int(np.ceil(np.log2(N + 2))))
        w = np.exp(2j * np.pi * np.arange(M) / M)
        gv = np.array([np.linalg.det(P + (rho * wm) * Q) for wm in w])
        a = (np.fft.fft(gv) / M)[:N + 1]             # a[j] = coeff of u^j
        hi = a[::-1]                                 # highest-order first
        tol = 1e-9 * np.max(np.abs(a))
        nz = np.nonzero(np.abs(hi) > tol)[0]
        if nz.size == 0 or hi.size - nz[0] < 2:
            return np.array([], dtype=complex)       # degenerate (no roots)
        u = np.roots(hi[nz[0]:])
        w = rho * u
        return w[np.argsort(np.abs(w))] / (2 * np.pi)

    return roots


def locus_family(system: MnaSystem, node_a: str, node_b: str | None):
    """One-time Z_port extraction for a pair; returns roots(C, R) giving the
    natural frequencies (Hz) with a series-RC (R=0 -> plain C) installed --
    the workhorse for sizing sweeps."""
    Nz, D = port_impedance_rational(system, node_a, node_b)
    s = S
    Dx, Nx = D.as_expr(), Nz.as_expr()

    def roots(C: float, R: float = 0.0) -> np.ndarray:
        Cq = sp.Rational(repr(float(C)))
        Rq = sp.Rational(repr(float(R)))
        poly = sp.Poly(Dx * (1 + s * Rq * Cq) + s * Cq * Nx, s)
        return _scaled_roots(poly) / (2 * np.pi)

    return roots


@dataclass
class Candidate:
    """One suggested OP-invariant compensation position, with semantics."""
    kind: str                        # "miller" | "shunt_rc" | "generic"
    node_a: str
    node_b: str | None               # None = ground
    rationale: str
    score: float                     # kind-specific ranking figure


def generate_candidates(system: MnaSystem, n_poles: int = 3,
                        min_gain: float = 2.0, top: int = 12):
    """Assemble the semantic candidate list for a bench:

    - Miller bridges across strongly inverting node pairs (pole splitting);
    - shunt series-RC at the nodes owning the dominant poles (pole-zero
      cancellation / damping);
    - the strongest generic movers of the dominant pole from the rank-one
      screen, for positions the templates do not explain.

    Sized/ranked against the MFM goal in N-C; this is the vocabulary."""
    cands: list[Candidate] = []
    seen = set()

    for (nin, nout, g) in miller_candidates(system, min_gain)[:top]:
        key = frozenset((nin, nout))
        if key in seen:
            continue
        seen.add(key)
        cands.append(Candidate(
            "miller", nin, nout,
            f"inverting gain {g:.1f} from {nin} to {nout}: a bridge cap is "
            f"Miller-multiplied (pole splitting; add series R for an LHP "
            f"zero)", abs(g)))

    for pole, ranked in pole_participation(system, n_poles):
        node, part = ranked[0]
        key = (node, None)
        if key in seen:
            continue
        seen.add(key)
        cands.append(Candidate(
            "shunt_rc", node, None,
            f"node owns the pole at {pole.real / 1e6:+.3g} MHz "
            f"(participation {part:.0%}): a shunt series-RC damps or "
            f"cancels it", float(part)))

    screens = cap_pole_screen(system, n_poles=1)
    for (pair, dp) in screens[0].ranked(top):
        key = frozenset(x for x in pair if x is not None)
        if key in seen:
            continue
        seen.add(key)
        cands.append(Candidate(
            "generic", pair[0], pair[1],
            f"strongest remaining mover of the dominant pole "
            f"(|dp/dC| = {abs(dp):.2g} rad/s per F)", abs(dp)))
    return cands


# ============================ N-C: goal inversion + area ranking =========


def _dominant_pair(poles: np.ndarray):
    """(f0, zeta) of the dominant dynamics: the smallest-|.| complex pair
    (zeta = -Re/|p|), or the smallest real pole treated as over-damped
    (zeta = 1)."""
    poles = poles[np.argsort(np.abs(poles))]
    for p in poles[:4]:
        if abs(p.imag) > 0.01 * abs(p):
            return float(abs(p)), float(-p.real / abs(p))
    p = poles[0]
    return float(abs(p)), 1.0


def _margins_of(freqs, T):
    """(pm_deg, f_unity, gm_db) in the stb convention; None where the
    crossing is absent."""
    m = 20 * np.log10(np.abs(T))
    ph = np.degrees(np.unwrap(np.angle(T)))
    x = np.log10(np.asarray(freqs, dtype=float))
    pm = fu = gm = None
    k = np.where(np.diff(np.sign(m)))[0]
    if k.size:
        k = k[0]
        xu = np.interp(0, [m[k + 1], m[k]], [x[k + 1], x[k]])
        pm = float(np.interp(xu, [x[k], x[k + 1]], [ph[k], ph[k + 1]]))
        fu = float(10 ** xu)
    j = np.where(np.diff(np.sign(ph)))[0]
    if j.size:
        j = j[0]
        xj = np.interp(0, [ph[j + 1], ph[j]], [x[j + 1], x[j]])
        gm = float(-np.interp(xj, [x[j], x[j + 1]], [m[j], m[j + 1]]))
    return pm, fu, gm


@dataclass
class Suggestion:
    """One sized, ranked compensation suggestion."""
    candidate: Candidate
    network: str                     # "C" | "series-RC"
    C: float
    R: float                         # 0 for plain C
    achieved: bool                   # met the goal within tolerance
    f0_hz: float                     # |dominant pair| after compensation
    zeta: float                      # damping of the dominant pair
    area: float                      # C/1pF + kappa*R/1kohm
    budget_hz: float                 # the Delft f_h for context
    pm_deg: float | None = None      # derived metrics at the sized values
    gm_db: float | None = None
    f_unity_hz: float | None = None
    spec_dev: float | None = None    # peak sensitivity Ms (goal="spec")

    def describe(self) -> str:
        net = (f"{self.C * 1e12:.3g} pF"
               if self.R == 0 else
               f"{self.C * 1e12:.3g} pF + {self.R / 1e3:.3g} kOhm in series")
        where = (f"{self.candidate.node_a} to "
                 f"{self.candidate.node_b or 'ground'}")
        pm = f", PM {self.pm_deg:.1f} deg" if self.pm_deg is not None else ""
        return (f"[{self.candidate.kind}] {net} across {where}: dominant "
                f"pair {self.f0_hz / 1e6:.2f} MHz at zeta={self.zeta:.2f} "
                f"({self.f0_hz / self.budget_hz:.0%} of the "
                f"{self.budget_hz / 1e6:.1f} MHz budget){pm} -- "
                f"{self.candidate.rationale}")


def _peak_sensitivity(freqs, T) -> float:
    """Ms = max_w |1/(1 - T(jw))| -- the peak of the sensitivity function,
    Middlebrook's discrepancy/tolerance quantity: |H/Hinf - 1| = |S| when
    direct feedthrough is negligible (the servo regime). Standard robustness
    metric: Ms ~ 1.3 <-> PM ~ 50 deg, Ms ~ 1.2 <-> PM ~ 60 deg; Ms >= 1
    always (the HF floor as |T| -> 0). Unlike the raw |H/Hinf-1|, it does
    NOT diverge at HF, so it needs no band trimming.

    The return difference is (1 - T), NOT (1 + T): loop_gain() uses the
    Spectre-stb convention (arg T = +180 deg at DC), where instability is
    T = +1 (arg 0), so 1 - T -> 0 there. The exact feedthrough-including
    |H/Hinf-1| is on GftQuartet.spec_deviation for inspection."""
    S = 1.0 / (1.0 - np.asarray(T, dtype=complex))
    return float(np.max(np.abs(S)))


def suggest_compensation(system: MnaSystem, probe: str, *,
                         goal: str = "mfm", pm_target: float = 60.0,
                         n_budget: int = 2, zeta_tol: float = 0.05,
                         candidates=None, kappa: float = 0.05,
                         c_grid=None, r_grid=None, mirror=None,
                         ms_target: float = 1.3, top: int = 5) -> list[Suggestion]:
    """Size every candidate against the goal and rank by AREA among the
    achievers (then by goal deviation for the rest).

    goal="mfm" (default, the structured-design formulation): place the
    dominant closed-loop pair at Butterworth damping zeta = 1/sqrt(2)
    (within zeta_tol), all natural frequencies LHP; among the feasible
    (C, R) family of each candidate, pick the LEAST-AREA point. The
    attainable-bandwidth budget f_h = |(1-L_MB) prod p_i|^(1/n) is
    computed from the UNCOMPENSATED loop gain and reported for context
    (budget utilization), not enforced.

    goal="pm": classic phase-margin floor pm_target at the probe (via the
    Sherman-Morrison loop-gain updater); least-area feasible point.

    goal="spec" (Middlebrook's GFT-native target): hold the peak
    sensitivity Ms = max|1/(1+T)| at or below `ms_target` -- the
    discrepancy/D-peaking tolerance, equal to max|H/Hinf-1| when direct
    feedthrough is negligible (the servo regime). A closed-loop robustness
    spec rather than a PM floor: Ms 1.3 ~ PM 50 deg, 1.2 ~ PM 60 deg.
    Least-area feasible point among achievers.

    `system` must be the CLOSED-loop bench built with input_name=probe
    (strip any existing compensation branch from the primitives first).

    `mirror` (fully-differential circuits): a node map {p-side: n-side,
    ...}. Each candidate is installed as the SYMMETRIC PAIR (itself +
    its mirror image, same value), the area counts both elements, the
    pole locus uses the even/odd-mode factorization
    (pair_locus_family), and the PM evaluation updates both branches.
    Self-symmetric candidates (mirror maps the pair onto itself) stay
    single-ended -- the natural CMFB/tail candidates.
    """
    from .loopgain import loop_gain

    T0 = loop_gain(system, probe, [])
    budget = servo_bandwidth(T0, n_budget)

    if candidates is None:
        candidates = generate_candidates(system)

    # Canonicalize to the probe's p side: its two nodes are one electrical
    # node (0-V branch), but a branch wired to the q side would BYPASS the
    # probe -- the measured loop gain then no longer cuts all loops (the
    # Tian paper's critical-wire requirement) and the PM readout lies.
    upd0 = LoopGainUpdater(system, probe, [1.0])
    inv_nodes = {i: n for n, i in system.node_index.items()}
    p_name, q_name = inv_nodes[upd0.ip], inv_nodes[upd0.iq]
    canon, seen_pairs = [], set()
    for c in candidates:
        na = p_name if c.node_a == q_name else c.node_a
        nb = p_name if c.node_b == q_name else c.node_b
        key = (frozenset(x for x in (na, nb) if x is not None), c.kind)
        if na == nb or key in seen_pairs:
            continue
        seen_pairs.add(key)
        canon.append(Candidate(c.kind, na, nb, c.rationale, c.score))
    candidates = canon

    twins: dict[int, tuple | None] = {}
    if mirror is not None:
        mm = {**mirror, **{v: k for k, v in mirror.items()}}
        paired_seen, kept = set(), []
        for c in candidates:
            twin = (mm.get(c.node_a, c.node_a), mm.get(c.node_b, c.node_b))
            key = frozenset((frozenset((c.node_a, c.node_b)),
                             frozenset(twin)))
            if key in paired_seen:
                continue                 # the mirror image of a kept one
            paired_seen.add(key)
            self_sym = frozenset(twin) == frozenset((c.node_a, c.node_b))
            twins[id(c)] = None if self_sym else twin
            kept.append(c)
        candidates = kept

    if c_grid is None:
        c_grid = np.geomspace(0.2e-12, 60e-12, 34)
    if r_grid is None:
        r_grid = np.concatenate(([0.0], np.geomspace(200.0, 20e3, 12)))

    upd = LoopGainUpdater(system, probe, np.geomspace(1e4, 1e9, 240))
    zeta_star = 1 / np.sqrt(2)

    def _y(C, R):
        return lambda s: s * C / (1 + s * R * C)

    def _T_of(pair, C, R, twin):
        Y = _y(C, R)
        if twin is None:
            return upd.with_branch(pair[0], pair[1], Y)
        return upd.with_branches([(pair[0], pair[1], Y),
                                  (twin[0], twin[1], Y)])

    def pm_at(pair, C, R, twin=None):
        return _margins_of(upd.freqs, _T_of(pair, C, R, twin))

    out: list[Suggestion] = []
    for cand in candidates:
        pair = (cand.node_a, cand.node_b)
        twin = twins.get(id(cand))
        mult = 1 if twin is None else 2
        if twin is None:
            roots = locus_family(system, *pair)
        else:
            roots = pair_locus_family(system, pair, twin)
        networks = (("C", [0.0]), ("series-RC", r_grid[1:]))
        for net, rs in networks:
            best = None                  # ((not ok, metric), C, R, f0, zeta)
            best_area = np.inf
            for R in rs:
                if mult * kappa * R / 1e3 >= best_area:
                    continue             # this whole row cannot beat it
                for C in c_grid:
                    area = mult * (C / 1e-12 + kappa * R / 1e3)
                    have_ok = best is not None and not best[0][0]
                    if have_ok and area >= best_area:
                        continue         # cannot improve the area ranking
                    poles = roots(C, R)
                    if _is_rhp(poles):
                        continue
                    f0, zeta = _dominant_pair(poles)
                    sdev = None
                    if goal == "mfm":
                        ok = abs(zeta - zeta_star) <= zeta_tol
                        dev = abs(zeta - zeta_star)
                    elif goal == "pm":
                        pm, fu, gm = pm_at(pair, C, R, twin)
                        ok = pm is not None and pm >= pm_target
                        dev = (pm_target - pm) if pm is not None else 1e9
                    elif goal == "spec":
                        T = _T_of(pair, C, R, twin)
                        sdev = _peak_sensitivity(upd.freqs, T)
                        pm, fu, gm = _margins_of(upd.freqs, T)
                        # a stable loop only: a non-crossing |T|<1 gives a
                        # trivially small Ms but is not a working amp
                        stable = fu is not None and (pm is None or pm > 0)
                        ok = stable and sdev <= ms_target
                        dev = sdev if stable else 1e9
                    else:
                        raise ValueError(f"unknown goal {goal!r}")
                    key = (not ok, area if ok else dev)
                    if best is None or key < best[0]:
                        best = (key, C, R, f0, zeta, ok, sdev)
                        if ok:
                            best_area = min(best_area, area)
            if best is None:
                continue
            _, C, R, f0, zeta, ok, sdev = best
            pm, fu, gm = pm_at(pair, C, R, twin)
            scand = cand
            if twin is not None:
                scand = Candidate(cand.kind, cand.node_a, cand.node_b,
                                  cand.rationale + " [symmetric pair with "
                                  f"({twin[0]}, {twin[1]})]", cand.score)
            out.append(Suggestion(
                candidate=scand, network=net, C=float(C), R=float(R),
                achieved=bool(ok), f0_hz=f0, zeta=zeta,
                area=float(mult * (C / 1e-12 + kappa * R / 1e3)),
                budget_hz=budget, pm_deg=pm, gm_db=gm, f_unity_hz=fu,
                spec_dev=(float(sdev) if sdev is not None else None)))

    out.sort(key=lambda s: (not s.achieved, s.area))
    return out[:top]


# ============================ N-E: multi-branch (NMC) synthesis ===========


@dataclass
class MultiBranch:
    """One installed branch of a multi-branch compensation network. On a
    fully-differential bench a branch is a mirrored PAIR (same value): `twin`
    is the mirror image's (node_a, node_b), or None for a single-ended /
    self-symmetric branch. `mult` (1 or 2) counts the physical elements."""
    kind: str
    node_a: str
    node_b: str | None
    network: str                     # "C" | "series-RC"
    C: float
    R: float
    rationale: str
    twin: tuple | None = None

    @property
    def mult(self) -> int:
        return 1 if self.twin is None else 2

    def physical(self) -> list:
        """The physical (node_a, node_b) branches this entry installs -- one,
        or the mirrored pair."""
        base = (self.node_a, self.node_b)
        return [base] if self.twin is None else [base, self.twin]

    def describe(self) -> str:
        net = (f"{self.C * 1e12:.3g} pF" if self.R == 0 else
               f"{self.C * 1e12:.3g} pF + {self.R / 1e3:.3g} kOhm series")
        pair = (f" (mirror {self.twin[0]} to {self.twin[1] or 'ground'})"
                if self.twin is not None else "")
        return (f"{net} across {self.node_a} to {self.node_b or 'ground'}"
                f"{pair} [{self.kind}]")


@dataclass
class MultiSuggestion:
    """A sized multi-branch (e.g. nested-Miller) compensation network, grown
    one OP-invariant branch at a time by successive rank-one updates."""
    branches: list[MultiBranch]
    achieved: bool
    area: float
    f0_hz: float
    zeta: float
    budget_hz: float
    steps: list[str] = field(default_factory=list)
    pm_deg: float | None = None
    gm_db: float | None = None
    f_unity_hz: float | None = None
    spec_dev: float | None = None

    def describe(self) -> str:
        head = " + ".join(b.describe() for b in self.branches)
        pm = f", PM {self.pm_deg:.1f} deg" if self.pm_deg is not None else ""
        ms = f", Ms {self.spec_dev:.2f}" if self.spec_dev is not None else ""
        return (f"[{len(self.branches)}-branch] {head}: dominant pair "
                f"{self.f0_hz / 1e6:.2f} MHz at zeta={self.zeta:.2f}"
                f"{pm}{ms} (area {self.area:.1f})")


def suggest_multi_compensation(system: MnaSystem, probe: str, *,
                               goal: str = "pm", k_max: int = 2,
                               pm_target: float = 60.0, ms_target: float = 1.3,
                               zeta_tol: float = 0.05, n_budget: int = 2,
                               candidates=None, kappa: float = 0.05,
                               c_grid=None, r_grid=None, mirror=None,
                               min_gain_improve: float = 0.02) -> MultiSuggestion:
    """Grow a multi-branch compensation network greedily -- the nested-Miller
    (NMC) story -- when one OP-invariant branch cannot meet the goal alone.

    Each step installs the single branch (sized least-area over the C / series-
    RC grid) that most improves the goal GIVEN the branches already placed;
    the joint effect is exact at every step (the rank-k determinant identity
    for the natural frequencies via multi_locus, the rank-k Woodbury loop gain
    via LoopGainUpdater.with_branches -- no re-solve of the network). Growth
    stops when the goal is met or an added branch improves the goal deviation
    by less than `min_gain_improve` (a further branch does not pay its area).

    goal="pm"/"spec" size against the loop-gain margins / peak sensitivity Ms
    at the probe; goal="mfm" places the dominant closed-loop pair at
    Butterworth damping. `system` must be the closed-loop bench built with
    input_name=probe (strip any existing compensation first).

    `mirror` (fully-differential NMC): a node map {p-side: n-side, ...}. Each
    installed branch is then the SYMMETRIC PAIR (itself + its mirror image,
    same value) -- so a fully-differential amplifier's nested-Miller network
    is synthesized as matched pairs. The pair's joint effect is exact by the
    SAME engines (multi_locus over all physical branches, with_branches over
    all physical Y's -- no even/odd factorization needed since multi_locus is
    already general), the area counts both elements, and self-symmetric
    candidates (mirror maps the pair onto itself -- CMFB/tail positions) stay
    single-ended.
    """
    from .loopgain import loop_gain

    if goal not in ("pm", "spec", "mfm"):
        raise ValueError(f"unknown goal {goal!r}")

    T0 = loop_gain(system, probe, [])
    budget = servo_bandwidth(T0, n_budget)
    zeta_star = 1 / np.sqrt(2)

    if candidates is None:
        candidates = generate_candidates(system)

    # canonicalize every candidate to the probe's p side (a branch on the q
    # side would bypass the probe and the loop-gain readout would lie -- the
    # Tian critical-wire requirement, exactly as in suggest_compensation)
    upd0 = LoopGainUpdater(system, probe, [1.0])
    inv = {i: n for n, i in system.node_index.items()}
    p_name, q_name = inv[upd0.ip], inv[upd0.iq]
    canon, seen = [], set()
    for c in candidates:
        na = p_name if c.node_a == q_name else c.node_a
        nb = p_name if c.node_b == q_name else c.node_b
        key = frozenset(x for x in (na, nb) if x is not None)
        if na == nb or key in seen:
            continue
        seen.add(key)
        canon.append(Candidate(c.kind, na, nb, c.rationale, c.score))
    candidates = canon

    # mirrored (fully-differential) candidates: keep one of each mirror pair,
    # record its twin (None if self-symmetric) -- same scheme as the
    # single-branch suggest_compensation
    twins: dict[int, tuple | None] = {}
    if mirror is not None:
        mm = {**mirror, **{v: k for k, v in mirror.items()}}
        paired_seen, kept = set(), []
        for c in candidates:
            twin = (mm.get(c.node_a, c.node_a), mm.get(c.node_b, c.node_b))
            key = frozenset((frozenset((c.node_a, c.node_b)), frozenset(twin)))
            if key in paired_seen:
                continue
            paired_seen.add(key)
            self_sym = frozenset(twin) == frozenset((c.node_a, c.node_b))
            twins[id(c)] = None if self_sym else twin
            kept.append(c)
        candidates = kept

    if c_grid is None:
        c_grid = np.geomspace(0.2e-12, 60e-12, 34)
    if r_grid is None:
        r_grid = np.concatenate(([0.0], np.geomspace(200.0, 20e3, 12)))

    upd = LoopGainUpdater(system, probe, np.geomspace(1e4, 1e9, 240))

    def _y(C, R):
        return lambda s: s * C / (1 + s * R * C)

    def _score(root_fn, vals, Ybr):
        """(ok, dev, f0, zeta, pm, gm, fu, sdev) for a trial branch set, or
        None if any natural frequency is in the RHP. `root_fn` is a prebuilt
        multi_locus closure (port numerators cached) so the (C, R) sweep only
        re-roots a polynomial -- no per-value Cramer solves."""
        poles = root_fn(vals)
        # a degenerate size (np.polydiv left a near-constant quotient at this
        # value) or any RHP root -> this size is unusable, skip it
        if poles.size == 0 or _is_rhp(poles):
            return None
        f0, zeta = _dominant_pair(poles)
        pm = gm = fu = sdev = None
        if goal == "mfm":
            dev = abs(zeta - zeta_star)
            ok = dev <= zeta_tol
        else:
            T = upd.with_branches(Ybr)
            pm, fu, gm = _margins_of(upd.freqs, T)
            if goal == "pm":
                ok = pm is not None and pm >= pm_target
                dev = (pm_target - pm) if pm is not None else 1e9
            else:                                    # spec
                sdev = _peak_sensitivity(upd.freqs, T)
                stable = fu is not None and (pm is None or pm > 0)
                ok = stable and sdev <= ms_target
                dev = sdev if stable else 1e9
        return ok, dev, f0, zeta, pm, gm, fu, sdev

    installed: list[MultiBranch] = []
    inst_pairs: list = []             # FLAT list of installed physical pairs
    inst_vals: list = []              # (C, R) aligned with inst_pairs
    inst_Y: list = []                 # (na, nb, Y) aligned with inst_pairs
    steps: list[str] = []
    prev_dev = np.inf
    last = None

    for step in range(k_max):
        placed = set()
        for b in installed:
            for na, nb in b.physical():
                placed.add(frozenset(x for x in (na, nb) if x is not None))
        best = None                  # (rk, cand, twin, net, C, R, sc)
        for cand in candidates:
            pair = (cand.node_a, cand.node_b)
            twin = twins.get(id(cand))
            phys = [pair] if twin is None else [pair, twin]
            if any(frozenset(x for x in p if x is not None) in placed
                   for p in phys):
                continue
            mult = len(phys)
            # port numerators for {installed + this candidate's physical
            # branches}: built ONCE, then the (C, R) grid is cheap rooting
            root_fn = multi_locus(system, inst_pairs + phys)
            for net, rs in (("C", [0.0]), ("series-RC", r_grid[1:])):
                for R in rs:
                    for C in c_grid:
                        vals = inst_vals + [(float(C), float(R))] * mult
                        Ybr = inst_Y + [(p[0], p[1], _y(C, R)) for p in phys]
                        sc = _score(root_fn, vals, Ybr)
                        if sc is None:
                            continue
                        ok, dev = sc[0], sc[1]
                        area = mult * (C / 1e-12 + kappa * R / 1e3)
                        # rank: achievers by least added area, else least dev
                        rk = (not ok, area if ok else dev)
                        if best is None or rk < best[0]:
                            best = (rk, cand, twin, net, float(C), float(R), sc)
        if best is None:
            break
        _, cand, twin, net, C, R, sc = best
        ok, dev, f0, zeta, pm, gm, fu, sdev = sc
        # stop if this branch neither achieves the goal nor meaningfully
        # improves the deviation (it would not pay its area)
        if step > 0 and not ok and (prev_dev - dev) < min_gain_improve * abs(prev_dev):
            break
        rationale = cand.rationale
        if twin is not None:
            rationale += f" [symmetric pair with ({twin[0]}, {twin[1]})]"
        installed.append(MultiBranch(cand.kind, cand.node_a, cand.node_b,
                                     net, C, R, rationale, twin=twin))
        phys = installed[-1].physical()
        inst_pairs.extend(phys)
        inst_vals.extend([(C, R)] * len(phys))
        inst_Y.extend((p[0], p[1], _y(C, R)) for p in phys)
        prev_dev = dev
        last = sc
        steps.append(f"step {step + 1}: add "
                     f"{installed[-1].describe()} -> dev {dev:.3g}"
                     + (" (goal met)" if ok else ""))
        if ok:
            break

    if last is None:                 # nothing installable (e.g. all RHP)
        return MultiSuggestion(branches=[], achieved=False, area=0.0,
                               f0_hz=float("nan"), zeta=float("nan"),
                               budget_hz=budget, steps=["no stable branch"])
    ok, dev, f0, zeta, pm, gm, fu, sdev = last
    area = sum(b.mult * (b.C / 1e-12 + kappa * b.R / 1e3) for b in installed)
    return MultiSuggestion(
        branches=installed, achieved=bool(ok), area=float(area),
        f0_hz=f0, zeta=zeta, budget_hz=budget, steps=steps,
        pm_deg=pm, gm_db=gm, f_unity_hz=fu, spec_dev=sdev)
