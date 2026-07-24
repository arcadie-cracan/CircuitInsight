"""D-C: the 2x2 mode loop matrix at a probe pair (DM/CM stability).

Generalizes the Eq.-30 Tian loop gain to TWO 0-V probe branches (the two
mode wires of an fd_probe, or any pair that jointly breaks the loops of
interest): per complex frequency, four injections (series unit voltage
at each branch, unit current into each branch's p node) yield the 2x2
readout blocks

    B[j,k] = -i_b(j) | series V at k      D[j,k] = v_p(j) | series V at k
    A[j,k] = -i_b(j) | current into p(k)  C[j,k] = v_p(j) | current into p(k)

and the block Eq. 30

    L = -[2(AD - BC) - A + D] . [2(BC - AD) + A - D + I]^{-1}.

Validation status (empirically established on the fd fixture; see
tests/test_modes.py and docs/loopgain-plan.md Sec. 8):

  * RETURN-DIFFERENCE EXACT: det(I - L(s)) -> 0 at every closed-loop
    natural frequency (roots of det A), under perfect symmetry AND
    under strong deliberate mismatch -- so the generalized-Nyquist
    verdict from the eigenloci is trustworthy. (Critical point is +1:
    the stb convention, arg T(DC) = +180 deg, confirmed by |1 - T| -> 0
    at the flagship's closed-loop poles.)
  * SYMMETRY EXACT: for a matched circuit L is numerically diagonal
    (off-diagonals at machine zero) and L_ii equals the scalar Tian
    measurement at branch i exactly.
  * ATTRIBUTION APPROXIMATE: under mismatch the Schur closure
    L_ii + L_ij (1 - L_jj)^{-1} L_ji reproduces the directly-measured
    closed-other-branch scalar T_eff only to O(mismatch^2) (2e-4 at a
    brutal 50% gm mismatch; block orderings differ at the same order).
    This is not a defect of the ordering: NO exact MIMO extension of
    the Tian convention exists. The four natural identities (two
    Schur closures + two chain-rule factorizations of the pair
    return difference) overdetermine (L11, L22, L12 L21), and the
    chain rule fails for Tian scalars at O(coupling^2) -- see
    docs/loopgain-plan.md Sec. 10.1. `schur_residual` is precisely a
    measurement of that obstruction; the Spectre-matching per-mode
    numbers remain the scalar closed-other-branch measurements.
    (Algebraic bonus: num + den = I identically for any ordering, so
    det(I - L) = 1/det(den) -- the reason the det-nulling oracle
    passes, with linear decay at symmetric poles and quadratic at
    mismatched ones, as observed.)

With mismatch the off-diagonals quantify cross-mode coupling; the
SISO-per-mode picture is valid while `coupling`
r = |L12 L21| / (|1-L11| |1-L22|) stays small.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import sympy as sp

from ..engine.mna import MnaError, MnaSystem, S, hybrid_split
from .compensate import _margins_of


def _probe_rows(system: MnaSystem, probe: str):
    if probe not in system.branch_index:
        raise MnaError(f"probe {probe!r} is not a voltage-defined branch")
    ib = system.branch_index[probe]
    ip = iq = None
    for i in range(len(system.node_index)):
        e = system.A[i, ib]
        if e == 1:
            ip = i
        elif e == -1:
            iq = i
    if ip is None or iq is None:
        raise MnaError(f"probe {probe!r}: could not identify both nodes")
    return ip, iq, ib


def _numeric_fn(system: MnaSystem, scale=None):
    """lambdify(S, A) with optional per-symbol value scaling
    (scale = {"gm_I0_MN1P": 1.1, ...} -- the mismatch knob)."""
    subs, _ = hybrid_split(system, [])
    if scale:
        for name, fac in scale.items():
            sym = system.symbols[name]
            if sym not in subs:
                raise MnaError(f"unknown symbol {name!r}")
            # keep the substitution EXACT: a python-float factor would
            # push the matrix off the rational domain and make _det
            # (DomainMatrix) fall back to the catastrophically slow EX
            # domain
            subs = {**subs, sym: subs[sym] * sp.Rational(str(fac))}
    A_v = system.A.xreplace(subs)
    return sp.lambdify(S, A_v, "numpy"), A_v


def _blocks(M: np.ndarray, rows):
    """The four 2x2 readout blocks at one complex frequency."""
    dim = M.shape[0]
    n = len(rows)
    Ab = np.empty((n, n), complex); Bb = np.empty((n, n), complex)
    Cb = np.empty((n, n), complex); Db = np.empty((n, n), complex)
    lu = np.linalg.inv(M)               # 4 solves share the factorization
    for k, (ipk, iqk, ibk) in enumerate(rows):
        zv = np.zeros(dim, complex); zv[ibk] = 1.0
        zi = np.zeros(dim, complex); zi[ipk] = 1.0
        xv = lu @ zv
        xi = lu @ zi
        for j, (ipj, iqj, ibj) in enumerate(rows):
            Bb[j, k] = -xv[ibj]
            Db[j, k] = xv[ipj]
            Ab[j, k] = -xi[ibj]
            Cb[j, k] = xi[ipj]
    return Ab, Bb, Cb, Db


def loop_matrix_at(system: MnaSystem, probes, s_values, *, scale=None,
                   fn=None) -> np.ndarray:
    """L(s) for each complex s in s_values; shape (len(s), n, n)."""
    rows = [_probe_rows(system, p) for p in probes]
    if fn is None:
        fn, _ = _numeric_fn(system, scale)
    n = len(rows)
    eye = np.eye(n)
    out = np.empty((len(s_values), n, n), complex)
    for i, s in enumerate(s_values):
        M = np.asarray(fn(s), complex)
        Ab, Bb, Cb, Db = _blocks(M, rows)
        num = 2 * (Ab @ Db - Bb @ Cb) - Ab + Db
        den = 2 * (Bb @ Cb - Ab @ Db) + Ab - Db + eye
        out[i] = -num @ np.linalg.inv(den)
    return out


def eigenloci(L: np.ndarray) -> np.ndarray:
    """Continuously-tracked eigenvalue loci; shape (npts, n). Adjacent
    points are matched by nearest distance so each column is one locus."""
    npts, n, _ = L.shape
    loci = np.empty((npts, n), complex)
    loci[0] = np.linalg.eigvals(L[0])
    for i in range(1, npts):
        ev = np.linalg.eigvals(L[i])
        prev = loci[i - 1]
        if n == 2:
            d0 = abs(ev[0] - prev[0]) + abs(ev[1] - prev[1])
            d1 = abs(ev[1] - prev[0]) + abs(ev[0] - prev[1])
            loci[i] = ev if d0 <= d1 else ev[::-1]
        else:                            # general small-n greedy match
            remaining = list(range(n))
            row = np.empty(n, complex)
            for j in range(n):
                k = min(remaining, key=lambda m: abs(ev[m] - prev[j]))
                row[j] = ev[k]
                remaining.remove(k)
            loci[i] = row
    return loci


@dataclass
class ModeLoopReport:
    probes: tuple
    freqs: np.ndarray
    L: np.ndarray                       # (npts, 2, 2)
    loci: np.ndarray                    # (npts, 2), continuity-tracked
    margins: list                       # per locus: (pm_deg, f_unity, gm_db)
    coupling: np.ndarray                # r(w) = |L12 L21|/(|1-L11||1-L22|)
    T_eff: np.ndarray                   # (npts, 2): scalar Tian per branch
    schur_residual: float               # accuracy certificate (see module doc)
    labels: list = field(default_factory=list)

    @property
    def max_coupling(self) -> float:
        return float(np.max(self.coupling))

    def summary(self) -> str:
        parts = []
        for lab, (pm, fu, gm) in zip(self.labels, self.margins):
            if pm is None:
                parts.append(f"{lab}: no crossing")
            else:
                parts.append(f"{lab}: PM {pm:.2f} deg @ {fu:.4g} Hz")
        parts.append(f"max cross-mode coupling r = {self.max_coupling:.3g}")
        return "; ".join(parts)


def mode_loop(system: MnaSystem, probe_a: str, probe_b: str, *,
              freqs=None, scale=None) -> ModeLoopReport:
    """The 2x2 mode analysis at a probe pair. Under symmetry the loci
    ARE the scalar per-branch loop gains; labels are assigned by which
    branch dominates each locus at the lowest frequency."""
    if freqs is None:
        freqs = np.logspace(2, 9, 281)
    freqs = np.asarray(freqs, dtype=float)
    probes = (probe_a, probe_b)
    fn, _ = _numeric_fn(system, scale)
    L = loop_matrix_at(system, probes, 2j * np.pi * freqs, fn=fn)
    loci = eigenloci(L)
    # label loci by proximity to the diagonal entries at the first point
    d0 = np.array([L[0, 0, 0], L[0, 1, 1]])
    if (abs(loci[0, 0] - d0[0]) + abs(loci[0, 1] - d0[1])
            > abs(loci[0, 0] - d0[1]) + abs(loci[0, 1] - d0[0])):
        loci = loci[:, ::-1]
    labels = [probe_a, probe_b]
    margins = [_margins_of(freqs, loci[:, k]) for k in range(2)]
    coupling = (np.abs(L[:, 0, 1] * L[:, 1, 0])
                / np.maximum(np.abs((1 - L[:, 0, 0]) * (1 - L[:, 1, 1])),
                             1e-300))
    # scalar closed-other-branch measurements (the Spectre-matching
    # per-mode loop gains) + the Schur self-certificate
    s_values = 2j * np.pi * freqs
    T_eff = np.empty((len(freqs), 2), complex)
    T_eff[:, 0] = loop_matrix_at(system, (probe_a,), s_values,
                                 fn=fn)[:, 0, 0]
    T_eff[:, 1] = loop_matrix_at(system, (probe_b,), s_values,
                                 fn=fn)[:, 0, 0]
    resid = 0.0
    for i, j in ((0, 1), (1, 0)):
        closed = (L[:, i, i]
                  + L[:, i, j] * L[:, j, i] / (1 - L[:, j, j]))
        resid = max(resid, float(np.max(
            np.abs(closed - T_eff[:, i])
            / np.maximum(np.abs(T_eff[:, i]), 1e-300))))
    return ModeLoopReport(probes=probes, freqs=freqs, L=L, loci=loci,
                          margins=margins, coupling=coupling,
                          T_eff=T_eff, schur_residual=resid,
                          labels=labels)
