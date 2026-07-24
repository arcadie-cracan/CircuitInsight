"""M-G: probe-adequacy advisor -- grade a designated stb probe.

Two axes (docs/loopgain-plan.md Sec. 9), both judged numerically because
CircuitInsight holds the exact closed-loop det A -- something a plain
Spectre run cannot do:

* Trustworthiness: do the probe's margins tell the truth about the
  closed-loop damping? We compare the PM-implied second-order damping
  with the actual damping of every dominant closed-loop pole pair, and
  we check for RHP poles outright.
* Coverage: which feedback dynamics is the probe blind to? For every
  active device we compute the elasticity of the dominant poles to its
  gm (first-order, via the same left/right null vectors used by the
  cap screen) and the elasticity of the probe's loop gain T. A device
  that moves closed-loop poles but leaves T unchanged belongs to a loop
  the probe does not observe (e.g. the CMFB loop seen from a DM probe,
  or a bypassed probe's main loop).

Graph connectivity is deliberately NOT used: removing a probe that sits
inside a loop always leaves its terminals connected around the loop, so
p-q connectivity is vacuous; and mode probes (fd_probe) legitimately
leave the other mode's loop closed. The numeric detectors handle both.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import sympy as sp

from ..engine.mna import MnaError, MnaSystem, S, _det, hybrid_split
from .compensate import LoopGainUpdater, _margins_of, _numeric_A
from .gft import _probe_indices


def zeta_from_pm(pm_deg: float) -> float:
    """Second-order damping implied by a phase margin (exact relation
    PM = atan(2 z / sqrt(sqrt(1+4 z^4) - 2 z^2)), inverted by bisection;
    the familiar z ~ PM/100 is its small-z tangent)."""
    pm = np.radians(pm_deg)
    if pm <= 0:
        return 0.0

    def pm_of(z):
        return np.arctan2(2 * z, np.sqrt(np.sqrt(1 + 4 * z ** 4)
                                         - 2 * z ** 2))
    lo, hi = 0.0, 5.0
    if pm >= pm_of(hi):
        return hi
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if pm_of(mid) < pm:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


@dataclass
class DeviceVisibility:
    """Per-device elasticities: relative pole shift and relative loop-gain
    shift per unit relative gm change (dimensionless)."""
    name: str
    pole_elasticity: float
    t_elasticity: float

    @property
    def unobserved(self) -> bool:
        """Moves the closed-loop poles but is (nearly) invisible to T."""
        return (self.pole_elasticity > 0.02
                and self.t_elasticity < 0.02 * self.pole_elasticity)


@dataclass
class GftCheck:
    """Design-meaningfulness of the probe's GFT designation (Montagne's
    axis, plan Sec. 9 detector 3): is Hinf the intended ideal transfer,
    and does direct feedthrough stay out of band?"""
    hinf_lf: complex                    # Hinf at the lowest grid point
    hinf_flat_dev: float                # max |Hinf/Hinf_lf - 1| in band
    feedthrough_crossover_hz: float | None
    identity_residual: float            # exact-rational; 0.0 = exact

    def notes(self) -> list[str]:
        out = []
        if self.identity_residual != 0.0:
            out.append("GFT designation INCONSISTENT: identity residual "
                       f"{self.identity_residual:.2e} (error signal does "
                       "not straddle the probe?)")
        if self.hinf_flat_dev > 0.05:
            out.append(f"Hinf deviates {self.hinf_flat_dev:.1%} from flat "
                       "in band -- the designated ideal is not what the "
                       "loop enforces")
        if self.feedthrough_crossover_hz is not None:
            out.append("direct feedthrough dominates beyond "
                       f"{self.feedthrough_crossover_hz:.3g} Hz -- "
                       "loop-gain shaping cannot change H past there")
        return out


@dataclass
class ProbeReport:
    probe: str
    pm_deg: float | None
    pm_freq_hz: float | None
    gm_db: float | None
    poles_hz: np.ndarray            # dominant closed-loop poles (Hz)
    min_zeta: float                 # worst damping among dominant pairs
    min_zeta_freq_hz: float
    zeta_implied: float | None      # from the PM, second-order
    rhp_poles_hz: np.ndarray
    visibility: list[DeviceVisibility] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    gft_check: "GftCheck | None" = None

    @property
    def unobserved(self) -> list[str]:
        return [v.name for v in self.visibility if v.unobserved]

    @property
    def margins_consistent(self) -> bool:
        """Margins do not overstate the damping of any dominant pole."""
        if self.rhp_poles_hz.size:
            return False
        if self.zeta_implied is None:
            return False
        return self.min_zeta >= 0.5 * min(self.zeta_implied, 1.0)

    def verdict(self) -> str:
        if self.pm_deg is None:
            return "no unity crossing found -- probe sees no loop here"
        parts = []
        if self.rhp_poles_hz.size:
            parts.append(f"UNSTABLE: {self.rhp_poles_hz.size} RHP pole(s) "
                         "while the probe reports margins")
        elif not self.margins_consistent:
            parts.append(
                f"MISLEADING: PM {self.pm_deg:.1f} deg implies zeta "
                f"~{self.zeta_implied:.2f} but a closed-loop pair at "
                f"{self.min_zeta_freq_hz:.3g} Hz has zeta {self.min_zeta:.2f}")
        else:
            parts.append(f"margins consistent (PM {self.pm_deg:.1f} deg, "
                         f"worst closed-loop zeta {self.min_zeta:.2f})")
        if self.unobserved:
            parts.append("unobserved loop dynamics via: "
                         + ", ".join(self.unobserved)
                         + " -- probe these separately (Hurst: valid "
                         "only as stable embedded/other-mode loops)")
        else:
            parts.append("common break: every strong pole-mover is "
                         "visible to this probe")
        if self.gft_check is not None:
            parts.extend(self.gft_check.notes())
        return "; ".join(parts)


def _dominant_poles(A_v: sp.Matrix, f_lo=1.0, f_hi=1e11, n=12):
    """Closed-loop natural frequencies from det A, windowed and sorted by
    magnitude (rad/s)."""
    den = sp.Poly(_det(A_v), S)
    coeffs = [complex(c) for c in den.all_coeffs()]
    scale = max(abs(c) for c in coeffs if c != 0)
    roots = np.roots([c / scale for c in coeffs])
    w = np.abs(roots)
    keep = (w > 2 * np.pi * f_lo) & (w < 2 * np.pi * f_hi)
    roots = roots[keep]
    return roots[np.argsort(np.abs(roots))][:n]


def _gft_check(system, A_v, probe, gft, freqs) -> "GftCheck":
    """Numeric quartet on the band + exact identity at rational points
    (nested_gft machinery)."""
    from .nested_gft import (_exact_A, _node, _num_quartet, _point_quartet,
                             _residual_of)

    inp_z, out, error = gft["z_in"], gft["out"], gft["error"]
    io = _node(system, out)
    pr = _probe_indices(system, probe)
    err = (_node(system, error[0]), int(error[1]))
    z_pr = np.zeros(A_v.rows); z_pr[pr[2]] = 1.0
    fn = sp.lambdify(S, A_v, "numpy")
    z_in = np.asarray(inp_z, dtype=complex).ravel()
    qs = [_num_quartet(np.asarray(fn(2j * np.pi * f), complex),
                       z_in, z_pr, io, pr, err) for f in freqs]
    Hinf = np.array([q["Hinf"] for q in qs])
    T = np.array([q["T"] for q in qs])
    H0 = np.array([q["H0"] for q in qs])
    flat = float(np.max(np.abs(Hinf / Hinf[0] - 1)))
    loop_part = np.abs(Hinf * T / (1 + T))
    ft_part = np.abs(H0 / (1 + T))
    k = np.where(ft_part > loop_part)[0]
    ft = float(freqs[k[0]]) if k.size else None

    worst = 0.0
    zin_m = sp.Matrix(inp_z)
    zpr_m = sp.zeros(A_v.rows, 1); zpr_m[pr[2], 0] = 1
    for sv in (2, 3):
        A0 = A_v.xreplace({S: sp.Rational(sv)})
        q = _point_quartet(A0, zin_m, zpr_m, io, pr, err)
        r = _residual_of(q)
        if r != 0:
            worst = max(worst, abs(float((r / q["H"]).evalf())))
    return GftCheck(hinf_lf=complex(Hinf[0]), hinf_flat_dev=flat,
                    feedthrough_crossover_hz=ft,
                    identity_residual=worst)


def assess_probe(system: MnaSystem, probe: str, *,
                 freqs=None, eps: float = 0.02,
                 n_poles: int = 12, gft=None) -> ProbeReport:
    """Grade the probe (a 0-V branch) against the exact closed loop.

    gft (optional): {"z_in": <input z vector>, "out": node,
    "error": (ref_node, c), "band": (f_lo, f_hi)} adds the
    design-meaningfulness check (Montagne's axis) via the GFT quartet
    at this probe."""
    if probe not in system.branch_index:
        raise MnaError(f"probe {probe!r} is not a voltage-defined branch")
    if freqs is None:
        freqs = np.logspace(2, 9, 281)
    freqs = np.asarray(freqs, dtype=float)

    A_v = _numeric_A(system)

    # ---- probe view: T on the grid, margins
    upd = LoopGainUpdater(system, probe, freqs)
    T = upd.baseline()
    pm, fu, gmdb = _margins_of(freqs, T)

    # ---- truth view: dominant closed-loop poles
    poles = _dominant_poles(A_v, n=n_poles)
    rhp = poles[poles.real > 1e-3 * np.abs(poles)]
    zmin, fz = 1.0, 0.0
    for p in poles:
        if abs(p.imag) > 0.01 * abs(p):
            z = float(-p.real / abs(p))
            if z < zmin:
                zmin, fz = z, float(abs(p) / (2 * np.pi))
    z_impl = None if pm is None else zeta_from_pm(pm)

    # ---- per-device visibility: elasticity of poles vs elasticity of T
    gm_names = [k for k in system.values if k.startswith("gm_")]
    subs, _ = hybrid_split(system, [])
    # stamp pattern of each gm (constant matrices; A is multilinear)
    stamps = {}
    for name in gm_names:
        sym = system.symbols[name]
        stamps[name] = np.array(
            system.A.diff(sym).xreplace(subs).xreplace(
                {S: sp.Integer(0)}).tolist(), dtype=complex)
    Q = np.array(A_v.diff(S).tolist(), dtype=complex)

    # left/right null vectors per dominant pole (cap_pole_screen pattern)
    P0 = np.array(A_v.xreplace({S: sp.Integer(0)}).tolist(), dtype=complex)
    pole_sens = {name: 0.0 for name in gm_names}
    for p in poles:
        M = P0 + p * Q
        U, sv, Vh = np.linalg.svd(M)
        v = Vh[-1].conj()
        w = U[:, -1]
        denom = np.vdot(w, Q @ v)
        if abs(denom) == 0:
            continue
        for name in gm_names:
            dp = -np.vdot(w, stamps[name] @ v) / denom
            el = abs(dp) * abs(system.values[name]) / abs(p)
            if el > pole_sens[name]:
                pole_sens[name] = float(el)

    # T elasticity: finite difference at a few in-band points around the
    # crossover (direct solves; stamps are tiny updates of A(s))
    if fu is not None:
        fpts = np.geomspace(max(fu / 100, freqs[0]),
                            min(fu * 10, freqs[-1]), 6)
    else:
        fpts = np.geomspace(freqs[0], freqs[-1], 6)
    fn = sp.lambdify(S, A_v, "numpy")
    dim = A_v.rows
    ib = system.branch_index[probe]
    ip = upd.ip
    zv = np.zeros(dim, complex); zv[ib] = 1.0
    zi = np.zeros(dim, complex); zi[ip] = 1.0
    base_T = {}
    Ms = {}
    for f in fpts:
        M = np.asarray(fn(2j * np.pi * f), complex)
        Ms[f] = M
        base_T[f] = LoopGainUpdater._tian(np.linalg.solve(M, zv),
                                          np.linalg.solve(M, zi), ip, ib)
    vis = []
    for name in gm_names:
        dM = eps * system.values[name] * stamps[name]
        rel = 0.0
        for f in fpts:
            M2 = Ms[f] + dM
            t2 = LoopGainUpdater._tian(np.linalg.solve(M2, zv),
                                       np.linalg.solve(M2, zi), ip, ib)
            t0 = base_T[f]
            if abs(t0) > 0:
                rel = max(rel, abs(t2 - t0) / abs(t0))
        vis.append(DeviceVisibility(
            name=name[3:], pole_elasticity=pole_sens[name],
            t_elasticity=float(rel / eps)))
    vis.sort(key=lambda d: -d.pole_elasticity)

    check = None
    if gft is not None:
        lo, hi = gft.get("band", (freqs[0], (fu or freqs[-1])))
        check = _gft_check(system, A_v, probe, gft,
                           np.geomspace(lo, hi, 25))

    return ProbeReport(
        probe=probe, pm_deg=pm, pm_freq_hz=fu, gm_db=gmdb,
        poles_hz=poles / (2 * np.pi), min_zeta=zmin, min_zeta_freq_hz=fz,
        zeta_implied=z_impl, rhp_poles_hz=rhp / (2 * np.pi),
        visibility=vis, gft_check=check)
