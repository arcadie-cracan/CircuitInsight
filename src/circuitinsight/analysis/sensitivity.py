"""Operating-point sensitivity ranking and keep-set suggestion.

Ranks every circuit parameter by the relative differential sensitivity of a
target quantity (dc gain or a chosen pole), evaluated at the true operating
point — and proposes the top-M as the `keep` list for a symbolic solve, in
the spirit of sensitivity-guided symbolic pole/zero extraction
(Gheorghe & Constantinescu, Mathematics 2025).

Method (no finite differences, no extra determinants):
- the MNA matrix A is AFFINE in each parameter x and in s, so
  dA/dx (= the element's stamp pattern, via sympy diff) is exact;
- pole sensitivity: for a simple root p of det A(s), with left/right null
  vectors w, v of A(p),   dp/dx = -(w^H (dA/dx) v) / (w^H (dA/ds) v);
- dc-gain sensitivity: Jacobi's formula at s=0 on the Cramer pair,
  d ln det(M)/dx = tr(M^{-1} dM/dx).

Float arithmetic throughout: this is a *ranking heuristic* that feeds the
exact symbolic machinery; the solves it suggests remain exact.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import sympy as sp

from ..engine.mna import MnaError, S, _det
from ..units import eng
from .criteria import make_criterion


def _np(mat: sp.Matrix, dtype=float) -> np.ndarray:
    return np.array(mat.tolist(), dtype=dtype)


@dataclass
class SensitivityReport:
    dc_gain: list[tuple[str, float]]          # (name, S_rel), |S| descending
    poles: np.ndarray                         # complex, Hz, |p| ascending
    pole_sens: list[list[tuple[str, float]]]  # per pole, |S| descending

    def suggest_keep(self, target: str = "dc_gain", top: int = 6) -> list[str]:
        """target: 'dc_gain' or 'p1', 'p2', ... (1-based, |p| ascending)."""
        if target == "dc_gain":
            ranked = self.dc_gain
        elif target.startswith("p") and target[1:].isdigit():
            ranked = self.pole_sens[int(target[1:]) - 1]
        else:
            raise ValueError(f"unknown target {target!r}")
        return [name for name, _ in ranked[:top]]

    def report(self, top: int = 8) -> str:
        lines = ["dc-gain sensitivities:"]
        for n, s_ in self.dc_gain[:top]:
            lines.append(f"  {n:24s} S = {s_:+.3f}")
        for i, (p, sens) in enumerate(zip(self.poles, self.pole_sens), 1):
            lines.append(f"p{i} = {eng(p, 'Hz')}:")
            for n, s_ in sens[:top]:
                lines.append(f"  {n:24s} S = {s_:+.3f}")
        return "\n".join(lines)


def sensitivities(analyzer, inp: str, out: str,
                  n_poles: int = 3) -> SensitivityReport:
    system = analyzer.system(inp)
    if out not in system.node_index:
        raise MnaError(f"output node {out!r} not found (or it is ground)")
    col = system.node_index[out]

    subs, missing = {}, []
    for name, symb in system.symbols.items():
        v = system.values.get(name)
        (missing.append(name) if v is None
         else subs.__setitem__(symb, sp.Rational(repr(v))))
    if missing:
        raise MnaError(f"sensitivities: no numeric value for {sorted(missing)}")

    A_v = system.A.xreplace(subs)             # numeric except s
    Q = _np(A_v.diff(S))
    P = _np(A_v.xreplace({S: sp.Integer(0)}))
    Ak_v = A_v.copy()
    Ak_v[:, col] = system.z
    Pk = _np(Ak_v.xreplace({S: sp.Integer(0)}))

    # parameter stamp derivatives, exact via sympy diff (A affine in each x)
    params = [(n, system.symbols[n], float(system.values[n]))
              for n in system.symbols if system.values.get(n)]
    stamps = []
    for name, symb, x0 in params:
        B = system.A.diff(symb).xreplace(subs)
        BP = _np(B.xreplace({S: sp.Integer(0)}))
        BQ = _np(B.diff(S))
        stamps.append((name, x0, BP, BQ))

    # ---------------- dc gain: Jacobi's formula on the Cramer pair --------
    dc = []
    Pinv_ok = Pkinv_ok = True
    try:
        P_lu = np.linalg.inv(P)
        Pk_lu = np.linalg.inv(Pk)
    except np.linalg.LinAlgError:              # pragma: no cover
        Pinv_ok = Pkinv_ok = False
    if Pinv_ok and Pkinv_ok:
        for name, x0, BP, _ in stamps:
            BPk = BP.copy()
            BPk[:, col] = 0.0                  # the z column is constant
            s_rel = x0 * (np.trace(Pk_lu @ BPk) - np.trace(P_lu @ BP))
            dc.append((name, float(s_rel)))
    dc.sort(key=lambda t: -abs(t[1]))

    # ---------------- poles: null-vector perturbation formula -------------
    den = sp.Poly(_det(A_v), S)
    roots = np.roots([complex(c) for c in den.all_coeffs()])
    roots = roots[np.argsort(np.abs(roots))]
    roots = roots[np.abs(roots) > 0][:n_poles]

    pole_sens = []
    for p in roots:
        M = P + p * Q
        # left/right null vectors from the smallest singular triplet
        U, sv, Vh = np.linalg.svd(M)
        v = Vh[-1].conj()
        w = U[:, -1]
        denom = np.vdot(w, Q @ v)
        sens = []
        for name, x0, BP, BQ in stamps:
            Bx = BP + p * BQ
            dp = -np.vdot(w, Bx @ v) / denom
            d_abs = (np.conj(p) * dp).real / abs(p)
            sens.append((name, float(x0 * d_abs / abs(p))))
        sens.sort(key=lambda t: -abs(t[1]))
        pole_sens.append(sens)

    return SensitivityReport(
        dc_gain=dc,
        poles=roots / (2 * np.pi),
        pole_sens=pole_sens,
    )


# ============================ band-sampled sensitivity ====================
# Rank parameters by how much they move the transfer function across the
# frequency band, evaluated with Jacobi's formula at a handful of
# feature-aligned frequency points -- all numeric, no symbolic determinant.
# This is the recommended ranking for keep-set selection: it is valid over
# the whole band (not just s=0, which a feedback rig desensitizes) and far
# cheaper than the pole path (no characteristic polynomial).

@dataclass
class BandSensitivity:
    ranking: list[tuple[str, float]]   # (name, score) descending, max over band
    freqs: np.ndarray                  # sample frequencies used (Hz)
    poles: np.ndarray                  # numeric pole frequencies (Hz)
    zeros: np.ndarray                  # numeric zero frequencies (Hz)
    metric: str                        # 'complex' or 'magnitude'
    symbols: list[str]                 # row order of `matrix`
    matrix: np.ndarray                 # [symbol, frequency] score |H|*|sens|

    def _scores(self, fmin, fmax) -> np.ndarray:
        mask = np.ones(len(self.freqs), dtype=bool)
        if fmin is not None:
            mask &= self.freqs >= fmin
        if fmax is not None:
            mask &= self.freqs <= fmax
        if not mask.any():
            mask[:] = True
        return self.matrix[:, mask].max(axis=1)

    def rank(self, fmin: float | None = None,
             fmax: float | None = None) -> list[tuple[str, float]]:
        """Ranking restricted to a frequency sub-range -- a symbol's
        importance is frequency-dependent (a compensation cap dominates the
        band edge, not the passband), so target where it matters."""
        scores = self._scores(fmin, fmax)
        return sorted(zip(self.symbols, scores), key=lambda t: -t[1])

    def suggest_keep(self, top: int = 6, fmin: float | None = None,
                     fmax: float | None = None) -> list[str]:
        return [n for n, _ in self.rank(fmin, fmax)[:top]]

    def peak_frequency(self, name: str) -> float:
        """Frequency at which `name` most influences the response."""
        i = self.symbols.index(name)
        return float(self.freqs[int(np.argmax(self.matrix[i]))])

    def report(self, top: int = 12) -> str:
        lines = [f"band sensitivity ({self.metric}), "
                 f"{len(self.freqs)} sample points "
                 f"{self.freqs.min():.3g}..{self.freqs.max():.3g} Hz "
                 f"(score, and where each peaks):"]
        for n, s_ in self.ranking[:top]:
            lines.append(f"  {n:24s} {s_:.4g}   peaks @ "
                         f"{self.peak_frequency(n):.3g} Hz")
        return "\n".join(lines)


def _stamp_derivatives(system, subs):
    """Per-parameter dA/dx at the operating point, split into its s^0 and
    s^1 parts (BP, BQ). One pass over the matrix entries -- much cheaper
    than differentiating the whole matrix once per parameter, since each
    entry involves only a few symbols."""
    n = system.A.shape[0]
    known = set(system.symbols.values())
    rev = {sym: name for name, sym in system.symbols.items()}
    BP: dict[str, np.ndarray] = {}
    BQ: dict[str, np.ndarray] = {}
    for i in range(n):
        for j in range(n):
            e = system.A[i, j]
            for sym in e.free_symbols & known:
                name = rev[sym]
                if name not in system.values:
                    continue
                if name not in BP:
                    BP[name] = np.zeros((n, n), dtype=complex)
                    BQ[name] = np.zeros((n, n), dtype=complex)
                de = e.diff(sym).xreplace(subs)
                BP[name][i, j] = complex(de.xreplace({S: sp.Integer(0)}))
                c1 = de.diff(S)
                if c1 != 0:
                    BQ[name][i, j] = complex(c1)
    return [(name, float(system.values[name]), BP[name], BQ[name])
            for name in BP]


def _critical_freqs(Pm: np.ndarray, Qm: np.ndarray) -> np.ndarray:
    """Frequencies (Hz) where det(Pm + s*Qm) = 0, via the numpy-only pencil
    reduction: (Pm + s Qm)v = 0  =>  (Pm^-1 Qm) v = -(1/s) v, so the finite
    critical s are -1/eig(Pm^-1 Qm). No scipy / generalized eig needed."""
    try:
        M = np.linalg.solve(Pm, Qm)
    except np.linalg.LinAlgError:
        return np.array([])
    mu = np.linalg.eigvals(M)
    mu = mu[np.abs(mu) > 1e-18]
    if mu.size == 0:
        return np.array([])
    s = -1.0 / mu
    f = np.abs(s) / (2 * np.pi)
    return np.sort(f[np.isfinite(f) & (f > 0)])


def _sample_freqs(features, fmin, fmax, per_decade=4):
    """Feature-aligned sample points: each critical frequency plus half a
    decade either side, clamped to [fmin, fmax] and de-duplicated. Falls
    back to a log grid when there are no reactive features."""
    lo = fmin if fmin else (features.min() if features.size else 1.0)
    hi = fmax if fmax else (features.max() if features.size else 1e6)
    pts = set()
    for fc in features:
        for m in (10 ** -0.5, 1.0, 10 ** 0.5):
            pts.add(fc * m)
    pts.update((lo, hi))
    if not features.size:                     # purely resistive-ish: log grid
        ndec = max(1, np.log10(hi / lo))
        pts.update(np.logspace(np.log10(lo), np.log10(hi),
                               int(per_decade * ndec) + 1))
    arr = np.array(sorted(f for f in pts if lo <= f <= hi))
    keep, last = [], -1.0
    for f in arr:                             # thin to >=0.1 decade apart
        if last < 0 or np.log10(f / last) > 0.1:
            keep.append(f)
            last = f
    return np.array(keep) if keep else np.array([np.sqrt(lo * hi)])


def _band_core(analyzer, inp: str, out: str, fmin, fmax,
               freqs=None) -> dict:
    """Shared numeric machinery for the band analyses: sample frequencies
    (feature-aligned by default, or the supplied `freqs`), the response
    H(f), and the complex sensitivity atoms dlnH[name][f] = d ln H / d ln x
    at each frequency. All numeric."""
    system = analyzer.system(inp)
    if out not in system.node_index:
        raise MnaError(f"output node {out!r} not found (or it is ground)")
    col = system.node_index[out]

    subs, missing = {}, []
    for name, symb in system.symbols.items():
        v = system.values.get(name)
        (missing.append(name) if v is None
         else subs.__setitem__(symb, sp.Rational(repr(v))))
    if missing:
        raise MnaError(f"band analysis: no numeric value for {sorted(missing)}")

    params = _stamp_derivatives(system, subs)
    A_v = system.A.xreplace(subs)
    P = _np(A_v.xreplace({S: sp.Integer(0)}), dtype=complex)
    Q = _np(A_v.diff(S), dtype=complex)
    z = np.array(system.z.T.tolist()[0], dtype=complex)

    Pk = P.copy(); Pk[:, col] = z
    Qk = Q.copy(); Qk[:, col] = 0.0
    poles = _critical_freqs(P, Q)
    zeros = _critical_freqs(Pk, Qk)
    if freqs is None:
        freqs = _sample_freqs(np.concatenate([poles, zeros]), fmin, fmax)
    freqs = np.asarray(freqs, dtype=float)

    names = [p[0] for p in params]
    nf = len(freqs)
    H = np.zeros(nf, dtype=complex)
    atoms = {n: np.zeros(nf, dtype=complex) for n in names}
    valid = np.ones(nf, dtype=bool)
    for j, f in enumerate(freqs):
        s = 2j * np.pi * f
        M = P + s * Q
        Mk = M.copy(); Mk[:, col] = z
        try:
            Minv = np.linalg.inv(M)
            Mkinv = np.linalg.inv(Mk)
        except np.linalg.LinAlgError:          # exactly on a feature; skip
            valid[j] = False
            continue
        H[j] = (Minv @ z)[col]
        for name, x0, BP, BQ in params:
            Bx = BP + s * BQ
            Bxk = Bx.copy(); Bxk[:, col] = 0.0
            atoms[name][j] = x0 * (np.trace(Mkinv @ Bxk) - np.trace(Minv @ Bx))
    stamps = {name: (x0, BP, BQ) for name, x0, BP, BQ in params}
    return dict(system=system, col=col, subs=subs, freqs=freqs, H=H,
                atoms=atoms, names=names, poles=poles, zeros=zeros, valid=valid,
                P=P, Q=Q, z=z, stamps=stamps)


def band_sensitivity(analyzer, inp: str, out: str, metric: str = "complex",
                     fmin: float | None = None, fmax: float | None = None
                     ) -> BandSensitivity:
    """Rank parameters by their influence on the transfer function across
    the band. metric='complex' (default) ranks by the whole complex TF
    sensitivity |d ln H / d ln x| (captures magnitude AND phase);
    metric='magnitude' ranks by |Re(...)| (the |H|-in-dB sensitivity only).
    fmin/fmax bound the band of interest (default: span the reactive
    features). Each sample is weighted by |H|, so frequencies where the
    response is negligible contribute nothing.

    The result carries the full [symbol, frequency] score matrix; the
    top-level ranking is the max over the band, but .rank(fmin, fmax) and
    .suggest_keep(fmin, fmax) target a sub-range, since a symbol's
    importance is frequency-dependent."""
    if metric not in ("complex", "magnitude"):
        raise ValueError(f"metric must be 'complex' or 'magnitude', got {metric!r}")
    core = _band_core(analyzer, inp, out, fmin, fmax)
    freqs, H, atoms, names = (core["freqs"], core["H"], core["atoms"],
                              core["names"])

    mat = np.zeros((len(names), len(freqs)))
    for i, name in enumerate(names):
        a = atoms[name]
        sval = np.abs(a) if metric == "complex" else np.abs(a.real)
        mat[i] = np.abs(H) * sval
    mat[~np.isfinite(mat)] = 0.0

    scores = mat.max(axis=1)
    ranking = sorted(zip(names, scores), key=lambda t: -t[1])
    return BandSensitivity(ranking=ranking, freqs=freqs, poles=core["poles"],
                           zeros=core["zeros"], metric=metric, symbols=names,
                           matrix=mat)


# ======================= reactive-element selection ========================
# "Throw away all reactances, add them back": EXACT greedy -- every round
# re-solves the response with each remaining candidate restored and keeps
# the best. (A first-order matching pursuit was tried first and rejected:
# the linear atom is unreliable exactly for the dominant, pole-setting
# elements; see the dominant_reactances docstring.)

@dataclass
class ReactanceReduction:
    selected: list[str]          # reactive symbols, in add-back order
    errors_db: list[float]       # band error after each addition (path[k])
    baseline_db: float           # error with NO reactances active
    tol_db: float
    freqs: np.ndarray
    metric: str
    floor_db: float = 60.0       # the enforcement level (see floor_is_abs)
    floor_is_abs: bool = False   # True: dB absolute; False: dB below peak
    floor_eff_db: float | None = None   # the level actually enforced to
                                        # (the stated floor minus tol_db)
    sig_lo: float = 0.0          # the band actually ENFORCED: where |H|
    sig_hi: float = 0.0          # stays within floor_db of its peak
    eps: float | None = None     # anchored mode: |dH| <= eps*(|H|+anchor)
    anchor: float | None = None  # min |H| at the band edges
    strategy: str | None = None  # 'plain' | 'stability' | 'rejection'
    met: bool = True             # tolerance reached within the order cap
    metrics: dict = field(default_factory=dict)   # strategy readouts

    def report(self) -> str:
        if self.strategy is not None:
            final = (self.errors_db[-1] if self.errors_db
                     else self.baseline_db)
            lines = [f"reactance reduction (strategy: {self.strategy}): "
                     f"score {self.baseline_db:.3g} with no reactances "
                     f"(normalized; <= 1 is within budget)"]
            lines.append(f"  band {eng(self.sig_lo, 'Hz')}-"
                         f"{eng(self.sig_hi, 'Hz')}")
            for name, e in zip(self.selected, self.errors_db):
                lines.append(f"  + {name:22s} -> {e:.3g}")
            if not self.selected:
                lines.append("  (already within tolerance; no reactance "
                             "needed)")
            elif not self.met:
                lines.append(f"  order cap reached at {final:.3g}x the "
                             f"budget")
            return '\n'.join(lines)
        if self.eps is not None:
            lines = [f"reactance reduction (anchored, eps {self.eps:.1%}, "
                     f"anchor {20 * math.log10(self.anchor):.1f} dB): "
                     f"{self.baseline_db:.3g} with no reactances"]
            lines.append(f"  band {eng(self.sig_lo, 'Hz')}-"
                         f"{eng(self.sig_hi, 'Hz')}")
            for name, e in zip(self.selected, self.errors_db):
                lines.append(f"  + {name:22s} -> {e:.3g}")
            if not self.selected:
                lines.append("  (already within tolerance; no reactance "
                             "needed)")
            return "\n".join(lines)
        lines = [f"reactance reduction ({self.metric}, tol {self.tol_db:g} dB): "
                 f"{self.baseline_db:.3g} dB with no reactances"]
        if self.sig_hi:
            how = (f"|H| >= {self.floor_db:g} dB" if self.floor_is_abs
                   else f"|H| within {self.floor_db:g} dB of peak")
            lines.append(f"  enforced over {eng(self.sig_lo, 'Hz')}-"
                         f"{eng(self.sig_hi, 'Hz')} ({how})")
        for name, e in zip(self.selected, self.errors_db):
            lines.append(f"  + {name:22s} -> {e:.3g} dB")
        if not self.selected:
            lines.append("  (already within tolerance; no reactance needed)")
        return "\n".join(lines)


def loop_margins(freqs, h):
    """(pm_deg, pm_freq, gm_db, gm_freq) from a gain sweep in the stb
    convention (phase +180 deg at DC). Entries are None when the
    respective crossing is not in the band. THE margin extraction: the
    session's loop-gain reporting, the Stability criterion, and (via
    compensate._margins_of) the compensation, modes and probe-adequacy
    advisors all read margins through this function."""
    f = np.asarray(freqs, dtype=float)
    m = 20 * np.log10(np.abs(h))
    ph = np.degrees(np.unwrap(np.angle(h)))
    x = np.log10(f)
    pm = fpm = gm = fgm = None
    k = np.where(np.diff(np.sign(m)))[0]
    if k.size:
        k = k[0]
        xu = np.interp(0, [m[k + 1], m[k]], [x[k + 1], x[k]])
        pm = float(np.interp(xu, [x[k], x[k + 1]], [ph[k], ph[k + 1]]))
        fpm = float(10 ** xu)
    j = np.where(np.diff(np.sign(ph)))[0]
    if j.size:
        j = j[0]
        xj = np.interp(0, [ph[j + 1], ph[j]], [x[j + 1], x[j]])
        gm = float(-np.interp(xj, [x[j], x[j + 1]], [m[j], m[j + 1]]))
        fgm = float(10 ** xj)
    return pm, fpm, gm, fgm


def _reactive_symbols(analyzer) -> set[str]:
    from ..engine.mna import symbol_name
    return {symbol_name(p, analyzer._alias)
            for p in analyzer.primitives if p.kind in ("c", "cx", "l")}


def dominant_reactances(analyzer, inp: str, out: str, tol_db: float = 1.0,
                        fmin: float | None = None, fmax: float | None = None,
                        metric: str = "complex",
                        exclude: tuple[str, ...] = (),
                        max_elements: int | None = None,
                        floor_db: float = 60.0,
                        floor_abs_db: float | None = None,
                        phase_tol_deg: float | None = None,
                        eps: float | None = None,
                        strategy: str | None = None,
                        gain_db: float = 1.0, phase_deg: float = 5.0,
                        pm_deg: float = 5.0, gm_db: float = 2.0,
                        rej_db: float = 3.0, fc_rel: float = 0.10,
                        cap: int = 6,
                        progress=None, note=None) -> ReactanceReduction:
    """The minimal set of reactive elements that reproduces H(f) over the
    band: start with every reactance removed and greedily add the one whose
    inclusion most reduces the Bode-magnitude band error, stopping within
    tol_db. Answers 'which caps actually shape this response?'.

    Selection is *exact* greedy, not the linear-sensitivity matching pursuit
    first considered: the first-order atom is unreliable for exactly the
    dominant, pole-setting elements (e.g. a Miller cap), so it can rank the
    single most important cap below the verification cutoff. Reactances are
    removed cheaply and exactly by subtracting their s-linear stamp from Q
    (a cap -> open, an inductance -> short), so trying every candidate each
    round is fast -- no re-solve of the symbolic system.

    metric='magnitude' scores |H| in dB; 'complex' also requires the phase
    to track (max of the dB error and phase/10). `exclude` names reactances
    kept always-on (e.g. a DC-feedback/AC-open measurement rig).

    eps (the ANCHORED mode, superseding the floor/budget knobs): accept
    when |H_red - H_full| <= eps * (|H_full| + anchor) at every band
    point, anchor = the smaller band-edge |H|. One dimensionless number
    bounds magnitude (~8.7*eps dB) and phase (~57*eps deg) jointly above
    the anchor and forgives the response below it -- the band placement
    IS the statement of where fidelity matters. floor/phase/metric are
    ignored in this mode.

    strategy (the TOLERANCE STRATEGIES; each gates on the quantity its
    designer actually reads, and every one is CAPPED at `cap` elements
    -- an unreadable model is a failed answer, reported with met=False
    and the achieved numbers, never dumped):
      'plain'      -- gain_db and phase_deg, each on its own axis at
                      every band point. Spec-sheet contract; trusts the
                      cursor literally (dB error is scale-free).
      'stability'  -- the reduced model must reproduce the full model's
                      margins: PM within pm_deg, GM within gm_db, the
                      unity crossing within fc_rel. The band's only job
                      is to contain the crossover. metrics records
                      full/reduced PM, GM, fc.
      'rejection'  -- the dB curve tracks within rej_db at every band
                      point, phase unconstrained: for CMRR/PSRR the
                      curve IS the deliverable and dB error is relative
                      error of the small quantity."""
    if fmin is not None and fmax is not None:
        lo, hi = fmin, fmax
    else:
        span = _band_core(analyzer, inp, out, fmin, fmax)["freqs"]
        lo = fmin if fmin is not None else float(span.min())
        hi = fmax if fmax is not None else float(span.max())
    ndec = max(1.0, math.log10(hi / lo))
    dense = np.logspace(math.log10(lo), math.log10(hi), int(8 * ndec) + 1)
    core = _band_core(analyzer, inp, out, fmin, fmax, freqs=dense)
    freqs, H_full, valid = core["freqs"], core["H"], core["valid"]
    P, Q, z, col, stamps = (core["P"], core["Q"], core["z"], core["col"],
                            core["stamps"])

    all_react = [n for n in _reactive_symbols(analyzer) if n in stamps]
    excl = {n for n in all_react
            if n in exclude or any(n == e or n.endswith("_" + e)
                                   for e in exclude)}
    react = sorted(n for n in all_react if n not in excl)
    # each reactance's contribution to the s-linear part: value * dQ/dvalue
    contrib = {c: stamps[c][0] * stamps[c][2] for c in all_react}
    w = 2j * np.pi * freqs

    # Q with every selectable reactance removed (excluded ones stay in)
    Q_base = Q.copy()
    for c in react:
        Q_base -= contrib[c]

    def response(Q_eff: np.ndarray) -> np.ndarray:
        out = np.empty(len(freqs), dtype=complex)
        for j, wj in enumerate(w):
            try:
                out[j] = np.linalg.solve(P + wj * Q_eff, z)[col]
            except np.linalg.LinAlgError:
                out[j] = np.nan
        return out

    m = valid & np.isfinite(H_full) & (np.abs(H_full) > 0)
    mag_full = np.abs(H_full)
    # ONE criterion object carries the whole tolerance contract -- the
    # enforcement window, the error score, the stopping rule and (in
    # the session) the report language. See analysis/criteria.py.
    crit = make_criterion(strategy=strategy,
                          strategy_opts={"gain_db": gain_db,
                                         "phase_deg": phase_deg,
                                         "pm_deg": pm_deg, "gm_db": gm_db,
                                         "rej_db": rej_db,
                                         "fc_rel": fc_rel},
                          eps=eps, tol_db=tol_db, metric=metric,
                          floor_db=floor_db, floor_abs_db=floor_abs_db,
                          phase_tol_deg=phase_tol_deg, cap=cap)
    # the ENFORCEMENT WINDOW is the criterion's own statement of where
    # its budget applies -- but the enforced sub-band is reported, never
    # silently narrower than the band the user asked for (that silence
    # once certified a 1-pole model at 0.6 dB over a band it visibly
    # left by 25 dB)
    sig = crit.window(freqs, mag_full, m, H_full)
    if not sig.any():
        sig = m
    anchor = getattr(crit, "anchor", None)
    crit.prepare(freqs, H_full, sig)

    def err(Hr: np.ndarray) -> float:
        return crit.error(freqs, H_full, mag_full, Hr, sig)

    tol = crit.tol
    if crit.cap is not None and max_elements is None:
        max_elements = crit.cap       # an unreadable model is a failure

    Qsel = Q_base.copy()
    Hr = response(Qsel)
    baseline = err(Hr)
    unit = crit.unit
    if note is not None:
        note(f"baseline error {baseline:.3g}{unit} with no reactances; "
             f"tolerance {tol:g}{unit}, {len(react)} candidates")
    selected: list[str] = []
    errors: list[float] = []
    remaining = set(react)
    # per-candidate progress: the number of ROUNDS is unknown up front
    # (the pursuit stops when the tolerance is met), so the reported
    # total is done + the current round's remainder -- a lower bound
    # that grows honestly when another round begins
    ev_done = 0

    def track():
        nonlocal ev_done
        ev_done += 1
        if progress is not None:
            progress(ev_done, ev_done + len(remaining))

    while remaining and (max_elements is None or len(selected) < max_elements):
        if baseline <= tol and not selected:
            break
        cur = err(Hr)
        best, best_err = None, cur
        for c in remaining:                     # exact greedy over all
            e = err(response(Qsel + contrib[c]))
            track()
            if e < best_err - 1e-9:
                best, best_err = c, e
        if best is None:                        # nothing improves -> stop
            break
        selected.append(best)
        remaining.discard(best)
        Qsel = Qsel + contrib[best]
        Hr = response(Qsel)
        errors.append(best_err)
        if note is not None:
            fate = ("tolerance met, stopping" if best_err <= tol else
                    f"still above tolerance, re-trying "
                    f"{len(remaining)} candidates")
            note(f"round {len(selected)}: + {best} -> error "
                 f"{best_err:.3g}{unit} ({fate})")
        if best_err <= tol:
            break
    final = errors[-1] if errors else baseline
    metrics = crit.metrics(freqs, Hr, sig)
    met = bool(final <= tol + 1e-12)
    if note is not None and strategy is not None and not met:
        note(f"order cap {max_elements} reached at score {final:.3g} "
             f"(> 1 = tolerance missed): the response needs more order "
             f"in this band — narrow the band or relax the tolerance")
    return ReactanceReduction(selected=selected, errors_db=errors,
                              baseline_db=baseline, tol_db=tol_db,
                              freqs=freqs, metric=metric,
                              floor_db=(floor_db if floor_abs_db is None
                                        else floor_abs_db),
                              floor_eff_db=(None if floor_abs_db is None
                                            else floor_abs_db - abs(tol_db)),
                              floor_is_abs=floor_abs_db is not None,
                              sig_lo=float(freqs[sig].min()),
                              sig_hi=float(freqs[sig].max()),
                              eps=eps, anchor=anchor,
                              strategy=strategy, met=met, metrics=metrics)
