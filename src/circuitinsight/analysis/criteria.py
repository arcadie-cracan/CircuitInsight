"""The band tolerance criteria, consolidated.

Three generations of tolerance contracts accreted across the reduction
machinery — the legacy dB pair with its enforcement floor, the anchored
one-knob ε, and the designer strategies (plain / stability / rejection).
Each was implemented twice: its math inside `dominant_reactances` (the
enforcement window, the error score, the stopping rule) and its language
inside `reduce_solve` (the strip headline, the Summary details, the
score units). Every new strategy paid that double tax, and the halves
drifted — the pm_spin bug lived exactly in that gap.

A BandCriterion is ONE tolerance contract carrying both halves:

- the math: `window` (where the budget applies), `prepare` (reference
  state from the full model, e.g. the margins a stability designer
  gates on), `error` (the normalized score of a candidate model),
  `tol`/`cap` (the stopping rule), `metrics` (the readouts);
- the language: `headline` (the one strip line), `details` (the
  Summary block), `score_fields` (band_score and a genuine-dB
  mag_err_db), `collapse_budgets` (what the coefficient collapse may
  spend), `eps_equivalent` (the certificate's comparable tolerance).

`make_criterion` maps the accreted keyword surface of
`dominant_reactances` / `reduce_solve` onto one object, so the public
API stays exactly as it was.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..units import eng


def _finite(sig, Hr, positive: bool = True):
    ok = sig & np.isfinite(Hr)
    if positive:
        ok = ok & (np.abs(Hr) > 0)
    return ok


def _dmag_db(Hr, mag_full, ok):
    return np.abs(20 * np.log10(np.abs(Hr[ok]) / mag_full[ok]))


def _dphase_deg(Hr, H_full, ok):
    d = np.abs(np.angle(Hr[ok]) - np.angle(H_full[ok]))
    return np.degrees(np.minimum(d, 2 * np.pi - d))


@dataclass
class BandCriterion:
    """Base contract. Subclasses override the math hooks and the
    language hooks; the greedy pursuit and the session report code call
    only this interface."""

    tol: float = 1.0             # threshold in error() units
    unit: str = ""               # narration suffix ("" or " dB")
    cap: int | None = None       # order cap; None = uncapped
    name: str = ""               # '' legacy, 'anchored', or the strategy

    # ---- math -----------------------------------------------------
    def window(self, freqs, mag_full, m, H_full) -> np.ndarray:
        """The enforcement window: where the budget applies. `m` is the
        valid-sample mask; the default enforces the whole band."""
        return m

    def prepare(self, freqs, H_full, sig) -> None:
        """Capture reference state from the FULL model (once)."""

    def error(self, freqs, H_full, mag_full, Hr, sig) -> float:
        raise NotImplementedError

    def metrics(self, freqs, Hr, sig) -> dict:
        return {}

    # ---- language -------------------------------------------------
    def collapse_budgets(self) -> tuple[float, float]:
        """(mag_db, phase_deg) the coefficient collapse may spend."""
        raise NotImplementedError

    def eps_equivalent(self) -> float:
        """The budget as a comparable relative tolerance, for the order
        certificate."""
        raise NotImplementedError

    def score_fields(self, band_err: float,
                     fallback_db: float) -> tuple[float | None, str, float]:
        """(band_score, band_score_unit, mag_err_db). mag_err_db is
        ALWAYS a genuine dB figure; the normalized score, when one
        exists, travels beside it with its unit named."""
        raise NotImplementedError

    def headline(self, red, band_err: float, fmin: float, fmax: float,
                 tol_db: float = 0.0) -> str:
        raise NotImplementedError

    def details(self, red, band_err: float, fmin: float,
                fmax: float) -> list[str]:
        raise NotImplementedError

    # shared fragments ----------------------------------------------
    def _names(self, red) -> str:
        n = len(red.selected)
        return f" [{', '.join(red.selected)}]" if 0 < n <= 3 else ""

    def _kept(self, red) -> str:
        return (f"kept reactances: "
                f"{', '.join(red.selected) or '(none needed)'}")

    def _cap_miss(self, red, band_err: float) -> tuple[str, list[str]]:
        """Headline suffix + detail lines when the order cap was hit
        before the tolerance."""
        if red.met:
            return " (details in Summary)", []
        return (f" — TOLERANCE NOT MET at the order cap: best score "
                f"{band_err:.2g}× the budget; narrow the band or relax "
                f"the tolerance (details in Summary)",
                [f"the order cap ({self.cap}) keeps the model readable "
                 f"— the response genuinely carries more structure in "
                 f"this band than the cap admits"])


@dataclass
class LegacyCriterion(BandCriterion):
    """The original dB-band contract: max |Δ|H|| in dB (optionally with
    phase/10 folded in), enforced where |H| stays within `floor_db` of
    its peak — or above an absolute floor, widened to the ±180°
    crossings when a phase budget is given."""

    tol_db: float = 1.0
    metric: str = "complex"
    floor_db: float = 60.0
    floor_abs_db: float | None = None
    phase_tol_deg: float | None = None

    def __post_init__(self):
        self.tol = self.tol_db
        self.unit = " dB"
        self.name = ""

    def window(self, freqs, mag_full, m, H_full) -> np.ndarray:
        peak = mag_full[m].max() if m.any() else 1.0
        if self.floor_abs_db is not None:
            # ABSOLUTE floor: "enforce wherever |H| is at least X dB".
            # A model allowed to err by tol_db could cross the floor
            # anywhere within tol_db of it, so enforcement reaches that
            # far BELOW the stated level.
            eff = self.floor_abs_db - abs(self.tol_db)
            sig = m & (mag_full >= 10.0 ** (eff / 20.0))
            if self.phase_tol_deg:
                # the gain-margin point lives where the phase crosses
                # ±180°, typically well below unity: include a bounded
                # neighbourhood of each crossing — not every frequency
                # whose phase is near 180, which on a rolloff
                # asymptoting there would sweep in the whole tail.
                ph = np.degrees(np.unwrap(np.angle(H_full)))
                d = np.abs(ph) - 180.0
                cross = np.nonzero(np.diff(np.sign(d)) != 0)[0]
                for i in cross:
                    fc = float(freqs[i])
                    if fc <= 0:
                        continue
                    span = 10.0 ** (0.5 + abs(self.phase_tol_deg) / 180.0)
                    sig = sig | (m & (freqs >= fc / span)
                                 & (freqs <= fc * span))
            return sig
        return m & (mag_full > peak * 10.0 ** (-self.floor_db / 20.0))

    def error(self, freqs, H_full, mag_full, Hr, sig) -> float:
        ok = _finite(sig, Hr)
        if not ok.any():
            return float("inf")
        dmag = _dmag_db(Hr, mag_full, ok)
        if self.metric == "magnitude":
            return float(dmag.max())
        dph = _dphase_deg(Hr, H_full, ok)
        return float(np.maximum(dmag, dph / 10.0).max())

    def collapse_budgets(self) -> tuple[float, float]:
        # the caller's mag_db/phase_deg pass through untouched in the
        # legacy path; reduce_solve keeps its own defaults there
        raise NotImplementedError("legacy collapse budgets are the "
                                  "caller's own mag_db/phase_deg")

    def eps_equivalent(self) -> float:
        return 10.0 ** (self.tol_db / 20.0) - 1.0

    def score_fields(self, band_err, fallback_db):
        return None, "", float(band_err)          # already dB

    def headline(self, red, band_err, fmin, fmax, tol_db=0.0):
        # the claim must name the band actually ENFORCED: the budget
        # applies where |H| stays within floor_db of its peak, and
        # silently claiming the full band certified a 1-pole model
        # over decades it visibly left
        span = f"{eng(fmin, 'Hz')}-{eng(fmax, 'Hz')}"
        if red.sig_hi and (red.sig_hi < 0.99 * fmax
                           or red.sig_lo > 1.01 * fmin):
            how = (f"|H| is at least {red.floor_eff_db:g} dB "
                   f"({red.floor_db:g} dB floor widened by the "
                   f"{abs(tol_db):g} dB budget), plus the "
                   f"+/-180 deg phase crossing"
                   if red.floor_is_abs else
                   f"|H| is within {red.floor_db:g} dB of peak")
            span = (f"{eng(red.sig_lo, 'Hz')}-{eng(red.sig_hi, 'Hz')}"
                    f" — the part of the band where {how}; lower the "
                    f"enforcement floor to cover more")
        return (f"reduced to {len(red.selected)} reactance(s) "
                f"[{', '.join(red.selected)}] -- {band_err:.3f} dB vs "
                f"the full model over {span}")

    def details(self, red, band_err, fmin, fmax) -> list[str]:
        return []


@dataclass
class AnchoredCriterion(BandCriterion):
    """The one-knob contract: |H_red − H_full| ≤ ε·(|H_full| + anchor)
    at every band point, anchor = the smaller band-edge |H|. One
    dimensionless number bounds magnitude (≈8.7·ε dB) and phase
    (≈57·ε deg) jointly above the anchor and forgives the response
    below it — the band placement IS the statement of where fidelity
    matters."""

    eps: float = 0.05
    anchor: float | None = None      # captured by window()

    def __post_init__(self):
        self.tol = self.eps
        self.unit = ""
        self.name = "anchored"

    def window(self, freqs, mag_full, m, H_full) -> np.ndarray:
        # the anchor level comes from the band EDGES — dragging the
        # cursor to a level is how the user declares the depth of
        # interest
        edges = mag_full[m]
        self.anchor = float(min(edges[0], edges[-1])) if edges.size else 1.0
        return m

    def error(self, freqs, H_full, mag_full, Hr, sig) -> float:
        ok = _finite(sig, Hr, positive=False)
        if not ok.any():
            return float("inf")
        return float((np.abs(Hr[ok] - H_full[ok])
                      / (mag_full[ok] + self.anchor)).max())

    def collapse_budgets(self) -> tuple[float, float]:
        # the collapse budgets derive from the SAME ε (dB/deg are its
        # projections)
        return (20.0 * math.log10(1.0 + self.eps),
                math.degrees(self.eps))

    def eps_equivalent(self) -> float:
        return self.eps

    def score_fields(self, band_err, fallback_db):
        return (float(band_err), "fraction",
                20.0 * math.log10(1.0 + band_err))

    def headline(self, red, band_err, fmin, fmax, tol_db=0.0):
        head = (f"reduced to {len(red.selected)} reactance(s)"
                f"{self._names(red)} — within {band_err:.1%} over "
                f"{eng(fmin, 'Hz')}–{eng(fmax, 'Hz')}")
        if band_err > self.eps:
            head += (f" — BUDGET NOT MET (asked {self.eps:.0%}): "
                     f"narrow the band or raise the tolerance"
                     f" (details in Summary)")
        else:
            head += " (details in Summary)"
        return head

    def details(self, red, band_err, fmin, fmax) -> list[str]:
        a_db = (20.0 * math.log10(red.anchor) if red.anchor > 0
                else float("-inf"))
        det = [
            self._kept(red),
            f"criterion: |ΔH| ≤ ε·(|H| + anchor) at every band "
            f"point; ε = {self.eps:.0%} "
            f"(≈ ±{20 * math.log10(1 + self.eps):.2g} dB, "
            f"±{math.degrees(self.eps):.2g}°), anchor {a_db:.0f} dB — "
            f"the smaller band-edge |H|",
            f"achieved: {band_err:.1%} "
            f"(≈ ±{20 * math.log10(1 + band_err):.2g} dB, "
            f"±{math.degrees(band_err):.2g}° above the anchor)",
        ]
        if band_err > self.eps:
            det.append(f"the band edge demands fidelity at "
                       f"{a_db:.0f} dB — every decade of cursor "
                       f"past the level you care about costs "
                       f"reactances")
        return det


@dataclass
class PlainCriterion(BandCriterion):
    """Spec-sheet contract: gain_db and phase_deg, each on its own axis
    at every band point. Trusts the cursor literally — dB error is
    scale-free."""

    gain_db: float = 1.0
    phase_deg: float = 5.0

    def __post_init__(self):
        self.tol = 1.0               # normalized: <= 1 means met
        self.unit = ""
        self.cap = self.cap or 6
        self.name = "plain"

    def error(self, freqs, H_full, mag_full, Hr, sig) -> float:
        ok = _finite(sig, Hr)
        if not ok.any():
            return float("inf")
        dmag = _dmag_db(Hr, mag_full, ok)
        dph = _dphase_deg(Hr, H_full, ok)
        return float(max(dmag.max() / self.gain_db,
                         dph.max() / self.phase_deg))

    def collapse_budgets(self) -> tuple[float, float]:
        return self.gain_db, self.phase_deg

    def eps_equivalent(self) -> float:
        return 10.0 ** (self.gain_db / 20.0) - 1.0

    def score_fields(self, band_err, fallback_db):
        return float(band_err), "x budget", band_err * self.gain_db

    def headline(self, red, band_err, fmin, fmax, tol_db=0.0):
        head = (f"reduced to {len(red.selected)} reactance(s)"
                f"{self._names(red)} — "
                f"within ±{band_err * self.gain_db:.2g} dB / "
                f"±{band_err * self.phase_deg:.2g}° over "
                f"{eng(fmin, 'Hz')}–{eng(fmax, 'Hz')}")
        tail, _ = self._cap_miss(red, band_err)
        return head + tail

    def details(self, red, band_err, fmin, fmax) -> list[str]:
        det = [self._kept(red),
               f"criterion: |Δ|H|| ≤ {self.gain_db:g} dB and "
               f"|Δphase| ≤ {self.phase_deg:g}° at every band "
               f"point (strategy: gain & phase)"]
        _, miss = self._cap_miss(red, band_err)
        return det + miss


@dataclass
class StabilityCriterion(BandCriterion):
    """The reduced model must reproduce the full model's margins: PM
    within pm_deg, GM within gm_db, the unity crossing within fc_rel.
    The band's only job is to contain the crossover."""

    pm_deg: float = 5.0
    gm_db: float = 2.0
    fc_rel: float = 0.10
    _full: tuple | None = None       # (pm, fpm, gm, fgm) of the full model

    def __post_init__(self):
        self.tol = 1.0
        self.unit = ""
        self.cap = self.cap or 6
        self.name = "stability"

    def prepare(self, freqs, H_full, sig) -> None:
        from .sensitivity import loop_margins
        self._full = loop_margins(freqs[sig], H_full[sig])

    def error(self, freqs, H_full, mag_full, Hr, sig) -> float:
        from .sensitivity import loop_margins
        ok = sig & np.isfinite(Hr)
        if not ok.any():
            return float("inf")
        pm_f, fpm_f, gm_f, _ = self._full
        pm, fpm, gm, _ = loop_margins(freqs[ok], Hr[ok])
        e = 0.0
        if pm_f is not None:
            if pm is None or fpm is None or fpm <= 0:
                return float("inf")
            e = max(abs(pm - pm_f) / self.pm_deg,
                    abs(math.log(fpm / fpm_f)) / self.fc_rel)
        if gm_f is not None:
            e = max(e, float("inf") if gm is None
                    else abs(gm - gm_f) / self.gm_db)
        return e

    def metrics(self, freqs, Hr, sig) -> dict:
        from .sensitivity import loop_margins
        pm_f, fpm_f, gm_f, fgm_f = self._full
        pm_r, fpm_r, gm_r, _ = loop_margins(freqs[sig], Hr[sig])
        return {"pm_full": pm_f, "pm_red": pm_r,
                "fc_full": fpm_f, "fc_red": fpm_r,
                "gm_full": gm_f, "gm_red": gm_r}

    def collapse_budgets(self) -> tuple[float, float]:
        return 1.0, self.pm_deg

    def eps_equivalent(self) -> float:
        # the margins criterion has no relative-tolerance equivalent:
        # it gates on PM/GM/fc reproduction, not on |dH|. 5% is the
        # display-only stand-in the order certificate is queried at --
        # a mid-range "reasonable fidelity" level, deliberately not
        # derived from the pm/gm spins.
        return 0.05

    def score_fields(self, band_err, fallback_db):
        # no dB budget to project the score through: report the
        # collapse's own achieved figure
        return float(band_err), "x budget", float(fallback_db)

    def _margin_bits(self, red) -> list[str]:
        mt = red.metrics
        pf, pr = mt.get("pm_full"), mt.get("pm_red")
        ff, fr_ = mt.get("fc_full"), mt.get("fc_red")
        gf, gr = mt.get("gm_full"), mt.get("gm_red")
        bits = []
        if pf is None:
            bits.append("no unity crossing in band — margins "
                        "undefined here, nothing to gate; "
                        "widen the band to the crossover")
        if pf is not None and pr is not None:
            bits.append(f"PM {pf:.1f}°→{pr:.1f}° "
                        f"(Δ{abs(pr - pf):.1f}°)")
        if ff and fr_:
            bits.append(f"fc {eng(ff, 'Hz')} "
                        f"(Δ{abs(fr_ / ff - 1):.1%})")
        if gf is not None and gr is not None:
            bits.append(f"GM {gf:.1f}→{gr:.1f} dB")
        elif gf is None:
            bits.append("GM: no ±180° crossing in band")
        return bits

    def headline(self, red, band_err, fmin, fmax, tol_db=0.0):
        head = (f"reduced to {len(red.selected)} reactance(s)"
                f"{self._names(red)} — "
                + ", ".join(self._margin_bits(red)))
        tail, _ = self._cap_miss(red, band_err)
        return head + tail

    def details(self, red, band_err, fmin, fmax) -> list[str]:
        det = [self._kept(red),
               "criterion: the reduced model reproduces "
               "the full model's margins (strategy: "
               "stability)"]
        det += [f"margins: {b}" for b in self._margin_bits(red)]
        _, miss = self._cap_miss(red, band_err)
        return det + miss


@dataclass
class RejectionCriterion(BandCriterion):
    """For CMRR/PSRR the dB curve IS the deliverable: it must track
    within rej_db at every band point, phase unconstrained — dB error
    is relative error of the small quantity."""

    rej_db: float = 3.0

    def __post_init__(self):
        self.tol = 1.0
        self.unit = ""
        self.cap = self.cap or 6
        self.name = "rejection"

    def error(self, freqs, H_full, mag_full, Hr, sig) -> float:
        ok = _finite(sig, Hr)
        if not ok.any():
            return float("inf")
        return float(_dmag_db(Hr, mag_full, ok).max() / self.rej_db)

    def collapse_budgets(self) -> tuple[float, float]:
        return self.rej_db, 30.0

    def eps_equivalent(self) -> float:
        return 10.0 ** (self.rej_db / 20.0) - 1.0

    def score_fields(self, band_err, fallback_db):
        return float(band_err), "x budget", band_err * self.rej_db

    def headline(self, red, band_err, fmin, fmax, tol_db=0.0):
        head = (f"reduced to {len(red.selected)} reactance(s)"
                f"{self._names(red)} — "
                f"tracks within {band_err * self.rej_db:.2g} dB over "
                f"{eng(fmin, 'Hz')}–{eng(fmax, 'Hz')}")
        tail, _ = self._cap_miss(red, band_err)
        return head + tail

    def details(self, red, band_err, fmin, fmax) -> list[str]:
        det = [self._kept(red),
               f"criterion: |Δ|H|| ≤ {self.rej_db:g} dB at every "
               f"band point, phase unconstrained "
               f"(strategy: rejection)"]
        _, miss = self._cap_miss(red, band_err)
        return det + miss


_STRATEGIES = {"plain": PlainCriterion, "stability": StabilityCriterion,
               "rejection": RejectionCriterion}


def make_criterion(*, strategy: str | None = None,
                   strategy_opts: dict | None = None,
                   eps: float | None = None,
                   tol_db: float = 1.0, metric: str = "complex",
                   floor_db: float = 60.0,
                   floor_abs_db: float | None = None,
                   phase_tol_deg: float | None = None,
                   cap: int = 6) -> BandCriterion:
    """Map the accreted keyword surface onto ONE criterion object.
    Precedence mirrors the historical dispatch: a strategy wins, then
    the anchored ε, then the legacy dB contract."""
    opts = dict(strategy_opts or {})
    if strategy is not None:
        try:
            cls = _STRATEGIES[strategy]
        except KeyError:
            raise ValueError(f"unknown strategy {strategy!r}; expected "
                             f"one of {sorted(_STRATEGIES)}") from None
        fields = {k: v for k, v in opts.items()
                  if k in cls.__dataclass_fields__}
        c = cls(**fields)
        c.cap = opts.get("cap", cap)
        return c
    if eps is not None:
        return AnchoredCriterion(eps=eps)
    return LegacyCriterion(tol_db=tol_db, metric=metric, floor_db=floor_db,
                           floor_abs_db=floor_abs_db,
                           phase_tol_deg=phase_tol_deg)
