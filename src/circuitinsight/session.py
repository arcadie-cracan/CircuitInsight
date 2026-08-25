"""Headless view-model for the CircuitInsight front ends.

`SessionController` is the single object both the PySide6 desktop app and the
teaching notebooks drive. It wraps the simulator-neutral core (`Analyzer` over a
CIN + operating-point data), holds the user's choices (input/output, matched
pairs, keep-set), runs solves on demand, caches results, and hands back a plain
`Result` the UIs render.

Independence contract (see docs/gui-virtuoso-integration-plan.md): this module and
everything it imports are simulator- and GUI-neutral. It never imports Qt, and
never imports the Cadence/Virtuoso integration layer. Simulator back ends are
reached only through the adapters (`open(..., simulator=...)`), so the
ngspice/LTspice/offline paths stay first-class.
"""
from __future__ import annotations

import math
import types
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import sympy as sp

from .keep import (ALL, is_all, norm_keep,  # noqa: F401
                   norm_keep_list)
from .units import db, eng

__all__ = ["SessionController", "Result", "DeviceInfo", "SolveTooLarge"]

# Above this symbol-space size a fully symbolic (keep=ALL) determinant is
# hopeless. The 5T OTA -- a 13-device circuit -- already sits at 2.8e14.
_SYMBOLIC_GRID_LIMIT = 1e6


@dataclass
class DeviceInfo:
    """One reconstructed device, for the source/topology view."""
    name: str
    device_type: str
    terminals: dict[str, str]


class SolveTooLarge(RuntimeError):
    """A solve was refused because its predicted cost blew the budget."""


@dataclass
class Result:
    """Everything a front end needs to render one solve. Plain data — no Qt,
    no sympy required to *display* it (LaTeX is pre-rendered to strings)."""
    inp: str
    out: str
    keep: object                # ALL = fully symbolic; [] = numeric
    # numeric summary
    dc_gain: complex
    dc_gain_db: float
    poles_hz: np.ndarray            # complex, ascending |.|, in Hz
    zeros_hz: np.ndarray
    n_terms: int                    # symbolic complexity (num+den monomials)
    # symbolic, pre-rendered
    tf_latex: str
    dc_gain_latex: str
    # Bode data
    freqs: np.ndarray               # Hz
    h: np.ndarray                   # complex TF over freqs
    h_ref: np.ndarray | None        # reference AC (sim), if available
    ref_label: str | None
    # advisories (pole separation, RHP zeros, missing reference, ...)
    warnings: list[str] = field(default_factory=list)
    # the underlying TransferFunction, opaque to UIs (simplification, export)
    tf: object = None
    # set by simplify(); None/False for a plain solve
    simplified: bool = False
    mag_err_db: float | None = None       # achieved simplification error
    phase_err_deg: float | None = None
    n_terms_full: int | None = None       # complexity before pruning
    # set by loop_gain(); None for ordinary transfer solves. Spectre stb
    # convention: PM = arg T at |T|=1, GM = -|T|dB where arg T crosses 0.
    pm_deg: float | None = None
    pm_freq_hz: float | None = None
    gm_db: float | None = None
    gm_freq_hz: float | None = None
    # simplify/reduce band (for plot shading); None for plain solves
    band_fmin: float | None = None
    band_fmax: float | None = None
    # the band the budget was actually ENFORCED over: inside the chosen
    # band, the part where |H| stays within the enforcement window of its
    # peak. Recorded for the report and the tests; the GUI's tolerance
    # tubes are drawn from the toolbar band, which the strategy and
    # anchored paths enforce in full (enforced_* == band_* there).
    enforced_fmin: float | None = None
    enforced_fmax: float | None = None
    # designer-form summary (A0, GBW, per-root formulas), pre-rendered by
    # attach_template(); None when not computed or not computable
    template_text: str | None = None
    # which working circuit produced this: "as imported" or "reduced"
    # (AC grounds applied + dead sources removed + passives lumped)
    circuit_state: str = "as imported"
    # True for lowest-order results (reduce_solve): certified only over
    # [band_fmin, band_fmax] AT THIS OPERATING POINT -- consumers that
    # move parameters (What-if) must refuse or warn
    reduced_order: bool = False
    # lowest-order criterion record (previously attribute-injected):
    # which strategy judged the reduction, its normalized score (<=1 met)
    # with its unit, and the anchored-mode parameters. mag_err_db keeps
    # ONE meaning -- an achieved error in dB -- it used to carry a dB, a
    # fraction, or a normalized score depending on the branch, and the
    # view printed all three as dB.
    strategy: str | None = None
    band_score: float | None = None
    band_score_unit: str = ""
    eps: float | None = None
    anchor: float | None = None
    details: list = field(default_factory=list)
    certificate: object = None
    # lowest-order: the DC-gain coefficient divided out (1.0 = none)
    snap_factor: float = 1.0


def _n_terms(tf) -> int:
    num, den = tf.num_den
    return sum(len(sp.Add.make_args(sp.expand(c)))
               for poly in (num, den) for _, c in poly.terms())




def _numeric_dc(tf) -> complex:
    """True dc gain: the TF at s=0 with every kept symbol at its OP value
    (frequency-independent, so unaffected by where the dominant pole sits)."""
    e = tf.dc_gain()
    subs = {tf.symbols[n]: sp.Float(v) for n, v in tf.values.items()
            if n in tf.symbols}
    if subs:
        e = e.xreplace(subs)
    return complex(e)


def _significance_floors(by_inst: dict) -> dict:
    """Per instance: 5% of the dominant conductance and of the dominant
    capacitance -- the line below which a parameter ratio is noise, not a
    mismatch. Shared by suggest_matches and match_conflicts so they cannot
    disagree about what counts."""
    sig: dict = {}
    for inst, params in by_inst.items():
        gmax = max((abs(v) for k, v in params.items()
                    if k.startswith("g")), default=0.0)
        cmax = max((abs(v) for k, v in params.items()
                    if k[:1] in "ck"), default=0.0)
        sig[inst] = {"g": 0.05 * gmax, "c": 0.05 * cmax}
    return sig


class SessionController:
    """Stateful, headless controller over one analysis session."""

    def __init__(self):
        self._run = None                 # simulator adapter run (opaque)
        self._analyzer = None            # built lazily, rebuilt when matches change
        self._matches: list[tuple[str, ...]] = []
        self._cache: dict[tuple, Result] = {}
        # Iordache's latency principle across a session: torn
        # sub-solves and the suggester's symbolic prelude are
        # keyed on a CONTENT fingerprint, so an edit re-solves
        # only what changed. Never cleared on its own -- a stale
        # entry is impossible, since the fingerprint IS the
        # identity of the circuit half it describes.
        self._tear_cache: dict = {}
        self._op_values = None               # op_values() lazy cache
        self._has_stb: bool | None = None    # lazily probed, cached
        #: net whose AC response the user DECLARED to be the return
        #: ratio; None = no declaration (see declare_ac_loop_gain)
        self.ac_loop_gain: str | None = None
        self.cin_path: Path | None = None
        self.op_path: Path | None = None
        self.simulator: str | None = None
        self.cap_model: str = "lumped"   # see open(cap_model=...)
        self.mos_model: str = "separate"  # see open(mos_model=...)
        self._reduction: dict | None = None   # apply_reduction state
        # matched-value policy: how a group's shared symbols get their
        # numeric values. Default "weighted" -- the band-sensitivity-weighted
        # mean, which found the load-bearing member on BOTH benches
        # (model-vs-sim: ota5t 0.039 dB vs first-wins 0.077; folded cascode
        # 0.347 vs 1.804). "representative" with no chosen representative is
        # the historic first-stamp-wins behavior, kept selectable.
        self._match_policy: str = "weighted"
        self._match_reps: dict[frozenset, str] = {}
        self._match_io: tuple[str, str] | None = None   # weighted's context
        self._match_orig: dict | None = None   # pre-policy values, for reports
        self._sens_wcache: dict = {}           # weights per (inp, out) -- the
                                               # pristine OP never changes
        #: instance/symbol -> LaTeX alias for expression rendering (GUI)
        self.sym_aliases: dict[str, str] = {}

    # ------------------------------------------------------------------ open
    @classmethod
    def open(cls, cin_path, op_path, *, simulator: str = "spectre",
             cap_model: str = "lumped", mos_model: str = "separate",
             **adapter_kw) -> "SessionController":
        """Open a session from a CIN topology + a simulator's OP results.

        `simulator` selects the adapter; only "spectre" exists today, but the
        entry point is neutral so ngspice/LTspice slot in the same way.

        `cap_model`: "lumped" (five-capacitor, default) or "matrix" (exact
        charge matrix). On strongly non-reciprocal processes (SKY130) the
        matrix model is the accurate one -- loop-gain margins in particular
        shift by ~0.1 deg / 0.6% between the two on the two-stage bench.

        `mos_model`: "separate" (gm and gmbs stamped independently,
        default) or "lumped-gmb" (EXACT per-device bundle: where gate
        and bulk sit at the same AC potential one symbol
        ghat = gm + gmb carries both, and a bulk-tied-to-source gmbs is
        dropped as inert; see models.small_signal._lump_gmb).
        """
        self = cls()
        self.simulator = simulator
        self.cap_model = cap_model
        self.mos_model = mos_model
        if simulator == "spectre":
            from .adapters.spectre import SpectreRun
            self._run = SpectreRun(cin_path, op_path, **adapter_kw)
        else:
            raise ValueError(f"unknown simulator adapter {simulator!r}")
        self.cin_path = Path(cin_path)
        self.op_path = Path(op_path)
        return self

    # -------------------------------------------------------- introspection
    @property
    def devices(self) -> list[DeviceInfo]:
        return [DeviceInfo(d.name, d.device_type, dict(d.terminals))
                for d in self._run.flat.devices]

    @property
    def ground(self) -> list[str]:
        return list(self._run.flat.ground)

    @property
    def nets(self) -> list[str]:
        gnd = set(self._run.flat.ground)
        seen: dict[str, None] = {}
        for d in self._run.flat.devices:
            for net in d.terminals.values():
                if net not in gnd:
                    seen.setdefault(net, None)
        return list(seen)

    def sources(self) -> list[str]:
        return [d.name for d in self._run.flat.devices
                if d.device_type in ("vsource", "isource")]

    def input_ports(self) -> list[str]:
        """Candidate inputs: excited sources first (nonzero AC magnitude), then
        the remaining independent sources."""
        srcs = self.sources()
        try:
            excited = [s for s in self._run.excited_sources() if s in srcs]
        except Exception:
            excited = []
        rest = [s for s in srcs if s not in excited]
        return excited + rest

    def suggested_input(self) -> str | None:
        ports = self.input_ports()
        return ports[0] if ports else None

    def output_nets(self) -> list[str]:
        return self.nets

    @property
    def reductions(self) -> list[str]:
        """Human-readable notes on netlist reductions applied when the run was
        opened -- e.g. simulator-pruned (0-valued) passives folded out. Empty
        when nothing was reduced."""
        return list(getattr(self._run, "reductions", None) or [])

    def suggested_output(self) -> str | None:
        """Best guess at the output net, so the first solve is meaningful rather
        than landing on the first net alphabetically (a bias node, typically).

        Heuristic only -- prefers out/vout-like names, penalizes inputs, bias,
        supplies and internal (dotted / netNN) nets. The user can override.
        """
        nets = self.output_nets()
        if not nets:
            return None
        gnd = {g.lower() for g in self._run.flat.ground}

        def score(n: str) -> int:
            ln, s = n.lower(), 0
            if ln in ("out", "vout", "vo", "output", "outp", "voutp"):
                s += 100
            if ln.startswith(("out", "vout")):
                s += 40
            if "out" in ln and "in" not in ln:
                s += 20
            if any(k in ln for k in ("vin", "in_", "inp", "inn",
                                     "bias", "vb", "cm", "dm", "ref", "cascn",
                                     "cascp")):
                s -= 30
            if "net" in ln or "." in n:                 # internal node
                s -= 25
            if ln in gnd or ln in ("vdd", "vss", "vcc", "gnd") or ln.endswith("!"):
                s -= 100
            return s

        best = max(nets, key=score)                     # ties keep first order
        return best

    # ----------------------------------------------------------- configuration
    def set_matches(self, *groups: tuple[str, ...]) -> None:
        """Declare matched-instance groups (each a tuple of instance names).
        Invalidates the analyzer and result cache."""
        self._matches = [tuple(g) for g in groups if len(g) >= 2]
        self._analyzer = None
        self._op_values = None
        self._cache.clear()

    # ------------------------------------------------- matched-value policy
    # Matching declares devices to share SYMBOLS; some single value set must
    # then stand for the whole group. Three policies, none universally best
    # (measured: first-wins beats the mean on the ota5t, the mean beats it
    # on the folded cascode -- the response follows whichever member's
    # parameter is load-bearing, which no fixed rule can know):
    #   representative -- one member's values stand for the group. With no
    #       representative chosen this is EXACTLY the historic engine
    #       behavior (first to stamp wins), so nothing re-pins.
    #   mean           -- the arithmetic group mean, parameter by parameter.
    #   weighted       -- the mean weighted by each member's band
    #       sensitivity: the member that shapes the response most pulls
    #       hardest. Needs an (inp, out) context; falls back to the
    #       suggested ones.

    @property
    def match_value_policy(self) -> str:
        return self._match_policy

    def set_match_value_policy(self, policy: str,
                               inp: str | None = None,
                               out: str | None = None) -> None:
        if policy not in ("representative", "mean", "weighted"):
            raise ValueError(f"unknown match value policy {policy!r}")
        self._match_policy = policy
        if inp and out:
            self._match_io = (inp, out)
        self._analyzer = None
        self._op_values = None
        self._cache.clear()

    def set_match_representative(self, inst: str) -> None:
        """Make `inst`'s values stand for its whole match group (and switch
        the policy to representative)."""
        for g in self._matches:
            if inst in g:
                self._match_reps[frozenset(g)] = inst
                self._match_policy = "representative"
                self._analyzer = None
                self._op_values = None
                self._cache.clear()
                return
        raise ValueError(f"{inst!r} is not in any match group")

    def match_representative(self, group) -> str | None:
        return self._match_reps.get(frozenset(group))

    def _sens_weights(self, an) -> dict[str, float]:
        """|band sensitivity| per UNMATCHED symbol name, for the weighted
        policy. Computed on a pristine analyzer so each member keeps its own
        symbol; failures degrade to {} (equal weights)."""
        try:
            io = self._match_io or (self.suggested_input(),
                                    self.suggested_output())
            if not (io[0] and io[1]):
                return {}
            if io in self._sens_wcache:
                return self._sens_wcache[io]
            base = self._run.analyzer(cap_model=self.cap_model,
                                    mos_model=self.mos_model)
            rep = base.band_sensitivities(io[0], io[1])
            w = {n: abs(s) for n, s in rep.ranking}
            self._sens_wcache[io] = w
            return w
        except Exception:
            return {}

    def _apply_match_values(self, an) -> list:
        """Rewrite the group members' parameter values per the active
        policy, so the engine's first-wins fusion becomes harmless (every
        member carries the same value). Representative-with-no-choice
        returns the primitives untouched -- the historic behavior."""
        import dataclasses

        if not self._matches or self._match_policy == "representative" \
                and not self._match_reps:
            return an.primitives
        by_inst: dict[str, dict[str, float]] = {}
        for p in an.primitives:
            if p.param and p.value is not None:
                by_inst.setdefault(p.inst, {})[p.param] = p.value
        weights = (self._sens_weights(an)
                   if self._match_policy == "weighted" else {})
        override: dict[tuple, float] = {}
        for g in self._matches:
            rep = self._match_reps.get(frozenset(g))
            params = set().union(*(by_inst.get(m, {}) for m in g))
            for param in params:
                vals = {m: by_inst[m][param] for m in g
                        if param in by_inst.get(m, {})}
                if len(vals) < 2:
                    continue
                if self._match_policy == "representative":
                    if rep is None or param not in by_inst.get(rep, {}):
                        continue
                    v = by_inst[rep][param]
                elif self._match_policy == "mean":
                    v = sum(vals.values()) / len(vals)
                else:                                  # weighted
                    def w_of(m):
                        key = f"{param}_{m.replace('.', '_')}"
                        return weights.get(key, 0.0)
                    tot = sum(w_of(m) for m in vals)
                    if tot > 0:
                        v = sum(w_of(m) * x for m, x in vals.items()) / tot
                    else:
                        v = sum(vals.values()) / len(vals)
                for m in vals:
                    # plain float: np.float64 sneaks in via the sensitivity
                    # weights, and sp.Rational(repr(np.float64)) is the
                    # known numpy2 crash
                    override[(m, param)] = float(v)
        if not override:
            return an.primitives
        return [dataclasses.replace(p, value=override[(p.inst, p.param)])
                if (p.inst, p.param) in override else p
                for p in an.primitives]

    def match_conflicts(self, tol: float = 0.05) -> list[tuple]:
        """Where the declared matches overwrite reality: matching shares the
        FIRST member's values, so every member parameter that differs from
        it by more than `tol` is a place the model just moved. Returns
        [(param, kept_inst, other_inst, ratio)] sorted worst first.

        Only SIGNIFICANT parameters count: a parameter below 5% of the
        device's dominant same-kind one (caps against the largest cap,
        conductances against the largest conductance) is skipped -- the
        ota5t's true input pair differs 4.6x on a kbd that is 4% of cgs,
        and reporting that as a conflict would teach users to ignore the
        report. The same floor governs suggest_matches.

        This is the pre-check for what engine.mna warns about at build
        time -- those warnings go to the Python layer and a GUI user never
        sees them; this is the same information as data, so a front end
        can say 'these matches fused devices whose gds differs 3.2x'
        BEFORE the wrong plot appears."""
        final: dict[str, dict[str, float]] = {}
        first_at: dict[str, int] = {}
        for i, p in enumerate(self._analyzer_ready().primitives):
            if p.param and p.value is not None:
                final.setdefault(p.inst, {})[p.param] = p.value
                first_at.setdefault(p.inst, i)
        orig = self._match_orig or final
        sig = _significance_floors(orig)
        out = []
        for g in self._matches:
            # what the model actually uses: post-policy, and under the
            # historic default that is the values of whichever member
            # STAMPS first (netlist order), not the first tuple name
            g = sorted(g, key=lambda n: first_at.get(n, 1 << 30))
            rep = self._match_reps.get(frozenset(g))
            src = (rep if rep is not None
                   else g[0] if self._match_policy == "representative"
                   else f"the {self._match_policy} value")
            used = final.get(g[0], {})
            for member in g:
                for param, v_used in used.items():
                    v_real = orig.get(member, {}).get(param)
                    if v_real is None or v_real == 0 or v_used == 0:
                        continue
                    kind = "g" if param.startswith("g") else "c"
                    floor = min(sig.get(g[0], {}).get(kind, 0.0),
                                sig.get(member, {}).get(kind, 0.0))
                    if abs(v_used) < floor and abs(v_real) < floor:
                        continue
                    ratio = abs(v_used / v_real)
                    if ratio > 0 and abs(ratio - 1.0) > tol:
                        out.append((param, src, member,
                                    max(ratio, 1.0 / ratio)))
        out.sort(key=lambda t: -t[3])
        return out

    @property
    def matches(self) -> list[tuple[str, ...]]:
        return list(self._matches)

    def suggest_matches(self) -> list[tuple[str, ...]]:
        """Heuristic matched sets to review: transistors that are structurally
        identical (same device type and parameters — polarity, multiplier, and
        W/L when the CIN carries them) are likely matched. A suggestion, not a
        decision — the user applies/edits it via set_matches()."""
        groups: dict[tuple, list[str]] = {}
        order: list[tuple] = []
        for d in self._run.flat.devices:
            if d.device_type not in ("mosfet", "bjt", "npn", "pnp"):
                continue
            # only siblings (same subckt scope) can be a matched set — keeps a
            # bench device from matching an identically-sized in-DUT one
            parent = d.name.rsplit(".", 1)[0] if "." in d.name else ""
            key = (parent, d.device_type,
                   tuple(sorted((d.params or {}).items())))
            if key not in groups:
                groups[key] = []
                order.append(key)
            groups[key].append(d.name)

        # Structural identity is necessary but not sufficient: aliasing is
        # only exact when the operating points MATCH. The params key cannot
        # see everything (e.g. L rides in meta, and bias branches differ),
        # so refine each group by gm proximity -- the fd bench's input
        # pair, CMFB tail, and CMFB error pair are all m=1 nfets that
        # would otherwise fuse into one 5-device alias and distort the CM
        # loop by ~6 deg.
        try:
            opv = self.op_values()
        except Exception:
            opv = {}

        # Matching shares the first member's VALUES, so a suggestion is only
        # sound when the operating points agree on EVERY shared parameter --
        # not just gm. The folded cascode is the calibration case: eight
        # NMOS with gm within 2% (same current) fused into one alias, but
        # gds differed 3.2x (different vds) and the junction caps up to 49x
        # (different drain bias), and the shared values cost 3.6 dB of DC
        # gain. Per-kind tolerances: gm 2% (the original criterion), other
        # conductances 25%, capacitances 50% -- true pairs on real silicon
        # show cap spreads up to ~15% (the ota5t flagship's cjd is 7%
        # apart), so 50% keeps them while splitting bias-plane strangers.
        try:
            prims = self._analyzer_ready().primitives
        except Exception:
            prims = []
        by_inst: dict[str, dict[str, float]] = {}
        for p in prims:
            if p.param and p.value is not None:
                by_inst.setdefault(p.inst, {})[p.param] = p.value

        # gm's 2% is the strict default (it kept the fd bench's input
        # pair from fusing with its CMFB pair). A MIRROR earns more: when
        # the set shares its gate net and V_GS and V_TH agree within
        # 1 mV (source-tied legs give exact equality), and the legs
        # carry the same current, the only gm difference left is
        # channel-length modulation, and its size is
        # predictable from the set's own operating point:
        # dgm/gm ~ dId/Id = gds*dVds/Id. The budget is 2% plus that
        # measured term, capped at 10% -- a tight-Vds mirror gets almost
        # no slack, a wide one gets what its physics justifies, and
        # gate-independent lookalikes never qualify.
        terms_by = {d.name: dict(d.terminals)
                    for d in self._run.flat.devices}

        def _op(name):
            try:
                return self.device_op(name) or {}
            except Exception:
                return {}

        def _gm_tol(members, cand):
            allm = list(members) + [cand]
            if len({terms_by.get(m, {}).get("g") for m in allm}) != 1:
                return 0.02
            ops = [_op(m) for m in allm]
            vgs = [o.get("vgs") for o in ops]
            vth = [o.get("vth") for o in ops]
            if any(v is None for v in vgs + vth):
                return 0.02
            if max(vgs) - min(vgs) > 1e-3 or max(vth) - min(vth) > 1e-3:
                return 0.02
            vds = [o.get("vds") for o in ops]
            gds = [o.get("gds") for o in ops]
            ids = [abs(o.get("ids") or 0.0) for o in ops]
            if any(v is None for v in vds + gds) or not all(ids):
                return 0.02
            dvds = max(abs(v) for v in vds) - min(abs(v) for v in vds)
            budget = 0.02 + dvds * (sum(gds) / len(gds))                 / (sum(ids) / len(ids))
            budget = min(budget, 0.10)
            # the currents must agree to within the same budget: a
            # ratioed or differently loaded leg is not this mirror
            if max(ids) / min(ids) - 1.0 > budget:
                return 0.02
            return budget

        def _tol(param, members=(), cand=None):
            if param == "gm":
                return _gm_tol(members, cand) if cand else 0.02
            if param.startswith("g"):
                return 0.25
            return 0.50

        # significance floor per instance and kind: a ratio test on a
        # parameter that is a few percent of the device's dominant one is
        # noise, not a mismatch -- the ota5t's true input pair differs 4.6x
        # on kbd, at 4% of the device's cgs, and physically it matters not
        # at all. Kinds: conductances (g*) against the largest g, caps
        # (c*/k*) against the largest cap.
        sig = _significance_floors(by_inst)

        def compatible(members, cand):
            """cand against the RUNNING MEAN of the subgroup so far --
            a fixed head made membership depend on netlist order: the
            fc cascode NM4 differed 2.8% in gm from head NM3 (out, at
            a 2% tolerance) but 0.9% from NM5; against the group mean
            it belongs. Genuinely different devices stay out (NM2's gm
            is 4x off any mean)."""
            pb = by_inst.get(cand)
            withop = [m for m in members if by_inst.get(m)]
            if not pb or not withop:
                return True                    # no OP data: keep old behavior
            params = set().union(*(by_inst[m].keys() for m in withop))
            for param in params:
                vals = [by_inst[m][param] for m in withop
                        if param in by_inst[m]]
                vb = pb.get(param)
                if not vals or vb is None or vb == 0:
                    continue
                va = sum(vals) / len(vals)
                if va == 0:
                    continue
                kind = "g" if param.startswith("g") else "c"
                floor = min([sig[m][kind] for m in withop]
                            + [sig[cand][kind]])
                if abs(va) < floor and abs(vb) < floor:
                    continue                   # both insignificant: skip
                r = abs(vb / va)
                r = max(r, 1.0 / r) if r > 0 else float("inf")
                if r - 1.0 > _tol(param, withop, cand):
                    return False
            return True

        def gm_of(name):
            return opv.get("gm_" + name.replace(".", "_"))

        # one more STRUCTURAL criterion: a matched set shares at least
        # one net. A true pair touches at gate, source or drain (the
        # input pair's common source, a mirror's common gate); two
        # same-sized devices in unrelated branches do not, and only the
        # bias plane distinguishes them. Ground and global nets do not
        # count -- every bulk ties to gnd!, which would connect
        # everything. Connectivity components via union-find.
        skip_nets = set(self.ground) |             set(getattr(self._run.flat, "global_nets", ()) or ())
        terms_of = {d.name: {n for n in d.terminals.values()
                             if n not in skip_nets}
                    for d in self._run.flat.devices}

        def _components(members):
            parent = {m: m for m in members}

            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            by_net: dict = {}
            for m in members:
                for n in terms_of.get(m, ()):
                    by_net.setdefault(n, []).append(m)
            for ms in by_net.values():
                for other in ms[1:]:
                    ra, rb = find(ms[0]), find(other)
                    if ra != rb:
                        parent[rb] = ra
            comps: dict = {}
            for m in members:
                comps.setdefault(find(m), []).append(m)
            return [c for c in comps.values()]

        refined: list[tuple[str, ...]] = []
        for k in order:
            for members in _components(groups[k]):
                if len(members) < 2:
                    continue
                self._refine_members(members, gm_of, compatible, refined)
        return refined

    @staticmethod
    def _refine_members(members, gm_of, compatible, refined):
        # greedy clustering against each subgroup's head, gm-sorted so
        # near-identical devices are adjacent
        members = sorted(members, key=lambda n: (gm_of(n) is None,
                                                 gm_of(n) or 0.0))
        remaining = list(members)
        while remaining:
            head = remaining.pop(0)
            sub = [head]
            still = []
            for cand in remaining:
                if compatible(sub, cand):
                    sub.append(cand)          # the mean now includes cand
                else:
                    still.append(cand)
            remaining = still
            if len(sub) >= 2:
                refined.append(tuple(sub))

    def name_bearer(self, group) -> str:
        """The member whose NAME labels the group's shared symbols: the
        one whose parameters shape the in->out response most (band
        sensitivity, the weighted policy's own numbers), so the formula
        reads g_m,NM5 for the cascode on the signal path rather than
        g_m,NM3 for its mirror twin. A user-chosen representative wins;
        ties and a missing context fall back to the group order."""
        rep = self._match_reps.get(frozenset(group))
        if rep is not None and rep in group:
            return rep
        w = self._sens_weights(None)
        if not w:
            return group[0]

        def score(m):
            key = m.replace(".", "_")
            return sum(v for n, v in w.items() if n.endswith("_" + key))
        best = max(group, key=score)
        return best if score(best) > 0 else group[0]

    def _analyzer_ready(self):
        if self._analyzer is None:
            an = self._run.analyzer(cap_model=self.cap_model,
                                    mos_model=self.mos_model)
            for group in self._matches:
                # the name bearer first: analyzer.match names the
                # shared symbols after its first argument
                lead = self.name_bearer(group)
                an.match(lead, *[m for m in group if m != lead])
            # capture pre-policy values FIRST: conflict reports must compare
            # against reality, not against what the policy already wrote
            self._match_orig = {}
            for p_ in an.primitives:
                if p_.param and p_.value is not None:
                    self._match_orig.setdefault(p_.inst, {})[p_.param] = p_.value
            an.primitives = self._apply_match_values(an)
            if self._reduction is not None:
                # the reduction is session state, not analyzer state: any
                # rebuild (e.g. matches changed) re-applies it from its
                # recorded node set, so it cannot silently evaporate
                an.primitives = self._reduced_primitives(
                    an, self._reduction["nodes"],
                    self._reduction.get("removals", ()))
            self._analyzer = an
        return self._analyzer

    # --------------------------------------------------------------- planning
    def estimate(self, inp: str, out: str, keep=ALL):
        """SolveEstimate for tf(inp, out, keep) — gate a solve on predicted cost."""
        return self._analyzer_ready().estimate_solve_time(inp, out, keep)

    def estimate_loop(self, probe: str, keep=()):
        """SolveEstimate for loop_gain(probe, keep): the Tian hybrid is
        FOUR rational solves on the probe system, so the estimate is
        one solve's with n_dets and seconds scaled by 4. The keep
        plan budgets the TRANSFER solve; this is what the same ticks
        actually cost at the probe."""
        import dataclasses

        from .analysis.loopgain import _probe_nodes

        an = self._analyzer_ready()
        system = an.system(probe)
        p, _q = _probe_nodes(system, probe)
        est = an.estimate_solve_time(probe, system.unknowns[p], keep)
        return dataclasses.replace(
            est, n_dets=est.n_dets * 4,
            seconds=(est.seconds * 4 if est.seconds is not None
                     else None))

    def estimate_z(self, spec, keep=()):
        """SolveEstimate for impedance_result(spec, keep): one solve on
        the probe-injected system. Raises for between-two-nets specs
        (two numerator solves, no single-probe system) -- callers fall
        back to the transfer estimate."""
        an2, probe, node = self._z_probe_analyzer(spec)
        return an2.estimate_solve_time(probe, node, keep)

    def _guard_loop_cost(self, probe, keep, max_seconds) -> None:
        """The loop-gain twin of _guard_cost: ticks budgeted for the
        transfer can cost ~4x at the probe (field report: a fast
        machine's auto plan froze the bench for minutes -- the plan
        gates tf(), nothing gated T). ALL is refused on grid size; a
        hybrid is gated only when a budget is given."""
        keep = norm_keep_list(keep)
        if max_seconds is None and not is_all(keep):
            return
        if not is_all(keep) and not keep:
            return                      # numeric: the fast path, never gated
        try:
            est = self.estimate_loop(probe, keep)
        except Exception:
            return                      # an estimate is a courtesy
        if is_all(keep):
            if est.grid_size <= _SYMBOLIC_GRID_LIMIT:
                return
            raise SolveTooLarge(
                f"loop gain at {probe}, fully symbolic (keep=ALL): "
                f"{len(est.kept_names)} symbols, symbol space "
                f"{est.grid_size:.3g}. The symbolic determinant is "
                f"intractable at this size — it will not finish.\n"
                f"Tick a keep set, keep=[] for a numeric T, or "
                f"max_seconds=None to insist.")
        if est.seconds is None or est.seconds <= max_seconds:
            return
        raise SolveTooLarge(
            f"loop gain at {probe}, keep={list(keep)}: estimated "
            f"{est.seconds:.0f} s for the Tian hybrid (four rational "
            f"solves, grid {est.grid_size:,}), over the "
            f"{max_seconds:.0f} s budget.\nThe keep plan budgeted the "
            f"TRANSFER solve — T costs ~4x with the same ticks. Untick "
            f"symbols, solve numeric (no ticks), or pass "
            f"max_seconds=None to insist.")

    def suggest_keep(self, inp: str, out: str, budget_s: float):
        """Largest band-ranked keep-set whose solve fits `budget_s` (KeepPlan)."""
        return self._analyzer_ready().plan_keep(inp, out, budget_s)

    # ------------------------------------------------------------------ solve
    def _input_net(self, inp: str) -> str | None:
        gnd = set(self._run.flat.ground)
        for d in self._run.flat.devices:
            if d.name == inp:
                p = d.terminals.get("p")
                n = d.terminals.get("n")
                if p is not None and p not in gnd:
                    return p
                if n is not None and n not in gnd:
                    return n
        return None

    def _reference(self, inp: str, out: str):
        """(h_ref, label) — the AC transfer v(out)/v(input net) from the same
        run, for overlay; (None, None) if unavailable."""
        try:
            in_net = self._input_net(inp)
            if in_net is None:
                return None, None
            ac = self._run.ac()
            h_ref = np.asarray(ac.wave(out)) / np.asarray(ac.wave(in_net))
            return np.asarray(ac.freq), (h_ref, f"AC sim  v({out})/v({in_net})")
        except Exception:
            return None, None

    def _assemble(self, tf, inp, out, keep, *, reference, fmin, fmax,
                  points) -> Result:
        """Package any TF-like (exact or simplified) into a Result."""
        freqs = np.logspace(math.log10(fmin), math.log10(fmax), points)
        h = np.asarray(tf.numeric(freqs))
        dc = _numeric_dc(tf)
        dc_db = db(dc)
        poles = tf.poles()
        zeros = tf.zeros()

        warns: list[str] = []
        ap = np.sort(np.abs(poles))
        if ap.size >= 2 and ap[0] > 0 and ap[1] / ap[0] < 10:
            warns.append(
                f"poles not well separated ({ap[1] / ap[0]:.1f}x): a "
                f"dominant-pole approximation is questionable")
        if np.any(np.real(zeros) > 0):
            warns.append("right-half-plane zero present (excess phase lag)")

        h_ref = ref_label = None
        if reference:
            fr, packed = self._reference(inp, out)
            if packed is not None:
                # resample onto the sim grid so overlay/error are point-aligned
                freqs = np.asarray(fr, dtype=float)
                h = np.asarray(tf.numeric(freqs))
                h_ref = np.asarray(packed[0])
                ref_label = packed[1]
            else:
                warns.append("no AC reference in this run (model only)")

        return Result(
            # None records "fully symbolic" — distinct from [] (fully numeric).
            # Coercing both to [] destroyed that, so a Result could not say which
            # solve produced it, and the summary mislabelled every symbolic one.
            inp=inp, out=out,
            keep=norm_keep_list(keep),
            dc_gain=dc, dc_gain_db=dc_db,
            poles_hz=poles, zeros_hz=zeros, n_terms=_n_terms(tf),
            tf_latex=sp.latex(tf.expr),
            dc_gain_latex=sp.latex(tf.dc_gain()),
            freqs=freqs, h=h, h_ref=h_ref, ref_label=ref_label,
            warnings=warns, tf=tf,
            circuit_state=self.circuit_state,
        )

    def _key(self, tag: str, *parts) -> tuple:
        """ONE composition rule for the analysis cache keys: the tag,
        the caller's own discriminators, then ALWAYS the full shared
        configuration -- matches, circuit state, match policy. The
        sixteen hand-built tuples covered three different subsets of
        that configuration; all were safe only because every mutator
        clears the cache, but a key that looks partial reads as an
        oversight and invites one."""
        return (tag, *parts, tuple(self._matches), self.circuit_state,
                self._match_policy)

    def solve(self, inp: str, out: str, keep=ALL, *,
              reference: bool = True, fmin: float = 1e3, fmax: float = 1e9,
              points: int = 400, max_seconds: float | None = None,
              progress=None) -> Result:
        """Solve tf(inp, out, keep) and package a `Result`. Cached.

        keep: ALL (the default) = fully symbolic, [] (or None, its
        alias) = fully numeric, [names] = hybrid.

        A keep=ALL solve over a large symbol space is ALWAYS refused: it cannot
        finish (a direct symbolic determinant does not terminate at this size),
        so running it is never what the caller wanted.

        progress: optional callable(done, total) over grid points -- a hybrid
        solve's cost IS the grid, and its size is known up front, so this is real
        progress rather than a spinner. Only the interpolation path reports; a
        direct symbolic determinant has no interior to report from.

        max_seconds additionally caps *hybrid* solves, which are slow but finite.
        It defaults to None -- no cap -- because a long hybrid solve is often
        exactly what the user asked for: keeping all twelve conductances of a
        two-stage amplifier symbolic takes ~250 s and is the tool's headline
        result. Set it when a front end would rather not block.
        """
        # norm_keep, NOT `keep or ()`: ALL and [] are opposites and both used to
        # hash to (), so a numeric result could be served for a symbolic request.
        key = self._key("solve", inp, out, norm_keep(keep),
                        float(fmin), float(fmax), int(points),
                        bool(reference))
        if key not in self._cache:
            self._guard_cost(inp, out, keep, max_seconds)
            H = self._analyzer_ready().tf(inp, out, keep=keep,
                                          progress=progress)
            self._cache[key] = self._assemble(
                H, inp, out, keep, reference=reference,
                fmin=fmin, fmax=fmax, points=points)
        return self._cache[key]

    @property
    def probes(self) -> list[str]:
        """Loop-gain probe candidates: ANY vsource branch is a valid Tian
        probe (Spectre's stb accepts any voltage source, not only an
        analogLib iprobe). iprobe-tagged instances come first -- they
        declare intent -- followed by every other vsource."""
        tagged, rest = [], []
        for d in self._run.flat.devices:
            if d.device_type != "vsource":
                continue
            if getattr(d, "meta", {}).get("cell") == "iprobe":
                tagged.append(d.name)
            else:
                rest.append(d.name)
        return tagged + rest

    @property
    def tagged_probes(self) -> list[str]:
        """iprobe-tagged vsources only — declared stb intent. The Tool
        dropdown filters the loop-analysis family on this, while `probes`
        keeps offering every vsource as a Tian candidate."""
        return [d.name for d in self._run.flat.devices
                if d.device_type == "vsource"
                and getattr(d, "meta", {}).get("cell") == "iprobe"]

    @property
    def ports(self) -> list[str]:
        """Impedance-port candidates: isources first (a 0 A isource is the
        parallel Thevenin port marker), then vsources (series ports --
        opened automatically by Analyzer.impedance)."""
        cur, volt = [], []
        for d in self._run.flat.devices:
            if d.device_type == "isource":
                cur.append(d.name)
            elif d.device_type == "vsource":
                volt.append(d.name)
        return cur + volt

    def analyses(self) -> list[str]:
        """What simulator truth the run carries (informative)."""
        try:
            return self._run.analyses()
        except Exception:
            return []

    def impedance_result(self, spec, keep=(), form: str = "exact", *,
                          tol_db: float = 0.5, mag_db: float = 1.0,
                          phase_deg: float = 5.0, strategy=None,
                          strategy_opts=None, fmin=None, fmax=None,
                          points: int = 400, max_elements=None,
                          progress=None, note=None) -> Result:
        """Driving-point impedance packaged as a Result. `spec` is a
        port-marker instance name (isource = parallel port, vsource =
        series port), or a dict: {"node": n} for Z(n -> ground),
        {"node": n, "ref": m} for the impedance between two nets,
        {"seen_by": inst} for what a two-terminal element sees at its
        terminals -- the element removed, sources zeroed (Thevenin
        looking outward).

        `form` gives the tool the Transfer solve's three contracts:
        "exact" (default), "simplified" (full order: prune coefficient
        terms within mag_db/phase_deg over [fmin, fmax]), "lowest"
        (reduced order under the tolerance strategy). The lowest form
        works by injecting a unit probe isource at the port -- Z
        becomes an ordinary transfer function and the reactance
        pursuit, the collapse, the snap and the certificate run
        exactly as they do for the Transfer tool. It needs a
        ground-referenced port; between-two-nets specs offer exact
        and full order.

        When the run carries an xf result with a marker port's
        transfer, the exact form overlays it as the simulator truth;
        everything else plots the model alone (never a show-stopper).
        Cached."""
        from .engine.mna import MnaError

        spec, node, ref, disable, label, terms_of = self._resolve_z(spec)
        # the applied reduction rewired its grounded nets onto ground:
        # Z there is 0 by construction and the MNA row is gone -- say
        # so instead of 'node not found'
        red_nodes = set((self._reduction or {}).get("nodes") or ())
        involved = ([n for n in (node, ref) if n] if node is not None
                    else list(terms_of.get(spec.get("port"), {}).values()))
        hit = sorted(set(involved) & red_nodes)
        if hit:
            raise MnaError(
                f"impedance: {', '.join(hit)} is AC-grounded by the "
                f"applied reduction -- Z there is 0 by construction. "
                f"Untick the net in the Reduce bench and Apply, or "
                f"Revert, to measure it.")

        opts = dict(strategy_opts or {})
        key = self._key("impedance", tuple(sorted(spec.items())),
                        norm_keep(keep), form, tol_db, mag_db, phase_deg,
                        strategy, tuple(sorted(opts.items())), fmin, fmax)
        if key in self._cache:
            return self._cache[key]
        an = self._analyzer_ready()
        lo = float(fmin) if fmin is not None else 1e3
        hi = float(fmax) if fmax is not None else             (1e7 if form == "lowest" else 1e9)
        if disable and all(p.inst != disable[0] for p in an.primitives):
            raise MnaError(
                f"impedance: {disable[0]} is not in the working "
                f"circuit (lumped or removed by the applied "
                f"reduction) -- Revert to look from its terminals.")
        if form == "lowest":
            r = self._impedance_lowest(
                spec, keep, an, node, ref, disable, label,
                tol_db=tol_db, mag_db=mag_db, phase_deg=phase_deg,
                strategy=strategy, opts=opts, fmin=lo, fmax=hi,
                points=points, max_elements=max_elements,
                progress=progress, note=note)
            self._cache[key] = r
            return r
        # the progress callback reaches the solver: without it every
        # impedance solve filed its whole life under 'preparing' and
        # the live estimate had nothing to refine from (field report:
        # 11 s promised, 91.6 s delivered, no phase to blame)
        if "port" in spec:
            tf = an.impedance(port=spec["port"], keep=keep,
                              progress=progress)
        else:
            tf = an.impedance(node=node, ref=ref, keep=keep,
                              disable=disable, progress=progress)
        if form == "simplified":
            H = tf
            tf = H.simplify(mag_tol_db=mag_db, phase_tol_deg=phase_deg,
                            fmin=lo, fmax=hi)
        r = self._assemble(tf, spec.get("port") or node, label, keep,
                           reference=False,
                           fmin=(lo if form == "simplified" else 1e3),
                           fmax=(hi if form == "simplified" else 1e9),
                           points=points)
        if form == "simplified":
            r.simplified = True
            r.mag_err_db = float(tf.achieved_mag_err_db)
            r.phase_err_deg = float(tf.achieved_phase_err_deg)
            r.n_terms_full = _n_terms(H)
            r.band_fmin, r.band_fmax = lo, hi
            # a simplified Z is judged against the exact one: the
            # reference curve IS the exact solution, on this grid
            try:
                full = self.impedance_result(dict(spec), keep=())
                r.h_ref = np.asarray(full.tf.numeric(r.freqs))
                r.ref_label = "exact Z"
            except Exception:                     # noqa: BLE001
                pass
        if "port" in spec and form == "exact":
            port = spec["port"]
            try:
                import warnings as _w

                with _w.catch_warnings():
                    _w.simplefilter("ignore")
                    xf = self._run.xf()
                if port in xf.transfers:
                    r.freqs = np.asarray(xf.freq, dtype=float)
                    r.h = np.asarray(tf.numeric(r.freqs))
                    r.h_ref = np.asarray(xf.tf(port))
                    r.ref_label = f"xf sim  Z via {port}"
            except Exception:
                pass                  # no xf truth in this run -- fine
        self._cache[key] = r
        return r

    def _resolve_z(self, spec):
        """One resolution for every Z consumer -- the solve, the
        pursuit, the ranking: (spec, node, ref, disable, label,
        terms_of). A marker port resolves node/ref lazily (the
        analyzer owns _port_nets); node stays None for it here."""
        from .engine.mna import MnaError

        if isinstance(spec, str):
            spec = {"port": spec}
        gnd = set(self.ground) | {"0"}
        terms_of = {d.name: dict(d.terminals)
                    for d in self._run.flat.devices}
        disable: tuple = ()
        node = ref = None
        if "seen_by" in spec:
            inst = spec["seen_by"]
            terms = terms_of.get(inst)
            if terms is None:
                raise MnaError(f"impedance: unknown instance {inst!r}")
            nets = list(dict.fromkeys(terms.values()))
            if len(nets) != 2:
                raise MnaError(f"impedance: {inst} is not a two-terminal "
                               f"element")
            a, b = nets
            if a in gnd:
                a, b = b, a
            if a in gnd:
                raise MnaError(f"impedance: both terminals of {inst} "
                               f"are ground")
            node, ref = a, (None if b in gnd else b)
            disable = (inst,)
            label = f"Z seen by {inst}"
        elif "node" in spec:
            node = spec["node"]
            ref = spec.get("ref") or None
            if ref in gnd:
                ref = None
            if node in gnd:
                raise MnaError("impedance: the node is ground")
            label = f"Z({node})" if ref is None else f"Z({node}, {ref})"
        else:
            label = f"Z({spec['port']})"
        return spec, node, ref, disable, label, terms_of

    def _z_probe_analyzer(self, spec, an=None):
        """(an2, probe, node): the working analyzer with a unit probe
        isource at the spec's port -- the ground-referenced form the
        single-node Z machinery (pursuit, ranking) shares. Raises for
        between-two-nets specs."""
        import copy

        from .engine.mna import MnaError
        from .engine.primitives import Primitive

        an = an if an is not None else self._analyzer_ready()
        spec, node, ref, disable, _label, _t = self._resolve_z(spec)
        if "port" in spec:
            node, ref, series = an._port_nets(spec["port"])
            disable = (spec["port"],) if series else ()
        if node is None or ref is not None:
            raise MnaError(
                "impedance: this needs a ground-referenced port -- "
                "between-two-nets impedance offers Exact and "
                "Simplified · full order.")
        off = set(disable)
        probe = "ZPROBE"
        prims = [p for p in an.primitives if p.inst not in off]
        prims.append(Primitive(inst=probe, param="", kind="isrc",
                               nodes=(an.flat.ground[0], node)))
        an2 = copy.copy(an)
        an2.primitives = prims
        return an2, probe, node

    def rank_symbols_z(self, spec, *, metric: str = "complex",
                       fmin: float | None = None,
                       fmax: float | None = None):
        """Band-sensitivity ranking for an IMPEDANCE response: the
        keep table's contributors must be Z's, not the transfer's --
        Z at a bias node is shaped by a different set than the
        VIND->vout path (field report: the ranking led with the
        transfer's gm while Z(net3) was built on another device).
        Same probe injection as the lowest-order solve; same list
        shape as rank_symbols."""
        an2, probe, node = self._z_probe_analyzer(spec)
        bs = an2.band_sensitivities(probe, node, metric=metric,
                                    fmin=fmin, fmax=fmax)
        return [(name, float(score), float(bs.peak_frequency(name)))
                for name, score in bs.rank(fmin, fmax)]

    def _impedance_lowest(self, spec, keep, an, node, ref, disable,
                          label, *, tol_db, mag_db, phase_deg, strategy,
                          opts, fmin, fmax, points, max_elements,
                          progress, note) -> Result:
        """Lowest-order Z: a unit probe isource at the port makes Z an
        ordinary transfer function -- the same MNA system, the same
        z-vector impedance() builds -- so the reactance pursuit, the
        coefficient collapse, the snap and the order certificate run
        exactly as for the Transfer tool. Ground-referenced ports
        only: the pursuit reads one output node."""
        import copy

        from .analysis.criteria import make_criterion
        from .analysis.simplify import snap_unity
        from .engine.mna import MnaError
        from .engine.primitives import Primitive

        an2, probe, node = self._z_probe_analyzer(spec, an)

        crit = make_criterion(strategy=strategy, strategy_opts=opts,
                              eps=None, tol_db=tol_db)
        if max_elements is not None and crit.cap is not None:
            crit.cap = max_elements
        if crit.name:
            mag_db, phase_deg = crit.collapse_budgets()
        H, red = an2.reduced_tf(probe, node, keep, tol_db=tol_db,
                                fmin=fmin, fmax=fmax,
                                max_elements=max_elements,
                                phase_tol_deg=phase_deg,
                                strategy=strategy, strategy_opts=opts,
                                progress=progress, note=note)
        Hs = H.simplify(mag_tol_db=mag_db, phase_tol_deg=phase_deg,
                        fmin=fmin, fmax=fmax)
        full = self.impedance_result(dict(spec), keep=())
        Hs, snap_k, snap_err = snap_unity(Hs, crit, full.freqs, full.h)
        r = self._assemble(Hs, probe, label, keep, reference=False,
                           fmin=fmin, fmax=fmax, points=points)
        # the reference curve IS the exact solution, on this grid --
        # the reduced model is read against the truth it approximates
        try:
            r.h_ref = np.asarray(full.tf.numeric(r.freqs))
            r.ref_label = "exact Z"
        except Exception:                         # noqa: BLE001
            pass
        r.snap_factor = float(snap_k)
        band_err = float(red.errors_db[-1]) if red.errors_db else             float(red.baseline_db)
        r.simplified = True
        r.reduced_order = True
        score, s_unit, mag_err = crit.score_fields(
            band_err, float(Hs.achieved_mag_err_db))
        if score is not None:
            r.band_score, r.band_score_unit = score, s_unit
        r.mag_err_db = float(mag_err)
        r.phase_err_deg = float(Hs.achieved_phase_err_deg)
        r.n_terms_full = _n_terms(H)
        r.band_fmin, r.band_fmax = float(fmin), float(fmax)
        if strategy is not None:
            r.strategy = strategy
        r.enforced_fmin, r.enforced_fmax = float(fmin), float(fmax)
        r.warnings.insert(0, crit.headline(red, band_err, fmin, fmax,
                                           tol_db=tol_db)
                          if not crit.name
                          else crit.headline(red, band_err, fmin, fmax))
        det = crit.details(red, band_err, fmin, fmax) if crit.name else []
        try:
            from .analysis.certificate import order_certificate as _cert

            cert = _cert(full.freqs, full.h, fmin, fmax,
                         poles_hz=full.poles_hz, zeros_hz=full.zeros_hz)
            det += self._certificate_details(
                r, None, None, fmin, fmax, crit.eps_equivalent(),
                len(red.selected), cert=cert)
        except Exception:                             # noqa: BLE001
            pass                        # advisory, never a show-stopper
        if snap_err is not None:
            det.append(
                f"coefficient snap: the DC gain's leading numeral "
                f"{snap_k:.4g} (a collapsed ratio of unkept parameters "
                f"on the path) divided out -- a uniform {snap_err:.3g} "
                f"dB shift, measured within the budget against the full "
                f"response")
        r.details = det
        return r

    def stb_probe(self) -> str | None:
        """The run's DESIGNATED stb probe (CIN name), when discoverable
        from the psfascii header or the run's netlist. None otherwise."""
        try:
            return self._run.stb_probe()
        except Exception:
            return None

    @property
    def has_stb(self) -> bool:
        """Whether the run carries Spectre stb results — the simulator
        ground truth the loop-analysis benches are gated on."""
        if self._has_stb is None:
            try:
                self._run.stb()
                self._has_stb = True
            except Exception:
                self._has_stb = False
        return self._has_stb

    def declare_ac_loop_gain(self, out_net: str | None):
        """The user's EXPLICIT statement that the run's AC data is a
        return-ratio capture: v(out_net)/v(input) is then the loop-gain
        reference, margins in the stb convention computed from it. None
        withdraws the declaration. This is the only way the loop benches
        open without stb results — a reconstructed loop gain is never
        shown without simulator ground truth to check it against."""
        if out_net != self.ac_loop_gain:
            self._cache.clear()      # cached overlays describe the other truth
        self.ac_loop_gain = out_net

    def loop_reference(self):
        """The run's loop-gain ground truth ALONE: (freqs, T, label,
        margins) from stb results, else from the declared-AC curve,
        else None. margins carries pm_deg/pm_freq_hz/gm_db/gm_freq_hz
        for the plot annotation, extracted from the curve itself. The
        loop benches' first light: simulator truth needs no model
        solve."""
        fr, h, _m, label = self._stb_reference(None)
        if fr is None:
            return None
        from .analysis.sensitivity import loop_margins

        pm, f1, gm, f2 = loop_margins(fr, h)
        margins = types.SimpleNamespace(pm_deg=pm, pm_freq_hz=f1,
                                        gm_db=gm, gm_freq_hz=f2)
        return fr, h, label, margins

    def _stb_reference(self, points_freqs):
        """(freqs, loopGain, margins, label) from the run's stb results,
        else from the DECLARED return-ratio AC data, else (None,)*4."""
        try:
            stb = self._run.stb()
            return (np.asarray(stb.freq, dtype=float),
                    np.asarray(stb.loop_gain),
                    stb, "Spectre stb loopGain")
        except Exception:
            pass
        if self.ac_loop_gain:
            try:
                fr, packed = self._reference(self.suggested_input(),
                                             self.ac_loop_gain)
                if packed is not None:
                    h = np.asarray(packed[0])
                    fr = np.asarray(fr, dtype=float)
                    from .analysis.sensitivity import loop_margins

                    pm, _f1, gm, _f2 = loop_margins(fr, h)
                    obj = types.SimpleNamespace(phase_margin_deg=pm,
                                                gain_margin_db=gm)
                    return (fr, h, obj,
                            f"AC declared as T  v({self.ac_loop_gain})")
            except Exception:
                pass
        return None, None, None, None

    def loop_gain(self, probe: str, keep=(), *, reference: bool = True,
                  fmin: float = 1.0, fmax: float = 1e10,
                  points: int = 600, form: str = "exact",
                  tol_db: float = 0.5, mag_db: float = 1.0,
                  phase_deg: float = 5.0, strategy=None,
                  strategy_opts=None, band_fmin=None, band_fmax=None,
                  max_elements=None, note=None,
                  max_seconds: float | None = None,
                  progress=None) -> Result:
        """Tian loop gain at `probe` packaged as a Result: Bode of T with
        the run's Spectre stb overlay when available, phase/gain margins in
        the stb convention, and stability advisories. Cached.

        `form` climbs the same ladder as the Transfer tool: "exact"
        (default), "simplified" (full order: prune T's coefficient
        terms within mag_db/phase_deg over [band_fmin, band_fmax]),
        "lowest" (reduced order under the tolerance strategy; the
        stability strategy gates on the margins themselves). Margins
        and advisories are then those of the simplified T -- the
        honest readout of the form on screen.

        The default band starts at 1 Hz so the +180-deg DC phase reference
        unwraps correctly even for sub-kHz dominant poles."""
        opts = dict(strategy_opts or {})
        key = self._key("loopgain", probe, norm_keep(keep),
                        float(fmin), float(fmax), int(points),
                        self.ac_loop_gain, form, tol_db, mag_db,
                        phase_deg, strategy,
                        tuple(sorted(opts.items())),
                        band_fmin, band_fmax)
        if key in self._cache:
            return self._cache[key]
        # gated for EVERY form: the lowest order zeroes reactance
        # VALUES but keeps the ticked symbols symbolic, so its final
        # Tian solve carries the same grid as the exact hybrid
        self._guard_loop_cost(probe, keep, max_seconds)

        lo = float(band_fmin) if band_fmin is not None else 1e3
        hi = float(band_fmax) if band_fmax is not None else 1e7
        red = crit = None
        n_terms_full = None
        if form == "lowest":
            from .analysis.criteria import make_criterion
            from .analysis.loopgain import reduced_loop_gain

            crit = make_criterion(strategy=strategy, strategy_opts=opts,
                                  eps=None, tol_db=tol_db)
            if max_elements is not None and crit.cap is not None:
                crit.cap = max_elements
            cmag, cph = (crit.collapse_budgets() if crit.name
                         else (mag_db, phase_deg))
            T, red = reduced_loop_gain(self._analyzer_ready(), probe,
                                       keep=keep, tol_db=tol_db,
                                       fmin=lo, fmax=hi,
                                       max_elements=max_elements,
                                       phase_tol_deg=cph,
                                       strategy=strategy,
                                       strategy_opts=opts,
                                       progress=progress, note=note)
            n_terms_full = _n_terms(T)
            T = T.simplify(mag_tol_db=cmag, phase_tol_deg=cph,
                           fmin=lo, fmax=hi)
        else:
            T = self._analyzer_ready().loop_gain(probe, keep=keep,
                                                 progress=progress)
            if form == "simplified":
                H0 = T
                n_terms_full = _n_terms(H0)
                T = T.simplify(mag_tol_db=mag_db, phase_tol_deg=phase_deg,
                               fmin=lo, fmax=hi)
        freqs = np.logspace(math.log10(fmin), math.log10(fmax), points)
        h_ref = ref_label = None
        stb_obj = None
        if reference:
            fr, lg, stb_obj, ref_label = self._stb_reference(freqs)
            if fr is not None:
                freqs = fr                     # point-aligned overlay
                h_ref = lg
        h = np.asarray(T.numeric(freqs))

        # margins from the model on a dense grid of its own -- the display /
        # overlay grid (e.g. a 20-per-decade stb sweep) is too coarse to place
        # the crossing frequencies accurately
        fd = np.logspace(math.log10(fmin), math.log10(fmax), 4001)
        from .analysis.sensitivity import loop_margins

        pm, fpm, gm, fgm = loop_margins(fd, np.asarray(T.numeric(fd)))
        warns: list[str] = []
        if pm is None:
            warns.append("no unity-gain crossing of |T| in the band: "
                         "margins undefined here")
        elif pm <= 0:
            warns.append(f"UNSTABLE: phase margin {pm:.1f} deg")
        elif pm < 45:
            warns.append(f"low phase margin: {pm:.1f} deg")
        if reference and h_ref is None:
            warns.append("no stb reference in this run (model only)")
        if stb_obj is not None and pm is not None and \
                stb_obj.phase_margin_deg is not None:
            dpm = pm - stb_obj.phase_margin_deg
            if abs(dpm) > 1.0:
                warns.append(f"model PM deviates from Spectre stb by "
                             f"{dpm:+.2f} deg")

        dc = _numeric_dc(T)
        dc_db = db(dc)
        r = Result(
            inp=probe, out=f"T@{probe}",
            keep=norm_keep_list(keep),
            dc_gain=dc, dc_gain_db=dc_db,
            poles_hz=T.poles(), zeros_hz=T.zeros(), n_terms=_n_terms(T),
            tf_latex=sp.latex(T.expr), dc_gain_latex=sp.latex(T.dc_gain()),
            freqs=freqs, h=h, h_ref=h_ref, ref_label=ref_label,
            warnings=warns, tf=T,
            pm_deg=pm, pm_freq_hz=fpm, gm_db=gm, gm_freq_hz=fgm,
        )
        if form != "exact":
            r.simplified = True
            r.mag_err_db = float(T.achieved_mag_err_db)
            r.phase_err_deg = float(T.achieved_phase_err_deg)
            r.n_terms_full = n_terms_full
            r.band_fmin, r.band_fmax = lo, hi
        if red is not None:
            r.reduced_order = True
            band_err = float(red.errors_db[-1]) if red.errors_db else                 float(red.baseline_db)
            score, s_unit, mag_err = crit.score_fields(
                band_err, float(T.achieved_mag_err_db))
            if score is not None:
                r.band_score, r.band_score_unit = score, s_unit
            r.mag_err_db = float(mag_err)
            if strategy is not None:
                r.strategy = strategy
            r.enforced_fmin, r.enforced_fmax = lo, hi
            r.warnings.insert(
                0, crit.headline(red, band_err, lo, hi) if crit.name
                else crit.headline(red, band_err, lo, hi, tol_db=tol_db))
            if crit.name:
                r.details = crit.details(red, band_err, lo, hi)
        self._cache[key] = r
        return r

    def assess_probe(self, probe: str, **kw):
        """Grade a designated stb probe (docs/loopgain-plan.md Sec. 9):
        margins-vs-closed-loop-pole consistency plus a per-device
        visibility scan naming loop dynamics the probe cannot see (e.g.
        the CMFB loop seen from a DM probe). Returns a ProbeReport whose
        .verdict() is the one-line summary. Cached."""
        key = self._key("adequacy", probe)
        if not kw and key in self._cache:
            return self._cache[key]
        report = self._analyzer_ready().assess_probe(probe, **kw)
        if not kw:                       # kw (grids, eps) may be unhashable
            self._cache[key] = report
        return report

    def template_form(self, inp: str, out: str, keep=ALL, *,
                      budget: float = 0.05):
        """The solved TF in the standard multistage form designers read:
        A0, GBW, and one short formula per pole/zero with the root
        displacement it accepted (analysis/template.py). Cached."""
        key = self._key("template", inp, out, norm_keep(keep), budget)
        if key in self._cache:
            return self._cache[key]
        from .analysis.template import template_form as _tpl

        tpl = _tpl(self.solve(inp, out, keep).tf, budget=budget)
        self._cache[key] = tpl
        return tpl

    #: attach_template skips TFs above this many monomials: the per-root
    #: formula pruning walks additive terms, and a fully symbolic solve of a
    #: real circuit has far too many for that walk to finish interactively
    TEMPLATE_MAX_TERMS = 1000

    def attach_template(self, result):
        """Fill result.template_text with the designer-form summary — A0,
        GBW, one formula per pole/zero — computed from the result's OWN tf,
        so a simplified or reduced result templates its pruned form, not the
        exact one it came from. Returns the result either way: the template
        is a bonus, never a reason for a solve to fail."""
        if result.tf is None or result.template_text is not None:
            return result
        if result.n_terms > self.TEMPLATE_MAX_TERMS:
            return result
        from .analysis.template import template_form as _tpl

        try:
            result.template_text = _tpl(result.tf).describe()
        except Exception:
            pass
        return result

    def scan_ac_grounds(self, inp: str, out: str, *,
                        budget_db: float = 0.1, strategy=None,
                        strategy_opts=None, fmin=None, fmax=None, **kw):
        """Rank the circuit's mirror/bias nodes by the EXACT error that
        declaring them AC grounds would introduce, and recommend the
        largest jointly-safe set (analysis/acground.py). With a
        `strategy`, pricing and gating run under the SAME BandCriterion
        as the order reduction, over the user's band -- one contract,
        one unit; `budget_db` gates only the legacy no-strategy path.
        Cached."""
        opts = dict(strategy_opts or {})
        include = tuple(kw.pop("include", ()) or ())
        key = self._key("acground", inp, out, budget_db, strategy,
                        tuple(sorted(opts.items())), fmin, fmax, include)
        if not kw and key in self._cache:
            return self._cache[key]
        crit, freqs = self._band_criterion(strategy, opts, fmin, fmax)
        if freqs is not None:
            kw.setdefault("freqs", freqs)
        rep = self._analyzer_ready().scan_ac_grounds(
            inp, out, budget_db=budget_db, criterion=crit,
            include=include, **kw)
        if set(kw) <= {"freqs"}:
            self._cache[key] = rep
        return rep

    def suggest_story_keep(self, inp: str, out: str, *,
                           fmin: float = 1e3, fmax: float = 1e7,
                           tol_db: float = 1.0,
                           max_symbols: int = 5) -> list[str]:
        """The keep set for a LOWEST-ORDER solve: the letters of the story.

        A lowest-order model is A0 and a dominant pole; the useful keeps
        are exactly the symbols that will appear in those two expressions.
        The band-sensitivity ranking answers a different question (what
        shapes the response anywhere in the band -- right for full order),
        and every extra keep multiplies the solve grid, so a large ranked
        set buys cost without letters.

        Composition: the matching pursuit names the reactances that
        survive the reduction over [fmin, fmax] (keeping them also
        PROTECTS them from being dropped); the sensitivity ranking over
        the same band, restricted to non-reactive symbols, names the
        conductances that set A0 and the pole. Parasitic reactances are
        excluded on purpose -- keeping one would force the order up.
        Cached."""
        key = self._key("storykeep", inp, out, float(fmin),
                        float(fmax), float(tol_db), max_symbols)
        if key in self._cache:
            return list(self._cache[key])
        from .analysis.sensitivity import _reactive_symbols

        an = self._analyzer_ready()
        red = an.dominant_reactances(inp, out, tol_db, fmin, fmax)
        reacts = list(red.selected)
        reactive = _reactive_symbols(an)
        bs = an.band_sensitivities(inp, out, fmin=fmin, fmax=fmax)
        conds = [n for n, _ in bs.rank(fmin, fmax)
                 if n not in reactive and n not in reacts]
        keep = reacts + conds[:max(0, max_symbols - len(reacts))]
        self._cache[key] = list(keep)
        return keep

    # ---------------------------------------------------- circuit reduction
    # The session holds ONE working circuit, in one of two named states:
    # "as imported" (straight from the CIN + OP) or "reduced" (chosen bias
    # nodes declared AC grounds, the controlled sources that kills removed,
    # parallel ground-referred passives lumped). Grounding is the only
    # approximation in that chain and its cost is MEASURED at apply time;
    # everything after it is exact for the rewritten circuit. Per-session by
    # design: declaring vbn an AC ground changes every subsequent analysis,
    # which is also the honest physics.

    @property
    def circuit_state(self) -> str:
        return "reduced" if self._reduction is not None else "as imported"

    def _band_criterion(self, strategy, strategy_opts, fmin, fmax):
        """(criterion, freqs) for a contract-priced scan: the criterion
        from the strategy surface, the frequency grid spanning the
        user's band. (None, None) when no strategy is given -- the
        legacy dB-budget path."""
        if strategy is None:
            return None, None
        from .analysis.criteria import make_criterion

        crit = make_criterion(strategy=strategy,
                              strategy_opts=strategy_opts)
        freqs = (np.geomspace(float(fmin), float(fmax), 41)
                 if fmin is not None and fmax is not None else None)
        return crit, freqs

    def acground_joint(self, inp: str, out: str, nodes, *,
                       strategy=None, strategy_opts=None,
                       fmin=None, fmax=None) -> dict:
        """Price grounding `nodes` TOGETHER — whatever set the user
        actually ticks, not just the scan's recommendation. Returns
        {"worst_db", "worst_deg", "score"}; the score is x-budget under
        the given contract, None without one."""
        from .analysis.acground import joint_metrics

        crit, freqs = self._band_criterion(strategy, strategy_opts,
                                           fmin, fmax)
        an = self._analyzer_ready()
        return joint_metrics(an.primitives, an.flat.ground, inp, out,
                             nodes, alias=an._alias, freqs=freqs,
                             criterion=crit)

    def reduction_joint(self, inp: str, out: str, nodes, removals=(), *,
                        strategy=None, strategy_opts=None,
                        fmin=None, fmax=None) -> dict:
        """Price the ticked rewrite -- groundings AND deletions --
        TOGETHER, by re-solving the rewritten circuit (no rank-one
        shortcut composes across both rewrites). Returns {"worst_db",
        "worst_deg", "score"}; the score is x-budget under the given
        contract, None without one."""
        crit, bandf = self._band_criterion(strategy, strategy_opts,
                                           fmin, fmax)
        f = (np.asarray(bandf, dtype=float) if bandf is not None
             else np.geomspace(1.0, 1e10, 41))
        hb, hr = self.reduction_responses(nodes, inp, out, f,
                                          removals=removals)
        hb = np.asarray(hb, dtype=complex)
        hr = np.asarray(hr, dtype=complex)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = hr / hb
            db = 20 * np.log10(np.abs(ratio))
            deg = np.degrees(np.angle(ratio))
        out_d = {"worst_db": float(np.nanmax(np.abs(db))),
                 "worst_deg": float(np.nanmax(np.abs(deg))),
                 "score": None}
        if crit is not None:
            out_d["score"] = float(crit.score(f, hb, hr))
        return out_d

    def _reduced_primitives(self, an, nodes, removals=()) -> list:
        """The reduction chain on `an`'s primitives: ground -> deactivate
        -> lump -> delete. Deterministic, so a rebuilt analyzer re-derives
        the same working circuit from the recorded node and removal sets.
        Removals are keyed by instance name AFTER the chain -- the removal
        scan priced them on exactly that circuit, lumped equivalents
        included."""
        from .analysis import tearing
        from .analysis.lumping import deactivate, lump_to_ground

        gnd = an.flat.ground
        lumped = an.primitives
        if nodes:
            grounded = tearing.ac_ground(an.primitives, gnd, nodes)
            live, _dead = deactivate(grounded, gnd)
            lumped, _rep = lump_to_ground(live, gnd)
        if self.mos_model == "lumped-gmb":
            # the COMPOSITION: ac_ground rewired the chosen nets to the
            # ground node, so gates that sat on internal bias nets now
            # meet the exact gm+gmb criterion ON THE REDUCED CIRCUIT --
            # the approximation was the reduction's, already measured
            # and reported; the bundle itself stays exact. Structural
            # grounding is input-independent, so these entries need no
            # input guard ("lumped (reduced)", skipped by it).
            from .models.small_signal import dc_nets, lump_gmb_primitives

            lumped = lump_gmb_primitives(lumped, dc_nets(an.flat),
                                         an.lumped_gmb,
                                         tag="lumped (reduced)")
        if removals:
            rm = set(removals)
            lumped = [p for p in lumped if p.inst not in rm]
        return lumped

    def preview_reduction(self, nodes, removals=()) -> dict:
        """What WOULD the chain do, without changing any state: primitive
        counts, the dead sources, the lump groups, the outright deletions.
        Pure inspection."""
        from .analysis import tearing
        from .analysis.lumping import deactivate, lump_report, lump_to_ground

        base = self._run.analyzer(cap_model=self.cap_model,
                                    mos_model=self.mos_model)
        for group in self._matches:
            base.match(*group)
        gnd = base.flat.ground
        nodes = list(nodes)
        if nodes:
            grounded = tearing.ac_ground(base.primitives, gnd, nodes)
            live, dead = deactivate(grounded, gnd)
            lrep = lump_report(live, gnd)
            lumped, _ = lump_to_ground(live, gnd)
        else:
            dead, live = [], list(base.primitives)
            lumped = live
            lrep = lump_report([], gnd)
        # deletions act on the chained circuit -- the removal scan priced
        # them there; a name that no longer exists deletes nothing
        rm = set(removals)
        removed = sorted({p.inst for p in lumped if p.inst in rm})
        lumped = [p for p in lumped if p.inst not in rm]
        pvals = {f"{p.param or p.kind}_{p.inst}": p.value
                 for p in base.primitives
                 if isinstance(p.value, (int, float))}
        return {
            "nodes": nodes,
            "removals": list(removals),
            "removed": removed,
            "prims_before": len(base.primitives),
            "prims_after": len(lumped),
            "dead_sources": [p.inst for p in dead],
            # per device: what died and what lives on. A device whose
            # gm died but whose gds and capacitances remain is a load,
            # not a removal; only a device with nothing live is gone
            "dead_primitives": [f"{p.param or p.kind}_{p.inst}"
                                for p in dead],
            "devices_silenced": sorted(
                {p.inst for p in dead}
                - {p.inst for p in live}),
            "devices_passive": sorted(
                {p.inst for p in dead}
                & {p.inst for p in live}),
            "lump_groups": [g.describe() for g in lrep.groups],
            "equivalents": [{"name": g.name, "kind": g.kind,
                             "node": g.node, "members": list(g.members),
                             "value": g.value,
                             # each member's own value, from the
                             # UNREDUCED circuit: op_values() describes
                             # the reduced one, where they are gone
                             "member_values": {
                                 m: pvals[m] for m in g.members
                                 if m in pvals}}
                            for g in lrep.groups if g.value is not None],
            "symbols_saved": lrep.symbols_saved,
        }

    def reduction_responses(self, nodes, inp: str, out: str, freqs,
                            removals=()):
        """(H_as_imported, H_reduced) over `freqs`: the numeric in->out
        response of the as-imported circuit (matches applied, no
        reduction) and of the circuit with `nodes` AC-grounded, dead
        sources removed, parallels lumped and the `removals` deleted --
        exactly what Apply would make the working circuit. What the
        Reduce bench PLOTS so a ticked set's cost is a curve inside the
        tolerance tube, not only a number. Both sets empty: both curves
        are the as-imported response."""
        from .analysis import tearing
        from .analysis.lumping import deactivate, lump_to_ground

        base = self._run.analyzer(cap_model=self.cap_model,
                                  mos_model=self.mos_model)
        for group in self._matches:
            lead = self.name_bearer(group)
            base.match(lead, *[m for m in group if m != lead])
        gnd = base.flat.ground
        f = np.asarray(freqs, dtype=float)
        h_base = tearing._numeric_response(base.primitives, gnd, inp, out,
                                           f, base._alias)
        nodes = list(nodes)
        removals = list(removals)
        if not nodes and not removals:
            return h_base, h_base
        prims = base.primitives
        if nodes:
            grounded = tearing.ac_ground(prims, gnd, nodes)
            live, _dead = deactivate(grounded, gnd)
            prims, _ = lump_to_ground(live, gnd)
        if removals:
            rm = set(removals)
            prims = [p for p in prims if p.inst not in rm]
        h_red = tearing._numeric_response(prims, gnd, inp, out, f,
                                          base._alias)
        return h_base, h_red

    def apply_reduction(self, nodes, *, inp: str, out: str,
                        removals=(), strategy=None, strategy_opts=None,
                        fmin=None, fmax=None) -> dict:
        """Make the reduced circuit THE working circuit for every analysis
        that follows, and measure what the rewrite cost end to end — full
        vs reduced numeric response over a wide grid, not the scans'
        estimates. `nodes` are AC-grounded, `removals` (instance names
        from the removal scan, lumped equivalents included) are deleted
        outright. Returns the summary the GUI banners."""
        from .analysis import tearing

        nodes = list(nodes)
        removals = list(removals)
        if not nodes and not removals:
            raise ValueError("apply_reduction: nothing to apply")
        base = self._run.analyzer(cap_model=self.cap_model,
                                    mos_model=self.mos_model)
        for group in self._matches:
            base.match(*group)
        gnd = base.flat.ground
        summary = self.preview_reduction(nodes, removals)

        self._reduction = {"nodes": nodes, "removals": removals}
        self._analyzer = None
        self._cache.clear()
        self._op_values = None       # OP symbols changed with the circuit
        an = self._analyzer_ready()          # builds the reduced circuit

        f = np.geomspace(1.0, 1e10, 41)
        a = tearing._numeric_response(base.primitives, gnd, inp, out, f,
                                      base._alias)
        b = tearing._numeric_response(an.primitives, gnd, inp, out, f,
                                      an._alias)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = b / a
            db = 20 * np.log10(np.abs(ratio))
            deg = np.degrees(np.angle(ratio))
        summary["worst_db"] = float(np.nanmax(np.abs(db)))
        summary["worst_deg"] = float(np.nanmax(np.abs(deg)))
        summary["inp"], summary["out"] = inp, out
        # the same change priced under the active contract, over the
        # user's band -- the x-budget figure every other approximation
        # reports
        crit, bandf = self._band_criterion(strategy, strategy_opts,
                                           fmin, fmax)
        summary["score"] = None
        if crit is not None:
            fb = bandf if bandf is not None else f
            ab = tearing._numeric_response(base.primitives, gnd, inp, out,
                                           fb, base._alias)
            bb = tearing._numeric_response(an.primitives, gnd, inp, out,
                                           fb, an._alias)
            summary["score"] = float(crit.score(fb, ab, bb))
            summary["criterion"] = crit.name or "dB band"
        self._reduction.update(summary)
        return dict(summary)

    def revert_reduction(self) -> None:
        """Back to the as-imported circuit. Every cache entry describes the
        other state, so the cache goes too."""
        self._reduction = None
        self._analyzer = None
        self._cache.clear()
        self._op_values = None       # OP symbols changed with the circuit

    def equivalent_elements(self) -> list:
        """The active reduction's lumped equivalents: each a dict with
        name (the symbol in H(s): Geq_I0_net8), kind, node, members
        (the symbols it replaced, e.g. gds_I0_M10) and value. Empty
        without a reduction. The lump is EXACT: parallel same-kind
        elements to ground sum."""
        if self._reduction is None:
            return []
        out = []
        for e in self._reduction.get("equivalents", []):
            d = dict(e)
            # the members as the join keys H(s) uses (dots -> underscores)
            d["members"] = [m.replace(".", "_") for m in e["members"]]
            d["member_values"] = {m.replace(".", "_"): v for m, v in
                                  (e.get("member_values") or {}).items()}
            out.append(d)
        return out

    def equivalent_definitions(self) -> list:
        """[(name, sympy expr over the member symbols, value)] for the
        active reduction's lumped equivalents: a G/C equivalent is the
        SUM of its members, an R equivalent the parallel combination.
        The member symbols are the engine's (positive=True), so the
        expression substitutes cleanly into a result's H(s)."""
        import sympy as sp

        out = []
        for e in self.equivalent_elements():
            syms = [sp.Symbol(m, positive=True) for m in e["members"]]
            if e["kind"] == "r":
                expr = 1 / sum(1 / s_ for s_ in syms)
            else:
                expr = sum(syms)
            out.append((e["name"], expr, e["value"]))
        return out

    def expand_equivalents(self, result):
        """A derived Result whose H(s) has every lumped equivalent
        replaced by its member expression, so A0, N(s), D(s), p1, z1
        read in the original device symbols. The member values join
        the value map (hover). Cached on the result; the result itself
        is returned when there is nothing to expand."""
        import sympy as sp
        from dataclasses import replace as _replace

        defs = self.equivalent_definitions()
        tf = getattr(result, "tf", None)
        if not defs or tf is None or not hasattr(tf, "expr"):
            return result
        cached = getattr(result, "_expanded_result", None)
        if cached is not None and cached[0] == [d[0] for d in defs]:
            return cached[1]
        by_name = {str(s_): s_ for s_ in tf.expr.free_symbols}
        subs = {}
        for name, expr, _val in defs:
            if name in by_name:
                subs[by_name[name]] = expr
        if not subs:
            return result
        new_expr = tf.expr.xreplace(subs)
        values = dict(getattr(tf, "values", {}) or {})
        symbols = dict(getattr(tf, "symbols", {}) or {})
        mvals = {}
        for e in self.equivalent_elements():
            mvals.update(e.get("member_values") or {})
        for name, expr, _val in defs:
            for s_ in expr.free_symbols:
                n = str(s_)
                symbols.setdefault(n, s_)
                if n in mvals and n not in values:
                    values[n] = float(mvals[n])
        try:
            new_tf = _replace(tf, expr=new_expr, values=values,
                              symbols=symbols)
        except TypeError:
            # a TransferFunction subclass without dataclass fields for
            # these: fall back to a plain TransferFunction
            from .engine.mna import TransferFunction
            new_tf = TransferFunction(expr=new_expr, values=values,
                                      symbols=symbols)
        new_result = _replace(result, tf=new_tf)
        try:
            object.__setattr__(result, "_expanded_result",
                               ([d[0] for d in defs], new_result))
        except Exception:                        # noqa: BLE001
            pass
        return new_result

    def reduction_summary(self) -> dict | None:
        """The apply_reduction summary of the active reduction, or None."""
        return dict(self._reduction) if self._reduction is not None else None

    def approximation_report(self, inp: str, out: str, *,
                             strategy=None, strategy_opts=None,
                             fmin: float = 1e3, fmax: float = 1e9,
                             result=None) -> dict:
        """The LEDGER: every approximation between the imported circuit
        and what the user is looking at, priced under ONE contract, and
        the honest total. Per-step scores are diagnostic; the totals
        are MEASURED end to end (full vs current response under the
        criterion), never summed -- errors do not add linearly, and a
        summed total would be a new lie.

        Entries in composition order: matches (when active), the
        applied AC-ground reduction (when active), the exact gmb bundle
        (no budget spent). circuit_score prices the working circuit
        against the imported one; with a reduced-order or collapsed
        `result`, solve_score carries that solve's own cost and
        grand_score prices the SHOWN response against the imported
        circuit -- the number that can exceed 1x while every step
        individually passed."""
        from .analysis import tearing
        from .analysis.criteria import make_criterion

        crit = make_criterion(strategy=strategy or "plain",
                              strategy_opts=strategy_opts)
        freqs = np.geomspace(float(fmin), float(fmax), 61)
        gnd = self._run.flat.ground

        def resp(an):
            return tearing._numeric_response(an.primitives, gnd, inp,
                                             out, freqs, an._alias)

        pristine = self._run.analyzer(cap_model=self.cap_model,
                                      mos_model=self.mos_model)
        working = self._analyzer_ready()
        h_pristine = resp(pristine)
        h_working = resp(working)
        entries = []
        if self._matches:
            matched = self._run.analyzer(cap_model=self.cap_model,
                                         mos_model=self.mos_model)
            for group in self._matches:
                matched.match(*group)
            matched.primitives = self._apply_match_values(matched)
            h_matched = resp(matched)
            entries.append({
                "step": f"matches ({len(self._matches)} group(s))",
                "score": float(crit.score(freqs, h_pristine, h_matched)),
                "exact": False})
        else:
            h_matched = h_pristine
        if self._reduction is not None:
            bits = []
            if self._reduction["nodes"]:
                bits.append("grounded "
                            + ", ".join(self._reduction["nodes"]))
            if self._reduction.get("removals"):
                bits.append("removed "
                            + ", ".join(self._reduction["removals"]))
            entries.append({
                "step": "reduction [" + "; ".join(bits) + "]",
                "score": float(crit.score(freqs, h_matched, h_working)),
                "exact": False})
        lump = working.lumped_gmb
        if lump:
            n_l = sum(1 for v in lump.values() if v.startswith("lumped"))
            n_d = len(lump) - n_l
            bits = ([f"ĝm on {n_l}"] if n_l else []) +                    ([f"inert gmbs dropped on {n_d}"] if n_d else [])
            entries.append({"step": "gmb bundle ("
                            + ", ".join(bits) + " device(s))",
                            "score": None, "exact": True})
        rep = {"criterion": crit.name or "dB band",
               "band": (float(fmin), float(fmax)),
               "entries": entries,
               "circuit_score": float(crit.score(freqs, h_pristine,
                                                 h_working))}
        if result is not None and getattr(result, "h", None) is not None                 and (getattr(result, "reduced_order", False)
                     or getattr(result, "simplified", False)):
            rf = np.asarray(result.freqs, dtype=float)
            m = (rf >= float(fmin)) & (rf <= float(fmax))
            if m.sum() >= 8:
                hp = tearing._numeric_response(
                    pristine.primitives, gnd, inp, out, rf[m],
                    pristine._alias)
                rep["solve_score"] = getattr(result, "band_score", None)
                rep["grand_score"] = float(
                    crit.score(rf[m], hp, np.asarray(result.h)[m]))
        return rep

    def scan_removals(self, inp: str, out: str, *,
                      budget_db: float = 0.1, strategy=None,
                      strategy_opts=None, fmin=None, fmax=None, **kw):
        """Which explicit elements can simply be DELETED: every netlist
        passive priced by the exact response shift its removal would cause
        (analysis/removal.py, the Lei & Wu setting-zero idea as a measured
        scan). With a `strategy`, pricing and gating run under the SAME
        BandCriterion as the order reduction, over the user's band.
        Cached."""
        opts = dict(strategy_opts or {})
        key = self._key("removal", inp, out, budget_db, strategy,
                        tuple(sorted(opts.items())), fmin, fmax)
        if not kw and key in self._cache:
            return self._cache[key]
        an = self._analyzer_ready()
        from .analysis.removal import scan_removals as _scan

        crit, freqs = self._band_criterion(strategy, opts, fmin, fmax)
        if freqs is not None:
            kw.setdefault("freqs", freqs)
        rep_ = _scan(an.primitives, an.flat.ground, inp, out,
                     budget_db=budget_db, alias=an._alias,
                     criterion=crit, **kw)
        if set(kw) <= {"freqs"}:
            self._cache[key] = rep_
        return rep_

    def pole_attribution(self, inp: str, *, n_poles: int = 4, **kw):
        """Which element establishes which pole, verified by nudging the
        top owner and re-rooting (analysis/attribution.py — the Manocha
        recursive-shunt intuition made exact). Cached; poles belong to the
        denominator, so only the input designation matters."""
        key = self._key("poleattr", inp, n_poles)
        if not kw and key in self._cache:
            return self._cache[key]
        from .analysis.attribution import pole_attribution as _attr

        atts = _attr(self._analyzer_ready().system(inp), n_poles=n_poles,
                     **kw)
        if not kw:
            self._cache[key] = atts
        return atts

    def explain_numerals(self, inp: str, out: str, keep=(), *,
                         progress=None, **kw):
        """Which collapsed parameters carry each numeral of the transfer
        function (analysis/explain.py). A hybrid solve substitutes every
        unkept parameter before solving, so the numbers in a simplified
        expression are sums of products whose names are gone; this ranks
        them back, per coefficient of N(s) and D(s), from one matrix
        inverse per s-sample. Kept symbols are excluded — they are
        already letters. Cached; `progress` stays outside the
        cache-bypass kwargs."""
        key = self._key("numerals", inp, out, tuple(sorted(keep)))
        if not kw and key in self._cache:
            return self._cache[key]
        from .analysis.explain import explain_coefficients
        from .engine.mna import hybrid_split

        sysm = self._analyzer_ready().system(inp)
        _, kept = hybrid_split(sysm, list(keep))
        stories = explain_coefficients(sysm, out, exclude=set(kept),
                                       progress=progress, **kw)
        if not kw:
            self._cache[key] = stories
        return stories

    def explain_per_numeral(self, inp: str, out: str, keep=(), *,
                            progress=None, fast: bool = False, **kw):
        """Per-numeral attribution: every collapsed numeral of the hybrid
        expression (coefficient of s^k · kept-monomial) with its
        contributors, from a derivative sweep over the hybrid grid
        (analysis/explain.py). On demand, cached; `progress` reports
        (grid point, total) and is deliberately outside the cache-bypass
        kwargs — a progress bar must not disable caching.

        fast=True runs the float64 circle kernel instead of the exact
        mpfr sweep (~20x, measured): the numeral slots and values come
        from the cached solve's exact expression, shares from the
        kernel, and any slot the kernel cannot confirm arrives with
        approx=True. It solves first if no solve is cached."""
        key = self._key("pernumeral", fast, inp, out,
                        tuple(sorted(keep)))
        if not kw and key in self._cache:
            return self._cache[key]
        sysm = self._analyzer_ready().system(inp)
        if fast:
            from .analysis.explain import explain_per_numeral_fast as _f

            # a lowest-order display leaves the plain solve uncached, so
            # this may BE a full hybrid solve — without the progress
            # thread the deep action sat on a dead bar for its duration
            r = self.solve(inp, out, keep=list(keep), progress=progress)
            stories = _f(sysm, out, keep, r.tf.expr,
                         progress=progress, **kw)
        else:
            from .analysis.explain import explain_per_numeral as _deep

            stories = _deep(sysm, out, keep, progress=progress, **kw)
        if not kw:
            self._cache[key] = stories
        return stories

    def cached_per_numeral(self, inp: str, out: str, keep=()):
        """The explain_per_numeral stories if already computed, else
        None — never computes. Exact stories win over fast ones when
        both exist."""
        tail = (inp, out, tuple(sorted(keep)),
                tuple(self._matches), self.circuit_state,
                self._match_policy)
        exact = self._cache.get(("pernumeral", False) + tail)
        if exact is not None:
            return exact
        return self._cache.get(("pernumeral", True) + tail)

    def cached_numerals(self, inp: str, out: str, keep=()):
        """The explain_numerals stories if already computed for this
        configuration, else None — never computes. The Expression view's
        numeral hover uses this to know whether to show contributors or
        the run-the-analysis prompt."""
        key = self._key("numerals", inp, out, tuple(sorted(keep)))
        return self._cache.get(key)

    def advise_split(self, inp: str, out: str, keep=(), **kw):
        """Should this solve be TORN, and where? Returns a SplitAdvice
        whose .verdict() is the one-line summary: the ranked cuts with
        their balance and keep-set split, plus -- when no cut pays as the
        circuit stands -- AC-ground designations that would fix the
        balance, each with the measured dB error it costs.

        Tearing pays only when the halves are balanced AND the keep-set
        is large enough that its exponential split beats the constant
        per-interface solve penalty; the advisory exists because that
        judgement is not obvious from the schematic. Cached."""
        key = self._key("split_advice", inp, out, norm_keep(keep))
        if not kw and key in self._cache:
            return self._cache[key]
        adv = self._analyzer_ready().advise_split(inp, out, keep, **kw)
        if not kw:
            self._cache[key] = adv
        return adv

    def suggest_compensation(self, probe: str, *, goal: str = "mfm",
                             pm_target: float = 60.0,
                             flat_tol_db: float = 1.0, exclude=(),
                             candidates=None, top: int = 5,
                             progress=None, **kw):
        """OP-invariant compensation suggestions at the designated loop
        probe (docs/compensation-synthesis-plan.md): sized C / series-RC
        branches ranked by area among goal achievers. goal="mfm" places the
        dominant closed-loop pair at Butterworth damping within the
        loop-gain bandwidth budget (the structured-design formulation);
        goal="pm" enforces the classic phase-margin floor; goal="spec"
        (Middlebrook) holds the peak sensitivity Ms = max|1/(1+T)| below
        `ms_target` (pass it via **kw) -- the discrepancy/tolerance target,
        = max|H/Hinf-1| in the feedthrough-free servo regime.

        `exclude`: instance names of EXISTING compensation branches to
        strip before suggesting (their removal is OP-invariant, so the
        reconstruction stays exact) -- the "re-compensate this amplifier"
        workflow. Returns analysis.compensate.Suggestion objects (plain
        data; .describe() renders a human line). Cached for the default
        candidate set."""
        from .analysis.compensate import suggest_compensation as _suggest
        from .engine.mna import build_mna

        an = self._analyzer_ready()
        prims = [p for p in an.primitives if p.inst not in set(exclude)]
        system = build_mna(prims, an.flat.ground, probe, an._alias)
        key = None
        # cache only the pristine default path: custom candidates or grid/
        # tolerance overrides (**kw) are not part of the key and must not
        # collide with it
        if candidates is None and not kw:
            key = self._key("suggest", probe, goal, pm_target,
                            flat_tol_db, tuple(sorted(exclude)), top)
            if key in self._cache:
                return self._cache[key]
        out = _suggest(system, probe, goal=goal, pm_target=pm_target,
                       flat_tol_db=flat_tol_db,
                       candidates=candidates, top=top,
                       torn=(prims, an.flat.ground, None),
                       cache=self._tear_cache, progress=progress, **kw)
        if key is not None:
            self._cache[key] = out
        return out

    def suggest_multi_compensation(self, probe: str, *, goal: str = "pm",
                                   k_max: int = 2, exclude=(),
                                   candidates=None, progress=None, **kw):
        """Grow a MULTI-branch (nested-Miller / NMC) compensation network at
        the probe, one OP-invariant branch at a time (analysis.compensate.
        suggest_multi_compensation): each step installs the least-area branch
        that most improves the goal given those already placed, the joint
        effect exact at every step (rank-k pole locus + Woodbury loop gain).
        Growth stops when the goal is met or a further branch would not pay
        its area. Use when one branch cannot reach the target -- otherwise it
        returns a single-branch network, same as suggest_compensation.

        `exclude`: existing compensation instances to strip first (the
        re-compensate workflow). Returns a MultiSuggestion (.describe()
        renders it). Not cached (multi-dimensional search; call directly)."""
        from .analysis.compensate import \
            suggest_multi_compensation as _multi
        from .engine.mna import build_mna

        an = self._analyzer_ready()
        prims = [p for p in an.primitives if p.inst not in set(exclude)]
        system = build_mna(prims, an.flat.ground, probe, an._alias)
        # the search itself is numeric (Woodbury + circle rooting); its
        # symbolic prelude (T0, the pole scale, the candidate screen) is
        # what costs, and it depends only on the circuit -- so serve it
        # from the session cache, keyed on the content fingerprint. A
        # re-run after a goal/target change then pays no determinants.
        return _multi(system, probe, goal=goal, k_max=k_max,
                      candidates=candidates,
                      torn=(prims, an.flat.ground, None),
                      cache=self._tear_cache, progress=progress, **kw)

    def _guard_cost(self, inp, out, keep, max_seconds) -> None:
        """Refuse a solve we can see will not finish, instead of hanging.

        Two different signals, because there are two solve paths:

        * hybrid/numeric -> interpolation, which `seconds` is calibrated for.
        * keep=ALL -> a DIRECT symbolic determinant, which the interp cost model
          does NOT describe. There `seconds` is an extrapolation of the wrong
          path, so we judge on `grid_size` (the size of the symbol space) and
          refuse to quote a wall-clock we cannot honestly predict.
        """
        # keep=ALL is checked even with no budget: it cannot finish, so "run it
        # anyway" is not a meaningful choice. A hybrid solve only gets capped if
        # the caller actually asked for a cap -- 250 s of hybrid solving is a
        # result, not a hang, and refusing it would block the tool's main use.
        if max_seconds is None and not is_all(keep):
            return
        try:
            est = self._analyzer_ready().estimate_solve_time(inp, out, keep)
        except Exception:
            return                      # an estimate is a courtesy, never a gate

        if is_all(keep):
            if est.grid_size <= _SYMBOLIC_GRID_LIMIT:
                return
            raise SolveTooLarge(
                f"{inp} → {out}, fully symbolic (keep=ALL): {len(est.kept_names)} "
                f"symbols, symbol space {est.grid_size:.3g}. The symbolic "
                f"determinant is intractable at this size — it will not finish.\n"
                f"Use suggest_keep()/plan_keep() to pick a keep set, keep=[] for "
                f"a numeric solve, or max_seconds=None to insist.")

        if est.seconds is None or est.seconds <= max_seconds:
            return
        raise SolveTooLarge(
            f"{inp} → {out}, keep={list(keep)}: estimated {est.seconds:.0f} s "
            f"(grid {est.grid_size:,}), over the {max_seconds:.0f} s budget.\n"
            f"Use suggest_keep()/plan_keep() to trim the keep set, or pass "
            f"max_seconds=None to run it anyway.")

    def simplify(self, inp: str, out: str, keep=ALL, *,
                 mag_db: float = 1.0, phase_deg: float = 5.0,
                 reference: bool = True, fmin: float = 1e3, fmax: float = 1e9,
                 points: int = 400, progress=None) -> Result:
        """Error-budgeted simplification of tf(inp, out, keep): prune negligible
        terms within `mag_db`/`phase_deg`. The Result carries the achieved error
        and the term count before/after pruning."""
        key = self._key("simplify", inp, out, norm_keep(keep),
                        mag_db, phase_deg, float(fmin), float(fmax))
        if key not in self._cache:
            H = self._analyzer_ready().tf(inp, out, keep=keep,
                                          progress=progress)
            Hs = H.simplify(mag_tol_db=mag_db, phase_tol_deg=phase_deg,
                            fmin=fmin, fmax=fmax)
            r = self._assemble(Hs, inp, out, keep, reference=reference,
                               fmin=fmin, fmax=fmax, points=points)
            r.simplified = True
            r.mag_err_db = float(Hs.achieved_mag_err_db)
            r.phase_err_deg = float(Hs.achieved_phase_err_deg)
            r.n_terms_full = _n_terms(H)
            r.band_fmin, r.band_fmax = float(fmin), float(fmax)
            self._cache[key] = r
        return self._cache[key]

    def order_certificate(self, inp: str, out: str, fmin: float,
                          fmax: float):
        """Loewner order certificate over [fmin, fmax]: how much order
        the band demands at a tolerance, from one cached wide numeric
        sweep (milliseconds per call after the first). See
        analysis/certificate.py — ships the doublet caveat."""
        from .analysis.certificate import order_certificate as _cert

        base = self.solve(inp, out, keep=[], reference=False,
                          fmin=1.0, fmax=1e10, points=800)
        return _cert(base.freqs, base.h, fmin, fmax,
                     poles_hz=base.poles_hz, zeros_hz=base.zeros_hz)

    def _certificate_details(self, r, inp, out, fmin, fmax, eps_eq,
                             n_sel, label="", cert=None) -> list[str]:
        """The order-certificate verdict + doublet caveat, shared by the
        anchored and strategy reduce reports (and the impedance lowest
        order, which brings its own certificate). Advisory: a
        certificate failure never breaks the solve that earned it."""
        try:
            if cert is None:
                cert = self.order_certificate(inp, out, fmin, fmax)
            r.certificate = cert
            k = cert.order_at(eps_eq)
            if n_sel <= k:
                det = [f"order-optimal for this band ({label})" if label
                       else "order-optimal for this band"]
            else:
                at = (f" for this band at {label}" if label
                      else " at comparable tolerance")
                det = [f"an abstract order-{k} fit exists{at}; physical "
                       f"elements need {n_sel} — the cost of named "
                       f"components"]
            if cert.doublets:
                d = cert.doublets[0]
                det.append(
                    f"pole/zero doublet near {eng(d[0], 'Hz')} "
                    f"(separation {d[2]:.1%}) — doublets affect "
                    f"settling, not this frequency response")
            return det
        except Exception:
            return []

    def reduce_solve(self, inp: str, out: str, keep=ALL, *,
                     tol_db: float = 0.5, max_elements: int | None = None,
                     mag_db: float = 1.0, phase_deg: float = 5.0,
                     reference: bool = True, fmin: float = 1e3, fmax: float = 1e7,
                     floor_db: float = 60.0,
                     floor_abs_db: float | None = None,
                     eps: float | None = None,
                     strategy: str | None = None,
                     strategy_opts: dict | None = None,
                     points: int = 400, progress=None,
                     note=None) -> Result:
        """Reduced-ORDER symbolic solve: keep only the reactances that shape H(s)
        over [fmin, fmax] (within tol_db), drop the rest, then collapse the
        coefficients with simplify. This lowers the pole count -- which Simplify
        alone never does -- to reach the textbook 2nd-order Miller form.

        The Result records the reactances kept and the reduced model's band error
        vs the FULL model in mag_err_db, and lists them in warnings. That error is
        the real cost of the lower order -- report it, do not hide it.
        """
        opts = dict(strategy_opts or {})
        key = self._key("reduce", inp, out, norm_keep(keep), tol_db,
                        max_elements, mag_db, phase_deg, fmin, fmax,
                        floor_db, floor_abs_db, eps, strategy,
                        tuple(sorted(opts.items())))
        if key not in self._cache:
            from .analysis.criteria import make_criterion

            an = self._analyzer_ready()
            # ONE criterion object carries the tolerance contract's math
            # (used by the pursuit, see analysis/criteria.py) AND its
            # language (the strip headline, the Summary details, the
            # score units) -- the two halves can no longer drift.
            crit = make_criterion(strategy=strategy, strategy_opts=opts,
                                  eps=eps, tol_db=tol_db)
            if max_elements is not None and crit.cap is not None:
                crit.cap = max_elements
            if crit.name:            # anchored or a strategy: the
                # collapse budgets derive from the SAME contract
                mag_db, phase_deg = crit.collapse_budgets()
            H, red = an.reduced_tf(inp, out, keep, tol_db=tol_db,
                                   fmin=fmin, fmax=fmax,
                                   max_elements=max_elements,
                                   floor_db=floor_db,
                                   floor_abs_db=floor_abs_db,
                                   phase_tol_deg=phase_deg,
                                   eps=eps, strategy=strategy,
                                   strategy_opts=opts,
                                   progress=progress, note=note)
            Hs = H.simplify(mag_tol_db=mag_db, phase_tol_deg=phase_deg,
                            fmin=fmin, fmax=fmax)
            # the coefficient snap: the leading numeral of the DC gain
            # (a collapsed ratio of unkept parameters on the path) goes
            # to exactly 1 when the snapped form still meets the
            # contract against the FULL response -- one more measured
            # approximation, never a cosmetic rounding
            from .analysis.simplify import snap_unity

            full = self.solve(inp, out, [])
            Hs, snap_k, snap_err = snap_unity(Hs, crit, full.freqs, full.h)
            r = self._assemble(Hs, inp, out, keep, reference=reference,
                               fmin=fmin, fmax=fmax, points=points)
            r.snap_factor = float(snap_k)
            band_err = float(red.errors_db[-1]) if red.errors_db else \
                float(red.baseline_db)
            r.simplified = True
            r.reduced_order = True
            # the normalized score travels beside mag_err_db with its
            # unit named; mag_err_db itself is ALWAYS a genuine dB figure
            score, s_unit, mag_err = crit.score_fields(
                band_err, float(Hs.achieved_mag_err_db))
            if score is not None:
                r.band_score, r.band_score_unit = score, s_unit
            r.mag_err_db = float(mag_err)
            r.phase_err_deg = float(Hs.achieved_phase_err_deg)
            r.n_terms_full = _n_terms(H)
            r.band_fmin, r.band_fmax = float(fmin), float(fmax)
            if crit.name:
                if strategy is not None:
                    r.strategy = strategy
                else:
                    r.eps = float(eps)
                    r.anchor = float(red.anchor)
                r.enforced_fmin, r.enforced_fmax = float(fmin), float(fmax)
                # the strip gets ONE actionable line; everything else --
                # element names, criterion, certificate verdict, doublet
                # caveat -- lands in r.details, which only the Summary
                # renders
                r.warnings.insert(
                    0, crit.headline(red, band_err, fmin, fmax))
                det = crit.details(red, band_err, fmin, fmax)
                det += self._certificate_details(
                    r, inp, out, fmin, fmax, crit.eps_equivalent(),
                    len(red.selected),
                    label=(f"{eps:.0%}" if crit.name == "anchored"
                           else ""))
                r.details = det
            else:
                if red.sig_hi:                 # the ENFORCED sub-band
                    r.enforced_fmin = float(red.sig_lo)
                    r.enforced_fmax = float(red.sig_hi)
                r.warnings.insert(
                    0, crit.headline(red, band_err, fmin, fmax,
                                     tol_db=tol_db))
            if snap_err is not None:
                r.details.append(
                    f"coefficient snap: the DC gain's leading numeral "
                    f"{snap_k:.4g} (a collapsed ratio of unkept parameters "
                    f"on the path) divided out -- a uniform {snap_err:.3g} "
                    f"dB shift, measured within the budget against the full "
                    f"response")
            self._cache[key] = r
        return self._cache[key]

    def rank_symbols(self, inp: str, out: str, *, metric: str = "complex",
                     fmin: float | None = None, fmax: float | None = None):
        """Band-sensitivity ranking for keep-set selection: a list of
        (name, score, peak_Hz) descending. All numeric; no symbolic solve."""
        bs = self._analyzer_ready().band_sensitivities(
            inp, out, metric=metric, fmin=fmin, fmax=fmax)
        return [(name, float(score), float(bs.peak_frequency(name)))
                for name, score in bs.rank(fmin, fmax)]

    def lumped_gmb(self) -> dict:
        """device -> "lumped" | "dropped (...)" under
        mos_model='lumped-gmb'; empty when separate. What the strip
        reports after the toggle."""
        return dict(self._analyzer_ready().lumped_gmb)

    def model_values(self, name: str) -> dict:
        """param -> the value the WORKING model uses for this device,
        after the match policy, the gmb lump and any reduction: what a
        symbol of it is worth in the next solve. Empty when the device
        has no live primitive (lumped away, reduced)."""
        out = {}
        try:
            prims = self._analyzer_ready().primitives
        except Exception:                        # noqa: BLE001
            return out
        for p in prims:
            if p.inst == name and p.value is not None                     and isinstance(p.value, (int, float)):
                out[p.param or p.kind] = float(p.value)   # a passive's
        return out                                        # value is its kind

    def device_op(self, name: str) -> dict:
        """The full OP record of one device (region, vds, vdsat, ... --
        the reporting metadata that never becomes a stamp), by CIN name."""
        for d in self._run.flat.devices:
            if d.name == name:
                return dict(self._run.op_data.get(d.sim_name) or {})
        return {}

    def impact_ionization_devices(self, tol: float = 0.005):
        """MOSFETs where the DC substrate (impact-ionization) current is a
        non-negligible fraction of Ids AND no gii conductance is modeled --
        the small-signal reconstruction is then INCOMPLETE for them (the
        r2r over-unity lesson: dcOpInfo's gm/gds/gmbs exclude the II
        derivatives the ac/stb linearization uses). Returns [(name, ratio)]
        for isub/ids >= tol, largest first -- a fast, pure-OP advisory; the
        exact gii still needs AC-injection identification."""
        out = []
        for d in self._run.flat.devices:
            if d.device_type != "mosfet":
                continue
            raw = self._run.op_data.get(d.sim_name) or {}
            isub = raw.get("isub", raw.get("iavl"))
            ids = raw.get("ids")
            try:
                ratio = abs(float(isub)) / abs(float(ids))
            except (TypeError, ValueError, ZeroDivisionError):
                continue
            if ratio < tol:
                continue
            params = d.params or {}
            has_gii = ("gii_d" in params or "gii_m" in params
                       or raw.get("gii_d") is not None)
            if not has_gii:
                out.append((d.name, ratio))
        out.sort(key=lambda t: -t[1])
        return out

    def op_values(self) -> dict[str, float]:
        """Operating-point value of every device symbol, name -> value (SI units:
        S for gm/gds, F for caps, ...). What each symbol IS numerically, for the
        ranking table. Device-level, so independent of input/output."""
        if getattr(self, "_op_values", None) is None:
            inp = self.suggested_input()
            src = self.sources()
            inp = inp or (src[0] if src else None)
            sysm = self._analyzer_ready().system(inp) if inp else None
            self._op_values = dict(sysm.values) if sysm is not None else {}
        return self._op_values
