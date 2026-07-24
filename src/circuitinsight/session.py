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
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import sympy as sp

from .keep import ALL, is_all, norm_keep  # noqa: F401  (ALL re-exported)

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


def _n_terms(tf) -> int:
    num, den = tf.num_den
    return sum(len(sp.Add.make_args(sp.expand(c)))
               for poly in (num, den) for _, c in poly.terms())


def _loop_margins(freqs, h):
    """(pm_deg, pm_freq, gm_db, gm_freq) from a loop-gain sweep in the stb
    convention (phase +180 deg at DC). Entries are None when the respective
    crossing is not in the band."""
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


def _numeric_dc(tf) -> complex:
    """True dc gain: the TF at s=0 with every kept symbol at its OP value
    (frequency-independent, so unaffected by where the dominant pole sits)."""
    e = tf.dc_gain()
    subs = {tf.symbols[n]: sp.Float(v) for n, v in tf.values.items()
            if n in tf.symbols}
    if subs:
        e = e.xreplace(subs)
    return complex(e)


class SessionController:
    """Stateful, headless controller over one analysis session."""

    def __init__(self):
        self._run = None                 # simulator adapter run (opaque)
        self._analyzer = None            # built lazily, rebuilt when matches change
        self._matches: list[tuple[str, ...]] = []
        self._cache: dict[tuple, Result] = {}
        self.cin_path: Path | None = None
        self.op_path: Path | None = None
        self.simulator: str | None = None
        self.cap_model: str = "lumped"   # see open(cap_model=...)
        #: instance/symbol -> LaTeX alias for expression rendering (GUI)
        self.sym_aliases: dict[str, str] = {}

    # ------------------------------------------------------------------ open
    @classmethod
    def open(cls, cin_path, op_path, *, simulator: str = "spectre",
             cap_model: str = "lumped", **adapter_kw) -> "SessionController":
        """Open a session from a CIN topology + a simulator's OP results.

        `simulator` selects the adapter; only "spectre" exists today, but the
        entry point is neutral so ngspice/LTspice slot in the same way.

        `cap_model`: "lumped" (five-capacitor, default) or "matrix" (exact
        charge matrix). On strongly non-reciprocal processes (SKY130) the
        matrix model is the accurate one -- loop-gain margins in particular
        shift by ~0.1 deg / 0.6% between the two on the two-stage bench.
        """
        self = cls()
        self.simulator = simulator
        self.cap_model = cap_model
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

        def gm_of(name):
            return opv.get("gm_" + name.replace(".", "_"))

        refined: list[tuple[str, ...]] = []
        for k in order:
            members = groups[k]
            if len(members) < 2:
                continue
            members = sorted(members, key=lambda n: (gm_of(n) is None,
                                                     gm_of(n) or 0.0))
            sub: list[str] = [members[0]]
            for prev, cur in zip(members, members[1:]):
                gp, gc = gm_of(prev), gm_of(cur)
                close = (gp is None or gc is None
                         or (gp > 0 and abs(gc / gp - 1.0) <= 0.02))
                if close:
                    sub.append(cur)
                else:
                    if len(sub) >= 2:
                        refined.append(tuple(sub))
                    sub = [cur]
            if len(sub) >= 2:
                refined.append(tuple(sub))
        return refined

    def _analyzer_ready(self):
        if self._analyzer is None:
            an = self._run.analyzer(cap_model=self.cap_model)
            for group in self._matches:
                an.match(*group)
            self._analyzer = an
        return self._analyzer

    # --------------------------------------------------------------- planning
    def estimate(self, inp: str, out: str, keep=ALL):
        """SolveEstimate for tf(inp, out, keep) — gate a solve on predicted cost."""
        return self._analyzer_ready().estimate_solve_time(inp, out, keep)

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
        dc_db = 20 * math.log10(abs(dc)) if dc != 0 else float("-inf")
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
            keep=(ALL if is_all(keep)
                  else list(() if keep is None else keep)),
            dc_gain=dc, dc_gain_db=dc_db,
            poles_hz=poles, zeros_hz=zeros, n_terms=_n_terms(tf),
            tf_latex=sp.latex(tf.expr),
            dc_gain_latex=sp.latex(tf.dc_gain()),
            freqs=freqs, h=h, h_ref=h_ref, ref_label=ref_label,
            warnings=warns, tf=tf,
        )

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
        key = ("solve", inp, out, norm_keep(keep), tuple(self._matches))
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

    def impedance_result(self, port: str, keep=()) -> Result:
        """Driving-point impedance at a port marker, packaged as a Result.
        When the run carries an xf result with this port's transfer, it is
        the overlay -- the simulator truth for Z; absent, the model stands
        alone (never a show-stopper). Cached."""
        key = ("impedance", port, norm_keep(keep), tuple(self._matches))
        if key in self._cache:
            return self._cache[key]
        an = self._analyzer_ready()
        tf = an.impedance(port=port, keep=keep)
        r = self._assemble(tf, port, f"Z({port})", keep, reference=False,
                           fmin=1e3, fmax=1e9, points=400)
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
            pass                          # no xf truth in this run -- fine
        self._cache[key] = r
        return r

    def stb_probe(self) -> str | None:
        """The run's DESIGNATED stb probe (CIN name), when discoverable
        from the psfascii header or the run's netlist. None otherwise."""
        try:
            return self._run.stb_probe()
        except Exception:
            return None

    def _stb_reference(self, points_freqs):
        """(freqs, loopGain, margins, label) from the run's stb results,
        or (None,)*4 when the run has none."""
        try:
            stb = self._run.stb()
            return (np.asarray(stb.freq, dtype=float),
                    np.asarray(stb.loop_gain),
                    stb, "Spectre stb loopGain")
        except Exception:
            return None, None, None, None

    def loop_gain(self, probe: str, keep=(), *, reference: bool = True,
                  fmin: float = 1.0, fmax: float = 1e10,
                  points: int = 600) -> Result:
        """Tian loop gain at `probe` packaged as a Result: Bode of T with
        the run's Spectre stb overlay when available, phase/gain margins in
        the stb convention, and stability advisories. Cached.

        The default band starts at 1 Hz so the +180-deg DC phase reference
        unwraps correctly even for sub-kHz dominant poles."""
        key = ("loopgain", probe, norm_keep(keep), tuple(self._matches))
        if key in self._cache:
            return self._cache[key]

        T = self._analyzer_ready().loop_gain(probe, keep=keep)
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
        pm, fpm, gm, fgm = _loop_margins(fd, np.asarray(T.numeric(fd)))
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
        dc_db = 20 * math.log10(abs(dc)) if dc != 0 else float("-inf")
        self._cache[key] = Result(
            inp=probe, out=f"T@{probe}",
            keep=(ALL if is_all(keep)
                  else list(() if keep is None else keep)),
            dc_gain=dc, dc_gain_db=dc_db,
            poles_hz=T.poles(), zeros_hz=T.zeros(), n_terms=_n_terms(T),
            tf_latex=sp.latex(T.expr), dc_gain_latex=sp.latex(T.dc_gain()),
            freqs=freqs, h=h, h_ref=h_ref, ref_label=ref_label,
            warnings=warns, tf=T,
            pm_deg=pm, pm_freq_hz=fpm, gm_db=gm, gm_freq_hz=fgm,
        )
        return self._cache[key]

    def assess_probe(self, probe: str, **kw):
        """Grade a designated stb probe (docs/loopgain-plan.md Sec. 9):
        margins-vs-closed-loop-pole consistency plus a per-device
        visibility scan naming loop dynamics the probe cannot see (e.g.
        the CMFB loop seen from a DM probe). Returns a ProbeReport whose
        .verdict() is the one-line summary. Cached."""
        key = ("adequacy", probe, tuple(self._matches))
        if not kw and key in self._cache:
            return self._cache[key]
        report = self._analyzer_ready().assess_probe(probe, **kw)
        if not kw:                       # kw (grids, eps) may be unhashable
            self._cache[key] = report
        return report

    def suggest_compensation(self, probe: str, *, goal: str = "mfm",
                             pm_target: float = 60.0, exclude=(),
                             candidates=None, top: int = 5, **kw):
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
            key = ("suggest", probe, goal, pm_target, tuple(sorted(exclude)),
                   top, tuple(self._matches))
            if key in self._cache:
                return self._cache[key]
        out = _suggest(system, probe, goal=goal, pm_target=pm_target,
                       candidates=candidates, top=top, **kw)
        if key is not None:
            self._cache[key] = out
        return out

    def suggest_multi_compensation(self, probe: str, *, goal: str = "pm",
                                   k_max: int = 2, exclude=(),
                                   candidates=None, **kw):
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
        return _multi(system, probe, goal=goal, k_max=k_max,
                      candidates=candidates, **kw)

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
                 points: int = 400) -> Result:
        """Error-budgeted simplification of tf(inp, out, keep): prune negligible
        terms within `mag_db`/`phase_deg`. The Result carries the achieved error
        and the term count before/after pruning."""
        key = ("simplify", inp, out, norm_keep(keep), mag_db, phase_deg,
               tuple(self._matches))
        if key not in self._cache:
            H = self._analyzer_ready().tf(inp, out, keep=keep)
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

    def reduce_solve(self, inp: str, out: str, keep=ALL, *,
                     tol_db: float = 0.5, max_elements: int | None = None,
                     mag_db: float = 1.0, phase_deg: float = 5.0,
                     reference: bool = True, fmin: float = 1e3, fmax: float = 1e7,
                     points: int = 400) -> Result:
        """Reduced-ORDER symbolic solve: keep only the reactances that shape H(s)
        over [fmin, fmax] (within tol_db), drop the rest, then collapse the
        coefficients with simplify. This lowers the pole count -- which Simplify
        alone never does -- to reach the textbook 2nd-order Miller form.

        The Result records the reactances kept and the reduced model's band error
        vs the FULL model in mag_err_db, and lists them in warnings. That error is
        the real cost of the lower order -- report it, do not hide it.
        """
        key = ("reduce", inp, out, norm_keep(keep), tol_db, max_elements,
               mag_db, phase_deg, fmin, fmax, tuple(self._matches))
        if key not in self._cache:
            an = self._analyzer_ready()
            H, red = an.reduced_tf(inp, out, keep, tol_db=tol_db,
                                   fmin=fmin, fmax=fmax, max_elements=max_elements)
            Hs = H.simplify(mag_tol_db=mag_db, phase_tol_deg=phase_deg,
                            fmin=fmin, fmax=fmax)
            r = self._assemble(Hs, inp, out, keep, reference=reference,
                               fmin=fmin, fmax=fmax, points=points)
            band_err = float(red.errors_db[-1]) if red.errors_db else 0.0
            r.simplified = True
            r.mag_err_db = band_err                 # reduced model vs full, in-band
            r.phase_err_deg = float(Hs.achieved_phase_err_deg)
            r.n_terms_full = _n_terms(H)
            r.band_fmin, r.band_fmax = float(fmin), float(fmax)
            r.warnings.insert(
                0, f"reduced to {len(red.selected)} reactance(s) "
                   f"[{', '.join(red.selected)}] -- {band_err:.3f} dB vs the full "
                   f"model over {fmin:g}-{fmax:g} Hz")
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
