"""High-level API: CIN circuit -> symbolic transfer functions.

    from circuitinsight import Analyzer

    an = Analyzer.from_cin("ota.cin.json")
    an.match("M1", "M2")                       # share symbols (matched pair)
    H = an.tf(inp="VIN", out="vout")           # fully symbolic
    H = an.tf(inp="VIN", out="vout", keep=["M1", "CL"])   # hybrid
    H.expr, H.dc_gain(), H.poles(), H.numeric([1e3, 1e6])
"""
from __future__ import annotations

from pathlib import Path

from dataclasses import dataclass

import sympy as sp

from .adapters.cin import CinDoc, FlatCircuit, flatten, load_cin, parse_cin
from .engine.mna import MnaError, MnaSystem, TransferFunction, build_mna, solve_tf
from .keep import ALL, is_all, norm_keep       # noqa: F401  (re-exported)
from .models import expand_circuit


@dataclass
class PortEquivalent:
    """Thevenin/Norton equivalent of the circuit seen at a port, per unit
    excitation: v_oc = vth, i_sc = isc, and zth = vth/isc."""
    vth: TransferFunction    # open-circuit (Thevenin) voltage
    zth: TransferFunction    # Thevenin/Norton impedance (sources zeroed)
    isc: TransferFunction    # short-circuit (Norton) current


class Analyzer:
    def __init__(self, flat: FlatCircuit, op_data: dict[str, dict] | None = None,
                 cap_model: str = "lumped", bjt_model: str = "intrinsic"):
        self.flat = flat
        self.primitives = expand_circuit(flat, op_data, cap_model, bjt_model)
        self._alias: dict[str, str] = {}

    # -------------------------------------------------- constructors
    @classmethod
    def from_cin(cls, source: str | Path | dict | CinDoc,
                 op_data: dict[str, dict] | None = None,
                 cap_model: str = "lumped",
                 bjt_model: str = "intrinsic") -> "Analyzer":
        if isinstance(source, CinDoc):
            doc = source
        elif isinstance(source, dict):
            doc = parse_cin(source)
        else:
            doc = load_cin(source)
        return cls(flatten(doc), op_data, cap_model=cap_model,
                   bjt_model=bjt_model)

    # -------------------------------------------------- symbol matching
    def match(self, *instances: str) -> None:
        """Declare instances matched: they share one set of symbols (the first
        instance's names). Enables textbook cancellations for matched pairs."""
        if len(instances) < 2:
            raise ValueError("match() needs at least two instance names")
        known = {p.inst for p in self.primitives}
        for inst in instances:
            if inst not in known:
                raise ValueError(f"match(): unknown instance {inst!r}")
        target = self._alias.get(instances[0], instances[0])
        for inst in instances[1:]:
            self._alias[inst] = target

    # -------------------------------------------------- analyses
    def system(self, inp: str) -> MnaSystem:
        return build_mna(self.primitives, self.flat.ground, inp, self._alias)

    def frequency_response(self, inp: str, out: str, freqs) -> "np.ndarray":
        """Numeric v(out)/excitation(inp) over freqs (Hz) by solving the MNA
        system directly at each frequency — no symbolic determinant.

        The fast path for large circuits (e.g. the extrinsic-BJT uA741,
        ~120 nodes) where the exact symbolic solve is impractical, and for
        validation. Returns a complex ndarray; use tf() when you want the
        symbolic transfer function itself."""
        import numpy as np

        from .engine.mna import S

        sysm = self.system(inp)
        subs = {sysm.symbols[n]: sp.Float(v) for n, v in sysm.values.items()
                if n in sysm.symbols}
        free = set(sysm.symbols.values()) - {S} - set(subs)
        if free:
            raise MnaError(
                f"frequency_response needs all values numeric; missing "
                f"{sorted(str(s) for s in free)}")
        if out not in sysm.node_index:
            raise MnaError(f"output node {out!r} not found (or it is ground)")
        k = sysm.node_index[out]
        A_of_s = sp.lambdify(S, sysm.A.xreplace(subs), "numpy")
        z = np.array(sysm.z.T.tolist()[0], dtype=complex)
        w = 2j * np.pi * np.atleast_1d(np.asarray(freqs, dtype=float))
        out_v = np.empty(w.shape, dtype=complex)
        for i, wi in enumerate(w):
            out_v[i] = np.linalg.solve(
                np.asarray(A_of_s(wi), dtype=complex), z)[k]
        return out_v

    def tf(self, inp: str, out: str, keep=ALL,
           method: str = "auto", progress=None) -> TransferFunction:
        """Transfer function v(out)/excitation(inp).

        inp: name of a vsource (voltage gain) or isource (transimpedance).
        out: net name.

        keep:
          ALL (the default) -- fully symbolic: every device symbol is
              retained. This is the point of the tool, and also the only mode
              whose cost is unbounded: on a big circuit it can run for hours.
              `estimate_solve_time()` will tell you before you commit.
          [] (or None, its alias) -- fully numeric: no symbols retained
              (cheap).
          [names] -- hybrid: only these instances/symbols stay symbolic, the
              rest become exact numerics.

        Note [] and ALL are opposites; do not test `keep` for truthiness.
        Fully symbolic must be spelled ALL -- None means numeric.

        method: 'auto' (multilinear interpolation for hybrid solves),
        'interp', or 'direct'.
        """
        keep = ALL if is_all(keep) else list(() if keep is None else keep)
        return solve_tf(self.system(inp), out, keep, method=method,
                        progress=progress)

    def loop_gain(self, probe: str, keep=(), progress=None) -> TransferFunction:
        """Loop gain T(s) at a designated probe (Tian double-injection on the
        MNA system; see analysis/loopgain.py). `probe` is a 0-V vsource
        branch inside the loop -- the CIN image of a schematic iprobe.
        Sign convention matches Spectre stb: arg T = +180 deg at DC, phase
        margin is arg T at |T| = 1.

        keep follows tf()'s convention -- [] (the default; None is its
        alias) numeric, [names] hybrid-symbolic, ALL fully symbolic -- so
        pole-splitting keep sets give symbolic phase-margin structure
        directly on T."""
        from .analysis.loopgain import loop_gain as _loop_gain

        keep = ALL if is_all(keep) else list(() if keep is None else keep)
        return _loop_gain(self.system(probe), probe, keep, progress=progress)

    def gft(self, probe: str, inp: str, out: str, error_ref: str, keep=()):
        """Middlebrook's GFT quartet {H, Hinf, T, Tn, H0} at the designated
        probe, with the error signal u_y = v(error_ref) - v(probe p side).
        Exact rationals; identity self-checked. See analysis/gft.py.

        keep follows the tf()/loop_gain() convention: [] (default) numeric,
        [names] keeps those symbols symbolic across the whole quartet, ALL
        fully symbolic."""
        from .analysis.gft import gft as _gft

        return _gft(self.system(inp), self.system(probe), probe, out,
                    error_ref, keep=keep)

    def nested_gft(self, probe1: str, probe2: str, inp: str, out: str,
                   error1, error2, **kw):
        """Nested (MIMO) GFT: dissect H against loop 1, then Hinf1
        against loop 2 on the loop-1-nulled system. Exact at both
        levels for any designation; see analysis/nested_gft.py."""
        from .analysis.nested_gft import nested_gft as _nested

        return _nested(self.system(inp), self.system(probe1),
                       self.system(probe2), probe1, probe2, out,
                       error1, error2, **kw)

    def nested_gft_deep(self, probes, inp: str, out: str, errors, **kw):
        """N-level nested (MIMO) GFT: dissect H against loop 1, then the
        running ideal gain against loops 2..N in turn, each on the
        higher-nulled system. `probes` and `errors` are equal-length lists
        (errors[k] = (ref_node, c)); exact at every level for any
        designation. See analysis/nested_gft.py (DeepNestedGft)."""
        from .analysis.nested_gft import nested_gft_deep as _deep

        systems_p = [self.system(pr) for pr in probes]
        return _deep(self.system(inp), systems_p, probes, out, errors, **kw)

    def mode_loop(self, probe_a: str, probe_b: str, **kw):
        """2x2 mode loop matrix at a probe pair (e.g. the DM and CM
        iprobes of an fd_probe): eigenloci, per-locus margins, cross-
        mode coupling, and the per-branch scalar loop gains, with a
        Schur self-certificate. `scale={"gm_...": factor}` applies a
        deliberate mismatch. See analysis/modes.py."""
        from .analysis.modes import mode_loop as _mode_loop

        return _mode_loop(self.system(None), probe_a, probe_b, **kw)

    def assess_probe(self, probe: str, gft=None, **kw):
        """Grade a designated stb probe against the exact closed loop:
        margins-vs-pole-damping consistency plus a per-device visibility
        scan naming loops the probe cannot see. Optional
        gft={"inp": src, "out": node, "error": (ref, c)} adds the
        design-meaningfulness check (Hinf flatness, feedthrough
        crossover, exact identity). See analysis/probeadequacy.py."""
        from .analysis.probeadequacy import assess_probe as _assess

        if gft is not None:
            gft = {**gft, "z_in": self.system(gft.pop("inp")).z}
        return _assess(self.system(probe), probe, gft=gft, **kw)

    def return_ratio(self, source: str, keep=()) -> TransferFunction:
        """Bode return ratio of a controlled source (e.g. ``gm_I0.MP2``) --
        the textbook loop gain 'of the gain device'. Positive real at DC for
        negative feedback; the stb-convention loop gain is ~ -RR wherever
        the loop signal flows through the source. See analysis/loopgain.py."""
        from .analysis.loopgain import return_ratio as _return_ratio

        keep = ALL if is_all(keep) else list(() if keep is None else keep)
        return _return_ratio(self.system(None), source, keep)

    def reduced_tf(self, inp: str, out: str, keep=ALL, *, tol_db: float = 0.5,
                   fmin: float | None = None, fmax: float | None = None,
                   metric: str = "complex", exclude: tuple[str, ...] = (),
                   max_elements: int | None = None):
        """Transfer function of the REDUCED-ORDER model: keep only the reactances
        that actually shape H(s) over the band, zero the rest, then solve keeping
        `keep` symbolic. This is what produces the textbook 2nd-order Miller form
        -- Simplify prunes coefficient terms but never lowers the pole count; this
        does, by dropping the parasitic caps the response doesn't need.

        Returns (TransferFunction, ReactanceReduction). The reduction's
        errors_db[-1] is the *honest* band error of the reduced model -- report
        it; a lower-order model is an approximation, not a free lunch.
        """
        from dataclasses import replace

        from .analysis.sensitivity import _reactive_symbols

        red = self.dominant_reactances(inp, out, tol_db, fmin, fmax, metric,
                                       exclude=exclude, max_elements=max_elements)
        keep = ALL if is_all(keep) else list(() if keep is None else keep)
        keep_set = set(() if is_all(keep) else keep)
        reactive = _reactive_symbols(self)
        sysm = self.system(inp)
        # drop every reactance that is neither dominant (selected) nor explicitly
        # kept symbolic; zero its value so the substitution folds it out.
        dropped = {n for n in reactive if n in sysm.values} \
            - set(red.selected) - keep_set
        vals = dict(sysm.values)
        for n in dropped:
            vals[n] = 0.0
        H = solve_tf(replace(sysm, values=vals), out, keep)
        return H, red

    def _port_nets(self, port: str) -> tuple[str, str | None, bool]:
        """Resolve a port-marker instance to (node, ref, series).

        Two marker types, both electrically invisible in the schematic:
        a 0 A isource in parallel is a *parallel* port (its terminals are
        the port; natively open, so it measures open-circuit quantities);
        a 0 V vsource in series with a branch is a *series* port (the
        marker is removed to open the branch and the port is the gap;
        natively a short, so the intact circuit carries its short-circuit
        current). series=True means callers must disable the marker."""
        gnd = set(self.flat.ground)
        for d in self.flat.devices:
            if d.name == port:
                if d.device_type not in ("isource", "vsource"):
                    raise ValueError(
                        f"port {port!r}: must be an isource (parallel port) "
                        f"or vsource (series port), not {d.device_type!r}")
                t = d.terminals
                p, n = t["p"], t["n"]
                if p in gnd:              # grounded terminal is the reference
                    p, n = n, p
                return p, (None if n in gnd else n), \
                    d.device_type == "vsource"
        raise ValueError(f"unknown port instance {port!r}")

    def impedance(self, node: str | None = None, ref: str | None = None,
                  keep=ALL, method: str = "auto",
                  disable: tuple[str, ...] = (),
                  port: str | None = None) -> TransferFunction:
        """Driving-point impedance Z(s) looking into `node` (referenced to
        `ref`, default ground): a unit test current is injected into the
        port and Z = v(port). All independent sources are zeroed (vsources
        become shorts, isources opens), so a vsource sitting directly across
        the port shorts it — pass its instance name in `disable` to remove
        it first (e.g. the driving source, for input impedance).

        keep/method: as in tf(). A differential port (ref not ground) costs
        two numerator solves. Instead of node/ref, `port` names a marker
        instance (see _port_nets): a parallel port (isource) gives the
        shunt driving-point impedance; a series port (vsource) is opened
        automatically and gives the loop impedance through its branch.
        """
        if (node is None) == (port is None):
            raise ValueError("impedance(): give exactly one of node or port")
        if port is not None:
            node, ref, series = self._port_nets(port)
            if series and port not in disable:
                disable = (*disable, port)
        known = {p.inst for p in self.primitives}
        missing = [i for i in disable if i not in known]
        if missing:
            raise ValueError(f"impedance(): unknown instance(s) {missing}")
        off = set(disable)
        prims = [p for p in self.primitives if p.inst not in off]
        system = build_mna(prims, self.flat.ground, None, self._alias)

        gnd = set(self.flat.ground)

        def port_index(n: str | None) -> int | None:
            if n is None or n in gnd:
                return None
            if n not in system.node_index:
                raise MnaError(f"impedance(): node {n!r} not found")
            return system.node_index[n]

        ip = port_index(node)
        if ip is None:
            raise MnaError("impedance(): port node is ground")
        system.z[ip, 0] += 1
        iref = port_index(ref)
        if iref is not None:
            system.z[iref, 0] -= 1

        zp = solve_tf(system, node, keep, method=method)
        if iref is None:
            return zp
        zn = solve_tf(system, ref, keep, method=method)
        return TransferFunction(expr=sp.cancel(zp.expr - zn.expr),
                                values=zp.values, symbols=zp.symbols)

    def equivalent(self, inp: str, node: str | None = None,
                   ref: str | None = None, keep=ALL,
                   method: str = "auto", disable: tuple[str, ...] = (),
                   port: str | None = None) -> PortEquivalent:
        """Thevenin/Norton equivalent seen at a port, per unit excitation
        of `inp`: vth is the open-circuit voltage (a plain transfer
        function), zth the impedance with sources zeroed, and
        isc = vth/zth the short-circuit (Norton) current — vth and zth
        share the network determinant, so isc cancels to a compact
        rational.

        Ports can be marked in the schematic with invisible elements
        (both accepted as `port`): a 0 A isource in parallel is a
        parallel port — natively open, the Thevenin probe; a 0 V vsource
        in series is a series port — the branch is opened for vth/zth,
        and isc is the current that branch carries in the intact circuit
        (the Norton probe). `disable` separately excludes instances from
        the equivalent (choosing the boundary, not the measurement).
        """
        if (node is None) == (port is None):
            raise ValueError("equivalent(): give exactly one of node or port")
        if port is not None:
            if port == inp:
                raise ValueError("equivalent(): port cannot be the input")
            node, ref, series = self._port_nets(port)
            if series and port not in disable:
                disable = (*disable, port)

        known = {p.inst for p in self.primitives}
        missing = [i for i in disable if i not in known]
        if missing:
            raise ValueError(f"equivalent(): unknown instance(s) {missing}")
        off = set(disable)
        prims = [p for p in self.primitives if p.inst not in off]
        sys_v = build_mna(prims, self.flat.ground, inp, self._alias)
        vth = solve_tf(sys_v, node, keep, method=method)
        if ref is not None and ref not in self.flat.ground:
            vn = solve_tf(sys_v, ref, keep, method=method)
            vth = TransferFunction(expr=sp.cancel(vth.expr - vn.expr),
                                   values=vth.values, symbols=vth.symbols)
        zth = self.impedance(node, ref, keep=keep, method=method,
                             disable=disable)
        if zth.expr == 0:
            raise MnaError("equivalent(): port is shorted (zth = 0); "
                           "disable the shorting element?")
        isc = TransferFunction(expr=sp.cancel(vth.expr / zth.expr),
                               values=vth.values, symbols=vth.symbols)
        return PortEquivalent(vth=vth, zth=zth, isc=isc)

    def sensitivities(self, inp: str, out: str, n_poles: int = 3):
        """OP-point sensitivity ranking of every parameter w.r.t. dc gain and
        the first poles; .suggest_keep() proposes symbolic keep-sets.

        For keep-set selection prefer band_sensitivities(): it ranks over
        the whole frequency band (not just s=0) and is far cheaper (no
        symbolic characteristic polynomial)."""
        from .analysis.sensitivity import sensitivities

        return sensitivities(self, inp, out, n_poles)

    def band_sensitivities(self, inp: str, out: str, metric: str = "complex",
                           fmin: float | None = None,
                           fmax: float | None = None):
        """Band-sampled parameter ranking for keep-set selection: how much
        each element moves the transfer function across the frequency band,
        via Jacobi's formula at feature-aligned (pole/zero) sample points --
        all numeric, no symbolic determinant. metric='complex' (default)
        ranks by the whole complex TF (magnitude and phase);
        'magnitude' ranks by |H|-in-dB alone. Returns a BandSensitivity
        with .suggest_keep(top) and .report()."""
        from .analysis.sensitivity import band_sensitivity

        return band_sensitivity(self, inp, out, metric, fmin, fmax)

    def dominant_reactances(self, inp: str, out: str, tol_db: float = 1.0,
                            fmin: float | None = None,
                            fmax: float | None = None, metric: str = "complex",
                            exclude: tuple[str, ...] = (),
                            max_elements: int | None = None):
        """The minimal set of capacitors/inductors that reproduces the
        transfer function over the band, by frequency-domain matching
        pursuit: remove every reactance, then add back the one whose
        first-order sensitivity best cancels the current residual (verified
        by an exact solve), until the band error is within tol_db. Answers
        'which caps actually shape this response?'. `exclude` names
        reactances kept always-on (e.g. a measurement rig). Returns a
        ReactanceReduction (.selected, .errors_db, .report())."""
        from .analysis.sensitivity import dominant_reactances

        return dominant_reactances(self, inp, out, tol_db, fmin, fmax, metric,
                                   exclude=exclude, max_elements=max_elements)

    def estimate_solve_time(self, inp: str, out: str, keep=ALL):
        """Estimate the interpolation solver's wall-clock for tf(inp, out,
        keep) from the grid size (∏ stamp_count+1 over kept symbols), the
        s-degree, and a cheap numeric-determinant timing probe. Returns a
        SolveEstimate. See analysis/estimate.py.

        keep=ALL (fully symbolic) is costed over EVERY symbol -- that is the only
        mode whose cost is unbounded, so it is the one worth estimating. It used
        to be silently costed as `[]` (the cheapest case), which meant the
        estimator reported milliseconds for a solve that would run for hours.
        """
        from .analysis.estimate import estimate_solve

        return estimate_solve(self, inp, out, norm_keep(keep))

    def plan_keep(self, inp: str, out: str, budget_s: float,
                  metric: str = "complex", fmin: float | None = None,
                  fmax: float | None = None, max_keep: int = 16):
        """Pick the largest band-ranked keep-set whose estimated symbolic
        solve fits `budget_s` seconds: rank symbols by band_sensitivities(),
        add them most-impactful-first while the time estimate stays within
        budget. Returns a KeepPlan (.keep, .dropped, .estimate)."""
        from .analysis.estimate import plan_keep

        return plan_keep(self, inp, out, budget_s, metric, fmin, fmax, max_keep)

    def bandwidth_report(self):
        """Per-capacitor/inductor bandwidth attribution via zero-value time
        constants (exact, from the network denominator; no output node or
        symbolic solve needed). See analysis/bandwidth.py."""
        from .analysis.bandwidth import bandwidth_contributions

        return bandwidth_contributions(self)
