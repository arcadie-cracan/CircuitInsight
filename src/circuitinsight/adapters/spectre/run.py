"""Join Spectre OP data onto a CIN circuit; end-to-end SpectreRun facade.

Join policy (PLAN.md): strict. Every OP-requiring CIN device must find its
record and every record must be consumed — orphans on either side are hard
errors, never silent skips. Wrapper-style PDKs (device expands to `M1.m0` +
parasitics) are handled by prefix matching with model-class disambiguation.
"""
from __future__ import annotations

import warnings
from pathlib import Path

from ...analyzer import Analyzer
from ..cin import FlatCircuit, flatten, load_cin
from .opdata import OpRecord, load_dcopinfo

# device types that must find OP data
_NEEDS_OP = {"mosfet", "bjt", "diode", "resistor", "capacitor", "inductor"}
# device types whose records exist but carry nothing we need
_SOURCE_TYPES = {"vsource", "isource", "vcvs", "vccs", "ccvs", "cccs"}

# saturation region code per docs/spectre-op-mapping.md
_REGION_SAT = 2
_REGION_NAMES = {0: "off", 1: "triode", 2: "saturation", 3: "subthreshold", 4: "breakdown"}


class SpectreError(ValueError):
    pass


def _match_op(flat, records, rename):
    """The matching loop, shared by join_op and join_op_reduced.

    Returns (op_data, used, missing, errors): op_data by sim_name, the record
    names consumed, the FlatDevices that NEED an OP record but got none, and any
    hard errors (type mismatch, ambiguous wrapper). It does not decide what a
    missing device means -- that is the caller's policy (fail, or reduce)."""
    rename = rename or {}
    op_data: dict[str, dict] = {}
    used: set[str] = set()
    missing: list = []
    errors: list[str] = []

    for dev in flat.devices:
        target = rename.get(dev.sim_name, dev.sim_name)
        rec = records.get(target)
        if rec is None:
            # wrapper expansion: records below this instance's hierarchy
            candidates = [r for n, r in records.items() if n.startswith(target + ".")]
            primary = [r for r in candidates if r.device_type == dev.device_type]
            if len(primary) == 1:
                rec = primary[0]
                # secondary expansion records (parasitic diodes/resistors) are
                # consumed with the wrapper: lumped-wrapper default
                for r in candidates:
                    used.add(r.name)
            elif candidates:
                if dev.device_type in _NEEDS_OP:
                    errors.append(
                        f"{dev.sim_name}: ambiguous wrapper expansion "
                        f"{sorted(r.name for r in candidates)}"
                    )
                    continue
                # ideal testbench macro (e.g. balun -> transformer subckt):
                # consume the internals, nothing to extract
                for r in candidates:
                    used.add(r.name)

        if rec is None:
            if dev.device_type in _NEEDS_OP:
                missing.append(dev)
            continue

        used.add(rec.name)
        # a switch may legitimately be reported as a resistor at its DC state
        compatible = {dev.device_type}
        if dev.device_type == "switch":
            compatible.add("resistor")
        if rec.device_type is not None and rec.device_type not in compatible:
            errors.append(
                f"{dev.sim_name}: CIN says {dev.device_type}, OP record "
                f"{rec.name!r} is model {rec.model_type!r} ({rec.device_type})"
            )
            continue

        if dev.device_type == "mosfet":
            region = rec.params.get("region")
            if region is not None and int(region) != _REGION_SAT:
                warnings.warn(
                    f"{dev.sim_name}: operating region is "
                    f"{_REGION_NAMES.get(int(region), region)} — small-signal "
                    f"params are still valid at this bias but check the design"
                )
        op_data[dev.sim_name] = rec.params

    return op_data, used, missing, errors


_PASSIVE_VALUE_KEYS = {"resistor": ("r", "res"), "inductor": ("l",),
                       "capacitor": ("c", "cap")}


def _passive_value(dev, params: dict | None) -> float | None:
    """The R/C/L value of a passive from its OP record (or CIN params), or None
    if not stated. Used to spot 0-valued passives the simulator kept."""
    for src in (params or {}, dev.params):
        for k in _PASSIVE_VALUE_KEYS.get(dev.device_type, ()):
            if k in src and src[k] is not None:
                from ...values import parse_value
                v = src[k]
                return float(v) if isinstance(v, (int, float)) else parse_value(v)
    return None


def _no_record_error(dev, rename) -> str:
    t = (rename or {}).get(dev.sim_name, dev.sim_name)
    return f"{dev.sim_name}: no OP record found (searched {t!r} and {t!r}.*)"


def join_op(
    flat: FlatCircuit,
    records: dict[str, OpRecord],
    rename: dict[str, str] | None = None,
    allow_unmatched: tuple[str, ...] = (),
) -> dict[str, dict]:
    """Match OP records to flat devices by sim_name. Returns op_data keyed by
    the device sim_name, ready for Analyzer/expand_circuit. Any device that needs
    an OP record and has none is an error (use join_op_reduced to instead reduce
    simulator-pruned passives).

    rename: CIN sim_name -> results name, for known naming differences.
    allow_unmatched: record names tolerated as orphans (e.g. testbench-only
    instrumentation).
    """
    op_data, used, missing, errors = _match_op(flat, records, rename)
    errors += [_no_record_error(d, rename) for d in missing]
    errors += [f"OP record {n!r} matched no CIN device"
               for n in sorted(set(records) - used - set(allow_unmatched))]
    if errors:
        raise SpectreError("OP join failed:\n  " + "\n  ".join(errors))
    return op_data


def join_op_reduced(
    flat: FlatCircuit,
    records: dict[str, OpRecord],
    rename: dict[str, str] | None = None,
    allow_unmatched: tuple[str, ...] = (),
) -> tuple[dict[str, dict], FlatCircuit, list[str]]:
    """Like join_op, but reduce simulator-pruned passives instead of failing on
    them. Returns (op_data, reduced_flat, notes).

    A resistor/inductor/capacitor with no OP record is a 0-valued component the
    simulator optimized away; reduce.py folds it out (short for R/L, open for C)
    and produces a note. A missing ACTIVE device (mosfet/bjt/diode) is a real
    error and still raises -- we never silently drop a transistor.
    """
    from ...reduce import REDUCIBLE_IF_MISSING, drop_unmatched_passives

    op_data, used, missing, errors = _match_op(flat, records, rename)
    reducible = [d for d in missing if d.device_type in REDUCIBLE_IF_MISSING]
    errors += [_no_record_error(d, rename)
               for d in missing if d.device_type not in REDUCIBLE_IF_MISSING]
    errors += [f"OP record {n!r} matched no CIN device"
               for n in sorted(set(records) - used - set(allow_unmatched))]
    if errors:
        raise SpectreError("OP join failed:\n  " + "\n  ".join(errors))

    # missing passives were 0-valued and pruned by the simulator; fold them out.
    flat, notes = drop_unmatched_passives(flat, reducible)
    # some simulators KEEP a 0 F / 0 ohm passive as an OP record (SKY130 does)
    # instead of pruning it; fold those the same way (short R/L, open C).
    zero = [d for d in flat.devices
            if d.device_type in REDUCIBLE_IF_MISSING
            and _passive_value(d, op_data.get(d.sim_name)) == 0.0]
    if zero:
        flat, znotes = drop_unmatched_passives(
            flat, zero, cause="0-valued in the OP")
        notes += znotes
    return op_data, flat, notes


class SpectreRun:
    """One simulated design: CIN topology + psfascii results directory."""

    def __init__(
        self,
        cin: str | Path,
        results: str | Path,
        rename: dict[str, str] | None = None,
        allow_unmatched: tuple[str, ...] = (),
        backend: str = "auto",
    ):
        self.flat = flatten(load_cin(cin))
        self.results = Path(results)
        self.backend = backend
        self.records = load_dcopinfo(results, backend)
        # reduce simulator-pruned passives (0-valued R/C/L absent from the OP
        # data) rather than crash on them; self.reductions records what changed.
        self.op_data, self.flat, self.reductions = join_op_reduced(
            self.flat, self.records, rename, allow_unmatched)
        for note in self.reductions:
            warnings.warn(f"netlist reduction: {note}")

    def analyzer(self, cap_model: str = "lumped",
                 bjt_model: str = "intrinsic") -> Analyzer:
        return Analyzer(self.flat, self.op_data, cap_model=cap_model,
                        bjt_model=bjt_model)

    @property
    def _base(self) -> Path:
        return self.results if self.results.is_dir() else self.results.parent

    def excited_sources(self) -> list[str]:
        """CIN sources with a nonzero AC magnitude (`acm` param), if the
        netlist recorded them. Empty for CINs without acm info."""
        from ...values import parse_value

        out = []
        for d in self.flat.devices:
            if d.device_type in _SOURCE_TYPES and d.params.get("acm"):
                try:
                    mag = parse_value(d.params["acm"])
                except (ValueError, TypeError):
                    continue
                if mag != 0:
                    out.append(d.name)
        return out

    def has_xf(self, name: str = "xf.xf") -> bool:
        return (self._base / name).exists()

    def ac(self, name: str = "ac.ac"):
        """AC results from the same run (for validation).

        Policy: with multiple AC-excited sources the ac waves are a
        superposition — prefer xf() for per-source references when an xf
        result exists; a lone ac result is accepted as intended."""
        from .acdata import load_ac

        excited = self.excited_sources()
        if len(excited) > 1:
            hint = ("an xf result exists — use xf() for per-source "
                    "references" if self.has_xf() else
                    "single-source validation against these waves may be "
                    "wrong unless the run was reconfigured")
            warnings.warn(
                f"CIN records {len(excited)} AC-excited sources "
                f"({', '.join(sorted(excited))}): the ac waves superpose "
                f"them; {hint}"
            )
        return load_ac(self._base / name, self.backend)

    def xf(self, name: str = "xf.xf"):
        """Spectre xf results: isolated per-source transfer functions to
        the xf output, independent of the run's AC magnitudes. For a 0 A
        parallel port marker, xf().tf(marker) is the port impedance."""
        from .acdata import load_xf

        return load_xf(self._base / name, self.backend)

    def analyses(self) -> list[str]:
        """Names of the analyses this psf directory carries -- informative
        only (the GUI shows what simulator truth is available). psfascii
        file names first; the SRR result list when the dir is binary."""
        found = []
        base = Path(self._base)
        for fname, tag in (("dcOp.dc", "dcOp"), ("dcOpInfo.info", "dcOpInfo"),
                           ("ac.ac", "ac"), ("xf.xf", "xf"),
                           ("stb.stb", "stb"), ("tran.tran", "tran"),
                           ("noise.noise", "noise")):
            if (base / fname).exists():
                found.append(tag)
        if not found:
            try:
                from .backends import srr_available

                if srr_available():
                    from cdspythonsrr.core.ocean import openResults, results

                    openResults(str(base))
                    found = list(results())
            except Exception:
                pass
        return found

    def stb_probe(self) -> str | None:
        """The stb analysis's designated probe as a CIN device name, or None.

        Sources, in order: the psfascii stb header (carries "probe"), then
        the run's own netlist (netlist/input.scs beside the psf dir, or
        input.scs in/above it) -- the definitive record of the analyses as
        DEFINED, which is what the ADE launch flow needs since binary
        results don't surface the setting. The sim name is mapped back to
        the CIN name through the flat devices."""
        import re

        sim = None
        try:
            sim = self.stb().probe
        except Exception:
            sim = None
        if not sim:
            base = Path(self._base)
            for cand in (base.parent / "netlist" / "input.scs",
                         base / "input.scs", base.parent / "input.scs"):
                try:
                    if not cand.is_file():
                        continue
                    for line in cand.read_text(
                            encoding="utf-8", errors="replace").splitlines():
                        if re.match(r"\s*\S+\s+stb\b", line):
                            m = re.search(r"\bprobe=([\w./]+)", line)
                            if m:
                                sim = m.group(1)
                                break
                except OSError:
                    continue
                if sim:
                    break
        if not sim:
            return None
        for d in self.flat.devices:
            if d.sim_name == sim or d.name == sim:
                return d.name
        return sim

    def stb(self, name: str = "stb.stb"):
        """Spectre stb results: Tian-probe loop gain at the run's iprobe,
        with the simulator's phase/gain margins -- the validation reference
        for reconstructed loop gain (docs/loopgain-plan.md)."""
        from .stbdata import load_stb

        return load_stb(self._base / name, self.backend)
