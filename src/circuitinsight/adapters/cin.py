"""CircuitInsight Netlist (CIN) — reference loader, validator, and flattener.

CIN is the only way topology enters the core. Format spec: docs/cin-spec.md,
machine-readable schema: schema/cin-0.1.schema.json. This module is the
authoritative validator.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

CIN_VERSION = "0.1"
HIER_SEP = "."

DEVICE_TERMINALS: dict[str, frozenset[str]] = {
    "mosfet": frozenset({"d", "g", "s", "b"}),
    "bjt": frozenset({"c", "b", "e"}),
    "diode": frozenset({"p", "n"}),
    "resistor": frozenset({"p", "n"}),
    "capacitor": frozenset({"p", "n"}),
    "inductor": frozenset({"p", "n"}),
    "vsource": frozenset({"p", "n"}),
    "isource": frozenset({"p", "n"}),
    "vcvs": frozenset({"p", "n", "cp", "cn"}),
    "vccs": frozenset({"p", "n", "cp", "cn"}),
    "ccvs": frozenset({"p", "n"}),
    "cccs": frozenset({"p", "n"}),
    "balun": frozenset({"d", "c", "p", "n"}),   # ideal balun (testbenches)
    "switch": frozenset({"p", "n"}),            # spectre ideal switch (SPST)
}

# terminals a device may carry beyond its required set (default to ground when
# absent). bjt substrate enables the collector-substrate junction cap.
OPTIONAL_TERMINALS: dict[str, frozenset[str]] = {
    "bjt": frozenset({"s"}),
}

REQUIRED_POLARITY: dict[str, frozenset[str]] = {
    "mosfet": frozenset({"n", "p"}),
    "bjt": frozenset({"npn", "pnp"}),
}


class CinError(ValueError):
    """Raised for any malformed or inconsistent CIN document."""

    def __init__(self, errors: list[str] | str):
        self.errors = [errors] if isinstance(errors, str) else errors
        super().__init__("\n".join(self.errors))


@dataclass(frozen=True)
class Instance:
    name: str
    terminals: dict[str, str]
    device_type: str | None = None
    subckt: str | None = None
    sim_name: str | None = None
    params: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    @property
    def is_device(self) -> bool:
        return self.device_type is not None

    @property
    def effective_sim_name(self) -> str:
        return self.sim_name if self.sim_name is not None else self.name


@dataclass(frozen=True)
class Definition:
    name: str
    ports: tuple[str, ...]
    instances: tuple[Instance, ...]


@dataclass(frozen=True)
class CinDoc:
    version: str
    top: str
    ground: tuple[str, ...]
    global_nets: tuple[str, ...]
    definitions: dict[str, Definition]
    design: dict = field(default_factory=dict)


@dataclass(frozen=True)
class FlatDevice:
    """A leaf device after hierarchy elaboration."""

    name: str  # hierarchical instance path, HIER_SEP-joined
    sim_name: str  # join key against simulator OP results
    device_type: str
    terminals: dict[str, str]  # terminal -> flat net name
    params: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)


@dataclass(frozen=True)
class FlatCircuit:
    devices: tuple[FlatDevice, ...]
    nets: frozenset[str]
    ground: tuple[str, ...]


def load_cin(path: str | Path) -> CinDoc:
    path = Path(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CinError(f"{path}: not valid JSON: {exc}") from exc
    return parse_cin(raw, source=str(path))


def parse_cin(raw: dict, source: str = "<memory>") -> CinDoc:
    errors: list[str] = []

    def err(msg: str) -> None:
        errors.append(f"{source}: {msg}")

    if not isinstance(raw, dict):
        raise CinError(f"{source}: top level must be a JSON object")

    version = raw.get("cin_version")
    if version != CIN_VERSION:
        raise CinError(
            f"{source}: unsupported cin_version {version!r} (this loader supports {CIN_VERSION!r})"
        )

    top = raw.get("top")
    if not isinstance(top, str) or not top:
        err("'top' must be a non-empty string")

    ground = raw.get("ground")
    if not isinstance(ground, list) or not ground or not all(isinstance(g, str) and g for g in ground):
        err("'ground' must be a non-empty list of net names")
        ground = []

    global_nets = raw.get("globals", [])
    if not isinstance(global_nets, list) or not all(isinstance(g, str) and g for g in global_nets):
        err("'globals' must be a list of net names")
        global_nets = []

    raw_defs = raw.get("definitions")
    if not isinstance(raw_defs, dict) or not raw_defs:
        err("'definitions' must be a non-empty object")
        raw_defs = {}

    definitions: dict[str, Definition] = {}
    for dname, rdef in raw_defs.items():
        definitions[dname] = _parse_definition(dname, rdef, err)

    if errors:
        raise CinError(errors)

    doc = CinDoc(
        version=version,
        top=top,
        ground=tuple(ground),
        global_nets=tuple(global_nets),
        definitions=definitions,
        design=raw.get("design", {}) or {},
    )
    _validate(doc, source)
    return doc


def _parse_definition(dname: str, rdef, err) -> Definition:
    if not isinstance(rdef, dict):
        err(f"definition '{dname}' must be an object")
        return Definition(dname, (), ())

    ports = rdef.get("ports", [])
    if not isinstance(ports, list) or not all(isinstance(p, str) and p for p in ports):
        err(f"definition '{dname}': 'ports' must be a list of net names")
        ports = []

    instances: list[Instance] = []
    for i, rinst in enumerate(rdef.get("instances", [])):
        if not isinstance(rinst, dict):
            err(f"definition '{dname}' instance #{i}: must be an object")
            continue
        name = rinst.get("name")
        if not isinstance(name, str) or not name:
            err(f"definition '{dname}' instance #{i}: 'name' must be a non-empty string")
            continue
        terminals = rinst.get("terminals")
        if not isinstance(terminals, dict) or not all(
            isinstance(k, str) and isinstance(v, str) and v for k, v in terminals.items()
        ):
            err(f"definition '{dname}' instance '{name}': 'terminals' must map terminal -> net name")
            terminals = {}
        instances.append(
            Instance(
                name=name,
                terminals=dict(terminals),
                device_type=rinst.get("device_type"),
                subckt=rinst.get("subckt"),
                sim_name=rinst.get("sim_name"),
                params=rinst.get("params", {}) or {},
                meta=rinst.get("meta", {}) or {},
            )
        )
    return Definition(name=dname, ports=tuple(ports), instances=tuple(instances))


def _validate(doc: CinDoc, source: str) -> None:
    errors: list[str] = []

    def err(msg: str) -> None:
        errors.append(f"{source}: {msg}")

    if doc.top not in doc.definitions:
        err(f"top definition '{doc.top}' not found in definitions")

    for dname, defn in doc.definitions.items():
        seen: set[str] = set()
        for inst in defn.instances:
            where = f"'{dname}'/'{inst.name}'"
            if inst.name in seen:
                err(f"{where}: duplicate instance name")
            seen.add(inst.name)
            if HIER_SEP in inst.name:
                err(f"{where}: instance name must not contain '{HIER_SEP}'")
            if inst.sim_name is not None and HIER_SEP in inst.sim_name:
                err(f"{where}: sim_name must not contain '{HIER_SEP}'")

            if (inst.device_type is None) == (inst.subckt is None):
                err(f"{where}: exactly one of 'device_type' or 'subckt' is required")
                continue

            if inst.is_device:
                required = DEVICE_TERMINALS.get(inst.device_type)
                if required is None:
                    err(f"{where}: unknown device_type '{inst.device_type}'")
                    continue
                got = frozenset(inst.terminals)
                optional = OPTIONAL_TERMINALS.get(inst.device_type, frozenset())
                if got != required and got != (required | optional):
                    exp = sorted(required)
                    if optional:
                        exp = f"{sorted(required)} (+ optional {sorted(optional)})"
                    err(
                        f"{where}: device_type '{inst.device_type}' requires terminals "
                        f"{exp}, got {sorted(got)}"
                    )
                polarities = REQUIRED_POLARITY.get(inst.device_type)
                if polarities is not None:
                    pol = inst.params.get("polarity")
                    if pol not in polarities:
                        err(
                            f"{where}: '{inst.device_type}' requires params.polarity in "
                            f"{sorted(polarities)}, got {pol!r}"
                        )
            else:
                target = doc.definitions.get(inst.subckt)
                if target is None:
                    err(f"{where}: unknown subckt '{inst.subckt}'")
                    continue
                got = frozenset(inst.terminals)
                want = frozenset(target.ports)
                if got != want:
                    err(
                        f"{where}: subckt '{inst.subckt}' has ports {sorted(want)}, "
                        f"got terminals {sorted(got)}"
                    )

    if not errors:
        cycle = _find_cycle(doc)
        if cycle:
            err("recursive subckt instantiation: " + " -> ".join(cycle))

    if errors:
        raise CinError(errors)


def _find_cycle(doc: CinDoc) -> list[str] | None:
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {name: WHITE for name in doc.definitions}

    def visit(name: str, stack: list[str]) -> list[str] | None:
        color[name] = GRAY
        stack.append(name)
        for inst in doc.definitions[name].instances:
            if inst.subckt is None or inst.subckt not in doc.definitions:
                continue
            child = inst.subckt
            if color[child] == GRAY:
                return stack[stack.index(child):] + [child]
            if color[child] == WHITE:
                found = visit(child, stack)
                if found:
                    return found
        stack.pop()
        color[name] = BLACK
        return None

    for name in doc.definitions:
        if color[name] == WHITE:
            found = visit(name, [])
            if found:
                return found
    return None


def flatten(doc: CinDoc) -> FlatCircuit:
    """Elaborate the hierarchy from `doc.top` into a flat device list.

    Naming semantics (docs/cin-spec.md): instance paths and sim_names are
    HIER_SEP-joined; ground/global nets are never prefixed; ports map to the
    parent nets bound at the instantiation site; other local nets get the
    instance-path prefix.
    """
    passthrough = set(doc.ground) | set(doc.global_nets)
    devices: list[FlatDevice] = []
    nets: set[str] = set(doc.ground)

    def expand(defn: Definition, path: str, sim_path: str, portmap: dict[str, str]) -> None:
        def resolve(local: str) -> str:
            if local in passthrough:
                return local
            if local in portmap:
                return portmap[local]
            return f"{path}{HIER_SEP}{local}" if path else local

        for inst in defn.instances:
            ipath = f"{path}{HIER_SEP}{inst.name}" if path else inst.name
            isim = (
                f"{sim_path}{HIER_SEP}{inst.effective_sim_name}"
                if sim_path
                else inst.effective_sim_name
            )
            bound = {t: resolve(n) for t, n in inst.terminals.items()}
            if inst.is_device:
                nets.update(bound.values())
                devices.append(
                    FlatDevice(
                        name=ipath,
                        sim_name=isim,
                        device_type=inst.device_type,
                        terminals=bound,
                        params=dict(inst.params),
                        meta=dict(inst.meta),
                    )
                )
            else:
                child = doc.definitions[inst.subckt]
                expand(child, ipath, isim, bound)

    expand(doc.definitions[doc.top], "", "", {})
    return FlatCircuit(devices=tuple(devices), nets=frozenset(nets), ground=doc.ground)
