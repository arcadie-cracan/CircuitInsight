"""Expand flattened devices into linear small-signal primitives.

Parameter names follow the verified Spectre dcOpInfo mapping
(docs/spectre-op-mapping.md): canonical values are magnitudes; polarity does
not change the small-signal element values, only the DC quantities — hence
NMOS and PMOS expand identically with the (d, g, s, b) orientation.

Values come from `op` (simulator OP data keyed by sim_name, wins) or from the
device's CIN `params` (hand-authored golden circuits). Parameters that are
absent or zero simply emit no primitive.
"""
from __future__ import annotations

from ..adapters.cin import FlatCircuit, FlatDevice
from ..engine.primitives import Primitive
from ..values import parse_value

# MOSFET small-signal expansion: (param, kind, terminal template)
_MOSFET = [
    ("gm", "vccs", ("d", "s", "g", "s")),    # i(d->s) = gm * v(g,s)
    ("gmbs", "vccs", ("d", "s", "b", "s")),  # i(d->s) = gmbs * v(b,s)
    ("gds", "g", ("d", "s")),
    ("cgs", "c", ("g", "s")),
    ("cgd", "c", ("g", "d")),
    ("cgb", "c", ("g", "b")),
    ("cdb", "c", ("d", "b")),
    ("csb", "c", ("s", "b")),
]

# Impact-ionization (substrate-current) linearization: the ac/stb small-signal
# model of the simulator includes dI_sub/dv terms that dcOpInfo never reports
# (r2r_bias_pos_loop investigation, 2026-07-19). The current flows d->b and is
# controlled by v(d,s) and v(b,s); with this orientation both polarities take
# POSITIVE coefficients. Values come from CIN params (or a future OP source);
# absent params emit nothing, so ordinary circuits are unaffected. Symbols
# follow the gm convention: gii_d_<inst>, gii_m_<inst>.
_MOSFET_II = [
    ("gii_d", "vccs", ("d", "b", "d", "s")),  # i(d->b) = gii_d * v(d,s)
    ("gii_m", "vccs", ("d", "b", "b", "s")),  # i(d->b) = gii_m * v(b,s)
]

_BJT = [
    ("gm", "vccs", ("c", "e", "b", "e")),
    ("gpi", "g", ("b", "e")),
    ("go", "g", ("c", "e")),
    ("cpi", "c", ("b", "e")),
    ("cmu", "c", ("b", "c")),
]


class ModelError(ValueError):
    pass


def _val(dev: FlatDevice, param: str, op: dict | None) -> float | None:
    if op is not None and param in op:
        return float(op[param])
    if param in dev.params:
        return parse_value(dev.params[param])
    return None


def _expand_table(dev: FlatDevice, table, op,
                  terminals: dict | None = None) -> list[Primitive]:
    tmap = terminals if terminals is not None else dev.terminals
    prims = []
    for param, kind, terms in table:
        value = _val(dev, param, op)
        if value is None or value == 0.0:
            continue
        nodes = tuple(tmap[t] for t in terms)
        prims.append(Primitive(dev.name, param, kind, nodes, abs(value)))
    if not prims:
        raise ModelError(
            f"{dev.name}: no small-signal parameters found (need at least gm or gds "
            f"via OP data or CIN params)"
        )
    return prims


def _expand_bjt(dev: FlatDevice, op: dict | None, extrinsic: bool,
                ground: str | None) -> list[Primitive]:
    """Hybrid-pi BJT. bjt_model='intrinsic' (default) stamps the core
    directly on the external terminals; 'extrinsic' inserts the series
    base/emitter/collector resistances (rb/re/rc) through per-terminal
    internal nodes, so the intrinsic core sees the internal voltages --
    matching the simulator's device more closely at the cost of extra
    nodes."""
    t = dev.terminals
    prims: list[Primitive] = []
    tmap = dict(t)
    if extrinsic:
        for term, key in (("c", "rc"), ("b", "rb"), ("e", "re")):
            r = _val(dev, key, op)
            if r and r > 0:
                internal = f"{dev.name}#{term}i"     # unique internal net
                prims.append(Primitive(dev.name, key, "r",
                                       (t[term], internal), r))
                tmap[term] = internal
    prims += _expand_table(dev, _BJT, op, terminals=tmap)
    # collector-substrate junction cap sits at the external collector
    # (outside rc), referenced to the substrate terminal or ground
    csub = _val(dev, "csub", op)
    sub = t.get("s", ground)
    if csub and sub is not None:
        prims.append(Primitive(dev.name, "csub", "c", (t["c"], sub), abs(csub)))
    return prims


def _expand_mosfet_matrix(dev: FlatDevice, op: dict | None) -> list[Primitive]:
    """Exact charge-matrix MOS model (docs/transcap-analysis.md): the nine
    signed source-referenced K_ij trans-capacitances plus junction caps,
    with the conductances of the lumped model."""
    t = dev.terminals
    prims = []
    for param, kind, terms in _MOSFET[:3] + _MOSFET_II:   # gm, gmbs, gds, gii_*
        v = _val(dev, param, op)
        if v not in (None, 0.0):
            nodes = tuple(dev.terminals[x] for x in terms)
            prims.append(Primitive(dev.name, param, kind, nodes, abs(v)))
    for i in "dgb":
        for j in "dgb":
            v = _val(dev, f"k{i}{j}", op)
            if v is None:
                raise ModelError(
                    f"{dev.name}: cap_model='matrix' needs the full "
                    f"trans-capacitance matrix (missing k{i}{j}); use "
                    f"cap_model='lumped' or provide OP data with c{i}{j}")
            if v:
                prims.append(Primitive(dev.name, f"k{i}{j}", "cx",
                                       (t[i], t["s"], t[j], t["s"]), v))
    for param, a, b in (("cjd", "d", "b"), ("cjs", "s", "b")):
        v = _val(dev, param, op)
        if v:
            prims.append(Primitive(dev.name, param, "c", (t[a], t[b]), abs(v)))
    return prims


def expand_device(dev: FlatDevice, op: dict | None = None,
                  cap_model: str = "lumped",
                  ground: str | None = None,
                  bjt_model: str = "intrinsic") -> list[Primitive]:
    t = dev.terminals
    dt = dev.device_type
    if dt == "mosfet":
        if cap_model == "matrix":
            return _expand_mosfet_matrix(dev, op)
        return _expand_table(dev, _MOSFET + _MOSFET_II, op)
    if dt == "bjt":
        return _expand_bjt(dev, op, bjt_model == "extrinsic", ground)
    if dt == "diode":
        prims = []
        gd = _val(dev, "gd", op)
        cd = _val(dev, "cd", op)
        if gd:
            prims.append(Primitive(dev.name, "gd", "g", (t["p"], t["n"]), abs(gd)))
        if cd:
            prims.append(Primitive(dev.name, "cd", "c", (t["p"], t["n"]), abs(cd)))
        return prims
    if dt == "resistor":
        v = _val(dev, "r", op)                       # not `or`: r=0 is a value
        v = _val(dev, "res", op) if v is None else v
        return [Primitive(dev.name, "", "r", (t["p"], t["n"]), v)]
    if dt == "capacitor":
        v = _val(dev, "c", op)
        v = _val(dev, "cap", op) if v is None else v
        return [Primitive(dev.name, "", "c", (t["p"], t["n"]), v)]
    if dt == "inductor":
        return [Primitive(dev.name, "", "l", (t["p"], t["n"]), _val(dev, "l", op))]
    if dt == "vsource":
        return [Primitive(dev.name, "", "vsrc", (t["p"], t["n"]), 0.0)]
    if dt == "isource":
        return [Primitive(dev.name, "", "isrc", (t["p"], t["n"]), 0.0)]
    if dt == "vccs":
        return [Primitive(dev.name, "", "vccs", (t["p"], t["n"], t["cp"], t["cn"]),
                          _val(dev, "gm", op))]
    if dt == "vcvs":
        return [Primitive(dev.name, "", "vcvs", (t["p"], t["n"], t["cp"], t["cn"]),
                          _val(dev, "gain", op))]
    if dt == "balun":
        # ideal constraint element: no parameters, no symbol
        return [Primitive(dev.name, "", "balun",
                          (t["d"], t["c"], t["p"], t["n"]), None)]
    if dt == "switch":
        # spectre ideal switch: 0 = open, 1 = closed (zero resistance).
        # ac_position dominates position (per spectre docs); the small-signal
        # circuit is the AC-analysis configuration.
        state = _val(dev, "ac_position", None)
        if state is None:
            state = _val(dev, "position", None) or 0.0
        state = int(state)
        if state == 0:
            return []                                    # open: element absent
        if state == 1:
            # closed: ideal short = 0 V source branch
            return [Primitive(dev.name, "", "vsrc", (t["p"], t["n"]), 0.0)]
        raise ModelError(f"{dev.name}: multi-throw switch position {state} "
                         f"not supported (SPST only)")
    if dt in ("ccvs", "cccs"):
        raise ModelError(f"{dev.name}: {dt} not supported yet (planned: probe-branch controls)")
    raise ModelError(f"{dev.name}: no small-signal model for device_type {dt!r}")


def expand_circuit(flat: FlatCircuit, op_data: dict[str, dict] | None = None,
                   cap_model: str = "lumped",
                   bjt_model: str = "intrinsic") -> list[Primitive]:
    """Expand every device; op_data maps sim_name -> {param: value}.
    cap_model: 'lumped' (reciprocal 5-cap MOS model) or 'matrix' (exact
    charge-based trans-capacitance matrix). bjt_model: 'intrinsic'
    (hybrid-pi on the external terminals) or 'extrinsic' (adds rb/re/rc
    series resistances via internal nodes)."""
    if cap_model not in ("lumped", "matrix"):
        raise ModelError(f"unknown cap_model {cap_model!r}")
    if bjt_model not in ("intrinsic", "extrinsic"):
        raise ModelError(f"unknown bjt_model {bjt_model!r}")
    gnd = flat.ground[0] if flat.ground else None
    prims: list[Primitive] = []
    for dev in flat.devices:
        op = op_data.get(dev.sim_name) if op_data else None
        prims.extend(expand_device(dev, op, cap_model, ground=gnd,
                                   bjt_model=bjt_model))
    return prims
