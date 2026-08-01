"""Canonical small-signal parameters from Spectre dcOpInfo records.

Implements the verified mapping of docs/spectre-op-mapping.md:
- trans-capacitances are reported as -dQi/dVj (negative off-diagonals): take
  magnitudes;
- junction caps cjd/cjs are separate and dominate the intrinsic cdb/csb;
- unset INT fields read INT_MAX, several FLOATs are nan: treat both as absent.

Backend-independent: records come from backends.read_dcopinfo_raw (psfascii
or cdspythonsrr), which normalizes to RawRecord.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

from .backends import RawRecord, read_dcopinfo_raw

_INT_SENTINEL = 2147483647

# psf struct name -> model class (a CIN device_type, or None = don't check)
_STRUCT_CLASSES = {
    "bsim4": "mosfet",
    "bsim6": "mosfet",
    "bsim3v3": "mosfet",
    "psp103": "mosfet",
    "utsoi": "mosfet",
    "resistor": "resistor",
    "capacitor": "capacitor",
    "inductor": "inductor",
    "relay": "switch",
    "switch": "switch",
    "vsource": "vsource",
    "isource": "isource",
    "diode": "diode",
    "bjt": "bjt",
    "vbic": "bjt",
    "hicum": "bjt",
    "mextram": "bjt",
}


def model_class(type_name: str) -> str | None:
    """Map a record's type to the CIN device_type it may join with.
    None means unknown/source-like: the join skips the type check."""
    if type_name.startswith("srr:"):
        cls = type_name[4:]
        return cls if cls in set(_STRUCT_CLASSES.values()) else None
    return _STRUCT_CLASSES.get(type_name)


@dataclass
class OpRecord:
    name: str
    model_type: str                      # struct name or pseudo "srr:<class>"
    params: dict[str, float]             # canonical small-signal params
    raw: dict[str, object] = field(default_factory=dict)
    props: dict[str, object] = field(default_factory=dict)

    @property
    def device_type(self) -> str | None:
        return model_class(self.model_type)


def _num(raw: dict, key: str) -> float | None:
    v = raw.get(key)
    if not isinstance(v, (int, float)):
        return None
    if isinstance(v, int) and abs(v) == _INT_SENTINEL:
        return None
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return float(v)


def _canonical_mosfet(raw: dict) -> dict[str, float]:
    out: dict[str, float] = {}
    for k in ("gm", "gds", "gmbs"):
        v = _num(raw, k)
        if v is not None:
            out[k] = abs(v)
    for k in ("cgs", "cgd", "cgb"):
        v = _num(raw, k)
        if v is not None:
            out[k] = abs(v)
    for canonical, junction, intrinsic in (("cdb", "cjd", "cdb"), ("csb", "cjs", "csb")):
        j = _num(raw, junction) or 0.0
        i = _num(raw, intrinsic) or 0.0
        total = abs(j) + abs(i)
        if total:
            out[canonical] = total
    # exact charge-matrix model inputs: junction-only caps plus the SIGNED
    # source-referenced K entries (K_ij = dQ_i/dV_j, i,j in {d,g,b})
    for canonical, junction in (("cjd", "cjd"), ("cjs", "cjs")):
        v = _num(raw, junction)
        if v is not None:
            out[canonical] = abs(v)
    for i in "dgb":
        for j in "dgb":
            v = _num(raw, f"c{i}{j}" if i != j else f"c{i}{i}")
            if v is not None:
                out[f"k{i}{j}"] = v
    # DC / reporting metadata (never becomes a primitive: names distinct).
    # isub/iavl are the substrate (impact-ionization) current -- carried so
    # the reconstruction can FLAG devices where II is significant but no
    # gii conductance is modeled (session.impact_ionization_devices).
    for k in ("ids", "vgs", "vds", "vth", "vdsat", "region", "gmoverid",
              "self_gain", "vearly", "ft", "isub", "iavl"):
        v = _num(raw, k)
        if v is not None:
            out[k] = v
    return out


def _canonical_bjt(raw: dict) -> dict[str, float]:
    out: dict[str, float] = {}
    # hybrid-pi core: transconductance, input/output conductances, junction
    # caps. Spectre names the output conductance g0 (not go); rpi/ro are the
    # reciprocals it also reports — prefer the conductances.
    for canonical, src in (("gm", "gm"), ("gpi", "gpi"), ("go", "g0"),
                           ("cpi", "cpi"), ("cmu", "cmu")):
        v = _num(raw, src)
        if v is not None:
            out[canonical] = abs(v)
    # extrinsic base-collector cap folds into Cmu; collector-substrate
    # junction cap (csub) is separate — it shunts the collector to the
    # substrate (AC ground) and sets much of the high-frequency rolloff
    v = _num(raw, "cmux")
    if v is not None and "cmu" in out:
        out["cmu"] += abs(v)
    v = _num(raw, "csub")
    if v is not None:
        out["csub"] = abs(v)
    # series parasitics (rb/re/rc) become explicit resistors in the model when
    # nonzero; keep them canonical so the expander can stamp them
    for k in ("rb", "re", "rc"):
        v = _num(raw, k)
        if v is not None and abs(v) > 0:
            out[k] = abs(v)
    # DC / reporting metadata (distinct names: never become primitives)
    for k in ("ic", "ib", "vbe", "vbc", "vce", "betadc", "betaac", "ft",
              "region"):
        v = _num(raw, k)
        if v is not None:
            out[k] = v
    return out


def _canonical_single(raw: dict, src: str, dst: str) -> dict[str, float]:
    v = _num(raw, src)
    return {dst: abs(v)} if v is not None else {}


def canonicalize(rec: RawRecord) -> OpRecord:
    cls = model_class(rec.type_name)
    if cls == "mosfet":
        params = _canonical_mosfet(rec.values)
    elif cls == "resistor":
        params = _canonical_single(rec.values, "res", "r")
    elif cls == "capacitor":
        params = _canonical_single(rec.values, "cap", "c")
    elif cls == "bjt":
        params = _canonical_bjt(rec.values)
    elif cls == "inductor":
        params = _canonical_single(rec.values, "ind", "l")
    elif cls == "switch":
        params = _canonical_single(rec.values, "res", "r") \
            or _canonical_single(rec.values, "r", "r")
    else:
        # sources, and model classes we don't map yet (bjt/diode need fixtures)
        params = {}
    return OpRecord(rec.name, rec.type_name, params, raw=rec.values, props=rec.props)


def load_dcopinfo(path: str | Path, backend: str = "auto") -> dict[str, OpRecord]:
    """Read a dcOpInfo results file/dir into canonical OP records by instance."""
    return {
        name: canonicalize(rec)
        for name, rec in read_dcopinfo_raw(path, backend).items()
    }
