"""Subcircuit blocks drawn as library symbols: the mapping and its proposal.

A subcircuit instance draws as a plain BOX until the user confirms a
symbol for its cell: which library symbol, which pin goes to which
anchor, and what to do with the pins the symbol has no anchor for.
That mapping is user-owned, per CELL (every instance of the OTA draws
the same way), and lives in a sidecar next to the CIN::

    tb_fc.symbols.json
    {"symbols_version": "0.1",
     "cells": {"ota_folded_cascode": {
         "symbol": "gm.svg",
         "pins": {"vin_p": "in+", "vin_n": "in-", "vout": "out"},
         "stubs": {"ib": "auto"},
         "confirmed": true,
         "evidence": {...}}}}

The tool PROPOSES the mapping from evidence the user can check, ranked
by trust, and writes it unconfirmed; nothing heuristic reaches the
picture until ``confirmed`` is true:

1. netlist roles, name-independent -- inside the subcircuit a port
   that reaches only MOS gates is an input, only drains an output, a
   gate+drain diode a bias; at the parent, a port on an isource is a
   bias, on the balun or a vsource an input, carrying the load an
   output;
2. geometry -- the symbol's orientation is whichever of the eight
   places its anchors nearest the harvested pins (done in compose);
3. names -- only to split in+ from in- (p/+/inp vs n/-/inn), since a
   mirror makes either polarity fit geometrically.

Pins without an anchor (bias, supplies) become named STUBS on the
symbol edge nearest the harvested pin: real connections, never dropped.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

SYMBOLS_VERSION = "0.1"

#: library anchor name -> role it accepts
ANCHOR_ROLES = {"in+": "input", "in-": "input", "in": "input",
                "out": "output", "out1": "output", "out2": "output"}

#: symbol by (inputs, outputs) among the ports with an anchor role
SYMBOL_BY_SHAPE = {(2, 1): "gm.svg", (2, 2): "gm-diff.svg",
                   (1, 1): "buf.svg", (1, 2): "gm-diff.svg"}

_PLUS = re.compile(r"(^|[^a-z])(p|plus|pos|inp|ip)$|\+$", re.I)
_MINUS = re.compile(r"(^|[^a-z])(n|minus|neg|inn|im|in)$|-$", re.I)


@dataclass
class BlockSymbol:
    symbol: str                       # library file
    pins: dict                        # port -> anchor
    stubs: dict = field(default_factory=dict)   # port -> edge | "auto"
    confirmed: bool = False
    evidence: dict = field(default_factory=dict)  # port -> [str]

    def to_json(self) -> dict:
        return {"symbol": self.symbol, "pins": dict(self.pins),
                "stubs": dict(self.stubs), "confirmed": self.confirmed,
                "evidence": {k: list(v) for k, v in self.evidence.items()}}

    @classmethod
    def from_json(cls, d: dict) -> "BlockSymbol":
        return cls(symbol=d["symbol"], pins=dict(d.get("pins", {})),
                   stubs=dict(d.get("stubs", {})),
                   confirmed=bool(d.get("confirmed", False)),
                   evidence={k: list(v)
                             for k, v in d.get("evidence", {}).items()})


class BlockSymbolsError(ValueError):
    pass


def sidecar_path(cin_path) -> Path:
    """``tb_fc.cin.json`` -> ``tb_fc.symbols.json`` beside it."""
    p = Path(cin_path)
    stem = p.name[:-len(".cin.json")] if p.name.endswith(".cin.json") \
        else p.stem
    return p.with_name(stem + ".symbols.json")


def load_block_symbols(path) -> dict:
    """cell -> BlockSymbol; {} when the sidecar does not exist."""
    p = Path(path)
    if not p.exists():
        return {}
    d = json.loads(p.read_text(encoding="utf-8"))
    if d.get("symbols_version") != SYMBOLS_VERSION:
        raise BlockSymbolsError(
            f"{p}: symbols_version {d.get('symbols_version')!r}, "
            f"expected {SYMBOLS_VERSION!r}")
    return {cell: BlockSymbol.from_json(v)
            for cell, v in d.get("cells", {}).items()}


def save_block_symbols(path, mapping: dict) -> None:
    d = {"symbols_version": SYMBOLS_VERSION,
         "cells": {cell: bs.to_json() for cell, bs in mapping.items()}}
    Path(path).write_text(json.dumps(d, indent=1) + "\n", encoding="utf-8")


def confirmed_only(mapping: dict) -> dict:
    """What compose may draw: the box stays until the user confirms."""
    return {c: b for c, b in mapping.items() if b.confirmed}


# ---------------------------------------------------------------- roles

def _internal_roles(cin: dict, cell: str) -> dict:
    """port -> (role, why) from what the port reaches INSIDE the cell."""
    cdef = cin["definitions"][cell]
    supplies = set(cin.get("globals", [])) | set(cin.get("ground", []))
    out = {}
    for port in cdef["ports"]:
        if port in supplies:
            out[port] = ("supply", f"{port} is a global net")
            continue
        touch = []                    # (device_type, terminal, name)
        per_dev: dict = {}
        for inst in cdef["instances"]:
            for term, net in inst.get("terminals", {}).items():
                if net == port:
                    touch.append((inst.get("device_type", "subckt"),
                                  term, inst["name"]))
                    per_dev.setdefault(inst["name"], set()).add(term)
        mos = [t for t in touch if t[0] == "mosfet"]
        if not touch:
            out[port] = ("unknown", "reaches nothing inside")
        elif mos and all(t[1] in ("g", "b") for t in mos) \
                and len(mos) == len(touch):
            out[port] = ("input", f"reaches only MOS gates "
                                  f"({', '.join(t[2] for t in mos)})")
        elif any({"g", "d"} <= terms for terms in per_dev.values()):
            dev = next(n for n, t in per_dev.items() if {"g", "d"} <= t)
            out[port] = ("bias", f"gate+drain of {dev} (diode)")
        elif mos and all(t[1] in ("d", "s") for t in mos):
            out[port] = ("output", f"reaches only MOS drains/sources "
                                   f"({', '.join(t[2] for t in mos)})")
        else:
            out[port] = ("unknown", "mixed: " + ", ".join(
                f"{t[2]}.{t[1]}" for t in touch))
    return out


def _parent_roles(cin: dict, cell: str) -> dict:
    """port -> (role, why) from what the FIRST instance of the cell
    sees at its parent: isource = bias, balun/vsource = input, a load
    to ground = output. Switches are walked through."""
    supplies = set(cin.get("globals", [])) | set(cin.get("ground", []))
    for pdef in cin["definitions"].values():
        for inst in pdef["instances"]:
            if inst.get("subckt") != cell:
                continue
            devs = [d for d in pdef["instances"] if d is not inst]
            out = {}
            for port, net in inst["terminals"].items():
                if net in supplies:
                    out[port] = ("supply", f"on {net}")
                    continue
                # direct devices first; only a net that carries
                # nothing but switches is walked through them (the
                # feedback switch on an output must not drag the
                # balun's evidence onto it)
                seen, todo, found = set(), [net], []
                while todo:
                    n = todo.pop()
                    if n in seen:
                        continue
                    seen.add(n)
                    here = [d for d in devs
                            if n in d.get("terminals", {}).values()]
                    direct = [d for d in here
                              if d.get("device_type") != "switch"]
                    for d in direct:
                        found.append((d.get("device_type", "subckt"),
                                      d["name"], d.get("terminals", {})))
                    if not direct:
                        for d in here:
                            todo += [v for v in d["terminals"].values()
                                     if v != n]
                kinds = {f[0] for f in found}
                if "isource" in kinds:
                    who = next(f[1] for f in found if f[0] == "isource")
                    out[port] = ("bias", f"driven by {who} (isource)")
                elif kinds & {"balun", "vsource"}:
                    who = next(f[1] for f in found
                               if f[0] in ("balun", "vsource"))
                    out[port] = ("input", f"driven by {who}")
                elif kinds & {"capacitor", "resistor"} and all(
                        f[0] in ("capacitor", "resistor") for f in found):
                    who = ", ".join(f[1] for f in found)
                    out[port] = ("output", f"loaded by {who}")
                elif not found:
                    out[port] = ("unknown", "open at the parent")
                else:
                    out[port] = ("unknown", "sees " + ", ".join(
                        f"{f[1]} ({f[0]})" for f in found))
            return out
    return {}


def pin_roles(cin: dict, cell: str) -> dict:
    """port -> (role, [evidence]) merging internal and parent views.
    Internal wins a conflict (the parent is one testbench; the cell is
    the cell); the conflict is recorded."""
    internal = _internal_roles(cin, cell)
    parent = _parent_roles(cin, cell)
    out = {}
    for port in cin["definitions"][cell]["ports"]:
        ri, wi = internal.get(port, ("unknown", "no internal view"))
        rp, wp = parent.get(port, ("unknown", "no parent view"))
        ev = [f"inside: {wi}", f"parent: {wp}"]
        if ri != "unknown" and rp != "unknown" and ri != rp:
            ev.append(f"CONFLICT: inside says {ri}, parent says {rp}")
        role = ri if ri != "unknown" else rp
        out[port] = (role, ev)
    return out


# ------------------------------------------------------------- proposal

def _polarity(name: str):
    if _PLUS.search(name):
        return "+"
    if _MINUS.search(name):
        return "-"
    return None


def propose(cin: dict, cell: str, available: set,
            harvested_terms: dict | None = None) -> BlockSymbol | None:
    """A BlockSymbol proposal (confirmed=False) for one cell, or None
    when no library symbol fits the port roles.

    available: library file names present.
    harvested_terms: port -> (x, y) in harvest coordinates, used only
    to order two inputs when their names carry no polarity (upper
    pin -> in+), recorded as evidence so the user knows it was guessed.
    """
    roles = pin_roles(cin, cell)
    ins = [p for p, (r, _) in roles.items() if r == "input"]
    outs = [p for p, (r, _) in roles.items() if r == "output"]
    symbol = SYMBOL_BY_SHAPE.get((len(ins), len(outs)))
    if not symbol or symbol not in available:
        return None
    evidence = {p: list(ev) for p, (_, ev) in roles.items()}
    pins = {}
    # inputs: polarity by name, else by harvested position (upper = +)
    if len(ins) == 1:
        pins[ins[0]] = "in"
    elif len(ins) == 2:
        pol = {p: _polarity(p) for p in ins}
        if set(pol.values()) == {"+", "-"}:
            for p, s in pol.items():
                pins[p] = "in" + s
                evidence[p].append(f"name says {s}")
        else:
            order = ins
            if harvested_terms and all(p in harvested_terms for p in ins):
                order = sorted(ins, key=lambda p: -harvested_terms[p][1])
                note = "by position (upper pin), names carry no polarity"
            else:
                note = "by port order, names carry no polarity"
            pins[order[0]], pins[order[1]] = "in+", "in-"
            for p in ins:
                evidence[p].append(note)
    if len(outs) == 1:
        pins[outs[0]] = "out"
    else:
        for i, p in enumerate(outs):
            pins[p] = f"out{i + 1}"
    stubs = {p: "auto" for p in roles if p not in pins}
    for p in stubs:
        evidence[p].append("no anchor on the symbol: drawn as a stub")
    return BlockSymbol(symbol=symbol, pins=pins, stubs=stubs,
                       confirmed=False, evidence=evidence)


def propose_all(cin: dict, available: set, hints=None) -> dict:
    """cell -> proposal for every subcircuit definition that has a
    fitting symbol. hints: loaded Hints, for pin positions."""
    out = {}
    top = cin.get("top")
    for cell in cin["definitions"]:
        if cell == top:
            continue
        terms = None
        if hints is not None:
            for dd in hints.definitions.values():
                for inst in dd.instances:
                    if inst.cell == cell and inst.terms:
                        terms = inst.terms
                        break
                if terms:
                    break
        bs = propose(cin, cell, available, terms)
        if bs:
            out[cell] = bs
    return out
