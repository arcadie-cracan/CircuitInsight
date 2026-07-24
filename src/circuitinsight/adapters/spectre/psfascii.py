"""Minimal PSFASCII reader for Spectre info/dc results.

Handles the two VALUE styles seen in real files (checked-in fixtures):

    "vout" "V" 9.96e-01                       # scalar-typed entry (dcOp.dc)
    "M1" "bsim4" (                            # struct-typed entry (dcOpInfo.info)
    4.68e-04
    ...
    ) PROP("model" "M1.nfet_01v8.0")

Sweep/trace sections (AC results) are not handled here yet — that is the M3
validation-harness work.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

_TOKEN = re.compile(r'"(?:[^"\\]|\\.)*"|\(|\)|[^\s()]+')

_SECTIONS = {"HEADER", "TYPE", "SWEEP", "TRACE", "VALUE", "END"}


class PsfError(ValueError):
    pass


@dataclass
class StructType:
    name: str
    fields: list[str] = field(default_factory=list)


@dataclass
class PsfEntry:
    name: str
    type_name: str
    value: float | dict[str, object]     # dict of field -> value for structs
    props: dict[str, object] = field(default_factory=dict)


@dataclass
class PsfFile:
    header: dict[str, object]
    types: dict[str, StructType | None]  # None marks scalar types
    entries: dict[str, PsfEntry]
    # swept results (AC etc.):
    sweeps: list[str] = field(default_factory=list)
    traces: dict[str, str] = field(default_factory=dict)      # name -> type
    sweep_values: list[float] = field(default_factory=list)
    trace_values: dict[str, list] = field(default_factory=dict)


class _Tokens:
    def __init__(self, text: str):
        self.toks = _TOKEN.findall(text)
        self.i = 0

    def peek(self) -> str | None:
        return self.toks[self.i] if self.i < len(self.toks) else None

    def next(self) -> str:
        tok = self.peek()
        if tok is None:
            raise PsfError("unexpected end of file")
        self.i += 1
        return tok


def _unquote(tok: str) -> str:
    return tok[1:-1] if tok.startswith('"') else tok


def _is_quoted(tok: str) -> bool:
    return tok.startswith('"')


def _atom(tok: str) -> object:
    """Convert a bare token to int/float; quoted tokens stay strings."""
    if _is_quoted(tok):
        return _unquote(tok)
    try:
        return int(tok)
    except ValueError:
        pass
    try:
        return float(tok)          # handles 'nan', 'inf', scientific notation
    except ValueError:
        return tok


def _skip_prop(t: _Tokens) -> dict[str, object]:
    """Parse PROP( "k" v ... ) into a dict (nested parens do not occur)."""
    props: dict[str, object] = {}
    t.next()  # (
    while t.peek() != ")":
        key = _unquote(t.next())
        props[key] = _atom(t.next())
    t.next()  # )
    return props


def parse_psfascii(path: str | Path) -> PsfFile:
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    t = _Tokens(text)

    if t.next() != "HEADER":
        raise PsfError(f"{path}: not a PSFASCII file (missing HEADER)")

    header: dict[str, object] = {}
    while t.peek() is not None and t.peek() not in _SECTIONS:
        key = _unquote(t.next())
        header[key] = _atom(t.next())

    psf = PsfFile(header=header, types={}, entries={})

    while t.peek() is not None:
        section = t.next()
        if section == "END":
            break
        if section == "TYPE":
            _parse_types(t, psf.types)
        elif section == "SWEEP":
            _parse_sweep(t, psf)
        elif section == "TRACE":
            _parse_trace(t, psf)
        elif section == "VALUE":
            if psf.sweeps:
                _parse_swept_values(t, psf)
            else:
                _parse_values(t, psf.types, psf.entries)
        else:
            raise PsfError(f"{path}: unexpected section {section!r}")

    return psf


def _parse_types(t: _Tokens, types: dict) -> None:
    while t.peek() is not None and t.peek() not in _SECTIONS:
        name = _unquote(t.next())
        kind = t.next()
        if kind == "STRUCT":
            st = StructType(name)
            t.next()  # (
            while t.peek() != ")":
                fname = _unquote(t.next())
                while t.peek() not in ("PROP", ")") and not _is_quoted(t.peek()):
                    t.next()  # consume FLOAT DOUBLE / INT LONG / INT BYTE ...
                if t.peek() == "PROP":
                    t.next()
                    _skip_prop(t)
                st.fields.append(fname)
            t.next()  # )
            if t.peek() == "PROP":
                t.next()
                _skip_prop(t)
            types[name] = st
        else:
            # scalar type: e.g. "V" FLOAT DOUBLE PROP( ... )
            while t.peek() not in ("PROP",) and t.peek() not in _SECTIONS \
                    and not _is_quoted(t.peek()):
                t.next()
            if t.peek() == "PROP":
                t.next()
                _skip_prop(t)
            types[name] = None


def _parse_sweep(t: _Tokens, psf: PsfFile) -> None:
    while t.peek() is not None and t.peek() not in _SECTIONS:
        name = _unquote(t.next())
        t.next()  # sweep type name, e.g. "sweep"
        if t.peek() == "PROP":
            t.next()
            _skip_prop(t)
        psf.sweeps.append(name)


def _parse_trace(t: _Tokens, psf: PsfFile) -> None:
    while t.peek() is not None and t.peek() not in _SECTIONS:
        name = _unquote(t.next())
        type_name = _unquote(t.next())
        if t.peek() == "PROP":
            t.next()
            _skip_prop(t)
        psf.traces[name] = type_name
        psf.trace_values[name] = []


def _parse_swept_values(t: _Tokens, psf: PsfFile) -> None:
    if len(psf.sweeps) != 1:
        raise PsfError(f"only single-variable sweeps supported, got {psf.sweeps}")
    sweep_var = psf.sweeps[0]
    while t.peek() is not None and t.peek() not in _SECTIONS:
        name = _unquote(t.next())
        if t.peek() == "(":                      # complex value (re im)
            t.next()
            re_ = _atom(t.next())
            im_ = _atom(t.next())
            if t.next() != ")":
                raise PsfError(f"trace {name!r}: malformed complex value")
            value: object = complex(re_, im_)
        else:
            value = _atom(t.next())
        if name == sweep_var:
            psf.sweep_values.append(value)
        elif name in psf.trace_values:
            psf.trace_values[name].append(value)
        # unknown names (grouped traces etc.) are ignored


def _parse_values(t: _Tokens, types: dict, entries: dict) -> None:
    while t.peek() is not None and t.peek() not in _SECTIONS:
        name = _unquote(t.next())
        type_name = _unquote(t.next())
        st = types.get(type_name)
        if st is not None:                       # struct entry
            if t.next() != "(":
                raise PsfError(f"entry {name!r}: expected '(' for struct value")
            raw = []
            while t.peek() != ")":
                raw.append(_atom(t.next()))
            t.next()  # )
            if len(raw) != len(st.fields):
                raise PsfError(
                    f"entry {name!r}: {len(raw)} values for {len(st.fields)} "
                    f"fields of struct {type_name!r}"
                )
            value: float | dict = dict(zip(st.fields, raw))
        else:                                    # scalar entry
            value = _atom(t.next())
        props = {}
        if t.peek() == "PROP":
            t.next()
            props = _skip_prop(t)
        entries[name] = PsfEntry(name, type_name, value, props)
