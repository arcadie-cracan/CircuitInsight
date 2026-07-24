"""Parsing of component values with SPICE SI suffixes ("10k", "1.5p", "2meg")."""
from __future__ import annotations

import re

# SPICE convention: case-insensitive, 'm' is always milli, mega is 'meg'.
_SUFFIX = {
    "t": 1e12,
    "g": 1e9,
    "meg": 1e6,
    "k": 1e3,
    "m": 1e-3,
    "u": 1e-6,
    "n": 1e-9,
    "p": 1e-12,
    "f": 1e-15,
}

_VALUE_RE = re.compile(
    r"^\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*([a-zA-Z]*)\s*$"
)


def parse_value(v: str | int | float) -> float:
    """Parse a CIN params value: plain number or string with SPICE suffix."""
    if isinstance(v, (int, float)):
        return float(v)
    m = _VALUE_RE.match(v)
    if not m:
        raise ValueError(f"cannot parse value {v!r}")
    mantissa, suffix = float(m.group(1)), m.group(2).lower()
    if not suffix:
        return mantissa
    # ignore trailing unit letters after a valid suffix ("10kohm", "100meghz")
    for sfx in ("meg", "t", "g", "k", "m", "u", "n", "p", "f"):
        if suffix.startswith(sfx):
            return mantissa * _SUFFIX[sfx]
    raise ValueError(f"unknown SI suffix in value {v!r}")
