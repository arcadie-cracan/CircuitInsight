"""Linear small-signal primitives — the only elements the MNA engine knows.

Each primitive carries the owning instance, the parameter it represents (which
determines its symbol name, e.g. param="gm" of inst "M1" -> symbol gm_M1), its
node tuple, and an optional numeric value from OP data or CIN params.
"""
from __future__ import annotations

from dataclasses import dataclass

# kind -> number of nodes
KINDS = {
    "r": 2,      # resistance; symbol is the resistance, stamped as 1/R
    "g": 2,      # conductance, stamped directly
    "c": 2,      # capacitance, stamped as s*C
    "l": 2,      # inductance (adds a branch current)
    "vccs": 4,   # (p, n, cp, cn): i(p->n) = gm * v(cp, cn)
    "cx": 4,     # trans-capacitance: i(p->n) = s * C * v(cp, cn); C signed
    "vcvs": 4,   # (p, n, cp, cn): v(p, n) = gain * v(cp, cn); adds a branch
    "vsrc": 2,   # independent V source (p, n); adds a branch
    "isrc": 2,   # independent I source (p, n); current flows p->n inside
    "balun": 4,  # ideal balun (d, c, p, n); adds two branches, no symbol
}

# kinds whose symbol is inherently positive (helps sympy cancel/simplify)
POSITIVE_KINDS = {"r", "g", "c", "l"}
POSITIVE_PARAMS = {"gm", "gmbs", "gds", "gpi", "go"}


@dataclass(frozen=True)
class Primitive:
    inst: str            # owning flattened instance name
    param: str           # parameter name; "" for single-valued elements (R, C…)
    kind: str
    nodes: tuple[str, ...]
    value: float | None = None

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"unknown primitive kind {self.kind!r}")
        if len(self.nodes) != KINDS[self.kind]:
            raise ValueError(
                f"{self.inst}/{self.param}: kind {self.kind!r} takes "
                f"{KINDS[self.kind]} nodes, got {len(self.nodes)}"
            )

    @property
    def is_positive(self) -> bool:
        return self.kind in POSITIVE_KINDS or self.param in POSITIVE_PARAMS
