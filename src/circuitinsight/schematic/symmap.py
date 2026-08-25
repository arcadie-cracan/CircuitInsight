"""CIN device -> SVG symbol mapping for Arcadie's component library.

The CIN already classified every PDK cell into a device_type +
polarity, so the map keys on THAT, never on foundry cell names (the
public repo must stay PDK-free). Terminal names map CIN terminals to
the library's anchor names (../terminals.csv conventions: 1/2 bipoles,
+/- sources, G/D/S/B MOS, B/C/E bipolars).

The library root is personal and never ships with the repo: it comes
from the CIN_SYMLIB environment variable or an explicit path.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SymbolSpec:
    file: str                 # svg file name inside the library root
    terms: dict               # CIN terminal -> library anchor name
    #: the symbol has no canonical size (a balun, a block): fit it to
    #: the harvested pin pitch like a block, orientation by residual
    scalable: bool = False


_MOS_N = SymbolSpec("nmos-4t.svg", {"g": "G", "d": "D", "s": "S", "b": "B"})
_MOS_P = SymbolSpec("pmos-4t.svg", {"g": "G", "d": "D", "s": "S", "b": "B"})

SYMBOLS = {
    ("mosfet", "n"): _MOS_N,
    ("mosfet", "p"): _MOS_P,
    ("bjt", "npn"): SymbolSpec("npn.svg", {"b": "B", "c": "C", "e": "E"}),
    ("bjt", "pnp"): SymbolSpec("pnp.svg", {"b": "B", "c": "C", "e": "E"}),
    ("resistor", None): SymbolSpec("res.svg", {"p": "1", "n": "2"}),
    ("capacitor", None): SymbolSpec("cap.svg", {"p": "1", "n": "2"}),
    ("inductor", None): SymbolSpec("ind.svg", {"p": "1", "n": "2"}),
    ("diode", None): SymbolSpec("diode.svg", {"p": "A", "n": "K"}),
    ("vsource", None): SymbolSpec("vsrc.svg", {"p": "+", "n": "-"}),
    ("isource", None): SymbolSpec("isrc.svg", {"p": "+", "n": "-"}),
    ("vccs", None): SymbolSpec("ccs.svg", {"p": "+", "n": "-"}),
    # analogLib ideal_balun: d = the single-ended differential port
    # (the library's primary p), c = the centre tap (scom), p/n = the
    # balanced pair (s+/s-); the harvest puts c,d left and p,n right,
    # as the symbol does
    ("balun", None): SymbolSpec("ideal-balun.svg",
                                {"d": "p", "c": "scom", "p": "s+",
                                 "n": "s-"}, scalable=True),
    # sp1tswitch: the exporter maps its CDF terminals positionally,
    # t0 -> p (drawn on the right), t1 -> n (left); the library's
    # anchors are 1 (left) and 2 (right). The polarity slot carries the
    # resolved AC state (models.small_signal.switch_ac_state) so the
    # drawing shows the configuration the solver actually sees; a
    # switch with no stated position falls back to closed
    ("switch", "open"): SymbolSpec("switch-open.svg",
                                   {"p": "2", "n": "1"}),
    ("switch", "closed"): SymbolSpec("switch-closed.svg",
                                     {"p": "2", "n": "1"}),
    ("switch", None): SymbolSpec("switch-closed.svg",
                                 {"p": "2", "n": "1"}),
}


#: hint instances the CIN deliberately ignores (rail glyphs, pins,
#: no-connects): they exist only as drawing, so they map by their
#: GENERIC Cadence cell name — analogLib/basic are stock Cadence
#: libraries, never foundry cells, so the public repo stays PDK-free.
#: their ONE anchor registers onto the instance ORIGIN: for rails and
#: pins the harvest shows the terminal coinciding with xy, so the origin
#: IS the connection point (the label-inclusive bbox center is not).
CELL_SYMBOLS = {
    "vdd": SymbolSpec("vdd.svg", {"@origin": "1"}),
    "vss": SymbolSpec("vss.svg", {"@origin": "1"}),
    "gnd": SymbolSpec("gnd.svg", {"@origin": "1"}),
    "iopin": SymbolSpec("pin-ixo.svg", {"@origin": "p"}),
    "ipin": SymbolSpec("pin-in.svg", {"@origin": "p"}),
    "opin": SymbolSpec("pin-out.svg", {"@origin": "p"}),
    "noConn": SymbolSpec("no-connect.svg", {"@origin": "1"}),
}

#: cells that draw anonymously: a ground, supply or no-connect symbol
#: is pure decoration, its instance name (I3, I14, ...) names nothing
#: anyone references, so the composer writes no designator for it
UNLABELLED_CELLS = frozenset({"vdd", "vss", "gnd", "noConn"})


#: harvested (Virtuoso pin) names -> CIN canonical terminal names. The
#: CIN adapter canonicalizes on export; the hints carry the raw names.
RAW_TERMS = {"PLUS": "p", "MINUS": "n", "D": "d", "G": "g", "S": "s",
             "B": "b", "C": "c", "E": "e",
             "t0": "p", "t1": "n"}          # sp1tswitch, positional


def anchor_for(spec: SymbolSpec, raw_term: str):
    """Library anchor name for a harvested terminal name, or None."""
    canon = RAW_TERMS.get(raw_term, raw_term)
    return spec.terms.get(canon, spec.terms.get(raw_term))


def resolve_cell(cell: str):
    """Symbol for a drawing-only hint instance (absent from the CIN)."""
    return CELL_SYMBOLS.get(cell)


def resolve(device_type: str, polarity: str | None = None):
    """SymbolSpec for a CIN device, or None when the map has no entry
    (the composer then draws a labelled placeholder box — visible and
    honest, never silently dropped)."""
    return SYMBOLS.get((device_type, polarity),
                       SYMBOLS.get((device_type, None)))
