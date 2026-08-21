"""Schematic layout hints: load, flatten, compose.

The composer needs the personal SVG symbol library (CIN_SYMLIB); those
tests skip cleanly where it is absent — the library never ships with
the repo."""
import os
import warnings
from pathlib import Path

import pytest

from circuitinsight.schematic import (flatten, load_hints, resolve)
from circuitinsight.schematic.hints import ORIENTS

FIX = Path(__file__).resolve().parent / "fixtures"
HINTS = FIX / "hints" / "tb_ota5t.hints.json"


def test_load_and_flatten_joins_cin_names():
    h = load_hints(HINTS)
    assert h.top == "tb_ota5t"
    flat = flatten(h)
    names = sorted(i.name for i in flat.instances)
    # hierarchy flattens with dot-joined names, exactly like the netlist
    assert "I0.MN0" in names and "I0.MP1" in names
    assert "VIND" in names and "I5" in names
    assert len(flat.wires) == 24
    # a subckt instance itself never survives flattening
    assert "I0" not in names


def test_flatten_composes_world_orientation():
    h = load_hints(HINTS)
    flat = flatten(h)
    by = {i.name: i for i in flat.instances}
    # the fixture mirrors MP0/MN0 locally (MY) inside an R0 parent:
    # the world matrix IS the mirror
    assert by["I0.MP0"].m == ORIENTS["MY"]
    assert by["I0.MP1"].m == ORIENTS["R0"]
    # geometry landed in top coordinates: the ota5t core sits inside
    # the I0 instance's bbox from the tb definition
    x0, y0, x1, y1 = by["I0.MP0"].bbox
    assert 2.2 <= x0 and x1 <= 4.4 and 1.2 <= y0 and y1 <= 3.2


def test_symbol_map_covers_the_cin_device_types():
    assert resolve("mosfet", "n").file == "nmos-4t.svg"
    assert resolve("mosfet", "p").file == "pmos-4t.svg"
    assert resolve("capacitor", None).file == "cap.svg"
    assert resolve("vsource", None).terms == {"p": "+", "n": "-"}
    assert resolve("frobnicator", None) is None    # placeholder box path


def test_strip_equations_is_balanced():
    from circuitinsight.schematic.compose import _strip_equations

    inner = ('<path d="M0 0"/>'
             '<g lib:role="equation"><text>M_1</text>'
             '<g lib:role="render"><path d="M1 1"/></g></g>'
             '<circle r="1"/>')
    out = _strip_equations(inner)
    assert "M_1" not in out and "render" not in out
    assert '<path d="M0 0"/>' in out and '<circle r="1"/>' in out


def _have_lib() -> bool:
    from circuitinsight.schematic.compose import BUNDLED_SYMLIB
    return bool(os.environ.get("CIN_SYMLIB"))         or (BUNDLED_SYMLIB / "terminals.csv").exists()


needs_lib = pytest.mark.skipif(
    not _have_lib(),
    reason="no symbol library (CIN_SYMLIB unset and the symbols "
           "submodule not checked out)")


@needs_lib
def test_compose_renders_every_mapped_instance():
    from circuitinsight.schematic import SymbolLibrary, compose
    from circuitinsight.session import SessionController

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        c = SessionController.open(
            FIX / "spectre" / "ota5t" / "tb_ota5t.cin.json",
            FIX / "spectre" / "ota5t" / "psf")
    devtypes = {d.name: (d.device_type, d.params.get("polarity"))
                for d in c._run.flat.devices}
    flat = flatten(load_hints(HINTS))
    svg = compose(flat, devtypes, SymbolLibrary())
    assert svg.startswith("<svg") and svg.rstrip().endswith("</svg>")
    # every hint instance is a CIN device here, so no placeholder
    # boxes (the placeholder's dashed-attribute form, not the symbols'
    # internal style strings)
    assert 'stroke-dasharray="1,1"' not in svg
    # the library's placeholder designators are stripped
    assert 'lib:role="equation"' not in svg
    # wires, dots and instance labels are present
    assert svg.count("<path d=\"M ") >= len(flat.wires)
    assert "I0.MP0" in svg and "VIND" in svg


def test_pages_keep_the_source_hierarchy():
    """The schematic is hierarchical like its source: one page per
    definition, subcircuit instances stay as instances (drawn as
    blocks with their harvested pins), never inlined."""
    from circuitinsight.schematic import page

    h = load_hints(HINTS)
    tb = page(h, "tb_ota5t")
    names = {i.name for i in tb.instances}
    assert "I0" in names                     # the subcircuit instance
    assert "I0.MN0" not in names             # nothing inlined
    core = page(h, "ota5t")
    assert {i.name for i in core.instances} == {"MP0", "MP1", "MN0",
                                                "MN1", "MN2"}
    # page geometry is the definition's own frame; orientation matrices
    # are the local orients
    by = {i.name: i for i in core.instances}
    assert by["MP0"].m == ORIENTS["MY"]
    # the page's own ports come along by NAME: the composer labels
    # port glyphs with these, never with the Cadence instance name
    assert sorted(pn["name"] for pn in core.pins) == ["inn", "inp",
                                                       "out", "vbn"]
    assert tb.pins == []


def test_port_glyph_takes_the_nearest_pin_name():
    from circuitinsight.schematic.compose import _pin_named
    from circuitinsight.schematic.hints import HintInstance, ORIENTS

    port = HintInstance(name="PIN1", lib="basic", cell="ipin",
                        xy=(0.0, 0.85), orient="R0",
                        bbox=(-0.2, 0.8, 0.1, 0.9), terms={},
                        m=ORIENTS["R0"])
    pins = [{"name": "inp", "xy": (0.05, 0.85)},
            {"name": "vbn", "xy": (0.0, 0.3)}]
    assert _pin_named(port, pins)["name"] == "inp"
    assert _pin_named(port, []) is None
    far = [{"name": "out", "xy": (2.2, 1.2)}]
    assert _pin_named(port, far) is None        # no guessing at range


@needs_lib
def test_compose_draws_subcircuit_blocks():
    from circuitinsight.schematic import SymbolLibrary, compose, page

    h = load_hints(HINTS)
    tb = page(h, "tb_ota5t")
    devtypes = {"VIND": ("vsource", None), "VINC": ("vsource", None),
                "VSUP": ("vsource", None), "IB": ("isource", None),
                "MN2": ("mosfet", "n"), "I5": ("balun", None),
                "S0": ("switch", None), "S1": ("switch", None)}
    svg = compose(tb, devtypes, SymbolLibrary(), subckts={"ota5t"})
    assert "I0 (ota5t)" in svg              # the block, named
    assert 'stroke-dasharray="1,1"' not in svg   # not a placeholder


# ---------------------------------------------------------------- blocks

def _fc_cin():
    import json
    return json.loads((FIX / "spectre" / "fc" / "tb_fc.cin.json")
                      .read_text(encoding="utf-8"))


def test_pin_roles_come_from_the_netlist_not_the_names():
    """Inside the OTA the inputs reach only gates, the output only
    drains, the bias a diode; the parent agrees (balun, load, isource).
    Names play no part here."""
    from circuitinsight.schematic.blocks import pin_roles

    roles = pin_roles(_fc_cin(), "ota_folded_cascode")
    assert {p: r for p, (r, _) in roles.items()} == {
        "vin_p": "input", "vin_n": "input", "vout": "output",
        "ib": "bias"}
    # the feedback switch on vout must not drag the balun onto it
    assert not any("CONFLICT" in e for ev in roles.values() for e in ev[1])
    assert any("loaded by CL" in e for e in roles["vout"][1])
    assert any("isource" in e for e in roles["ib"][1])


def test_proposal_maps_pins_and_stubs_the_rest():
    from circuitinsight.schematic.blocks import propose_all

    props = propose_all(_fc_cin(), {"gm.svg", "buf.svg"})
    bs = props["ota_folded_cascode"]
    assert bs.symbol == "gm.svg"
    assert bs.pins == {"vin_p": "in+", "vin_n": "in-", "vout": "out"}
    assert bs.stubs == {"ib": "auto"}
    assert bs.confirmed is False                 # never pre-confirmed
    assert any("name says +" in e for e in bs.evidence["vin_p"])
    # no fitting symbol in the library -> no proposal, no guess
    assert propose_all(_fc_cin(), {"res.svg"}) == {}


def test_sidecar_round_trip_and_confirmed_filter(tmp_path):
    from circuitinsight.schematic.blocks import (
        BlockSymbol, confirmed_only, load_block_symbols,
        save_block_symbols, sidecar_path)

    side = sidecar_path(tmp_path / "tb_x.cin.json")
    assert side.name == "tb_x.symbols.json"
    assert load_block_symbols(side) == {}
    m = {"a": BlockSymbol("gm.svg", {"p": "in+"}, {"b": "top"}, True,
                          {"p": ["why"]}),
         "b": BlockSymbol("buf.svg", {"i": "in"})}
    save_block_symbols(side, m)
    back = load_block_symbols(side)
    assert back["a"] == m["a"] and back["b"].confirmed is False
    assert list(confirmed_only(back)) == ["a"]


@needs_lib
def test_block_is_a_box_until_confirmed_then_a_symbol():
    from circuitinsight.schematic import SymbolLibrary, compose, page
    from circuitinsight.schematic.blocks import BlockSymbol

    h = load_hints(HINTS)
    tb = page(h, "tb_ota5t")
    devtypes = {"VIND": ("vsource", None), "VINC": ("vsource", None),
                "VSUP": ("vsource", None), "IB": ("isource", None),
                "MN2": ("mosfet", "n"), "I5": ("balun", None),
                "S0": ("switch", None), "S1": ("switch", None)}
    bs = BlockSymbol("gm.svg", {"inp": "in+", "inn": "in-", "out": "out"},
                     {"vbn": "auto"}, confirmed=False)
    lib = SymbolLibrary()
    boxed = compose(tb, devtypes, lib, subckts={"ota5t"},
                    block_symbols={"ota5t": bs})
    assert "I0 (ota5t)" in boxed             # still the honest box
    bs.confirmed = True
    drawn = compose(tb, devtypes, lib, subckts={"ota5t"},
                    block_symbols={"ota5t": bs})
    assert "I0 (ota5t)" not in drawn
    assert ">(ota5t)<" in drawn and ">I0<" in drawn
    assert ">vbn<" in drawn                  # the stub keeps its name


@needs_lib
def test_t_junction_gets_a_dot_without_a_wire_split():
    """In the fixture the vbn wire from MN2's gate ends on the INTERIOR
    of the vbn bus (no split, two ends only): the source draws a dot
    there and so must the composer."""
    from circuitinsight.schematic import SymbolLibrary, compose, page
    from circuitinsight.schematic.compose import compose as _c

    h = load_hints(HINTS)
    tb = page(h, "tb_ota5t")
    devtypes = {"VIND": ("vsource", None), "VINC": ("vsource", None),
                "VSUP": ("vsource", None), "IB": ("isource", None),
                "MN2": ("mosfet", "n"), "I5": ("balun", None),
                "S0": ("switch", None), "S1": ("switch", None)}
    compose(tb, devtypes, SymbolLibrary(), subckts={"ota5t"})
    x, y = _c.last_mm((1.1, 2.5))
    assert (round(x * 2) / 2, round(y * 2) / 2) in _c.last_dots
