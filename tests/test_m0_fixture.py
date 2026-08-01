"""End-to-end M0 fixture checks.

The CIN produced by the Virtuoso SKILL exporter (circuitinsight/skill/cin_export.il) must
resolve against the instance names in the Spectre dcOpInfo psfascii dump.
Both files are checked-in fixtures captured on <sim-host> (SKY130, 2026-07-18).

On SKY130 the join is not name-identical: nfet_01v8/pfet_01v8 are sky130_fd_pr
subckt wrappers, so a MOSFET's dcOpInfo key is "<inst>.<primitive-model>"
(M1.msky130_fd_pr__nfet_01v8) while passives/sources keep their exact name.
"""
import re
from pathlib import Path

from circuitinsight.adapters.cin import flatten, load_cin

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "spectre" / "m0"


def dcopinfo_instance_names(path: Path) -> set[str]:
    text = path.read_text()
    value_section = text[text.index("\nVALUE\n"):]
    return set(re.findall(r'^"([^"]+)" "[^"]+" \($', value_section, re.M))


def test_exporter_cin_loads_and_flattens():
    flat = flatten(load_cin(FIXTURES / "tb_m0.cin.json"))
    assert len(flat.devices) == 10
    by_name = {d.name: d for d in flat.devices}
    assert by_name["M1"].params["polarity"] == "n"
    assert by_name["M2"].params["polarity"] == "p"
    assert by_name["M1"].terminals == {"d": "vout", "g": "vin", "s": "gnd!", "b": "gnd!"}


def test_exporter_sim_names_resolve_against_dcopinfo():
    """Every CIN sim name must resolve to exactly one OP record: passives and
    sources by an exact match, MOSFETs by an "<inst>.<primitive>" prefix (the
    SKY130 subckt-wrapper form the join must handle)."""
    flat = flatten(load_cin(FIXTURES / "tb_m0.cin.json"))
    op_names = dcopinfo_instance_names(FIXTURES / "psf" / "dcOpInfo.info")
    for d in flat.devices:
        matches = [n for n in op_names
                   if n == d.sim_name or n.startswith(d.sim_name + ".")]
        assert len(matches) == 1, f"{d.sim_name}: {matches}"
    # exactly the two MOSFETs gained a wrapper suffix; everything else is exact
    assert {n for n in op_names if "." in n} == {
        "M1.msky130_fd_pr__nfet_01v8", "M2.msky130_fd_pr__pfet_01v8"}


def test_hierarchical_result_names_use_dot_separator():
    names = dcopinfo_instance_names(FIXTURES / "psf_hier" / "dcOpInfo.info")
    assert "I0.RL" in names                              # passive: exact
    assert any(n.startswith("I0.M1.") for n in names)    # MOSFET: I0.M1.<prim>


def test_dcopinfo_has_expected_mosfet_params():
    text = (FIXTURES / "psf" / "dcOpInfo.info").read_text()
    type_section = text[text.index("\nTYPE\n"): text.index("\nVALUE\n")]
    bsim4 = type_section[type_section.index('"bsim4" STRUCT('):]
    fields = set(re.findall(r'^"(\w+)" (?:FLOAT|INT)', bsim4, re.M))
    required = {"gm", "gds", "gmbs", "cgs", "cgd", "cgb", "cjd", "cjs", "region"}
    assert required <= fields, f"missing: {required - fields}"
