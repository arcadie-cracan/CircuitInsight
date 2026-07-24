"""Spectre adapter: psfascii parsing, canonical OP mapping, join, end-to-end TF.

All against the real SKY130 fixtures captured in M0. SKY130's nfet_01v8/
pfet_01v8 are sky130_fd_pr subckt wrappers, so a MOSFET's raw dcOpInfo key
carries the expanded primitive name; the join resolves it back to the CIN
instance, and post-join op_data is keyed by the plain instance name.
"""
import math
from pathlib import Path

import pytest

from circuitinsight.adapters.cin import flatten, parse_cin
from circuitinsight.adapters.spectre import SpectreError, SpectreRun, join_op
from circuitinsight.adapters.spectre.opdata import load_dcopinfo
from circuitinsight.adapters.spectre.psfascii import parse_psfascii

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "spectre" / "m0"
NFET = "M1.msky130_fd_pr__nfet_01v8"    # raw dcOpInfo keys (subckt-wrapper form)
PFET = "M2.msky130_fd_pr__pfet_01v8"


# ------------------------------------------------------------------ psfascii

def test_parse_dcopinfo_structs():
    psf = parse_psfascii(FIXTURES / "psf" / "dcOpInfo.info")
    assert len(psf.entries) == 10
    m1 = psf.entries[NFET]
    assert m1.type_name == "bsim4"
    assert m1.value["gm"] == pytest.approx(4.177149904678507e-04)
    assert m1.value["region"] == 2
    assert math.isnan(m1.value["ft"])   # unreported fields come through as nan
    assert m1.props["model"].startswith("M1.sky130_fd_pr__nfet_01v8__model")
    assert psf.entries["RLP"].value["res"] == pytest.approx(1e4)


def test_parse_dcop_dc_scalars():
    psf = parse_psfascii(FIXTURES / "psf" / "dcOp.dc")
    assert psf.entries["vout"].type_name == "V"
    assert psf.entries["vout"].value == pytest.approx(1.36885020245337)
    assert psf.entries["VDD:p"].type_name == "I"


# ------------------------------------------------------------ canonical map

def test_canonical_mosfet_caps():
    records = load_dcopinfo(FIXTURES / "psf" / "dcOpInfo.info")
    m1 = records[NFET]
    assert m1.device_type == "mosfet"
    assert m1.params["gm"] == pytest.approx(4.177149904678507e-04)
    # magnitudes of negative trans-caps
    assert m1.params["cgs"] == pytest.approx(abs(m1.raw["cgs"]))
    assert m1.raw["cgs"] < 0
    # cdb = junction + |intrinsic|
    assert m1.params["cdb"] == pytest.approx(abs(m1.raw["cjd"]) + abs(m1.raw["cdb"]))
    # sentinel ints and nan never become params
    assert "OPdef" not in m1.params
    r = records["RL"]
    assert r.device_type == "resistor" and r.params["r"] == pytest.approx(1e4)


# ------------------------------------------------------------------- join

def test_join_full_testbench():
    run = SpectreRun(FIXTURES / "tb_m0.cin.json", FIXTURES / "psf")
    assert set(run.op_data) == {"M1", "M2", "RD1", "RD2", "RL", "RLP",
                                "VDD", "VIN", "VINP", "C0"}
    assert run.op_data["C0"]["c"] == pytest.approx(1e-13)
    assert run.op_data["M2"]["gm"] == pytest.approx(3.200512e-04, rel=1e-6)


def test_join_missing_record_is_hard_error():
    flat = flatten(parse_cin({
        "cin_version": "0.1", "top": "main", "ground": ["0"],
        "definitions": {"main": {"ports": [], "instances": [
            {"name": "RX", "device_type": "resistor",
             "terminals": {"p": "a", "n": "0"}},
        ]}}}))
    with pytest.raises(SpectreError, match="RX: no OP record"):
        join_op(flat, load_dcopinfo(FIXTURES / "psf" / "dcOpInfo.info"))


def test_join_orphan_record_is_hard_error():
    run_records = load_dcopinfo(FIXTURES / "psf" / "dcOpInfo.info")
    flat = flatten(parse_cin({
        "cin_version": "0.1", "top": "main", "ground": ["0"],
        "definitions": {"main": {"ports": [], "instances": [
            {"name": "RL", "device_type": "resistor",
             "terminals": {"p": "a", "n": "0"}},
        ]}}}))
    with pytest.raises(SpectreError, match="matched no CIN device"):
        join_op(flat, run_records)


def test_join_type_mismatch_is_hard_error():
    # lie about a passive, whose OP record matches by exact name, so the direct
    # model-class check fires (SKY130 MOSFETs are wrappers -- covered below)
    records = load_dcopinfo(FIXTURES / "psf" / "dcOpInfo.info")
    flat = flatten(parse_cin({
        "cin_version": "0.1", "top": "main", "ground": ["0"],
        "definitions": {"main": {"ports": [], "instances": [
            {"name": "RL", "device_type": "capacitor",    # lies about RL
             "terminals": {"p": "a", "n": "0"}},
        ]}}}))
    with pytest.raises(SpectreError, match="model 'resistor'"):
        join_op(flat, records, allow_unmatched=tuple(n for n in records if n != "RL"))


def test_join_wrapper_type_mismatch_is_hard_error():
    # a MOSFET declared as the wrong type can't bind its subckt-wrapper record
    # (no same-type expansion candidate) -- a hard error unique to SKY130 wrappers
    records = load_dcopinfo(FIXTURES / "psf" / "dcOpInfo.info")
    flat = flatten(parse_cin({
        "cin_version": "0.1", "top": "main", "ground": ["0"],
        "definitions": {"main": {"ports": [], "instances": [
            {"name": "M1", "device_type": "resistor",     # lies about M1
             "terminals": {"p": "a", "n": "0"}},
        ]}}}))
    with pytest.raises(SpectreError, match="ambiguous wrapper expansion"):
        join_op(flat, records, allow_unmatched=tuple(records))


def test_join_hierarchical_names():
    # matches tb_m0_hier.scs: csstage(M1, RL) instantiated as I0
    flat = flatten(parse_cin({
        "cin_version": "0.1", "top": "tb", "ground": ["0"],
        "definitions": {
            "csstage": {"ports": ["out", "in", "vddl"], "instances": [
                {"name": "M1", "device_type": "mosfet",
                 "terminals": {"d": "out", "g": "in", "s": "0", "b": "0"},
                 "params": {"polarity": "n"}},
                {"name": "RL", "device_type": "resistor",
                 "terminals": {"p": "vddl", "n": "out"}},
            ]},
            "tb": {"ports": [], "instances": [
                {"name": "VDD", "device_type": "vsource",
                 "terminals": {"p": "vdd", "n": "0"}},
                {"name": "VIN", "device_type": "vsource",
                 "terminals": {"p": "vin", "n": "0"}},
                {"name": "I0", "subckt": "csstage",
                 "terminals": {"out": "vout", "in": "vin", "vddl": "vdd"}},
            ]},
        }}))
    op = join_op(flat, load_dcopinfo(FIXTURES / "psf_hier" / "dcOpInfo.info"))
    assert op["I0.M1"]["gm"] > 0
    assert op["I0.RL"]["r"] == pytest.approx(1e4)


# ------------------------------------------------------------- end to end

def test_end_to_end_nmos_stage_tf():
    run = SpectreRun(FIXTURES / "tb_m0.cin.json", FIXTURES / "psf")
    # hybrid keep=[]: exact-rational numeric solve; full symbolic on the whole
    # testbench (~15 symbols incl. 1/R fraction fields) is the documented
    # explosion case and takes minutes
    H = run.analyzer().tf("VIN", "vout", keep=[])

    p = run.op_data["M1"]
    gout = p["gds"] + 1 / run.op_data["RL"]["r"]
    cout = run.op_data["C0"]["c"] + p["cgd"] + p["cdb"]

    a0 = complex(H.numeric([0.01])[0])
    assert a0.real == pytest.approx(-p["gm"] / gout, rel=1e-9)

    (p1,) = sorted(H.poles(), key=abs)[:1]
    assert abs(p1) == pytest.approx(gout / (2 * math.pi * cout), rel=1e-6)

    # RHP zero from cgd feedforward
    (z1,) = H.zeros()
    assert z1.real == pytest.approx(p["gm"] / (2 * math.pi * p["cgd"]), rel=1e-6)


def test_end_to_end_hierarchical_exporter_cin():
    # exporter's hierarchical CIN (tb_m2 schematic: csstage wrapped as I7)
    # joined against the matching Spectre run — no rename needed at all
    run = SpectreRun(FIXTURES / "tb_hier.cin.json", FIXTURES / "psf_tbm2")
    assert "I7.M1" in run.op_data and "I7.RL" in run.op_data
    H = run.analyzer().tf("VIN", "net1", keep=[])
    p = run.op_data["I7.M1"]
    gout = p["gds"] + 1 / run.op_data["I7.RL"]["r"]
    a0 = complex(H.numeric([0.01])[0])
    assert a0.real == pytest.approx(-p["gm"] / gout, rel=1e-9)


def test_end_to_end_pmos_stage_tf():
    run = SpectreRun(FIXTURES / "tb_m0.cin.json", FIXTURES / "psf")
    H = run.analyzer().tf("VINP", "voutp", keep=[])
    p = run.op_data["M2"]
    gout = p["gds"] + 1 / run.op_data["RLP"]["r"]
    a0 = complex(H.numeric([0.01])[0])
    assert a0.real == pytest.approx(-p["gm"] / gout, rel=1e-9)
