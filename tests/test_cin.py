import json
from pathlib import Path

import pytest

from circuitinsight.adapters.cin import (
    CIN_VERSION,
    CinError,
    flatten,
    load_cin,
    parse_cin,
)

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def minimal_doc(**overrides):
    doc = {
        "cin_version": CIN_VERSION,
        "top": "main",
        "ground": ["0"],
        "definitions": {
            "main": {
                "ports": [],
                "instances": [
                    {
                        "name": "R1",
                        "device_type": "resistor",
                        "terminals": {"p": "a", "n": "0"},
                        "params": {"r": "1k"},
                    }
                ],
            }
        },
    }
    doc.update(overrides)
    return doc


# ---------------------------------------------------------------- loading

def test_load_all_examples():
    paths = sorted(EXAMPLES.glob("*.cin.json"))
    assert paths, "no example CIN files found"
    for p in paths:
        doc = load_cin(p)
        flat = flatten(doc)
        assert flat.devices


def test_load_rc_lowpass():
    doc = load_cin(EXAMPLES / "rc_lowpass.cin.json")
    assert doc.top == "main"
    assert doc.ground == ("0",)
    assert [i.name for i in doc.definitions["main"].instances] == ["V1", "R1", "C1"]


def test_flatten_flat_design():
    flat = flatten(load_cin(EXAMPLES / "rc_lowpass.cin.json"))
    by_name = {d.name: d for d in flat.devices}
    assert set(by_name) == {"V1", "R1", "C1"}
    assert by_name["R1"].sim_name == "R1"
    assert by_name["R1"].terminals == {"p": "vin", "n": "vout"}
    assert flat.nets == {"vin", "vout", "0"}


# ---------------------------------------------------------------- hierarchy

def test_flatten_hierarchical_names_and_nets():
    flat = flatten(load_cin(EXAMPLES / "hier_diffpair.cin.json"))
    by_name = {d.name: d for d in flat.devices}

    # instance path and sim_name compose with '.'
    assert "I0.M1" in by_name
    assert by_name["I0.M1"].sim_name == "I0.M1"

    m1 = by_name["I0.M1"]
    # port net maps to the parent binding
    assert m1.terminals["g"] == "vip"
    # ground is never prefixed
    assert m1.terminals["b"] == "0"
    # definition-internal net gets the instance-path prefix
    assert m1.terminals["s"] == "I0.s1"
    # port bound to a top-level net
    assert by_name["I0.RS1"].terminals == {"p": "I0.s1", "n": "ntail"}
    # global net is never prefixed
    assert by_name["I0.RL1"].terminals["p"] == "vdd!"

    assert "I0.s1" in flat.nets and "vdd!" in flat.nets and "ntail" in flat.nets


def test_sim_name_override_composes():
    raw = minimal_doc()
    raw["definitions"] = {
        "leafdef": {
            "ports": ["a"],
            "instances": [
                {
                    "name": "M1",
                    "sim_name": "M1_par1",
                    "device_type": "mosfet",
                    "terminals": {"d": "a", "g": "a", "s": "0", "b": "0"},
                    "params": {"polarity": "n"},
                }
            ],
        },
        "main": {
            "ports": [],
            "instances": [
                {"name": "X1", "subckt": "leafdef", "terminals": {"a": "n1"}}
            ],
        },
    }
    flat = flatten(parse_cin(raw))
    (dev,) = flat.devices
    assert dev.name == "X1.M1"
    assert dev.sim_name == "X1.M1_par1"


# ---------------------------------------------------------------- validation

def expect_error(raw, fragment):
    with pytest.raises(CinError) as exc:
        parse_cin(raw)
    assert fragment in str(exc.value)


def test_rejects_unknown_version():
    expect_error(minimal_doc(cin_version="9.9"), "unsupported cin_version")


def test_rejects_missing_top():
    expect_error(minimal_doc(top="nope"), "top definition 'nope' not found")


def test_rejects_bad_terminal_set():
    raw = minimal_doc()
    raw["definitions"]["main"]["instances"][0]["terminals"] = {"p": "a", "x": "0"}
    expect_error(raw, "requires terminals")


def test_rejects_unknown_device_type():
    raw = minimal_doc()
    raw["definitions"]["main"]["instances"][0]["device_type"] = "memristor"
    expect_error(raw, "unknown device_type")


def test_rejects_mosfet_without_polarity():
    raw = minimal_doc()
    raw["definitions"]["main"]["instances"][0] = {
        "name": "M1",
        "device_type": "mosfet",
        "terminals": {"d": "a", "g": "a", "s": "0", "b": "0"},
    }
    expect_error(raw, "params.polarity")


def test_rejects_device_and_subckt_together():
    raw = minimal_doc()
    raw["definitions"]["main"]["instances"][0]["subckt"] = "main"
    expect_error(raw, "exactly one of")


def test_rejects_unknown_subckt():
    raw = minimal_doc()
    raw["definitions"]["main"]["instances"].append(
        {"name": "X1", "subckt": "ghost", "terminals": {}}
    )
    expect_error(raw, "unknown subckt 'ghost'")


def test_rejects_port_terminal_mismatch():
    raw = minimal_doc()
    raw["definitions"]["sub"] = {
        "ports": ["a", "b"],
        "instances": [
            {
                "name": "R1",
                "device_type": "resistor",
                "terminals": {"p": "a", "n": "b"},
                "params": {"r": "1"},
            }
        ],
    }
    raw["definitions"]["main"]["instances"].append(
        {"name": "X1", "subckt": "sub", "terminals": {"a": "n1"}}
    )
    expect_error(raw, "has ports")


def test_rejects_duplicate_instance_names():
    raw = minimal_doc()
    inst = raw["definitions"]["main"]["instances"][0]
    raw["definitions"]["main"]["instances"].append(dict(inst))
    expect_error(raw, "duplicate instance name")


def test_rejects_dot_in_instance_name():
    raw = minimal_doc()
    raw["definitions"]["main"]["instances"][0]["name"] = "a.b"
    expect_error(raw, "must not contain")


def test_rejects_recursive_subckt():
    raw = minimal_doc()
    raw["definitions"]["a"] = {
        "ports": [],
        "instances": [{"name": "X1", "subckt": "b", "terminals": {}}],
    }
    raw["definitions"]["b"] = {
        "ports": [],
        "instances": [{"name": "X1", "subckt": "a", "terminals": {}}],
    }
    expect_error(raw, "recursive subckt")


def test_error_lists_all_problems():
    raw = minimal_doc(top="nope")
    raw["definitions"]["main"]["instances"][0]["device_type"] = "memristor"
    with pytest.raises(CinError) as exc:
        parse_cin(raw)
    assert len(exc.value.errors) >= 2


# ---------------------------------------------------------------- schema sync

def test_examples_conform_to_json_schema():
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(
        (Path(__file__).resolve().parents[1] / "schema" / "cin-0.1.schema.json").read_text()
    )
    for p in sorted(EXAMPLES.glob("*.cin.json")):
        jsonschema.validate(json.loads(p.read_text()), schema)
