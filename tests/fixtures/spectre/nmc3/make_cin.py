"""Author the nmc3 CIN (hand-authored, matching tb_nmc3.scs exactly).

Behavioral gm-C 3-stage nested-Miller amplifier: ideal transconductors
(device_type vccs, gm from CIN params -- a source type, not OP-extracted),
output conductances (resistor), node/load caps (capacitor), analogLib
iprobes (vsource) marking the three nested loops. PDK-free -> public.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def vccs(name, p, n, cp, cn, gm):
    return {"name": name, "device_type": "vccs",
            "terminals": {"p": p, "n": n, "cp": cp, "cn": cn},
            "params": {"gm": gm},
            "meta": {"lib": "analogLib", "cell": "vccs"}}


def two(name, dtype, p, n, cell, params=None):
    e = {"name": name, "device_type": dtype,
         "terminals": {"p": p, "n": n},
         "meta": {"lib": "analogLib", "cell": cell}}
    if params:
        e["params"] = params
    return e


def iprobe(name, p, n):
    e = two(name, "vsource", p, n, "iprobe", {})
    e["meta"]["note"] = ("Tian probe: 0 V series branch marking a loop; "
                         "joins the stb run's iprobe OP record")
    return e


INSTANCES = [
    two("VIN", "vsource", "inp", "0", "vdc", {"acm": "1"}),
    # stage 1
    vccs("G1", "n1", "0", "inp", "inn", "100u"),
    two("R1", "resistor", "n1", "0", "res", {"r": "200k"}),
    two("CN1", "capacitor", "n1", "0", "cap", {"c": "300f"}),
    # stage 2 (non-inverting into n2)
    vccs("G2", "0", "n2", "n1", "0", "300u"),
    two("R2", "resistor", "n2", "0", "res", {"r": "200k"}),
    two("CN2", "capacitor", "n2", "0", "cap", {"c": "300f"}),
    # stage 3 (inverting output)
    vccs("G3", "out", "0", "n2", "0", "3m"),
    two("R3", "resistor", "out", "0", "res", {"r": "10k"}),
    two("CL", "capacitor", "out", "0", "cap", {"c": "10p"}),
    # inner Miller Cm1 (out<->n2) through IPRB1
    two("CM1", "capacitor", "out", "n2i", "cap", {"c": "2p"}),
    iprobe("IPRB1", "n2i", "n2"),
    # outer Miller Cm2 (out<->n1) through IPRB2
    two("CM2", "capacitor", "out", "n1i", "cap", {"c": "6p"}),
    iprobe("IPRB2", "n1i", "n1"),
    # outer unity feedback out->inn through IPRB0
    iprobe("IPRB0", "out", "inn"),
]


def build():
    return {
        "cin_version": "0.1",
        "design": {"name": "tb_nmc3",
                   "source": {"kind": "hand", "note": "behavioral gm-C "
                              "3-stage nested-Miller amplifier; matches "
                              "tb_nmc3.scs"}},
        "top": "tb_nmc3",
        "ground": ["0"],
        "globals": [],
        "definitions": {"tb_nmc3": {"ports": [], "instances": INSTANCES}},
    }


if __name__ == "__main__":
    out = HERE / "tb_nmc3.cin.json"
    out.write_text(json.dumps(build(), indent=1))
    print("wrote", out)
