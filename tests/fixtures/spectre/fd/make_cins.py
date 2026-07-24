"""Generate the fdota CINs (hand-authored topology matching the decks)."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

SKY = {"lib": "sky130_fd_pr", "w": 'iPar("totalW")', "nf": "1"}


def mos(name, d, g, s, b, pol, m, l):
    cell = ("nfet_01v8" if pol == "n" else "pfet_01v8")
    return {"name": name, "device_type": "mosfet",
            "terminals": {"d": d, "g": g, "s": s, "b": b},
            "params": {"polarity": pol, "m": str(m)},
            "meta": {**SKY, "cell": cell, "l": l}}


def two(name, dtype, p, n, cell):
    return {"name": name, "device_type": dtype,
            "terminals": {"p": p, "n": n},
            "meta": {"lib": "analogLib", "cell": cell}}


def iprobe(name, p, n):
    e = two(name, "vsource", p, n, "iprobe")
    e["params"] = {}
    e["meta"]["note"] = ("Tian probe: 0 V series branch, joins the stb "
                        "run's iprobe OP record")
    return e


def balun(name, d, c, p, n):
    return {"name": name, "device_type": "balun",
            "terminals": {"d": d, "c": c, "p": p, "n": n},
            "meta": {"lib": "analogLib", "cell": "ideal_balun"}}


def fdota2s(cmfb_probe=False):
    """The OTA subckt; cmfb_probe adds IPRB_CMFB in the vcmfb mirror wire."""
    load_gate = "vcmfbg" if cmfb_probe else "vcmfb"
    ins = [
        mos("MN2", "tail", "vbn", "vss", "vss", "n", 2, "1u"),
        mos("MN1P", "x1p", "inp", "tail", "vss", "n", 1, "2u"),
        mos("MN1N", "x1n", "inn", "tail", "vss", "n", 1, "2u"),
        mos("MPLP", "x1p", load_gate, "vdd", "vdd", "p", 2, "2u"),
        mos("MPLN", "x1n", load_gate, "vdd", "vdd", "p", 2, "2u"),
        mos("MP2P", "outp", "x1p", "vdd", "vdd", "p", 4, "1u"),
        mos("MN3P", "outp", "vbn", "vss", "vss", "n", 4, "1u"),
        mos("MP2N", "outn", "x1n", "vdd", "vdd", "p", 4, "1u"),
        mos("MN3N", "outn", "vbn", "vss", "vss", "n", 4, "1u"),
        two("CCP", "capacitor", "netp", "outp", "cap"),
        two("RZP", "resistor", "x1p", "netp", "res"),
        two("CCN", "capacitor", "netn", "outn", "cap"),
        two("RZN", "resistor", "x1n", "netn", "res"),
        mos("MNT", "ctail", "vbn", "vss", "vss", "n", 1, "1u"),
        mos("MNS", "vcmfb", "vcmsense", "ctail", "vss", "n", 1, "2u"),
        mos("MNR", "dumm", "vcmref", "ctail", "vss", "n", 1, "2u"),
        mos("MPA", "vcmfb", "vcmfb", "vdd", "vdd", "p", 1, "2u"),
        mos("MPB", "dumm", "dumm", "vdd", "vdd", "p", 1, "2u"),
    ]
    if cmfb_probe:
        ins.insert(13, iprobe("IPRB_CMFB", "vcmfb", "vcmfbg"))
    ports = ["outp", "outn", "vdd", "inp", "inn", "vss", "vbn",
             "vcmsense", "vcmref"]
    return {"ports": ports, "instances": ins}


FD_PROBE = {"ports": ["in1", "in2", "out1", "out2"], "instances": [
    balun("BI", "dmi", "cmi", "in1", "in2"),
    balun("BO", "dmo", "cmo", "out1", "out2"),
    iprobe("IPRB_DM", "dmi", "dmo"),
    iprobe("IPRB_CM", "cmi", "cmo"),
]}


def bench():
    vind = two("VIND", "vsource", "vin_dm", "gnd!", "vdc")
    vind["params"] = {"acm": "1"}
    return {"ports": [], "instances": [
        two("VSUP", "vsource", "vdd!", "gnd!", "vdc"),
        vind,
        two("VINC", "vsource", "vin_cm", "gnd!", "vdc"),
        two("VCMR", "vsource", "vcmref", "gnd!", "vdc"),
        balun("I5", "vin_dm", "vin_cm", "srcp", "srcn"),
        two("R1A", "resistor", "srcp", "inn", "res"),
        two("R1B", "resistor", "srcn", "inp", "res"),
        two("R2A", "resistor", "voutp", "inn", "res"),
        two("R2B", "resistor", "voutn", "inp", "res"),
        {"name": "I0", "subckt": "fdota2s",
         "terminals": {"outp": "outpi", "outn": "outni", "vdd": "vdd!",
                       "inp": "inp", "inn": "inn", "vss": "gnd!",
                       "vbn": "vbn", "vcmsense": "vcm_sense",
                       "vcmref": "vcmref"}},
        {"name": "FDPRB", "subckt": "fd_probe",
         "terminals": {"in1": "outpi", "in2": "outni",
                       "out1": "voutp", "out2": "voutn"}},
        two("CLP", "capacitor", "voutp", "gnd!", "cap"),
        two("CLN", "capacitor", "voutn", "gnd!", "cap"),
        two("RCMA", "resistor", "voutp", "vcm_sense", "res"),
        two("RCMB", "resistor", "voutn", "vcm_sense", "res"),
        two("CCMA", "capacitor", "voutp", "vcm_sense", "cap"),
        two("CCMB", "capacitor", "voutn", "vcm_sense", "cap"),
        {"name": "IB", "device_type": "isource",
         "terminals": {"p": "vdd!", "n": "vbn"},
         "meta": {"lib": "analogLib", "cell": "idc"}},
        mos("MN2", "vbn", "vbn", "gnd!", "gnd!", "n", 2, "1u"),
    ]}


def doc(name, cmfb_probe):
    return {
        "cin_version": "0.1",
        "design": {"name": name,
                   "source": {"kind": "hand", "deck": name + ".scs"}},
        "top": "tb_fdota",
        "ground": ["0", "gnd!"],
        "globals": ["vdd!"],
        "definitions": {"tb_fdota": bench(),
                        "fdota2s": fdota2s(cmfb_probe),
                        "fd_probe": FD_PROBE},
    }


for name, probe in [("tb_fdota_stb", False), ("tb_fdota_stb_cmfb", True)]:
    p = HERE / f"{name}.cin.json"
    p.write_text(json.dumps(doc(name, probe), indent=1))
    print("wrote", p.name)
