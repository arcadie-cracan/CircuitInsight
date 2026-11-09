"""Author the psrldo CIN by hand, matching tb_psrldo.scs exactly.

SKY130 re-design of LDO1 of the ROMJIST 2022 high-PSR capacitor-less LDO
paper (Sularea, Raducan, Neag), device for device the paper's Fig. 5. One
CIN serves every (IL, CL) operating point of one CL value: the load current
lives in the OP data, the load capacitance is a CIN parameter, so a CIN is
written per CL value. A second family, with the PSR enhancer's Cin removed
(cin = 0, which Spectre prunes), serves the "without enhancer" runs.

    python make_cin.py <out dir>   -> tb_psrldo_cl<CL>.cin.json for CL in
                                      10p, 100p, 1n; tb_psrldo_cl100p_noenh;
                                      tb_psrldo_cl10p_v1 (the first sizing)
"""
import json
import sys
from pathlib import Path

P = dict(r34="80k", r1="20k", c1="2p", c2="30p", c3="3p", rf1="300k",
         rf2="450k", rin="60k", cin="1p", wpass="5u", mpass="800")
# the first sizing (R3,4 = 200k, C3 = 2 pF, Rin = 30k): the paper's PSR at
# 1 MHz, and four unstable points; psf_10u_10p_v1 keeps one of them for
# the stability certificate
P_V1 = dict(P, r34="200k", c3="2p", rin="30k")


def mos(name, d, g, s, b, cell, w, l, m="1"):
    pol = "n" if cell.startswith("nfet") else "p"
    return {"name": name, "device_type": "mosfet",
            "terminals": {"d": d, "g": g, "s": s, "b": b},
            "params": {"polarity": pol, "m": m},
            "meta": {"lib": "sky130_fd_pr", "cell": cell, "w": w, "l": l,
                     "nf": "1"}}


def two(name, dtype, p, n, cell, params=None):
    e = {"name": name, "device_type": dtype,
         "terminals": {"p": p, "n": n},
         "meta": {"lib": "analogLib", "cell": cell}}
    if params:
        e["params"] = params
    return e


def iprobe(name, p, n):
    e = two(name, "vsource", p, n, "iprobe", {})
    e["meta"]["note"] = ("Tian probe: 0 V series branch marking the main "
                         "loop between OUT and the top of the divider; "
                         "joins the stb run's iprobe OP record")
    return e


def instances(cl, enhancer=True, P=P):
    NL, N, PL, PP = "nfet_01v8_lvt", "nfet_01v8", "pfet_01v8_lvt", "pfet_01v8"
    vin = two("VIN", "vsource", "vin", "0", "vdc")
    vin["params"] = {"acm": "1"}
    vin["meta"]["note"] = "the supply: AC magnitude 1 for the PSR run"
    out = [
        vin,
        two("VREF", "vsource", "ref", "0", "vdc"),
        # the PMOS bias voltages are drops below the supply
        two("VBP", "vsource", "vin", "vbp", "vdc"),
        two("VCP", "vsource", "vin", "vcp", "vdc"),
        two("VCN", "vsource", "vcn", "0", "vdc"),
        # input pair on the cascoded tail
        mos("M17", "s18", "vbp", "vin", "vin", PL, "10u", "1u", m="2"),
        mos("M18", "tail", "vcp", "s18", "vin", PL, "10u", "0.5u", m="2"),
        mos("M1", "a", "ref", "tail", "vin", PL, "10u", "0.5u", m="2"),
        mos("M2", "b", "fb", "tail", "vin", PL, "10u", "0.5u", m="2"),
        # PSR enhancer
        two("IPTAT", "isource", "vin", "tailp", "idc"),
        two("Rin", "resistor", "ref", "hp", "res", {"r": P["rin"]}),
    ]
    if enhancer:
        out.append(two("Cin", "capacitor", "vin", "hp", "cap",
                       {"c": P["cin"]}))
    out += [
        mos("M3", "a", "ref", "tailp", "vin", PL, "10u", "1u"),
        mos("M4", "b", "hp", "tailp", "vin", PL, "10u", "1u"),
        # first-stage loads with the LCMFB resistors
        mos("M5", "a", "vcn", "s5", "0", NL, "10u", "0.5u"),
        mos("M7", "s5", "c", "0", "0", N, "10u", "1u"),
        two("R4", "resistor", "a", "c", "res", {"r": P["r34"]}),
        mos("M6", "b", "vcn", "s6", "0", NL, "10u", "0.5u"),
        mos("M8", "s6", "c", "0", "0", N, "10u", "1u"),
        two("R3", "resistor", "b", "c", "res", {"r": P["r34"]}),
        # second stage
        mos("M10", "s9", "a", "0", "0", N, "10u", "1u", m="2"),
        mos("M9", "bp", "vcn", "s9", "0", NL, "10u", "0.5u"),
        mos("M11", "bp", "vcp", "s11", "vin", PL, "10u", "0.5u", m="2"),
        mos("M12", "s11", "bp", "vin", "vin", PL, "10u", "1u", m="2"),
        mos("M13", "s14", "bp", "vin", "vin", PL, "10u", "1u", m="2"),
        mos("M14", "vb", "vcp", "s14", "vin", PL, "10u", "0.5u", m="2"),
        mos("M16", "va1", "b", "0", "0", N, "10u", "1u", m="2"),
        mos("M15", "vb", "vcn", "va1", "0", NL, "10u", "0.5u"),
        # R1-C1 from VB to the supply
        two("C1", "capacitor", "vin", "p1", "cap", {"c": P["c1"]}),
        two("R1", "resistor", "p1", "vb", "res", {"r": P["r1"]}),
        # voltage buffer
        mos("M19", "vin", "vb", "s19", "0", NL, "10u", "0.5u", m="3"),
        mos("M23", "s19", "b", "0", "0", N, "10u", "1u", m="2"),
        mos("M22", "va2", "bp", "vin", "vin", PL, "10u", "1u", m="5"),
        mos("M21", "gate", "vcp", "va2", "vin", PL, "10u", "0.5u", m="5"),
        mos("M20", "0", "s19", "gate", "vin", PL, "10u", "0.5u"),
        # pass device, local loops, load, divider, probe
        mos("MP", "out", "gate", "vin", "vin", PP, P["wpass"], "0.15u",
            m=P["mpass"]),
        two("C2", "capacitor", "out", "va1", "cap", {"c": P["c2"]}),
        two("C3", "capacitor", "out", "va2", "cap", {"c": P["c3"]}),
        two("CL", "capacitor", "out", "0", "cap", {"c": cl}),
        two("IL", "isource", "out", "0", "idc"),
        iprobe("IPRB0", "out", "fb_x"),
        two("Rf1", "resistor", "fb_x", "fb", "res", {"r": P["rf1"]}),
        two("Rf2", "resistor", "fb", "0", "res", {"r": P["rf2"]}),
    ]
    return out


def build(cl, enhancer=True, P=P, suffix=""):
    name = f"tb_psrldo_cl{cl}" + ("" if enhancer else "_noenh") + suffix
    return {
        "cin_version": "0.1",
        "design": {"name": name,
                   "source": {"kind": "hand", "note": (
                       "SKY130 re-design of LDO1 of Sularea, Raducan, Neag, "
                       "ROMJIST 25(2) 2022, Fig. 5; matches tb_psrldo.scs"
                       + ("" if enhancer else "; PSR enhancer's Cin removed")
                       + ("; first sizing, R3,4 = 200k, C3 = 2 pF, Rin = 30k"
                          if suffix else ""))}},
        "top": "tb_psrldo",
        "ground": ["0"],
        "globals": [],
        "definitions": {"tb_psrldo": {"ports": [],
                                      "instances": instances(cl, enhancer, P)}},
    }


if __name__ == "__main__":
    out = Path(sys.argv[1])
    docs = [build(cl, enh) for cl, enh in (("10p", True), ("100p", True),
                                           ("1n", True), ("100p", False))]
    docs.append(build("10p", True, P_V1, "_v1"))
    for doc in docs:
        p = out / (doc["design"]["name"] + ".cin.json")
        p.write_text(json.dumps(doc, indent=1))
        print("wrote", p)
