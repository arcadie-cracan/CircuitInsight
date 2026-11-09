"""Author the fastldo CIN by hand, matching tb_fastldo.scs exactly.

SKY130 re-design of the IEEE Access 2022 fast LDO (Grajdeanu et al.),
complementary image of its Fig. 5 without the class-AB input halves. One
CIN serves every operating point: the load current and capacitance live in
the OP data (IL) and the CIN parameter (CL) respectively, so a CIN is written
per CL value, with the dcOpInfo of each (IL, CL) run in its own psf dir.

    python make_cin.py <out dir>   -> tb_fastldo_cl<CL>.cin.json for CL in
                                   1p, 10n, 100n, 1u
"""
import json
import sys
from pathlib import Path

R = "10k"
RCM = "300k"
C1 = C2 = "40p"


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
                         "loop between OUT and the FB input; joins the stb "
                         "run's iprobe OP record")
    return e


def instances(cl):
    NL, P, PL = "nfet_01v8_lvt", "pfet_01v8", "pfet_01v8_lvt"
    return [
        two("VIN", "vsource", "vin", "0", "vdc"),
        two("VREF", "vsource", "ref", "0", "vdc"),
        two("VBN", "vsource", "vbn", "0", "vdc"),
        two("VCP", "vsource", "vcp", "0", "vdc"),
        two("VCN", "vsource", "vcn", "0", "vdc"),
        # input pair and tail
        mos("M1b", "a", "ref", "tail", "0", NL, "10u", "0.5u"),
        mos("M1a", "b", "fb_x", "tail", "0", NL, "10u", "0.5u"),
        mos("MT", "tail", "vbn", "0", "0", NL, "5u", "1u"),
        # PMOS degenerated mirrors, diode side, cascoded into A and B
        two("Rp1b", "resistor", "vin", "p1b", "res", {"r": R}),
        mos("M3xPb", "d3b", "c", "p1b", "vin", PL, "10u", "1u"),
        mos("M12", "a", "vcp", "d3b", "vin", PL, "10u", "0.5u"),
        two("Rp1a", "resistor", "vin", "p1a", "res", {"r": R}),
        mos("M3xPa", "d3a", "c", "p1a", "vin", PL, "10u", "1u"),
        mos("M11", "b", "vcp", "d3a", "vin", PL, "10u", "0.5u"),
        # LCMFB: a virtual ground at c for a differential signal, so A
        # and B see RCM and the error amplifier's Gm is 2 gm1 gm4 RCM
        two("Rcmb", "resistor", "a", "c", "res", {"r": RCM}),
        two("Rcma", "resistor", "b", "c", "res", {"r": RCM}),
        # mirror outputs, k = 2, degenerated
        two("Rp2b", "resistor", "vin", "p1", "res", {"r": R}),
        mos("M4yPb", "iom", "a", "p1", "vin", PL, "10u", "1u", m="2"),
        two("Rp2a", "resistor", "vin", "p2", "res", {"r": R}),
        mos("M4yPa", "iop", "b", "p2", "vin", PL, "10u", "1u", m="2"),
        # C1 into the source of the y-device that drives GATE directly
        two("C1", "capacitor", "out", "p2", "cap", {"c": C1}),
        # PMOS cascodes carrying Io- and Io+ down
        mos("M5", "min", "vcp", "iom", "vin", PL, "10u", "0.5u"),
        mos("M6", "gate", "vcp", "iop", "vin", PL, "10u", "0.5u"),
        # degenerated NMOS cascoded mirror into GATE; C2 enters at n2
        mos("M7", "min", "vcn", "d9", "0", NL, "5u", "0.5u"),
        mos("M9xN", "d9", "min", "n1", "0", NL, "5u", "1u"),
        two("Rn1", "resistor", "n1", "0", "res", {"r": R}),
        mos("M8", "gate", "vcn", "d10", "0", NL, "5u", "0.5u"),
        mos("M10yN", "d10", "min", "n2", "0", NL, "5u", "1u"),
        two("Rn2", "resistor", "n2", "0", "res", {"r": R}),
        two("C2", "capacitor", "out", "n2", "cap", {"c": C2}),
        # pass device, load, and the main-loop probe
        mos("MPASS", "out", "gate", "vin", "vin", P, "5u", "0.15u",
            m="14000"),
        two("CL", "capacitor", "out", "0", "cap", {"c": cl}),
        two("IL", "isource", "out", "0", "idc"),
        iprobe("IPRB0", "out", "fb_x"),
    ]


def build(cl):
    return {
        "cin_version": "0.1",
        "design": {"name": f"tb_fastldo_cl{cl}",
                   "source": {"kind": "hand", "note": (
                       "SKY130 re-design of the IEEE Access 2022 fast LDO "
                       "(Grajdeanu, Raducan, Plesa, Neag, Varzaru, Topa), "
                       "complementary image of its Fig. 5 without the "
                       "class-AB input halves; matches tb_fastldo.scs")}},
        "top": "tb_fastldo",
        "ground": ["0"],
        "globals": [],
        "definitions": {"tb_fastldo": {"ports": [], "instances": instances(cl)}},
    }


if __name__ == "__main__":
    out = Path(sys.argv[1])
    for cl in ("1p", "10n", "100n", "1u"):
        p = out / f"tb_fastldo_cl{cl}.cin.json"
        p.write_text(json.dumps(build(cl), indent=1))
        print("wrote", p)
