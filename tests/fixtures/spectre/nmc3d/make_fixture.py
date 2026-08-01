"""Generate the nmc3d fixture: a behavioral fully-differential gm-C THREE-stage
nested-Miller amplifier -- the mirror of nmc3, with TWO nested Miller PAIRS
and a clean-room fd_probe (two ideal baluns + a DM iprobe). Emits BOTH the
Spectre deck (tb_nmc3d.scs) and the matching CIN (tb_nmc3d.cin.json) from one
element list, so they cannot drift. PDK-free -> ships public.

Its DM half-circuit is exactly nmc3 (gm/go/C values reused), so the DM loop
needs the same two nested Miller caps -- here per side, as mirrored pairs.
Differential unity feedback (voutp->inn, voutn->inp) gives negative DM
feedback; the DM loop gain is probed at FDPRB.IPRB_DM.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

# --- element list: (kind, name, nodes, params) shared by deck + CIN ----------
# kinds: gm (vccs), r (resistor), c (capacitor), iprobe (vsource), balun, vdc
GM1, GM2, GM3 = "100u", "300u", "3m"
R1, R2, R3 = "200k", "200k", "10k"          # 1/go1, 1/go2, 1/go3
CN1, CN2, CLD = "300f", "300f", "10p"
CM1, CM2 = "2p", "6p"

E = []


def side(s, other):
    inx = ("inp", "inn") if s == "p" else ("inn", "inp")
    E.append(("gm", f"G1{s}", (f"n1{s}", "0", inx[0], inx[1]), GM1))
    E.append(("r", f"R1{s}", (f"n1{s}", "0"), R1))
    E.append(("c", f"CN1{s}", (f"n1{s}", "0"), CN1))
    E.append(("gm", f"G2{s}", ("0", f"n2{s}", f"n1{s}", "0"), GM2))   # s2=-1
    E.append(("r", f"R2{s}", (f"n2{s}", "0"), R2))
    E.append(("c", f"CN2{s}", (f"n2{s}", "0"), CN2))
    E.append(("gm", f"G3{s}", (f"out{s}", "0", f"n2{s}", "0"), GM3))  # s3=+1
    E.append(("r", f"R3{s}", (f"out{s}", "0"), R3))
    E.append(("c", f"CL{s}", (f"out{s}", "0"), CLD))
    # inner Miller Cm1 (out<->n2) then outer Miller Cm2 (out<->n1), each iprobe
    E.append(("c", f"CM1{s}", (f"out{s}", f"n2i{s}"), CM1))
    E.append(("iprobe", f"IPRB1{s}", (f"n2i{s}", f"n2{s}"), None))
    E.append(("c", f"CM2{s}", (f"out{s}", f"n1i{s}"), CM2))
    E.append(("iprobe", f"IPRB2{s}", (f"n1i{s}", f"n1{s}"), None))


side("p", "n")
side("n", "p")
# fd_probe (instantiated flat here): two baluns + DM/CM iprobes
E.append(("balun", "BI", ("dmi", "cmi", "outp", "outn"), None))
E.append(("balun", "BO", ("dmo", "cmo", "voutp", "voutn"), None))
E.append(("iprobe", "IPRB_DM", ("dmi", "dmo"), None))
E.append(("iprobe", "IPRB_CM", ("cmi", "cmo"), None))
# differential unity feedback (0 V sources = wires) voutp->inn, voutn->inp
E.append(("iprobe", "FBP", ("voutp", "inn"), None))
E.append(("iprobe", "FBN", ("voutn", "inp"), None))


# --- Spectre deck ------------------------------------------------------------
def emit_scs():
    L = ["""// tb_nmc3d.scs -- behavioral fully-differential gm-C THREE-stage
// nested-Miller (NMC) amplifier: the mirror of nmc3, two nested Miller PAIRS,
// a clean-room fd_probe (two ideal baluns + a DM iprobe). PDK-free.
// DM half-circuit == nmc3; differential unity feedback gives negative DM
// feedback; the DM loop gain is probed at FDPRB.IPRB_DM. Neither Miller pair
// alone stabilizes the DM loop -- both nested pairs are needed.
//   spectre tb_nmc3d.scs -format psfascii -raw ./psf +log nmc3d.log
simulator lang=spectre
global 0

subckt ideal_balun d c p n
    K0 (d 0 p c) transformer n1=2
    K1 (d 0 c n) transformer n1=2
ends ideal_balun
"""]
    bal = {}
    for kind, name, nodes, val in E:
        if kind == "gm":
            L.append(f"{name} ({' '.join(nodes)}) vccs gm={val}")
        elif kind == "r":
            L.append(f"{name} ({' '.join(nodes)}) resistor r={val}")
        elif kind == "c":
            L.append(f"{name} ({' '.join(nodes)}) capacitor c={val}")
        elif kind == "iprobe":
            L.append(f"{name} ({' '.join(nodes)}) iprobe")
        elif kind == "balun":
            L.append(f"{name} ({' '.join(nodes)}) ideal_balun")
    L.append("""
simulatorOptions options psfversion="1.4.0" temp=27 tnom=27 scalem=1.0 \\
    scale=1.0 gmin=1e-12 rforce=1 maxnotes=5 maxwarns=5 digits=8 cols=80 \\
    pivrel=1e-3
stb stb start=1 stop=10G dec=40 probe=IPRB_DM annotate=status
dcOp dc write="spectre.dc" maxiters=150 maxsteps=10000 annotate=status
dcOpInfo info what=oppoint where=rawfile
saveOptions options save=allpub
""")
    return "\n".join(L)


# --- CIN ---------------------------------------------------------------------
def cin_inst(kind, name, nodes, val):
    if kind == "gm":
        return {"name": name, "device_type": "vccs",
                "terminals": {"p": nodes[0], "n": nodes[1],
                              "cp": nodes[2], "cn": nodes[3]},
                "params": {"gm": val}, "meta": {"lib": "analogLib", "cell": "vccs"}}
    if kind == "r":
        return {"name": name, "device_type": "resistor",
                "terminals": {"p": nodes[0], "n": nodes[1]},
                "params": {"r": val}, "meta": {"lib": "analogLib", "cell": "res"}}
    if kind == "c":
        return {"name": name, "device_type": "capacitor",
                "terminals": {"p": nodes[0], "n": nodes[1]},
                "params": {"c": val}, "meta": {"lib": "analogLib", "cell": "cap"}}
    if kind == "iprobe":
        return {"name": name, "device_type": "vsource",
                "terminals": {"p": nodes[0], "n": nodes[1]}, "params": {},
                "meta": {"lib": "analogLib", "cell": "iprobe"}}
    if kind == "balun":
        return {"name": name, "device_type": "balun",
                "terminals": {"d": nodes[0], "c": nodes[1],
                              "p": nodes[2], "n": nodes[3]},
                "meta": {"lib": "analogLib", "cell": "ideal_balun"}}
    raise ValueError(kind)


def emit_cin():
    return {"cin_version": "0.1",
            "design": {"name": "tb_nmc3d",
                       "source": {"kind": "hand", "note": "behavioral fd gm-C "
                                  "3-stage nested-Miller; matches tb_nmc3d.scs"}},
            "top": "tb_nmc3d", "ground": ["0"], "globals": [],
            "definitions": {"tb_nmc3d": {"ports": [],
                            "instances": [cin_inst(*e) for e in E]}}}


if __name__ == "__main__":
    (HERE / "tb_nmc3d.scs").write_text(emit_scs())
    (HERE / "tb_nmc3d.cin.json").write_text(json.dumps(emit_cin(), indent=1))
    print("wrote tb_nmc3d.scs + tb_nmc3d.cin.json;", len(E), "elements")
