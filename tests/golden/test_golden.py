"""Golden circuits: engine output vs textbook closed forms.

Small circuits are checked for EXACT symbolic equality against hand-derived
transfer functions. Larger ones (cascode, OTAs) are checked against an
independently hand-written KCL solve and/or textbook approximations with
element values chosen so the approximations are excellent (gm/gds = 1000).
"""
import math
from pathlib import Path

import pytest
import sympy as sp

from circuitinsight import Analyzer

CIRCUITS = Path(__file__).resolve().parent / "circuits"
s = sp.Symbol("s")


def load(name):
    return Analyzer.from_cin(CIRCUITS / f"{name}.cin.json")


# ------------------------------------------------------------------ CS amp

def test_cs_amp_exact_symbolic():
    H = load("cs_amp").tf("V1", "vout")
    y = H.symbols
    gm, gds, cgd, cdb = y["gm_M1"], y["gds_M1"], y["cgd_M1"], y["cdb_M1"]
    RL, CL = y["RL"], y["CL"]
    expected = (s * cgd - gm) / (gds + 1 / RL + s * (cgd + cdb + CL))
    assert sp.simplify(H.expr - expected) == 0
    # cgs never appears: input driven by an ideal source
    assert "cgs_M1" not in {str(x) for x in H.expr.free_symbols}


def test_cs_amp_numeric_dc_gain():
    H = load("cs_amp").tf("V1", "vout")
    gm, gds, RL = 469e-6, 8.5e-6, 1e4
    expected = -gm / (gds + 1 / RL)          # -gm * (ro || RL)
    assert complex(H.numeric([1.0])[0]).real == pytest.approx(expected, rel=1e-4)


# ------------------------------------------------------------ source follower

def test_source_follower_exact_symbolic():
    H = load("source_follower").tf("V1", "vout")
    y = H.symbols
    gm, gmbs, gds = y["gm_M1"], y["gmbs_M1"], y["gds_M1"]
    cgs, csb = y["cgs_M1"], y["csb_M1"]
    RS, CL = y["RS"], y["CL"]
    expected = (gm + s * cgs) / (gm + gmbs + gds + 1 / RS + s * (cgs + csb + CL))
    assert sp.simplify(H.expr - expected) == 0


# ------------------------------------------------------------------ cascode

def test_cascode_vs_hand_kcl():
    H = load("cascode").tf("V1", "vout")
    y = H.symbols
    gm1, gds1 = y["gm_M1"], y["gds_M1"]
    gm2, gmbs2, gds2 = y["gm_M2"], y["gmbs_M2"], y["gds_M2"]
    RL, CL = y["RL"], y["CL"]

    # independent hand-written KCL (device current i(d->s) = gm*vgs + gmbs*vbs
    # leaves the drain node and enters the source node)
    vin, vx, vout = sp.symbols("vin vx vout")
    i1 = gm1 * vin                            # M1: g=vin, s=0
    i2 = (gm2 + gmbs2) * (0 - vx)             # M2: g=0, b=0, s=vx
    eq_vx = sp.Eq(i1 + gds1 * vx + gds2 * (vx - vout) - i2, 0)
    eq_vout = sp.Eq(vout / RL + s * CL * vout + gds2 * (vout - vx) + i2, 0)
    sol = sp.solve([eq_vx, eq_vout], [vx, vout], dict=True)[0]
    assert sp.simplify(H.expr - sol[vout] / vin) == 0


def test_cascode_textbook_dc_gain():
    H = load("cascode").tf("V1", "vout")
    # Rout(cascode) ~ 2.5 GOhm >> RL: |A0| ~ gm1*RL to well under 0.1%
    a0 = complex(H.numeric([0.01])[0])
    assert a0.real == pytest.approx(-400e-6 * 1e4, rel=1e-3)


# ------------------------------------------------------------------- 5T OTA

def test_ota5t_textbook():
    an = load("ota5t")
    an.match("M1", "M2")
    an.match("M3", "M4")
    H = an.tf("V1", "vout")

    # matching merged the symbols
    names = {str(x) for x in H.expr.free_symbols}
    assert "gm_M2" not in names and "gm_M4" not in names

    gm, gds, CL = 400e-6, 0.4e-6, 1e-12
    a0 = complex(H.numeric([0.01])[0])
    assert abs(a0) == pytest.approx(gm / (2 * gds), rel=0.02)      # ~500

    (p1,) = H.poles()
    f_p1 = 2 * gds / (2 * math.pi * CL)                            # ~127 kHz
    assert abs(p1) == pytest.approx(f_p1, rel=0.02)
    assert p1.real < 0


# ------------------------------------------------------------ two-stage Miller

def test_miller_ota_textbook():
    an = load("miller_ota")
    an.match("M1", "M2")
    an.match("M3", "M4")
    H = an.tf("V1", "vout", keep=["CC", "CL"])   # hybrid: fast and exact

    gm1, gds12 = 400e-6, 0.4e-6
    gm5, gds56 = 2e-3, 2e-6
    Cc = 1e-12
    R1, R2 = 1 / (2 * gds12), 1 / (2 * gds56)

    a0 = complex(H.numeric([1e-3])[0])
    assert abs(a0) == pytest.approx(gm1 * R1 * gm5 * R2, rel=0.02)     # ~250k

    poles = H.poles()
    f_p1 = 1 / (2 * math.pi * R1 * Cc * gm5 * R2)                      # ~255 Hz
    assert abs(poles[0]) == pytest.approx(f_p1, rel=0.03)
    assert poles[0].real < 0

    # RHP zero at gm5/Cc
    zs = [z for z in H.zeros() if z.real > 0]
    assert len(zs) == 1
    assert abs(zs[0]) == pytest.approx(gm5 / (2 * math.pi * Cc), rel=0.03)

    # GBW = A0 * p1 ~ gm1/Cc
    gbw = abs(a0) * abs(poles[0])
    assert gbw == pytest.approx(gm1 / (2 * math.pi * Cc), rel=0.05)
