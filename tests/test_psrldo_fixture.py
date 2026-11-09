"""psrldo: a SKY130 re-design of LDO1 of Sularea, Raducan, Neag, "High
Power Supply Rejection Capacitor-less Low Dropout Regulators" (ROMJIST
25(2) 2022), captured at fourteen (IL, CL) operating points with the stb
sweep at the main-loop probe and a supply-driven ac sweep for the PSR,
plus two points without the PSR enhancer and one point of the first
sizing that the simulator and the certificate both read as unstable.

The circuit is the paper's Fig. 5 device for device: PMOS input pair on a
cascoded tail, cascoded first-stage loads with the LCMFB resistors R3/R4,
the second-stage transconductors M10/M16 mirrored through them, a
cascoded PMOS diode mirrored into the OTA output VB, R1-C1 from VB to the
supply, the NMOS/PMOS follower pair driving the gate, C2 and C3 from OUT
into the two cascode sources, the PSR enhancer's PMOS pair on the Rin-Cin
high-pass, a 4 mm pfet_01v8 pass device, Vout = 1.5 V from 1.8 V, 21.5 uA
quiescent.

Ground truth captured on <sim-host> (Spectre 25.1, sky130 tt): see the
numbers asserted below, read off stb.margin.stb and ac.ac of each run.
"""
import warnings
from pathlib import Path

import numpy as np
import pytest

from circuitinsight.adapters.spectre import SpectreRun
from circuitinsight.analysis.marginledger import margin_ledger

FIX = Path(__file__).resolve().parent / "fixtures" / "spectre" / "psrldo"


def _run(point):
    cl = point.split("_")[1]
    tag = ("_noenh" if point.endswith("_noenh")
           else "_v1" if point.endswith("_v1") else "")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SpectreRun(FIX / f"tb_psrldo_cl{cl}{tag}.cin.json",
                          FIX / f"psf_{point}")


def _db(x):
    return 20 * np.log10(np.abs(x))


def test_full_load_loop_gain_psr_and_certificate():
    """At 50 mA and 100 pF the Tian loop gain reproduces stb wherever |T|
    is above -20 dB, the supply-to-output transfer reproduces the ac
    sweep over the whole band, both with the charge-matrix capacitances,
    and the certificate agrees with the closed-loop poles of the PSR
    transfer: stable."""
    run = _run("50m_100p")
    an = run.analyzer(cap_model="matrix")
    stb = run.stb()
    T = an.loop_gain("IPRB0", [])
    f = np.asarray(stb.freq)
    h = T.numeric_hp(f)
    Ts = np.asarray(stb.loop_gain)
    band = np.abs(Ts) > 0.1
    assert band.sum() > 40
    assert np.max(np.abs(_db(h[band]) - _db(Ts[band]))) < 0.01
    assert np.max(np.abs(np.angle(h[band], deg=True)
                         - np.angle(Ts[band], deg=True))) < 0.1
    ac = run.ac()
    H = an.tf("VIN", "out", [])
    fa = np.asarray(ac.freq)
    psr = H.numeric_hp(fa)
    assert np.max(np.abs(_db(psr) - _db(np.asarray(ac.waves["out"])))) < 0.01
    led = margin_ledger(T)
    assert led.pm_deg == pytest.approx(stb.phase_margin_deg, abs=0.3)
    st = led.stability
    assert st["verdict"] == "stable"
    assert st["closed_loop_rhp"] == 0
    assert not [p for p in H.poles() if p.real > 0]


def test_the_first_sizing_is_unstable_and_the_certificate_says_so():
    """At 10 uA and 10 pF the first sizing (R3,4 = 200k) has a closed-loop
    pole pair in the right half-plane near 4.6 MHz: Spectre's stb reports
    no margins ("positive feedback system"), the PSR transfer's poles show
    the pair, and the certificate counts it once, not twice -- the raw
    Tian rational carries the closed-loop determinant in both numerator
    and denominator, so D - N of the raw form roots every closed-loop pole
    twice, which is why the certificate roots the minimal form."""
    run = _run("10u_10p_v1")
    an = run.analyzer(cap_model="matrix")
    T = an.loop_gain("IPRB0", [])
    H = an.tf("VIN", "out", [])
    rhp = [p for p in H.poles() if p.real > 0]
    assert len(rhp) == 2
    assert abs(rhp[0]) == pytest.approx(4.57e6, rel=0.02)
    st = margin_ledger(T).stability
    assert st["verdict"] == "unstable"
    assert st["closed_loop_rhp"] == 2
    assert [abs(x) for x in st["closed_loop_rhp_hz"]] == pytest.approx(
        [abs(p) for p in rhp], rel=1e-3)
    # one unity crossing, and the phase crosses the critical line once
    # below it: the phase margin would mislead, which is the reason line
    assert st["unity_crossings"] == 1 and st["parity"] == "odd"
    assert "misleads" in st["reason"]
