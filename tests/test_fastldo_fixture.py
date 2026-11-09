"""fastldo: a SKY130 re-design of the IEEE Access 2022 fast LDO (Grajdeanu,
Raducan, Plesa, Neag, Varzaru, Topa, "Fast LDO Handles a Wide Range of Load
Currents and Load Capacitors, up to 100 mA and Over 1 uF"), captured at
eight (IL, CL) operating points with the stb sweep at the main-loop probe.

The circuit is the complementary image of the paper's Fig. 5 without its
class-AB input halves: NMOS input pair, PMOS degenerated mirrors with the
LCMFB resistors and the cascodes at A and B, PMOS cascodes carrying Io+/Io-
down into a degenerated NMOS cascoded mirror that drives GATE, C1 and C2
from OUT into the two mirror source nodes, R = 10k, C1 = C2 = 40 pF, a
70 mm pfet_01v8 pass device, Vout = 1 V from Vin = 1.2 V, 3.3 uA quiescent.

Ground truth captured on <sim-host> (Spectre 25.1, sky130 tt): PM 115.0 deg at
525 kHz and GM 32.7 dB at 18.4 MHz at IL = 1 mA, CL = 1 pF; PM 4.2 deg at
20.7 kHz at IL = 1 mA, CL = 1 uF; PM 31.9 deg at 100 mA with 1 uF; the
dominant pole is the GATE node's, below 1 kHz at every point.
"""
import warnings
from pathlib import Path

import numpy as np
import pytest

from circuitinsight.adapters.spectre import SpectreRun
from circuitinsight.analysis.marginledger import margin_ledger

FIX = Path(__file__).resolve().parent / "fixtures" / "spectre" / "fastldo"
POINTS = ["20u_1p", "20u_10n", "20u_1u", "1m_1p", "1m_100n", "1m_1u",
          "100m_1p", "100m_1u"]


def _run(point):
    cl = point.split("_")[1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SpectreRun(FIX / f"tb_fastldo_cl{cl}.cin.json",
                          FIX / f"psf_{point}")


@pytest.mark.parametrize("point", ["20u_1p", "1m_1u", "100m_1p", "100m_1u"])
def test_loop_gain_matches_stb(point):
    """The Tian loop gain at the main-loop probe reproduces Spectre's stb
    wherever |T| is above -20 dB, to the noise of the capture, once the
    capacitances are taken as the charge MATRIX: the lumped model reads
    0.11 dB high at 2 MHz on the light-load point, sky130's gate-drain
    non-reciprocity, which the band of a 500 kHz loop reaches."""
    run = _run(point)
    stb = run.stb()
    T = run.analyzer(cap_model="matrix").loop_gain("IPRB0", []).numeric_hp(
        stb.freq)
    Ts = np.asarray(stb.loop_gain)
    band = np.abs(Ts) > 0.1
    assert band.sum() > 40
    dberr = np.max(np.abs(20 * np.log10(np.abs(T[band]))
                          - 20 * np.log10(np.abs(Ts[band]))))
    pherr = np.max(np.abs(np.angle(T[band], deg=True)
                          - np.angle(Ts[band], deg=True)))
    assert dberr < 0.01 and pherr < 0.1


def test_the_ledger_reads_the_margins_spectre_reports():
    """At 1 mA and 1 pF the ledger's PM and f_u land on the stb margins,
    and the dominant pole is the GATE node's, well below the unity
    crossing."""
    run = _run("1m_1p")
    stb = run.stb()
    led = margin_ledger(run.analyzer(cap_model="matrix").loop_gain("IPRB0",
                                                                   []))
    assert led.pm_deg == pytest.approx(stb.phase_margin_deg, abs=0.2)
    assert led.f_unity_hz == pytest.approx(stb.phase_margin_freq_hz,
                                           rel=0.01)
    assert led.pm_deg == pytest.approx(115.0, abs=0.5)
    dom = min((e for e in led.entries if e.kind == "pole" and e.f0_hz > 0),
              key=lambda e: e.f0_hz)
    assert dom.f0_hz < 1e3
    assert led.stability["verdict"] == "stable"
    assert led.stability["bode_applies"] is True
