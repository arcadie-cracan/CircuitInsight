"""The full uA741 op-amp, symbolically — the canonical bipolar amplifier.

Circuit and reference numbers from Gross & Roberts, "Analog Integrated
Circuits" (McGill SPICE decks, Ch.10): 26 BJTs (Gummel-Poon npn/pnp),
open-loop gain 294.6 kV/V (109.4 dB), unity-gain 0.652 MHz. The fixture
uses the textbook open-loop-gain rig: a huge feedback inductor (Lf, shorts
at DC so the operating point is the linear closed-loop bias -- essential,
since the amp railed open-loop) and a huge cap grounding the inverting
input at AC (loop opens). CircuitInsight reproduces both regimes from its
L/C primitives, so tf(Vin, "22") is the open-loop gain above the rig
corner.

This is the first real-silicon exercise of the BJT hybrid-pi path
(gm/gpi/go/cpi/cmu + collector-substrate junction cap) and of the
overflow-safe numeric evaluation (44-device exact determinant).
"""
import warnings
from pathlib import Path

import numpy as np
import pytest

from circuitinsight.adapters.spectre import SpectreRun

FIX = Path(__file__).resolve().parent / "fixtures" / "spectre" / "ua741"


@pytest.fixture(scope="module")
def run():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return SpectreRun(FIX / "ua741.cin.json", FIX / "psf")


def test_join_and_bjt_canonicalization(run):
    assert len(run.op_data) == 42          # 25 BJT + 11 R + Cc + Lf/Cf + 3 V
    q1 = run.op_data["Q1"]
    # hybrid-pi core present, ib matches the reference's 34.5 nA bias current
    for k in ("gm", "gpi", "go", "cpi", "cmu", "csub"):
        assert q1[k] > 0, k
    assert q1["ib"] == pytest.approx(34.5e-9, rel=0.05)


def test_output_stage_is_biased_active(run):
    # the closed-loop-DC rig must put the output devices in forward-active
    # (region 1); open-loop DC railed them into cutoff
    for dev in ("Q13A", "Q14", "Q20"):
        p = run.op_data[dev]
        assert int(p["region"]) == 1, dev
        assert abs(p["ic"]) > 1e-5, dev


def test_open_loop_gain_matches_spectre(run):
    H = run.analyzer().tf("Vin", "22", keep=[])
    ac = run.ac()
    f = ac.freq
    hs = np.asarray(ac.wave("22"))
    hm = H.numeric(f)

    # DC gain: reference 294.6 kV/V (109.4 dB); Spectre agrees; the pure
    # hybrid-pi reconstruction lands within ~1.6 dB (residual is the
    # un-modeled rb/re/rc series parasitics)
    g_sim = 20 * np.log10(abs(hs[0]))
    g_mod = 20 * np.log10(abs(hm[0]))
    assert g_sim == pytest.approx(109.4, abs=0.5)     # sim reproduces ref
    assert g_mod == pytest.approx(g_sim, abs=2.0)
    assert abs(hm[0]) > 2e5                            # > 106 dB

    # dominant-pole band: the substrate cap makes the reconstruction track
    # Spectre to a tenth of a dB from 10 Hz through 100 kHz
    band = (f >= 10) & (f <= 1e5)
    err = np.abs(20 * np.log10(np.abs(hm[band]) / np.abs(hs[band])))
    assert err.max() < 0.25


def test_extrinsic_model_closes_the_dc_gain_gap(run):
    # the intrinsic residual (~1.5 dB) is exactly the rb/re/rc series
    # parasitics: bjt_model='extrinsic' reproduces the simulator's
    # open-loop gain to the digit. Evaluated with the direct numeric MNA
    # solve (the symbolic determinant is impractical at ~120 nodes).
    ac = run.ac()
    f = ac.freq
    hs = np.asarray(ac.wave("22"))
    plateau = (f >= 0.05) & (f <= 1)          # between rig corner and p1
    g_sim = 20 * np.log10(np.abs(hs[plateau]).max())

    fp = f[plateau]
    g_intr = 20 * np.log10(np.abs(
        run.analyzer("lumped", "intrinsic").frequency_response(
            "Vin", "22", fp)).max())
    g_extr = 20 * np.log10(np.abs(
        run.analyzer("lumped", "extrinsic").frequency_response(
            "Vin", "22", fp)).max())

    assert g_intr == pytest.approx(g_sim - 1.54, abs=0.2)   # ~1.5 dB low
    assert g_extr == pytest.approx(g_sim, abs=0.1)          # exact


def test_band_sensitivity_ranks_the_signal_path(run):
    # feature-aligned band sensitivity (numeric, sub-second) must find the
    # dominant pole and rank the high-gain second stage + Miller cap
    an = run.analyzer()
    bs = an.band_sensitivities("Vin", "22", fmin=1, fmax=1e6)
    assert np.any((bs.poles > 1) & (bs.poles < 10))       # dom. pole ~ few Hz
    keep = bs.suggest_keep(8)
    assert any(k in ("gm_Q16", "gm_Q17") for k in keep)   # 2nd stage
    assert "Cc" in keep                                   # compensation cap

    # the whole-complex metric surfaces the phase-critical dominant-pole cap
    # higher than the magnitude-only metric does
    kc = an.band_sensitivities("Vin", "22", "complex", 1, 1e6).suggest_keep(5)
    km = an.band_sensitivities("Vin", "22", "magnitude", 1, 1e6).suggest_keep(5)
    assert "Cc" in kc and "Cc" not in km


def test_substrate_cap_improves_high_frequency(run):
    # the collector-substrate cap is what rolls the gain off correctly past
    # the dominant pole: without it the model stays far too high at 10 MHz
    ac = run.ac()
    f = ac.freq
    hs = np.asarray(ac.wave("22"))
    i = np.argmin(np.abs(f - 1e7))

    with_sub = run.analyzer()
    hm = with_sub.tf("Vin", "22", keep=[]).numeric(f)
    err_with = abs(20 * np.log10(abs(hm[i]) / abs(hs[i])))

    # drop csub from every BJT record -> intrinsic-only model
    stripped = {n: {k: v for k, v in p.items() if k != "csub"}
                for n, p in run.op_data.items()}
    from circuitinsight import Analyzer
    hm0 = Analyzer(run.flat, stripped).tf("Vin", "22", keep=[]).numeric(f)
    err_without = abs(20 * np.log10(abs(hm0[i]) / abs(hs[i])))

    assert err_with < err_without - 10        # csub cuts >10 dB of HF error
