"""D-A: fully-differential two-stage OTA + CT CMFB fixture (SKY130).

Three loops, one circuit: the DM loop and the CM loop measured at a
probe PAIR (fd_probe = two ideal baluns back-to-back with an iprobe in
the DM wire and one in the CM wire -- a clean-room analogLib
diffstbprobe; Spectre margins from the Cadence probe and from fd_probe
agree to every printed digit), and the CM loop measured again at the
scalar iprobe cutting the vcmfb mirror wire (a different, equally valid
Hurst break of the same loop).

Spectre stb references (<sim-host>, SKY130 tt, 27C):
  DM   (FDPRB.IPRB_DM): PM 64.9197 deg @ 1.65254 MHz, GM 13.5715 dB
  CM   (FDPRB.IPRB_CM): PM 61.4719 deg @ 1.25091 MHz, GM 16.2558 dB
  CMFB (I0.IPRB_CMFB):  PM 60.9064 deg @ 1.26547 MHz, GM 16.3279 dB
"""
import warnings
from pathlib import Path

import numpy as np
import pytest

from circuitinsight.adapters.spectre import SpectreRun

FIX = Path(__file__).resolve().parent / "fixtures" / "spectre" / "fd"

CASES = {
    "dm": ("tb_fdota_stb.cin.json", "psf_dm", "FDPRB.IPRB_DM", 64.9197),
    "cm": ("tb_fdota_stb.cin.json", "psf_cm", "FDPRB.IPRB_CM", 61.4719),
    "cmfb": ("tb_fdota_stb_cmfb.cin.json", "psf_cmfb", "I0.IPRB_CMFB",
             60.9064),
}


def _tf_and_stb(case):
    cin, psf, probe, _ = CASES[case]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run = SpectreRun(FIX / cin, FIX / psf)
        an = run.analyzer(cap_model="matrix")
        return an.loop_gain(probe), run.stb()


@pytest.fixture(scope="module", params=list(CASES))
def loop(request):
    T, stb = _tf_and_stb(request.param)
    return request.param, T, stb


def test_stb_margins_pinned(loop):
    case, _, stb = loop
    assert stb.phase_margin_deg == pytest.approx(CASES[case][3], abs=1e-3)


def test_engine_matches_stb_in_band(loop):
    """Eq.-30 Tian loop gain vs the Spectre stb sweep, on the stb grid,
    up to 100 MHz (well past all crossovers): sub-0.01 dB / sub-0.1 deg."""
    _, T, stb = loop
    f = stb.freq
    inb = f <= 1e8
    ours = T.numeric(f[inb])
    theirs = stb.loop_gain[inb]
    mdb = 20 * np.log10(np.abs(ours)) - 20 * np.log10(np.abs(theirs))
    ph = ((np.angle(ours, deg=True) - np.angle(theirs, deg=True))
          + 180) % 360 - 180
    assert np.max(np.abs(mdb)) < 0.01
    assert np.max(np.abs(ph)) < 0.1


def test_engine_margin_matches_spectre(loop):
    """Unity-crossover phase margin computed on our own dense grid
    agrees with Spectre's margin to 0.05 deg (Spectre convention:
    arg T(DC) = +180, PM = arg T at |T| = 1)."""
    case, T, stb = loop
    fd = np.logspace(3, 8, 4001)
    Td = T.numeric(fd)
    mag = np.abs(Td)
    phs = np.degrees(np.unwrap(np.angle(Td)))
    k = int(np.where(mag <= 1.0)[0][0])
    lf = np.log10(fd)
    lfu = np.interp(0.0, [np.log10(mag[k]), np.log10(mag[k - 1])],
                    [lf[k], lf[k - 1]])
    # arg T(DC) = +180 (Spectre loopGain convention) is already in the
    # data: the interpolated phase at crossover IS the phase margin
    pm = np.interp(lfu, lf[k - 1:k + 1], phs[k - 1:k + 1])
    assert pm == pytest.approx(stb.phase_margin_deg, abs=0.05)
    assert 10 ** lfu == pytest.approx(stb.phase_margin_freq_hz, rel=2e-3)


def test_dm_cm_probes_share_one_op():
    """One netlist, one OP: the DM and CM measurements come from the
    same fixture files, so the probe pair is OP-invariant by
    construction (both iprobes are DC shorts)."""
    T_dm, _ = _tf_and_stb("dm")
    T_cm, _ = _tf_and_stb("cm")
    # both are defined, distinct loops on the same circuit
    f = [1e3]
    dm_db = 20 * np.log10(abs(complex(T_dm.numeric(f)[0])))
    cm_db = 20 * np.log10(abs(complex(T_cm.numeric(f)[0])))
    assert dm_db != pytest.approx(cm_db, abs=0.5)


def test_cmfb_and_cm_probe_agree_on_the_loop():
    """The vcmfb-wire cut and the output-pair CM cut are two breaks of
    the same CM loop: crossover frequencies within 2%, PM within 1 deg
    (they need not be identical -- different break points)."""
    _, stb_cm = _tf_and_stb("cm")
    _, stb_fb = _tf_and_stb("cmfb")
    assert (stb_fb.phase_margin_freq_hz
            == pytest.approx(stb_cm.phase_margin_freq_hz, rel=0.02))
    assert (stb_fb.phase_margin_deg
            == pytest.approx(stb_cm.phase_margin_deg, abs=1.0))
