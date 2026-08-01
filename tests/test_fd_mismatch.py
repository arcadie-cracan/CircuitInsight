"""D-D: deliberately mismatched fd OTA (MN1P w +10%) vs Spectre stb.

The mismatch enters through the REAL asymmetric operating point (the
psf_dmm/psf_cmm captures), not the engine's scale knob: the same
topology CIN picks up per-side device data and the whole chain --
per-mode margins, mode matrix, coupling -- is validated against
Spectre's own verdicts on the same deck.
"""
import warnings
from pathlib import Path

import numpy as np
import pytest

from circuitinsight.adapters.spectre import SpectreRun
from circuitinsight.session import SessionController

FIX = Path(__file__).resolve().parent / "fixtures" / "spectre" / "fd"
DM, CM = "FDPRB.IPRB_DM", "FDPRB.IPRB_CM"


@pytest.mark.parametrize("psf,probe,pm_ref,fu_ref", [
    ("psf_dmm", DM, 64.1189, 1.66716e6),
    ("psf_cmm", CM, 61.4784, 1.25118e6),
])
def test_margins_match_spectre_on_the_mismatched_op(psf, probe,
                                                    pm_ref, fu_ref):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = SessionController.open(FIX / "tb_fdota_stb.cin.json",
                                   FIX / psf, cap_model="matrix")
        r = s.loop_gain(probe)
    assert r.pm_deg == pytest.approx(pm_ref, abs=0.02)
    assert r.pm_freq_hz == pytest.approx(fu_ref, rel=1e-3)


def test_mismatch_shifts_the_dm_margin():
    """+10% MN1P width moves the DM margin by ~-0.8 deg vs the
    symmetric fixture (64.92 -> 64.12) while CM barely moves -- the
    diff-pair mismatch lives in the DM loop."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sym = SpectreRun(FIX / "tb_fdota_stb.cin.json",
                         FIX / "psf_dm").stb()
        mis = SpectreRun(FIX / "tb_fdota_stb.cin.json",
                         FIX / "psf_dmm").stb()
    assert sym.phase_margin_deg - mis.phase_margin_deg == pytest.approx(
        0.80, abs=0.05)


def test_mode_matrix_on_the_true_asymmetric_op():
    """The 2x2 machinery on the real mismatched OP: eigenloci margins
    agree with the per-probe effective ones, the cross-mode coupling is
    finite but tiny (real 10% W mismatch ~ 3e-6, far below the 50%
    gm-knob's 1e-3 -- Hurst's 'rarely destabilizing' quantified), and
    the Schur certificate stays at its documented level."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run = SpectreRun(FIX / "tb_fdota_stb.cin.json", FIX / "psf_dmm")
        rep = run.analyzer(cap_model="matrix").mode_loop(DM, CM)
    (pm_dm, fu_dm, _), (pm_cm, fu_cm, _) = rep.margins
    assert pm_dm == pytest.approx(64.12, abs=0.05)
    assert pm_cm == pytest.approx(61.48, abs=0.05)
    assert 1e-8 < rep.max_coupling < 1e-4
    assert rep.schur_residual < 1e-3
