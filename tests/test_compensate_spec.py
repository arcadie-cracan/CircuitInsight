"""M-F(b): goal="spec" compensation -- Middlebrook's peak-sensitivity
target Ms = max|1/(1-T)| (the discrepancy/tolerance criterion), an exact
robustness spec rather than a PM floor.
"""
import warnings
from pathlib import Path

import numpy as np
import pytest

from circuitinsight.analysis.compensate import (Candidate, LoopGainUpdater,
                                                _margins_of,
                                                _peak_sensitivity)
from circuitinsight.engine.mna import build_mna
from circuitinsight.session import SessionController

FIX = Path(__file__).resolve().parent / "fixtures" / "spectre" / "miller"
CAND = [Candidate("miller", "I0.net1", "vout", "Miller port", 1.0)]


@pytest.fixture(scope="module")
def sess():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SessionController.open(FIX / "tb_ota2s_stb.cin.json",
                                      FIX / "psf_stb", cap_model="matrix")


def test_peak_sensitivity_uses_the_stb_return_difference(sess):
    """Ms = max|1/(1-T)| (NOT 1/(1+T)): loop_gain is stb-convention, where
    instability is T=+1, so a near-unstable loop must give a huge Ms and
    a well-damped one ~1.6 -- monotone in PM."""
    an = sess._analyzer_ready()
    prims = [p for p in an.primitives if p.inst != "I0.Cc"]
    sysp = build_mna(prims, an.flat.ground, "IPRB0", an._alias)
    lg = LoopGainUpdater(sysp, "IPRB0", np.geomspace(1e4, 1e9, 600))
    got = []
    for Cpf in (1, 8, 16, 32):
        T = lg.with_branch("I0.net1", "vout", lambda s, C=Cpf * 1e-12: s * C)
        pm, _, _ = _margins_of(lg.freqs, T)
        got.append((pm, _peak_sensitivity(lg.freqs, T)))
    ms = [m for _, m in got]
    assert ms[0] > 20                       # PM ~ -1 deg: near instability
    assert ms == sorted(ms, reverse=True)   # falls monotonically with C/PM
    assert 1.4 < ms[2] < 2.0                # PM ~ 60 deg -> Ms ~ 1.7


def test_spec_goal_tightening_costs_area(sess):
    """A tighter Ms target needs more compensation (larger area), and each
    achiever meets its target -- the least-area feasible point."""
    prev_area = 0.0
    for ms in (1.5, 1.3, 1.2):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sg = sess.suggest_compensation("IPRB0", goal="spec",
                                           exclude=("I0.Cc",),
                                           candidates=CAND, ms_target=ms)
        b = sg[0]
        assert b.achieved and b.spec_dev <= ms + 1e-6
        assert b.pm_deg is not None and b.pm_deg > 45
        assert b.area >= prev_area - 1e-9    # tighter target -> more area
        prev_area = b.area


def test_spec_rejects_a_near_unstable_candidate(sess):
    """An impossible target (Ms <= 1.0, unreachable since Ms>=1) yields no
    achiever, and no suggestion claims success."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sg = sess.suggest_compensation("IPRB0", goal="spec",
                                       exclude=("I0.Cc",), candidates=CAND,
                                       ms_target=1.0)
    assert all(not s.achieved for s in sg)
