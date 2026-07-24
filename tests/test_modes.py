"""D-C: 2x2 mode loop matrix at the fd probe pair (analysis/modes.py).

Locked empirical facts (see the module docstring): the block Eq.-30
matrix is return-difference exact (det(I-L) nulls at the closed-loop
poles even under strong mismatch), exactly diagonal under matched
symmetry with the scalar Tian on its diagonal, and its Schur closure
reproduces the closed-other-branch scalar measurement to O(mismatch^2)
-- the report's own accuracy certificate.
"""
import warnings
from pathlib import Path

import numpy as np
import pytest

from circuitinsight.adapters.spectre import SpectreRun
from circuitinsight.analysis.modes import (_numeric_fn, loop_matrix_at,
                                           mode_loop)
from circuitinsight.analysis.probeadequacy import _dominant_poles

FIX = Path(__file__).resolve().parent / "fixtures" / "spectre" / "fd"
DM, CM = "FDPRB.IPRB_DM", "FDPRB.IPRB_CM"
MISMATCH = {"gm_I0_MN1P": 1.5, "gm_I0_MPLN": 1.3}


@pytest.fixture(scope="module")
def system():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run = SpectreRun(FIX / "tb_fdota_stb.cin.json", FIX / "psf_dm")
        return run.analyzer(cap_model="matrix").system(None)


@pytest.fixture(scope="module")
def sym(system):
    return mode_loop(system, DM, CM)


@pytest.fixture(scope="module")
def mis(system):
    return mode_loop(system, DM, CM, scale=MISMATCH)


def test_symmetric_L_is_exactly_diagonal(sym):
    off = np.max(np.abs(sym.L[:, 0, 1])) + np.max(np.abs(sym.L[:, 1, 0]))
    diag = np.max(np.abs(sym.L[:, 0, 0]))
    assert off < 1e-10 * diag
    assert sym.schur_residual < 1e-10
    assert sym.max_coupling < 1e-12


def test_symmetric_loci_are_the_mode_loop_gains(sym):
    """Eigenloci == diagonal == scalar Tian per branch; margins match
    the Spectre stb references (DM 64.9197 deg, CM 61.4719 deg)."""
    assert np.allclose(sym.loci[:, 0], sym.T_eff[:, 0], rtol=1e-9)
    assert np.allclose(sym.loci[:, 1], sym.T_eff[:, 1], rtol=1e-9)
    (pm_dm, fu_dm, _), (pm_cm, fu_cm, _) = sym.margins
    assert pm_dm == pytest.approx(64.9197, abs=0.05)
    assert pm_cm == pytest.approx(61.4719, abs=0.05)
    assert fu_dm == pytest.approx(1.65254e6, rel=2e-3)
    assert fu_cm == pytest.approx(1.25091e6, rel=2e-3)


def test_return_difference_nulls_at_closed_poles(system):
    """det(I - L(s)) -> 0 approaching every dominant closed-loop pole,
    with and without mismatch: the generalized-Nyquist verdict is
    anchored to the exact det A."""
    for scale in (None, MISMATCH):
        fn, A_v = _numeric_fn(system, scale)
        poles = _dominant_poles(A_v, n=10)
        pairs = [p for p in poles if p.imag > 0.01 * abs(p)][:2]
        assert pairs
        for p in pairs:
            d_far, d_near = 3e-2, 1e-3
            g = loop_matrix_at(system, (DM, CM),
                               [p * (1 + d_far), p * (1 + d_near)], fn=fn)
            v_far = abs(np.linalg.det(np.eye(2) - g[0]))
            v_near = abs(np.linalg.det(np.eye(2) - g[1]))
            assert v_near < 0.2 * v_far          # decays toward the pole
            assert v_near < 5e-3


def test_mismatch_couples_the_modes(mis, sym):
    """Deliberate mismatch: off-diagonals rise from machine zero,
    coupling becomes finite, margins shift, and the Schur certificate
    stays at the documented O(mismatch^2) level."""
    assert np.max(np.abs(mis.L[:, 0, 1])) > 1e3 * np.max(
        np.abs(sym.L[:, 0, 1]))
    assert mis.max_coupling > 1e-4
    assert mis.schur_residual < 5e-3             # 50%/30% gm mismatch
    (pm_dm_m, _, _), _ = mis.margins
    (pm_dm_s, _, _), _ = sym.margins
    assert pm_dm_m != pytest.approx(pm_dm_s, abs=0.01)


def test_certificate_scales_with_mismatch(system):
    """The Schur residual is O(mismatch^2): a 2% mismatch must certify
    at least two orders tighter than the 50% one."""
    small = mode_loop(system, DM, CM, scale={"gm_I0_MN1P": 1.02})
    big = mode_loop(system, DM, CM, scale={"gm_I0_MN1P": 1.5})
    assert small.schur_residual < 1e-2 * big.schur_residual


def test_analyzer_wrapper(system):
    from circuitinsight.analyzer import Analyzer  # noqa: F401  (API smoke)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run = SpectreRun(FIX / "tb_fdota_stb.cin.json", FIX / "psf_dm")
        rep = run.analyzer(cap_model="matrix").mode_loop(DM, CM)
    assert rep.labels == [DM, CM]
    assert "PM" in rep.summary()
