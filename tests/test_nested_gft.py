"""M-H: nested (MIMO) GFT on the fd bench (analysis/nested_gft.py).

The headline vs the MIMO-Tian obstruction (plan Secs. 10.1/10.2): the
nested null dissection stays EXACT at both levels for any mismatch --
nulls compose in determinant algebra where terminated wire measurements
do not. The corner gain is order-invariant, the level-2 loop gain
commutes with the plain one exactly under symmetry, and their ratio is
a zero-error coupling diagnostic under mismatch.
"""
import warnings
from pathlib import Path

import numpy as np
import pytest

FIX = Path(__file__).resolve().parent / "fixtures" / "spectre" / "fd"
DM, CM = "FDPRB.IPRB_DM", "FDPRB.IPRB_CM"
ERR_DM, ERR_CM = ("vin_dm", +1), ("vcmref", -1)
MISMATCH = {"gm_I0_MN1P": 1.5, "gm_I0_MPLN": 1.3}
FREQS = [1e2, 1e4, 1e6, 1e8]


@pytest.fixture(scope="module")
def an():
    from circuitinsight.adapters.spectre import SpectreRun

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run = SpectreRun(FIX / "tb_fdota_stb.cin.json", FIX / "psf_dm")
        return run.analyzer(cap_model="matrix")


@pytest.fixture(scope="module")
def sym(an):
    return an.nested_gft(DM, CM, "VIND", "FDPRB.dmo", ERR_DM, ERR_CM)


@pytest.fixture(scope="module")
def mis(an):
    return an.nested_gft(DM, CM, "VIND", "FDPRB.dmo", ERR_DM, ERR_CM,
                         scale=MISMATCH)


def test_both_levels_exact_symmetric(sym):
    assert sym.identity_residual(1, s_points=(2, 3)) == 0.0
    assert sym.identity_residual(2, s_points=(2, 3)) == 0.0


def test_both_levels_exact_under_mismatch(mis):
    """THE contrast with MIMO Tian: 50%/30% gm mismatch and the nested
    dissection is still exact at both levels -- no O(mismatch^2)
    residual anywhere."""
    assert mis.identity_residual(1, s_points=(2, 3)) == 0.0
    assert mis.identity_residual(2, s_points=(2, 3)) == 0.0


def test_ideal_gain_is_the_inverting_unity(sym):
    """Hinf1 flat at the ideal inverting gain (0 dB) through the band --
    the M-F flatness hallmark for a well-designated error."""
    h = sym.level1([1e1, 1e2, 1e3, 1e4])["Hinf"]
    assert np.allclose(np.abs(h), 10 ** (-0.0019 / 20), rtol=1e-3)


def test_corner_is_order_invariant(an, mis):
    """Hinf12 (both nulls) is the same whether DM or CM is dissected
    first -- same doubly-constrained system -- even under mismatch."""
    rev = an.nested_gft(CM, DM, "VIND", "FDPRB.dmo", ERR_CM, ERR_DM,
                        scale=MISMATCH)
    c12 = mis.level2(FREQS)["Hinf"]
    c21 = rev.level2(FREQS)["Hinf"]
    assert np.allclose(c12, c21, rtol=1e-9)


def test_symmetric_loops_commute(sym):
    """Under matched symmetry, idealizing the DM loop does not change
    the CM loop gain: T2' == plain T2 to solver precision, and the
    coupling diagnostic is numerically zero."""
    t2p = sym.level2(FREQS)["T"]
    t2 = sym.plain2(FREQS)["T"]
    assert np.allclose(t2p, t2, rtol=1e-9)
    assert np.max(sym.coupling(FREQS)) < 1e-9


def test_mismatch_shows_up_only_in_coupling(mis):
    """With mismatch the coupling diagnostic is finite (loop 2 changes
    when loop 1 is idealized) while both identities remain exact --
    the diagnostic itself carries no approximation error."""
    cpl = mis.coupling(FREQS)
    assert np.max(cpl) > 1e-5            # peaks ~2e-4 near crossover
