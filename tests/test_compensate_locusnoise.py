"""D-E follow-on: scale-invariant RHP classification of the compensation loci.

The suggester rejects a sized candidate when its closed-loop natural
frequencies leave the left half plane. The test that decides this used a
fixed ABSOLUTE threshold (poles.real > 1e-3 Hz), which does not scale: a
high-degree characteristic polynomial rooted numerically gives its large
(fast) roots an absolute real-part error of order |p|*eps, which on a
GHz-scale circuit is many orders above 1 mHz -- so a genuinely stable pole
would read as unstable and its candidate be dropped for a numerical reason.

`_is_rhp` replaces it with a scale-invariant damping test: unstable iff
Re(p) > rel_tol*|p|. The same relative root noise (Re/|p| ~ 1e-9..1e-12)
never trips it, while a real instability (Re/|p| from ~1e-4 up to O(1))
always does. These tests pin that behavior and confirm the decisions on the
real fd bench are unchanged.
"""
import warnings
from pathlib import Path

import numpy as np
import pytest

from circuitinsight.adapters.spectre import SpectreRun
from circuitinsight.analysis.compensate import (_is_rhp, pair_locus_family)
from circuitinsight.engine.mna import build_mna

FIX = Path(__file__).resolve().parent / "fixtures" / "spectre" / "fd"


def test_catches_genuine_instability_at_any_scale():
    """A right-half-plane pair (damping ratio clearly negative) is flagged
    whether it sits at kHz or at GHz."""
    for scale in (1e3, 1e6, 1e9):
        poles = np.array([-2 * scale,
                          complex(0.1 * scale, scale),
                          complex(0.1 * scale, -scale)])   # Re/|p| ~ 0.1
        assert _is_rhp(poles)


def test_ignores_high_frequency_root_noise():
    """A stable (here marginal) GHz-scale pair whose real part carries only
    numerical noise is NOT flagged -- though the old absolute test (any
    real > 1e-3 Hz) WOULD have dropped it."""
    poles = np.array([-1e9,
                      complex(0.5, 1e9),        # +0.5 Hz of root noise
                      complex(0.5, -1e9)])
    assert not _is_rhp(poles)                    # Re/|p| = 5e-10, below floor
    assert np.any(poles.real > 1e-3)             # the old test fails here


def test_ignores_poles_negligible_against_the_spectral_scale():
    """A near-origin numerical-zero root with sign noise does not, by its
    tiny magnitude, masquerade as an instability."""
    poles = np.array([complex(1e-6, 1e-9),       # |p| ~ 1e-6, dwarfed
                      -1e8, complex(-2e8, 3e8)])
    assert not _is_rhp(poles)


def test_empty_and_all_lhp():
    assert not _is_rhp(np.array([]))
    assert not _is_rhp(np.array([-1e6, complex(-2e6, 1e6), complex(-2e6, -1e6)]))


@pytest.fixture(scope="module")
def stripped():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run = SpectreRun(FIX / "tb_fdota_stb.cin.json", FIX / "psf_dm")
        an = run.analyzer(cap_model="matrix")
        return build_mna(
            [p for p in an.primitives if p.inst not in ("I0.CCP", "I0.CCN")],
            an.flat.ground, "FDPRB.IPRB_DM", an._alias)


def test_fd_decisions_are_unchanged_by_the_relative_test(stripped):
    """On the real fd bench the reduced characteristic polynomial is low
    degree and well conditioned, so absolute and relative tests AGREE: the
    stabilizing Miller pair is stable (both say LHP), the uncompensated bench
    is genuinely unstable (both say RHP). The relative test changes nothing
    here -- it only adds headroom for faster / higher-degree systems."""
    pair, twin = ("I0.netp", "outpi"), ("I0.netn", "outni")
    stable = pair_locus_family(stripped, pair, twin)(16e-12, 2.5e3)
    assert not _is_rhp(stable)
    assert not np.any(stable.real > 1e-3)        # absolute agrees: stable

    tiny = pair_locus_family(stripped, pair, twin)(0.2e-12, 0.0)
    assert _is_rhp(tiny)                          # too little C: still RHP
    assert np.any(tiny.real > 1e-3)              # absolute agrees: unstable
