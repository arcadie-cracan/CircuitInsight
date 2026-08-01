"""D-E: per-mode compensation for fully-differential circuits --
symmetric candidate pairs (analysis/compensate.py mirror support).

A mirrored pair of series-RC branches is sized as ONE candidate: the
pole locus uses the even/odd-mode factorization of the rank-2
determinant (pair_locus_family), the PM evaluation uses the rank-k
Woodbury updater, and the area counts both elements. Validated by
re-discovering the fd fixture's own Miller compensation from the
stripped bench.
"""
import warnings
from pathlib import Path

import numpy as np
import pytest

from circuitinsight.adapters.spectre import SpectreRun
from circuitinsight.analysis.compensate import (Candidate, LoopGainUpdater,
                                                pair_locus_family,
                                                suggest_compensation)
from circuitinsight.engine.mna import build_mna

FIX = Path(__file__).resolve().parent / "fixtures" / "spectre" / "fd"
PROBE = "FDPRB.IPRB_DM"
PAIR, TWIN = ("I0.netp", "outpi"), ("I0.netn", "outni")
MIRROR = {"I0.netp": "I0.netn", "outpi": "outni", "voutp": "voutn"}


@pytest.fixture(scope="module")
def rig():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run = SpectreRun(FIX / "tb_fdota_stb.cin.json", FIX / "psf_dm")
        an = run.analyzer(cap_model="matrix")
        stripped = build_mna(
            [p for p in an.primitives if p.inst not in ("I0.CCP",
                                                        "I0.CCN")],
            an.flat.ground, PROBE, an._alias)
        full = an.system(PROBE)
    return stripped, full


def test_pair_locus_reproduces_the_fixture_poles(rig):
    """Installing the fixture's own 16 pF pair on the stripped bench
    must reproduce the intact fixture's closed-loop poles (known from
    D-C): -0.956+/-0.606j and -2.69+/-2.073j MHz."""
    stripped, _ = rig
    roots = pair_locus_family(stripped, PAIR, TWIN)(16e-12)
    dom = roots[np.argsort(np.abs(roots))][:4] / 1e6
    ref = np.array([-0.956 + 0.606j, -0.956 - 0.606j,
                    -2.690 + 2.073j, -2.690 - 2.073j])
    for r in ref:
        assert np.min(np.abs(dom - r)) < 5e-3


def test_rank2_woodbury_reproduces_the_intact_fixture(rig):
    """The stripped system + the 16 pF pair via with_branches must give
    the SAME loop gain as the intact fixture -- rank-2 update vs
    direct build, to solver precision."""
    stripped, full = rig
    freqs = np.array([1e3, 1e5, 1e6, 1e7])
    Y = lambda s: s * 16e-12
    t_upd = LoopGainUpdater(stripped, PROBE, freqs).with_branches(
        [(PAIR[0], PAIR[1], Y), (TWIN[0], TWIN[1], Y)])
    t_ref = LoopGainUpdater(full, PROBE, freqs).baseline()
    assert np.allclose(t_upd, t_ref, rtol=1e-9)


def test_with_branches_singleton_matches_with_branch(rig):
    stripped, _ = rig
    freqs = np.array([1e4, 1e6])
    Y = lambda s: s * 2e-12
    upd = LoopGainUpdater(stripped, PROBE, freqs)
    a = upd.with_branch(PAIR[0], PAIR[1], Y)
    b = upd.with_branches([(PAIR[0], PAIR[1], Y)])
    assert np.allclose(a, b, rtol=1e-12)


def test_recompensation_finds_the_symmetric_miller_pair(rig):
    """MFM re-compensation of the stripped fd bench: the suggester must
    find an achieving symmetric pair at the Miller port, annotate the
    mirror, count both elements in the area, and rank the series-RC
    (phantom-zero) pair above the plain-C pair on area."""
    stripped, _ = rig
    cands = [Candidate("miller", PAIR[0], PAIR[1],
                       "stage-1 to output", 1.0)]
    sugg = suggest_compensation(stripped, PROBE, goal="mfm",
                                candidates=cands, mirror=MIRROR, top=4)
    assert sugg and sugg[0].achieved
    best = sugg[0]
    assert "symmetric pair with (I0.netn, outni)" in best.candidate.rationale
    assert best.network == "series-RC"
    assert abs(best.zeta - 1 / np.sqrt(2)) <= 0.05
    assert best.area == pytest.approx(
        2 * (best.C / 1e-12 + 0.05 * best.R / 1e3), rel=1e-9)
    plain = [s for s in sugg if s.network == "C" and s.achieved]
    assert plain and best.area < plain[0].area
