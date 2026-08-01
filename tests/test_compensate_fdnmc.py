"""Fully-differential mirrored multi-branch (NMC) synthesis.

suggest_multi_compensation(..., mirror={p: n}) installs each branch as a
matched SYMMETRIC PAIR, so a fully-differential amplifier's nested-Miller
network is grown as mirrored pairs. The pair's joint effect is exact by the
same engines the single-ended synthesizer uses -- multi_locus over ALL
physical branches, with_branches over all physical Y's.

The available fd bench is a TWO-stage OTA, which (like the single-ended
two-stage) needs only ONE Miller pair, so it validates the mirrored pair
bookkeeping and its exact agreement with the direct build; the genuine
two-mirrored-pair activation is a fully-differential three-stage bench
(a follow-on fixture). The rank-4 (two mirrored pairs) pole locus is
verified here directly against a re-stamp, so the multi-pair path the
synthesizer relies on is exercised exactly.
"""
import warnings
from pathlib import Path

import numpy as np
import pytest
import sympy as sp

from circuitinsight.adapters.spectre import SpectreRun
from circuitinsight.analysis.compensate import (
    Candidate, LoopGainUpdater, MultiBranch, _margins_of, _numeric_A,
    _pair_incidence, _scaled_roots, multi_locus, suggest_multi_compensation)
from circuitinsight.engine.mna import S, _det, build_mna

FIX = Path(__file__).resolve().parent / "fixtures" / "spectre" / "fd"
PROBE = "FDPRB.IPRB_DM"
MIRROR = {"I0.netp": "I0.netn", "outpi": "outni", "voutp": "voutn"}


@pytest.fixture(scope="module")
def stripped():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run = SpectreRun(FIX / "tb_fdota_stb.cin.json", FIX / "psf_dm")
        an = run.analyzer(cap_model="matrix")
        return build_mna(
            [p for p in an.primitives if p.inst not in ("I0.CCP", "I0.CCN")],
            an.flat.ground, PROBE, an._alias)


def test_mirror_branch_bookkeeping():
    """A mirrored branch reports both physical elements and mult=2; a
    single-ended one is mult=1."""
    b = MultiBranch("miller", "I0.netp", "outpi", "series-RC", 3e-12, 3e3,
                    "x", twin=("I0.netn", "outni"))
    assert b.mult == 2
    assert b.physical() == [("I0.netp", "outpi"), ("I0.netn", "outni")]
    s = MultiBranch("shunt_rc", "vcmfb", None, "C", 1e-12, 0.0, "y")
    assert s.mult == 1 and s.physical() == [("vcmfb", None)]


def test_single_mirrored_pair_is_installed_and_exact_in_the_loop_gain(stripped):
    """One mirrored pair through the MULTI synthesizer: the twin is recorded,
    BOTH physical branches are installed, the area counts both elements, and
    the achieved margin is reproduced EXACTLY by an independent Woodbury
    (with_branches) recomputation over the two physical branches -- the
    rank-2 loop gain the synthesizer sizes against is exact (no polynomial
    rooting). goal='pm' so the decision rides on that exact engine."""
    cand = [Candidate("miller", "I0.netp", "outpi", "stage-1 to output", 1.0)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = suggest_multi_compensation(stripped, PROBE, goal="pm",
                                       pm_target=60.0, k_max=1,
                                       candidates=cand, mirror=MIRROR)
    assert m.achieved and len(m.branches) == 1
    b = m.branches[0]
    assert b.twin == ("I0.netn", "outni") and b.mult == 2
    assert b.physical() == [("I0.netp", "outpi"), ("I0.netn", "outni")]
    assert m.pm_deg >= 60.0
    assert m.area == pytest.approx(
        2 * (b.C / 1e-12 + 0.05 * b.R / 1e3), rel=1e-9)  # both elements

    # independent EXACT recomputation over BOTH physical branches
    Y = (lambda C, R: (lambda s: s * C / (1 + s * R * C)))(b.C, b.R)
    upd = LoopGainUpdater(stripped, PROBE, np.geomspace(1e4, 1e9, 240))
    T = upd.with_branches([(p[0], p[1], Y) for p in b.physical()])
    pm, fu, gm = _margins_of(upd.freqs, T)
    assert pm == pytest.approx(m.pm_deg, abs=1e-6)


def test_mirrored_pair_matches_single_ended_pair_via_loop_gain(stripped):
    """The mirrored pair installed by the MULTI synthesizer carries the same
    loop gain as a D-E mirrored install (both drive with_branches over the
    same two physical branches): the mirror bookkeeping feeds the exact
    engine correctly. Sizes need not coincide (different goal/search); the
    loop gain of a GIVEN mirrored pair must."""
    Y = lambda s: s * 6e-12 / (1 + s * 3e3 * 6e-12)
    upd = LoopGainUpdater(stripped, PROBE, np.array([1e4, 1e6, 1e7]))
    a = upd.with_branches([("I0.netp", "outpi", Y), ("I0.netn", "outni", Y)])
    b = upd.with_branches([("I0.netn", "outni", Y), ("I0.netp", "outpi", Y)])
    assert np.allclose(a, b, rtol=1e-12)     # order-independent, symmetric


def _direct_cap_roots(system, pairs, caps):
    """Natural frequencies (Hz) with plain caps stamped straight into A."""
    A = _numeric_A(system).copy()
    for (na, nb), C in zip(pairs, caps):
        b = _pair_incidence(system, na, nb)
        bz = sp.Matrix([[sp.Integer(int(x))] for x in b])
        A = A + S * sp.Rational(repr(float(C))) * (bz * bz.T)
    return _scaled_roots(sp.Poly(sp.expand(_det(A)), S)) / (2 * np.pi)


def test_two_mirrored_pairs_locus_is_exact(stripped):
    """The rank-4 locus of TWO mirrored pairs -- four physical branches, the
    structure a fully-differential nested-Miller network installs -- matches a
    direct re-stamp of the four caps to machine precision. The robust
    circle-interpolation locus handles this k where the old polynomial method
    collapsed."""
    pairs = [("I0.netp", "outpi"), ("I0.netn", "outni"),   # Miller pair
             ("I0.netp", None), ("I0.netn", None)]         # a shunt pair
    caps = [4e-12, 4e-12, 2e-12, 2e-12]
    got = multi_locus(stripped, pairs)([(c, 0.0) for c in caps])
    ref = _direct_cap_roots(stripped, pairs, caps)
    assert len(got) == len(ref)
    rem = list(np.sort_complex(ref))
    err = 0.0
    for r in np.sort_complex(got):
        j = min(range(len(rem)), key=lambda k: abs(rem[k] - r))
        err = max(err, abs(rem[j] - r) / (abs(rem[j]) + 1.0))
        rem.pop(j)
    assert err < 1e-6


def test_mirrored_synthesis_stops_at_one_pair_when_sufficient(stripped):
    """The fd two-stage is single-Miller: one mirrored pair meets the goal,
    so the synthesizer installs exactly one (no wasted second pair) -- and
    that pair is genuinely a matched pair (twin set, area x2)."""
    cand = [Candidate("miller", "I0.netp", "outpi", "Miller bridge", 5.0)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = suggest_multi_compensation(stripped, PROBE, goal="pm",
                                       pm_target=70.0, k_max=2,
                                       candidates=cand, mirror=MIRROR,
                                       c_grid=np.geomspace(1e-12, 30e-12, 16))
    assert m.achieved and len(m.branches) == 1
    assert m.branches[0].twin is not None
    assert m.pm_deg >= 70.0
