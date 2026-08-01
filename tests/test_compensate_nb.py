"""N-B: semantic candidate generation on the SKY130 two-stage.

The detectors must rediscover the amplifier's structure from the matrix
alone: the Miller-bridge detector finds the second stage (I0.net1 -> vout,
strongly inverting), pole participation localizes the dominant poles, the
Sherman-Morrison loop-gain updater reproduces a direct re-solve exactly,
and generate_candidates assembles a ranked, explained list.
"""
import warnings
from pathlib import Path

import numpy as np
import pytest

from circuitinsight.adapters.spectre import SpectreRun
from circuitinsight.analysis.compensate import (Candidate, LoopGainUpdater,
                                                generate_candidates,
                                                miller_candidates,
                                                pole_participation)

FIX = Path(__file__).resolve().parent / "fixtures" / "spectre"


@pytest.fixture(scope="module")
def unc():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run = SpectreRun(FIX / "miller_unc" / "tb_ota2s.cin.json",
                         FIX / "miller_unc" / "psf")
        an = run.analyzer(cap_model="matrix")
        return an, an.system("VIND")


@pytest.fixture(scope="module")
def stb_sys():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run = SpectreRun(FIX / "miller" / "tb_ota2s_stb.cin.json",
                         FIX / "miller" / "psf_stb")
        an = run.analyzer(cap_model="matrix")
        return an, an.system("IPRB0")


def test_miller_detector_finds_the_second_stage(unc):
    """The classic Miller position -- across the inverting second stage --
    must surface from the DC gain matrix with a large negative gain."""
    an, sys_ = unc
    cands = miller_candidates(sys_, min_gain=2.0)
    pairs = [(a, b) for a, b, _ in cands]
    assert ("I0.net1", "vout") in pairs
    g = dict(((a, b), g) for a, b, g in cands)[("I0.net1", "vout")]
    assert g < -50               # a real gain stage, not a coupling artifact


def test_pole_participation_localizes_the_dominant_poles(unc):
    """Both dominant modes live mostly on the high-impedance first-stage
    output I0.net1 (measured 82%/85% -- they are COUPLED through cgd of the
    second stage, so the textbook one-node-per-pole picture does not hold),
    with vout the clear second participant."""
    an, sys_ = unc
    parts = pole_participation(sys_, n_poles=2)
    (p1, r1), (p2, r2) = parts
    assert abs(p1) == pytest.approx(1.095e5, rel=0.01)
    assert r1[0][0] == "I0.net1" and r1[0][1] > 0.5
    assert abs(p2) == pytest.approx(1.037e6, rel=0.01)
    assert r2[0][0] == "I0.net1"
    assert r2[1][0] == "vout" and r2[1][1] > 0.05


def test_sherman_morrison_matches_direct_resolve(stb_sys):
    """The SM fast path must equal a from-scratch loop gain with the branch
    actually installed -- exact algebra, not an approximation. Candidate:
    the Miller bridge itself (add 5 pF on top of the fixture's 16 pF)."""
    from circuitinsight.engine.mna import build_mna
    from circuitinsight.engine.primitives import Primitive
    from circuitinsight.analysis.loopgain import loop_gain

    an, sys_ = stb_sys
    freqs = np.logspace(3, 9, 25)
    upd = LoopGainUpdater(sys_, "IPRB0", freqs)

    C = 5e-12
    pair = ("I0.net1", "vout")
    sm = upd.with_branch(*pair, lambda s: s * C)

    prims = list(an.primitives) + [Primitive("Cx", "", "c", pair, C)]
    sys2 = build_mna(prims, an.flat.ground, "IPRB0", an._alias)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        T2 = loop_gain(sys2, "IPRB0", [])
    direct = T2.numeric(freqs)
    assert np.max(np.abs(sm - direct) / np.abs(direct)) < 1e-9


def test_sherman_morrison_baseline_is_the_plain_loop_gain(stb_sys):
    from circuitinsight.analysis.loopgain import loop_gain

    an, sys_ = stb_sys
    freqs = np.logspace(3, 9, 15)
    upd = LoopGainUpdater(sys_, "IPRB0", freqs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        T = loop_gain(sys_, "IPRB0", [])
    assert np.max(np.abs(upd.baseline() - T.numeric(freqs))
                  / np.abs(T.numeric(freqs))) < 1e-9


def test_generate_candidates_speaks_semantics(unc):
    """The assembled vocabulary covers the classic Miller second-stage
    bridge (final ranking against the MFM+area cost is N-C's job -- the DC
    gain matrix legitimately finds other strongly inverting pairs too),
    covers the dominant-pole owner with a shunt-RC suggestion, and every
    entry carries a human rationale."""
    an, sys_ = unc
    cands = generate_candidates(sys_)
    assert all(isinstance(c, Candidate) and c.rationale for c in cands)
    kinds = {c.kind for c in cands}
    assert {"miller", "shunt_rc"} <= kinds
    millers = [(c.node_a, c.node_b) for c in cands if c.kind == "miller"]
    assert ("I0.net1", "vout") in millers
    assert max(c.score for c in cands if c.kind == "miller") > 100
    shunt_nodes = {c.node_a for c in cands if c.kind == "shunt_rc"}
    assert "I0.net1" in shunt_nodes
