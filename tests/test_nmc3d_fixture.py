"""nmc3d: behavioral fully-differential gm-C THREE-stage nested-Miller
amplifier -- the fully-differential mirrored-NMC showcase.

The mirror of nmc3: two nested Miller PAIRS, a clean-room fd_probe (two ideal
baluns + a DM iprobe), differential unity feedback. Its DM half-circuit is
exactly nmc3, so the DM loop needs the same two nested Miller caps -- here as
mirrored pairs. Being linear, its reconstruction is EXACT against Spectre stb
at the DM probe. Ground truth (Spectre IC25.1): DM PM 75.5102 deg @ 5.3998
MHz, GM 12.71 dB @ 28.47 MHz.

This is the end-to-end validation the fd two-stage could not give: a circuit
that genuinely NEEDS two mirrored Miller pairs (neither alone stabilizes the
DM loop), on which suggest_multi_compensation(mirror=...) grows both.
"""
import warnings
from pathlib import Path

import numpy as np
import pytest

from circuitinsight.adapters.spectre import SpectreRun
from circuitinsight.analysis.compensate import (Candidate, LoopGainUpdater,
                                                _is_rhp, _margins_of,
                                                multi_locus,
                                                suggest_multi_compensation)
from circuitinsight.engine.mna import build_mna

FIX = Path(__file__).resolve().parent / "fixtures" / "spectre" / "nmc3d"
PROBE = "IPRB_DM"
MIRROR = {"outp": "outn", "n1p": "n1n", "n2p": "n2n"}
MILLER = {"CM1p", "CM1n", "CM2p", "CM2n",
          "IPRB1p", "IPRB1n", "IPRB2p", "IPRB2n"}


def _run():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SpectreRun(FIX / "tb_nmc3d.cin.json", FIX / "psf")


def test_dm_loop_gain_is_exact_against_stb():
    """Linear circuit -> the reconstructed DM loop gain equals Spectre's stb
    loopGain to machine precision: 0 dB / 0 deg."""
    run = _run()
    stb = run.stb()
    T = run.analyzer().loop_gain(PROBE, []).numeric(stb.freq)
    Ts = np.asarray(stb.loop_gain)
    band = np.abs(Ts) > 0.1
    assert band.sum() > 40
    dberr = np.max(np.abs(20 * np.log10(np.abs(T[band]))
                          - 20 * np.log10(np.abs(Ts[band]))))
    pherr = np.max(np.abs(np.angle(T[band], deg=True)
                          - np.angle(Ts[band], deg=True)))
    assert dberr < 1e-4 and pherr < 1e-4


def test_dm_margin_matches_spectre():
    run = _run()
    assert run.stb().phase_margin_deg == pytest.approx(75.5102, abs=0.02)


@pytest.fixture(scope="module")
def uncomp():
    """The DM bench with both Miller pairs stripped (uncompensated)."""
    an = _run().analyzer()
    prims = [p for p in an.primitives if p.inst not in MILLER]
    return build_mna(prims, an.flat.ground, PROBE, an._alias)


def test_neither_miller_pair_alone_stabilizes_the_dm_loop(uncomp):
    """Genuine three-stage NMC: with a single mirrored Miller pair (inner OR
    outer) the DM loop still has right-half-plane poles; both nested pairs are
    needed. (Checked via the exact multi-branch locus over the physical pair.)"""
    upd_pairs = {
        "outer": [("outp", "n1p"), ("outn", "n1n")],
        "inner": [("outp", "n2p"), ("outn", "n2n")],
    }
    for pairs in upd_pairs.values():
        # a generous plain cap on just this mirrored pair
        poles = multi_locus(uncomp, pairs)([(6e-12, 0.0)] * 2)
        assert _is_rhp(poles)                       # one pair alone: unstable


def test_mirrored_nmc_synthesis_grows_two_pairs(uncomp):
    """The payoff: an aggressive peak-sensitivity spec (Ms <= 1.1) that no
    single pair can meet drives suggest_multi_compensation to grow the nested
    network as TWO mirrored Miller pairs -- a cap at BOTH Miller ports, each
    with its mirror twin -- stabilizing the fully-differential amplifier. The
    achieved margin is reproduced EXACTLY by an independent Woodbury
    recomputation over all four physical branches."""
    cands = [Candidate("miller", "outp", "n1p", "outer Miller (stages 2-3)", 50.0),
             Candidate("miller", "outp", "n2p", "inner Miller (stage 3)", 10.0)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = suggest_multi_compensation(
            uncomp, PROBE, goal="spec", ms_target=1.1, k_max=2,
            candidates=cands, mirror=MIRROR,
            c_grid=np.geomspace(0.5e-12, 20e-12, 18))
    assert m.achieved and len(m.branches) == 2 and m.spec_dev <= 1.1
    ports = {(b.node_a, b.node_b) for b in m.branches}
    assert ports == {("outp", "n1p"), ("outp", "n2p")}
    assert all(b.twin is not None and b.mult == 2 for b in m.branches)

    # independent EXACT recomputation over ALL FOUR physical branches
    phys, Ybr = [], []
    for b in m.branches:
        Y = (lambda C, R: (lambda s: s * C / (1 + s * R * C)))(b.C, b.R)
        for p in b.physical():
            phys.append(p); Ybr.append((p[0], p[1], Y))
    assert len(phys) == 4
    upd = LoopGainUpdater(uncomp, PROBE, np.geomspace(1e4, 1e9, 240))
    pm, fu, gm = _margins_of(upd.freqs, upd.with_branches(Ybr))
    assert pm == pytest.approx(m.pm_deg, abs=1e-6)
    assert not _is_rhp(multi_locus(uncomp, phys)(
        [(b.C, b.R) for b in m.branches for _ in b.physical()]))
