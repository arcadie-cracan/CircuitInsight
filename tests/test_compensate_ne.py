"""N-E: multi-branch (nested-Miller / NMC) compensation synthesis.

Two layers are exercised:

* the exact multi-branch pole locus (multi_port_rational / multi_locus): the
  rank-k determinant identity char = det(D I + Y-hat M)/D^(k-1), assembled and
  root-found in numpy from port numerators extracted once. Ground truth is an
  explicit internal-node stamp of every branch -- the same natural frequencies
  to machine precision, for plain, mixed, and all-series-RC branch sets.

* the greedy synthesizer (suggest_multi_compensation): grows the network one
  OP-invariant branch at a time by successive rank-one updates. It must reduce
  to the single-branch suggester at k=1, must NOT add a branch the goal does
  not need, and -- when it does grow two branches -- the two independent exact
  engines (multi_locus poles, the Woodbury loop gain) must agree.

The available bench is the single-Miller two-stage, which physically needs
only ONE branch; the genuine 3-stage NMC showcase waits for the three-loop
fixture. So the two-branch path here is exercised by excluding the dominant
Miller port, forcing the synthesizer to combine two weaker positions.
"""
import warnings
from pathlib import Path

import numpy as np
import pytest
import sympy as sp

from circuitinsight.adapters.spectre import SpectreRun
from circuitinsight.analysis.compensate import (
    Candidate, LoopGainUpdater, _dominant_pair, _margins_of, _numeric_A,
    _pair_incidence, _scaled_roots, locus_family, multi_locus,
    suggest_compensation, suggest_multi_compensation)
from circuitinsight.engine.mna import S, _det, build_mna

FIX = Path(__file__).resolve().parent / "fixtures" / "spectre" / "miller"

CLASSIC = Candidate("miller", "I0.net1", "vout", "classic Miller bridge", 106.0)


@pytest.fixture(scope="module")
def sys_uncomp():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run = SpectreRun(FIX / "tb_ota2s_stb.cin.json", FIX / "psf_stb")
        an = run.analyzer(cap_model="matrix")
        prims = [p for p in an.primitives if p.inst != "I0.Cc"]
        return build_mna(prims, an.flat.ground, "IPRB0", an._alias)


def _augmented_roots(system, pairs, vals):
    """Exact natural frequencies (Hz) by stamping each branch with an explicit
    internal RC node -- the independent ground truth for multi_locus."""
    A = _numeric_A(system)
    dim = A.rows
    extra = sum(1 for _, R in vals if R != 0)
    Aug = sp.zeros(dim + extra, dim + extra)
    Aug[:dim, :dim] = A
    m = dim
    for (na, nb), (C, R) in zip(pairs, vals):
        ia = system.node_index[na]
        ib = None if nb is None else system.node_index[nb]
        Cq = sp.Rational(repr(float(C)))
        if R == 0:
            Y = S * Cq
            Aug[ia, ia] += Y
            if ib is not None:
                Aug[ib, ib] += Y; Aug[ia, ib] -= Y; Aug[ib, ia] -= Y
        else:
            g = 1 / sp.Rational(repr(float(R)))
            Aug[ia, ia] += g; Aug[m, m] += g
            Aug[ia, m] -= g; Aug[m, ia] -= g
            Y = S * Cq
            Aug[m, m] += Y
            if ib is not None:
                Aug[ib, ib] += Y; Aug[m, ib] -= Y; Aug[ib, m] -= Y
            m += 1
    return _scaled_roots(sp.Poly(sp.expand(_det(Aug)), S)) / (2 * np.pi)


def _match(a, b):
    """Max relative mismatch after nearest-neighbour pairing of two root sets
    of equal length."""
    a = np.sort_complex(np.asarray(a))
    rem = list(np.sort_complex(np.asarray(b)))
    err = 0.0
    for r in a:
        j = min(range(len(rem)), key=lambda k: abs(rem[k] - r))
        err = max(err, abs(rem[j] - r) / (abs(rem[j]) + 1.0))
        rem.pop(j)
    return err


@pytest.mark.parametrize("vals", [
    [(8e-12, 0.0), (3e-12, 0.0)],            # two plain caps
    [(8e-12, 2.5e3), (3e-12, 0.0)],          # one series-RC, one plain
    [(8e-12, 2.5e3), (3e-12, 1.2e3)],        # two series-RC
])
def test_multi_locus_matches_the_internal_node_stamp(sys_uncomp, vals):
    """char = det(D I + Y-hat M)/D^(k-1) reproduces the explicit-internal-node
    natural frequencies exactly, across plain / mixed / all-RC branch sets."""
    pairs = [("I0.net1", "vout"), ("I0.tail", None)]
    got = multi_locus(sys_uncomp, pairs)(vals)
    gt = _augmented_roots(sys_uncomp, pairs, vals)
    assert len(got) == len(gt)
    assert _match(got, gt) < 1e-6


def test_multi_locus_reduces_to_locus_family_at_k1(sys_uncomp):
    """A single branch through multi_locus is the single-branch locus_family."""
    one = multi_locus(sys_uncomp, [("I0.net1", "vout")])([(6e-12, 2.0e3)])
    ref = locus_family(sys_uncomp, "I0.net1", "vout")(6e-12, 2.0e3)
    assert _match(one, ref) < 1e-9


def test_greedy_k1_reduces_to_single_suggester(sys_uncomp):
    """k_max=1 with the classic port must land on the SAME sized branch the
    single-branch suggester picks -- the multi path is a strict generalization."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = suggest_multi_compensation(sys_uncomp, "IPRB0", goal="pm",
                                       pm_target=60.0, k_max=1,
                                       candidates=[CLASSIC])
        s = suggest_compensation(sys_uncomp, "IPRB0", goal="pm",
                                 pm_target=60.0, candidates=[CLASSIC])[0]
    assert len(m.branches) == 1 and m.achieved
    b = m.branches[0]
    assert b.node_a == s.candidate.node_a and b.node_b == s.candidate.node_b
    assert b.C == pytest.approx(s.C, rel=1e-9)
    assert b.R == pytest.approx(s.R, rel=1e-9)
    assert m.pm_deg == pytest.approx(s.pm_deg, abs=1e-6)


def test_greedy_adds_no_branch_it_does_not_need(sys_uncomp):
    """With the dominant Miller port available, one branch meets PM 60 -- the
    synthesizer must stop at one, not spend area on a second."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = suggest_multi_compensation(sys_uncomp, "IPRB0", goal="pm",
                                       pm_target=60.0, k_max=2,
                                       candidates=[CLASSIC])
    assert m.achieved and len(m.branches) == 1


def test_greedy_grows_two_branches_and_the_two_engines_agree(sys_uncomp):
    """Excluding the dominant Miller port forces two weaker branches to
    combine. The result must be stable, meet the goal, and -- the real
    invariant -- the reported margins (from the Woodbury loop gain) and the
    reported dominant pair (from the multi-branch locus) must be reproduced
    by an INDEPENDENT recomputation over the installed set."""
    cset = [Candidate("shunt_rc", "I0.net1", None, "gain-node shunt", 1.0),
            Candidate("shunt_rc", "vout", None, "load-node shunt", 1.0),
            Candidate("miller", "I0.tail", "vout", "alt bridge", 10.0)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = suggest_multi_compensation(
            sys_uncomp, "IPRB0", goal="pm", pm_target=70.0, k_max=2,
            candidates=cset, c_grid=np.geomspace(0.5e-12, 30e-12, 24))
    assert m.achieved and len(m.branches) == 2 and m.pm_deg >= 70.0

    # independent recomputation of the joint loop gain over the installed set
    upd = LoopGainUpdater(sys_uncomp, "IPRB0", np.geomspace(1e4, 1e9, 240))
    Ybr = [(b.node_a, b.node_b,
            (lambda C, R: (lambda s: s * C / (1 + s * R * C)))(b.C, b.R))
           for b in m.branches]
    T = upd.with_branches(Ybr)
    pm, fu, gm = _margins_of(upd.freqs, T)
    assert pm == pytest.approx(m.pm_deg, abs=1e-6)

    # independent recomputation of the joint natural frequencies (all LHP)
    poles = multi_locus(sys_uncomp, [(b.node_a, b.node_b) for b in m.branches]
                        )([(b.C, b.R) for b in m.branches])
    assert np.all(poles.real < 1e-3)
    f0, zeta = _dominant_pair(poles)
    assert f0 == pytest.approx(m.f0_hz, rel=1e-9)
    assert zeta == pytest.approx(m.zeta, rel=1e-9)
