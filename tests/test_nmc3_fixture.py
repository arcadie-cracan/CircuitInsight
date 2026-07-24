"""nmc3: behavioral gm-C THREE-STAGE nested-Miller amplifier -- the
three-loop nesting fixture.

A PDK-free ideal-transconductor amplifier with two nested Miller caps and
three nested loops (outer feedback IPRB0, outer Miller IPRB2/Cm2, inner
Miller IPRB1/Cm1). Being linear, its reconstruction is EXACT against Spectre
stb (no quasi-static model error), which makes it the clean bench for:

  * loop_gain == stb to machine precision at every probe;
  * the depth-3 nested GFT (DeepNestedGft): the dissection identity exact at
    all three nesting levels, the ideal follower gain 1 preserved as each
    loop is idealized in turn;
  * the multi-branch (N-E) synthesizer rediscovering the nested-Miller pair
    on the uncompensated amplifier.

Ground truth captured on <sim-host> (Spectre IC25.1): outer-loop PM 82.9513 deg
@ 2.6513 MHz, GM 18.73 dB @ 28.47 MHz.
"""
import warnings
from pathlib import Path

import numpy as np
import pytest

from circuitinsight.adapters.spectre import SpectreRun
from circuitinsight.analysis.compensate import (Candidate,
                                                suggest_multi_compensation)
from circuitinsight.engine.mna import build_mna

FIX = Path(__file__).resolve().parent / "fixtures" / "spectre" / "nmc3"

# nesting order: outer feedback (IPRB0), outer Miller Cm2 (IPRB2), inner
# Miller Cm1 (IPRB1); probe-aligned follower errors (all give Hinf = 1)
PROBES = ["IPRB0", "IPRB2", "IPRB1"]
ERRORS = [("inp", -1), ("out", -1), ("out", -1)]
PSF = {"IPRB0": "psf", "IPRB2": "psf_cm2", "IPRB1": "psf_cm1"}


def _run(sub="psf"):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SpectreRun(FIX / "tb_nmc3.cin.json", FIX / sub)


@pytest.mark.parametrize("probe", PROBES)
def test_loop_gain_is_exact_against_stb(probe):
    """The circuit is linear, so the reconstructed loop gain equals Spectre's
    stb loopGain to machine precision at EVERY nested-loop probe -- 0 dB /
    0 deg, no quasi-static residual."""
    run = _run(PSF[probe])
    stb = run.stb()
    T = run.analyzer().loop_gain(probe, []).numeric(stb.freq)
    Ts = np.asarray(stb.loop_gain)
    band = np.abs(Ts) > 0.1                     # |T| > -20 dB
    assert band.sum() > 40
    dberr = np.max(np.abs(20 * np.log10(np.abs(T[band]))
                          - 20 * np.log10(np.abs(Ts[band]))))
    pherr = np.max(np.abs(np.angle(T[band], deg=True)
                          - np.angle(Ts[band], deg=True)))
    assert dberr < 1e-4 and pherr < 1e-4


def test_outer_margin_matches_spectre():
    """The designed outer-loop margins reproduce Spectre stb."""
    run = _run("psf")
    stb = run.stb()
    assert stb.phase_margin_deg == pytest.approx(82.9513, abs=0.02)
    T = run.analyzer().loop_gain("IPRB0", [])
    f = np.geomspace(1e2, 1e10, 4000)
    Tv = T.numeric(f)
    m = 20 * np.log10(np.abs(Tv))
    ph = np.degrees(np.unwrap(np.angle(Tv)))
    x = np.log10(f)
    k = np.where(np.diff(np.sign(m)))[0][0]
    xu = np.interp(0, [m[k + 1], m[k]], [x[k + 1], x[k]])
    pm = np.interp(xu, [x[k], x[k + 1]], [ph[k], ph[k + 1]])
    assert pm == pytest.approx(82.9513, abs=0.05)


def test_depth3_nested_gft_is_exact():
    """The three-level nested dissection H -> Hinf1 -> Hinf12 -> Hinf123 is
    exact at every level (identity residual identically 0 in rational
    arithmetic), and the ideal follower gain is 1 at each level -- idealizing
    any nested loop leaves the unity-follower ideal unchanged."""
    an = _run("psf").analyzer()
    g = an.nested_gft_deep(PROBES, "VIN", "out", ERRORS)
    assert g.depth == 3
    f = np.array([1e4])
    for k in (1, 2, 3):
        assert g.identity_residual(k) == 0.0
        assert complex(g.level(k, f)["Hinf"][0]) == pytest.approx(1.0, abs=1e-9)


def test_depth3_first_two_levels_match_the_depth2_dissection():
    """DeepNestedGft is a strict generalization: its levels 1-2, on the same
    two designations, reproduce the two-level NestedGft T's exactly."""
    an = _run("psf").analyzer()
    deep = an.nested_gft_deep(PROBES[:2], "VIN", "out", ERRORS[:2])
    two = an.nested_gft("IPRB0", "IPRB2", "VIN", "out", ERRORS[0], ERRORS[1])
    f = np.geomspace(1e3, 1e9, 60)
    assert np.allclose(deep.level(1, f)["T"], two.level1(f)["T"], rtol=1e-9)
    assert np.allclose(deep.level(2, f)["T"], two.level2(f)["T"], rtol=1e-9)


def test_nested_loops_are_genuinely_coupled():
    """The inner Miller loop's gain shifts when the outer loops are idealized
    (nonzero coupling) -- a real 3-loop nesting, not three independent loops.
    The coupling diagnostic is exact (ratio of two exact loop gains)."""
    an = _run("psf").analyzer()
    g = an.nested_gft_deep(PROBES, "VIN", "out", ERRORS)
    c = g.coupling(3, np.geomspace(1e5, 1e8, 40))
    assert np.max(c) > 1e-3


def test_multibranch_rediscovers_the_nested_miller_pair():
    """On the UNCOMPENSATED amplifier (both Miller caps stripped -- unstable),
    an aggressive peak-sensitivity spec (Ms <= 1.1) cannot be met by one
    branch: the synthesizer grows the nested pair, placing a cap at BOTH
    Miller ports (out-n2 inner, out-n1 outer) and stabilizing it."""
    an = _run("psf").analyzer()
    strip = {"CM1", "CM2", "IPRB1", "IPRB2"}
    prims = [p for p in an.primitives if p.inst not in strip]
    sysu = build_mna(prims, an.flat.ground, "IPRB0", an._alias)
    cands = [Candidate("miller", "out", "n1", "outer Miller (stages 2-3)", 50.0),
             Candidate("miller", "out", "n2", "inner Miller (stage 3)", 10.0)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = suggest_multi_compensation(
            sysu, "IPRB0", goal="spec", ms_target=1.1, k_max=2,
            candidates=cands, c_grid=np.geomspace(0.5e-12, 20e-12, 26))
    assert m.achieved and len(m.branches) == 2
    ports = {(b.node_a, b.node_b) for b in m.branches}
    assert ports == {("out", "n1"), ("out", "n2")}
    assert m.spec_dev <= 1.1
