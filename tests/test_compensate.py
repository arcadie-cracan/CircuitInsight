"""Compensation-candidate screening (N-A): rank-one cap loci + pole
sensitivities on the uncompensated SKY130 two-stage.

The validation spine: the uncompensated bench + a VIRTUAL 16 pF cap across
the Miller pair (I0.net1, vout), computed by the rank-one locus with NO
re-solve, must reproduce (a) a direct solve with the capacitor actually
installed in the primitive list, and (b) the poles of the real plain-Miller
fixture -- an independent Spectre run whose operating point matches to
~1e-13 (the OP-invariance this whole feature is built on).
"""
import warnings
from pathlib import Path

import numpy as np
import pytest

from circuitinsight.adapters.spectre import SpectreRun
from circuitinsight.analysis.compensate import (butterworth_targets,
                                                cap_locus, cap_pole_screen,
                                                rc_locus, servo_bandwidth)

FIX = Path(__file__).resolve().parent / "fixtures" / "spectre"
PAIR = ("I0.net1", "vout")              # the Miller pair (2nd-stage gate, out)


def _sets_close(a, b, k, rtol, atol=0.0):
    """Greedy nearest-matching of the k dominant roots of two lists --
    immune to conjugate-pair ordering and ULP-level |.| ties."""
    a = np.asarray(a)[np.argsort(np.abs(np.asarray(a)))][:k]
    b = list(np.asarray(b)[np.argsort(np.abs(np.asarray(b)))][:k])
    for x in a:
        i = int(np.argmin(np.abs(np.array(b) - x)))
        if abs(b[i] - x) > rtol * abs(x) + atol:
            return False
        b.pop(i)
    return True


@pytest.fixture(scope="module")
def unc_system():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run = SpectreRun(FIX / "miller_unc" / "tb_ota2s.cin.json",
                         FIX / "miller_unc" / "psf")
        an = run.analyzer(cap_model="matrix")
        return an, an.system("VIND")


def test_cap_locus_matches_direct_install(unc_system):
    """Rank-one locus vs actually installing the capacitor primitive."""
    from circuitinsight.engine.mna import build_mna, solve_tf
    from circuitinsight.engine.primitives import Primitive

    an, sys_ = unc_system
    (locus,) = cap_locus(sys_, *PAIR, [16e-12])

    prims = list(an.primitives) + [Primitive("Ccand", "", "c", PAIR, 16e-12)]
    sys2 = build_mna(prims, an.flat.ground, "VIND", an._alias)
    direct = solve_tf(sys2, "vout", []).poles()

    assert _sets_close(locus, direct, 6, rtol=1e-9)


def test_cap_locus_reproduces_the_real_miller_fixture(unc_system):
    """Cross-fixture: the virtual 16 pF on the uncompensated bench must land
    on the plain-Miller fixture's actual poles (independent Spectre run;
    OPs agree to ~1e-13 because the cap carries no DC)."""
    an, sys_ = unc_system
    (locus,) = cap_locus(sys_, *PAIR, [16e-12])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run_m = SpectreRun(FIX / "miller" / "tb_ota2s.cin.json",
                           FIX / "miller" / "psf")
        pm = run_m.analyzer(cap_model="matrix").tf(
            "VIND", "vout", keep=[]).poles()
    assert _sets_close(locus, pm, 4, rtol=1e-4, atol=1.0)
    la = np.asarray(locus)[np.argsort(np.abs(np.asarray(locus)))][:4]
    # the canonical signature: dominant pole driven to ~178.8 Hz and a split
    # upper pair at ~-29 +/- 2.6j MHz
    assert abs(la[0]) == pytest.approx(178.8, rel=1e-3)
    assert la[1].real == pytest.approx(-2.899e7, rel=0.01)
    assert abs(la[1].imag) == pytest.approx(2.56e6, rel=0.05)


def test_screen_discovers_the_miller_pair(unc_system):
    """The O(N^2) screen must rank the classic Miller position at the top
    for the dominant pole pair, with the split signature: a unit cap moves
    p1 DOWN in |.| and p2 UP."""
    an, sys_ = unc_system
    screens = cap_pole_screen(sys_, n_poles=2)
    p1, p2 = screens

    def dmag(scr, pair):
        dp = scr.dpdc[pair]
        p = scr.pole_hz * 2 * np.pi
        return (np.conj(p) * dp).real / abs(p)

    # the Miller pair is the strongest mover of the dominant pole
    top_p1 = p1.ranked(3)
    assert PAIR in [k for k, _ in top_p1] or (PAIR[1], PAIR[0]) in \
        [k for k, _ in top_p1]
    # and it splits: |p1| decreases, |p2| increases
    assert dmag(p1, PAIR) < 0
    assert dmag(p2, PAIR) > 0


def test_screen_consistent_with_locus(unc_system):
    """dp/dC from the null-vector formula must match a finite difference of
    the exact locus at small C."""
    an, sys_ = unc_system
    screens = cap_pole_screen(sys_, n_poles=2)
    dC = 1e-16      # small enough to sit before the locus curvature
    (l0,), (l1,) = cap_locus(sys_, *PAIR, [0.0]), cap_locus(sys_, *PAIR, [dC])
    for scr in screens:
        p0 = scr.pole_hz
        k0 = np.argmin(np.abs(l0 - p0))
        k1 = np.argmin(np.abs(l1 - p0))
        fd = (l1[k1] - l0[k0]) * 2 * np.pi / dC     # rad/s per F
        pred = scr.dpdc[PAIR]
        assert abs(fd - pred) / abs(pred) < 0.02


def test_rc_locus_places_the_zero_pole(unc_system):
    """A series-RC branch adds one natural frequency (the branch's own
    1/RC-scale pole) and reshapes the split -- sanity: the locus stays LHP
    and gains exactly one root vs the pure-cap locus."""
    an, sys_ = unc_system
    (lc,) = cap_locus(sys_, *PAIR, [3e-12])
    (lrc,) = rc_locus(sys_, *PAIR, 3e-12, [2.5e3])
    assert len(lrc) == len(lc) + 1
    assert np.all(lrc.real < 1e-3)                  # no RHP natural freqs


def test_servo_bandwidth_and_targets():
    """The Delft-style budget on the compensated closed-loop bench: the
    midband loop gain x dominant loop poles give the attainable MFM
    bandwidth; sanity-pin the n=2 value and the Butterworth geometry."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run = SpectreRun(FIX / "miller" / "tb_ota2s_stb.cin.json",
                         FIX / "miller" / "psf_stb")
        T = run.analyzer(cap_model="matrix").loop_gain("IPRB0")
    f_h = servo_bandwidth(T, 2)
    assert 5e6 < f_h < 2e7                          # ~10 MHz scale
    tgt = butterworth_targets(f_h, 2)
    assert np.allclose(np.abs(tgt), f_h)
    assert np.allclose(sorted(np.degrees(np.angle(tgt))), [-135, 135])
