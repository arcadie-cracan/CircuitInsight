"""M-F: Middlebrook's GFT quartet, exact on the MNA (analysis/gft.py).

The stb bench with the error designated as u_y = v(vin_p) - v(vout): the
ideal closed-loop gain must be exactly the balun half at every frequency,
the identity H = Hinf*T/(1+T) + H0/(1+T) must hold in EXACT rational
arithmetic, and the spec-level deviation |H/Hinf - 1| gives the
D-peaking/tolerance view Middlebrook argues is the real design target.
"""
import warnings
from pathlib import Path

import numpy as np
import pytest

from circuitinsight.adapters.spectre import SpectreRun

FIX = Path(__file__).resolve().parent / "fixtures" / "spectre" / "miller"


@pytest.fixture(scope="module")
def quartet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run = SpectreRun(FIX / "tb_ota2s_stb.cin.json", FIX / "psf_stb")
        an = run.analyzer(cap_model="matrix")
        return an.gft("IPRB0", "VIND", "vout", "vin_p")


def test_identity_holds_exactly(quartet):
    """H = Hinf*T/(1+T) + H0/(1+T) at rational s points, in exact rational
    arithmetic: residual must be identically zero. This is the self-check
    that caught two wrong null designations and a sign convention during
    development -- nothing inexact survives it."""
    assert quartet.identity_residual() == 0.0


def test_ideal_gain_is_flat_at_the_balun_half(quartet):
    """Hinf = the ideal unity-buffer gain through the balun (0.5), flat at
    ALL frequencies -- Middlebrook's hallmark of a correctly designated
    error null."""
    h = quartet.Hinf.numeric([1e1, 1e5, 1e9])
    assert np.allclose(np.abs(h), 0.5, rtol=1e-9)


def test_second_level_shapes(quartet):
    """Pinned magnitudes of the dissection on this bench."""
    db = lambda tfq, f: 20 * np.log10(abs(complex(tfq.numeric([f])[0])))
    assert db(quartet.T, 1e3) == pytest.approx(70.39, abs=0.05)
    assert db(quartet.Tn, 1e3) == pytest.approx(75.85, abs=0.05)
    assert db(quartet.H0, 1e3) == pytest.approx(-11.48, abs=0.05)
    # null loop gain stays healthy through crossover (Design Step 4)
    assert db(quartet.Tn, 1e7) > 30


def test_spec_deviation_is_the_design_view(quartet):
    """|H/Hinf - 1|: negligible in band, ~30% at 1 MHz, ~1 at the loop
    crossover -- the tolerance-limit view of stability."""
    dev = quartet.spec_deviation([1e3, 1e6, 3.5e6])
    assert dev[0] < 1e-3
    assert dev[1] == pytest.approx(0.297, abs=0.02)
    assert dev[2] == pytest.approx(1.00, abs=0.05)
    d = np.abs(quartet.discrepancy([1e3]))[0]
    assert d == pytest.approx(1.0, abs=1e-3)


def test_bad_designations_raise(quartet):
    from circuitinsight.engine.mna import MnaError

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run = SpectreRun(FIX / "tb_ota2s_stb.cin.json", FIX / "psf_stb")
        an = run.analyzer(cap_model="matrix")
        with pytest.raises(MnaError):
            an.gft("IB", "VIND", "vout", "vin_p")      # not a branch
        with pytest.raises(MnaError):
            an.gft("IPRB0", "VIND", "vout", "nope")    # unknown error ref
