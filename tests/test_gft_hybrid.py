"""Hybrid-symbolic GFT: the quartet kept symbolic in a chosen keep-set.

The scalar GFT (test_gft.py) dissects H numerically at the OP. Here the same
dissection keeps designated device parameters symbolic across ALL FIVE
members -- routed through the same multilinear solver tf()/loop_gain() use --
so the quartet becomes a closed form in the kept parameters (the paper's
compensation-inversion story, generalized from H(s) to the whole
dissection). Ground truth is the numeric quartet: substituting the kept
symbols back to their OP values must reproduce it exactly, and the GFT
identity self-check must still be zero.
"""
import warnings
from pathlib import Path

import numpy as np
import pytest
import sympy as sp

from circuitinsight.adapters.spectre import SpectreRun
from circuitinsight.engine.mna import S

FIX = Path(__file__).resolve().parent / "fixtures" / "spectre" / "miller"


@pytest.fixture(scope="module")
def bench():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run = SpectreRun(FIX / "tb_ota2s_stb.cin.json", FIX / "psf_stb")
        an = run.analyzer(cap_model="matrix")
        q0 = an.gft("IPRB0", "VIND", "vout", "vin_p")            # numeric
        gms = sorted(n for n in an.system("IPRB0").symbols
                     if n.startswith("gm_"))
        qh = an.gft("IPRB0", "VIND", "vout", "vin_p", keep=[gms[0]])
        return q0, qh, gms


def test_hybrid_reproduces_the_numeric_quartet(bench):
    """Every member, evaluated over a wide sweep, matches the keep=[] quartet
    to machine precision -- the hybrid solves are exact, only kept symbolic."""
    q0, qh, _ = bench
    f = np.array([1e2, 1e3, 1e5, 1e6, 3.5e6, 1e8])
    for m in ("H", "Hinf", "T", "Tn", "H0"):
        a = getattr(q0, m).numeric(f)
        b = getattr(qh, m).numeric(f)
        assert np.allclose(a, b, rtol=1e-9, atol=0), m


def test_identity_still_holds_exactly(bench):
    """The GFT identity is algebraic in the matrix entries, so it survives
    the keep-set; the self-check collapses the kept symbols to the OP and is
    exactly zero, just as for keep=[]."""
    _, qh, _ = bench
    assert qh.identity_residual() == 0.0


def test_kept_symbol_is_carried_symbolically(bench):
    """The kept gm appears in the loop-dependent members (H, T, Tn, H0). It
    is ABSENT from Hinf -- correctly: Hinf is the ideal gain (loop -> inf),
    here the exact buffer 0.5, independent of any forward transconductance.
    That clean split is the GFT's whole point, now visible symbolically."""
    _, qh, gms = bench
    x = qh.T.symbols[gms[0]]
    assert x in qh.T.expr.free_symbols
    assert x in qh.H.expr.free_symbols
    assert x in qh.Tn.expr.free_symbols
    assert x in qh.H0.expr.free_symbols
    # Hinf is the loop-independent ideal: no forward gm in it
    assert x not in qh.Hinf.expr.free_symbols
    assert complex(qh.Hinf.numeric([1e3])[0]) == pytest.approx(0.5, abs=1e-9)


def test_kept_form_is_a_live_what_if(bench):
    """Because T(s) is closed-form in the kept gm, halving it (a what-if the
    OP cannot show) drops the loop gain -- less forward transconductance,
    less loop gain -- with the rest of the circuit still exact at the OP."""
    _, qh, gms = bench
    x = qh.T.symbols[gms[0]]
    val = qh.T.values[gms[0]]
    f = sp.lambdify(S, qh.T.expr.subs(x, sp.Float(val * 0.5)), "numpy")
    t_half = abs(complex(f(2j * np.pi * 1e3)))
    t_full = abs(complex(qh.T.numeric([1e3])[0]))
    assert t_half < t_full


def test_two_symbol_keep_exercises_the_grid(bench):
    """A two-symbol keep drives the multilinear grid in two variables; both
    survive in T and the numeric reduction is still exact."""
    q0, _, gms = bench
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run = SpectreRun(FIX / "tb_ota2s_stb.cin.json", FIX / "psf_stb")
        an = run.analyzer(cap_model="matrix")
        q2 = an.gft("IPRB0", "VIND", "vout", "vin_p", keep=gms[:2])
    for n in gms[:2]:
        assert q2.T.symbols[n] in q2.T.expr.free_symbols
    f = np.array([1e3, 1e6, 3.5e6])
    assert np.allclose(q2.T.numeric(f), q0.T.numeric(f), rtol=1e-9, atol=0)
    assert q2.identity_residual() == 0.0
