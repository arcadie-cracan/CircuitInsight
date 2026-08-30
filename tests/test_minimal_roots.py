"""Minimal-form rooting (engine.mna.minimal_pz): assembly-artifact
pole/zero pairs cancel, genuine doublets survive, and the hybrid and
numeric solve routes agree on the SAME true roots -- ghost roots in
the ledger and sensitivity maps sent the designer investigating
nothing (field case, fc loop gain)."""
from pathlib import Path

import numpy as np
import pytest
import sympy as sp

FIX = Path(__file__).resolve().parent / "fixtures" / "spectre" / "miller"


def test_exact_cancellation_goes_genuine_doublet_stays():
    from circuitinsight.engine.mna import minimal_pz

    a, b, c = sp.Rational(100), sp.Rational(1000), sp.Rational(5000)
    d = b * sp.Rational(1005, 1000)              # 0.5% doublet partner
    s = sp.symbols("s")
    num = sp.Poly((s + a) * (s + c) * (s + d), s)
    den = sp.Poly((s + b) * (s + c) * (s + sp.Rational(70000)), s)
    zeros, poles = minimal_pz(num.all_coeffs(), den.all_coeffs())
    # the exact common factor (s + c) cancelled from BOTH
    assert len(zeros) == 2 and len(poles) == 2
    assert not any(abs(abs(z) - 5000) / 5000 < 1e-6 for z in zeros)
    assert not any(abs(abs(p) - 5000) / 5000 < 1e-6 for p in poles)
    # the 0.5% doublet SURVIVED on both sides, at true positions
    assert any(abs(z + float(d)) / float(d) < 1e-10 for z in zeros)
    assert any(abs(p + float(b)) / float(b) < 1e-10 for p in poles)


def test_roots_at_the_origin_survive():
    from circuitinsight.engine.mna import minimal_pz

    s = sp.symbols("s")
    num = sp.Poly(s * (s + 10), s)               # zero at s = 0
    den = sp.Poly((s + 3) * (s + 7), s)
    zeros, poles = minimal_pz(num.all_coeffs(), den.all_coeffs())
    assert any(abs(z) < 1e-12 for z in zeros)
    assert len(zeros) == 2 and len(poles) == 2


def _poisoned_loop_gain():
    """A clean two-pole one-RHP-zero loop gain plus a term that is
    IDENTICALLY ZERO -- but whose float64 evaluation cancels
    catastrophically at high frequency (the folded-cascode field
    case: numeric() read up to 25 dB / 179 deg wrong wherever |T|
    was small, and the ledger anchor absorbed the error)."""
    from circuitinsight.engine.mna import TransferFunction

    s = sp.Symbol("s")
    p1, p2, z0 = (sp.Integer(10) ** 4, sp.Integer(10) ** 7,
                  sp.Integer(10) ** 8)          # rad/s, exact
    clean = -1000 * (1 - s / z0) / ((1 + s / p1) * (1 + s / p2))
    c = sp.Integer(10) ** 9 + 7
    poison = 100 * ((s + c) * (s - c) + c * c - s * s)
    tf = TransferFunction(expr=clean + poison, values={}, symbols={})
    truth = sp.lambdify(s, clean, "numpy")
    return tf, truth


def test_factored_and_hp_reads_survive_float64_cancellation():
    tf, truth = _poisoned_loop_gain()
    f = np.array([3e6, 3e7])                     # where |T| is small
    want = np.asarray(truth(2j * np.pi * f), complex)
    # float64 on the raw expression is garbage here -- that is the
    # premise of the test; if numpy ever evaluates this exactly the
    # assertion below flags the fixture as no longer probative
    got_f64 = np.asarray(tf.numeric(f))
    assert np.max(np.abs(got_f64 - want) / np.abs(want)) > 0.9
    for read in (tf.numeric_hp, tf.numeric_min):
        got = np.asarray(read(f))
        assert np.max(np.abs(got / want - 1)) < 1e-6, read.__name__


def test_tf_pickle_sheds_derived_caches():
    """Results ship a tf across the engine pipe; the exact-coefficient
    and rooted-spectrum caches can dwarf the expression itself and are
    recomputable, so pickling must strip them."""
    import pickle

    tf, truth = _poisoned_loop_gain()
    tf.poles()
    tf.numeric_min([1e6])                        # warm every cache
    assert "num_den" in tf.__dict__ and "_minimal_pz" in tf.__dict__
    tf2 = pickle.loads(pickle.dumps(tf))
    for k in ("num_den", "_exact_coeffs", "_minimal_pz",
              "_minimal_exact"):
        assert k not in tf2.__dict__
    f = np.array([3e6])
    want = np.asarray(truth(2j * np.pi * f), complex)
    got = np.asarray(tf2.numeric_min(f))         # recomputes cleanly
    assert np.max(np.abs(got / want - 1)) < 1e-6


def test_margin_ledger_anchor_is_high_precision():
    """The certificate residuals stay tiny on the poisoned loop gain:
    the anchor and residual reads no longer trust float64."""
    from circuitinsight.analysis.marginledger import margin_ledger

    tf, _ = _poisoned_loop_gain()
    led = margin_ledger(tf)
    assert led.f_unity_hz is not None and led.f_180_hz is not None
    assert led.phase_residual_deg < 1e-3
    assert led.mag_residual_db < 1e-3


@pytest.mark.parametrize("keep_n", [0, 3])
def test_loop_gain_routes_agree_and_are_ghost_free(keep_n):
    """The acceptance for the field report: numeric and hybrid solves
    now expose the SAME minimal roots, and no surviving zero sits
    within 1e-6 of a pole (a ghost pair would)."""
    from circuitinsight.session import SessionController

    s = SessionController.open(str(FIX / "tb_ota2s_stb.cin.json"),
                               str(FIX / "psf_stb"))
    probe = s.stb_probe() or s.probes[0]
    keep = []
    if keep_n:
        an = s._analyzer_ready()
        keep = [str(x) for x in list(an.system(probe).symbols)[:keep_n]]
    r = s.loop_gain(probe, keep=keep)
    poles = np.asarray(r.poles_hz)
    zeros = np.asarray(r.zeros_hz)
    assert len(poles) == 5                       # the minimal count
    for z in zeros:
        for p in poles:
            assert abs(z - p) / max(abs(p), 1.0) > 1e-6, "ghost pair"
    # the numeric route pins the truth; the hybrid must match it
    r0 = s.loop_gain(probe, keep=[])
    ref = sorted(np.asarray(r0.poles_hz), key=abs)
    got = sorted(poles, key=abs)
    for a, b in zip(ref, got):
        assert abs(a - b) / max(abs(a), 1.0) < 1e-8