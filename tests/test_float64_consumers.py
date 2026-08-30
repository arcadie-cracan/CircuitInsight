"""The remaining consumers of big assembled rationals survive float64
cancellation (the audit that followed the margin-ledger fix): the
tearing reduction gate, the simplify base leg, the tolerance-margin
full-original leg, the GFT quartet curves and the servo-bandwidth DC
read all evaluate through the exact-coefficient reads. Mirrors
test_minimal_roots.py::test_factored_and_hp_reads_survive_float64_cancellation:
the poison term is IDENTICALLY ZERO, so any measured error against the
clean form is pure evaluation noise."""
import numpy as np
import sympy as sp

from circuitinsight.engine.mna import TransferFunction

S = sp.Symbol("s")
P1, P2, Z0 = (sp.Integer(10) ** 4, sp.Integer(10) ** 7,
              sp.Integer(10) ** 8)               # rad/s, exact
CLEAN = -1000 * (1 - S / Z0) / ((1 + S / P1) * (1 + S / P2))
_C = sp.Integer(10) ** 9 + 7
POISON = 100 * ((S + _C) * (S - _C) + _C * _C - S * S)
TRUTH = sp.lambdify(S, CLEAN, "numpy")


def _tf(expr) -> TransferFunction:
    return TransferFunction(expr=expr, values={}, symbols={})


def _assert_probative(tf):
    """float64 on the raw expression must still be garbage where |T| is
    small -- otherwise the fixture no longer probes anything."""
    f = np.array([3e6, 3e7])
    want = np.asarray(TRUTH(2j * np.pi * f), complex)
    got = np.asarray(tf.numeric(f))
    assert np.max(np.abs(got - want) / np.abs(want)) > 0.9


def test_reduction_gate_measures_zero_for_identical_forms():
    """The tearing budget gate: exact and approx differ by the poison
    only, i.e. by NOTHING -- the measured reduction error must be ~0 on
    either leg (float64 reported 248 dB here, wrong in both roles)."""
    from circuitinsight.analysis.tearing import _measured_error

    poisoned, clean = _tf(CLEAN + POISON), _tf(CLEAN)
    _assert_probative(poisoned)
    freqs = np.geomspace(10, 1e10, 60)
    for exact, approx in ((poisoned, clean), (clean, poisoned)):
        d_db, d_deg = _measured_error(exact, approx, freqs)
        assert d_db < 1e-9 and d_deg < 1e-9


def test_simplify_base_leg_is_high_precision():
    """simplify's reference curve: with a float64 base leg the eps=0
    rung compares the (clean, expanded) candidate against a poisoned
    reference and even the UNPRUNED form fails verification; the exact
    Horner base leg prices the pruning correctly."""
    tf = _tf(CLEAN + POISON)
    _assert_probative(tf)
    res = tf.simplify(mag_tol_db=1.0, phase_tol_deg=5.0,
                      fmin=10.0, fmax=1e10)
    assert res.achieved_mag_err_db < 1e-9
    assert res.achieved_phase_err_deg < 1e-9


def test_tolerance_margin_full_leg_is_high_precision():
    """The Kolka region probe re-evaluates the FULL original at pushed
    parameter values; that leg reads through the exact Horner, so a
    simplification that dropped nothing reports ~0 margin."""
    from circuitinsight.analysis.simplify import SimplifiedTF

    A = sp.Symbol("A")
    clean_a = -A * (1 - S / Z0) / ((1 + S / P1) * (1 + S / P2))
    orig = TransferFunction(expr=clean_a + POISON, values={"A": 1000.0},
                            symbols={"A": A})
    _assert_probative(orig)
    simp = SimplifiedTF(expr=clean_a, values={"A": 1000.0},
                        symbols={"A": A}, original=orig,
                        band_hz=(10.0, 1e10))
    assert simp.tolerance_margin(spread=0.2, top=1) < 1e-9


def test_gft_quartet_curves_are_high_precision():
    """discrepancy / spec_deviation / feedthrough_crossover are read
    exactly where |T| is near or below unity -- the region float64
    mangled. Every member carries the poison; the curves must match the
    clean truth."""
    from circuitinsight.analysis.gft import GftQuartet

    clean_h = sp.Rational(1, 2) * CLEAN / (1 + CLEAN)
    q = GftQuartet(H=_tf(clean_h + POISON),
                   Hinf=_tf(sp.Rational(1, 2) + POISON),
                   T=_tf(CLEAN + POISON),
                   Tn=_tf(CLEAN),
                   H0=_tf(sp.Rational(1, 100) + POISON),
                   error_ref="x", probe="p")
    _assert_probative(q.T)
    f = np.array([3e6, 3e7])
    t = np.asarray(TRUTH(2j * np.pi * f), complex)
    d = np.asarray(q.discrepancy(f))
    assert np.max(np.abs(d / (t / (1 + t)) - 1)) < 1e-9
    dev = np.asarray(q.spec_deviation(f))
    assert np.max(np.abs(dev / np.abs(1 / (1 + t)) - 1)) < 1e-9
    # crossover where |H0| > |Hinf*T|: reproduce the rule from truth
    grid = np.geomspace(1e3, 1e10, 121)
    tg = np.asarray(TRUTH(2j * np.pi * grid), complex)
    k = np.where(np.abs(0.01 / (1 + tg))
                 > np.abs(0.5 * tg / (1 + tg)))[0]
    assert q.feedthrough_crossover(grid) == float(grid[k[0]])


def test_servo_bandwidth_dc_read_is_high_precision():
    """The near-DC loop-gain read (measured safe in float64 at 9e-11
    relative on this fixture, but the exact point is free -- poles()
    pays for the machinery anyway): poisoned and clean loop gains give
    the identical MFM bandwidth."""
    from circuitinsight.analysis.compensate import servo_bandwidth

    poisoned, clean = _tf(CLEAN + POISON), _tf(CLEAN)
    _assert_probative(poisoned)
    f_p = servo_bandwidth(poisoned, 2)
    f_c = servo_bandwidth(clean, 2)
    assert abs(f_p / f_c - 1) < 1e-12
