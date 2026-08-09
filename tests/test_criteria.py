"""The BandCriterion family: one object per tolerance contract.

Fixture-free on purpose — these run in the public snapshot's gate. The
end-to-end behavior (pursuit + report) is covered by test_sensitivity,
test_session and test_reduce; here the criterion's own math and mapping
are pinned so a future strategy cannot drift from the contract."""
import math

import numpy as np
import pytest

from circuitinsight.analysis.criteria import (AnchoredCriterion,
                                              LegacyCriterion,
                                              PlainCriterion,
                                              RejectionCriterion,
                                              StabilityCriterion,
                                              make_criterion)


def test_factory_precedence_mirrors_the_historical_dispatch():
    assert isinstance(make_criterion(strategy="plain"), PlainCriterion)
    assert isinstance(make_criterion(strategy="stability", eps=0.05),
                      StabilityCriterion)      # a strategy wins over eps
    assert isinstance(make_criterion(eps=0.05), AnchoredCriterion)
    assert isinstance(make_criterion(tol_db=1.0), LegacyCriterion)
    with pytest.raises(ValueError):
        make_criterion(strategy="vibes")


def test_stopping_rules_and_caps():
    for s in ("plain", "stability", "rejection"):
        c = make_criterion(strategy=s)
        assert c.tol == 1.0 and c.cap == 6 and c.unit == ""
    c = make_criterion(strategy="plain", strategy_opts={"cap": 4})
    assert c.cap == 4
    a = make_criterion(eps=0.07)
    assert a.tol == 0.07 and a.cap is None
    l = make_criterion(tol_db=0.5)
    assert l.tol == 0.5 and l.cap is None and l.unit == " dB"


def test_eps_equivalents_and_collapse_budgets():
    p = make_criterion(strategy="plain", strategy_opts={"gain_db": 1.0,
                                                        "phase_deg": 5.0})
    assert p.eps_equivalent() == pytest.approx(10 ** (1 / 20) - 1)
    assert p.collapse_budgets() == (1.0, 5.0)
    s = make_criterion(strategy="stability", strategy_opts={"pm_deg": 7.0})
    assert s.eps_equivalent() == 0.05
    assert s.collapse_budgets() == (1.0, 7.0)
    rj = make_criterion(strategy="rejection", strategy_opts={"rej_db": 3.0})
    assert rj.eps_equivalent() == pytest.approx(10 ** (3 / 20) - 1)
    assert rj.collapse_budgets() == (3.0, 30.0)
    a = make_criterion(eps=0.05)
    assert a.collapse_budgets() == pytest.approx(
        (20 * math.log10(1.05), math.degrees(0.05)))


def test_score_fields_units():
    """band_score carries the normalized score with its unit named;
    mag_err_db is ALWAYS a genuine dB figure."""
    p = make_criterion(strategy="plain", strategy_opts={"gain_db": 2.0})
    assert p.score_fields(0.5, 9.9) == (0.5, "x budget", 1.0)
    s = make_criterion(strategy="stability")
    assert s.score_fields(0.5, 0.123) == (0.5, "x budget", 0.123)
    a = make_criterion(eps=0.05)
    score, unit, db = a.score_fields(0.05, 9.9)
    assert (score, unit) == (0.05, "fraction")
    assert db == pytest.approx(20 * math.log10(1.05))
    l = make_criterion(tol_db=1.0)
    assert l.score_fields(0.42, 9.9) == (None, "", 0.42)


def _flat(n=64, mag=100.0):
    freqs = np.logspace(3, 7, n)
    H = np.full(n, mag, dtype=complex)
    m = np.ones(n, dtype=bool)
    return freqs, H, m


def test_plain_error_is_the_worse_of_the_two_axes():
    freqs, H, m = _flat()
    c = PlainCriterion(gain_db=1.0, phase_deg=5.0)
    sig = c.window(freqs, np.abs(H), m, H)
    Hr = H * 10 ** (2.0 / 20)                 # +2 dB, no phase error
    assert c.error(freqs, H, np.abs(H), Hr, sig) == pytest.approx(2.0)
    Hr = H * np.exp(1j * np.radians(10.0))    # 10 deg, no mag error
    assert c.error(freqs, H, np.abs(H), Hr, sig) == pytest.approx(2.0)


def test_rejection_ignores_phase():
    freqs, H, m = _flat()
    c = RejectionCriterion(rej_db=3.0)
    sig = c.window(freqs, np.abs(H), m, H)
    Hr = H * np.exp(1j * np.radians(90.0))    # phase-only deviation
    assert c.error(freqs, H, np.abs(H), Hr, sig) == pytest.approx(0.0)


def test_anchored_window_captures_the_band_edge_anchor():
    freqs, H, m = _flat()
    H = H * np.linspace(1.0, 0.1, H.size)     # falling response
    c = AnchoredCriterion(eps=0.05)
    sig = c.window(freqs, np.abs(H), m, H)
    assert sig.all()
    assert c.anchor == pytest.approx(np.abs(H)[-1])   # smaller edge
    # the criterion formula itself: |dH| / (|H| + anchor)
    Hr = H + 0.5 * c.anchor
    e = c.error(freqs, H, np.abs(H), Hr, sig)
    assert e == pytest.approx(
        (0.5 * c.anchor / (np.abs(H) + c.anchor)).max())


def test_legacy_window_enforces_above_the_relative_floor():
    freqs, H, m = _flat()
    H = H * np.concatenate([np.ones(32), np.full(32, 1e-4)])  # -80 dB tail
    c = LegacyCriterion(tol_db=1.0, floor_db=60.0)
    sig = c.window(freqs, np.abs(H), m, H)
    assert sig[:32].all() and not sig[32:].any()
