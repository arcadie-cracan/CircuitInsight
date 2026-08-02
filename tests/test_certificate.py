"""Order certificate: Loewner rank over a band + doublet caveat."""
import numpy as np
import pytest

from circuitinsight.analysis.certificate import order_certificate


def _rational(f, poles_hz, zeros_hz, k=1.0):
    s = 2j * np.pi * f
    H = np.full(f.shape, k, dtype=complex)
    for z in zeros_hz:
        H *= 1 + s / (2 * np.pi * z)
    for p in poles_hz:
        H /= 1 + s / (2 * np.pi * p)
    return H


def test_certificate_counts_the_orders_a_band_can_see():
    """A synthetic 3-pole lowpass: within a band covering two poles the
    certificate demands ~2 at a tight tolerance and less at a loose one;
    the third pole far above the band never counts."""
    f = np.logspace(1, 7, 400)
    H = _rational(f, poles_hz=[1e3, 1e5, 1e9], zeros_hz=[], k=1e3)
    c = order_certificate(f, H, 1e1, 1e7)
    assert c.shape == "lowpass"
    assert c.order_at(0.01) in (2, 3)
    assert c.order_at(0.30) <= c.order_at(0.01)
    # sv must collapse after the visible orders
    assert c.sv[5] < 1e-3


def test_certificate_names_the_doublet():
    """A near-cancelling pole/zero pair inside the band is reported with
    its separation — the settling caveat's raw material."""
    f = np.logspace(1, 7, 400)
    H = _rational(f, poles_hz=[1e3, 1.0e5], zeros_hz=[1.05e5], k=1e3)
    c = order_certificate(
        f, H, 1e1, 1e7,
        poles_hz=np.array([1e3, 1.0e5], dtype=complex),
        zeros_hz=np.array([1.05e5], dtype=complex))
    assert c.doublets, "the 5% pole/zero pair must be named"
    fp, fz, sep = c.doublets[0]
    assert fp == pytest.approx(1.0e5, rel=1e-6)
    assert sep == pytest.approx(0.05, abs=0.01)
    assert "doublet" in c.describe(0.05)


def test_certificate_classifies_the_shapes():
    f = np.logspace(1, 7, 400)
    lp = _rational(f, poles_hz=[1e4], zeros_hz=[])
    hp = _rational(f, poles_hz=[1e4], zeros_hz=[1e0])
    bp = _rational(f, poles_hz=[1e3, 1e5], zeros_hz=[1e0])
    assert order_certificate(f, lp, 1e1, 1e7).shape == "lowpass"
    assert order_certificate(f, hp, 1e1, 1e7).shape == "highpass"
    assert order_certificate(f, bp, 1e1, 1e7).shape == "bandpass"
    assert order_certificate(f, lp, 1e1, 1e3).shape == "flat"
