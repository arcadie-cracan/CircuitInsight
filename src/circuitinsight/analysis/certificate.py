"""Order certificate: how much dynamical order a response needs over a
band, at a tolerance — before any reduction runs.

Data-driven, from the band samples every solve already computes. The
numerical rank of the Loewner pencil built from the samples bounds the
minimal McMillan degree that can reproduce the band data (Mayo-Antoulas);
weighting the samples by the anchored criterion 1/(|H| + anchor) makes
the rank speak the same error language as the reduction knob: the number
of singular values above eps is the order the band demands at eps.

This is a practical certificate, not a sharp theorem: the exact
statement (AAK) bounds the Hankel-norm error by singular values of the
Hankel operator; the Loewner numerical rank is its band-limited,
sampled counterpart and is standard practice in rational approximation.
Validated against the pursuit on the benches (tests) rather than
trusted axiomatically.

The certificate also names DOUBLETS: near-cancelling pole/zero pairs of
the full model inside the band. A response certified low-order through
cancellation is frequency-faithful and transient-blind — doublets
govern settling, which no Bode-band criterion sees. The GUI must ship
that caveat with the order, or the certificate misleads.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

__all__ = ["OrderCertificate", "order_certificate"]


@dataclass
class OrderCertificate:
    fmin: float
    fmax: float
    anchor: float                 # min band-edge |H| (the criterion's)
    sv: np.ndarray                # normalized Loewner singular values
    shape: str                    # 'lowpass' | 'highpass' | 'bandpass' | 'flat'
    doublets: list = field(default_factory=list)   # [(f_pole, f_zero, sep)]

    def order_at(self, eps: float) -> int:
        """Minimal order the band demands at tolerance eps."""
        return int(np.sum(self.sv > eps))

    def describe(self, eps: float) -> str:
        from ..units import eng

        k = self.order_at(eps)
        bits = [f"{self.shape} band {eng(self.fmin, 'Hz')}–"
                f"{eng(self.fmax, 'Hz')} needs order {k} at {eps:.0%}"]
        if self.doublets:
            worst = min(self.doublets, key=lambda d: d[2])
            bits.append(f"note: pole/zero doublet near "
                        f"{eng(worst[0], 'Hz')} (sep {worst[2]:.1%}) — "
                        f"doublets affect settling, not this response")
        return "; ".join(bits)


def _classify(mag_band: np.ndarray) -> str:
    """lowpass / highpass / bandpass / flat from edge levels vs peak."""
    lo, hi, pk = mag_band[0], mag_band[-1], mag_band.max()
    near = 10.0 ** (3.0 / 20.0)              # within 3 dB counts as "at"
    lo_at = pk / lo < near
    hi_at = pk / hi < near
    if lo_at and hi_at:
        return "flat"
    if lo_at:
        return "lowpass"
    if hi_at:
        return "highpass"
    return "bandpass"


def order_certificate(freqs, H, fmin: float, fmax: float,
                      poles_hz=None, zeros_hz=None,
                      doublet_sep: float = 0.25) -> OrderCertificate:
    """Certificate from band samples (freqs, H) restricted to
    [fmin, fmax]; poles/zeros of the full model (complex, Hz) enable the
    doublet note. doublet_sep: |p−z|/|p| below this counts as a doublet."""
    f = np.asarray(freqs, dtype=float)
    Hv = np.asarray(H)
    m = (f >= fmin) & (f <= fmax) & np.isfinite(Hv) & (np.abs(Hv) > 0)
    if m.sum() < 8:
        raise ValueError("order_certificate: too few band samples")
    f, Hv = f[m], Hv[m]
    if len(f) > 400:                  # SVD stays interactive (~10 ms)
        pick = np.linspace(0, len(f) - 1, 400).astype(int)
        f, Hv = f[pick], Hv[pick]
    mag = np.abs(Hv)
    anchor = float(min(mag[0], mag[-1]))
    w = 1.0 / (mag + anchor)
    s = 2j * np.pi * f

    li = np.arange(0, len(f), 2)
    ri = np.arange(1, len(f), 2)
    n = min(len(li), len(ri))
    li, ri = li[:n], ri[:n]
    wh = w * Hv
    L = (wh[li, None] - wh[None, ri]) / (s[li, None] - s[None, ri])
    sv = np.linalg.svd(L, compute_uv=False)
    sv = sv / sv[0] if sv[0] > 0 else sv

    doublets = []
    if poles_hz is not None and zeros_hz is not None:
        ps = np.asarray(poles_hz, dtype=complex)
        zs = np.asarray(zeros_hz, dtype=complex)
        for p in ps:
            ap = abs(p)
            if not (fmin <= ap <= fmax) or ap == 0:
                continue
            if zs.size == 0:
                continue
            j = int(np.argmin(np.abs(zs - p)))
            sep = abs(zs[j] - p) / ap
            if sep < doublet_sep:
                doublets.append((float(ap), float(abs(zs[j])), float(sep)))
        doublets.sort(key=lambda d: d[2])

    return OrderCertificate(fmin=float(f[0]), fmax=float(f[-1]),
                            anchor=anchor, sv=sv,
                            shape=_classify(mag), doublets=doublets)
