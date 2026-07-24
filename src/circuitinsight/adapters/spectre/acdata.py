"""AC-analysis results: frequency axis + complex node waveforms.

Same backend pair as OP data: psfascii (fixtures/CI) and cdspythonsrr
(binary ADE results, Cadence environment only).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .backends import BackendError, srr_available


@dataclass
class AcResult:
    freq: np.ndarray                 # Hz, ascending
    waves: dict[str, np.ndarray]     # node/branch name -> complex response

    def wave(self, name: str) -> np.ndarray:
        if name not in self.waves:
            raise KeyError(
                f"no AC wave {name!r}; available: {sorted(self.waves)[:20]}"
            )
        return self.waves[name]


def _load_psfascii(path: Path) -> AcResult:
    from .psfascii import parse_psfascii

    psf = parse_psfascii(path)
    if not psf.sweeps:
        raise BackendError(f"{path}: not a swept (AC) result file")
    freq = np.asarray(psf.sweep_values, dtype=float)
    waves = {
        name: np.asarray(vals, dtype=complex)
        for name, vals in psf.trace_values.items()
        if len(vals) == len(freq)
    }
    return AcResult(freq=freq, waves=waves)


def _load_srr(psf_dir: Path, analysis: str | None = None) -> AcResult:
    from cdspythonsrr.core.ocean import (getData, openResults, outputs,
                                         results, selectResult)

    openResults(str(psf_dir))
    names = results()
    if analysis is None:
        cands = [n for n in names if n.endswith("-ac") or n == "ac"]
        if not cands:
            raise BackendError(f"no AC analysis in {psf_dir}; found {names}")
        analysis = cands[0]
    selectResult(analysis)

    freq = None
    waves: dict[str, np.ndarray] = {}
    for out in outputs():
        d = getData(out)
        if not isinstance(d, dict):
            continue
        x = d.get("x")
        y = d.get("y")
        if x is None or y is None:
            continue
        if freq is None:
            freq = np.asarray(x, dtype=float)
        waves[out] = np.asarray(y, dtype=complex)
    if freq is None:
        raise BackendError(f"{analysis} in {psf_dir}: no waveform data found")
    return AcResult(freq=freq, waves=waves)


@dataclass
class XfResult:
    """Spectre `xf` results: per-source transfer functions to one output.

    Each independent source's transfer is isolated (one adjoint solve),
    regardless of the run's AC magnitudes — the preferred per-source
    validation reference. Units: V/V for vsources, V/A (impedance) for
    isources when the xf output is a voltage."""
    freq: np.ndarray
    transfers: dict[str, np.ndarray]   # source instance -> complex transfer

    def tf(self, source: str) -> np.ndarray:
        if source not in self.transfers:
            raise KeyError(
                f"no xf transfer from {source!r}; "
                f"available: {sorted(self.transfers)}"
            )
        return self.transfers[source]


def load_xf(path: str | Path, backend: str = "auto") -> XfResult:
    """`path` is a psf results directory (default file xf.xf) or the file.
    An xf result is a swept psfascii file with one complex trace per
    independent source, so the AC loader machinery applies unchanged."""
    path = Path(path)
    xffile = path / "xf.xf" if path.is_dir() else path

    if backend in ("auto", "psfascii") and xffile.exists():
        with open(xffile, "rb") as fh:
            if fh.read(6) == b"HEADER":
                r = _load_psfascii(xffile)
                return XfResult(freq=r.freq, transfers=r.waves)
    if backend == "psfascii":
        raise BackendError(f"{xffile}: missing or not a psfascii xf result")
    if srr_available():
        from cdspythonsrr.core.ocean import openResults, results

        openResults(str(xffile.parent if xffile.exists() else path))
        cands = [n for n in results() if n.endswith("-xf") or n == "xf"]
        if not cands:
            raise BackendError(f"no xf analysis in {path}")
        r = _load_srr(xffile.parent if xffile.exists() else path, cands[0])
        return XfResult(freq=r.freq, transfers=r.waves)
    raise BackendError(
        f"{xffile}: binary or missing xf results and no cdspythonsrr backend; "
        f"rerun spectre with '-format psfascii' or run under the Cadence "
        f"environment."
    )


def load_ac(path: str | Path, backend: str = "auto") -> AcResult:
    """`path` is a psf results directory (default file ac.ac) or the AC file."""
    path = Path(path)
    acfile = path / "ac.ac" if path.is_dir() else path

    if backend == "psfascii":
        return _load_psfascii(acfile)
    if backend == "cdspythonsrr":
        return _load_srr(acfile.parent if acfile.is_file() else path)
    if backend != "auto":
        raise BackendError(f"unknown backend {backend!r}")

    if acfile.exists():
        with open(acfile, "rb") as fh:
            if fh.read(6) == b"HEADER":
                return _load_psfascii(acfile)
    if srr_available():
        return _load_srr(acfile.parent if acfile.exists() else path)
    raise BackendError(
        f"{acfile}: binary or missing AC results and no cdspythonsrr backend; "
        f"rerun spectre with '-format psfascii' or run under the Cadence "
        f"environment."
    )
