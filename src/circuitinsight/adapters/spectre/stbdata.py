"""Spectre `stb` (stability) results: Tian-probe loop gain + margins.

`stb` inserts its two injections at a designated probe (analogLib `iprobe`)
and reports the true loop gain seen there (Tian's method), plus the phase and
gain margins it derives. Two psfascii files:

- ``stb.stb``        swept traces in the ac.ac layout: ``loopGain`` (complex),
                     and the impedances/admittances looking both ways from the
                     probe (``ZL``/``ZG``, ``YL``/``YG``);
- ``stb.margin.stb`` scalar margins: gainMargin (dB) / gainMarginFreq,
                     phaseMargin (deg) / phaseMarginFreq, and a stability
                     verdict string.

The sweep reuses the psfascii AC loader unchanged; the margin file is a
scalar-VALUE psfascii like dcOp.dc. This is the validation reference for the
reconstructed loop gain (docs/loopgain-plan.md), the same per-result pattern
as `ac` for transfer functions and `xf` for impedances.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .acdata import _load_psfascii, _load_srr
from .backends import BackendError, srr_available


@dataclass
class StbResult:
    freq: np.ndarray                     # Hz, ascending
    waves: dict[str, np.ndarray]         # loopGain, ZL, ZG, YL, YG, ...
    margins: dict[str, float | str] = field(default_factory=dict)
    probe: str | None = None             # designated probe (sim name), if known

    @property
    def loop_gain(self) -> np.ndarray:
        """Complex loop gain T(j2*pi*f) at the probe."""
        if "loopGain" not in self.waves:
            raise KeyError(
                f"no loopGain trace; available: {sorted(self.waves)}")
        return self.waves["loopGain"]

    # margin accessors return None when the margin file was absent
    @property
    def phase_margin_deg(self) -> float | None:
        v = self.margins.get("phaseMargin")
        return float(v) if v is not None else None

    @property
    def phase_margin_freq_hz(self) -> float | None:
        v = self.margins.get("phaseMarginFreq")
        return float(v) if v is not None else None

    @property
    def gain_margin_db(self) -> float | None:
        v = self.margins.get("gainMargin")
        return float(v) if v is not None else None

    @property
    def gain_margin_freq_hz(self) -> float | None:
        v = self.margins.get("gainMarginFreq")
        return float(v) if v is not None else None


def _load_margins(path: Path) -> dict[str, float | str]:
    from .psfascii import parse_psfascii

    return {name: e.value for name, e in parse_psfascii(path).entries.items()}


def load_stb(path: str | Path, backend: str = "auto") -> StbResult:
    """`path` is a psf results directory (default file stb.stb) or the file.
    The companion margin file (stb.margin.stb) is read when present.

    Backends mirror load_ac/load_xf: psfascii when the file is there and
    ascii; cdspythonsrr for native binary ADE results (the sweep comes
    from the "-stb" result; margins and the probe name are best-effort
    there -- the psfascii header carries the probe, SRR results may not).
    NEEDS LIVE VERIFICATION under a Cadence environment."""
    path = Path(path)
    stbfile = path / "stb.stb" if path.is_dir() else path

    if backend not in ("auto", "psfascii", "cdspythonsrr"):
        raise BackendError(f"unknown backend {backend!r} for stb results")

    if backend in ("auto", "psfascii") and stbfile.exists():
        with open(stbfile, "rb") as fh:
            ascii_ = fh.read(6) == b"HEADER"
        if ascii_:
            from .psfascii import parse_psfascii

            r = _load_psfascii(stbfile)
            probe = parse_psfascii(stbfile).header.get("probe")
            marginfile = stbfile.parent / "stb.margin.stb"
            margins = _load_margins(marginfile) if marginfile.exists() else {}
            return StbResult(freq=r.freq, waves=r.waves, margins=margins,
                             probe=probe if isinstance(probe, str) else None)
    if backend == "psfascii":
        raise BackendError(
            f"{stbfile}: missing or binary stb results; rerun spectre with "
            f"'-format psfascii' and an `stb` analysis on a vsource probe.")

    if srr_available():
        base = stbfile.parent if stbfile.exists() else path
        from cdspythonsrr.core.ocean import openResults, results

        openResults(str(base))
        names = results()
        cands = [n for n in names if n.endswith("-stb") or n == "stb"]
        if not cands:
            raise BackendError(f"no stb analysis in {base}; found {names}")
        r = _load_srr(base, cands[0])
        margins: dict[str, float | str] = {}
        try:                              # margins live in a sibling result
            mcands = [n for n in names if "margin" in n and "stb" in n]
            if mcands:
                from cdspythonsrr.core.ocean import (getData, outputs,
                                                     selectResult)

                selectResult(mcands[0])
                for out in outputs():
                    d = getData(out)
                    if not isinstance(d, dict):
                        margins[out] = d
        except Exception:
            pass
        return StbResult(freq=r.freq, waves=r.waves, margins=margins)

    raise BackendError(
        f"{stbfile}: binary or missing stb results and no cdspythonsrr "
        f"backend; rerun spectre with '-format psfascii' or run under the "
        f"Cadence environment.")
