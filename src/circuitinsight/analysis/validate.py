"""Validate a symbolic transfer function against the simulator's AC analysis.

This is the project's permanent correctness gate (PLAN.md §2.5): the
reconstructed small-signal model evaluated over frequency must match the
simulator's own AC sweep of the same design. Disagreement means a bug in the
model mapping, the stamps, or the join — no matter how plausible the symbolic
expression looks.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..engine.mna import TransferFunction


@dataclass
class ValidationReport:
    freq: np.ndarray
    mag_err_db: np.ndarray
    phase_err_deg: np.ndarray

    @property
    def worst_mag_db(self) -> float:
        return float(np.max(np.abs(self.mag_err_db)))

    @property
    def worst_phase_deg(self) -> float:
        return float(np.max(np.abs(self.phase_err_deg)))

    def ok(self, mag_tol_db: float = 0.5, phase_tol_deg: float = 3.0) -> bool:
        return (self.worst_mag_db <= mag_tol_db
                and self.worst_phase_deg <= phase_tol_deg)

    def summary(self) -> str:
        i_m = int(np.argmax(np.abs(self.mag_err_db)))
        i_p = int(np.argmax(np.abs(self.phase_err_deg)))
        return (
            f"worst |mag| err {self.worst_mag_db:.3f} dB @ {self.freq[i_m]:.3g} Hz; "
            f"worst phase err {self.worst_phase_deg:.2f} deg @ {self.freq[i_p]:.3g} Hz "
            f"({len(self.freq)} points, {self.freq[0]:.3g}..{self.freq[-1]:.3g} Hz)"
        )


def compare_tf(
    tf: TransferFunction,
    freq: np.ndarray,
    sim_out: np.ndarray,
    sim_in: np.ndarray | None = None,
    fmin: float | None = None,
    fmax: float | None = None,
) -> ValidationReport:
    """Compare tf(j2πf) against the simulator response sim_out (optionally
    normalized by sim_in, for testbenches where the source mag is not 1)."""
    freq = np.asarray(freq, dtype=float)
    h_sim = np.asarray(sim_out, dtype=complex)
    if sim_in is not None:
        h_in = np.asarray(sim_in, dtype=complex)
        if np.max(np.abs(h_in)) == 0.0:
            raise ValueError(
                "reference input wave is identically zero — is that source "
                "actually AC-driven (mag=...) in this analysis?"
            )
        h_sim = h_sim / h_in

    mask = np.ones_like(freq, dtype=bool)
    if fmin is not None:
        mask &= freq >= fmin
    if fmax is not None:
        mask &= freq <= fmax
    if not mask.any():
        raise ValueError("empty frequency band after fmin/fmax masking")
    f = freq[mask]
    h_sim = h_sim[mask]

    h_model = tf.numeric(f)
    mag_err = 20 * np.log10(np.abs(h_model)) - 20 * np.log10(np.abs(h_sim))
    phase_err = np.degrees(np.angle(h_model * np.conj(h_sim)))
    return ValidationReport(freq=f, mag_err_db=mag_err, phase_err_deg=phase_err)


def assert_tf_matches(
    tf: TransferFunction,
    freq: np.ndarray,
    sim_out: np.ndarray,
    sim_in: np.ndarray | None = None,
    mag_tol_db: float = 0.5,
    phase_tol_deg: float = 3.0,
    fmin: float | None = None,
    fmax: float | None = None,
) -> ValidationReport:
    report = compare_tf(tf, freq, sim_out, sim_in, fmin, fmax)
    if not report.ok(mag_tol_db, phase_tol_deg):
        raise AssertionError(
            f"symbolic TF disagrees with simulator AC: {report.summary()} "
            f"(tolerances: {mag_tol_db} dB / {phase_tol_deg} deg)"
        )
    return report
