"""Full-chain regression: real 5T OTA (SKY130) from Virtuoso to validated TF.

Fixtures from worklib/tb_ota5t on the open-source SKY130 PDK (pair 40u/2u,
mirror 40u/2u m=2, tail 40u/1u m=2, 40 uA bias -> 46.1 dB): CInExport-topology
CIN (ideal_balun drive, DC-feedback/AC-open ideal switches) + psfascii results
of the identical ADE netlist. Exercises hierarchy, balun stamps,
analysis-dependent switches, and the OP join in one test.

The tail node swings with the input-pair sources, so the 5-cap LUMPED model
carries large charge-non-reciprocity error here (tens of dB at 10 GHz on
SKY130); the exact charge-matrix model stays below 0.1 dB, so the AC match is
validated against `cap_model="matrix"`.
"""
import warnings
from pathlib import Path

import numpy as np
import pytest

from circuitinsight.adapters.spectre import SpectreRun
from circuitinsight.analysis import assert_tf_matches

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "spectre" / "ota5t"


@pytest.fixture(scope="module")
def run():
    with warnings.catch_warnings():
        # the input pair sits in moderate inversion (region=3): by design
        warnings.simplefilter("ignore", UserWarning)
        return SpectreRun(FIXTURES / "tb_ota5t.cin.json", FIXTURES / "psf")


def test_join_and_ideal_elements(run):
    # 5 OTA transistors + bench diode + 3 vsources + isource
    assert len(run.op_data) == 10
    # ideal balun and switches need no OP records
    names = {d.name for d in run.flat.devices}
    assert {"I5", "S0", "S1"} <= names
    assert run.op_data["I0.MN2"]["gm"] > 5e-4          # tail, m=2 totals


def test_ota_tf_matches_spectre_ac(run):
    H = run.analyzer(cap_model="matrix").tf("VIND", "vout", keep=[])
    ac = run.ac()
    report = assert_tf_matches(
        H, ac.freq, ac.wave("vout"), ac.wave("vin_dm"),
        mag_tol_db=0.1, phase_tol_deg=3.5,             # matrix: 0.077 / 3.29
    )
    a0 = complex(H.numeric([1e3])[0])
    assert a0.real == pytest.approx(202.55, rel=1e-3)  # matches sim +46.1 dB
    assert report.worst_mag_db < 0.1


def test_switch_states_shape_the_circuit(run):
    # S1 (out->inn feedback) must be ABSENT in AC: unity feedback would make
    # the gain ~1 instead of ~72. S0 must be a short: opening it floats inn.
    H = run.analyzer().tf("VIND", "vin_n", keep=[])    # inn
    assert complex(H.numeric([1e3])[0]).real == pytest.approx(-0.5, abs=1e-6)
