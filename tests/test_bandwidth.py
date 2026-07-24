"""ZVTC bandwidth attribution: exact per-element time constants."""
import warnings
from pathlib import Path

import pytest

from circuitinsight import Analyzer
from circuitinsight.adapters.spectre import SpectreRun

GOLDEN = Path(__file__).resolve().parent / "golden" / "circuits"
MILLER = Path(__file__).resolve().parent / "fixtures" / "spectre" / "miller"


def circuit(instances):
    return {"cin_version": "0.1", "top": "main", "ground": ["0"],
            "definitions": {"main": {"ports": [], "instances": instances}}}


def test_rc_single_pole_exact():
    an = Analyzer.from_cin(circuit([
        {"name": "V1", "device_type": "vsource",
         "terminals": {"p": "vin", "n": "0"}},
        {"name": "R1", "device_type": "resistor",
         "terminals": {"p": "vin", "n": "vout"}, "params": {"r": "1k"}},
        {"name": "C1", "device_type": "capacitor",
         "terminals": {"p": "vout", "n": "0"}, "params": {"c": "1n"}},
    ]))
    rep = an.bandwidth_report()
    (row,) = rep.rows
    assert row.tau == pytest.approx(1e-6, rel=1e-12)        # R*C
    assert row.share == pytest.approx(1.0)
    assert rep.validity == pytest.approx(1.0, rel=1e-9)     # single pole


def test_rc_ladder_textbook_zvtc():
    # tau1 = C1*R1, tau2 = C2*(R1+R2) — the classic ZVTC example
    an = Analyzer.from_cin(circuit([
        {"name": "V1", "device_type": "vsource",
         "terminals": {"p": "vin", "n": "0"}},
        {"name": "R1", "device_type": "resistor",
         "terminals": {"p": "vin", "n": "n1"}, "params": {"r": "1k"}},
        {"name": "C1", "device_type": "capacitor",
         "terminals": {"p": "n1", "n": "0"}, "params": {"c": "1n"}},
        {"name": "R2", "device_type": "resistor",
         "terminals": {"p": "n1", "n": "n2"}, "params": {"r": "2k"}},
        {"name": "C2", "device_type": "capacitor",
         "terminals": {"p": "n2", "n": "0"}, "params": {"c": "2n"}},
    ]))
    rep = an.bandwidth_report()
    taus = {r.name: r.tau for r in rep.rows}
    assert taus["C1"] == pytest.approx(1e-6, rel=1e-12)      # 1n * 1k
    assert taus["C2"] == pytest.approx(6e-6, rel=1e-12)      # 2n * 3k
    assert rep.tau_total == pytest.approx(7e-6, rel=1e-12)


def test_cs_amp_attribution():
    an = Analyzer.from_cin(GOLDEN / "cs_amp.cin.json")
    rep = an.bandwidth_report()
    taus = {r.name: r.tau for r in rep.rows}
    ro = 1 / (8.5e-6 + 1e-4)                                 # gds || RL
    assert taus["cgs_M1"] == 0.0                             # ideal drive
    assert taus["CL"] == pytest.approx(1e-13 * ro, rel=1e-9)
    assert taus["cgd_M1"] == pytest.approx(0.75e-15 * ro, rel=1e-9)
    assert taus["cdb_M1"] == pytest.approx(1.7e-15 * ro, rel=1e-9)
    assert rep.rows[0].name == "CL"


def test_two_stage_miller_dominates():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        run = SpectreRun(MILLER / "tb_ota2s.cin.json", MILLER / "psf")
        an = run.analyzer()
        an.match("I0.MN0", "I0.MN1")
        an.match("I0.MP0", "I0.MP1")
    rep = an.bandwidth_report()
    # Miller multiplication: the 14.7 pF Cc (multiplied by the 2nd-stage gain)
    # so dominates the bandwidth that even the 5 pF CL is a minor contributor
    assert rep.rows[0].name == "I0_Cc"
    assert rep.rows[0].share > 0.95
    assert next(r for r in rep.rows if r.name == "CL").share < 0.05
    assert 0.9 < rep.validity < 1.1
    assert len(rep.rows) > 25                                # all device caps
