"""Multilinear-interpolation solver: exact equivalence with the direct path.

The interp path must produce SYMBOLICALLY IDENTICAL transfer functions —
it is reconstruction, not approximation (docs/multilinear-solver-plan.md).
"""
import warnings
from pathlib import Path

import pytest
import sympy as sp

from circuitinsight import Analyzer
from circuitinsight.adapters.spectre import SpectreRun
from circuitinsight.analysis import compare_tf

GOLDEN = Path(__file__).resolve().parent / "golden" / "circuits"
OTA = Path(__file__).resolve().parent / "fixtures" / "spectre" / "ota5t"


def assert_equivalent(an, inp, out, keep):
    H_d = an.tf(inp, out, keep=keep, method="direct")
    H_i = an.tf(inp, out, keep=keep, method="interp")
    diff = sp.cancel(sp.together(H_d.expr - H_i.expr))
    assert sp.simplify(diff) == 0, f"paths disagree for keep={keep}"
    return H_i


def test_cs_amp_with_kept_resistor():
    # exercises the reciprocal (u = 1/R) path: RL kept symbolic
    an = Analyzer.from_cin(GOLDEN / "cs_amp.cin.json")
    H = assert_equivalent(an, "V1", "vout", keep=["M1", "RL", "CL"])
    # sanity: textbook dc gain still exact
    y = H.symbols
    dc = sp.simplify(H.expr.subs(sp.Symbol("s"), 0))
    expected = -y["RL"] * y["gm_M1"] / (y["RL"] * y["gds_M1"] + 1)
    assert sp.simplify(dc - expected) == 0


def test_ota5t_golden_matched_pairs():
    # matched symbols -> degree-2 axes in the tensor grid
    an = Analyzer.from_cin(GOLDEN / "ota5t.cin.json")
    an.match("M1", "M2")
    an.match("M3", "M4")
    assert_equivalent(an, "V1", "vout", keep=["M1", "M3", "CL"])


def test_transimpedance_kept_resistor():
    an = Analyzer.from_cin({
        "cin_version": "0.1", "top": "main", "ground": ["0"],
        "definitions": {"main": {"ports": [], "instances": [
            {"name": "I1", "device_type": "isource",
             "terminals": {"p": "0", "n": "vout"}},
            {"name": "R1", "device_type": "resistor",
             "terminals": {"p": "vout", "n": "0"}, "params": {"r": "1k"}},
            {"name": "C1", "device_type": "capacitor",
             "terminals": {"p": "vout", "n": "0"}, "params": {"c": "1n"}},
        ]}}})
    H = assert_equivalent(an, "I1", "vout", keep=["R1", "C1"])
    y = H.symbols
    s = sp.Symbol("s")
    assert sp.simplify(
        sp.cancel(H.expr) - y["R1"] / (1 + s * y["R1"] * y["C1"])) == 0


def test_real_bench_with_balun_and_switches():
    # equivalence proxy on the full testbench: the interp result must match
    # the simulator AC exactly as well as the exact model does
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        run = SpectreRun(OTA / "tb_ota5t.cin.json", OTA / "psf")
        an = run.analyzer(cap_model="matrix")      # SKY130 lumped errs ~28 dB
        an.match("I0.MN0", "I0.MN1")
        an.match("I0.MP0", "I0.MP1")
        H = an.tf("VIND", "vout",
                  keep=["gm_I0_MN0", "gds_I0_MN0", "gm_I0_MP0", "gds_I0_MP0"],
                  method="interp")
    ac = run.ac()
    # interp must reproduce the sim as well as the exact model does (charge matrix)
    r = compare_tf(H, ac.freq, ac.wave("vout"), ac.wave("vin_dm"))
    assert r.worst_mag_db < 0.1 and r.worst_phase_deg < 3.5


def test_empty_keep_falls_back_to_direct():
    an = Analyzer.from_cin(GOLDEN / "cs_amp.cin.json")
    H = an.tf("V1", "vout", keep=[], method="interp")
    assert not (H.expr.free_symbols - {sp.Symbol("s")})


def test_unknown_method_rejected():
    from circuitinsight.engine.mna import MnaError
    an = Analyzer.from_cin(GOLDEN / "cs_amp.cin.json")
    with pytest.raises(MnaError, match="method"):
        an.tf("V1", "vout", keep=["M1"], method="magic")
