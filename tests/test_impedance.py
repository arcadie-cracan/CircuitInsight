"""Driving-point impedance analysis (Analyzer.impedance).

Analytic checks on hand circuits (divider, source follower), then a
solver-path consistency check on the real 5T OTA fixture.
"""
import warnings
from pathlib import Path

import pytest
import sympy as sp

from circuitinsight import Analyzer
from circuitinsight.adapters.spectre import SpectreRun
from circuitinsight.engine.mna import MnaError

GOLDEN = Path(__file__).resolve().parent / "golden" / "circuits"
OTA5T = Path(__file__).resolve().parent / "fixtures" / "spectre" / "ota5t"

DIVIDER = {
    "cin_version": "0.1",
    "design": {"name": "divider", "source": {"kind": "hand"}},
    "top": "main",
    "ground": ["0"],
    "definitions": {"main": {"ports": [], "instances": [
        {"name": "V1", "device_type": "vsource",
         "terminals": {"p": "vin", "n": "0"}},
        {"name": "R1", "device_type": "resistor",
         "terminals": {"p": "vin", "n": "out"}, "params": {"r": "1k"}},
        {"name": "R2", "device_type": "resistor",
         "terminals": {"p": "out", "n": "0"}, "params": {"r": "3k"}},
    ]}},
}


def test_zout_divider_symbolic():
    an = Analyzer.from_cin(DIVIDER)
    Z = an.impedance("out")
    R1, R2 = Z.symbols["R1"], Z.symbols["R2"]
    # V1 is zeroed (short), so looking into `out` sees R1 || R2
    assert sp.simplify(Z.expr - R1 * R2 / (R1 + R2)) == 0
    assert complex(Z.numeric([1.0])[0]).real == pytest.approx(750.0)


def test_driven_port_is_shorted_by_its_source():
    an = Analyzer.from_cin(DIVIDER)
    assert Analyzer.from_cin(DIVIDER).impedance("vin").expr == 0
    # removing the driving source exposes the real input impedance
    Z = an.impedance("vin", disable=("V1",))
    R1, R2 = Z.symbols["R1"], Z.symbols["R2"]
    assert sp.simplify(Z.expr - (R1 + R2)) == 0


def test_differential_port():
    an = Analyzer.from_cin(DIVIDER)
    # current driven in at `out`, out at `vin`: only R1 lies in the loop
    Z = an.impedance("out", ref="vin", disable=("V1",))
    assert sp.simplify(Z.expr - Z.symbols["R1"]) == 0


def test_bad_arguments():
    an = Analyzer.from_cin(DIVIDER)
    with pytest.raises(ValueError, match="unknown instance"):
        an.impedance("out", disable=("NOPE",))
    with pytest.raises(MnaError, match="ground"):
        an.impedance("0")
    with pytest.raises(MnaError, match="not found"):
        an.impedance("nonexistent_net")


def test_source_follower_zout():
    an = Analyzer.from_cin(GOLDEN / "source_follower.cin.json")
    Z = an.impedance("vout")
    y = Z.symbols
    # textbook: Rout = 1 / (gm + gmbs + gds + 1/RS), gate at ac ground
    expected = 1 / (y["gm_M1"] + y["gmbs_M1"] + y["gds_M1"] + 1 / y["RS"])
    assert sp.simplify(Z.dc_gain() - expected) == 0
    rout = complex(Z.numeric([1.0])[0]).real
    assert rout == pytest.approx(1 / (469e-6 + 87e-6 + 8.5e-6 + 1 / 20e3),
                                 rel=1e-6)


def test_ota5t_zout_solver_paths_agree():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        run = SpectreRun(OTA5T / "tb_ota5t.cin.json", OTA5T / "psf")
    an = run.analyzer()
    freqs = [1e3, 1e8]
    keep = ["gm_I0_MN2"]           # one kept symbol: small grid, fast dets
    zd = an.impedance("vout", keep=keep, method="direct").numeric(freqs)
    zi = an.impedance("vout", keep=keep, method="interp").numeric(freqs)
    assert zd == pytest.approx(zi, rel=1e-9)
    rout = zd[0].real
    assert 1e3 < rout < 1e7          # a real OTA output resistance
    # |Z| must fall with frequency once the output caps take over
    assert abs(zd[1]) < abs(zd[0])
