"""Thevenin/Norton equivalents (Analyzer.equivalent) and port markers.

The schematic-level idiom — two dual, electrically invisible markers:
a 0 A isource in parallel is a *parallel* port (natively open: the
Thevenin probe); a 0 V vsource in series is a *series* port (natively a
short: the Norton probe — the branch is auto-opened for vth/zth and isc
is the current it carries in the intact circuit). disable= separately
chooses the equivalent's boundary. Both flow through the netlister and
OP join as ordinary sources.
"""
import warnings
from pathlib import Path

import pytest
import sympy as sp

from circuitinsight import Analyzer
from circuitinsight.adapters.spectre import SpectreRun

# V1--R1--out--R2--gnd source network; IPORT marks the parallel port at
# `out`; VPORT (0 V, series) marks the load branch through RL
PORTED = {
    "cin_version": "0.1",
    "design": {"name": "ported_divider", "source": {"kind": "hand"}},
    "top": "main",
    "ground": ["0"],
    "definitions": {"main": {"ports": [], "instances": [
        {"name": "V1", "device_type": "vsource",
         "terminals": {"p": "vin", "n": "0"}},
        {"name": "R1", "device_type": "resistor",
         "terminals": {"p": "vin", "n": "out"}, "params": {"r": "1k"}},
        {"name": "R2", "device_type": "resistor",
         "terminals": {"p": "out", "n": "0"}, "params": {"r": "3k"}},
        {"name": "IPORT", "device_type": "isource",
         "terminals": {"p": "out", "n": "0"}},
        {"name": "VPORT", "device_type": "vsource",
         "terminals": {"p": "out", "n": "nl"}},
        {"name": "RL", "device_type": "resistor",
         "terminals": {"p": "nl", "n": "0"}, "params": {"r": "6k"}},
    ]}},
}


def _r(eq):
    y = eq.vth.symbols
    return y["R1"], y["R2"], y["RL"]


def test_parallel_port_with_load_excluded():
    an = Analyzer.from_cin(PORTED)
    eq = an.equivalent("V1", port="IPORT", disable=("VPORT",))
    R1, R2, _ = _r(eq)
    assert sp.simplify(eq.vth.expr - R2 / (R1 + R2)) == 0
    assert sp.simplify(eq.zth.expr - R1 * R2 / (R1 + R2)) == 0
    assert sp.simplify(eq.isc.expr - 1 / R1) == 0


def test_parallel_port_isc_invariant_to_parallel_load():
    # with the load attached, vth and zth both change but the short-circuit
    # current cannot: isc stays exactly 1/R1
    an = Analyzer.from_cin(PORTED)
    eq = an.equivalent("V1", port="IPORT")
    R1 = eq.vth.symbols["R1"]
    assert sp.simplify(eq.isc.expr - 1 / R1) == 0
    assert complex(eq.vth.numeric([1.0])[0]).real == pytest.approx(2 / 3)
    assert complex(eq.zth.numeric([1.0])[0]).real == pytest.approx(2000 / 3)


def test_series_port_is_the_norton_probe():
    # port=VPORT: branch auto-opened for vth/zth; isc is the current the
    # branch carries in the intact circuit
    an = Analyzer.from_cin(PORTED)
    eq = an.equivalent("V1", port="VPORT")
    R1, R2, RL = _r(eq)
    assert sp.simplify(eq.vth.expr - R2 / (R1 + R2)) == 0        # open gap
    assert sp.simplify(eq.zth.expr - (R1 * R2 / (R1 + R2) + RL)) == 0
    # intact circuit: I(RL) = [(R2||RL)/(R1+R2||RL)] / RL = 1/9 mA
    assert complex(eq.isc.numeric([1.0])[0]).real == pytest.approx(1 / 9000)


def test_series_port_loop_impedance():
    an = Analyzer.from_cin(PORTED)
    Z = an.impedance(port="VPORT")
    R1, R2, RL = (Z.symbols[k] for k in ("R1", "R2", "RL"))
    assert sp.simplify(Z.expr - (R1 * R2 / (R1 + R2) + RL)) == 0


def test_port_markers_are_invisible():
    # neither marker may change any transfer function of the circuit
    bare = {**PORTED, "definitions": {"main": {"ports": [], "instances": [
        i for i in PORTED["definitions"]["main"]["instances"]
        if i["name"] != "IPORT"]}}}
    H_marked = Analyzer.from_cin(PORTED).tf("V1", "out")
    H_bare = Analyzer.from_cin(bare).tf("V1", "out")
    assert sp.simplify(H_marked.expr - H_bare.expr) == 0


def test_port_resolution_and_errors():
    an = Analyzer.from_cin(PORTED)
    assert an._port_nets("IPORT") == ("out", None, False)
    assert an._port_nets("VPORT") == ("out", "nl", True)
    with pytest.raises(ValueError, match="unknown port"):
        an._port_nets("NOPE")
    with pytest.raises(ValueError, match="isource .*or vsource"):
        an._port_nets("R1")
    with pytest.raises(ValueError, match="exactly one"):
        an.equivalent("V1", node="out", port="IPORT")
    with pytest.raises(ValueError, match="port cannot be the input"):
        an.equivalent("V1", port="V1")
    with pytest.raises(ValueError, match="unknown instance"):
        an.equivalent("V1", port="IPORT", disable=("NOPE",))


def test_follower_equivalent_consistency():
    FIX = Path(__file__).resolve().parent / "fixtures" / "spectre" / "follower3"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        run = SpectreRun(FIX / "tb_follower3.cin.json", FIX / "psf")
    an = run.analyzer()
    eq = an.equivalent("VIN", node="out", keep=[])
    f = [1e3, 1e8]
    vth, zth, isc = (x.numeric(f) for x in (eq.vth, eq.zth, eq.isc))
    assert isc == pytest.approx(vth / zth, rel=1e-9)
    # zth must equal the standalone impedance analysis
    assert zth == pytest.approx(an.impedance("out", keep=[]).numeric(f),
                                rel=1e-12)
    # three followers: positive sub-unity gain behind ~1 kOhm
    assert 0.3 < vth[0].real < 1.0
    assert zth[0].real == pytest.approx(986.8, rel=1e-3)
