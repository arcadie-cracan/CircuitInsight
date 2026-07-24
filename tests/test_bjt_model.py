"""BJT model options: intrinsic hybrid-pi vs extrinsic (with rb/re/rc).

Fast, simulator-free checks on a single common-emitter stage with
hand-set operating-point parameters. The full-uA741 exercise of the same
option lives in test_ua741; here we pin the structure and the physics.
"""
import numpy as np
import pytest

from circuitinsight import Analyzer
from circuitinsight.models.small_signal import ModelError

CE = {
    "cin_version": "0.1",
    "design": {"name": "ce", "source": {"kind": "hand"}},
    "top": "m", "ground": ["0"],
    "definitions": {"m": {"ports": [], "instances": [
        {"name": "Vin", "device_type": "vsource",
         "terminals": {"p": "in", "n": "0"}},
        {"name": "Rs", "device_type": "resistor",
         "terminals": {"p": "in", "n": "b"}, "params": {"r": "1k"}},
        {"name": "RC", "device_type": "resistor",
         "terminals": {"p": "out", "n": "0"}, "params": {"r": "10k"}},
        {"name": "Q1", "device_type": "bjt",
         "terminals": {"c": "out", "b": "b", "e": "0"},
         "params": {"polarity": "npn", "gm": "40m", "gpi": "200u",
                    "go": "10u", "cpi": "1p", "cmu": "0.1p",
                    "rb": "200", "re": "5", "rc": "150"}},
    ]}},
}


def _gain(bjt_model):
    an = Analyzer.from_cin(CE, bjt_model=bjt_model)
    q1 = [p for p in an.primitives if p.inst == "Q1"]
    a0 = complex(an.tf("Vin", "out", keep=[]).numeric([1.0])[0]).real
    return an, q1, a0


def test_intrinsic_is_default_and_unchanged():
    an, q1, a0 = _gain("intrinsic")
    # exactly the five hybrid-pi elements on the external terminals
    assert {p.param for p in q1} == {"gm", "gpi", "go", "cpi", "cmu"}
    for p in q1:
        assert not any(n.startswith("Q1#") for n in p.nodes)
    # textbook: -(rpi/(Rs+rpi)) * gm * (RC||ro) with rpi=5k, RC||ro=9.09k
    assert a0 == pytest.approx(-303.0, rel=0.02)


def test_extrinsic_adds_series_resistances():
    an, q1, a0 = _gain("extrinsic")
    series = {p.param: p for p in q1 if p.param in ("rb", "re", "rc")}
    assert set(series) == {"rb", "re", "rc"}
    # each series R bridges an external terminal to a per-terminal internal
    assert series["rb"].nodes == ("b", "Q1#bi")
    assert series["re"].nodes == ("0", "Q1#ei")
    assert series["rc"].nodes == ("out", "Q1#ci")
    # the intrinsic core now sees the internal nodes
    gm = next(p for p in q1 if p.param == "gm")
    assert gm.nodes == ("Q1#ci", "Q1#ei", "Q1#bi", "Q1#ei")
    # emitter degeneration (gm*re = 0.2) lowers the gain vs intrinsic
    assert -290 < a0 < -230


def test_zero_series_resistance_inserts_no_node():
    # a device whose rb/re/rc are absent/zero must not gain internal nodes
    cin = {**CE}
    import copy
    cin = copy.deepcopy(CE)
    q = cin["definitions"]["m"]["instances"][-1]["params"]
    for k in ("rb", "re", "rc"):
        q.pop(k)
    an = Analyzer.from_cin(cin, bjt_model="extrinsic")
    q1 = [p for p in an.primitives if p.inst == "Q1"]
    assert not any(n.startswith("Q1#") for p in q1 for n in p.nodes)


def test_unknown_bjt_model_rejected():
    with pytest.raises(ModelError, match="bjt_model"):
        Analyzer.from_cin(CE, bjt_model="nope")
