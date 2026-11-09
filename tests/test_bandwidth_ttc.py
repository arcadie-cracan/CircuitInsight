"""Hajimiri's time- and transfer-constant output of the bandwidth report.

Every constant is checked against the TTC class of tests/ttc_reference.py,
which extracts it symbolically from the exact transfer function and whose
expressions match the paper's. The numbers are the paper's (A. Hajimiri,
IEEE TCAS-I 57(6), 2010, Section VII).
"""
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import sympy as sp

from circuitinsight.engine.mna import MnaError

BENCH = Path(__file__).resolve().parent / "ttc_reference.py"

_spec = importlib.util.spec_from_file_location("ttc_reference", BENCH)
bench = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("ttc_reference", bench)
_spec.loader.exec_module(bench)
C, G, L, R, V, TTC, analyzer = (bench.C, bench.G, bench.L, bench.R, bench.V,
                                bench.TTC, bench.analyzer)
PI2 = 2 * np.pi


def cross_check(an, inp, out, names):
    """Every constant of the report against the symbolic extraction."""
    H = an.tf(inp, out)
    T = TTC(H, names)
    v = bench.values(H)
    rep = an.bandwidth_report(inp, out)
    rows = {r.name: r for r in rep.rows}

    def num(e):
        return float(sp.sympify(e).xreplace(v))

    assert set(rows) == set(names)
    assert rep.h0 == pytest.approx(num(T.h0()), rel=1e-12, abs=1e-12)
    assert rep.b1 == pytest.approx(num(T.b(1)), rel=1e-12)
    assert rep.b2 == pytest.approx(num(T.b(2)), rel=1e-12)
    for i in names:
        assert rows[i].tau == pytest.approx(num(T.tau0(i)), rel=1e-12, abs=1e-24)
        hi = T.hi(i)
        if hi is not None:
            assert rows[i].h == pytest.approx(num(hi), rel=1e-12, abs=1e-12)
        for j in names:
            if i == j:
                continue
            tji = T.tauij(j, i)
            if tji is None:
                assert rep.tau_second(j, i) is None
            else:
                assert rep.tau_second(j, i) == pytest.approx(num(tji), rel=1e-12, abs=1e-24)
    return rep


def ce(extra=()):
    return analyzer("ce", bench.ce_inst(extra))


def sf():
    return analyzer("sf", [V("V1", "vin"), R("R1", "vin", "g", "2k"), C("Cp", "g", "out", "50f"),
                           G("Gm", "0", "out", "g", "out", "20m"), C("CL", "out", "0", "50f")])


def bandpass():
    return analyzer("bp", [V("V1", "vin"), R("R", "vin", "a", "50"), C("C1", "a", "b", "1p"), L("L2", "b", "0", "1n"),
                           L("L3", "b", "out", "1n"), R("Rb", "out", "0", "50")])


def negative():
    return analyzer("neg", [{"name": "I1", "device_type": "isource", "terminals": {"p": "0", "n": "out"}, "params": {"acm": "1"}},
                            R("Ro", "out", "0", "1k"), L("L", "out", "0", "1n"), C("C", "out", "0", "1p"),
                            G("Gn", "0", "out", "out", "0", "2.5m")])


def trap():
    return analyzer("trap", [V("V1", "vin"), G("Gm", "d", "0", "vin", "0", "10m"), R("R1", "d", "0", "1k"),
                             L("L", "d", "out", "1n"), C("C", "d", "out", "1p"), R("RL", "out", "0", "1k")])


def test_example1_common_emitter():
    rep = cross_check(ce(), "V1", "c", ["Cp", "Cm", "CL"])
    rows = {r.name: r for r in rep.rows}
    assert rep.h0 == pytest.approx(-57.14, abs=0.01)
    assert rows["Cm"].tau == pytest.approx(1197.1e-12, rel=1e-4)
    assert rep.f_zvtc == pytest.approx(95.4e6, rel=1e-3)
    assert rep.tau_second("Cm", "Cp") == pytest.approx(20e-15 * 2e3)
    # no zero to speak of: the correction changes little
    assert rep.f_modified == pytest.approx(rep.f_zvtc, rel=1e-3)
    assert 0.9 < rep.validity < 1.1


def test_example3_modified_zvt_restores_the_bandwidth():
    rep = cross_check(ce([C("C1", "vin", "b", "4.3p")]), "V1", "c", ["Cp", "Cm", "CL", "C1"])
    rows = {r.name: r for r in rep.rows}
    assert rep.f_zvtc == pytest.approx(33.6e6, rel=2e-3)
    assert rep.f_modified == pytest.approx(362e6, rel=2e-3)
    assert abs(rows["Cm"].h / rep.h0) == pytest.approx(0.0004, abs=1e-4)
    assert abs(rows["C1"].h / rep.h0) == pytest.approx(1.4, rel=1e-6)
    assert rows["C1"].tau_bar == pytest.approx(-1.23e-9, rel=2e-3)
    assert rep.tau_bar_total == pytest.approx(sum(r.tau_bar for r in rep.rows))
    # the exact bandwidth, for the record: the estimate is now in its range
    H = ce([C("C1", "vin", "b", "4.3p")]).tf("V1", "c")
    assert bench.f3db(bench.response(H)) == pytest.approx(479e6, rel=2e-3)
    assert "362 MHz" in rep.report()


def test_example5_source_follower_second_order():
    rep = cross_check(sf(), "V1", "out", ["Cp", "CL"])
    assert rep.b1 == pytest.approx(5e-12)
    assert rep.b2 == pytest.approx(250e-24)
    assert rep.q == pytest.approx(3.16, abs=0.005)
    assert rep.f_n == pytest.approx(10.07e9, rel=1e-3)
    assert rep.w_n == pytest.approx(PI2 * rep.f_n)
    assert rep.tau_second("Cp", "CL") == pytest.approx(2e3 * 50e-15)
    # H^pi = H^0 cancels tau_pi, so (41) leaves tau_L: 63.7 GHz, see the
    # survey's note on the paper's printed 15.9 GHz
    assert {r.name: r for r in rep.rows}["Cp"].tau_bar == pytest.approx(0, abs=1e-24)
    assert rep.f_modified == pytest.approx(63.7e9, rel=1e-3)
    # a complex dominant pair: the old ratio is not a validity
    assert rep.dominant_kind == "complex"
    assert math.isnan(rep.validity)
    assert "not applicable" in rep.report()


def test_example6_bandpass_inductors():
    rep = cross_check(bandpass(), "V1", "out", ["C1", "L2", "L3"])
    assert {r.name: r.tau for r in rep.rows} == pytest.approx({"C1": 50e-12, "L2": 20e-12, "L3": 20e-12})
    assert rep.b2 == pytest.approx(3e-21)            # (2 L2 + L3) C1
    # a bandpass: H^0 = 0, so the ratio of (42) does not exist
    assert rep.h0 == 0
    assert all(r.h == 0 and r.tau_bar is None for r in rep.rows)
    assert rep.tau_bar_total is None and rep.f_modified is None
    assert "H^0 = 0" in rep.report()


def test_example7_negative_resistance_guards_validity():
    # the report used to print validity -21.08 here
    an = negative()
    rep = cross_check(an, "I1", "out", ["L", "C"])
    rows = {r.name: r for r in rep.rows}
    assert rows["L"].tau == pytest.approx(-1.5e-12)
    assert rows["L"].h == pytest.approx(-1 / 1.5e-3)  # -1/Geff, see the survey
    assert rep.tau_second("C", "L") == pytest.approx(1e-12 / -1.5e-3)
    assert rep.b2 == pytest.approx(1e-21)
    assert rep.q == pytest.approx(-21.08, abs=0.01)   # eq. (53), negative: RHP
    assert rep.dominant_kind == "rhp"
    assert math.isnan(rep.validity)
    assert "not applicable" in rep.report()
    assert "-21" not in rep.report().splitlines()[0]
    assert math.isnan(an.bandwidth_report().validity)


def test_example8_zero_value_time_constant():
    rep = cross_check(trap(), "V1", "out", ["L", "C"])
    rows = {r.name: r for r in rep.rows}
    assert rows["C"].tau == 0                         # shorted by the zero-value L
    assert rows["L"].tau == pytest.approx(1e-9 / 2e3)
    # the paper's note: tau_C^0 tau_L^C is 0 * inf, so go through tau_L^0 tau_C^L
    assert rep.tau_second("L", "C") is None
    assert rep.tau_second("C", "L") == pytest.approx(1e-12 * 2e3)
    (pair,) = rep.pairs
    assert pair.term == pytest.approx(1e-21)
    assert rep.b2 == pytest.approx(1e-21)             # LC
    # H^C = H^0: the short across a short changes nothing. H^L: no path.
    assert rep.h0 == pytest.approx(-5.0)
    assert rows["C"].h == pytest.approx(-5.0)
    assert rows["L"].h == 0
    assert rows["C"].tau_bar == 0
    assert rows["L"].tau_bar == pytest.approx(rows["L"].tau)
    assert "indeterminate" in rep.report()


def test_unbounded_transfer_constant_keeps_a_finite_product():
    # a -- 1k -- gnd and b -- (-1k) -- gnd: C sees 1k - 1k = 0, so tau_C^0 = 0,
    # but its current still moves v(a), so N1 has an x_C term: tau_C H^C = N1
    # is finite with H^C unbounded. N1 = -(1k)(1k) C and H^0 = 1k.
    an = analyzer("unb", [{"name": "I1", "device_type": "isource", "terminals": {"p": "0", "n": "a"}, "params": {"acm": "1"}},
                          R("Ra", "a", "0", "1k"), G("Gn", "b", "0", "b", "0", "-1m"), C("C", "a", "b", "1p")])
    rep = an.bandwidth_report("I1", "a")
    (row,) = rep.rows
    assert row.tau == 0
    assert row.h == -math.inf
    assert rep.h0 == pytest.approx(1e3)
    assert row.tau_bar == pytest.approx(-1e-9)        # -|N1/H^0|
    assert rep.f_modified is None                     # the sum is negative
    assert "not positive" in rep.report()
    # D(s) is a constant here, so there is no pole to compare with either
    assert rep.dominant_kind == "none"
    assert math.isnan(rep.validity)
    assert "no finite pole" in rep.report()


def test_no_arguments_needs_no_output():
    rep = ce().bandwidth_report()
    assert rep.h0 is None and rep.f_modified is None and rep.tau_bar_total is None
    assert all(r.h is None and r.tau_bar is None for r in rep.rows)
    assert rep.b2 > 0 and rep.q is not None
    first = rep.report().splitlines()[0]
    assert "true dominant pole" in first and "estimate/true" in first
    assert "Modified ZVT" not in rep.report()


def test_arguments_are_checked():
    with pytest.raises(MnaError):
        ce().bandwidth_report("V1")
    with pytest.raises(MnaError):
        ce().bandwidth_report(out="c")
    with pytest.raises(MnaError):
        ce().bandwidth_report("V1", "nosuch")
    with pytest.raises(MnaError):
        ce().bandwidth_report("R1", "c")


def test_matched_row_keeps_the_sums_exact():
    # two matched caps share one symbol: b2 then has a self term, which only
    # the row's own sum check can place
    an = analyzer("m", [V("V1", "vin"), R("R1", "vin", "a", "1k"), C("Ca", "a", "b", "1p"), R("R2", "b", "0", "1k"),
                        C("Cb", "b", "c", "1p"), R("R3", "c", "0", "1k")])
    an.match("Ca", "Cb")
    rep = an.bandwidth_report("V1", "c")
    (row,) = rep.rows
    (pair,) = rep.pairs
    assert pair.i == pair.j == row.name
    assert pair.term == pytest.approx(rep.b2)
    assert rep.b1 == pytest.approx(row.tau)
