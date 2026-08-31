"""The conductance a capacitor sees at its own terminals.

Where a pole's capacitance side reads and its conductance side does
not, the reconstruction names the remainder G_eff and stops. That
quantity is the driving-point conductance at the capacitor's port, and
the circuit can be asked for it instead of having it inferred from
second-order couplings. The algebra is

    Z = u^T G^-1 u = -det([[G, u], [u^T, 0]]) / det(G)

at s = 0, where every capacitor is an open -- which is what makes it
cheap: the s-degree drives the interpolation grid and here it is zero.
Both the value and the sign convention are pinned against a direct
numeric inverse, and the symbolic form must agree with the numeric one
at the operating point.
"""
import warnings
from pathlib import Path

import numpy as np
import pytest
import sympy as sp

from circuitinsight.analysis.portg import (eet_at_port,
                                            port_conductance)
from circuitinsight.engine.mna import S, build_mna, hybrid_split

FIX = Path(__file__).resolve().parent / "fixtures" / "spectre" / "miller"


@pytest.fixture(scope="module")
def bench():
    from circuitinsight.analysis.marginledger import _collect_stamps
    from circuitinsight.session import SessionController

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = SessionController.open(str(FIX / "tb_ota2s_stb.cin.json"),
                                   str(FIX / "psf_stb"))
        s.set_matches(*s.suggest_matches())
        probe = s.stb_probe() or s.probes[0]
        an = s._analyzer_ready()
        system = build_mna(an.primitives, an.flat.ground, probe,
                           an._alias)
    stamps, vals, _m = _collect_stamps(system, an.primitives, an._alias)
    return system, stamps, vals


def _numeric_port(system, u):
    """1 / (u^T G^-1 u) straight from a numeric inverse."""
    subs, _ = hybrid_split(system, [])
    G = system.A.xreplace(subs).xreplace({S: sp.Integer(0)})
    Gn = np.array(G.tolist(), dtype=complex)
    uv = np.asarray(u, dtype=complex)
    z = uv @ np.linalg.solve(Gn, uv)
    return 1.0 / z


def test_port_conductance_matches_a_direct_numeric_inverse(bench):
    """The determinant identity and its SIGN, against the obvious
    computation. A sign slip here would invert every conductance the
    aggregate is given."""
    system, stamps, _vals = bench
    u = stamps["CL"][0][1]
    num, den, val = port_conductance(system, u, [])
    assert num is not None and val is not None
    want = _numeric_port(system, u)
    assert val == pytest.approx(want.real, rel=1e-9)
    assert val > 0, "a passive port conductance is positive"


def test_the_symbolic_form_agrees_at_the_operating_point(bench):
    """Keeping the sensitivity-list conductances symbolic must not
    change the value -- only expose where it comes from."""
    system, stamps, vals = bench
    u = stamps["CL"][0][1]
    keep = [n for n in ("gds_I0_MN3", "gds_I0_MP2") if n in vals]
    assert keep
    num, den, val = port_conductance(system, u, keep)
    assert num is not None
    expr = num / den
    free = {str(x) for x in expr.free_symbols}
    assert set(keep) & free, "the kept symbols must survive"
    _n0, _d0, v0 = port_conductance(system, u, [])
    assert val == pytest.approx(v0, rel=1e-6)
    # and the kept symbols really move it: this is an OUTPUT node, so
    # raising an output conductance raises what the capacitor sees
    sub = {system.symbols[k]: sp.Float(vals[k]) for k in keep}
    hi = dict(sub)
    hi[system.symbols[keep[0]]] = sp.Float(vals[keep[0]] * 2)
    assert float(sp.re(sp.N(expr.xreplace(hi)))) > val


def test_a_degenerate_port_reports_nothing(bench):
    """A port with no incidence is not an error to guess around."""
    system, _stamps, _vals = bench
    num, den, val = port_conductance(system, np.zeros(system.A.rows), [])
    assert num is None and den is None and val is None


def test_opening_the_loop_needs_the_far_side_held(bench):
    """A loop-gain pole is an OPEN-loop quantity, so the port must be
    read with the loop cut at the probe. Cutting alone leaves the
    amplifier input floating with no DC path and the matrix goes
    singular, so the probe node that is not the port under test is
    held. All three states are pinned here because the difference
    between them is enormous: on a folded cascode the closed loop
    read 522 times the conductance the pole actually implies."""
    from circuitinsight.analysis.loopgain import _probe_nodes

    system, stamps, _vals = bench
    u = stamps["CL"][0][1]
    probe = "IPRB0"
    _n, _d, closed = port_conductance(system, u, [])
    _n, _d, opened = port_conductance(system, u, [], open_branch=probe)
    assert closed and opened
    # the loop's own low impedance dominates the closed reading
    assert closed > opened
    # and the held node is the one AWAY from the port
    p, q = _probe_nodes(system, probe)
    support = {i for i, x in enumerate(u) if x}
    assert (p in support) != (q in support), \
        "the probe must straddle the port for this bench"
    with pytest.raises(KeyError):
        port_conductance(system, u, [], open_branch="NOT_A_BRANCH")


def test_eet_interpolation_reproduces_the_direct_reading(bench):
    """The EET route never builds a symbolic determinant: it evaluates
    the two determinants NUMERICALLY at a few scaled values of one
    element and interpolates, which is exact because each element
    enters through a rank-one stamp. So the fitted polynomials, read
    back at the element's own value, must return exactly what the
    direct port computation gives -- and do it in a fraction of the
    time the symbolic form needs."""
    import numpy as npy

    system, stamps, vals = bench
    u = stamps["CL"][0][1]
    _n, _d, direct = port_conductance(system, u, [], open_branch="IPRB0")
    assert direct
    for sym in ("gds_I0_MN3", "gds_I0_MP2"):
        if sym not in vals:
            continue
        r = eet_at_port(system, u, sym, open_branch="IPRB0")
        assert r is not None
        # the degree IS the stamp count: one term per matched member
        assert r["degree"] == max(1, system.stamp_counts.get(sym, 1))
        back = (npy.polyval(r["num_poly"], r["value"])
                / npy.polyval(r["den_poly"], r["value"]))
        assert back == pytest.approx(direct, rel=1e-6), sym
        # removing an OUTPUT conductance must lower what the port sees
        assert r["y_removed"] < direct


def test_nested_eet_splits_a_port_into_parallel_branches(bench):
    """The first level asks what a port keeps when one element is
    removed; where two removals leave conductances that ADD to the
    whole, the port is two parallel branches. Whatever the topology,
    the arithmetic of the split has to hold and every branch form that
    is emitted must be dimensionally sound."""
    from circuitinsight.analysis.portg import branch_form, nested_eet

    system, stamps, vals = bench
    u = stamps["CL"][0][1]
    syms = [n for n in ("gds_I0_MN3", "gds_I0_MP2", "gm_I0_MP2",
                        "gds_I0_MN0") if n in vals]
    r = nested_eet(system, u, syms, open_branch="IPRB0")
    assert r and r["total"] > 0
    assert r["removed"] and all(v is not None
                                for v in r["removed"].values())
    if r["split"]:
        a, b = r["split"]["gates"]
        assert (r["removed"][a] + r["removed"][b]) == pytest.approx(
            r["total"], rel=0.05)
        for gate, info in r["branches"].items():
            assert info["conductance"] > 0
            f = branch_form(info, gate, vals)
            if f is not None:
                # a conductance branch nets exactly one conductance
                assert len(f["num"]) - len(f["den"]) == 1
                assert f["constant"] and f["constant"] > 0
                assert gate in f["num"]


def test_a_sum_is_recovered_from_the_null_and_the_pole():
    """Members of one sum imply the SAME total. A branch going as
    1/(x+c) has its pole at x = -c and one going as (x+c) has its null
    there, so each member implies value + |root|, and agreement across
    members is what proves them a sum rather than a product. Pinned on
    a synthetic reading so the rule stands on its own arithmetic."""
    from circuitinsight.analysis.portg import branch_form

    def rd(removed, shorted, null=(), pole=()):
        return {"y_removed": removed, "y_shorted": shorted,
                "g_null": list(null), "g_pole": list(pole)}

    vals = {"g_gate": 2.0, "a": 3.0, "b": 1.0, "d1": 8.0, "d2": 2.0}
    y = 2.0 * 4.0 / 10.0            # g_gate * (a + b) / (d1 + d2)
    info = {"conductance": y, "reads": {
        # a and b each null where the OTHER remains: one sum, total 4
        "a": rd(removed=y * 0.25, shorted=y * 9, null=[-1.0]),
        "b": rd(removed=y * 0.75, shorted=y * 9, null=[-3.0]),
        # d1 and d2 each pole where the other remains: one sum, total 10
        "d1": rd(removed=y * 5, shorted=0.0, pole=[-2.0]),
        "d2": rd(removed=y * 1.25, shorted=0.0, pole=[-8.0])}}
    f = branch_form(info, "g_gate", vals)
    assert f is not None, "a sum on each side must still read"
    # one net conductance: two numerator groups over one denominator
    assert f["text"].count("/") == 1
    assert "(a+b)" in f["text"] or "(b+a)" in f["text"]
    assert "d1+d2" in f["text"] or "d2+d1" in f["text"]
    # and the sums are recovered as totals, not as products
    assert f["constant"] == pytest.approx(1.0, rel=1e-6)


def test_port_expression_opens_the_loop_at_the_probe(monkeypatch, bench):
    """The session entry point MUST forward its probe.

    A driving-point conductance read with the feedback still closed is
    the conductance the loop presents, not the one the pole is made
    of. On the folded cascode the two differ by 522x, which makes
    every root report as not matching and the aggregate never expand.
    The algorithm grew `open_branch` for exactly this, and the API
    that calls it silently did not pass it -- so the contract is
    pinned here rather than left to the next reader to rediscover."""
    from circuitinsight.analysis import portg
    from circuitinsight.session import SessionController

    seen = {}
    real = portg.conductance_for_aggregate

    def spy(system, rec, stamps, g, **kw):
        seen.update(kw)
        return real(system, rec, stamps, g, **kw)

    monkeypatch.setattr(portg, "conductance_for_aggregate", spy)
    s = SessionController.open(str(FIX / "tb_ota2s_stb.cin.json"),
                               str(FIX / "psf_stb"))
    s.set_matches(*s.suggest_matches())
    probe = s.stb_probe() or s.probes[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            s.port_expression(probe, 1e6)
        except ValueError:
            pass                # no aggregate here; the call still ran
    assert seen, "port_expression never reached the port algebra"
    assert seen.get("probe") == probe, (
        "the loop was left CLOSED: probe not forwarded")


def test_the_eet_is_not_starved_by_the_determinant_cap():
    """The two routes need SEPARATE symbol lists.

    `keep` is capped to bound the determinants, whose cost explodes --
    on the folded cascode, 22 s at three symbols against 562 s at
    five. The EET costs milliseconds. Spending the determinant's
    budget on it is a category error with a QUIET symptom: starved of
    symbols, branch_form cannot name a branch and reports it unnamed,
    which reads like a limit of the method rather than a malformed
    question. Pinned on a synthetic reading so it asserts on every
    run, whatever a fixture happens to certify."""
    from circuitinsight.analysis.portg import eet_symbols

    # a cascode branch's worth of conductances, plus a capacitor that
    # must NOT be selected and a mover below the floor
    g = {"gds_NM14": 0.9, "gm_NM14": -0.7, "gmbs_NM14": -0.2,
         "gds_NM10": 0.05, "gds_PM11": 0.04, "gds_NM13": 0.03,
         "CL": -1.0, "dust": 1e-9}
    stamps = {n: [("g", None, None, "M")] for n in g if n != "CL"}
    stamps["CL"] = [("c", None, None, "C")]

    syms = eet_symbols(g, stamps)
    # every conductance that carries weight -- the three small ones
    # that FORM the sum included; that is the whole point
    for n in ("gds_NM14", "gm_NM14", "gmbs_NM14",
              "gds_NM10", "gds_PM11", "gds_NM13"):
        assert n in syms, n
    assert "CL" not in syms          # reactive: not a port conductance
    assert "dust" not in syms        # below the floor
    assert len(syms) > 3, "a determinant-sized cap cannot name a sum"
    # ordered by weight, and the cap still bounds it
    assert syms[0] == "gds_NM14"
    assert len(eet_symbols(g, stamps, cap=2)) == 2
