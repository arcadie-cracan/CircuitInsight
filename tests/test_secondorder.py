"""The level-2 primitive against exact symbolic differentiation.

logdet_jet claims the first and second log-log derivatives of
ln det A(s) from ONE inverse and a K x K inner-product matrix. Here a
small synthetic system with rank-one stamps is differentiated exactly
by sympy and the two must agree to machine precision -- including the
cases that carry the structure of the real problem: a symbol SHARED by
two stamps (matched devices), a capacitive stamp whose factor carries
s, and a RESISTIVE stamp, the only kind whose admittance is nonlinear
in its own parameter and so the only one with a second-derivative
term of its own.
"""
import numpy as np
import pytest
import sympy as sp
from pathlib import Path

from circuitinsight.analysis.secondorder import (coupling_ratios,
                                                 loop_pole_jet,
                                                 logdet_jet)


def _fixture():
    """A 4x4 base plus four rank-one stamps over three symbols."""
    dim = 4                                    # used by vec() below
    A0 = sp.Matrix([[7, 1, 0, 2],
                    [1, 5, 1, 0],
                    [0, 1, 6, 1],
                    [2, 0, 1, 8]]) / 10
    v = {}

    def vec(*ix):
        out = np.zeros(dim)
        for i, sgn in ix:
            out[i] = sgn
        return out

    # gA is MATCHED: two stamps, one symbol -- degree two, not affine
    stamps = {
        "gA": [("g", vec((0, 1.0), (1, -1.0)), vec((0, 1.0), (1, -1.0)),
                "MA0"),
               ("g", vec((1, 1.0), (2, -1.0)), vec((1, 1.0), (2, -1.0)),
                "MA1")],
        "cB": [("c", vec((2, 1.0)), vec((2, 1.0)), "CB")],
        "rC": [("r", vec((3, 1.0), (0, -1.0)), vec((3, 1.0), (0, -1.0)),
                "RC")],
    }
    vals = {"gA": 3e-4, "cB": 2e-13, "rC": 4e3}
    return A0, stamps, vals, v


def _exact(A0, stamps, vals, s_val):
    """Exact log-log jet of ln det by symbolic differentiation."""
    S = sp.Symbol("s")
    syms = {n: sp.Symbol(n, positive=True) for n in vals}
    A = A0 + sp.zeros(*A0.shape)
    for name, sts in stamps.items():
        th = syms[name]
        for kind, a, b, _inst in sts:
            Y = {"g": th, "c": S * th, "r": 1 / th}[kind]
            A = A + Y * sp.Matrix(a.reshape(-1, 1)) * sp.Matrix(
                b.reshape(1, -1))
    L = sp.log(A.det())
    sub = {syms[n]: sp.Float(v, 30) for n, v in vals.items()}
    sub[S] = sp.nsimplify(s_val.real) + sp.I * sp.nsimplify(s_val.imag)
    g, H = {}, {}
    for n1 in vals:
        d1 = sp.diff(L, syms[n1])
        g[n1] = complex((syms[n1] * d1).xreplace(sub).evalf(30))
        for n2 in vals:
            d2 = sp.diff(d1, syms[n2])
            val = (syms[n1] * syms[n2] * d2).xreplace(sub).evalf(30)
            v = complex(val)
            if n1 == n2:
                v += g[n1]
            H[n1, n2] = v
    return g, H


def _numeric_M(A0, stamps, vals, s_val):
    A = np.array(A0.tolist(), dtype=complex)
    for name, sts in stamps.items():
        th = vals[name]
        for kind, a, b, _inst in sts:
            Y = {"g": th, "c": s_val * th, "r": 1.0 / th}[kind]
            A = A + Y * np.outer(a, b)
    return np.linalg.inv(A)


def test_logdet_jet_matches_exact_symbolic_derivatives():
    A0, stamps, vals, _ = _fixture()
    s_val = 2j * np.pi * 1e6
    M = _numeric_M(A0, stamps, vals, s_val)
    g, H = logdet_jet(M, stamps, vals, s_val)
    ge, He = _exact(A0, stamps, vals, s_val)

    for n in vals:
        assert abs(g[n] - ge[n]) <= 1e-9 * max(abs(ge[n]), 1e-12), n
    for n1 in vals:
        for n2 in vals:
            assert abs(H[n1, n2] - He[n1, n2]) <= \
                1e-9 * max(abs(He[n1, n2]), 1e-12), (n1, n2)


def test_hessian_is_symmetric_and_both_orders_resolve():
    A0, stamps, vals, _ = _fixture()
    s_val = 2j * np.pi * 1e6
    M = _numeric_M(A0, stamps, vals, s_val)
    _g, H = logdet_jet(M, stamps, vals, s_val)
    for n1 in vals:
        for n2 in vals:
            assert H[n1, n2] == H[n2, n1]


def test_single_conductance_follows_the_sum_member_signature():
    """One lone conductance makes ln det an affine-in-theta log, so
    g = u and H = u(1 - u) exactly: the same share-times-one-minus-
    share signature a SUM MEMBER shows. The quantization the whole
    programme rests on is visible in the smallest possible case."""
    A0 = sp.Matrix([[4, 1, 0], [1, 3, 1], [0, 1, 5]]) / 10
    a = np.array([1.0, -1.0, 0.0])
    stamps = {"g1": [("g", a, a, "M0")]}
    vals = {"g1": 7e-4}
    s_val = 2j * np.pi * 1e5
    A = np.array(A0.tolist(), dtype=complex) + vals["g1"] * np.outer(a, a)
    g, H = logdet_jet(np.linalg.inv(A), stamps, vals, s_val)
    u = g["g1"]
    assert abs(H["g1", "g1"] - u * (1 - u)) <= 1e-12 * abs(u)


MILLER = Path(__file__).resolve().parent / "fixtures" / "spectre" / "miller"

# rho of the miller 24.5 MHz pair, measured independently by finite
# differences on exactly-perturbed hybrid solves at mpmath dps 60
# (fc_local/jet_probe.py). Printed to two decimals, which is the
# tolerance this can be checked to.
FD_TRUTH = {("gm_I0_MP1", "gm_I0_MP2"): -0.01,
            ("gm_I0_MP1", "I0_Cc"): +0.02,
            ("gm_I0_MP1", "CL"): -0.02,
            ("gm_I0_MP1", "cgs_I0_MP1"): -0.03,
            ("gm_I0_MP2", "I0_Cc"): +0.07,
            ("gm_I0_MP2", "CL"): -0.10,
            ("gm_I0_MP2", "cgs_I0_MP2"): -0.09,
            ("I0_Cc", "CL"): -0.31,
            ("CL", "cgs_I0_MP2"): +1.29,
            ("cgs_I0_MP1", "cgs_I0_MP2"): -0.02}


@pytest.fixture(scope="module")
def miller_jet():
    """(ledger, context, stamps, vals) of the two-stage loop gain."""
    import warnings

    from circuitinsight.analysis.marginledger import (_collect_stamps,
                                                      _tian_context)
    from circuitinsight.engine.mna import build_mna
    from circuitinsight.session import SessionController

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = SessionController.open(str(MILLER / "tb_ota2s_stb.cin.json"),
                                   str(MILLER / "psf_stb"))
        s.set_matches(*s.suggest_matches())
        probe = s.stb_probe() or s.probes[0]
        an = s._analyzer_ready()
        system = build_mna(an.primitives, an.flat.ground, probe,
                           an._alias)
        led = s.margin_ledger(probe)["ledger"]
    return (led, _tian_context(system, probe),
            *_collect_stamps(system, an.primitives, an._alias)[:2])


def _jet_of(entry, ctx, stamps, vals):
    fn, Q, ib, ip = ctx
    x = max(entry.roots, key=lambda r: r.imag)
    return loop_pole_jet(fn, Q, ib, ip, stamps, vals,
                         2 * np.pi * complex(x))


def test_pole_jet_first_order_matches_the_ledger(miller_jet):
    """The analytic first derivatives must reproduce the ledger's own,
    computed by entirely separate code. The ledger reads 1e-6 off the
    root to keep its diverging ratio finite; this evaluates AT the
    root, where the Eq. 30 denominator is merely zero, so agreement to
    that offset is the expected -- and best possible -- result."""
    led, ctx, stamps, vals = miller_jet
    e = min((x for x in led.entries
             if x.kind == "pole" and x.f0_hz > 0 and x.root_sens),
            key=lambda x: x.f0_hz)
    g, _H = _jet_of(e, ctx, stamps, vals)
    checked = 0
    for m in e.root_sens[:8]:
        if abs(m["df0"]) < 1e-3:
            continue
        assert m["symbol"] in g
        assert abs(g[m["symbol"]].real - m["df0"]) < 1e-4, m["symbol"]
        checked += 1
    assert checked >= 4


def test_pole_jet_rho_matches_finite_difference_truth(miller_jet):
    """The second-order claim, against a measurement made by a wholly
    different route: perturbed hybrid solves rooted at 60 digits."""
    led, ctx, stamps, vals = miller_jet
    e = next(x for x in led.entries if x.pair)
    g, H = _jet_of(e, ctx, stamps, vals)
    rho = coupling_ratios(g, H, floor=0.0)
    for (a, b), ref in FD_TRUTH.items():
        got = rho.get((a, b), rho.get((b, a)))
        assert got is not None, (a, b)
        assert abs(got - ref) <= 0.02, (a, b, got, ref)


def test_miller_dominant_pole_grammar_is_read_from_the_jet(miller_jet):
    """The structural payoff, on a root whose textbook formula is not
    in doubt. The Miller-compensated two-stage has

        f_p1 = (gds_MN0 + gds_MP1)(gds_MN3 + gds_MP2)
               / (2 pi gm_MP2 Cc)

    i.e. p1 = 1/(gm2 Ro1 Ro2 Cc) with each output resistance a
    PARALLEL pair. The jet must reproduce that grammar with no
    candidate vocabulary and no fitting: exponent -1 on gm and Cc,
    zero coupling across factors, -1 within each conductance sum, and
    shares that close to one inside each sum."""
    led, ctx, stamps, vals = miller_jet
    e = min((x for x in led.entries
             if x.kind == "pole" and x.f0_hz > 0 and x.root_sens),
            key=lambda x: x.f0_hz)
    g, H = _jet_of(e, ctx, stamps, vals)
    rho = coupling_ratios(g, H, floor=0.0)

    def r(a, b):
        return rho.get((a, b), rho.get((b, a)))

    stage1 = ("gds_I0_MN0", "gds_I0_MP1")
    stage2 = ("gds_I0_MN3", "gds_I0_MP2")
    # the two monomial factors: exponent -1, coupled to nothing
    for lone in ("I0_Cc", "gm_I0_MP2"):
        assert abs(g[lone].real + 1.0) < 0.02, lone
        for other in stage1 + stage2:
            assert abs(r(lone, other)) < 0.05, (lone, other)
    assert abs(r("I0_Cc", "gm_I0_MP2")) < 0.05
    # each output conductance is a SUM: rho = -1 inside, 0 across
    for sm in (stage1, stage2):
        assert abs(r(*sm) + 1.0) < 0.05, sm
        assert abs(sum(g[x].real for x in sm) - 1.0) < 0.02, sm
    for a in stage1:
        for b in stage2:
            assert abs(r(a, b)) < 0.05, (a, b)


def test_coupling_ratios_skip_vanishing_denominators():
    A0, stamps, vals, _ = _fixture()
    s_val = 2j * np.pi * 1e6
    M = _numeric_M(A0, stamps, vals, s_val)
    g, H = logdet_jet(M, stamps, vals, s_val)
    rho = coupling_ratios(g, H, floor=1e9)      # nothing can clear this
    assert rho == {}
    rho = coupling_ratios(g, H, floor=0.0)
    assert set(rho) == {("cB", "gA"), ("cB", "rC"), ("gA", "rC")}
