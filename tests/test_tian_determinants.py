"""The five Tian determinants, exposed for parametric studies.

loop_gain's numeric path assembles Eq. 30 from five Cramer
determinants and throws them away. Exposing them lets a parametric
study exploit what the assembled rational hides: each determinant is a
polynomial of degree d_i in element i, where d_i is that element's
stamp count, so a fixed, small grid of determinant reads pins the
whole parametric family exactly -- instead of one assembled solve per
parameter point, whose symbolic cost is exponential in kept symbols.
"""
import warnings
from pathlib import Path

import sympy as sp

from circuitinsight.adapters.spectre import SpectreRun
from circuitinsight.engine.mna import S, build_mna
from circuitinsight.analysis.loopgain import (loop_gain, tian_combine,
                                              tian_determinants)

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "spectre" / "miller"
PROBE = "IPRB0"


def _system():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run = SpectreRun(FIXTURES / "tb_ota2s_stb.cin.json",
                         FIXTURES / "psf_stb")
        an = run.analyzer(cap_model="matrix")
    return an, build_mna(an.primitives, an.flat.ground, PROBE, an._alias)


def test_determinants_reassemble_the_numeric_loop_gain():
    """The exposed pieces ARE what loop_gain combines: same rational."""
    an, system = _system()
    num, den = tian_combine(tian_determinants(system, PROBE))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        T = loop_gain(system, PROBE, [])
    assert sp.simplify(sp.together(-num / den - T.expr)) == 0


def test_determinant_degree_equals_the_stamp_count():
    """The load-bearing structural fact: a determinant is degree d_i in
    element i, d_i being its stamp count -- 1 for a lone element, one
    per member for a symbol SHARED by matched devices. This is what
    fixes how many reads a parametric model needs per dimension."""
    an, system = _system()
    name = "CL"                                   # a lone load cap
    assert system.stamp_counts[name] == 1
    rows = []
    for scale in (1, 2, 4):                       # powers of two: exact
        sysk = build_mna(an.primitives, an.flat.ground, PROBE, an._alias)
        sysk.values[name] = sysk.values[name] * scale
        rows.append(sp.Poly(tian_determinants(sysk, PROBE)["Dt"],
                            S).all_coeffs())
    w = max(len(r) for r in rows)
    M = sp.Matrix([[sp.Integer(0)] * (w - len(r)) + list(r)
                   for r in rows])
    # rank 2 == degree 1 == affine: two reads pin the whole family
    assert M.rank() == 2


def test_dt_divides_both_eq30_parts_exactly():
    """The ghost population is one factor of Dt: it divides the Eq. 30
    numerator and denominator with zero remainder, so a parametric
    study can deflate exactly instead of cancelling numerically."""
    _an, system = _system()
    d = tian_determinants(system, PROBE)
    num, den = tian_combine(d)
    Dt = sp.Poly(d["Dt"], S)
    for part in (num, den):
        _q, r = sp.div(sp.Poly(part, S), Dt)
        assert r.is_zero
