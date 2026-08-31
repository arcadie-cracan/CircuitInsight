"""The ray test: what a SHARED verdict has to prove.

"No partition stood" is an absence. The verdict a designer can act on
needs evidence -- where the root's attribution actually lives, and
whether any element gathers it somewhere. The three answers differ in
what they ask of the designer: a localized root means the READER has a
gap, a collapsible one names the direction that would condition it,
and a distributed one says no resizing will.
"""
import warnings
from pathlib import Path

import numpy as np
import pytest

from circuitinsight.analysis import rays

FIX = Path(__file__).resolve().parent / "fixtures" / "spectre" / "miller"


def test_a_floating_element_is_its_own_location():
    """A compensation capacitor localizes a pole onto a PORT. Splitting
    its weight between the two ends would make every Miller-compensated
    pole read as distributed by construction, so a floating element
    counts as the node pair it spans."""
    node_index = {"a": 0, "b": 1}
    stamps = {"Cab": [("c", np.array([1.0, -1.0]), None, "X")],
              "Ca": [("c", np.array([1.0, 0.0]), None, "Y")]}
    sh = rays.node_shares({"Cab": -0.7, "Ca": -0.3}, stamps, node_index)
    assert set(sh) == {"a ↔ b", "a"}
    loc, place = rays.localization(sh)
    assert place == "a ↔ b" and loc == pytest.approx(0.7)


def test_conductances_do_not_dilute_the_node_measure():
    """Where a pole LIVES is a statement about node capacitance; the
    conductances spread over branches and only blur it."""
    node_index = {"a": 0, "b": 1}
    stamps = {"C": [("c", np.array([1.0, 0.0]), None, "X")],
              "G": [("g", np.array([1.0, -1.0]), None, "Y")]}
    sh = rays.node_shares({"C": -1.0, "G": 5.0}, stamps, node_index)
    assert set(sh) == {"a"} and sh["a"] == pytest.approx(1.0)


@pytest.fixture(scope="module")
def bench():
    from circuitinsight.analysis.marginledger import (_collect_stamps,
                                                      _tian_context)
    from circuitinsight.engine.mna import build_mna
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
        led = s.margin_ledger(probe)["ledger"]
    return (s, probe, an, system, led,
            _tian_context(system, probe),
            _collect_stamps(system, an.primitives, an._alias))


def test_a_ray_follows_the_root_without_resolving(bench):
    """The pole is tracked by Newton on the Eq. 30 denominator as the
    element walks, so a whole ray costs inverses rather than solves --
    and it must actually MOVE, monotonically, with a load capacitance."""
    _s, probe, an, system, led, _ctx, (stamps, _v, _m) = bench
    e = min((x for x in led.entries
             if x.kind == "pole" and x.f0_hz > 0), key=lambda x: x.f0_hz)
    s0 = 2 * np.pi * complex(max(e.roots, key=lambda r: r.imag))
    path = rays.ray(system, probe, an.primitives, an._alias, "CL", s0,
                    factors=(0.25, 4.0))
    assert len(path) == 2
    up = next(p for p in path if p.scale > 1)
    down = next(p for p in path if p.scale < 1)
    # a bigger load capacitance can only lower an output pole
    assert up.f0_hz < e.f0_hz < down.f0_hz
    assert all(0.0 <= p.localization <= 1.0 for p in path)


def test_the_verdict_is_three_valued_and_carries_its_evidence(bench):
    """Whatever the answer, it must name a node, a number, and -- when
    it claims collapsibility -- the ray that does the gathering."""
    s, probe, an, system, led, (fn, Q, ib, ip), (stamps, vals, _m) = bench
    from circuitinsight.analysis.secondorder import loop_pole_jet

    e = min((x for x in led.entries
             if x.kind == "pole" and x.f0_hz > 0), key=lambda x: x.f0_hz)
    s0 = 2 * np.pi * complex(max(e.roots, key=lambda r: r.imag))
    g, _H = loop_pole_jet(fn, Q, ib, ip, stamps, vals, s0)
    v = rays.assess(system, probe, an.primitives, an._alias, g, stamps,
                    s0, top=2, factors=(0.2, 5.0))
    assert v.kind in ("localized", "collapsible", "distributed")
    assert v.node_nominal and 0.0 <= v.l_nominal <= 1.0
    assert v.describe()
    if v.kind == "collapsible":
        assert v.best_symbol and v.path
        assert v.best_localization > v.l_nominal
    # the dominant pole of a Miller-compensated stage is owned by the
    # compensated node, so it must not read as distributed
    assert v.kind != "distributed"


def test_session_exposes_the_assessment_on_demand(bench):
    """A ray sweep costs seconds where the ledger stays interactive,
    so it is a verb the designer asks for, not part of every solve."""
    s, probe, _an, _system, led, _ctx, _st = bench
    e = min((x for x in led.entries
             if x.kind == "pole" and x.f0_hz > 0), key=lambda x: x.f0_hz)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        v = s.assess_root(probe, e.f0_hz)
    assert v.kind in ("localized", "collapsible", "distributed")
    assert v.describe()


def test_the_verdict_does_not_overclaim_against_the_reader():
    """Two calibration bugs, both of which accused the reader falsely.

    The `localized` bar is 0.60, but the message blamed the reader for
    a missing formula at every level above it. At 68% a third of the
    weight is elsewhere and a single-location formula would be
    materially wrong, so the reader declining is CORRECT -- and the
    accusation sent a reader hunting a bug that was not there. Only a
    root OWNED by its location earns that.

    And the ray is computed without reference to the reconstruction,
    so it cannot know whether a formula exists. It used to assert
    there was none, including about roots the reader had named."""
    from circuitinsight.analysis.rays import RayVerdict

    owned = RayVerdict(kind="localized", l_nominal=0.99,
                       node_nominal="net1 ↔ vout")
    mixed = RayVerdict(kind="localized", l_nominal=0.68,
                       node_nominal="mirr ↔ vdd!")

    # owned: the reader is on the hook, but only CONDITIONALLY -- the
    # verdict cannot see the reconstruction
    assert "gap in the reader" in owned.describe()
    assert "if no formula" in owned.describe()
    assert "no formula named it, which is" not in owned.describe()

    # merely centred: no accusation, and the spread is quantified
    assert "gap in the reader" not in mixed.describe()
    assert "32%" in mixed.describe()
    assert "would not be exact" in mixed.describe()

    # the band itself
    from circuitinsight.analysis.rays import _OWNED
    assert 0.60 < _OWNED <= 1.0, "owning must be stricter than localized"
