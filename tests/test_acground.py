"""AC-ground scan: propose structurally, decide numerically.

The scan's whole claim is that its closed-form cost is EXACT -- so the
central test compares it against an independent two-solve measurement of
the same quantity, and the rest pin the physics it must get right (a
tail-bias gate is groundable, an active-load mirror gate is not, an
ideally driven node is not a candidate at all)."""
import warnings
from pathlib import Path

import numpy as np
import pytest

from circuitinsight.adapters.spectre import SpectreRun
from circuitinsight.analysis import tearing
from circuitinsight.analysis.acground import scan_ac_grounds

FIX = Path(__file__).resolve().parent / "fixtures" / "spectre"


def _an(name, cin, psf, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run = SpectreRun(FIX / name / cin, FIX / name / psf)
        return run.analyzer(**kw)


@pytest.fixture(scope="module")
def ota5t():
    return _an("ota5t", "tb_ota5t.cin.json", "psf", cap_model="matrix")


def test_predicted_cost_equals_the_independent_measurement(ota5t):
    """Sherman-Morrison closed form vs actually rebuilding the circuit and
    re-solving it: the two must agree, or the cheap scan is not usable."""
    freqs = np.geomspace(1.0, 1e10, 41)
    rep = scan_ac_grounds(ota5t.primitives, ota5t.flat.ground, "VIND",
                          "vout", freqs=freqs, alias=ota5t._alias)
    assert rep.candidates
    for c in rep.candidates:
        e = tearing.ac_ground_error(ota5t.primitives, ota5t.flat.ground,
                                    "VIND", "vout", [c.node], freqs,
                                    alias=ota5t._alias)
        assert c.worst_db == pytest.approx(e["worst_db"], rel=1e-6,
                                           abs=1e-9), c.node


def test_tail_bias_is_groundable_but_the_load_mirror_is_not(ota5t):
    """Both are diode-connected mirror gates; only the measurement can
    tell them apart -- the load mirror carries half the signal."""
    rep = scan_ac_grounds(ota5t.primitives, ota5t.flat.ground, "VIND",
                          "vout", alias=ota5t._alias, budget_db=0.1)
    by_node = {c.node: c for c in rep.candidates}
    assert by_node["vbn"].within_budget
    assert by_node["vbn"].worst_db < 0.01
    assert not by_node["I0.net1"].within_budget
    assert by_node["I0.net1"].worst_db > 1.0
    # and the structural labelling is right
    assert by_node["vbn"].kind == "mirror reference"
    assert "I0.MN2" in by_node["vbn"].controls
    assert rep.recommended == ["vbn"]


def test_ideally_driven_nodes_are_not_candidates(ota5t):
    """A node held by a source/balun cannot be 'declared' an AC ground --
    the driver wins, so grounding it is a no-op that would otherwise
    score a misleading 0 dB."""
    rep = scan_ac_grounds(ota5t.primitives, ota5t.flat.ground, "VIND",
                          "vout", alias=ota5t._alias)
    nodes = {c.node for c in rep.candidates}
    assert "vin_p" not in nodes and "vin_n" not in nodes
    assert "vdd!" not in nodes


def test_joint_set_error_is_measured_not_summed():
    """The 741: five bias nodes, recommended TOGETHER, with the joint cost
    computed by the Woodbury form -- and the set must actually deliver a
    balanced cut where none existed."""
    an = _an("ua741", "ua741.cin.json", "psf")
    rep = scan_ac_grounds(an.primitives, an.flat.ground, "Vin", "22",
                          alias=an._alias, budget_db=0.1)
    assert len(rep.recommended) >= 3
    assert rep.joint_db <= 0.1
    e = tearing.ac_ground_error(an.primitives, an.flat.ground, "Vin", "22",
                                rep.recommended,
                                np.geomspace(1.0, 1e10, 41),
                                alias=an._alias)
    assert e["worst_db"] == pytest.approx(rep.joint_db, rel=1e-6, abs=1e-9)

    before = tearing.rank_cuts(an.primitives, an.flat.ground, "1", "22",
                               keep_src={"Vin"})
    after = tearing.rank_cuts(tearing.ac_ground(an.primitives,
                                                an.flat.ground,
                                                rep.recommended),
                              an.flat.ground, "1", "22", keep_src={"Vin"})
    assert before[0]["balance"] < 0.2                  # hopeless as it is
    assert any(len(c["cut"]) == 1 for c in after)      # a ONE-node cut
    assert after[0]["balance"] > 0.5


def test_session_scan_is_cached():
    from circuitinsight.session import SessionController

    s = SessionController.open(FIX / "ota5t" / "tb_ota5t.cin.json",
                               FIX / "ota5t" / "psf")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = s.scan_ac_grounds("VIND", "vout")
        b = s.scan_ac_grounds("VIND", "vout")
    assert a is b
    assert "AC-ground scan" in a.describe()


def test_include_prices_a_non_structural_node():
    """A Nets-tree right-click must always come back with a price. The
    structural filter nominates only transconductor-control nodes; a
    passives-only junction was silently dropped ('unknown nets simply
    are not candidates') — exactly the node a user experiments with.
    `include` carries it through, priced by the same single inverse."""
    from circuitinsight.engine.primitives import Primitive

    prims = [
        Primitive("VIN", "", "vsrc", ("in", "0"), 0.0),
        Primitive("R1", "", "r", ("in", "mid"), 1e3),
        Primitive("R2", "", "r", ("mid", "out"), 1e3),
        Primitive("Cm", "", "c", ("mid", "0"), 1e-9),
        Primitive("RL", "", "r", ("out", "0"), 1e4),
    ]
    # structural-only: no transconductors, no candidates — the request
    # would have vanished
    rep0 = scan_ac_grounds(prims, ("0",), "VIN", "out")
    assert not rep0.candidates
    # include: the junction comes back WITH its measured price
    rep = scan_ac_grounds(prims, ("0",), "VIN", "out", include=("mid",))
    by = {c.node: c for c in rep.candidates}
    assert "mid" in by
    assert by["mid"].worst_db > 3.0        # grounding a divider mid is loud
    assert not by["mid"].within_budget     # and the verdict says so
    # a request for a held or unknown node is filtered, not an error
    rep2 = scan_ac_grounds(prims, ("0",), "VIN", "out",
                           include=("in", "nosuch"))
    assert all(c.node not in ("in", "nosuch") for c in rep2.candidates)


def test_include_survives_the_report_cap(ota5t):
    """The requested node must appear even when the cap would have cut
    it: ask for the most expensive candidate with max_report=1."""
    full = scan_ac_grounds(ota5t.primitives, ota5t.flat.ground, "VIND",
                           "vout", structural_only=False, max_report=99)
    worst = full.candidates[-1].node
    rep = scan_ac_grounds(ota5t.primitives, ota5t.flat.ground, "VIND",
                          "vout", structural_only=False, max_report=1,
                          include=(worst,))
    assert any(c.node == worst for c in rep.candidates)
