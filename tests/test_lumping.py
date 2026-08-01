"""Lumping parallel ground-referred passives into single symbols.

The claim is EXACTNESS: unlike AC-grounding (an approximation whose cost
we measure), folding parallel same-kind elements changes nothing about
the circuit -- only how many symbols describe it. So the central test is
that the response is bit-for-bit unchanged, and the rest guard the one
place it could stop being exact (mixing kinds)."""
import warnings
from pathlib import Path

import numpy as np

from circuitinsight import Analyzer
from circuitinsight.analysis import tearing
from circuitinsight.analysis.acground import scan_ac_grounds
from circuitinsight.analysis.lumping import lump_report, lump_to_ground

FIX = Path(__file__).resolve().parent / "fixtures" / "spectre"


def _node_cin():
    """Three caps and two resistors at one node, plus a floating cap."""
    inst = [
        {"name": "V1", "device_type": "vsource",
         "terminals": {"p": "in", "n": "0"}},
        {"name": "RS", "device_type": "resistor",
         "terminals": {"p": "in", "n": "x"}, "params": {"r": "1k"}},
        {"name": "C1", "device_type": "capacitor",
         "terminals": {"p": "x", "n": "0"}, "params": {"c": "1p"}},
        {"name": "C2", "device_type": "capacitor",
         "terminals": {"p": "x", "n": "0"}, "params": {"c": "2p"}},
        {"name": "C3", "device_type": "capacitor",
         "terminals": {"p": "0", "n": "x"}, "params": {"c": "3p"}},
        {"name": "R1", "device_type": "resistor",
         "terminals": {"p": "x", "n": "0"}, "params": {"r": "10k"}},
        {"name": "R2", "device_type": "resistor",
         "terminals": {"p": "x", "n": "0"}, "params": {"r": "10k"}},
        {"name": "CF", "device_type": "capacitor",
         "terminals": {"p": "in", "n": "x"}, "params": {"c": "5p"}},
    ]
    return {"cin_version": "0.1", "top": "main", "ground": ["0"],
            "definitions": {"main": {"ports": [], "instances": inst}}}


def test_lumping_is_exact_and_folds_per_kind():
    an = Analyzer.from_cin(_node_cin())
    rep = lump_report(an.primitives, an.flat.ground)
    by = {(g.node, g.kind): g for g in rep.groups}
    # capacitors and resistors at x lump SEPARATELY (a single primitive
    # cannot carry a mixed R+C admittance)
    assert set(by) == {("x", "c"), ("x", "r")}
    assert by[("x", "c")].value == 6e-12            # 1p + 2p + 3p
    assert by[("x", "r")].value == 5e3              # 10k || 10k
    assert rep.symbols_saved == 3                   # 3 caps + 2 res -> 2

    lumped, applied = lump_to_ground(an.primitives, an.flat.ground)
    assert len(lumped) == len(an.primitives) - applied.symbols_saved
    # the floating cap is untouched
    assert any(p.inst == "CF" for p in lumped)

    f = np.geomspace(1e3, 1e10, 30)
    a = tearing._numeric_response(an.primitives, an.flat.ground, "V1", "x",
                                  f)
    b = tearing._numeric_response(lumped, an.flat.ground, "V1", "x", f)
    assert np.max(np.abs(20 * np.log10(np.abs(b / a)))) < 1e-9


def test_lumping_shrinks_the_keep_grid():
    """The point for the solver: one C_node axis instead of three."""
    an = Analyzer.from_cin(_node_cin())
    lumped, rep = lump_to_ground(an.primitives, an.flat.ground)
    names = {p.inst for p in lumped}
    assert rep.groups
    assert any(n.startswith("Ceq_") for n in names)
    assert not {"C1", "C2", "C3"} & names           # folded away
    assert not {"R1", "R2"} & names


def test_grounding_unlocks_lumping_on_the_741():
    """The synergy, measured: AC-grounding the recommended bias nodes
    (0.018 dB) converts cross-coupled passives into ground-referred ones,
    and the lumping that follows is exact."""
    from circuitinsight.adapters.spectre import SpectreRun

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run = SpectreRun(FIX / "ua741" / "ua741.cin.json",
                         FIX / "ua741" / "psf")
        an = run.analyzer()
    prims, gnd = an.primitives, an.flat.ground
    before = lump_report(prims, gnd).symbols_saved

    rep = scan_ac_grounds(prims, gnd, "Vin", "22", alias=an._alias,
                          budget_db=0.1)
    assert rep.recommended
    grounded = tearing.ac_ground(prims, gnd, rep.recommended)
    after = lump_report(grounded, gnd).symbols_saved
    assert after > before, "grounding must expose more lumpable groups"

    lumped, applied = lump_to_ground(grounded, gnd)
    assert applied.symbols_saved == after
    f = np.geomspace(1.0, 1e8, 20)
    a = tearing._numeric_response(grounded, gnd, "Vin", "22", f, an._alias)
    b = tearing._numeric_response(lumped, gnd, "Vin", "22", f, an._alias)
    assert np.max(np.abs(20 * np.log10(np.abs(b / a)))) < 1e-9


def test_analyzer_exposes_the_report():
    an = Analyzer.from_cin(_node_cin())
    rep = an.lump_report()
    assert rep.symbols_saved == 3
    assert "Ceq_x" in rep.describe()


def test_grounding_turns_a_bias_transistor_into_a_load_impedance():
    """The two-stage OTA chain Arcadie pointed at: once vbn is an AC
    ground the output-stage current source has v_gs = 0, so its gm is
    dead; removing it is EXACT, and what is left of the device -- gds and
    its drain caps -- lumps with CL into one load impedance."""
    from circuitinsight.adapters.spectre import SpectreRun
    from circuitinsight.analysis.lumping import deactivate

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run = SpectreRun(FIX / "miller" / "tb_ota2s_stb.cin.json",
                         FIX / "miller" / "psf_stb")
        an = run.analyzer(cap_model="matrix")
    prims, gnd = an.primitives, an.flat.ground
    grounded = tearing.ac_ground(prims, gnd, ["vbn"])

    live, dead = deactivate(grounded, gnd)
    assert dead, "grounding vbn must kill the bias transconductances"
    assert any(p.inst == "I0.MN3" and p.param == "gm" for p in dead)

    f = np.geomspace(1e3, 1e10, 30)
    ref = tearing._numeric_response(grounded, gnd, "VIND", "vout", f,
                                    an._alias)
    got = tearing._numeric_response(live, gnd, "VIND", "vout", f,
                                    an._alias)
    assert np.max(np.abs(20 * np.log10(np.abs(got / ref)))) < 1e-9

    lumped, rep = lump_to_ground(live, gnd, nodes={"vout"})
    members = {m for g in rep.groups for m in g.members}
    assert "c_CL" in members and any("I0.MN3" in m for m in members)
    got2 = tearing._numeric_response(lumped, gnd, "VIND", "vout", f,
                                     an._alias)
    assert np.max(np.abs(20 * np.log10(np.abs(got2 / ref)))) < 1e-9
    assert len(lumped) < len(prims)
