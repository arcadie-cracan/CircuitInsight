"""Spectre adapter: psfascii parsing, canonical OP mapping, join, end-to-end TF.

All against the real SKY130 fixtures captured in M0. SKY130's nfet_01v8/
pfet_01v8 are sky130_fd_pr subckt wrappers, so a MOSFET's raw dcOpInfo key
carries the expanded primitive name; the join resolves it back to the CIN
instance, and post-join op_data is keyed by the plain instance name.
"""
import math
from pathlib import Path

import pytest

from circuitinsight.adapters.cin import flatten, parse_cin
from circuitinsight.adapters.spectre import SpectreError, SpectreRun, join_op
from circuitinsight.adapters.spectre.opdata import load_dcopinfo
from circuitinsight.adapters.spectre.psfascii import parse_psfascii

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "spectre" / "m0"
NFET = "M1.msky130_fd_pr__nfet_01v8"    # raw dcOpInfo keys (subckt-wrapper form)
PFET = "M2.msky130_fd_pr__pfet_01v8"


# ------------------------------------------------------------------ psfascii

def test_parse_dcopinfo_structs():
    psf = parse_psfascii(FIXTURES / "psf" / "dcOpInfo.info")
    assert len(psf.entries) == 10
    m1 = psf.entries[NFET]
    assert m1.type_name == "bsim4"
    assert m1.value["gm"] == pytest.approx(4.177149904678507e-04)
    assert m1.value["region"] == 2
    assert math.isnan(m1.value["ft"])   # unreported fields come through as nan
    assert m1.props["model"].startswith("M1.sky130_fd_pr__nfet_01v8__model")
    assert psf.entries["RLP"].value["res"] == pytest.approx(1e4)


def test_parse_dcop_dc_scalars():
    psf = parse_psfascii(FIXTURES / "psf" / "dcOp.dc")
    assert psf.entries["vout"].type_name == "V"
    assert psf.entries["vout"].value == pytest.approx(1.36885020245337)
    assert psf.entries["VDD:p"].type_name == "I"


# ------------------------------------------------------------ canonical map

def test_canonical_mosfet_caps():
    records = load_dcopinfo(FIXTURES / "psf" / "dcOpInfo.info")
    m1 = records[NFET]
    assert m1.device_type == "mosfet"
    assert m1.params["gm"] == pytest.approx(4.177149904678507e-04)
    # magnitudes of negative trans-caps
    assert m1.params["cgs"] == pytest.approx(abs(m1.raw["cgs"]))
    assert m1.raw["cgs"] < 0
    # cdb = junction + |intrinsic|
    assert m1.params["cdb"] == pytest.approx(abs(m1.raw["cjd"]) + abs(m1.raw["cdb"]))
    # sentinel ints and nan never become params
    assert "OPdef" not in m1.params
    r = records["RL"]
    assert r.device_type == "resistor" and r.params["r"] == pytest.approx(1e4)


# ------------------------------------------------------------------- join

def test_join_full_testbench():
    run = SpectreRun(FIXTURES / "tb_m0.cin.json", FIXTURES / "psf")
    assert set(run.op_data) == {"M1", "M2", "RD1", "RD2", "RL", "RLP",
                                "VDD", "VIN", "VINP", "C0"}
    assert run.op_data["C0"]["c"] == pytest.approx(1e-13)
    assert run.op_data["M2"]["gm"] == pytest.approx(3.200512e-04, rel=1e-6)


def test_join_missing_record_is_hard_error():
    flat = flatten(parse_cin({
        "cin_version": "0.1", "top": "main", "ground": ["0"],
        "definitions": {"main": {"ports": [], "instances": [
            {"name": "RX", "device_type": "resistor",
             "terminals": {"p": "a", "n": "0"}},
        ]}}}))
    with pytest.raises(SpectreError, match="RX: no OP record"):
        join_op(flat, load_dcopinfo(FIXTURES / "psf" / "dcOpInfo.info"))


def test_join_orphan_record_is_hard_error():
    run_records = load_dcopinfo(FIXTURES / "psf" / "dcOpInfo.info")
    flat = flatten(parse_cin({
        "cin_version": "0.1", "top": "main", "ground": ["0"],
        "definitions": {"main": {"ports": [], "instances": [
            {"name": "RL", "device_type": "resistor",
             "terminals": {"p": "a", "n": "0"}},
        ]}}}))
    with pytest.raises(SpectreError, match="matched no CIN device"):
        join_op(flat, run_records)


def test_join_type_mismatch_is_hard_error():
    # lie about a passive, whose OP record matches by exact name, so the direct
    # model-class check fires (SKY130 MOSFETs are wrappers -- covered below)
    records = load_dcopinfo(FIXTURES / "psf" / "dcOpInfo.info")
    flat = flatten(parse_cin({
        "cin_version": "0.1", "top": "main", "ground": ["0"],
        "definitions": {"main": {"ports": [], "instances": [
            {"name": "RL", "device_type": "capacitor",    # lies about RL
             "terminals": {"p": "a", "n": "0"}},
        ]}}}))
    with pytest.raises(SpectreError, match="model 'resistor'"):
        join_op(flat, records, allow_unmatched=tuple(n for n in records if n != "RL"))


def test_join_wrapper_type_mismatch_is_hard_error():
    # a MOSFET declared as the wrong type can't bind its subckt-wrapper record
    # (no same-type expansion candidate) -- a hard error unique to SKY130 wrappers
    records = load_dcopinfo(FIXTURES / "psf" / "dcOpInfo.info")
    flat = flatten(parse_cin({
        "cin_version": "0.1", "top": "main", "ground": ["0"],
        "definitions": {"main": {"ports": [], "instances": [
            {"name": "M1", "device_type": "resistor",     # lies about M1
             "terminals": {"p": "a", "n": "0"}},
        ]}}}))
    with pytest.raises(SpectreError, match="ambiguous wrapper expansion"):
        join_op(flat, records, allow_unmatched=tuple(records))


def test_join_hierarchical_names():
    # matches tb_m0_hier.scs: csstage(M1, RL) instantiated as I0
    flat = flatten(parse_cin({
        "cin_version": "0.1", "top": "tb", "ground": ["0"],
        "definitions": {
            "csstage": {"ports": ["out", "in", "vddl"], "instances": [
                {"name": "M1", "device_type": "mosfet",
                 "terminals": {"d": "out", "g": "in", "s": "0", "b": "0"},
                 "params": {"polarity": "n"}},
                {"name": "RL", "device_type": "resistor",
                 "terminals": {"p": "vddl", "n": "out"}},
            ]},
            "tb": {"ports": [], "instances": [
                {"name": "VDD", "device_type": "vsource",
                 "terminals": {"p": "vdd", "n": "0"}},
                {"name": "VIN", "device_type": "vsource",
                 "terminals": {"p": "vin", "n": "0"}},
                {"name": "I0", "subckt": "csstage",
                 "terminals": {"out": "vout", "in": "vin", "vddl": "vdd"}},
            ]},
        }}))
    op = join_op(flat, load_dcopinfo(FIXTURES / "psf_hier" / "dcOpInfo.info"))
    assert op["I0.M1"]["gm"] > 0
    assert op["I0.RL"]["r"] == pytest.approx(1e4)


# ------------------------------------------------------------- end to end

def test_end_to_end_nmos_stage_tf():
    run = SpectreRun(FIXTURES / "tb_m0.cin.json", FIXTURES / "psf")
    # hybrid keep=[]: exact-rational numeric solve; full symbolic on the whole
    # testbench (~15 symbols incl. 1/R fraction fields) is the documented
    # explosion case and takes minutes
    H = run.analyzer().tf("VIN", "vout", keep=[])

    p = run.op_data["M1"]
    gout = p["gds"] + 1 / run.op_data["RL"]["r"]
    cout = run.op_data["C0"]["c"] + p["cgd"] + p["cdb"]

    a0 = complex(H.numeric([0.01])[0])
    assert a0.real == pytest.approx(-p["gm"] / gout, rel=1e-9)

    (p1,) = sorted(H.poles(), key=abs)[:1]
    assert abs(p1) == pytest.approx(gout / (2 * math.pi * cout), rel=1e-6)

    # RHP zero from cgd feedforward
    (z1,) = H.zeros()
    assert z1.real == pytest.approx(p["gm"] / (2 * math.pi * p["cgd"]), rel=1e-6)


def test_end_to_end_hierarchical_exporter_cin():
    # exporter's hierarchical CIN (tb_m2 schematic: csstage wrapped as I7)
    # joined against the matching Spectre run — no rename needed at all
    run = SpectreRun(FIXTURES / "tb_hier.cin.json", FIXTURES / "psf_tbm2")
    assert "I7.M1" in run.op_data and "I7.RL" in run.op_data
    H = run.analyzer().tf("VIN", "net1", keep=[])
    p = run.op_data["I7.M1"]
    gout = p["gds"] + 1 / run.op_data["I7.RL"]["r"]
    a0 = complex(H.numeric([0.01])[0])
    assert a0.real == pytest.approx(-p["gm"] / gout, rel=1e-9)


def test_end_to_end_pmos_stage_tf():
    run = SpectreRun(FIXTURES / "tb_m0.cin.json", FIXTURES / "psf")
    H = run.analyzer().tf("VINP", "voutp", keep=[])
    p = run.op_data["M2"]
    gout = p["gds"] + 1 / run.op_data["RLP"]["r"]
    a0 = complex(H.numeric([0.01])[0])
    assert a0.real == pytest.approx(-p["gm"] / gout, rel=1e-9)


FIX_SP = Path(__file__).resolve().parent / "fixtures" / "spectre"


def _op_tree(tmp_path, n=2):
    """A synthetic maestro ExplorerRun tree: n design points, each a
    copy of the miller stb psf plus the runObjFile/variables_file that
    label it."""
    import shutil

    src = FIX_SP / "miller" / "psf_stb"
    corners = ["tt 27", "nom", "nom"]
    temps = [27, -40, 85]
    secs = ["tt", "ss", "ff"]
    root = tmp_path / "ExplorerRun.0"
    for i in range(1, n + 1):
        psf = root / str(i) / "work_tb" / "psf"
        shutil.copytree(src, psf)
        (psf / "runObjFile").write_text(
            'HEADER\n"PSFversion" "1.00"\n'
            f'"#maeCorner" "{corners[i - 1]}"\n'
            f' "#maeDesignPoint" {i}\n')
        (psf / "variables_file").write_text(
            'HEADER\nVALUE\n'
            f'"temperature" "variable" ( {temps[i - 1]} ) PROP( )\nEND\n')
        nl = psf.parent / "netlist"
        nl.mkdir()
        (nl / ".modelFiles").write_text(
            f'include "models.spice" section={secs[i - 1]}\n')
    return root


def test_op_point_discovery_labels_the_corners(tmp_path):
    """A maestro corner tree yields one labeled point per design point;
    a plain psf directory yields exactly one, so opens stay uniform."""
    from circuitinsight.adapters.spectre.points import discover_op_points

    root = _op_tree(tmp_path, 3)
    pts = discover_op_points(root)
    assert [p.label for p in pts] == ["tt 27", "nom ss -40", "nom ff 85"]
    assert [p.index for p in pts] == [1, 2, 3]
    assert all((p.psf / "dcOpInfo.info").exists() for p in pts)
    assert pts[1].variables["temperature"] == -40.0
    assert pts[0].process == "tt"    # in the corner name, not repeated
    # the netlister's toolArgs.ils keeps the corner name the USER wrote
    # even where runObjFile recorded "nom" (fc field case: every 125C
    # skew point); when present it wins, and a truthful name needs no
    # section or temperature appended
    (root / "2" / "work_tb" / "netlist" / "toolArgs.ils").write_text(
        '((("sim" "spectre")) nil)'
        '(("CornerName" "ss -40" "temperature" "-40"))(2)')
    pts = discover_op_points(root)
    assert [p.label for p in pts] == ["tt 27", "ss -40", "nom ff 85"]
    single = discover_op_points(FIX_SP / "miller" / "psf_stb")
    assert len(single) == 1


def test_session_binds_and_switches_op_points(tmp_path):
    """One session over the point set: opens on the first point,
    switches with structure preserved (matches, declaration) and every
    cache dropped -- the new point's analyses re-derive."""
    import warnings

    from circuitinsight.session import SessionController

    root = _op_tree(tmp_path, 2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c = SessionController.open(
            FIX_SP / "miller" / "tb_ota2s_stb.cin.json", root,
            cap_model="matrix")
    assert c.op_point_labels == ["tt 27", "nom ss -40"]
    assert c.op_path == root / "1" / "work_tb" / "psf"
    assert c.has_stb
    r1 = c.loop_gain("IPRB0")
    c.declare_ac_loop_gain(None)
    c.set_op_point(1)
    assert c.op_path == root / "2" / "work_tb" / "psf"
    r2 = c.loop_gain("IPRB0")
    assert r2 is not r1              # caches dropped, re-derived
    # identical copied data: the physics must agree across the rebind
    assert abs(r1.pm_deg - r2.pm_deg) < 1e-9
    # single-psf sessions have no point set and refuse to switch
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c1 = SessionController.open(FIX_SP / "miller" / "tb_ota2s.cin.json",
                                    FIX_SP / "miller" / "psf")
    assert c1.op_point_labels == []
    with pytest.raises(ValueError):
        c1.set_op_point(1)


def test_points_manifest_is_the_official_point_set(tmp_path):
    """A launcher-written manifest (from the simulator's OWN results
    database) names the points officially; the session loads it as the
    point set with no reconstruction. Entries without OP data are
    skipped, and a manifest with none is a clear error."""
    import json
    import warnings

    from circuitinsight.adapters.points import load_points_manifest
    from circuitinsight.session import SessionController

    root = _op_tree(tmp_path, 2)
    man = tmp_path / "ExplorerRun.0.points.json"
    man.write_text(json.dumps({
        "cin_points_version": "0.1", "history": "ExplorerRun.0",
        "points": [
            {"corner": "ff 125", "psf": str(root / "1" / "work_tb" / "psf")},
            {"corner": "tt", "psf": str(root / "2" / "work_tb" / "psf")},
            {"corner": "ghost", "psf": str(root / "3" / "work_tb" / "psf")},
        ]}))
    pts = load_points_manifest(man)
    assert [p.label for p in pts] == ["ff 125", "tt"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c = SessionController.open(
            FIX_SP / "miller" / "tb_ota2s_stb.cin.json", man,
            cap_model="matrix")
    assert c.op_point_labels == ["ff 125", "tt"]
    assert c.op_path == root / "1" / "work_tb" / "psf"
    c.set_op_point(1)
    assert c.op_path == root / "2" / "work_tb" / "psf"
    empty = tmp_path / "empty.points.json"
    empty.write_text(json.dumps({"points": [
        {"corner": "x", "psf": str(root / "9" / "work_tb" / "psf")}]}))
    with pytest.raises(ValueError, match="manifest"):
        SessionController.open(
            FIX_SP / "miller" / "tb_ota2s_stb.cin.json", empty)


def test_margin_ledger_set_spans_the_points(tmp_path):
    """Ledgers over the whole set without moving the bound point; the
    per-point results are content-addressed, so the bound point's own
    ledger is THE SAME object margin_ledger served, and identical
    copied points read identically. A single-psf session refuses."""
    import warnings

    from circuitinsight.session import SessionController

    root = _op_tree(tmp_path, 2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c = SessionController.open(
            FIX_SP / "miller" / "tb_ota2s_stb.cin.json", root,
            cap_model="matrix")
        bound = c.margin_ledger("IPRB0")
        out = c.margin_ledger_set("IPRB0")
    assert out["labels"] == ["tt 27", "nom ss -40"]
    assert out["ref_index"] == 0
    assert c._op_index == 0                    # the session stayed put
    assert out["ledgers"][0] is bound["ledger"]
    assert out["ledgers"][1].pm_deg == pytest.approx(
        out["ledgers"][0].pm_deg, abs=1e-9)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c1 = SessionController.open(FIX_SP / "miller" / "tb_ota2s.cin.json",
                                    FIX_SP / "miller" / "psf")
    with pytest.raises(ValueError):
        c1.margin_ledger_set("IPRB0")


def test_suggestion_robustness_spans_the_point_set(tmp_path):
    """Already-sized suggestions re-measured on EVERY operating point,
    values held fixed. The synthetic tree copies one psf, so every
    point must read identically and the worst case must equal the
    suggester's own nominal certificate; a single-psf session has no
    set and refuses."""
    import warnings

    from circuitinsight.analysis.compensate import Candidate
    from circuitinsight.session import SessionController

    root = _op_tree(tmp_path, 2)
    cand = Candidate("miller", "I0.net1", "vout", "bridge", 1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c = SessionController.open(
            FIX_SP / "miller" / "tb_ota2s_stb.cin.json", root,
            cap_model="matrix")
        sugg = c.suggest_compensation("IPRB0", goal="mfm",
                                      exclude=("I0.Cc",),
                                      candidates=[cand])
        rob = c.suggestion_robustness("IPRB0", sugg, exclude=("I0.Cc",))
    assert len(rob) == len(sugg) > 0
    r0 = rob[0]
    assert r0["labels"] == ["tt 27", "nom ss -40"]
    flats = [p["flat_db"] for p in r0["points"]]
    assert flats[0] == pytest.approx(flats[1], abs=1e-9)
    assert r0["worst_flat_db"] == pytest.approx(sugg[0].flat_db, abs=1e-6)
    assert r0["worst_pm_deg"] == pytest.approx(sugg[0].pm_deg, abs=1e-6)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c1 = SessionController.open(FIX_SP / "miller" / "tb_ota2s.cin.json",
                                    FIX_SP / "miller" / "psf")
    with pytest.raises(ValueError):
        c1.suggestion_robustness("IPRB0", sugg)
    with pytest.raises(ValueError):
        c1.suggest_robust_compensation("IPRB0")
    # identical copied points: worst-corner sizing changes nothing
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rob = c.suggest_robust_compensation("IPRB0", goal="mfm",
                                            exclude=("I0.Cc",),
                                            candidates=[cand])
    assert [(s.C, s.R, s.achieved) for s in rob] == \
        [(s.C, s.R, s.achieved) for s in sugg]
