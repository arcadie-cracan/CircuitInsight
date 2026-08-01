"""S-E tearing: cut discovery + exact split composition.

Everything here is exact-or-refused: split solves must reproduce the
monolithic solver bit-for-bit (loading, bidirectional coupling and all),
and cases the 1-node theory does not cover raise TearingError instead of
approximating."""
import warnings
from pathlib import Path

import pytest
import sympy as sp

from circuitinsight import Analyzer
from circuitinsight.analysis import tearing
from circuitinsight.analysis.loopgain import loop_gain
from circuitinsight.keep import ALL

FIX = Path(__file__).resolve().parent / "fixtures" / "spectre"


def _fixture(name, cin, psf):
    from circuitinsight.adapters.spectre import SpectreRun
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run = SpectreRun(FIX / name / cin, FIX / name / psf)
        return run.analyzer(cap_model="matrix")


# ------------------------------------------------------------ cut discovery
def test_nmc3_opened_loop_has_the_one_node_cut():
    an = _fixture("nmc3", "tb_nmc3.cin.json", "psf")
    cuts = tearing.find_cuts(an.primitives, an.flat.ground, "inn", "out",
                             open_insts={"IPRB0"})
    ones = [c for c in cuts if len(c) == 1]
    assert ones and ones[0][0] in ("n1", "n1i")


def test_miller_opened_loop_needs_two_nodes():
    an = _fixture("miller", "tb_ota2s_stb.cin.json", "psf_stb")
    cuts = tearing.find_cuts(an.primitives, an.flat.ground,
                             "vin_n", "vout", open_insts={"IPRB0"})
    assert not any(len(c) == 1 for c in cuts)
    assert any(set(c) == {"I0.net1", "vbn"} for c in cuts)


def test_find_cuts_refuses_shorted_terminals():
    an = _fixture("nmc3", "tb_nmc3.cin.json", "psf")
    with pytest.raises(tearing.TearingError):
        tearing.find_cuts(an.primitives, an.flat.ground, "inn", "out")


# ------------------------------------------------------------- split_tf
def _ladder_cin(extra=()):
    inst = [
        {"name": "V1", "device_type": "vsource",
         "terminals": {"p": "in", "n": "0"}},
        {"name": "R1", "device_type": "resistor",
         "terminals": {"p": "in", "n": "m"}, "params": {"r": "1k"}},
        {"name": "C1", "device_type": "capacitor",
         "terminals": {"p": "m", "n": "0"}, "params": {"c": "1n"}},
        {"name": "R2", "device_type": "resistor",
         "terminals": {"p": "m", "n": "out"}, "params": {"r": "2k"}},
        {"name": "C2", "device_type": "capacitor",
         "terminals": {"p": "out", "n": "0"}, "params": {"c": "2n"}},
    ] + list(extra)
    return {"cin_version": "0.1", "top": "main", "ground": ["0"],
            "definitions": {"main": {"ports": [], "instances": inst}}}


def _assert_split_equals_monolithic(an, cut, keep):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        h_mono = an.tf("V1", "out", keep=keep, method="interp")
        h_split = tearing.split_tf(an.primitives, an.flat.ground,
                                   "V1", "out", cut, keep=keep)
    diff = sp.simplify(sp.together(h_mono.expr - h_split.expr))
    assert diff == 0, f"split and monolithic disagree for keep={keep}"
    return h_split


def test_split_tf_loaded_ladder_exact():
    """R2/C2 genuinely load node m: a naive H1*H2 product would be wrong;
    the composed split must match the monolithic solve exactly."""
    an = Analyzer.from_cin(_ladder_cin())
    h = _assert_split_equals_monolithic(an, "m", ["R1", "R2", "C2"])
    free = {str(s) for s in h.expr.free_symbols}
    assert {"R1", "R2", "C2"} <= free            # kept on BOTH sides


def test_split_tf_bidirectional_half_exact():
    """A feedback cap from out back onto the cut makes side B
    bidirectional; the Thevenin composition must still be exact."""
    cf = {"name": "CF", "device_type": "capacitor",
          "terminals": {"p": "out", "n": "m"}, "params": {"c": "500p"}}
    an = Analyzer.from_cin(_ladder_cin([cf]))
    _assert_split_equals_monolithic(an, "m", ["R1", "CF"])


def test_split_tf_active_stages_exact():
    """Two vccs stages with a bridging feedback cap: active, inverting,
    non-unilateral -- the full 1-node-cut stress."""
    inst = [
        {"name": "V1", "device_type": "vsource",
         "terminals": {"p": "in", "n": "0"}},
        {"name": "G1", "device_type": "vccs",
         "terminals": {"p": "0", "n": "m", "cp": "in", "cn": "0"},
         "params": {"gm": "1m"}},
        {"name": "RM", "device_type": "resistor",
         "terminals": {"p": "m", "n": "0"}, "params": {"r": "10k"}},
        {"name": "CM", "device_type": "capacitor",
         "terminals": {"p": "m", "n": "0"}, "params": {"c": "1p"}},
        {"name": "G2", "device_type": "vccs",
         "terminals": {"p": "0", "n": "out", "cp": "m", "cn": "0"},
         "params": {"gm": "2m"}},
        {"name": "RL", "device_type": "resistor",
         "terminals": {"p": "out", "n": "0"}, "params": {"r": "5k"}},
        {"name": "CF", "device_type": "capacitor",
         "terminals": {"p": "out", "n": "m"}, "params": {"c": "3p"}},
    ]
    cin = {"cin_version": "0.1", "top": "main", "ground": ["0"],
           "definitions": {"main": {"ports": [], "instances": inst}}}
    an = Analyzer.from_cin(cin)
    _assert_split_equals_monolithic(an, "m", ["G1", "G2", "CF", "RL"])


def test_split_tf_rejects_output_on_input_side():
    an = Analyzer.from_cin(_ladder_cin())
    with pytest.raises(tearing.TearingError):
        tearing.split_tf(an.primitives, an.flat.ground, "V1", "in", "m")


# ------------------------------------------------------------- 2-node tear
def _two_path_cin():
    """Two coupling paths A->B (signal node v1 AND a finite-impedance bias
    node v2), plus an element BETWEEN the cut nodes and a feedback cap:
    no one-node cut exists, and the 2x2 tear must handle every term."""
    inst = [
        {"name": "V1", "device_type": "vsource",
         "terminals": {"p": "in", "n": "0"}},
        {"name": "G1", "device_type": "vccs",
         "terminals": {"p": "0", "n": "v1", "cp": "in", "cn": "0"},
         "params": {"gm": "1m"}},
        {"name": "GA", "device_type": "vccs",
         "terminals": {"p": "0", "n": "v2", "cp": "in", "cn": "0"},
         "params": {"gm": "0.1m"}},
        {"name": "R1", "device_type": "resistor",
         "terminals": {"p": "v1", "n": "0"}, "params": {"r": "10k"}},
        {"name": "R2", "device_type": "resistor",
         "terminals": {"p": "v2", "n": "0"}, "params": {"r": "50k"}},
        {"name": "R12", "device_type": "resistor",
         "terminals": {"p": "v1", "n": "v2"}, "params": {"r": "100k"}},
        {"name": "G2", "device_type": "vccs",
         "terminals": {"p": "0", "n": "out", "cp": "v1", "cn": "0"},
         "params": {"gm": "2m"}},
        {"name": "G2B", "device_type": "vccs",
         "terminals": {"p": "0", "n": "out", "cp": "v2", "cn": "0"},
         "params": {"gm": "0.5m"}},
        {"name": "RL", "device_type": "resistor",
         "terminals": {"p": "out", "n": "0"}, "params": {"r": "5k"}},
        {"name": "CL", "device_type": "capacitor",
         "terminals": {"p": "out", "n": "0"}, "params": {"c": "2p"}},
        {"name": "CF", "device_type": "capacitor",
         "terminals": {"p": "out", "n": "v1"}, "params": {"c": "1p"}},
        {"name": "CB", "device_type": "capacitor",
         "terminals": {"p": "out", "n": "v2"}, "params": {"c": "0.2p"}},
    ]
    return {"cin_version": "0.1", "top": "main", "ground": ["0"],
            "definitions": {"main": {"ports": [], "instances": inst}}}


def test_two_node_cut_is_minimal_on_the_synthetic():
    an = Analyzer.from_cin(_two_path_cin())
    cuts = tearing.find_cuts(an.primitives, an.flat.ground, "in", "out",
                             keep_src={"V1"})
    assert not any(len(c) == 1 for c in cuts)
    assert any(set(c) == {"v1", "v2"} for c in cuts)


def test_split_tf2_synthetic_exact():
    an = Analyzer.from_cin(_two_path_cin())
    keep = ["G1", "G2", "CF", "R12", "CB"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        h_mono = an.tf("V1", "out", keep=keep, method="interp")
        h_split = tearing.split_tf(an.primitives, an.flat.ground,
                                   "V1", "out", ("v1", "v2"), keep=keep)
    diff = sp.simplify(sp.together(h_mono.expr - h_split.expr))
    assert diff == 0
    free = {str(s) for s in h_split.expr.free_symbols}
    assert {"G1", "G2", "CF", "R12", "CB"} <= free


def test_split_tf2_miller_transistor_level_exact():
    """THE transistor-level gate: the opened miller loop torn at its
    measured minimal cut {I0.net1, vbn} (inter-stage node + shared bias
    rail) must reproduce the monolithic solve of the same opened chain."""
    from circuitinsight.engine.mna import build_mna, solve_tf
    from circuitinsight.engine.primitives import Primitive

    an = _fixture("miller", "tb_ota2s_stb.cin.json", "psf_stb")
    gnd = an.flat.ground
    opened = [p for p in an.primitives if p.inst != "IPRB0"]
    drv = Primitive(inst="__drv", param="", kind="vsrc",
                    nodes=("vin_n", gnd[0]))
    prims = opened + [drv]
    keep = ["gm_I0_MN0", "gm_I0_MN3"]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sys_mono = build_mna(prims, gnd, "__drv")
        h_mono = solve_tf(sys_mono, "vout", keep=keep)
        h_split = tearing.split_tf(prims, gnd, "__drv", "vout",
                                   ("I0.net1", "vbn"), keep=keep)
    diff = sp.simplify(sp.together(h_mono.expr - h_split.expr))
    assert diff == 0
    free = {str(s) for s in h_split.expr.free_symbols}
    assert {"gm_I0_MN0", "gm_I0_MN3"} <= free


def _three_path_cin():
    """THREE coupling paths in->out (v1, v2, v3) with cross elements among
    them and feedback caps: the minimal cut is 3 nodes, so only the
    general m-node interface solve can compose it."""
    inst = [
        {"name": "V1", "device_type": "vsource",
         "terminals": {"p": "in", "n": "0"}},
    ]
    for i, gm in enumerate(("1m", "0.4m", "0.2m"), start=1):
        inst += [
            {"name": f"G{i}", "device_type": "vccs",
             "terminals": {"p": "0", "n": f"v{i}", "cp": "in", "cn": "0"},
             "params": {"gm": gm}},
            {"name": f"R{i}", "device_type": "resistor",
             "terminals": {"p": f"v{i}", "n": "0"},
             "params": {"r": f"{10 * i}k"}},
            {"name": f"GO{i}", "device_type": "vccs",
             "terminals": {"p": "0", "n": "out", "cp": f"v{i}", "cn": "0"},
             "params": {"gm": gm}},
            {"name": f"CF{i}", "device_type": "capacitor",
             "terminals": {"p": "out", "n": f"v{i}"},
             "params": {"c": f"{i}p"}},
        ]
    inst += [
        {"name": "R12", "device_type": "resistor",
         "terminals": {"p": "v1", "n": "v2"}, "params": {"r": "70k"}},
        {"name": "R23", "device_type": "resistor",
         "terminals": {"p": "v2", "n": "v3"}, "params": {"r": "90k"}},
        {"name": "RL", "device_type": "resistor",
         "terminals": {"p": "out", "n": "0"}, "params": {"r": "5k"}},
        {"name": "CL", "device_type": "capacitor",
         "terminals": {"p": "out", "n": "0"}, "params": {"c": "2p"}},
    ]
    return {"cin_version": "0.1", "top": "main", "ground": ["0"],
            "definitions": {"main": {"ports": [], "instances": inst}}}


def test_three_node_cut_is_minimal_and_composes_exactly():
    """m = 3: no 1- or 2-node cut exists, and the general interface solve
    (polynomial-cleared Cramer) must reproduce the monolithic result."""
    an = Analyzer.from_cin(_three_path_cin())
    cuts = tearing.find_cuts(an.primitives, an.flat.ground, "in", "out",
                             keep_src={"V1"}, max_size=3)
    assert not any(len(c) < 3 for c in cuts)
    assert any(set(c) == {"v1", "v2", "v3"} for c in cuts)

    keep = ["G1", "GO3", "CF1", "R23"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        h_mono = an.tf("V1", "out", keep=keep, method="interp")
        h_split = tearing.split_tf(an.primitives, an.flat.ground, "V1",
                                   "out", ("v1", "v2", "v3"), keep=keep)
    assert sp.simplify(sp.together(h_mono.expr - h_split.expr)) == 0


def test_rank_cuts_flags_the_lopsided_cut():
    """A cut is valid ≠ a cut pays. The 741's only 2-node cut is wildly
    unbalanced (a handful of primitives against nearly the whole
    circuit), and ranking must say so -- the measured 87 s vs 24 s
    regression that motivated the rule."""
    an = _fixture("ua741", "ua741.cin.json", "psf")
    ranked = tearing.rank_cuts(an.primitives, an.flat.ground, "1", "22",
                               keep_src={"Vin"})
    assert ranked, "the 741 has a 2-node cut"
    top = ranked[0]
    assert top["cut"] == ("12", "4")
    assert min(top["n_a"], top["n_b"]) * 4 < max(top["n_a"], top["n_b"])
    assert top["balance"] < 0.25 and not top["pays"]


def test_rank_cuts_accepts_a_balanced_cut():
    an = Analyzer.from_cin(_two_path_cin())
    ranked = tearing.rank_cuts(an.primitives, an.flat.ground, "in", "out",
                               keep_src={"V1"}, max_size=2)
    top = ranked[0]
    assert set(top["cut"]) == {"v1", "v2"}
    assert top["balance"] >= 0.25


# --------------------------------------------------------------- AC ground
def test_ac_ground_error_separates_bias_rails_from_signal_nodes():
    """The measurement that makes the AC-ground assumption honest: on the
    741, tying a true bias rail to ground costs ~0.02 dB, while tying a
    SIGNAL node to ground is catastrophic. The tool must be able to tell
    them apart by measurement, not by naming convention."""
    import numpy as np

    an = _fixture("ua741", "ua741.cin.json", "psf")
    freqs = np.geomspace(10.0, 1e7, 40)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bias = tearing.ac_ground_error(an.primitives, an.flat.ground,
                                       "Vin", "22", ["6"], freqs,
                                       alias=an._alias)
        signal = tearing.ac_ground_error(an.primitives, an.flat.ground,
                                         "Vin", "22", ["8"], freqs,
                                         alias=an._alias)
    assert bias["worst_db"] < 0.1
    assert signal["worst_db"] > 20.0


def test_ac_ground_rebalances_the_741_and_stays_exact_for_the_model():
    """Shi's AC-ground designation as a tearing lever: declaring one
    measured bias rail an AC ground turns the 741's hopeless 15/155 cut
    into a balanced one, and the split is then EXACT for the rewritten
    circuit."""
    an = _fixture("ua741", "ua741.cin.json", "psf")
    prims, gnd = an.primitives, an.flat.ground
    pg = tearing.ac_ground(prims, gnd, ["6"])
    assert len(pg) == len(prims)

    ranked = tearing.rank_cuts(pg, gnd, "1", "22", keep_src={"Vin"})
    top = ranked[0]
    assert set(top["cut"]) == {"2", "8"}
    assert top["balance"] > 0.9 and top["pays"]

    from circuitinsight.engine.mna import build_mna, solve_tf
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        h_mono = solve_tf(build_mna(pg, gnd, "Vin", an._alias), "22",
                          keep=[])
        h_split = tearing.split_tf(pg, gnd, "Vin", "22", ("2", "8"),
                                   keep=[])
    assert _eq_ratio(h_mono.expr, h_split.expr)


def test_rank_cuts_predicts_the_measured_741_outcomes():
    """The advisor's verdict was validated against wall-clock: on the
    balanced 741 cut, four kept symbols (2 per side) were measured at
    141 s split vs 33 s monolithic -- a loss, because the exponential
    keep-set win (16 -> 8) cannot pay the constant (1+m)*m = 6 solve
    penalty. Twelve kept symbols (4096 -> 128) flips it, matching the
    miller k=12 win. The heuristic must keep saying exactly that."""
    an = _fixture("ua741", "ua741.cin.json", "psf")
    pg = tearing.ac_ground(an.primitives, an.flat.ground, ["6"])
    small = tearing.rank_cuts(pg, an.flat.ground, "1", "22",
                              keep_src={"Vin"},
                              keep=["gm_Q1", "gm_Q10", "gm_Q13A",
                                    "gm_Q13B"])[0]
    assert not small["pays"]
    big = tearing.rank_cuts(
        pg, an.flat.ground, "1", "22", keep_src={"Vin"},
        keep=["gm_Q1", "gm_Q10", "gm_Q11", "gm_Q12", "gm_Q2", "gm_Q3",
              "gm_Q13A", "gm_Q13B", "gm_Q14", "gm_Q15", "gm_Q16",
              "gm_Q17"])[0]
    assert big["pays"]
    assert big["grid_mono"] > 6 * big["grid_split"]


# ---------------------------------------------------------- split advisory
def test_advise_split_names_the_ac_ground_that_would_fix_the_741():
    """The advisory the measurements earned: the 741's own cut does not
    pay, and the tool must say so AND name the bias node that would fix
    the balance, with the dB it costs -- while never proposing the
    signal node that looks just as good topologically."""
    an = _fixture("ua741", "ua741.cin.json", "psf")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        adv = an.advise_split("Vin", "22", keep=["gm_Q1", "gm_Q17"])
    assert adv.cuts and not adv.cuts[0]["pays"]
    assert adv.cuts[0]["balance"] < 0.25
    assert adv.ac_grounds, "a fixing AC ground exists on the 741"
    nodes = {g["node"] for g in adv.ac_grounds}
    assert "8" not in nodes, "the 252 dB signal node must be filtered out"
    top = adv.ac_grounds[0]
    assert top["balance"] > 0.75 and top["worst_db"] < 0.5
    v = adv.verdict()
    assert "do not tear" in v and "AC ground" in v


def test_advise_split_recommends_a_balanced_cut_with_a_large_keep_set():
    an = Analyzer.from_cin(_two_path_cin())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        adv = an.advise_split("V1", "out",
                              keep=["G1", "GA", "R1", "R2", "R12",
                                    "G2", "G2B", "CF", "CB", "RL"])
    assert adv.cuts[0]["pays"]
    assert "tear at" in adv.verdict()


def test_session_split_advice_is_cached():
    from circuitinsight.session import SessionController

    s = SessionController.open(FIX / "nmc3" / "tb_nmc3.cin.json",
                               FIX / "nmc3" / "psf")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a1 = s.advise_split("VIN", "vout")
        a2 = s.advise_split("VIN", "vout")
    assert a1 is a2
    assert isinstance(a1.verdict(), str)


# ------------------------------------------------------------ latency cache
def test_latency_cache_reuses_the_untouched_side(monkeypatch):
    """Iordache's latency principle: changing a side-B element must
    re-solve ONLY side B; the cached side-A batch is reused verbatim and
    the composed result stays exact."""
    import dataclasses

    from circuitinsight.engine import mna as mna_mod

    calls = []
    orig = mna_mod.solve_tf_batch

    def counting(tasks, **kw):
        calls.append(len(tasks))
        return orig(tasks, **kw)

    monkeypatch.setattr(mna_mod, "solve_tf_batch", counting)
    monkeypatch.setattr(tearing, "solve_tf_batch", counting)

    an = Analyzer.from_cin(_ladder_cin())
    cache = {}
    keep = ["R1", "C2"]
    h1 = tearing.split_tf(an.primitives, an.flat.ground, "V1", "out", "m",
                          keep=keep, cache=cache)
    n_first = len(calls)
    assert n_first == 2                          # one batch per side

    # identical repeat: fully served from the cache
    tearing.split_tf(an.primitives, an.flat.ground, "V1", "out", "m",
                     keep=keep, cache=cache)
    assert len(calls) == n_first

    # change a side-B element value (an UNKEPT one, so the expression
    # itself changes): only side B recomputes
    prims2 = [dataclasses.replace(p, value=3000.0) if p.inst == "R2" else p
              for p in an.primitives]
    h2 = tearing.split_tf(prims2, an.flat.ground, "V1", "out", "m",
                          keep=keep, cache=cache)
    assert len(calls) == n_first + 1             # side B only

    # and the cached composition is still exact vs a fresh monolithic
    an2 = Analyzer.from_cin(_ladder_cin())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sys2 = __import__("circuitinsight.engine.mna",
                          fromlist=["build_mna"]).build_mna(
            prims2, an.flat.ground, "V1")
        h_ref = orig([(sys2, "out")], keep=keep)[0]
    assert sp.simplify(sp.together(h2.expr - h_ref.expr)) == 0
    assert sp.simplify(sp.together(h1.expr - h_ref.expr)) != 0


# --------------------------------------------------------- split_loop_gain
def test_split_loop_gain_nmc3_matches_monolithic_fully_symbolic():
    """The S-E(c) gate, now through product code: the loop gain composed
    from the torn chain equals the monolithic Tian T with EVERY symbol
    kept symbolic."""
    an = _fixture("nmc3", "tb_nmc3.cin.json", "psf")
    T_ref = loop_gain(an.system("IPRB0"), "IPRB0", keep=ALL)
    T_split = tearing.split_loop_gain(an.primitives, an.flat.ground,
                                      "IPRB0", "n1", keep=ALL)
    diff = sp.simplify(sp.together(T_ref.expr - T_split.expr))
    assert diff == 0


def test_split_loop_gain_numeric_keep():
    an = _fixture("nmc3", "tb_nmc3.cin.json", "psf")
    T_ref = loop_gain(an.system("IPRB0"), "IPRB0", keep=["CM2"])
    T_split = tearing.split_loop_gain(an.primitives, an.flat.ground,
                                      "IPRB0", "n1", keep=["CM2"])
    diff = sp.simplify(sp.together(T_ref.expr - T_split.expr))
    assert diff == 0
    assert {str(s) for s in T_split.expr.free_symbols} == {"CM2", "s"}


def _eq_ratio(a, b):
    na, da = sp.fraction(sp.together(a))
    nb, db = sp.fraction(sp.together(b))
    return sp.expand(na * db - nb * da) == 0


def test_split_loop_gain_tian_matches_on_nmc3():
    """The double-injection composition must agree with loop_gain even
    where the cheap chain route also applies."""
    an = _fixture("nmc3", "tb_nmc3.cin.json", "psf")
    T_ref = loop_gain(an.system("IPRB0"), "IPRB0", keep=ALL)
    T_tian = tearing._split_loop_gain_tian(an.primitives, an.flat.ground,
                                           "IPRB0", "n1", keep=ALL)
    assert _eq_ratio(T_ref.expr, T_tian.expr)


def test_split_loop_gain_miller_transistor_loop_exact():
    """THE transistor-loop gate: the miller stb loop gain composed through
    the 2-node tear {I0.net1, vbn} with the Tian double-injection
    reconnection (the probe's sense node draws current, so the chain
    route does not apply) must equal the monolithic Tian loop_gain."""
    an = _fixture("miller", "tb_ota2s_stb.cin.json", "psf_stb")
    T_ref = loop_gain(an.system("IPRB0"), "IPRB0", keep=[])
    T_split = tearing.split_loop_gain(an.primitives, an.flat.ground,
                                      "IPRB0", ("I0.net1", "vbn"),
                                      keep=[])
    assert _eq_ratio(T_ref.expr, T_split.expr)


def test_split_loop_gain_tian_rejects_probe_terminal_in_cut():
    an = _fixture("miller", "tb_ota2s_stb.cin.json", "psf_stb")
    with pytest.raises(tearing.TearingError):
        tearing._split_loop_gain_tian(an.primitives, an.flat.ground,
                                      "IPRB0", ("vout", "vbn"))


# ---------------------------------------------- suggester on the cache
def test_suggester_rerun_pays_zero_symbolic_dets(monkeypatch):
    """The wired candidate sweep: a re-run of suggest_multi_compensation
    with torn=(prims, ground, cut) and a shared cache must serve every
    symbolic quantity (T0 loop gain, pole scale, candidate screen) from
    the cache -- zero symbolic determinants -- and reproduce the fresh
    run's suggestion."""
    from circuitinsight.analysis.compensate import suggest_multi_compensation
    from circuitinsight.engine import mna as mna_mod
    from circuitinsight.engine.mna import build_mna

    an = _fixture("nmc3", "tb_nmc3.cin.json", "psf")
    prims = [p for p in an.primitives if p.inst not in ("CM1", "CM2")]
    gnd = an.flat.ground
    system = build_mna(prims, gnd, "IPRB0")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = suggest_multi_compensation(system, "IPRB0", goal="pm",
                                         pm_target=55.0, k_max=1)

        cache = {}
        s1 = suggest_multi_compensation(system, "IPRB0", goal="pm",
                                        pm_target=55.0, k_max=1,
                                        torn=(prims, gnd, "n1"),
                                        cache=cache)

        dets = []
        orig = mna_mod._det
        monkeypatch.setattr(mna_mod, "_det",
                            lambda *a, **kw: dets.append(1) or orig(*a, **kw))
        s2 = suggest_multi_compensation(system, "IPRB0", goal="pm",
                                        pm_target=55.0, k_max=1,
                                        torn=(prims, gnd, "n1"),
                                        cache=cache)
    assert not dets, f"re-run performed {len(dets)} symbolic dets"
    for s in (s1, s2):
        assert s.achieved == ref.achieved
        assert len(s.branches) == len(ref.branches)
        if s.branches:
            assert s.branches[0].physical() == ref.branches[0].physical()
            assert abs(s.branches[0].C - ref.branches[0].C) < 1e-15


def test_session_owns_a_tearing_cache_for_the_suggester(monkeypatch):
    """The session hands its content-fingerprinted cache to the
    compensation suggester, so a SECOND search on the same circuit pays
    no symbolic determinants for the prelude (T0, pole scale, candidate
    screen) -- the amortization the GUI inherits without knowing."""
    from circuitinsight.engine import mna as mna_mod
    from circuitinsight.session import SessionController

    s = SessionController.open(FIX / "nmc3" / "tb_nmc3.cin.json",
                               FIX / "nmc3" / "psf")
    assert s._tear_cache == {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s.suggest_multi_compensation("IPRB0", goal="pm", pm_target=55.0,
                                     k_max=1, exclude=("CM1", "CM2"))
        assert s._tear_cache, "the prelude must be cached on the session"

        dets = []
        orig = mna_mod._det
        monkeypatch.setattr(mna_mod, "_det",
                            lambda *a, **k: dets.append(1) or orig(*a, **k))
        s.suggest_multi_compensation("IPRB0", goal="pm", pm_target=70.0,
                                     k_max=1, exclude=("CM1", "CM2"))
    assert not dets, f"re-search performed {len(dets)} symbolic dets"


def test_tearing_pays_on_the_741_with_spread_keeps_post_sf():
    """The S-E closing measurement (2026-07-30). The 2026-07-28 verdict —
    tearing loses on the 741 — predates the all-numeric s-sweep; with it,
    single-level tearing through the AC-ground-enabled 1-node cut PAYS on
    spread keeps (measured 18.4 s -> 5.7 s at 6 keeps, grid 64 -> 8+8),
    and stays exact. NESTING does not pay at interactive keep counts: the
    clamp adds a grid axis per cut, so a 3-keep side factors 8 -> 9 — and
    at the keep counts where nesting would win, the sparse backend is
    already the winner. The advisor knows all of this; this test pins its
    verdicts and the exactness, not wall-clock."""
    import sympy as sp

    from circuitinsight.analysis.acground import scan_ac_grounds
    from circuitinsight.engine.mna import build_mna, solve_tf

    an = _fixture("ua741", "ua741.cin.json", "psf")
    prims, gnd = an.primitives, an.flat.ground
    rep = scan_ac_grounds(prims, gnd, "Vin", "22", alias=an._alias,
                          budget_db=0.1)
    g = tearing.ac_ground(prims, gnd, rep.recommended)
    keep = ["gm_Q1", "gm_Q3", "cpi_Q1", "gm_Q16", "gm_Q17", "cpi_Q16"]

    # advise_split, not rank_cuts directly: it resolves the SOURCE
    # instance to its net before seeding the graph -- handing rank_cuts
    # an instance name lands the seed in the wrong component and the
    # ranking silently degrades (found the hard way writing this test)
    adv = tearing.advise_split(g, gnd, "Vin", "22", keep=keep,
                               alias=an._alias)
    best = adv.cuts[0]
    assert best["cut"] == ("8",) and best["pays"]
    assert best["grid_mono"] == 64 and best["grid_split"] == 16

    h_mono = solve_tf(build_mna(g, gnd, "Vin", an._alias), "22", keep)
    h_split = tearing.split_tf(g, gnd, "Vin", "22", best["cut"], keep=keep)
    n1, d1 = h_mono.num_den
    n2, d2 = h_split.num_den
    assert sp.expand(n1.as_expr() * d2.as_expr()
                     - d1.as_expr() * n2.as_expr()) == 0

    # nesting: the big side has cuts, none pays at this keep count
    _, B = tearing.partition(g, gnd, best["cut"], "1")
    nested = tearing.rank_cuts(B, gnd, best["cut"][0], "22",
                               keep=["gm_Q16", "gm_Q17", "cpi_Q16"])
    assert nested and not any(r["pays"] for r in nested)


def test_find_cuts_rejects_instance_names_loudly():
    """Handing rank_cuts a SOURCE INSTANCE name used to seed the graph in
    the wrong component and silently degrade every verdict (it cost an
    hour writing the post-S-F closing test). Unknown seeds now raise with
    the advise_split hint instead."""
    an = _fixture("ua741", "ua741.cin.json", "psf")
    with pytest.raises(tearing.TearingError, match="NET names"):
        tearing.find_cuts(an.primitives, an.flat.ground, "Vin", "22")
    with pytest.raises(tearing.TearingError, match="advise_split"):
        tearing.rank_cuts(an.primitives, an.flat.ground, "Q16", "22")
