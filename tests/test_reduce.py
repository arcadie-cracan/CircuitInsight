"""Netlist reduction: fold out simulator-pruned (0-valued) passives.

A device with no OP record is a 0-valued passive the simulator optimized away.
Reflect it electrically: R/L -> short (merge nets), C -> open (delete). A missing
ACTIVE device is a real error and must still raise.
"""
import warnings
from pathlib import Path

import pytest

from circuitinsight.adapters.cin import FlatCircuit, FlatDevice, flatten, load_cin
from circuitinsight.adapters.spectre import SpectreError, join_op, join_op_reduced
from circuitinsight.adapters.spectre.opdata import load_dcopinfo
from circuitinsight.reduce import drop_unmatched_passives

FIX = Path(__file__).resolve().parent / "fixtures" / "spectre"


def _dev(name, dtype, p, n):
    return FlatDevice(name=name, sim_name=name, device_type=dtype,
                      terminals={"p": p, "n": n})


def _circuit(devs):
    nets = frozenset(x for d in devs for x in d.terminals.values())
    return FlatCircuit(tuple(devs), nets, ("0",))


# ---------------------------------------------------------------- unit level

def test_missing_resistor_is_shorted():
    r = _dev("R1", "resistor", "a", "b")
    keeper = _dev("C1", "capacitor", "b", "0")
    flat, notes = drop_unmatched_passives(_circuit([r, keeper]), [r])
    names = {d.name for d in flat.devices}
    assert "R1" not in names and "C1" in names          # R removed, C kept
    # a and b are now one net (survivor 'a', lexicographically first); C1's 'b'
    # terminal moves onto it.
    c1 = next(d for d in flat.devices if d.name == "C1")
    assert c1.terminals["p"] == "a" and c1.terminals["n"] == "0"
    assert set(flat.nets) == {"a", "0"}                 # b is gone, merged into a
    assert "SHORT" in notes[0]


def test_missing_capacitor_is_opened():
    c = _dev("Cc", "capacitor", "vout", "vx")
    r = _dev("Rload", "resistor", "vout", "0")
    flat, notes = drop_unmatched_passives(_circuit([c, r]), [c])
    assert "Cc" not in {d.name for d in flat.devices}   # removed (open)
    assert "Rload" in {d.name for d in flat.devices}
    assert "OPEN" in notes[0]


def test_short_chain_collapses_and_ground_wins():
    # R1: a-b, R2: b-0  -> both shorted -> a,b,0 all become ground
    r1 = _dev("R1", "resistor", "a", "b")
    r2 = _dev("R2", "resistor", "b", "0")
    load = _dev("Cl", "capacitor", "a", "0")
    flat, notes = drop_unmatched_passives(_circuit([r1, r2, load]), [r1, r2])
    cl = next(d for d in flat.devices if d.name == "Cl")
    assert cl.terminals["p"] == cl.terminals["n"] == "0"   # collapsed to ground
    assert len(notes) == 2


# --------------------------------------------------- through the OP join

def test_join_op_reduced_reduces_where_join_op_raises():
    flat = flatten(load_cin(FIX / "miller_unc" / "tb_ota2s.cin.json"))
    recs = load_dcopinfo(FIX / "miller_unc" / "psf" / "dcOpInfo.info")
    cc = next(d for d in flat.devices if d.device_type == "capacitor"
              and d.name.endswith("Cc"))
    pruned = {k: v for k, v in recs.items() if cc.sim_name not in k}  # Spectre pruned Cc

    with pytest.raises(SpectreError, match="no OP record"):
        join_op(flat, pruned)                            # old behaviour: crash

    op, flat2, notes = join_op_reduced(flat, pruned)     # new: reduce
    assert cc.name not in {d.name for d in flat2.devices}
    assert notes and "OPEN" in notes[0]


def test_zero_valued_passive_in_op_is_folded():
    """Some simulators (SKY130's Spectre) KEEP a 0 F / 0 ohm passive as an OP
    record instead of pruning it. join_op_reduced must fold it the same way a
    missing one is folded: cap 0 F -> open, resistor 0 ohm -> short."""
    from circuitinsight.adapters.spectre.opdata import OpRecord
    mos = _dev("M1", "mosfet", "d", "s")               # keeps the join non-empty
    cc = _dev("Cc", "capacitor", "d", "x")
    rz = _dev("Rz", "resistor", "x", "y")
    flat = _circuit([mos, cc, rz])

    def rec(name, mt, params):
        return OpRecord(name=name, model_type=mt, params=params)
    recs = {"M1": rec("M1", "mosfet", {"gm": 1e-3, "gds": 1e-5, "region": 2}),
            "Cc": rec("Cc", "capacitor", {"c": 0.0}),      # kept, but zero
            "Rz": rec("Rz", "resistor", {"r": 0.0})}       # kept, but zero

    op, flat2, notes = join_op_reduced(flat, recs)
    names = {d.name for d in flat2.devices}
    assert "Cc" not in names                               # 0 F -> open (removed)
    assert "Rz" not in names                               # 0 ohm -> short (merged)
    assert any("0-valued in the OP" in n for n in notes)
    assert "M1" in names                                   # transistor untouched


def test_missing_active_device_still_errors():
    """A pruned passive is fine; a missing transistor is a real error."""
    flat = flatten(load_cin(FIX / "ota5t" / "tb_ota5t.cin.json"))
    recs = load_dcopinfo(FIX / "ota5t" / "psf" / "dcOpInfo.info")
    mos = next(d for d in flat.devices if d.device_type == "mosfet")
    pruned = {k: v for k, v in recs.items() if mos.sim_name not in k}
    with pytest.raises(SpectreError, match="no OP record"):
        join_op_reduced(flat, pruned)


def test_reduced_circuit_still_solves():
    """After folding out a pruned cap, the analyzer must build and solve."""
    from circuitinsight import Analyzer

    flat = flatten(load_cin(FIX / "miller_unc" / "tb_ota2s.cin.json"))
    recs = load_dcopinfo(FIX / "miller_unc" / "psf" / "dcOpInfo.info")
    cc = next(d for d in flat.devices if d.name.endswith("Cc"))
    pruned = {k: v for k, v in recs.items() if cc.sim_name not in k}
    op, flat2, _ = join_op_reduced(flat, pruned)
    an = Analyzer(flat2, op)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        H = an.tf("VIND", "vout", keep=[])               # numeric solve
    assert H.dc_gain() != 0


# ------------------------------------------- reduced-ORDER symbolic solve

def _miller_session():
    from circuitinsight import SessionController
    c = SessionController.open(FIX / "miller" / "tb_ota2s.cin.json",
                              FIX / "miller" / "psf")
    c.set_matches(*c.suggest_matches())
    return c


def test_reduced_tf_lowers_the_pole_count():
    """Simplify never lowers order; reduced_tf does, by dropping parasitic caps.
    Keeping only Cc+CL yields the 2nd-order Miller form (eq 2)."""
    import sympy as sp

    c = _miller_session()
    an = c._analyzer_ready()
    keep = ["gm_I0_MN1", "gm_I0_MP2", "I0_Cc", "CL",
            "gds_I0_MN1", "gds_I0_MP1", "gds_I0_MP2", "gds_I0_MN3"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        H_full = an.tf("VIND", "vout", keep=keep)
        H_red, red = an.reduced_tf("VIND", "vout", keep=keep, tol_db=0.001,
                                   fmin=1e3, fmax=1e7, max_elements=2)
    s = sp.Symbol("s")
    full_order = sp.Poly(H_full.num_den[1].as_expr(), s).degree()
    red_order = sp.Poly(H_red.num_den[1].as_expr(), s).degree()
    assert red.selected == ["I0_Cc", "CL"]
    assert red_order == 2 and full_order > 2         # order actually dropped
    # the two kept caps survive as symbols; parasitics are gone
    syms = {str(x) for x in H_red.expr.free_symbols}
    assert "I0_Cc" in syms and "CL" in syms
    assert not any(x.startswith(("cgs_", "cgd_", "cdb_")) for x in syms)


def test_reduce_solve_reports_honest_band_error():
    c = _miller_session()
    keep = ["gm_I0_MN1", "gm_I0_MP2", "I0_Cc", "CL",
            "gds_I0_MN1", "gds_I0_MP1", "gds_I0_MP2", "gds_I0_MN3"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = c.reduce_solve("VIND", "vout", keep, tol_db=0.35, mag_db=1.0,
                           phase_deg=5.0, fmax=1e7)
    assert r.simplified and r.mag_err_db > 0          # a reduction is not free
    assert r.warnings and "reduced to" in r.warnings[0] \
        and "I0_Cc" in r.warnings[0]                  # Cc is the dominant reactance
    assert r.dc_gain_db == pytest.approx(85.5, abs=0.3)
