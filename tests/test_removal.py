"""Element-removal advisory: the Lei & Wu setting-zero idea as a measured
scan. Two things to establish: the rank-one pricing agrees with an exact
re-solve of the pruned circuit, and the scan separates elements that shape
the response from those that never mattered."""
import warnings
from pathlib import Path

import numpy as np
import pytest

from circuitinsight import Analyzer
from circuitinsight.analysis import tearing
from circuitinsight.analysis.removal import scan_removals

FIX = Path(__file__).resolve().parent / "fixtures" / "spectre"


def _cin():
    """An RC divider where C2 shapes the response and CX never can:
    CX sits in parallel with C2 but is 10000x smaller."""
    inst = [
        {"name": "V1", "device_type": "vsource",
         "terminals": {"p": "in", "n": "0"}},
        {"name": "R1", "device_type": "resistor",
         "terminals": {"p": "in", "n": "out"}, "params": {"r": "1k"}},
        {"name": "C2", "device_type": "capacitor",
         "terminals": {"p": "out", "n": "0"}, "params": {"c": "1n"}},
        {"name": "CX", "device_type": "capacitor",
         "terminals": {"p": "out", "n": "0"}, "params": {"c": "0.1f"}},
        {"name": "RB", "device_type": "resistor",
         "terminals": {"p": "out", "n": "0"}, "params": {"r": "1e9"}},
    ]
    return {"cin_version": "0.1", "top": "m", "ground": ["0"],
            "definitions": {"m": {"ports": [], "instances": inst}}}


def test_scan_separates_load_bearing_from_negligible():
    an = Analyzer.from_cin(_cin())
    rep = scan_removals(an.primitives, an.flat.ground, "V1", "out",
                        budget_db=0.1)
    by = {c.inst: c for c in rep.candidates}
    assert by["CX"].within_budget and by["RB"].within_budget
    assert not by["C2"].within_budget            # the pole lives on C2
    assert not by["R1"].within_budget
    assert set(rep.recommended) == {"CX", "RB"}
    assert rep.joint_db <= 0.1


def test_rank_one_pricing_matches_the_exact_resolve():
    """The Sherman-Morrison price per element must agree with actually
    deleting the element and re-solving -- signs and denominators have no
    place to hide."""
    an = Analyzer.from_cin(_cin())
    prims, gnd = an.primitives, an.flat.ground
    f = np.geomspace(1e2, 1e9, 25)
    rep = scan_removals(prims, gnd, "V1", "out", freqs=f, budget_db=0.1)
    a = tearing._numeric_response(prims, gnd, "V1", "out", f)
    for c in rep.candidates:
        pruned = [p for p in prims if p.inst != c.inst]
        b = tearing._numeric_response(pruned, gnd, "V1", "out", f)
        with np.errstate(divide="ignore", invalid="ignore"):
            db = np.abs(20 * np.log10(np.abs(b / a)))
        finite = db[np.isfinite(db)]
        if finite.size == 0:
            # removal disconnects the output entirely (deleting R1 leaves
            # nothing driving `out`): the scan must price that as huge
            assert c.worst_db > 60, c.inst
            continue
        exact = float(finite.max())
        assert c.worst_db == pytest.approx(exact, rel=1e-6, abs=1e-9), c.inst


def test_scan_protects_device_model_elements():
    """gds/cpi are passive by stamp but they ARE the transistor: the scan
    must only ever propose explicit netlist elements."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from circuitinsight.adapters.spectre import SpectreRun
        run = SpectreRun(FIX / "fc" / "tb_fc.cin.json", FIX / "fc" / "psf")
        an = run.analyzer(cap_model="matrix")
    rep = scan_removals(an.primitives, an.flat.ground, "VIND", "vout",
                        budget_db=0.5, alias=an._alias)
    for c in rep.candidates:
        assert c.kind in ("c", "r", "g", "l")
        assert "." not in c.inst or not c.inst.startswith("I0.")  \
            or c.inst.count(".") == 1        # explicit elements only
    named = {c.inst for c in rep.candidates}
    assert not any(n.endswith(("MN0", "MP0")) for n in named)


def test_session_exposes_both_advisories():
    from circuitinsight.session import SessionController

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = SessionController.open(FIX / "fc" / "tb_fc.cin.json",
                                   FIX / "fc" / "psf", cap_model="matrix")
        rep = s.scan_removals("VIND", "vout", budget_db=0.5)
        assert rep.candidates and "removal scan" in rep.describe()
        assert rep is s.scan_removals("VIND", "vout", budget_db=0.5)  # cached
        atts = s.pole_attribution("VIND", n_poles=2)
    assert atts and atts[0].owners[0].symbol == "CL"
    assert atts is s.pole_attribution("VIND", n_poles=2)              # cached
