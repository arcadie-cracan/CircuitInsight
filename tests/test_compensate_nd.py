"""N-D: the session surface + the Spectre confirmation loop.

The full circle: SessionController.suggest_compensation strips the
existing Miller cap (exclude -- OP-invariant), re-derives the Sec.-VI
series-RC network from one DC solve, and the psf_sugg fixture -- a real
Spectre stb run of exactly that network (3.177302 pF + 3747.635 ohm
through the schematic's Cc/Rz chain) -- confirms the predicted phase
margin to a hundredth of a degree.
"""
import warnings
from pathlib import Path

import numpy as np
import pytest

from circuitinsight import SessionController
from circuitinsight.adapters.spectre.stbdata import load_stb
from circuitinsight.analysis.compensate import Candidate

FIX = Path(__file__).resolve().parent / "fixtures" / "spectre" / "miller"

CLASSIC = Candidate("miller", "I0.net1", "vout",
                    "classic second-stage bridge", 106.0)


@pytest.fixture(scope="module")
def top_suggestion():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c = SessionController.open(FIX / "tb_ota2s_stb.cin.json",
                                   FIX / "psf_stb", cap_model="matrix")
        sugg = c.suggest_compensation(
            "IPRB0", goal="pm", pm_target=60.0,
            exclude=("I0.Cc",), candidates=[CLASSIC])
    return sugg


def test_session_surface_rederives_the_network(top_suggestion):
    s = top_suggestion[0]
    assert s.network == "series-RC" and s.achieved
    assert s.C == pytest.approx(3.177302e-12, rel=1e-6)
    assert s.R == pytest.approx(3747.635, rel=1e-6)
    assert s.pm_deg == pytest.approx(66.74, abs=0.05)
    assert "pF" in s.describe()


def test_spectre_confirms_the_suggested_network(top_suggestion):
    """psf_sugg is a real Spectre stb run of the suggested network. The
    tool's PM prediction -- from the OP of a DIFFERENTLY-compensated run --
    must land on the simulator's verdict."""
    s = top_suggestion[0]
    stb = load_stb(FIX / "psf_sugg")
    assert "stable" in stb.margins["stb_state"]
    assert stb.phase_margin_deg == pytest.approx(66.751, abs=0.01)  # pinned
    assert s.pm_deg == pytest.approx(stb.phase_margin_deg, abs=0.05)
    assert stb.phase_margin_deg >= 60.0                  # the goal held


def test_suggested_network_preserves_the_operating_point():
    """The whole premise: the suggested branch carries no DC, so the
    psf_sugg OP equals the psf_stb OP (different Cc/Rz values, same
    bias)."""
    from circuitinsight.adapters.spectre.opdata import load_dcopinfo

    a = load_dcopinfo(FIX / "psf_stb" / "dcOpInfo.info")
    b = load_dcopinfo(FIX / "psf_sugg" / "dcOpInfo.info")
    for name, ra in a.items():
        if ra.device_type != "mosfet":
            continue
        rb = b[name]
        for p in ("gm", "gds"):
            assert rb.params[p] == pytest.approx(ra.params[p], rel=1e-9)


def test_session_cache_guard_and_exclude():
    """Grid/tolerance overrides and custom candidate lists are NOT part of
    the cache key, so those calls must bypass the cache entirely."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c = SessionController.open(FIX / "tb_ota2s_stb.cin.json",
                                   FIX / "psf_stb", cap_model="matrix")
        s1 = c.suggest_compensation("IPRB0", goal="mfm",
                                    exclude=("I0.Cc",),
                                    candidates=[CLASSIC])
        d1 = c.suggest_compensation(
            "IPRB0", goal="mfm", exclude=("I0.Cc",), candidates=[CLASSIC],
            top=2, c_grid=np.geomspace(1e-12, 30e-12, 10),
            r_grid=np.array([0.0, 2.5e3]))
    assert s1 and s1[0].achieved
    assert d1 and len(d1) <= 2
    assert not any(k[0] == "suggest" for k in c._cache)   # nothing cached
