"""N-C: goal inversion + area ranking -- the pilot rediscovers Sec. VI.

On the UNCOMPENSATED closed-loop two-stage (stb bench primitives with
I0.Cc stripped -- valid because compensation branches are OP-invariant),
`suggest_compensation` must produce, unprompted, the two networks the
paper's Sec. VI derived by hand: a small series-RC Miller bridge (the
area winner) and the plain Miller capacitor -- sized against either goal
formulation (Delft MFM damping, or the classic PM floor).
"""
import warnings
from pathlib import Path

import numpy as np
import pytest

from circuitinsight.adapters.spectre import SpectreRun
from circuitinsight.analysis.compensate import (Candidate,
                                                suggest_compensation)
from circuitinsight.engine.mna import build_mna

FIX = Path(__file__).resolve().parent / "fixtures" / "spectre" / "miller"

CLASSIC = Candidate("miller", "I0.net1", "vout",
                    "classic second-stage bridge", 106.0)


@pytest.fixture(scope="module")
def sys_uncomp():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run = SpectreRun(FIX / "tb_ota2s_stb.cin.json", FIX / "psf_stb")
        an = run.analyzer(cap_model="matrix")
        prims = [p for p in an.primitives if p.inst != "I0.Cc"]
        return build_mna(prims, an.flat.ground, "IPRB0", an._alias)


def test_pm_goal_rediscovers_section_vi(sys_uncomp):
    """goal='pm' at 60 deg, classic pair only: the area winner must be the
    series-RC network in the 3-pF/3-kOhm class (Sec. VI: 3 pF + 2.5 kOhm),
    with the plain Miller capacitor in the 15-20 pF class (Sec. VI: 16 pF)
    ranked behind it on area. Both must actually meet the margin."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sugg = suggest_compensation(sys_uncomp, "IPRB0", goal="pm",
                                    pm_target=60.0, candidates=[CLASSIC])
    assert len(sugg) == 2 and all(s.achieved for s in sugg)
    rc = sugg[0]
    assert rc.network == "series-RC"
    assert 2e-12 < rc.C < 5e-12
    assert 1.5e3 < rc.R < 6e3
    assert rc.pm_deg >= 60.0
    c = sugg[1]
    assert c.network == "C" and c.R == 0
    assert 13e-12 < c.C < 21e-12
    assert c.pm_deg >= 60.0
    # area ordering is the point
    assert rc.area < c.area
    # and the RC network buys bandwidth on top: larger dominant pair
    assert rc.f0_hz > 2 * c.f0_hz


def test_mfm_goal_holds_the_measured_response_flat(sys_uncomp):
    """goal='mfm' judges the MEASURED closed-loop magnitude: peaking of
    |T/(1-T)| at most 1 dB (the dominant-pair zeta proxy certified fc
    designs peaking +6.6 dB off a doublet, and rejected the flattest
    design as 'overdamped'). The winner is flat, well-margined, and
    CHEAPER than the proxy's pick -- the series R's zero flattens the
    response at less capacitance, Sec. VI's own economy."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sugg = suggest_compensation(sys_uncomp, "IPRB0", goal="mfm",
                                    candidates=[CLASSIC])
    best = sugg[0]
    assert best.achieved
    assert best.budget_hz == pytest.approx(44.95e6, rel=0.01)
    assert best.network == "series-RC" and best.area < 6
    # a flat loop carries real margins by construction
    assert all(s.pm_deg is None or s.pm_deg > 45
               for s in sugg if s.achieved)


def test_probe_side_canonicalization(sys_uncomp):
    """A candidate wired to the probe's q side (vin_n) must be canonicalized
    to the p side (vout): a branch to vin_n would bypass the probe and the
    measured loop gain would no longer cut all loops -- the Tian paper's
    critical-wire requirement."""
    qside = Candidate("miller", "I0.net1", "vin_n", "wired to q side", 106.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sugg = suggest_compensation(sys_uncomp, "IPRB0", goal="mfm",
                                    candidates=[qside])
    assert sugg and all(s.candidate.node_b == "vout" for s in sugg)


def test_describe_is_human(sys_uncomp):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sugg = suggest_compensation(sys_uncomp, "IPRB0", goal="mfm",
                                    candidates=[CLASSIC], top=1)
    text = sugg[0].describe()
    assert "pF" in text and "zeta" in text and "budget" in text


def test_flatness_measures_the_servo_peak():
    """_flatness_db: 0 for a clean single-pole loop, the doublet's
    peak for a resonant one -- in the stb sign convention (critical
    point T = +1)."""
    from circuitinsight.analysis.compensate import _flatness_db

    f = np.geomspace(1e3, 1e9, 800)

    def loop(f2):
        # stb convention: T(0) negative real, instability at T = +1
        return -1000.0 / ((1 + 1j * f / 1e4) * (1 + 1j * f / f2))

    assert _flatness_db(loop(1e8)) < 0.1     # well-separated poles: flat
    assert _flatness_db(loop(1e6)) > 3.0     # tight second pole peaks
