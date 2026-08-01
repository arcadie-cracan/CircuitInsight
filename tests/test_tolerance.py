"""The Kolka tolerance-region term: a simplification's error budget was
met at the NOMINAL operating point; tolerance_margin() reports what it
becomes when parameters drift — sampled exactly, since both sides of the
comparison are numeric."""
import warnings
from pathlib import Path

import pytest

from circuitinsight import Analyzer

GOLDEN = Path(__file__).resolve().parent / "golden" / "circuits"


def test_tolerance_margin_grows_with_spread_and_stays_sane():
    H = Analyzer.from_cin(GOLDEN / "cs_amp.cin.json").tf("V1", "vout")
    Hs = H.simplify(mag_tol_db=0.5, phase_tol_deg=3, fmin=1e3, fmax=1e9)

    m0 = Hs.tolerance_margin(spread=0.0)
    m20 = Hs.tolerance_margin(spread=0.2)
    m50 = Hs.tolerance_margin(spread=0.5)
    # zero spread reproduces the nominal error; the region error can only
    # grow with the region
    assert m0 == pytest.approx(Hs.achieved_mag_err_db, abs=1e-9)
    assert m0 <= m20 <= m50
    # cs_amp's pruning dropped cdb (1.7% of CL): a 50% drift on anything
    # cannot blow the error up by orders of magnitude on this circuit
    assert m50 < 3.0


def test_tolerance_margin_flags_a_marginal_prune():
    """Force a prune that was BARELY legal at nominal (gds at 8.5% of
    1/RL survives 0.5 dB but dies at 1.5 dB): the region check must then
    report a growing error as RL drifts, where the nominal check said
    everything was fine."""
    H = Analyzer.from_cin(GOLDEN / "cs_amp.cin.json").tf("V1", "vout")
    Hs = H.simplify(mag_tol_db=1.5, phase_tol_deg=10, fmin=1e3, fmax=1e9)
    free = {str(x) for x in Hs.expr.free_symbols}
    assert "gds_M1" not in free                    # the marginal prune
    m = Hs.tolerance_margin(spread=0.5)
    assert m > Hs.achieved_mag_err_db * 1.05       # region error exceeds nominal
