import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# the solve-time calibration loads lazily from the user cache and every
# observed solve writes back to it: point both at a throwaway directory,
# so the suite neither depends on this machine's learned model nor
# teaches it test-sized lessons (set in the environment, so spawned
# engine processes inherit it)
import os
import tempfile

os.environ["CIRCUITINSIGHT_CACHE_DIR"] = tempfile.mkdtemp(prefix="ci-cal-")


import pytest


@pytest.fixture(autouse=True)
def _no_state_bleed_between_tests():
    """Autosave writes <cin>.last.cistate BESIDE the CIN, and ticks now
    restore from it by default -- a test that opens a fixture in place
    would inherit whatever the previous test left there. Sweep the
    fixture tree before every test; tests that WANT persistence copy
    their fixture into tmp_path and exercise it there."""
    for p in (Path(__file__).parent / "fixtures").rglob("*.cistate"):
        try:
            p.unlink()
        except OSError:
            pass
    yield


@pytest.fixture(autouse=True)
def _no_calibration_bleed_between_tests():
    """Every estimate-then-tf() now teaches the in-process solve-time
    model: restore it after each test, so one test's lessons cannot move
    another's estimates."""
    est = sys.modules.get("circuitinsight.analysis.estimate")
    saved = (est._CAL, est._CAL_SETTLED) if est is not None else None
    yield
    if saved is not None:
        est._CAL, est._CAL_SETTLED = saved
