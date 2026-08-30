import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


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
