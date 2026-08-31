"""The CI configuration must PARSE, and must still carry its gates.

An invalid workflow file does not degrade gracefully: GitHub refuses
the whole file, so no job runs at all. The lint gate added in the
audit did exactly that -- an unquoted "only: undefined names" in a
step name is a colon-space inside a plain YAML scalar, which is not
allowed -- and the result was silent. Nothing failed; CI simply
stopped happening.

So the guard is not about style. It is about a config whose failure
mode is invisible from inside the repository.
"""
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

WF = Path(__file__).resolve().parents[1] / ".github" / "workflows"


def _workflows():
    if not WF.is_dir():
        pytest.skip("no .github/workflows here (snapshot or export)")
    files = sorted(WF.glob("*.yml")) + sorted(WF.glob("*.yaml"))
    if not files:
        pytest.skip("no workflow files")
    return files


@pytest.mark.parametrize("path", _workflows(),
                         ids=lambda p: p.name)
def test_the_workflow_parses(path):
    """The failure this exists for: yaml.safe_load raising."""
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(doc, dict), f"{path.name} is not a mapping"
    assert doc.get("jobs"), f"{path.name} declares no jobs"
    for job, spec in doc["jobs"].items():
        assert spec.get("steps"), f"{path.name}:{job} has no steps"
        for st in spec["steps"]:
            assert "run" in st or "uses" in st, (
                f"{path.name}:{job} has a step that neither runs nor uses")


def test_ci_still_runs_the_lint_and_the_suite():
    """A parseable file that quietly lost a gate is the next failure
    along, and just as silent. Both gates are named here so removing
    one has to be deliberate."""
    ci = WF / "ci.yml"
    if not ci.is_file():
        pytest.skip("no ci.yml here")
    doc = yaml.safe_load(ci.read_text(encoding="utf-8"))
    runs = " ".join(st.get("run", "")
                    for spec in doc["jobs"].values()
                    for st in spec["steps"])
    assert "ruff check" in runs, "the lint gate is gone"
    assert "pytest" in runs, "the test gate is gone"
