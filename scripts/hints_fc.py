"""Dev driver: compose the REAL fc harvest against the fc CIN.

Usage: python scripts/hints_fc.py <symlib-dir> <hints.json> [out.svg]
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.simplefilter("ignore")

from circuitinsight import SessionController                    # noqa: E402
from circuitinsight.schematic import (SymbolLibrary, compose,   # noqa: E402
                                      flatten, load_hints)

root = Path(__file__).resolve().parents[1]
h = load_hints(sys.argv[2])

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    c = SessionController.open(
        root / "tests/fixtures/spectre/fc/tb_fc.cin.json",
        root / "tests/fixtures/spectre/fc/psf")
devtypes = {d.name: (d.device_type, d.params.get("polarity"))
            for d in c._run.flat.devices}
cin_defs = set(c._run.flat and [])   # placeholder; real defs below
import json                                                     # noqa: E402
cin_doc = json.loads(
    (root / "tests/fixtures/spectre/fc/tb_fc.cin.json").read_text())
cin_defs = set(cin_doc["definitions"])
print("cin defs:", sorted(cin_defs))

flat = flatten(h, descend=cin_defs)
hint_names = {i.name for i in flat.instances}
joined = hint_names & set(devtypes)
print(f"hints: {len(hint_names)} instances, {len(flat.wires)} wires; "
      f"joined with CIN: {len(joined)}")
print("hints-only:", sorted(hint_names - set(devtypes))[:12])
print("cin-only:", sorted(set(devtypes) - hint_names)[:12])

lib = SymbolLibrary(sys.argv[1])
svg = compose(flat, devtypes, lib)
out = Path(sys.argv[3] if len(sys.argv) > 3 else "fc_hints_preview.svg")
out.write_text(svg, encoding="utf-8")
print("wrote", out, len(svg), "bytes")
