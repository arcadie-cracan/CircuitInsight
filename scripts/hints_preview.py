"""Dev driver: compose the ota5t hints fixture into an SVG preview.

Usage: python scripts/hints_preview.py <symlib-dir> [out.svg]
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
h = load_hints(root / "tests/fixtures/hints/tb_ota5t.hints.json")
flat = flatten(h)
print("flat instances:", sorted(i.name for i in flat.instances))
print("wires:", len(flat.wires), "labels:", len(flat.labels))

c = SessionController.open(
    root / "tests/fixtures/spectre/ota5t/tb_ota5t.cin.json",
    root / "tests/fixtures/spectre/ota5t/psf")
devtypes = {d.name: (d.device_type, d.params.get("polarity"))
            for d in c._run.flat.devices}
print("cin devices:", sorted(devtypes))

lib = SymbolLibrary(sys.argv[1])
svg = compose(flat, devtypes, lib)
out = Path(sys.argv[2] if len(sys.argv) > 2 else "ota5t_hints_preview.svg")
out.write_text(svg, encoding="utf-8")
print("wrote", out, len(svg), "bytes |",
      svg.count("<svg ") - 1, "nested symbols")
