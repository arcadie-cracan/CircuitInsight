"""Dev driver: one SVG page per CIN definition from a real harvest.

Usage: python scripts/hints_pages.py <symlib-dir> <hints.json> <cin.json> <outdir>
           [--overlay] [--symbols <file>]

Block symbols: <stem>.symbols.json beside the CIN (or --symbols) is
read; only CONFIRMED cells draw as symbols, the rest stay boxes.
"""
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.simplefilter("ignore")

from circuitinsight.schematic import (SymbolLibrary, compose,   # noqa: E402
                                      load_hints, page)
from circuitinsight.schematic.blocks import (               # noqa: E402
    confirmed_only, load_block_symbols, sidecar_path)

argv = [a for a in sys.argv[1:] if not a.startswith("--")]
lib = SymbolLibrary(argv[0])
h = load_hints(argv[1])
cin = json.loads(Path(argv[2]).read_text(encoding="utf-8"))
out = Path(argv[3])
side = (sys.argv[sys.argv.index("--symbols") + 1]
        if "--symbols" in sys.argv else sidecar_path(argv[2]))
mapping = load_block_symbols(side)
drawn = confirmed_only(mapping)
if mapping:
    print(f"block symbols: {len(drawn)} confirmed of {len(mapping)} "
          f"in {side}")
out.mkdir(parents=True, exist_ok=True)
subckts = frozenset(cin["definitions"])

for defname, cdef in cin["definitions"].items():
    if defname not in h.definitions:
        print(f"{defname}: no hints page")
        continue
    devtypes = {i["name"]: (i["device_type"],
                            (i.get("params") or {}).get("polarity"))
                for i in cdef["instances"] if "device_type" in i}
    pg = page(h, defname)
    svg = compose(pg, devtypes, lib, subckts=subckts,
                  overlay=("--overlay" in sys.argv), block_symbols=drawn)
    f = out / f"{defname}.svg"
    f.write_text(svg, encoding="utf-8")
    blocks = [i.name for i in pg.instances if i.cell in subckts]
    print(f"{defname}: {len(pg.instances)} instances "
          f"({len(blocks)} block(s): {blocks}), {len(pg.wires)} wires "
          f"-> {f.name}")
