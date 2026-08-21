"""Propose library symbols for subcircuit blocks, with the evidence.

Usage: python scripts/hints_symbols.py <symlib-dir> <cin.json> [hints.json] [--write]

Prints, per subcircuit cell, the proposed symbol, the pin -> anchor
map, the stubs, and WHY (netlist roles inside and at the parent, name
polarity, position fallbacks). --write stores the proposals UNCONFIRMED
in <stem>.symbols.json beside the CIN, never touching a cell the user
has already confirmed; flip "confirmed" to true to see the symbol.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from circuitinsight.schematic import load_hints            # noqa: E402
from circuitinsight.schematic.blocks import (               # noqa: E402
    load_block_symbols, propose_all, save_block_symbols, sidecar_path)

args = [a for a in sys.argv[1:] if not a.startswith("--")]
lib_dir, cin_path = Path(args[0]), Path(args[1])
hints = load_hints(args[2]) if len(args) > 2 else None
cin = json.loads(cin_path.read_text(encoding="utf-8"))
available = {f.name for f in lib_dir.glob("*.svg")}

proposals = propose_all(cin, available, hints)
if not proposals:
    print("no subcircuit cell matches a library symbol shape")
for cell, bs in proposals.items():
    print(f"{cell}: {bs.symbol}")
    for port, anchor in bs.pins.items():
        print(f"  {port:>10} -> {anchor}")
        for ev in bs.evidence.get(port, []):
            print(f"  {'':>13} {ev}")
    for port, edge in bs.stubs.items():
        print(f"  {port:>10} -> stub ({edge})")
        for ev in bs.evidence.get(port, []):
            print(f"  {'':>13} {ev}")

if "--write" in sys.argv:
    side = sidecar_path(cin_path)
    existing = load_block_symbols(side)
    kept = 0
    for cell, bs in proposals.items():
        if cell in existing and existing[cell].confirmed:
            kept += 1
            continue
        existing[cell] = bs
    save_block_symbols(side, existing)
    print(f"wrote {side} ({kept} confirmed cell(s) left untouched); "
          f"set \"confirmed\": true to draw a symbol")
