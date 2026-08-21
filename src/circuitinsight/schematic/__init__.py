"""Schematic re-creation from harvested Virtuoso layout hints.

See docs/schematic-hints-plan.md. The pipeline: cin_hints.il harvests
the original schematic geometry; hints.py loads and flattens it;
symmap.py maps CIN devices to the personal SVG symbol library;
compose.py renders the schematic the human already drew;
blocks.py maps subcircuit cells to library symbols (user-confirmed,
proposed from netlist roles) so a block can draw as an amplifier
rather than a box.
"""
from .blocks import (BlockSymbol, BlockSymbolsError,  # noqa: F401
                     confirmed_only, load_block_symbols, propose_all,
                     save_block_symbols, sidecar_path)
from .compose import ComposeError, SymbolLibrary, compose  # noqa: F401
from .hints import (FlatHints, Hints, HintsError, flatten,  # noqa: F401
                    load_hints, page)
from .symmap import SYMBOLS, SymbolSpec, resolve  # noqa: F401
