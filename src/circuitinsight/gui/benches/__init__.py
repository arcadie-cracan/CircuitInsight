"""The analysis benches and the schematic pane, one mixin each, mixed into
MainWindow. Factored out of app.py -- each bench owns
its page builder, its run verbs, its completion
handlers and its drawing."""
from .whatif import WhatIfMixin  # noqa: F401
from .reduce import ReduceBenchMixin  # noqa: F401
from .compensate import CompensateBenchMixin  # noqa: F401
from .modes import ModesBenchMixin  # noqa: F401
from .gft import GFTBenchMixin  # noqa: F401
from .schematic import SchematicMixin  # noqa: F401
