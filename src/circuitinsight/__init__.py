"""CircuitInsight — symbolic small-signal circuit analyzer driven by simulator OP data.

The public names resolve lazily (PEP 562): eager imports here pulled
sympy + numpy into EVERY process that touched the package, including
the GUI launcher whose whole point is showing a banner before the
heavy imports are paid for. ``from circuitinsight import Analyzer``
works exactly as before; it just pays on first use."""

__version__ = "0.0.1"

_EXPORTS = {
    "Analyzer": ".analyzer", "PortEquivalent": ".analyzer",
    "ALL": ".keep", "is_all": ".keep",
    "calibrate": ".analysis.estimate",
    "get_calibration": ".analysis.estimate",
    "TransferFunction": ".engine.mna",
    "DeviceInfo": ".session", "Result": ".session",
    "SessionController": ".session", "SolveTooLarge": ".session",
}


def __getattr__(name):
    mod = _EXPORTS.get(name)
    if mod is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(mod, __name__), name)


def __dir__():
    return sorted(set(globals()) | set(_EXPORTS))
