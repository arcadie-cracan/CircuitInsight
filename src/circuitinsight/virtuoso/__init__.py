"""Optional Virtuoso coupling (M10): schematic cross-probe over skillbridge.

Nothing in the core, the SessionController or the GUI depends on this package
importing successfully -- `xprobe.AVAILABLE` reports whether the bridge is
installed, and every entry point degrades to a no-op with a reason when it is
not. Install with the extra:  pip install circuitinsight[virtuoso]
"""
