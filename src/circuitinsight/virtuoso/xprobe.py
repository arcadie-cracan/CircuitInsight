"""Schematic cross-probe: select in CircuitInsight, highlight in Virtuoso.

Transport is `skillbridge <https://github.com/unihd-cag/skillbridge>`_, per
docs/gui-virtuoso-integration-plan.md Sec. 3.1 (phase 2) and Sec. 3.4: it ships
as the ``circuitinsight[virtuoso]`` extra, so the standalone app keeps zero
Cadence dependency. Importing this module NEVER fails -- ``AVAILABLE`` says
whether the bridge is installed, and ``CrossProbe.connect`` returns None with a
reason when no Virtuoso is listening.

Division of labour: the database traversal stays in SKILL (skill/cin_xprobe.il
-- descending a hierarchical instance path, selecting, zooming and flashing is
what geSelectObject/hiZoomIn are for), and Python only resolves *which*
instance a symbol names and calls the helper. So the fragile, Cadence-version-
dependent part is one loadable .il file, and this module stays unit-testable
against a fake workspace.

Name correlation is the standing risk (plan Sec. 3.2). We do NOT reconstruct an
instance path by string surgery on the symbol name: the join-key spelling
replaces the hierarchy separator with an underscore, which is ambiguous the
moment an instance name legitimately contains one. Instead we match the symbol
against the KNOWN instance list of the reconstructed circuit, longest first, so
the answer is always a device that actually exists.
"""
from __future__ import annotations

try:                                            # optional extra
    from skillbridge import Symbol, Workspace   # type: ignore
    AVAILABLE = True
except Exception:                               # pragma: no cover - env dependent
    Symbol = None                               # type: ignore
    Workspace = None                            # type: ignore
    AVAILABLE = False

__all__ = ["AVAILABLE", "instance_for_symbol", "CrossProbe"]


def instance_for_symbol(symbol: str, instances) -> str | None:
    """The device instance a keep-set symbol belongs to, or None.

    `symbol` is a join key such as ``gm_I0_MN0`` or a bare passive name such as
    ``I0_Cc``; `instances` is the reconstruction's instance list (``I0.MN0``,
    ``I0.Cc``). The instance whose join-key spelling (dots as underscores) ends
    the symbol wins, longest first -- so ``gm_I0_MN0`` resolves to ``I0.MN0``
    and never to a shorter ``MN0`` that happens to exist elsewhere in the
    hierarchy, and an instance containing an underscore still matches because
    we compare against real names rather than splitting the symbol.
    """
    if not symbol:
        return None
    best = None
    for inst in instances:
        key = inst.replace(".", "_")
        if symbol == key or symbol.endswith("_" + key):
            if best is None or len(key) > len(best.replace(".", "_")):
                best = inst
    return best


class CrossProbe:
    """A live connection to a Virtuoso session, or nothing at all.

    Every method is safe to call on a connection that has gone away: Virtuoso
    exiting must degrade the GUI to "cross-probe unavailable", never raise into
    a click handler.
    """

    def __init__(self, ws):
        self.ws = ws
        self._warned = False

    # ------------------------------------------------------------ lifecycle
    @classmethod
    def connect(cls, workspace_id=None):
        """(CrossProbe, None) on success, (None, reason) otherwise."""
        if not AVAILABLE:
            return None, ("skillbridge is not installed; "
                          "pip install circuitinsight[virtuoso]")
        try:
            ws = Workspace.open(workspace_id)
        except Exception as exc:                # no server, bad id, socket gone
            return None, f"no Virtuoso skill server: {exc}"
        probe = cls(ws)
        if not probe.ready():
            return None, ("connected, but CInHighlight is not defined -- "
                          "load skill/cin_xprobe.il in the CIW")
        probe.pin_root()
        return probe, None

    def pin_root(self) -> bool:
        """Pin the cross-probe root to Virtuoso's current window.

        Paths are resolved relative to the top of the exported hierarchy, so
        without pinning, the designer switching to a sub-schematic to look at a
        highlight would break the next probe against that sub-cellview. Best
        effort: an unpinned session still works as long as the top schematic
        stays current."""
        try:
            return bool(self.ws["CInXprobeHere"]())
        except Exception:
            return False

    def ready(self) -> bool:
        """True when the session has our SKILL helpers loaded.

        isCallable takes a SYMBOL, not a string: a Python str serialises to a
        SKILL string ("CInHighlight") and isCallable answers nil for it, which
        looked exactly like "the file was never loaded" even with the helpers
        sitting there working. Symbol() serialises to 'CInHighlight.
        """
        name = Symbol("CInHighlight") if Symbol is not None else "CInHighlight"
        try:
            if bool(self.ws["isCallable"](name)):
                return True
        except Exception:
            pass
        # older/odd releases: fall back to a plain unbound-variable check,
        # which is true for a defined function too
        try:
            return bool(self.ws["boundp"](Symbol("CInHighlight")))
        except Exception:
            return False

    def close(self) -> None:
        try:
            self.ws.close()
        except Exception:
            pass

    # --------------------------------------------------------------- probing
    def highlight(self, instances) -> bool:
        """Select and flash `instances` (hierarchical paths) in the schematic.

        Returns False rather than raising when the call does not land, so a
        selection change in the GUI can never take the app down with it.
        """
        paths = [i for i in dict.fromkeys(instances or []) if i]
        if not paths:
            return False
        try:
            return bool(self.ws["CInHighlight"](paths))
        except Exception:
            return False

    def clear(self) -> bool:
        try:
            return bool(self.ws["CInHighlightClear"]())
        except Exception:
            return False

    def selection(self):
        """Instance paths (and net names) currently selected in Virtuoso, so
        the GUI can adopt the designer's schematic selection. [] on failure."""
        try:
            got = self.ws["CInSelection"]()
        except Exception:
            return []
        if not got:
            return []
        if isinstance(got, str):
            return [got]
        return [str(x) for x in got]
