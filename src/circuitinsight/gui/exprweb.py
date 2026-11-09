"""Optional KaTeX expression view (QtWebEngine).

Crisp vector math where every device symbol is a live DOM node: hover shows its
operating-point value, a click round-trips to Python (``ExprBridge.symbolClicked``)
-- the identity handle the keep table and the schematic share.

Feature-detected: importing this module never fails. ``WEBENGINE`` tells the app
whether the web view is available; when it is not (PySide6 without the
QtWebEngine addon, e.g. a minimal conda install), the app keeps the matplotlib
mathtext canvas. The core/SessionController never import this module.
"""
from __future__ import annotations

import json
from pathlib import Path

import os

try:
    if os.environ.get("CIN_NO_WEBENGINE"):
        # opt-out for displays where Chromium's start-up stalls the GUI
        # thread (X11 over VNC without GL): the matplotlib/QTextEdit
        # surfaces come back, nothing of Chromium is touched
        raise ImportError("CIN_NO_WEBENGINE set")
    from PySide6.QtCore import QObject, QUrl, Signal, Slot
    from PySide6.QtWebChannel import QWebChannel
    from PySide6.QtWebEngineWidgets import QWebEngineView
    WEBENGINE = True
except Exception:                                     # pragma: no cover
    WEBENGINE = False

if WEBENGINE:
    _SHELL = Path(__file__).resolve().parent / "assets" / "expr.html"

    class ExprBridge(QObject):
        """JS -> Python: clicks on identity-tagged symbols in the expression."""

        symbolClicked = Signal(str)
        selectionChanged = Signal(list)
        termClicked = Signal(str)      # a theta link in the PM sum
        frootClicked = Signal(str)     # an f_p/f_z symbol
        aggClicked = Signal(str)       # RIGHT-click on a G_eff/C_eff

        @Slot(str)
        def on_symbol_click(self, name: str) -> None:
            self.symbolClicked.emit(name)

        @Slot(str)
        def on_term_click(self, name: str) -> None:
            self.termClicked.emit(name)

        @Slot(str)
        def on_froot_click(self, name: str) -> None:
            self.frootClicked.emit(name)

        @Slot(str)
        def on_agg_click(self, name: str) -> None:
            """A G_eff/C_eff asked to say what it is. Right-click, not
            left: the aggregate carries ONE expensive action, and the
            left button in this view already selects symbols."""
            self.aggClicked.emit(name)

        @Slot(str)
        def on_selection_change(self, names: str) -> None:
            """The FULL pinned set after a click, comma separated.

            Sent alongside symbolClicked rather than replacing it: the click
            drives single-symbol actions (scroll the keep table to that row),
            while the set drives anything that mirrors the selection, such as
            the schematic cross-probe."""
            self.selectionChanged.emit(
                [n for n in names.split(",") if n])

    class ExprWebView(QWebEngineView):
        """The Expression tab surface when QtWebEngine is present.

        Push ``view.expr_katex(result, base=...)`` via :meth:`set_payload`;
        rendering is queued until the local shell (assets/expr.html + bundled
        KaTeX, fully offline) finishes loading."""

        def __init__(self, parent=None):
            super().__init__(parent)
            self.bridge = ExprBridge(self)
            self._channel = QWebChannel(self.page())
            self._channel.registerObject("bridge", self.bridge)
            self.page().setWebChannel(self._channel)
            self._ready = False
            self._pending: dict | None = None
            self._underlay()          # no white flash before the shell paints
            self.loadFinished.connect(self._loaded)
            self.load(QUrl.fromLocalFile(str(_SHELL)))

        def _underlay(self) -> None:
            """Chromium paints WHITE until the document does -- on dark the
            pane flashes a white page on load and on theme switch. The
            page's background color is the underlay beneath the document."""
            from PySide6.QtGui import QColor
            from . import theme as _theme
            self.page().setBackgroundColor(QColor(_theme.BASE))

        def _loaded(self, ok: bool) -> None:
            self._ready = bool(ok)
            if self._ready:
                self.set_theme()               # the shell wakes themed
            if self._ready and self._pending is not None:
                self._push(self._pending)
                self._pending = None

        def set_theme(self, name: str | None = None) -> None:
            """Point the shell's CSS tokens at the active theme (or the
            named one); called on load and on a live switch."""
            from . import theme as _theme
            name = name or _theme.current
            self._underlay()
            if self._ready:
                self.page().runJavaScript(
                    f"window.setTheme({json.dumps(name)})")

        def set_payload(self, payload: dict) -> None:
            """Render ``{"lines": [...], "values": {...}}`` (see expr_katex)."""
            if self._ready:
                self._push(payload)
            else:
                self._pending = payload

        def _push(self, payload: dict) -> None:
            self.page().runJavaScript(
                f"window.renderExpr({json.dumps(payload)})")
