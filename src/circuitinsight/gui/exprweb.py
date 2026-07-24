"""Optional KaTeX expression view (QtWebEngine).

Crisp vector math where every device symbol is a live DOM node: hover shows its
operating-point value, a click round-trips to Python (``ExprBridge.symbolClicked``)
-- the same identity handle a future schematic cross-probe will use.

Feature-detected: importing this module never fails. ``WEBENGINE`` tells the app
whether the web view is available; when it is not (PySide6 without the
QtWebEngine addon, e.g. a minimal conda install), the app keeps the matplotlib
mathtext canvas. The core/SessionController never import this module.
"""
from __future__ import annotations

import json
from pathlib import Path

try:
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

        @Slot(str)
        def on_symbol_click(self, name: str) -> None:
            self.symbolClicked.emit(name)

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
            self.loadFinished.connect(self._loaded)
            self.load(QUrl.fromLocalFile(str(_SHELL)))

        def _loaded(self, ok: bool) -> None:
            self._ready = bool(ok)
            if self._ready and self._pending is not None:
                self._push(self._pending)
                self._pending = None

        def set_payload(self, payload: dict) -> None:
            """Render ``{"lines": [...], "values": {...}}`` (see expr_katex)."""
            if self._ready:
                self._push(payload)
            else:
                self._pending = payload

        def _push(self, payload: dict) -> None:
            self.page().runJavaScript(
                f"window.renderExpr({json.dumps(payload)})")
