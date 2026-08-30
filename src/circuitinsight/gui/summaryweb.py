"""The Summary surface: typeset HTML when QtWebEngine is present.

The summary is structured plain text -- a title line, aligned
``label : value`` rows, formula blocks, advisory sections, warnings.
Rendering it through a web view gives real typography and clean
printing while the CONTENT stays the same text the report export and
the tests read: the widget keeps a plain-text buffer and exposes the
QTextEdit API (``setPlainText``/``toPlainText``) so every caller and
fallback path is identical. Feature-detected like exprweb: without the
QtWebEngine addon this degrades to the read-only QTextEdit it replaced.
"""
from __future__ import annotations

import html as _html

import os

try:
    if os.environ.get("CIN_NO_WEBENGINE"):
        raise ImportError("CIN_NO_WEBENGINE set")   # see exprweb
    from PySide6.QtWebEngineWidgets import QWebEngineView
    WEBENGINE = True
except Exception:                                     # pragma: no cover
    WEBENGINE = False

def _css() -> str:
    """The stylesheet for the ACTIVE theme -- a function so a live
    switch re-renders with the new tokens (the widget re-sets its
    kept text; see MainWindow.apply_theme)."""
    from . import theme

    scheme = "dark" if theme.current == "dark" else "light"
    return f"""
:root {{ color-scheme: {scheme}; }}
body {{ font-family: 'Segoe UI', system-ui, sans-serif; color: {theme.TEXT};
       background: {theme.BASE}; margin: 10px 14px; }}
h1 {{ font-size: 11.5pt; font-weight: 600; color: {theme.INFO};
     margin: 0 0 6px 0; }}
pre {{ font-family: 'Cascadia Mono', Consolas, 'DejaVu Sans Mono',
      monospace; font-size: 9.5pt; line-height: 1.45; margin: 0;
      white-space: pre; }}
.sec {{ font-weight: 600; color: {theme.INFO}; }}
.warn {{ color: {theme.WARN}; }}
/* the webview's scrollbars wear the app's chrome, not Chromium's */
::-webkit-scrollbar {{ width: 12px; height: 12px;
                       background: {theme.BG}; }}
::-webkit-scrollbar-thumb {{ background: {theme.BORDER};
    border: 2px solid {theme.BG}; border-radius: 6px; }}
::-webkit-scrollbar-corner {{ background: {theme.BG}; }}
@media print {{ body {{ margin: 0; }} }}
"""


def summary_html(text: str) -> str:
    """The plain-text summary typeset: first line as the heading, the
    aligned body verbatim in a monospace block (formulas and columns
    keep their alignment), section leads bold, warnings amber."""
    lines = text.splitlines()
    title = _html.escape(lines[0]) if lines else ""
    body = []
    for ln in lines[1:]:
        esc = _html.escape(ln)
        if ln.startswith("⚠"):
            esc = f'<span class="warn">{esc}</span>'
        elif ln.rstrip().endswith(":") and " : " not in ln \
                and not ln.startswith(" "):
            esc = f'<span class="sec">{esc}</span>'
        body.append(esc)
    return (f"<!doctype html><html><head><meta charset='utf-8'>"
            f"<style>{_css()}</style></head><body><h1>{title}</h1>"
            f"<pre>" + "\n".join(body) + "</pre></body></html>")


from PySide6.QtWidgets import QTextEdit


class TextSummaryView(QTextEdit):
    """Fallback without the QtWebEngine addon: the old read-only
    QTextEdit, unchanged."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setLineWrapMode(QTextEdit.NoWrap)


if WEBENGINE:

    class SummaryView(QWebEngineView):
        """QTextEdit-compatible summary surface over a web page."""

        def __init__(self, parent=None):
            super().__init__(parent)
            self._text = ""
            self.setPlainText("")

        def setPlainText(self, text: str) -> None:
            self._text = text or ""
            # Chromium paints WHITE until the document does: keep the
            # page's underlay on the theme's base (re-read here so a
            # theme switch, which re-sets the text, refreshes it too)
            from PySide6.QtGui import QColor
            from . import theme
            self.page().setBackgroundColor(QColor(theme.BASE))
            self.setHtml(summary_html(self._text))

        def toPlainText(self) -> str:
            return self._text


def make(parent=None):
    """The best available summary surface: web when the addon loads,
    the plain text edit otherwise (same API either way)."""
    if WEBENGINE:
        try:
            return SummaryView(parent)
        except Exception:                             # pragma: no cover
            pass                                      # broken GL/WebEngine
    return TextSummaryView(parent)
