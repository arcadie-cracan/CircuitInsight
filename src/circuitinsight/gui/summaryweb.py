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

try:
    from PySide6.QtWebEngineWidgets import QWebEngineView
    WEBENGINE = True
except Exception:                                     # pragma: no cover
    WEBENGINE = False

_CSS = """
:root { color-scheme: light; }
body { font-family: 'Segoe UI', system-ui, sans-serif; color: #222;
       background: #ffffff; margin: 10px 14px; }
h1 { font-size: 11.5pt; font-weight: 600; color: #1d3a56;
     margin: 0 0 6px 0; }
pre { font-family: 'Cascadia Mono', Consolas, 'DejaVu Sans Mono',
      monospace; font-size: 9.5pt; line-height: 1.45; margin: 0;
      white-space: pre; }
.sec { font-weight: 600; color: #1d3a56; }
.warn { color: #a15c00; }
@media print { body { margin: 0; } }
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
            f"<style>{_CSS}</style></head><body><h1>{title}</h1>"
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
            self.setHtml(summary_html(""))

        def setPlainText(self, text: str) -> None:
            self._text = text or ""
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
