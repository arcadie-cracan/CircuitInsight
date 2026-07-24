"""Cadence-like styling, so the app sits beside Virtuoso instead of shouting.

Virtuoso is a Qt application too, so blending in is a palette + a compact font +
a few widget rules -- not a bitmap skin. Everything lives here and nothing else
in the GUI hard-codes a colour.

Switchable on purpose (`circuitinsight-gui --theme native`): launched from the
CIW you want it to look like part of the toolchain, but a designer running it
standalone may prefer their own desktop's look.

Qt-only: `view.py` stays import-clean (matplotlib, no Qt), so figure styling is
applied from the app, not from the view.
"""
from __future__ import annotations

from PySide6.QtGui import QColor, QFont, QPalette

# Chosen to sit next to Virtuoso IC23 panels rather than glow white beside them.
BG = "#d9d9d9"          # window / panel chrome
BASE = "#ffffff"        # editable surfaces: tables, text
ALT = "#f2f2f2"         # alternating rows
BORDER = "#9a9a9a"
TEXT = "#1a1a1a"
HEADER = "#cccccc"      # table + section headers
SELECT = "#3a6ea5"      # selection blue (tables, text)
ACCENT = "#a80000"      # the red rule under Virtuoso's toolbars, and submenu arrows
DISABLED = "#7a7a7a"

# Virtuoso menus do NOT highlight in blue -- they use a mid-grey bar with the
# text left black. White surface, hairline separators, thin dark border, and a
# dashed tear-off strip along the top.
MENU_BG = "#fdfdfd"
MENU_BORDER = "#5a5a5a"
MENU_SELECT = "#9a9a9a"
MENU_SEP = "#c4c4c4"

# Fusion + the palette above render every standard widget the way Virtuoso does
# -- buttons with Fusion's subtle gradient, native tabs, framed inputs -- so the
# stylesheet is deliberately MINIMAL: only what a palette cannot express. An
# earlier version restyled QPushButton/QComboBox/QTabBar with flat colour boxes,
# which fought Fusion and looked *less* like Virtuoso, not more. This is the same
# recipe Cadence uses: keep Fusion's widgets, recolour via palette, and reach for
# QSS only for the few Cadence-isms Fusion has no notion of.
_QSS = f"""
/* The red rule under Virtuoso's toolbars -- Fusion has no such accent. */
QToolBar {{ border-bottom: 2px solid {ACCENT}; spacing: 2px; }}

/* Grey header sections. Fusion would tint them with the button colour; Virtuoso's
   are a distinct flatter grey. */
QHeaderView::section {{
    background: {HEADER};
    border: 0;
    border-right: 1px solid {BORDER};
    border-bottom: 1px solid {BORDER};
    padding: 2px 4px;
}}

/* Menus: the one deliberate divergence from Fusion. Virtuoso highlights menu
   items with a GREY bar, not the palette's selection blue, and its menus tear
   off with a red submenu arrow. */
QMenuBar {{
    background: {BG};
    border-bottom: 1px solid {BORDER};
}}
QMenuBar::item {{
    background: transparent;
    padding: 3px 8px;
}}
QMenuBar::item:selected {{ background: {MENU_SELECT}; }}
QMenu {{
    background: {MENU_BG};
    border: 1px solid {MENU_BORDER};
    padding: 2px 0;
}}
QMenu::item {{
    padding: 3px 28px 3px 24px;
    background: transparent;
}}
QMenu::item:selected {{
    background: {MENU_SELECT};
    color: {TEXT};                      /* text stays black on the grey bar */
}}
QMenu::item:disabled {{ color: {DISABLED}; }}
QMenu::separator {{
    height: 1px;
    background: {MENU_SEP};
    margin: 3px 6px;
}}
QMenu::right-arrow {{
    width: 8px; height: 8px;
    /* Virtuoso's submenu arrow is red */
    image: none;
    border-left: 4px solid {ACCENT};
    border-top: 3px solid transparent;
    border-bottom: 3px solid transparent;
    margin-right: 6px;
}}
QMenu::tearoff {{
    height: 6px;
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                stop:0 {MENU_SEP}, stop:0.5 {MENU_BG},
                                stop:1 {MENU_SEP});
}}
QMenu::tearoff:selected {{ background: {MENU_SELECT}; }}
"""


def palette() -> QPalette:
    p = QPalette()
    p.setColor(QPalette.Window, QColor(BG))
    p.setColor(QPalette.WindowText, QColor(TEXT))
    p.setColor(QPalette.Base, QColor(BASE))
    p.setColor(QPalette.AlternateBase, QColor(ALT))
    p.setColor(QPalette.Button, QColor(BG))
    p.setColor(QPalette.ButtonText, QColor(TEXT))
    p.setColor(QPalette.Text, QColor(TEXT))
    p.setColor(QPalette.Highlight, QColor(SELECT))
    p.setColor(QPalette.HighlightedText, QColor("#ffffff"))
    p.setColor(QPalette.ToolTipBase, QColor("#ffffe1"))
    p.setColor(QPalette.ToolTipText, QColor(TEXT))
    p.setColor(QPalette.Disabled, QPalette.Text, QColor(DISABLED))
    p.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(DISABLED))
    return p


def apply(app, *, font_pt: int = 8) -> None:
    """Style CircuitInsight the way Cadence styles Virtuoso: the platform-neutral
    Fusion style, recoloured by a palette, with a thin QSS layer only for the few
    Cadence-isms Fusion has no notion of (the toolbar accent rule, grey menu
    selection). Fusion — not the native platform style — is deliberate: it renders
    identically on the Linux compute host and a Windows/Mac desktop, so the tool
    looks the same wherever it runs, next to Virtuoso or standalone.
    """
    app.setStyle("Fusion")
    app.setPalette(palette())
    f = QFont(app.font())
    f.setPointSize(font_pt)          # Virtuoso's chrome is noticeably compact
    app.setFont(f)
    app.setStyleSheet(_QSS)


def style_figure(fig, *, tick_pt: int = 7) -> None:
    """Match a Matplotlib canvas to the surrounding chrome.

    A default white figure reads as a pasted-in island next to grey panels; the
    plot area stays white (it is a data surface), but its surround does not.
    """
    fig.patch.set_facecolor(BG)
    for ax in fig.axes:
        ax.set_facecolor(BASE)
        ax.tick_params(labelsize=tick_pt, width=0.6)
        for spine in ax.spines.values():
            spine.set_linewidth(0.6)
            spine.set_color(BORDER)
        ax.xaxis.label.set_size(tick_pt + 1)
        ax.yaxis.label.set_size(tick_pt + 1)
