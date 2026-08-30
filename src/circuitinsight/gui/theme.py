"""Cadence-like styling, so the app sits beside Virtuoso instead of shouting.

Virtuoso is a Qt application too, so blending in is a palette + a compact font +
a few widget rules -- not a bitmap skin. The chrome palette, the severity
roles and the plot-overlay cycle live here; rangeslider/launch/summaryweb
keep small self-contained palettes of their own.

TWO palettes, one set of names: the module attributes (BG, TEXT, GOOD, ...)
are reassigned by ``set_theme("light"|"dark")``, and every styling site reads
them at RENDER time -- so a theme switch is a token swap plus a repaint, not
an edit of nineteen call sites. Light is the Cadence-like chrome; dark is its
sibling (Virtuoso's own schematic editor is famously black, so a dark
CircuitInsight sits beside it just as naturally).

Switchable on purpose (`circuitinsight-gui --theme native`): launched from the
CIW you want it to look like part of the toolchain, but a designer running it
standalone may prefer their own desktop's look.

Qt-only: `view.py` stays import-clean (matplotlib, no Qt), so figure styling is
applied from the app, not from the view.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtGui import QColor, QFont, QPalette

#: the checked indicator's tick (white, on the selection-blue fill):
#: QSS stops drawing the native check once the indicator is styled
_CHECK_SVG = (Path(__file__).resolve().parent / "assets"
              / "check.svg").as_posix()
#: collapsible group headers use DISCLOSURE chevrons, not checkboxes
#: (a checkbox says "option on", a chevron says "section open")
_CHEV_R = (Path(__file__).resolve().parent / "assets"
           / "chevron-right.svg").as_posix()
_CHEV_D = (Path(__file__).resolve().parent / "assets"
           / "chevron-down.svg").as_posix()
#: splitter-handle grips: three quiet dots say "drag me" without the
#: bright bars Fusion paints by default
_GRIP_H = (Path(__file__).resolve().parent / "assets"
           / "grip-h.svg").as_posix()
_GRIP_V = (Path(__file__).resolve().parent / "assets"
           / "grip-v.svg").as_posix()

# ---- the two palettes ---------------------------------------------------
# Light: chosen to sit next to Virtuoso IC23 panels rather than glow white
# beside them. Dark: Fusion-dark ground with the SAME selection blue and
# red accent, semantic roles lightened for contrast on the dark ground.
_LIGHT = dict(
    BG="#d9d9d9",           # window / panel chrome
    BASE="#ffffff",         # editable surfaces: tables, text
    ALT="#f2f2f2",          # alternating rows
    BORDER="#9a9a9a",
    TEXT="#1a1a1a",
    HEADER="#cccccc",       # table + section headers
    SELECT="#3a6ea5",       # selection blue (tables, text)
    ACCENT="#a80000",       # the red rule under Virtuoso's toolbars
    DISABLED="#7a7a7a",
    # Virtuoso menus do NOT highlight in blue -- a mid-grey bar, text black
    MENU_BG="#fdfdfd",
    MENU_BORDER="#5a5a5a",
    MENU_SELECT="#9a9a9a",
    MENU_SEP="#c4c4c4",
    TOOLTIP_BG="#ffffe1",
    TOOLTIP_TEXT="#1a1a1a",
    # semantic roles shared by the main window
    MUTED="#555",           # secondary labels, hints
    GOOD="#1e5c2f",         # done / within budget (quiet green)
    BAD="#8a1c12",          # over budget / failed
    WARN="#7a5200",         # close to the limit
    INFO="#1a466b",         # the next step, info strips
    PENDING="#888888",      # not yet reached
    WARN_TINT="#f0f4f8",    # value-conflict row tint
    TOL_BAND="#c9962a",     # tolerance-tube fill (with alpha)
    BAND_SPAN="#4a78a8",    # selected-band shading (with alpha)
    SEVERITY={"info": ("#eef4fa", "#1a466b"), "ok": ("#eaf6ec", "#1e5c2f"),
              "warn": ("#fdf3e0", "#7a5200"), "error": ("#fbe9e7", "#8a1c12")},
    # light, colorblind-friendly member tints for match groups (rows
    # in the Instances tree, plates behind matched symbols)
    GROUP_TINTS=("#dbe9f6", "#fbe4d5", "#e2efda", "#ece1f0",
                 "#fff2cc", "#dbeef4"),
    # the schematic's role palette: light is the paper drawing --
    # black ink for every role (identity re-ink), blue net names
    SCH={"paper": "#ffffff", "device": "#000000", "wire": "#000000",
         "net": "#1a466b", "label": "#000000", "port": "#000000",
         "select": "#d55e00"},
)

_DARK = dict(
    BG="#2f3136",
    BASE="#1e2023",
    ALT="#26282c",
    BORDER="#55585e",
    TEXT="#e2e2e2",
    HEADER="#3a3d42",
    SELECT="#3a6ea5",
    ACCENT="#c22b1d",       # the red rule reads against dark chrome too
    DISABLED="#77797d",
    MENU_BG="#26282c",
    MENU_BORDER="#0e0f10",
    MENU_SELECT="#4a4d52",
    MENU_SEP="#3d4045",
    TOOLTIP_BG="#3a3d42",
    TOOLTIP_TEXT="#e2e2e2",
    MUTED="#9aa0a6",
    GOOD="#7fc98f",
    BAD="#e08a7a",
    WARN="#d9b45a",
    INFO="#85b8e8",
    PENDING="#8b8e93",
    WARN_TINT="#333a42",
    TOL_BAND="#c9962a",     # alpha'd fills: the hue holds on dark
    BAND_SPAN="#4a78a8",
    SEVERITY={"info": ("#243342", "#85b8e8"), "ok": ("#24382a", "#7fc98f"),
              "warn": ("#3d3422", "#d9b45a"), "error": ("#40261f", "#e08a7a")},
    # the same six hues as deep washes: readable UNDER light text in
    # the tree, quiet plates behind the green symbols on the canvas
    GROUP_TINTS=("#24384d", "#4a3423", "#2b3d28", "#3a2e44",
                 "#4a4222", "#22404a"),
    # Virtuoso's own schematic-editor palette, inspected from a live
    # IC25 session: black canvas, #00cc66 device artwork, #38befe
    # wires (junction dots and net names ride them), #fe0000 ports,
    # #d8cc00 instance labels
    SCH={"paper": "#000000", "device": "#00cc66", "wire": "#38befe",
         "net": "#38befe", "label": "#d8cc00", "port": "#fe0000",
         "select": "#ffffff"},
)

PALETTES = {"light": _LIGHT, "dark": _DARK}
current = "light"

#: Okabe-Ito plot cycle -- identical in BOTH themes (it reads on dark), and
#: the SAME order as view._OVERLAY_COLORS, which must stay Qt-free and
#: therefore keeps its own copy of these values
OVERLAY = ("#D55E00", "#009E73", "#CC79A7", "#E69F00")
BLUE, GREEN, PINK, ORANGE = "#0072B2", "#009E73", "#CC79A7", "#E69F00"
VERMILION = "#D55E00"


def set_theme(name: str) -> None:
    """Reassign the module tokens to the named palette. Every styling
    site reads them at render time, so switching is this swap plus a
    repaint (MainWindow.apply_theme drives the repaint)."""
    global current
    pal = PALETTES[name]                      # KeyError = caller bug
    globals().update(pal)
    current = name


set_theme("light")                            # the module's ground state


# Fusion + the palette above render every standard widget the way Virtuoso does
# -- buttons with Fusion's subtle gradient, native tabs, framed inputs -- so the
# stylesheet is deliberately MINIMAL: only what a palette cannot express. An
# earlier version restyled QPushButton/QComboBox/QTabBar with flat colour boxes,
# which fought Fusion and looked *less* like Virtuoso, not more. This is the same
# recipe Cadence uses: keep Fusion's widgets, recolour via palette, and reach for
# QSS only for the few Cadence-isms Fusion has no notion of.
def _qss() -> str:
    """The stylesheet for the ACTIVE palette -- a function, not a
    constant, so a theme switch regenerates it."""
    # the header separator must sit DARKER than the header fill on
    # both themes: BORDER happens to on light but is LIGHTER on dark,
    # and a light line under a dark-topped bar is the shading of a
    # pressed bevel -- the header reads as sunk
    header_sep = QColor(HEADER).darker(150).name()
    # Fusion's derived pane outline (window darkened 40%) lands two
    # steps from BASE on the dark palette -- where the instances/nets
    # tree meets its pane's bottom edge the border vanishes. Dark
    # rules THAT pane's border explicitly, derived LIGHTER than the
    # window; light keeps Fusion's own drawing, which reads fine
    # there. main_tabs takes the same flat border: Fusion's pane
    # bevel (dark outer line, light inner highlight) read as a
    # raised button face around the bench. Scoped by name: a blanket
    # QSS pane restyles the result strip's South-tab pane too.
    pane_rule = "" if current == "light" else f"""
QTabWidget#dev_tabs::pane, QTabWidget#main_tabs::pane {{
    border: 1px solid {QColor(BG).lighter(135).name()};
}}
"""
    return f"""{pane_rule}
/* No stylesheet border on toolbars: Qt lays a toolbar's widget children
   over a QSS border (padding or not), which put the buttons on top of
   the red line. The Virtuoso red rule is a real 2px QFrame at the top
   of the central area instead (app.py), where nothing can overlap it. */
QToolBar {{ spacing: 2px; }}

/* Checkbox indicators: Fusion derives the outline by darkening the
   window colour, which on the dark ground is invisible -- an
   explicit box with the theme's border, filled with the selection
   blue when checked. */
QTreeView::indicator, QTableView::indicator, QCheckBox::indicator {{
    width: 11px; height: 11px;
    border: 1px solid {BORDER};
    border-radius: 2px;
    background: {BASE};
}}
QTreeView::indicator:checked, QTableView::indicator:checked,
QCheckBox::indicator:checked {{
    background: {SELECT};
    border-color: {SELECT};
    image: url({_CHECK_SVG});
}}

/* Splitter handles: the chrome color, not Fusion's bright bars,
   with a three-dot grip saying "drag me"; hover lifts them a shade. */
QSplitter::handle {{ background: {BG}; }}
QSplitter::handle:horizontal {{ width: 6px; image: url({_GRIP_V}); }}
QSplitter::handle:vertical {{ height: 6px; image: url({_GRIP_H}); }}
QSplitter::handle:hover {{ background: {HEADER}; }}

/* Collapsible group headers disclose, they do not tick: a chevron
   pointing right when collapsed, down when open. */
QGroupBox::indicator {{
    width: 10px; height: 10px;
    border: none;
    background: transparent;
    image: url({_CHEV_R});
}}
QGroupBox::indicator:checked {{ image: url({_CHEV_D}); }}

/* A checked icon-only tool button (the schematic's overlay toggles)
   needs a visible ON state beyond Fusion's faint sinking. */
QToolButton:checked {{
    background: {MENU_SELECT};
    border: 1px solid {BORDER};
    border-radius: 2px;
}}

/* Grey header sections. Fusion would tint them with the button colour; Virtuoso's
   are a distinct flatter grey. */
QHeaderView::section {{
    background: {HEADER};
    border: 0;
    border-right: 1px solid {header_sep};
    border-bottom: 1px solid {header_sep};
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
    color: {TEXT};                      /* text keeps its ink on the grey bar */
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
    p.setColor(QPalette.ToolTipBase, QColor(TOOLTIP_BG))
    p.setColor(QPalette.ToolTipText, QColor(TOOLTIP_TEXT))
    # Qt derives placeholder text from Text with 50% alpha ONLY when
    # the role is untouched; a themed palette must say it out loud
    # (field report: the filter hint was near-invisible on dark)
    p.setColor(QPalette.PlaceholderText, QColor(MUTED))
    p.setColor(QPalette.Disabled, QPalette.Text, QColor(DISABLED))
    p.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(DISABLED))
    return p


_ASSETS = Path(__file__).resolve().parent / "assets"
_ICON_INK = "#e2e2e2"      # the placeholder ink every icon SVG is drawn in
_icon_cache: dict = {}


def icon(name: str, size: int = 16):
    """A themed QIcon from ``assets/<name>.svg``: the placeholder ink
    re-inked to the active TEXT at load, rendered at 2x for crisp
    high-DPI, cached per (name, theme). Call again after a theme
    switch -- the cache key carries the theme, so stale pixmaps are
    never returned, but a button keeps the icon it was handed."""
    key = (name, current, size)
    ic = _icon_cache.get(key)
    if ic is None:
        from PySide6.QtCore import QByteArray, Qt
        from PySide6.QtGui import QIcon, QPainter, QPixmap
        from PySide6.QtSvg import QSvgRenderer

        svg = (_ASSETS / f"{name}.svg").read_text(encoding="utf-8")
        svg = svg.replace(_ICON_INK, TEXT)
        r = QSvgRenderer(QByteArray(svg.encode("utf-8")))
        pm = QPixmap(size * 2, size * 2)
        pm.fill(Qt.transparent)
        p = QPainter(pm)
        r.render(p)
        p.end()
        pm.setDevicePixelRatio(2.0)
        ic = _icon_cache[key] = QIcon(pm)
    return ic


def app_icon():
    """The CircuitInsight application icon -- Arcadie's chip-with-an-eye
    design -- as one QIcon carrying every exported PNG size, so the
    window manager and taskbar pick the sharpest fit. Theme-independent."""
    from PySide6.QtGui import QIcon

    ic = QIcon()
    for p in sorted((_ASSETS / "appicon").glob("circuitinsight-icon-*.png")):
        ic.addFile(p.as_posix())
    return ic


def frame_shade() -> str:
    """The outline shade Fusion frames its panes with -- the window
    color darkened 40%. BORDER matched that on light by coincidence
    and glared on dark; anything drawing its own frame (the trees'
    proxy style) takes this so it matches the neighbors by
    construction in both themes."""
    return QColor(BG).darker(140).name()


def apply(app, *, font_pt: int = 8) -> None:
    """Style CircuitInsight the way Cadence styles Virtuoso: the platform-neutral
    Fusion style, recoloured by the ACTIVE palette, with a thin QSS layer only
    for the few Cadence-isms Fusion has no notion of (the toolbar accent rule,
    grey menu selection). Fusion — not the native platform style — is deliberate:
    it renders identically on the Linux compute host and a Windows/Mac desktop,
    so the tool looks the same wherever it runs, next to Virtuoso or standalone.
    """
    app.setStyle("Fusion")
    app.setPalette(palette())
    f = QFont(app.font())
    f.setPointSize(font_pt)          # Virtuoso's chrome is noticeably compact
    app.setFont(f)
    app.setStyleSheet(_qss())


def style_figure(fig, *, tick_pt: int = 7) -> None:
    """Match a Matplotlib canvas to the surrounding chrome.

    A default white figure reads as a pasted-in island next to grey panels; the
    plot area stays a data surface (white in light, near-black in dark), but its
    surround does not. Text and spines take the theme's ink -- required on the
    dark ground, invisible-by-identity on the light one.
    """
    fig.patch.set_facecolor(BG)
    for ax in fig.axes:
        ax.set_facecolor(BASE)
        ax.tick_params(labelsize=tick_pt, width=0.6, colors=TEXT)
        for spine in ax.spines.values():
            spine.set_linewidth(0.6)
            spine.set_color(BORDER)
        ax.xaxis.label.set_size(tick_pt + 1)
        ax.yaxis.label.set_size(tick_pt + 1)
        ax.xaxis.label.set_color(TEXT)
        ax.yaxis.label.set_color(TEXT)
        ax.title.set_color(TEXT)
        leg = ax.get_legend()
        if leg is not None:
            for t in leg.get_texts():
                t.set_color(TEXT)
            leg.get_frame().set_facecolor(BASE)
            leg.get_frame().set_edgecolor(BORDER)
    # FIGURE-level legends too (fig.legend, the Bode's below-axis one):
    # the axes sweep above never saw them, so they stayed black-on-dark
    for leg in list(getattr(fig, "legends", []) or []):
        for t in leg.get_texts():
            t.set_color(TEXT)
        leg.get_frame().set_facecolor(BASE)
        leg.get_frame().set_edgecolor(BORDER)
