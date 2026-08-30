"""The two-palette theme registry (T-A) and the live switch (T-B):
same token names, render-time reads, a swap plus a repaint."""
import os

import pytest

pytest.importorskip("PySide6")
try:
    import PySide6.QtWidgets  # noqa: F401
except Exception as exc:                                  # pragma: no cover
    pytest.skip(f"PySide6.QtWidgets not loadable: {exc}",
                allow_module_level=True)
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from circuitinsight.gui import theme  # noqa: E402
from circuitinsight.gui import view  # noqa: E402


@pytest.fixture(autouse=True)
def _light_after():
    yield
    theme.set_theme("light")
    view.set_fig_theme("light")


def test_palettes_share_the_token_set_and_parse():
    from PySide6.QtGui import QColor

    light, dark = theme.PALETTES["light"], theme.PALETTES["dark"]
    assert set(light) == set(dark)
    for pal in (light, dark):
        for key, val in pal.items():
            if key == "SEVERITY":
                assert set(val) == {"info", "ok", "warn", "error"}
                for bg, fg in val.values():
                    assert QColor(bg).isValid() and QColor(fg).isValid()
            elif key == "GROUP_TINTS":
                assert len(val) == 6
                for c in val:
                    assert QColor(c).isValid()
            elif key == "SCH":
                assert set(val) == {"paper", "device", "wire", "net",
                                    "label", "port", "select"}
                for c in val.values():
                    assert QColor(c).isValid()
            else:
                assert QColor(val).isValid(), key


def test_set_theme_swaps_module_tokens_and_qss():
    light_bg = theme.BG
    theme.set_theme("dark")
    assert theme.BG != light_bg and theme.current == "dark"
    assert theme.PALETTES["dark"]["HEADER"] in theme._qss()
    assert theme.SEVERITY["ok"][1] == theme.PALETTES["dark"]["GOOD"]
    theme.set_theme("light")
    assert theme.BG == light_bg and theme.current == "light"
    assert theme.PALETTES["light"]["HEADER"] in theme._qss()


def test_fig_palette_follows_and_okabe_ito_stays():
    ref_light = view.FIGPAL["reference"]
    view.set_fig_theme("dark")
    assert view.FIGPAL["reference"] != ref_light
    view.set_fig_theme("light")
    assert view.FIGPAL["reference"] == ref_light
    # the data cycle is theme-invariant (reads on both grounds)
    assert theme.OVERLAY == view._OVERLAY_COLORS


def test_style_figure_inks_text_with_the_theme():
    from matplotlib.figure import Figure

    theme.set_theme("dark")
    fig = Figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel("f")
    theme.style_figure(fig)
    assert fig.get_facecolor()[:3] != (1.0, 1.0, 1.0)
    assert ax.xaxis.label.get_color() == theme.TEXT


def test_svg_darken_reinks_by_role():
    """The dark schematic follows Virtuoso's own conventions: wires
    (told apart by the verbatim WIRE_STYLE markup) and their junction
    dots go wire-blue, net names ride the wire family, mathtext
    labels go label-yellow, and the remaining black -- the symbol
    artwork -- goes device-green."""
    from circuitinsight.gui.benches.schematic import _svg_darken
    from circuitinsight.schematic.compose import WIRE_STYLE

    sch = theme.PALETTES["dark"]["SCH"]
    svg = ('<svg xmlns="http://www.w3.org/2000/svg">'
           f'<path d="M0 0" {WIRE_STYLE}/>'
           '<path stroke="#000000" fill="white"/>'
           '<path d="M1 1" fill="#000000" stroke="none"/>'
           '<text fill="#1a466b">net8</text>'
           '<circle fill="black"/><text>x</text></svg>')
    out = _svg_darken(svg, sch)
    assert f'stroke="{sch["wire"]}"' in out          # the wire run
    assert f'fill="{sch["wire"]}"' in out            # its junction dot
    assert f'fill="{sch["net"]}"' in out             # the net name
    assert f'fill="{sch["label"]}" stroke="none"' in out   # the label
    assert f'stroke="{sch["device"]}"' in out        # symbol artwork
    assert f'fill="{sch["paper"]}"' in out           # block plates
    # no black STROKE survives (the paper fill is itself black now,
    # true to Virtuoso's canvas)
    assert 'stroke="#000000"' not in out
    assert 'fill="black"' not in out
    assert out.startswith(f'<svg fill="{sch["label"]}" ')


def test_contrib_color_light_pins_hold_and_dark_lerps_from_ink():
    from circuitinsight.schematic.compose import _contrib_color

    # the light pins are unchanged by the lerp generalization
    assert _contrib_color(1.0) == "#D55E00"
    assert _contrib_color(2.0) == "#D55E00"
    # from a light ink the weakest contributor stays LIGHT, so it
    # reads on the dark ground instead of vanishing near black
    dark0 = _contrib_color(0.0, "#E2E2E2")
    assert dark0 != _contrib_color(0.0)
    assert int(dark0[1:3], 16) > 0x80


def test_web_shells_carry_the_dark_tokens():
    from pathlib import Path

    import circuitinsight.gui as g
    from circuitinsight.gui import summaryweb

    shell = (Path(g.__file__).parent / "assets" / "expr.html").read_text(
        encoding="utf-8")
    assert 'data-theme="dark"' in shell and "window.setTheme" in shell
    theme.set_theme("dark")
    dark_html = summaryweb.summary_html("t")
    assert theme.PALETTES["dark"]["BASE"] in dark_html
    theme.set_theme("light")
    assert theme.PALETTES["dark"]["BASE"] not in summaryweb.summary_html("t")


def test_live_switch_repaints_and_persists(tmp_path):
    from PySide6.QtWidgets import QApplication

    from circuitinsight.gui.app import MainWindow

    _ = QApplication.instance() or QApplication([])
    MainWindow.settings_path = str(tmp_path / "gui.ini")
    w = MainWindow()
    try:
        assert not w.a_dark.isChecked()
        w.apply_theme("dark")
        assert theme.current == "dark" and w.a_dark.isChecked()
        from matplotlib.colors import to_hex
        assert to_hex(w.canvas.figure.get_facecolor()) == \
            theme.PALETTES["dark"]["BG"].lower()
        assert str(w._settings().value("ui/theme")) == "dark"
        # and back: the light literals return exactly
        w.apply_theme("light")
        assert theme.BG == theme.PALETTES["light"]["BG"]
        assert to_hex(w.canvas.figure.get_facecolor()) == \
            theme.PALETTES["light"]["BG"].lower()
        assert not w.a_dark.isChecked()
    finally:
        w.close()
        MainWindow.settings_path = None
