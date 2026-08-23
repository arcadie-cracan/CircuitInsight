"""Labels as glyph outlines: matplotlib mathtext -> SVG paths.

The schematic's text used to be <text> elements: the viewer's fonts
decide how they look, a headless rasteriser shows tofu, and a device
alias like M_1 cannot be typeset at all. mathtext renders a TeX subset
(sub/superscripts, Greek, \\mathrm, fractions) to outlines with no
external TeX, in STIX -- the face the symbol library's own equation
placeholders were rendered with -- so labels and symbols match, and a
path is a path in every viewer.

One path per label, cached by (text, size). Coordinates are mm, the
baseline at y = 0 and the anchor at x = 0; the composer places them.
"""
from __future__ import annotations

import re
from functools import lru_cache

PT_PER_MM = 72.0 / 25.4
MM_PER_PT = 25.4 / 72.0


def tex_name(name: str, bold: bool = False) -> str:
    """A plain identifier (instance, net, port name) as mathtext:
    upright, underscores escaped, nothing interpreted."""
    esc = name.replace("\\", r"\backslash ").replace("_", r"\_") \
        .replace("#", r"\#").replace("%", r"\%").replace("&", r"\&")
    cmd = r"\mathbf" if bold else r"\mathrm"
    return f"${cmd}{{{esc}}}$"


def tex_alias(alias: str) -> str:
    """A user's LaTeX alias (M_1, g_{m1}) as mathtext, as written."""
    a = alias.strip()
    if a.startswith("$") and a.endswith("$"):
        return a
    return f"${a}$"


@lru_cache(maxsize=4096)
def render(text: str, size_mm: float) -> tuple:
    """(path_d, width_mm, ascent_mm, descent_mm) for `text` at the
    given x-height-equivalent size: the SVG path data of the outlines
    with the anchor at x = 0 and the baseline at y = 0 (y DOWN, as SVG
    wants), plus the extents the composer needs to anchor it.
    Falls back to upright plain text when mathtext rejects the input."""
    from matplotlib.font_manager import FontProperties
    from matplotlib.textpath import TextPath

    prop = FontProperties(family="STIXGeneral", math_fontfamily="stix",
                          size=size_mm * PT_PER_MM)
    try:
        tp = TextPath((0, 0), text, prop=prop, usetex=False)
    except Exception:                                # noqa: BLE001
        tp = TextPath((0, 0), re.sub(r"[$\\{}]", "", text), prop=prop,
                      usetex=False)
    verts = tp.vertices * MM_PER_PT
    codes = tp.codes
    if codes is None or len(verts) == 0:
        return "", 0.0, 0.0, 0.0
    # matplotlib's Path: MOVETO 1, LINETO 2, CURVE3 3, CURVE4 4, CLOSE 79
    out = []
    i = 0
    n = len(codes)
    while i < n:
        c = int(codes[i])
        x, y = verts[i]
        if c == 1:
            out.append(f"M{x:.3f} {-y:.3f}")
            i += 1
        elif c == 2:
            out.append(f"L{x:.3f} {-y:.3f}")
            i += 1
        elif c == 3:
            x2, y2 = verts[i + 1]
            out.append(f"Q{x:.3f} {-y:.3f} {x2:.3f} {-y2:.3f}")
            i += 2
        elif c == 4:
            x2, y2 = verts[i + 1]
            x3, y3 = verts[i + 2]
            out.append(f"C{x:.3f} {-y:.3f} {x2:.3f} {-y2:.3f} "
                       f"{x3:.3f} {-y3:.3f}")
            i += 3
        elif c == 79:
            out.append("Z")
            i += 1
        else:
            i += 1
    bb = tp.get_extents()
    width = float(bb.x1 - bb.x0) * MM_PER_PT
    ascent = float(max(bb.y1, 0.0)) * MM_PER_PT
    descent = float(max(-bb.y0, 0.0)) * MM_PER_PT
    return "".join(out), width, ascent, descent


def svg_label(x: float, y: float, text: str, size_mm: float,
              anchor: str = "start", baseline: str = "alphabetic",
              fill: str = "#000000", extra: str = "") -> str:
    """An SVG <path> element for `text` placed like a <text> at (x, y)
    with the given text-anchor and dominant-baseline (alphabetic,
    hanging, central). `extra` is appended to the element's attributes."""
    d, width, ascent, descent = render(text, size_mm)
    if not d:
        return ""
    dx = {"start": 0.0, "middle": -width / 2.0, "end": -width}.get(anchor,
                                                                    0.0)
    dy = {"alphabetic": 0.0, "hanging": ascent,
          "central": (ascent - descent) / 2.0}.get(baseline, 0.0)
    return (f'<path d="{d}" transform="translate({x + dx:.3f}, '
            f'{y + dy:.3f})" fill="{fill}" stroke="none"{extra}/>')


def label_width(text: str, size_mm: float) -> float:
    """Measured width in mm, for the label planner."""
    return render(text, size_mm)[1]
