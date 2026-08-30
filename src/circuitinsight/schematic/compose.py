"""Compose an SVG schematic from flattened layout hints + the symbol
library.

The layout is the ORIGINAL schematic's: wires are drawn exactly where
the human routed them (uu -> mm, y flipped); each instance's symbol is
registered so its terminal anchors land on the harvested pin positions
with the harvested orientation. Every coordinate goes through ONE
lane-rounding, so collinearity is exact; the scale is the most frequent
pin-pitch ratio, so the common pitch maps to whole millimetres and
symbol anchors land on their pins. An anchor the library spaces
differently from the PDK symbol is BRIDGED by a straight stub
extension; wires are never moved to meet a symbol.

Library conventions honored (reguli_schema.md): wire stroke
0.566929 px, junction dot r ~0.5 mm at >= 3 wire ends, labels 9 pt.
"""
from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, replace
from pathlib import Path

from .hints import ORIENTS, FlatHints
from .symmap import (CELL_SYMBOLS, UNLABELLED_CELLS, SymbolSpec, anchor_for,
                     resolve, resolve_cell)

CELL_GLYPHS = frozenset(CELL_SYMBOLS)

#: the library's terminal-stub style, verbatim but in page units: the
#: stubs carry stroke-width 0.566929 in the symbols' px-scaled frame,
#: which is 0.15 mm on the 1 mm grid (rule 17); drawing wires at
#: 0.567 MM made them four times heavier than the stubs they meet
WIRE_STROKE = 0.15               # mm
WIRE_STYLE = (f'fill="none" stroke="#000000" stroke-width="{WIRE_STROKE}" '
              f'stroke-linecap="round" stroke-linejoin="miter"')
DOT_R = 0.53                     # mm (~2 px)
BODY_PAD = 0.6                   # mm of air around a measured body rect

#: analysis-state styling on the SAME layout (the simplified view):
#: instance tags and net tags the GUI derives from the session
STYLE_REMOVED = "removed"        # nothing of the device is left: faded
STYLE_PASSIVE = "passive"        # its gm died, gds/caps remain: a load
STYLE_LUMPED = "lumped"          # gmb bundled into ĝm: badge
STYLE_SYMBOLIC = "symbolic"      # a symbol of it is in the shown formula
STYLE_NUMERIC = "numeric"        # priced numerically in the shown formula
NET_ACGROUND = "acground"        # AC-grounded net: gray dashed + earth
EMPH = "#1a466b"
ACG_WIRE_STYLE = (f'fill="none" stroke="#6f6f6f" stroke-width="{WIRE_STROKE}" '
                  f'stroke-dasharray="0.9,0.6" stroke-linecap="round"')


def _earth_mark(x, y, size=1.2):
    """A small earth: three shortening bars under a stem at (x, y)."""
    s = size
    return (f'<path d="M {x:.3f} {y:.3f} v {s * 0.8:.3f} '
            f'M {x - s:.3f} {y + s * 0.8:.3f} h {2 * s:.3f} '
            f'M {x - s * 0.62:.3f} {y + s * 1.25:.3f} h {2 * s * 0.62:.3f} '
            f'M {x - s * 0.25:.3f} {y + s * 1.7:.3f} h {2 * s * 0.25:.3f}" '
            f'fill="none" stroke="#6f6f6f" stroke-width="{WIRE_STROKE * 1.4}" '
            f'stroke-linecap="round"/>')
GRID = 1.0                       # mm: every wire end and node on the grid


def _grid(v: float) -> float:
    return round(v / GRID) * GRID


class ComposeError(ValueError):
    pass


#: the symbol library (github.com/arcadie-cracan/circuitinsight-symbols,
#: CC BY 4.0): the default when nothing is configured.
# The library lives INSIDE the package (the submodule's path is
# src/circuitinsight/symbols) so a WHEEL ships it: a pip install from
# git once delivered a GUI whose schematic pane could only shrug --
# the library sat at the repo root, outside everything hatchling packs
BUNDLED_SYMLIB = Path(__file__).resolve().parents[1] / "symbols"


def library_root(explicit=None) -> Path:
    root = explicit or os.environ.get("CIN_SYMLIB")
    if not root and (BUNDLED_SYMLIB / "terminals.csv").exists():
        root = BUNDLED_SYMLIB
    if not root:
        raise ComposeError(
            "no symbol library: set CIN_SYMLIB to the circuitinsight-"
            "symbols checkout (a git submodule of the source tree: "
            "git submodule update --init), or choose it in the GUI")
    root = Path(root)
    if not (root / "terminals.csv").exists():
        raise ComposeError(f"{root} has no terminals.csv — not the "
                           f"symbol library?")
    return root


@dataclass
class Symbol:
    inner: str                   # svg markup between <svg ...> and </svg>
    w: float                     # mm
    h: float                     # mm
    viewbox: str
    anchors: dict                # anchor name -> (x_mm, y_mm), DOCUMENT
    #                              coords (terminals.csv = circle cx/cy
    #                              over 96 dpi, viewBox origin included)
    origin: tuple = (0.0, 0.0)   # viewBox min-x/min-y in mm: pin-in.svg
    #                              starts at x = 10 mm, every other
    #                              symbol at 0 -- anchors must be taken
    #                              RELATIVE to this to land on the canvas
    #: where the library author put the designator: (x_mm, y_mm,
    #: text-anchor) in DOCUMENT coords like the anchors, from the first
    #: lib:role="equation" placeholder; None when the symbol has none
    label_spot: tuple | None = None

    def local(self, anchor: str) -> tuple:
        """Anchor position relative to the symbol canvas, in mm."""
        ax, ay = self.anchors[anchor]
        return (ax - self.origin[0], ay - self.origin[1])


class SymbolLibrary:
    """Loads symbols + their terminal anchors (terminals.csv)."""

    def __init__(self, root=None):
        self.root = library_root(root)
        self._anchors: dict = {}
        for line in (self.root / "terminals.csv").read_text(
                encoding="utf-8").splitlines()[1:]:
            parts = line.strip().split(",")
            if len(parts) == 4:
                f, term, x, y = parts
                self._anchors.setdefault(f, {})[term] = (float(x),
                                                         float(y))
        self._cache: dict = {}

    def anchors_of(self, filename: str) -> dict:
        """anchor name -> (x, y) mm from terminals.csv, {} if unknown."""
        return dict(self._anchors.get(filename, {}))

    def block_candidates(self) -> list:
        """Library files usable as BLOCK symbols: those with at least one
        anchor whose name carries an input/output role (in+/in-/in,
        out/out1/out2), sorted by name."""
        from .blocks import ANCHOR_ROLES
        return sorted(f for f, an in self._anchors.items()
                      if any(a in ANCHOR_ROLES for a in an)
                      and (self.root / f).exists())

    def load(self, filename: str) -> Symbol:
        if filename in self._cache:
            return self._cache[filename]
        text = (self.root / filename).read_text(encoding="utf-8")
        m = re.search(r"<svg\b[^>]*>", text, re.S)
        if not m:
            raise ComposeError(f"{filename}: no <svg> root")
        head = m.group(0)

        def attr(name):
            am = re.search(name + r'="([^"]+)"', head)
            return am.group(1) if am else None

        w = attr("width")
        h = attr("height")
        vb = attr("viewBox")
        if not (w and h):
            raise ComposeError(f"{filename}: missing width/height")

        def as_mm(v):
            # two library dialects: "13mm" (the rescaled symbols) and
            # raw px at 96 dpi (the unscaled glyphs: gnd/node/...)
            return (float(v[:-2]) if v.endswith("mm")
                    else float(v) / 3.7795275)

        if vb is None:
            # no viewBox: user units ARE the px width/height
            vb = (f"0 0 {float(w.replace('mm', ''))} "
                  f"{float(h.replace('mm', ''))}")
        inner = _strip_equations(text[m.end():text.rindex("</svg>")])
        vbv = [float(v) for v in vb.split()]
        w_mm, h_mm = as_mm(w), as_mm(h)
        # viewBox units -> mm through the canvas size; the origin offset
        # is what pin-in.svg's x = 37.8 px start turns into (10 mm)
        origin = (vbv[0] * (w_mm / vbv[2]) if vbv[2] else 0.0,
                  vbv[1] * (h_mm / vbv[3]) if vbv[3] else 0.0)
        sym = Symbol(inner=inner, w=w_mm, h=h_mm, viewbox=vb,
                     anchors=dict(self._anchors.get(filename, {})),
                     origin=origin,
                     label_spot=_designator_spot(
                         text, w_mm / vbv[2] if vbv[2] else 1.0,
                         h_mm / vbv[3] if vbv[3] else 1.0))
        self._cache[filename] = sym
        return sym


#: Cadence port glyph cells: their label is the pin name
PORT_CELLS = frozenset({"ipin", "opin", "iopin"})


def _parse_transform(s: str):
    """An SVG transform attribute as a 3x3 affine (translate, scale,
    matrix; rotate about the origin). Unknown functions are identity."""
    m = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]

    def mul(a, b):
        return [[a[0][0] * b[0][0] + a[0][1] * b[1][0],
                 a[0][0] * b[0][1] + a[0][1] * b[1][1],
                 a[0][0] * b[0][2] + a[0][1] * b[1][2] + a[0][2]],
                [a[1][0] * b[0][0] + a[1][1] * b[1][0],
                 a[1][0] * b[0][1] + a[1][1] * b[1][1],
                 a[1][0] * b[0][2] + a[1][1] * b[1][2] + a[1][2]]]
    for fn, args in re.findall(r"(\w+)\s*\(([^)]*)\)", s or ""):
        v = [float(x) for x in re.split(r"[\s,]+", args.strip()) if x]
        if fn == "translate":
            b = [[1, 0, v[0]], [0, 1, v[1] if len(v) > 1 else 0.0]]
        elif fn == "scale":
            b = [[v[0], 0, 0], [0, v[1] if len(v) > 1 else v[0], 0]]
        elif fn == "matrix" and len(v) == 6:
            b = [[v[0], v[2], v[4]], [v[1], v[3], v[5]]]
        elif fn == "rotate":
            c, s_ = math.cos(math.radians(v[0])), math.sin(math.radians(v[0]))
            b = [[c, -s_, 0], [s_, c, 0]]
        else:
            continue
        m = mul(m, b)
    return m


def _designator_spot(text: str, sx: float, sy: float):
    """(x_mm, y_mm, text-anchor) of the first equation placeholder's
    hidden <text>, composed through its ancestors' transforms, in
    DOCUMENT mm (viewBox units scaled like the anchors). None when
    the symbol carries no placeholder or the markup resists."""
    try:
        import xml.etree.ElementTree as ET
        root = ET.fromstring(text)
    except ET.ParseError:
        return None
    ns_role = [k for k in ("{https://arcadie.dev/symlib}role",) ]
    parent = {c: p for p in root.iter() for c in p}

    def is_eq(el):
        return any(el.get(k) == "equation" for k in ns_role) or \
            el.get("role") == "equation"
    for el in root.iter():
        if not is_eq(el):
            continue
        txt = next((c for c in el.iter()
                    if c.tag.endswith("text") and c.get("x") is not None),
                   None)
        if txt is None:
            continue
        chain, cur = [], txt
        while cur is not None:
            chain.append(cur)
            cur = parent.get(cur)
        m = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        for node in reversed(chain):          # root -> text
            tm = _parse_transform(node.get("transform", ""))
            m = [[m[0][0] * tm[0][0] + m[0][1] * tm[1][0],
                  m[0][0] * tm[0][1] + m[0][1] * tm[1][1],
                  m[0][0] * tm[0][2] + m[0][1] * tm[1][2] + m[0][2]],
                 [m[1][0] * tm[0][0] + m[1][1] * tm[1][0],
                  m[1][0] * tm[0][1] + m[1][1] * tm[1][1],
                  m[1][0] * tm[0][2] + m[1][1] * tm[1][2] + m[1][2]]]
        x, y = float(txt.get("x")), float(txt.get("y"))
        px = m[0][0] * x + m[0][1] * y + m[0][2]
        py = m[1][0] * x + m[1][1] * y + m[1][2]
        style = txt.get("style", "")
        am = re.search(r"text-anchor:\s*(start|middle|end)", style)
        anchor = am.group(1) if am else (txt.get("text-anchor") or "start")
        return (px * sx, py * sy, anchor)
    return None


def _pin_named(inst, pins):
    """The harvested pin whose figure sits on this port glyph (the
    nearest, within one symbol extent); None when the page has none."""
    if not pins:
        return None
    ext = max(abs(inst.bbox[2] - inst.bbox[0]),
              abs(inst.bbox[3] - inst.bbox[1]), 1e-9)
    best = min(pins, key=lambda pn: math.dist(pn["xy"], inst.xy))
    return best if math.dist(best["xy"], inst.xy) <= ext else None


def _strip_equations(inner: str) -> str:
    """Remove the library's label groups (lib:role="equation"): their
    M_1 / v_IN designators are placeholders, name the wrong instance,
    and mirror with a mirrored symbol. The composer places its own
    upright instance labels."""
    out = []
    i = 0
    while True:
        j = inner.find("<g", i)
        if j < 0:
            out.append(inner[i:])
            break
        tag_end = inner.find(">", j)
        tag = inner[j:tag_end + 1]
        if 'lib:role="equation"' not in tag:
            out.append(inner[i:tag_end + 1])
            i = tag_end + 1
            continue
        # balanced-group scan to the matching </g>
        depth = 1
        k = tag_end + 1
        while depth and k < len(inner):
            ng = inner.find("<g", k)
            ng_close = inner.find("</g>", k)
            if ng_close < 0:
                break
            if 0 <= ng < ng_close:
                depth += 1
                k = inner.find(">", ng) + 1
            else:
                depth -= 1
                k = ng_close + len("</g>")
        out.append(inner[i:j])
        i = k
    return "".join(out)


def _orient_svg(m) -> str:
    """The world 2x2 (y-up) as an SVG transform matrix (y-down): flip
    the y axis on both sides — M_svg = F · M · F with F = diag(1,-1)."""
    a, b = m[0]
    c, d = m[1]
    return f"matrix({a}, {-c}, {-b}, {d}, 0, 0)"


FONT_MM = 2.4                    # instance / net label size
PORT_FONT_MM = 2.0               # port names: secondary to instances
FONT_W = 0.56                    # mean glyph advance, em fraction


#: Cadence label justification -> (text-anchor, dominant-baseline)
_JUSTIFY = {
    "upperLeft": ("start", "hanging"),
    "centerLeft": ("start", "central"),
    "lowerLeft": ("start", "alphabetic"),
    "upperCenter": ("middle", "hanging"),
    "centerCenter": ("middle", "central"),
    "lowerCenter": ("middle", "alphabetic"),
    "upperRight": ("end", "hanging"),
    "centerRight": ("end", "central"),
    "lowerRight": ("end", "alphabetic"),
}


def _label_rot(orient: str) -> float:
    """SVG rotation (degrees about the anchor) for a label's Cadence
    orientation: R90 reads bottom-to-top (CCW), R270 top-to-bottom.
    The mirror-rotate composites approximate onto the same two."""
    o = orient or "R0"
    if o in ("R90", "MXR90"):
        return -90.0
    if o in ("R270", "MYR90"):
        return 90.0
    return 0.0


def _rot_box(x, y, text, size, anchor, baseline, rot):
    """The obstacle box of a ROTATED label: the horizontal text box
    around the origin, corners turned by rot, moved to (x, y)."""
    import math as _m

    x0, y0, x1, y1 = _text_box(0.0, 0.0, text, size, anchor, baseline)
    c = _m.cos(_m.radians(rot))
    s = _m.sin(_m.radians(rot))
    pts = [(px * c - py * s, px * s + py * c)
           for px, py in ((x0, y0), (x1, y1), (x0, y1), (x1, y0))]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return (x + min(xs), y + min(ys), x + max(xs), y + max(ys))


def _label_orient_flip(orient: str, anchor: str, baseline: str):
    """A mirrored or half-turned label flows the OTHER way from its
    origin: flip the anchor across the mirror. Rotated labels (R90/
    R270) stay upright at their spot -- an approximation, but an
    honest one."""
    o = orient or "R0"
    flip_h = o in ("MY", "R180")
    flip_v = o in ("MX", "R180")
    if flip_h:
        anchor = {"start": "end", "end": "start"}.get(anchor, anchor)
    if flip_v:
        baseline = {"hanging": "alphabetic",
                    "alphabetic": "hanging"}.get(baseline, baseline)
    return anchor, baseline


def _text_box(x, y, text, size, anchor="start", baseline="alphabetic"):
    """Estimated (x0, y0, x1, y1) of a text run: sans glyphs average
    ~0.56 em wide; alphabetic baseline sits 0.78 em under the top."""
    w = FONT_W * size * len(text)
    x0 = x if anchor == "start" else x - w if anchor == "end" else x - w / 2
    if baseline == "hanging":
        y0 = y
    elif baseline == "central":
        y0 = y - size / 2
    else:
        y0 = y - 0.78 * size
    return (x0, y0, x0 + w, y0 + size)


def _overlap(a, b) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return w * h if w > 0 and h > 0 else 0.0


def _place(text, size, candidates, obstacles, tolerance=0.0):
    """The candidate (x, y, anchor, baseline) covering the least
    obstacle area; the first candidate wins ties (it is the natural
    spot), and also wins outright when its overlap is within
    `tolerance` of its own area (a port name grazing a wire is better
    than a port name somewhere else). Returns (x, y, anchor,
    baseline, box)."""
    best = None
    for i, (x, y, an, bl) in enumerate(candidates):
        box = _text_box(x, y, text, size, an, bl)
        cost = sum(_overlap(box, o) for o in obstacles)
        area = (box[2] - box[0]) * (box[3] - box[1])
        if i == 0 and area > 0 and cost <= tolerance * area:
            return (x, y, an, bl, box)
        key = (round(cost, 3), i)
        if best is None or key < best[0]:
            best = (key, (x, y, an, bl, box))
        if cost == 0:
            break
    return best[1]


def _spot_candidate(sym, m, cx, cy, text, size, fit_body=True):
    """The library author's designator spot as a label candidate in
    world coordinates, carried through orientation m (a mirror swaps
    the text anchor's side); None when the symbol has none.
    fit_body: an end-anchored spot at the canvas edge suits the
    library's short value labels; a longer name that would run back
    over the body starts just right of the pin column instead."""
    if sym.label_spot is None:
        return None
    lx, ly, an = sym.label_spot
    lx -= sym.origin[0]
    ly -= sym.origin[1]
    if fit_body and an == "end" and sym.anchors:
        w_txt = FONT_W * size * len(text)
        right = max(sym.local(k)[0] for k in sym.anchors)
        if lx - w_txt < right + 3.0:       # a body reaches ~3 mm past
            lx, an = right + 3.5, "start"  # its pin column
    a, b = m[0]
    c, d = m[1]
    ox, oy = lx - sym.w / 2.0, ly - sym.h / 2.0
    wx = cx + a * ox + (-b) * oy
    wy = cy + (-c) * ox + d * oy
    if a < 0 and an in ("start", "end"):
        an = "end" if an == "start" else "start"
    return (wx, wy, an, "alphabetic")


def _symbol_rect(cx, cy, sym, m):
    """World-axis extent of a placed symbol canvas under orientation m."""
    a, b = m[0]
    c, d = m[1]
    ew = abs(a) * sym.w + abs(b) * sym.h
    eh = abs(c) * sym.w + abs(d) * sym.h
    return (cx - ew / 2, cy - eh / 2, cx + ew / 2, cy + eh / 2)


def _oriented_local_rect(cx, cy, sym, m, loc):
    """A rect given in the symbol's LOCAL mm frame (0..w, 0..h),
    placed: centered on the canvas, oriented by m, moved to (cx, cy).
    The matrix is applied EXACTLY as _orient_svg hands it to the SVG
    -- M_svg = F.M.F, the y-up world matrix conjugated with the
    y-flip -- so the rect lands where the artwork is drawn (the raw
    matrix put ROTATED bodies beside their symbol: mirrors have zero
    off-diagonals and hid the sign error). Axis-aligned matrices, so
    corner transforms and extents are exact."""
    x0, y0, x1, y1 = loc
    a, b = m[0]
    c, d = m[1]
    xs, ys = [], []
    for lx, ly in ((x0, y0), (x1, y1), (x0, y1), (x1, y0)):
        px, py = lx - sym.w / 2.0, ly - sym.h / 2.0
        xs.append(a * px - b * py)
        ys.append(-c * px + d * py)
    return (cx + min(xs), cy + min(ys), cx + max(xs), cy + max(ys))


#: mm per Virtuoso user unit when a page has no device symbol to
#: calibrate against (a pure-hierarchy page of blocks): a 0.5 uu PDK
#: transistor symbol vs the library's 13 mm one
DEFAULT_SCALE = 26.0

#: mm from a block symbol's canvas edge to its body (the library draws
#: 2 mm anchor stubs; a stub aimed here meets the body, not a stub)
BLOCK_BODY_INSET = 2.5
#: a block symbol grows to its harvested pin pitch, at most this much
BLOCK_MAX_SCALE = 3.0


def _scaled(sym, k: float):
    """The symbol drawn k times larger: canvas, anchors and viewBox
    origin scale together; the inner drawing follows through the
    w / viewBox ratio the embedding applies."""
    # the strokes must NOT grow with the body: divide every stroke
    # width in the drawing by k so the enlarged symbol keeps the
    # library's 0.15 mm line like its neighbours
    def thin(m):
        return (f"{m.group(1)}{float(m.group(2)) / k:.6g}"
                f"{m.group(3) or ''}")
    inner = re.sub(r'(stroke-width[:=]"?)([0-9.]+)(px)?', thin, sym.inner)
    return replace(sym, w=sym.w * k, h=sym.h * k, inner=inner,
                   anchors={n: (x * k, y * k) for n, (x, y) in
                            sym.anchors.items()},
                   origin=(sym.origin[0] * k, sym.origin[1] * k))


def _xml_id(name: str) -> str:
    """inst:NAME as an XML id: dots and brackets survive; anything
    else becomes an underscore."""
    return "inst:" + re.sub(r"[^A-Za-z0-9_.:<>\[\]-]", "_", name)


def _body_id(name: str) -> str:
    """body:NAME -- the id of the instance's ARTWORK alone (symbol
    drawing, block box), so the viewer's selection bounds hug the
    body instead of spanning the labels (field request)."""
    return "body:" + _xml_id(name)[5:]


def _offset_fn(sym, m):
    """anchor name -> (dx, dy) from the symbol center in the svg frame
    under orientation matrix m (viewBox origin removed)."""
    a, b = m[0]
    c, d = m[1]

    def offset(anchor):
        ax, ay = sym.local(anchor)
        lx, ly = ax - sym.w / 2.0, ay - sym.h / 2.0
        return (a * lx + (-b) * ly, (-c) * lx + d * ly)
    return offset


def _register(reg, mode=True):
    """Symbol center from [(harvested mm point, anchor offset)]: the
    MODE of the per-anchor centers, not their mean -- the anchors whose
    pitch the scale reproduces exactly (D-B-S) agree on one center and
    land ON their pins; an anchor the library spaces differently (G)
    is bridged, never allowed to pull the others off their pins.
    mode=False is least squares (the half-mm-rounded mean): for BLOCKS,
    whose library symbol shares no pitch with the PDK symbol, so the
    body sits between its pins and every side gets a short bridge.
    Returns (cx, cy, rms residual)."""
    if mode:
        cand: dict = {}
        for hpt, o in reg:
            c_ = (round((hpt[0] - o[0]) * 2) / 2,
                  round((hpt[1] - o[1]) * 2) / 2)
            cand.setdefault(c_, []).append(c_)
        cx, cy = max(cand.values(), key=len)[0]
    else:
        cx = round(sum(h[0] - o[0] for h, o in reg) / len(reg) * 2) / 2
        cy = round(sum(h[1] - o[1] for h, o in reg) / len(reg) * 2) / 2
    res = math.sqrt(sum(
        (h[0] - (cx + o[0])) ** 2 + (h[1] - (cy + o[1])) ** 2
        for h, o in reg) / len(reg))
    return cx, cy, res


def compose(flat: FlatHints, devtypes: dict, lib: SymbolLibrary,
            *, margin_mm: float = 8.0, subckts=frozenset(),
            overlay: bool = False, block_symbols: dict | None = None,
            styles: dict | None = None, net_styles: dict | None = None,
            proposed_cells=frozenset(), tints: dict | None = None,
            tex_labels: bool = False, aliases: dict | None = None,
            proposed=(), contrib: dict | None = None,
            contrib_base: str = "#000000",
            port_base: str | None = None,
            deactivated=frozenset(), body_measure=None) -> str:
    """SVG document string for one page (a definition, or the flat
    view).

    devtypes: instance name -> (device_type, polarity), the CIN join.
    subckts: cells that are subcircuit definitions — drawn as BLOCKS
    at their harvested symbol bbox with pin stubs at the harvested pin
    positions, exactly like the source schematic shows them.
    overlay: draw the RAW harvest on top — instance bboxes and origins
    (vermilion), harvested terminal positions (dots with names), the
    unsnapped wire polylines (blue) — so fidelity against the source
    can be judged: every deviation of the composed picture from the
    harvest is visible as a gap between black and color.
    Scale is self-calibrating: the most frequent ratio of library
    anchor pitch to harvested pin pitch, so the library symbols sit on
    the original pitch without a magic constant.
    block_symbols: cell -> blocks.BlockSymbol, CONFIRMED mappings only
    (pass through blocks.confirmed_only). Such a block draws as the
    mapped library symbol in whichever of the eight orientations puts
    its anchors nearest the harvested pins; pins without an anchor are
    drawn as named stubs on the nearest symbol edge. Unmapped cells
    stay boxes.
    styles: instance name -> STYLE_* tag, net_styles: net -> NET_*
    tag — the analysis state drawn on the same layout: removed
    instances fade, ĝm-lumped ones wear a badge, instances whose
    symbol is in the shown formula are emphasized, AC-grounded nets
    turn gray-dashed with an earth mark. The geometry never changes.
    proposed_cells: block cells that have a symbol PROPOSED but not
    confirmed — their boxes wear a "?" badge inviting the dialog.
    tints: instance name -> (fill colour, badge text): a matched device
    gets a tinted plate behind its symbol and the badge (🔗n) after its
    name, the same tint and number as the Instances tree.
    tex_labels: labels as mathtext outlines (STIX, the library's own
    face) instead of <text>: viewer-independent, and a device alias
    (aliases: device name -> LaTeX, e.g. M_1) is typeset as math.
    """
    tints = tints or {}
    aliases = aliases or {}

    def label(x, y, text, size=FONT_MM, anchor="start",
              baseline="alphabetic", fill="#000000", bold=False,
              tex_text=None, extra="", rotate=0.0):
        """One label: a <text> or, with tex_labels, the outlines.
        tex_text: the mathtext to use instead of tex_name(text).
        rotate: degrees about the anchor point -- a Cadence R90 net
        label reads along its vertical wire."""
        if tex_labels:
            from .mathtext import svg_label, tex_name
            return svg_label(x, y, tex_text or tex_name(text, bold), size,
                             anchor, baseline, fill, extra,
                             rotate=rotate)
        if rotate:
            extra += f' transform="rotate({rotate:.1f} {x:.2f} {y:.2f})"'
        attrs = (f' text-anchor="{anchor}" dominant-baseline="{baseline}"'
                 + (f' font-size="{size}"' if size != FONT_MM else "")
                 + (f' fill="{fill}"' if fill != "#000000" else "")
                 + (' font-weight="bold"' if bold else "") + extra)
        return f'<text x="{x:.2f}" y="{y:.2f}"{attrs}>{text}</text>'

    def tex_for(inst_name):
        """An instance label's mathtext: the device alias typeset as
        written when there is one, else the upright name."""
        from .mathtext import tex_alias
        a = aliases.get(inst_name) or aliases.get(inst_name.split(".")[-1])
        return tex_alias(a) if a else None

    def badge_text(x, y, badge_tspans, anchor="start"):
        """Badges (ĝ, 🔗n, ?) stay <text>: mathtext has no emoji. Placed
        after a path label at the measured width."""
        if not badge_tspans:
            return ""
        return (f'<text x="{x:.2f}" y="{y:.2f}" text-anchor="{anchor}" '
                f'dominant-baseline="alphabetic">{badge_tspans}</text>')

    block_symbols = block_symbols or {}
    styles = styles or {}
    net_styles = net_styles or {}
    if not flat.instances:
        raise ComposeError("no instances in the hints")

    placed = []                       # (inst, Symbol|None, spec|None)
    blocks = []                       # subcircuit instances
    ratios = []
    for inst in flat.instances:
        if inst.cell in subckts:
            blocks.append(inst)
            continue
        dt = devtypes.get(inst.name)
        spec = resolve(*dt) if dt else resolve_cell(inst.cell)
        sym = lib.load(spec.file) if spec else None
        placed.append((inst, sym, spec))
        if sym:
            bw = abs(inst.bbox[2] - inst.bbox[0])
            bh = abs(inst.bbox[3] - inst.bbox[1])
            if bw > 0 and bh > 0:
                ratios.append(max(sym.w, sym.h) / max(bw, bh))
    # the honest scale is PIN PITCH: harvested terminal distances vs the
    # symbol's anchor distances (the harvested bbox excludes the label
    # room the library canvas includes, so bbox ratios run ~20% large)
    pitch = []
    for inst, sym, spec in placed:
        if not sym or not spec or len(inst.terms) < 2:
            continue
        pairs = [(inst.terms[r], sym.anchors[anchor_for(spec, r)])
                 for r in inst.terms
                 if anchor_for(spec, r) in sym.anchors]
        for i in range(len(pairs)):
            for j in range(i + 1, len(pairs)):
                du = math.dist(pairs[i][0], pairs[j][0])
                dm = math.dist(pairs[i][1], pairs[j][1])
                if du > 1e-9 and dm > 1e-9:
                    pitch.append(dm / du)
    # the MODE, not the median: the library's MOS is 7:4 where the PDK
    # symbol is 4:3, so the median lands on a diagonal pair and no pitch
    # maps to whole millimetres; the most frequent ratio is the one the
    # 1 mm grid must reproduce exactly (D-B-S on every transistor)
    src = pitch or ratios
    s = DEFAULT_SCALE
    if src:
        bins: dict = {}
        for r in src:
            bins.setdefault(round(r, 3), []).append(r)
        best = max(bins.values(), key=lambda v: (len(v), -v[0]))
        s = sum(best) / len(best)

    xs, ys = [], []
    for inst in [p_[0] for p_ in placed] + blocks:
        xs += [inst.bbox[0], inst.bbox[2]]
        ys += [inst.bbox[1], inst.bbox[3]]
    for w in flat.wires:
        xs += [p[0] for p in w.points]
        ys += [p[1] for p in w.points]
    x0, y1 = min(xs), max(ys)

    def mm_raw(p):
        return ((p[0] - x0) * s + margin_mm, (y1 - p[1]) * s + margin_mm)

    compose.last_mm = lambda p: mm(p)

    def mm(p):
        """Harvest -> grid lanes. ONE rounding for every coordinate, so
        equal harvest coordinates are equal lanes (collinearity is
        exact) and pitches that scale to whole mm stay whole: a 1e-6
        pre-round keeps float noise from tipping a .5 case."""
        x, y = mm_raw(p)
        return (_grid(round(x, 6)), _grid(round(y, 6)))

    W = _grid((max(xs) - x0) * s) + 2 * margin_mm
    H = _grid((y1 - min(ys)) * s) + 2 * margin_mm

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" '
             f'width="{W:.2f}mm" height="{H:.2f}mm" '
             f'viewBox="0 0 {W:.3f} {H:.3f}" font-family="sans-serif" '
             f'font-size="2.4">']

    # ---- confirmed block symbols become placed symbols. The harvested
    # orient placed the PDK symbol; the library symbol may order its
    # anchors differently, so the orientation is chosen by geometry:
    # the one of the eight with the smallest registration residual,
    # the harvested orient winning ties.
    stubbed = []                      # (inst, sym, spec, stubs)
    fitted = []                       # scalable DEVICE symbols, refitted
    todo = [(inst, None) for inst in list(blocks)]
    # devices whose symbol has no canonical size (SymbolSpec.scalable:
    # the balun) are fitted the same way, shrinking when the library
    # draws them larger than the PDK footprint
    todo += [(inst, (sym, spec)) for inst, sym, spec in placed
             if sym and spec and spec.scalable and inst.terms]
    for inst, device in todo:
        if device is None:
            bs = block_symbols.get(inst.cell)
            if not bs or not bs.confirmed:
                continue
            sym = lib.load(bs.symbol)
            spec = SymbolSpec(bs.symbol, dict(bs.pins))
            kmin = 1.0
        else:
            sym, spec = device
            kmin = 0.4
        pairs = [(mm(inst.terms[r]), anchor_for(spec, r))
                 for r in inst.terms if anchor_for(spec, r) in sym.anchors]
        if not pairs:
            continue
        # a block symbol has no canonical size: scale it to the
        # harvested pin pitch -- the SMALLEST pair ratio, which matches
        # the tightest pitch (in+/in-) exactly and keeps the symbol
        # inside its footprint; wider pitches (the output) bridge
        # straight along their own axis
        ks = []
        for i in range(len(pairs)):
            for j in range(i + 1, len(pairs)):
                du = math.dist(pairs[i][0], pairs[j][0])
                dm = math.dist(sym.anchors[pairs[i][1]],
                               sym.anchors[pairs[j][1]])
                if du > 1e-9 and dm > 1e-9:
                    ks.append(du / dm)
        if ks:
            k = min(max(min(ks), kmin), BLOCK_MAX_SCALE)
            if abs(k - 1.0) > 1e-9:
                sym = _scaled(sym, k)
        best = None
        for name, m in ORIENTS.items():
            off = _offset_fn(sym, m)
            _cx, _cy, res = _register([(h, off(an)) for h, an in pairs],
                                      mode=False)
            key = (round(res, 6), 0 if m == inst.m else 1)
            if best is None or key < best[0]:
                best = (key, m)
        inst2 = replace(inst, m=best[1])
        if device is None:
            blocks.remove(inst)
            placed.append((inst2, sym, spec))
            stubbed.append((inst2, sym, spec,
                            {r: bs.stubs.get(r, "auto") for r in inst.terms
                             if anchor_for(spec, r) not in sym.anchors}))
        else:
            placed[[p_[0] for p_ in placed].index(inst)] = (inst2, sym, spec)
            fitted.append(inst2.name)

    # least-squares centering for everything fitted (blocks and
    # scalable devices): no exact pitch to preserve
    block_names = {s[0].name for s in stubbed} | set(fitted)

    # ---- wires first (symbols overlay them; rule 5's masking order)
    ends: dict = {}
    anchor_world: dict = {}
    residuals: dict = {}
    for inst, sym, spec in placed:
        if not sym or not spec:
            continue
        offset = _offset_fn(sym, inst.m)

        # REGISTER: the symbol center that puts its anchors on the
        # harvested terminal positions (least squares = mean offset).
        # Glyphs (rails, pins) register their one anchor onto the
        # instance ORIGIN, which is their connection point; the
        # bbox center is the last resort when nothing was harvested.
        reg = [(mm(inst.terms[r]), offset(anchor_for(spec, r)))
               for r in inst.terms
               if anchor_for(spec, r) in sym.anchors]
        if not reg and "@origin" in spec.terms \
                and spec.terms["@origin"] in sym.anchors:
            reg = [(mm(inst.xy), offset(spec.terms["@origin"]))]
        if reg:
            cx, cy, residuals[inst.name] = _register(
                reg, mode=inst.name not in block_names)
        else:
            cx, cy = mm((((inst.bbox[0] + inst.bbox[2]) / 2.0),
                         ((inst.bbox[1] + inst.bbox[3]) / 2.0)))
        pts = {anchor: (cx + offset(anchor)[0], cy + offset(anchor)[1])
               for anchor in spec.terms.values() if anchor in sym.anchors}
        anchor_world[inst.name] = (cx, cy, pts)
    compose.last_residuals = residuals

    def _term_box(inst, pad_mm=2.0):
        """A block's extent from its TERMINALS (the harvested bbox
        includes the property labels): the pin extent plus a margin."""
        if not inst.terms:
            q0 = mm((inst.bbox[0], inst.bbox[3]))
            q1 = mm((inst.bbox[2], inst.bbox[1]))
            return q0, q1
        ps = [mm(v) for v in inst.terms.values()]
        xs_ = [p_[0] for p_ in ps]
        ys_ = [p_[1] for p_ in ps]
        return ((min(xs_) - pad_mm, min(ys_) - pad_mm),
                (max(xs_) + pad_mm, max(ys_) + pad_mm))

    for inst in blocks:
        cx, cy = mm((((inst.bbox[0] + inst.bbox[2]) / 2.0),
                     ((inst.bbox[1] + inst.bbox[3]) / 2.0)))
        anchor_world[inst.name] = (
            cx, cy,
            {k: mm(v) for k, v in inst.terms.items()})

    # Wires go exactly where the designer routed them, lane-rounded by
    # mm(). Nothing is snapped or dragged: an anchor the library spaces
    # differently from the PDK symbol (the MOS gate, 7 mm vs 5.33) gets
    # a BRIDGE -- a straight stub extension from the anchor to the
    # harvested pin lane -- so the symbol never pulls a wire off its
    # column (fc PM6-NM6 drain jog, PM13/NM8 gate reroutes).
    #
    # The one exception to never-touch-a-wire (fc PM7 field report):
    # when the mismatch has the OPPOSITE sign -- the harvested pin
    # sits INSIDE our symbol, between the gate plate and the channel,
    # because the library spaces the anchor farther out than the PDK
    # -- the wire visibly pokes through the gate plate (a MOS symbol
    # is open strokes; rule 5's masking has nothing there to swallow
    # it). A wire END on such a pin is TRIMMED back to the anchor:
    # only an end, only collinear along its last segment, and only
    # when nothing else touches that point -- a through bus, a tap or
    # a junction is never cut. The pin's bridge is dropped with it
    # (the wire now ends ON the anchor).
    def _k05(p):
        return (round(p[0] * 2) / 2, round(p[1] * 2) / 2)

    wire_pts = [[mm(p) for p in w.points] for w in flat.wires]
    term_anchor: dict = {}            # pin key -> anchor point
    for inst, sym, spec in placed:
        if not sym or not spec or inst.name not in anchor_world:
            continue
        apts = anchor_world[inst.name][2]
        for r, xy in inst.terms.items():
            an = apts.get(anchor_for(spec, r))
            if an is None:
                continue
            k = _k05(mm(xy))
            # two pins on one point: ambiguous, leave alone
            term_anchor[k] = None if k in term_anchor else an
    end_count: dict = {}
    segs_all = []
    for pts_ in wire_pts:
        for p in (pts_[0], pts_[-1]):
            k = _k05(p)
            end_count[k] = end_count.get(k, 0) + 1
        segs_all += [(pts_[i], pts_[i + 1])
                     for i in range(len(pts_) - 1)]

    def _interior(p, a, b):
        if abs(a[0] - b[0]) < 1e-6:
            return (abs(p[0] - a[0]) < 1e-6
                    and min(a[1], b[1]) + 1e-6 < p[1]
                    < max(a[1], b[1]) - 1e-6)
        if abs(a[1] - b[1]) < 1e-6:
            return (abs(p[1] - a[1]) < 1e-6
                    and min(a[0], b[0]) + 1e-6 < p[0]
                    < max(a[0], b[0]) - 1e-6)
        return False

    def _between(v, lo, hi):
        lo, hi = (lo, hi) if lo <= hi else (hi, lo)
        return lo + 1e-6 < v < hi - 1e-6

    trimmed_pins: set = set()
    for pts_ in wire_pts:
        if len(pts_) < 2:
            continue
        for idx, prv in ((len(pts_) - 1, len(pts_) - 2), (0, 1)):
            e, pv = pts_[idx], pts_[prv]
            k = _k05(e)
            an = term_anchor.get(k)
            if not an or end_count.get(k, 0) != 1:
                continue
            if any(_interior(e, a_, b_) for a_, b_ in segs_all):
                continue                      # a tap on a bus lane
            if abs(e[0] - pv[0]) < 1e-6 and abs(an[0] - e[0]) < 1e-6                     and _between(an[1], pv[1], e[1]):
                pts_[idx] = (e[0], an[1])
            elif abs(e[1] - pv[1]) < 1e-6 and abs(an[1] - e[1]) < 1e-6                     and _between(an[0], pv[0], e[0]):
                pts_[idx] = (an[0], e[1])
            else:
                continue
            trimmed_pins.add(k)

    bridges = []
    for inst, sym, spec in placed:
        if not sym or not spec or inst.name not in anchor_world:
            continue
        pts = anchor_world[inst.name][2]
        pairs = [(mm(xy), pts[anchor_for(spec, r)])
                 for r, xy in inst.terms.items()
                 if anchor_for(spec, r) in pts]
        if not pairs and "@origin" in spec.terms                 and spec.terms["@origin"] in pts:
            pairs = [(mm(inst.xy), pts[spec.terms["@origin"]])]
        for pin, an in pairs:
            if math.dist(pin, an) < 1e-6 or _k05(pin) in trimmed_pins:
                continue
            dx, dy = abs(pin[0] - an[0]), abs(pin[1] - an[1])
            if dx > 1e-6 and dy > 1e-6:
                # an L whose SHORT leg sits at the anchor, against the
                # symbol's own stub, so the long leg runs on the pin's
                # lane and the jog hides at the symbol
                knee = (an[0], pin[1]) if dx > dy else (pin[0], an[1])
                bridges.append([an, knee, pin])
            else:
                bridges.append([an, pin])
    # the match-group plates are COLLECTED here and spliced in at this
    # index: a plate must sit UNDER the wire layer, or its fill covers
    # the wire tails beneath the symbol and the terminals read as
    # disconnected (field report, PM1's plate eating its wires)
    plate_slot = len(parts)
    plates: list = []

    # named STUBS for block pins the symbol has no anchor for (bias,
    # supplies): from the pin lane straight to the nearest edge of the
    # symbol's anchor extent, the pin name beside it. Real connections
    # stay visible; nothing is dropped.
    for inst, sym, spec, stubs in stubbed:
        cx, cy, _ = anchor_world[inst.name]
        a, b = inst.m[0]
        c, d = inst.m[1]
        inset = BLOCK_BODY_INSET
        corners = []
        for lx, ly in ((inset - sym.w / 2, inset - sym.h / 2),
                       (sym.w / 2 - inset, sym.h / 2 - inset)):
            corners.append((cx + a * lx + (-b) * ly,
                            cy + (-c) * lx + d * ly))
        for lx, ly in ((inset - sym.w / 2, sym.h / 2 - inset),
                       (sym.w / 2 - inset, inset - sym.h / 2)):
            corners.append((cx + a * lx + (-b) * ly,
                            cy + (-c) * lx + d * ly))
        xs_ = [q[0] for q in corners]
        ys_ = [q[1] for q in corners]
        bx0, bx1, by0, by1 = min(xs_), max(xs_), min(ys_), max(ys_)
        for r, edge in stubs.items():
            px, py = mm(inst.terms[r])
            if edge == "auto":
                edge = min((("top", by0 - py), ("bottom", py - by1),
                            ("left", bx0 - px), ("right", px - bx1)),
                           key=lambda e: -e[1])[0]
            if edge in ("top", "bottom"):
                ex = min(max(px, bx0), bx1)
                ey = by0 if edge == "top" else by1
                pts_ = [(px, py), (px, ey), (ex, ey)] if ex != px \
                    else [(px, py), (ex, ey)]
                tx, ty, ta = px + 0.8, (py + ey) / 2.0, "start"
            else:
                ey = min(max(py, by0), by1)
                ex = bx0 if edge == "left" else bx1
                pts_ = [(px, py), (ex, py), (ex, ey)] if ey != py \
                    else [(px, py), (ex, ey)]
                tx, ty, ta = (px + ex) / 2.0, py - 0.8, "middle"
            bridges.append(pts_)
            parts.append(label(tx, ty, r, 1.8, ta, "central"))
    for pts in bridges:
        d = "M " + " L ".join(f"{x:.3f} {y:.3f}" for x, y in pts)
        parts.append(f'<path d="{d}" {WIRE_STYLE}/>')

    raw_wires = [[mm_raw(p) for p in w.points] for w in flat.wires]
    earthed: set = set()
    segs: list = []                   # (net, (x0, y0), (x1, y1)) lanes
    wire_ends: list = []              # (net, point)
    net_wires: dict = {}              # net -> [polyline, ...] for the
    for w, pts in zip(flat.wires, wire_pts):   # viewer's net highlight
        net_wires.setdefault(w.net, []).append(pts)
        segs += [(w.net, pts[i], pts[i + 1]) for i in range(len(pts) - 1)]
        wire_ends += [(w.net, pts[0]), (w.net, pts[-1])]
        for p in (pts[0], pts[-1]):
            key = (round(p[0] * 2) / 2, round(p[1] * 2) / 2)
            ends[key] = ends.get(key, 0) + 1
        d = "M " + " L ".join(f"{x:.3f} {y:.3f}" for x, y in pts)
        if net_styles.get(w.net) == NET_ACGROUND:
            parts.append(f'<path d="{d}" {ACG_WIRE_STYLE}/>')
            if w.net not in earthed and len(pts) >= 2:
                # one earth per net, at the midpoint of its first segment
                mx = (pts[0][0] + pts[1][0]) / 2.0
                my = (pts[0][1] + pts[1][1]) / 2.0
                parts.append(_earth_mark(mx, my))
                earthed.add(w.net)
        else:
            parts.append(f'<path d="{d}" {WIRE_STYLE}/>')

    # the pin-anchor bridges BELONG to their nets: without them the
    # viewer's net runs stopped at the harvested wire ends, and both
    # the net highlight and the sensitivity map's joining run left
    # the last millimetres to every symbol unlit (field report: the
    # cyan gap between the OTA's output pin and the vout junction).
    # A bridge's pin end coincides with a wire end -- that names it.
    endnet = {}
    for net_, p_ in wire_ends:
        endnet[(round(p_[0], 3), round(p_[1], 3))] = net_
    for pts in bridges:
        for probe_pt in (pts[-1], pts[0]):
            net_ = endnet.get((round(probe_pt[0], 3),
                               round(probe_pt[1], 3)))
            if net_ is not None:
                net_wires.setdefault(net_, []).append(
                    [tuple(q) for q in pts])
                break

    # junction dots (rule 6): >= 3 wire ends meeting, or a T -- one
    # wire end landing on the INTERIOR of another wire's segment of
    # the same net (the source draws a dot there; end-counting alone
    # missed every T)
    dots = {k for k, n in ends.items() if n >= 3}
    for net, (x, y) in wire_ends:
        key = (round(x * 2) / 2, round(y * 2) / 2)
        if key in dots:
            continue
        for snet, (ax, ay), (bx, by) in segs:
            if snet != net:
                continue
            if abs(ax - bx) < 1e-6:                      # vertical
                on = (abs(x - ax) < 1e-6
                      and min(ay, by) + 1e-6 < y < max(ay, by) - 1e-6)
            elif abs(ay - by) < 1e-6:                    # horizontal
                on = (abs(y - ay) < 1e-6
                      and min(ax, bx) + 1e-6 < x < max(ax, bx) - 1e-6)
            else:
                on = False
            if on:
                dots.add(key)
                break
    compose.last_dots = set(dots)
    compose.last_net_wires = net_wires
    for (x, y) in dots:
        parts.append(f'<circle cx="{x:.3f}" cy="{y:.3f}" '
                     f'r="{DOT_R}" fill="black"/>')

    # ---- label planning. Obstacles: symbol canvases, wire segments,
    # then the designer's net labels (which keep their spot unless it
    # lies on a symbol body, then nudge vertically); instance labels
    # choose right / left / above / below by least overlap and claim
    # their box so later labels avoid them.
    obstacles: list = []
    for inst, sym, spec in placed:
        if sym is None or inst.name not in anchor_world:
            continue
        cx_, cy_, _ = anchor_world[inst.name]
        obstacles.append(_symbol_rect(cx_, cy_, sym, inst.m))
    sym_rects = list(obstacles)
    for _net, (ax, ay), (bx, by) in segs:
        obstacles.append((min(ax, bx) - 0.3, min(ay, by) - 0.3,
                          max(ax, bx) + 0.3, max(ay, by) + 0.3))
    net_label_pos: list = []
    label_boxes: list = []
    for lab in flat.labels:
        x, y = mm(lab["xy"])
        txt = lab["text"]
        # the label's own JUSTIFICATION and orientation decide which
        # way the text flows from its origin -- dropping them put
        # every re-drawn label lower-left of its spot, visibly wrong
        # on right-justified and mirrored labels (field report)
        an0, bl0 = _JUSTIFY.get(lab.get("just") or "lowerLeft",
                                ("start", "alphabetic"))
        an0, bl0 = _label_orient_flip(lab.get("orient") or "R0",
                                      an0, bl0)
        rot = _label_rot(lab.get("orient"))
        if rot:
            # a rotated label reads ALONG its wire: keep the
            # designer's spot verbatim, register the turned box
            box = _rot_box(x, y, txt, FONT_MM, an0, bl0, rot)
            net_label_pos.append((x, y, txt, an0, bl0, rot))
            label_boxes.append(box)
            obstacles.append(box)
            continue
        cands = [(x, y, an0, bl0)]
        cands += [(x, y + dy, an0, bl0)
                  for dy in (-2.0, 2.0, -4.0, 4.0)]
        # the designer's spot is kept unless it sits on a symbol body
        # or an earlier label; wires alone do not move it
        px, py, an, bl, box = _place(txt, FONT_MM, cands,
                                     sym_rects + label_boxes)
        net_label_pos.append((px, py, txt, an, bl, 0.0))
        label_boxes.append(box)
        obstacles.append(box)

    # the mapped blocks' FIXED captions (name + cell under the body)
    # are drawn without planning -- register their boxes BEFORE any
    # instance label places, in a pre-pass: doing it inside the
    # placement loop was order-dependent, and VPRB placed before I0
    # registered (field report: the name climbed onto the caption)
    for inst, sym, spec in placed:
        if (sym is None or inst.name not in anchor_world
                or inst.cell not in block_symbols):
            continue
        cxb, cyb, _ = anchor_world[inst.name]
        extb = max(sym.w, sym.h) / 2.0
        obstacles.append(_text_box(cxb, cyb + extb + 1.5,
                                   inst.name, FONT_MM,
                                   "middle", "hanging"))
        obstacles.append(_text_box(cxb, cyb + extb + 5.2,
                                   f"({inst.cell})", 1.8,
                                   "middle", "hanging"))

    inst_label_pos: dict = {}
    port_label_pos: dict = {}
    for inst, sym, spec in placed:
        if sym is None or inst.name not in anchor_world:
            continue
        if inst.cell in block_symbols:
            continue
        if inst.cell in UNLABELLED_CELLS:
            continue                   # rails/no-connects: anonymous
        cx_, cy_, pts_ = anchor_world[inst.name]
        r = _symbol_rect(cx_, cy_, sym, inst.m)
        if inst.cell in PORT_CELLS:
            # a port's name goes on its FREE end, away from the wire:
            # the side of the glyph opposite its connection anchor
            pn = _pin_named(inst, flat.pins)
            if not pn or not pts_:
                continue
            ax, ay = next(iter(pts_.values()))
            gap = 0.8
            spot = _spot_candidate(sym, inst.m, cx_, cy_, pn["name"],
                                   PORT_FONT_MM, fit_body=False)
            if abs(ax - cx_) >= abs(ay - cy_):          # horizontal glyph
                if ax > cx_:                            # wire on the right
                    first = (r[0] - gap, cy_ + 0.35 * PORT_FONT_MM, "end",
                             "alphabetic")
                else:
                    first = (r[2] + gap, cy_ + 0.35 * PORT_FONT_MM, "start",
                             "alphabetic")
                cands = [first,
                         (cx_, r[1] - gap, "middle", "alphabetic"),
                         (cx_, r[3] + gap, "middle", "hanging")]
                if spot:
                    cands.insert(0, spot)   # the author's spot first
            else:                                       # vertical glyph
                if ay > cy_:                            # wire below
                    first = (cx_, r[1] - gap, "middle", "alphabetic")
                else:
                    first = (cx_, r[3] + gap, "middle", "hanging")
                cands = [first,
                         (r[2] + gap, cy_ + 0.35 * PORT_FONT_MM, "start",
                          "alphabetic"),
                         (r[0] - gap, cy_ + 0.35 * PORT_FONT_MM, "end",
                          "alphabetic")]
                if spot:
                    cands.insert(0, spot)
            own = [o for o in obstacles if o != r]
            px, py, an, bl, box = _place(pn["name"], PORT_FONT_MM, cands,
                                         own, tolerance=0.15)
            port_label_pos[inst.name] = (px, py, an, bl, pn["name"])
            obstacles.append(box)
            continue
        gap = 1.0
        # where the library author put the designator, first. The
        # planned text includes the badges the label will carry
        label_text = inst.name + (" xxx" if inst.name in tints else "")
        cands = []
        spot = _spot_candidate(sym, inst.m, cx_, cy_, label_text, FONT_MM)
        if spot:
            cands.append(spot)
        cands += [
            (r[2] + gap, cy_ + 0.35 * FONT_MM, "start", "alphabetic"),
            (r[0] - gap, cy_ + 0.35 * FONT_MM, "end", "alphabetic"),
            (cx_, r[1] - gap, "middle", "alphabetic"),
            (cx_, r[3] + gap, "middle", "hanging"),
        ]
        # the symbol's own canvas is no obstacle to its own label: the
        # author's designator spot lies inside it by design
        own = [o for o in obstacles if o != r]
        px, py, an, bl, box = _place(label_text, FONT_MM, cands, own)
        inst_label_pos[inst.name] = (px, py, an, bl)
        obstacles.append(box)

    # every instance sits in <g id="inst:NAME"> so a viewer can hit-test
    # and highlight it (QSvgRenderer.boundsOnElement); class says what
    # a click may do: "block" descends, "device" cross-probes.
    # `bodies` carries each instance's ARTWORK rect in page mm -- the
    # geometric truth the viewer prefers over boundsOnElement, whose
    # stroke-padded union exceeded the drawn symbol (field report)
    instances: dict = {}
    bodies: dict = {}

    # ---- subcircuit blocks: the harvested symbol box + pin stubs
    for inst in blocks:
        q0, q1 = _term_box(inst)
        instances[inst.name] = inst.cell
        bodies[inst.name] = (q0[0], q0[1], q1[0], q1[1])
        parts.append(f'<g id="{_xml_id(inst.name)}" class="block">')
        parts.append(
            f'<rect id="{_body_id(inst.name)}" '
            f'x="{q0[0]:.2f}" y="{q0[1]:.2f}" '
            f'width="{q1[0] - q0[0]:.2f}" height="{q1[1] - q0[1]:.2f}" '
            f'fill="white" stroke="#000000" stroke-width="{WIRE_STROKE}"/>')
        if inst.name in deactivated:
            parts.append(_deact_cross(q0[0], q0[1], q1[0], q1[1]))
        cxb = (q0[0] + q1[0]) / 2.0
        badge = (f'<tspan fill="{EMPH}" font-weight="bold"> ?</tspan>'
                 if inst.cell in proposed_cells else "")
        if tex_labels:
            from .mathtext import label_width, tex_name
            txt = f"{inst.name} ({inst.cell})"
            parts.append(label(cxb, q0[1] - 1.2, txt, anchor="middle"))
            if badge:
                w = label_width(tex_name(txt), FONT_MM)
                parts.append(badge_text(cxb + w / 2.0, q0[1] - 1.2, badge))
        else:
            parts.append(f'<text x="{cxb:.2f}" y="{q0[1] - 1.2:.2f}" '
                         f'text-anchor="middle">{inst.name} '
                         f'({inst.cell}){badge}</text>')
        for pname, pxy in inst.terms.items():
            px, py = anchor_world[inst.name][2][pname]
            parts.append(f'<circle cx="{px:.3f}" cy="{py:.3f}" r="0.45" '
                         f'fill="white" stroke="#000000" '
                         f'stroke-width="{WIRE_STROKE}"/>')
            # pin name just inside the box edge nearest the pin
            inside_x = px + (1.0 if px < cxb else -1.0)
            anchor = "start" if px < cxb else "end"
            parts.append(label(inside_x, py + 1.0, pname, 1.8, anchor))
        parts.append("</g>")

    # ---- symbols
    for inst, sym, spec in placed:
        kind = ("block" if inst.cell in block_symbols else
                "glyph" if inst.cell in CELL_GLYPHS else "device")
        instances[inst.name] = inst.cell
        tag = styles.get(inst.name)
        extra = (' opacity="0.25"' if tag == STYLE_REMOVED else
                 ' opacity="0.55"' if tag == STYLE_PASSIVE else "")
        parts.append(f'<g id="{_xml_id(inst.name)}" class="{kind}"{extra}>')
        # in the formula: bold; folded into a number: gray. Blue stays
        # the harvested net labels' color, nothing else
        lab_attr = (' font-weight="bold"' if tag == STYLE_SYMBOLIC else
                    ' fill="#8a8a8a"' if tag == STYLE_NUMERIC else "")
        badge = (f'<tspan fill="{EMPH}" font-size="2.4"> ĝ</tspan>'
                 if tag == STYLE_LUMPED else "")
        if inst.name in tints and tints[inst.name][1]:
            badge += (f'<tspan fill="{EMPH}" font-size="2.2"> '
                      f'{tints[inst.name][1]}</tspan>')
        if sym is None:
            q0 = mm((inst.bbox[0], inst.bbox[3]))
            q1 = mm((inst.bbox[2], inst.bbox[1]))
            bodies[inst.name] = (q0[0], q0[1], q1[0], q1[1])
            parts.append(
                f'<rect id="{_body_id(inst.name)}" '
                f'x="{q0[0]:.2f}" y="{q0[1]:.2f}" '
                f'width="{q1[0] - q0[0]:.2f}" '
                f'height="{q1[1] - q0[1]:.2f}" fill="none" '
                f'stroke="#a80000" stroke-dasharray="1,1" '
                f'stroke-width="{WIRE_STROKE}"/>'
                f'<text x="{q0[0]:.2f}" y="{q0[1] - 1:.2f}" '
                f'fill="#a80000">{inst.name} ({inst.cell})</text>')
            parts.append("</g>")
            continue
        cx, cy, _ = anchor_world[inst.name]
        if inst.name in tints:
            # the match-group plate: a tinted rounded rectangle behind
            # the symbol, inset so it reads as a highlight, not a box
            # -- and UNDER the wires (spliced at plate_slot)
            fill, _badge = tints[inst.name]
            pr = _symbol_rect(cx, cy, sym, inst.m)
            plates.append(
                f'<rect x="{pr[0] + 0.6:.2f}" y="{pr[1] + 0.6:.2f}" '
                f'width="{pr[2] - pr[0] - 1.2:.2f}" '
                f'height="{pr[3] - pr[1] - 1.2:.2f}" rx="1.2" '
                f'fill="{fill}" stroke="none"/>')
        # a plain scaled <g>, not a nested <svg>: QtSvg (SVG Tiny)
        # ignores nested svg elements entirely — measured, not assumed
        vb = [float(v) for v in sym.viewbox.split()]
        sx = sym.w / (vb[2] or 1.0)
        sy = sym.h / (vb[3] or 1.0)
        inner = sym.inner
        cw = contrib.get(inst.name) if contrib else None
        if cw is not None:
            # a contributor's BODY wears the accent, weight-fading
            # toward the installed black -- the singularity story
            # drawn in the strokes themselves, no box
            col = _contrib_color(cw, contrib_base)
            inner = inner.replace("#000000", col).replace("#000", col)
        elif port_base is not None and inst.cell in PORT_CELLS:
            # a dark display hands Virtuoso's pin red for the port
            # glyphs (the canonical black-on-white export never does)
            inner = (inner.replace("#000000", port_base)
                     .replace("#000", port_base))
        parts.append(
            f'<g id="{_body_id(inst.name)}" '
            f'transform="translate({cx:.3f}, {cy:.3f}) '
            f'{_orient_svg(inst.m)} '
            f'translate({-sym.w / 2.0:.3f}, {-sym.h / 2.0:.3f}) '
            f'scale({sx:.6f}, {sy:.6f}) '
            f'translate({-vb[0]:.3f}, {-vb[1]:.3f})">'
            f'{inner}</g>')
        # the body rect: the ARTWORK's measured extent when a
        # measurer is handed in (rendered-alpha bbox, cached per
        # symbol file -- the canvas frame carries empty margin on
        # symbols like the isource circle), else the canvas frame.
        # A measured rect gets BODY_PAD of air: a box on the strokes
        # reads cramped, and the X inscribes the same padded square
        loc = body_measure(sym, inner) if body_measure else None
        if loc:
            tight = _oriented_local_rect(cx, cy, sym, inst.m, loc)
            rd = (tight[0] - BODY_PAD, tight[1] - BODY_PAD,
                  tight[2] + BODY_PAD, tight[3] + BODY_PAD)
        else:
            tight = rd = _symbol_rect(cx, cy, sym, inst.m)
        bodies[inst.name] = tuple(rd)
        if inst.name in deactivated:
            # the X marks the ARTWORK: it inscribes the tight rect,
            # not the padded selection box (field request)
            parts.append(_deact_cross(tight[0], tight[1],
                                      tight[2], tight[3]))
        if inst.cell in PORT_CELLS:
            # a port shows its PIN NAME on its free end, never the
            # Cadence instance name; the name wears the port ink
            if inst.name in port_label_pos:
                px, py, an, bl, txt = port_label_pos[inst.name]
                parts.append(label(px, py, txt, PORT_FONT_MM, an, bl,
                                   fill=port_base or "#000000"))
        elif inst.cell in UNLABELLED_CELLS:
            pass                       # rails/no-connects: anonymous
        elif inst.cell in block_symbols:
            # under the body, centered: the sides carry the bridges
            ext = max(sym.w, sym.h) / 2.0
            bold = tag == STYLE_SYMBOLIC
            fill = "#8a8a8a" if tag == STYLE_NUMERIC else "#000000"
            if tex_labels:
                from .mathtext import label_width, tex_name
                ttx = tex_for(inst.name) or tex_name(inst.name, bold)
                parts.append(label(cx, cy + ext + 1.5, inst.name,
                                   anchor="middle", baseline="hanging",
                                   fill=fill, bold=bold, tex_text=ttx))
                if badge:
                    w = label_width(ttx, FONT_MM)
                    parts.append(badge_text(cx + w / 2.0,
                                            cy + ext + 1.5 + 2.4, badge))
            else:
                parts.append(f'<text x="{cx:.2f}" y="{cy + ext + 1.5:.2f}" '
                             f'text-anchor="middle" '
                             f'dominant-baseline="hanging"{lab_attr}>'
                             f'{inst.name}{badge}</text>')
            parts.append(label(cx, cy + ext + 5.2, f"({inst.cell})", 1.8,
                               "middle", "hanging"))
        else:
            px, py, an, bl = inst_label_pos.get(
                inst.name, (cx + sym.w / 2 + 1, cy, "start", "alphabetic"))
            if tex_labels:
                from .mathtext import label_width, render, tex_name
                bold = tag == STYLE_SYMBOLIC
                fill = "#8a8a8a" if tag == STYLE_NUMERIC else "#000000"
                ttx = tex_for(inst.name) or tex_name(inst.name, bold)
                parts.append(label(px, py, inst.name, anchor=an,
                                   baseline=bl, fill=fill, bold=bold,
                                   tex_text=ttx))
                if badge:
                    w = label_width(ttx, FONT_MM)
                    bx = {"start": px + w, "middle": px + w / 2.0,
                          "end": px}.get(an, px + w)
                    by = py + (render(ttx, FONT_MM)[2] if bl == "hanging"
                               else 0.0)
                    parts.append(badge_text(bx, by, badge))
            else:
                parts.append(f'<text x="{px:.2f}" y="{py:.2f}" '
                             f'text-anchor="{an}" dominant-baseline="{bl}"'
                             f'{lab_attr}>{inst.name}{badge}</text>')
        parts.append("</g>")
    compose.last_instances = instances
    compose.last_bodies = bodies

    for x, y, txt, an, bl, rot in net_label_pos:
        parts.append(label(x, y, txt, anchor=an, baseline=bl,
                           fill="#1a466b", rotate=rot))

    if overlay:
        parts.append(_overlay_svg(flat, blocks, placed, raw_wires, mm_raw))

    # ---- proposed compensation branches: the selected suggestion
    # drawn ONTO the page it would change, in the Bode preview's
    # accent, so a table row has a physical place, not just a curve
    compose.last_proposed_drawn = 0
    if proposed:
        svg_p, n_drawn = _proposed_svg(proposed, net_wires, lib,
                                       sym_rects + label_boxes)
        if n_drawn:
            parts.append(svg_p)
        compose.last_proposed_drawn = n_drawn

    parts[plate_slot:plate_slot] = plates
    parts.append("</svg>")
    return "\n".join(parts)


#: the Bode preview's accent (Okabe-Ito vermilion): the proposed branch
#: on the schematic and the preview curve on the plot are ONE object
PROPOSED_COLOR = "#D55E00"


def _deact_cross(x0: float, y0: float, x1: float, y1: float) -> str:
    """Virtuoso's deactivation mark: a red X drawn as the diagonals
    of the largest CENTERED SQUARE inside the body rect (a rect's own
    diagonals splay on tall or wide bodies). Red in both themes (the
    dark re-ink leaves non-black strokes alone)."""
    half = min(x1 - x0, y1 - y0) / 2.0
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    x0, x1 = cx - half, cx + half
    y0, y1 = cy - half, cy + half
    return (f'<path d="M {x0:.2f} {y0:.2f} L {x1:.2f} {y1:.2f} '
            f'M {x0:.2f} {y1:.2f} L {x1:.2f} {y0:.2f}" '
            f'stroke="#d00000" stroke-width="0.5" fill="none" '
            f'stroke-linecap="round"/>')


def _contrib_color(w: float, base: str = "#000000") -> str:
    """ink -> accent, weight-interpolated: the strongest contributor
    wears the full accent, the weakest stays near the installed ink
    (black on the light paper; the display hands its own ink for a
    dark ground, where a near-black shade would vanish)."""
    t = 0.3 + 0.7 * max(0.0, min(1.0, float(w)))
    b = tuple(int(base[i:i + 2], 16) for i in (1, 3, 5))
    acc = (0xD5, 0x5E, 0x00)
    return "#%02X%02X%02X" % tuple(
        int(bc + t * (ac - bc)) for bc, ac in zip(b, acc))


def _proposed_body(sym, a1, a2, p1, theta) -> str:
    """The library symbol's own artwork as the proposed-branch body:
    anchor `a1` lands on world point p1 with the a1->a2 axis rotated
    onto `theta`, natural size, recolored to the accent (black is the
    installed circuit's color; the proposal must not wear it)."""
    la1 = sym.local(a1)
    la2 = sym.local(a2) if a2 is not None else (la1[0], la1[1] + 1.0)
    phi = math.degrees(math.atan2(la2[1] - la1[1], la2[0] - la1[0]))
    delta = math.degrees(theta) - phi
    vb = [float(v) for v in sym.viewbox.split()]
    sx = sym.w / (vb[2] or 1.0)
    sy = sym.h / (vb[3] or 1.0)
    inner = sym.inner.replace("#000000", PROPOSED_COLOR).replace(
        "#000", PROPOSED_COLOR)
    return (f'<g transform="translate({p1[0]:.3f}, {p1[1]:.3f}) '
            f'rotate({delta:.2f}) '
            f'translate({-la1[0]:.3f}, {-la1[1]:.3f}) '
            f'scale({sx:.6f}, {sy:.6f}) '
            f'translate({-vb[0]:.3f}, {-vb[1]:.3f})">{inner}</g>')


def _seg_block(p, q, rects) -> float:
    """Length of the axis-aligned segment p-q lying INSIDE any of the
    rectangles -- the routing score's currency (diagonals score 0;
    the fallback diagonal is already the last resort)."""
    total = 0.0
    if abs(p[1] - q[1]) < 1e-6:                   # horizontal
        y, x0, x1 = p[1], min(p[0], q[0]), max(p[0], q[0])
        for rx0, ry0, rx1, ry1 in rects:
            if ry0 <= y <= ry1:
                total += max(0.0, min(x1, rx1) - max(x0, rx0))
    elif abs(p[0] - q[0]) < 1e-6:                 # vertical
        x, y0, y1 = p[0], min(p[1], q[1]), max(p[1], q[1])
        for rx0, ry0, rx1, ry1 in rects:
            if rx0 <= x <= rx1:
                total += max(0.0, min(y1, ry1) - max(y0, ry0))
    return total


def _proposed_svg(branches, net_wires, lib, rects=()) -> tuple[str, int]:
    """Proposed compensation branches drawn between their nets'
    existing wires: dashed accent leads joining the LIBRARY's own
    capacitor / resistor / ground artwork (recolored to the accent),
    the sized value as a label. `branches`: (net_a, net_b_or_None,
    label, with_r) with PAGE-LOCAL net names; entries whose nets have
    no wires on this page are skipped (the count of DRAWN branches is
    returned beside the markup)."""
    C = PROPOSED_COLOR
    try:
        cap = lib.load("cap.svg")
        res = lib.load("res.svg") if any(b[3] for b in branches) else None
        gnd = (lib.load("gnd.svg")
               if any(b[1] is None for b in branches) else None)
    except Exception:                             # noqa: BLE001
        return "", 0

    def points_of(net):
        """Candidate tap points with their segment's unit direction."""
        pts = []
        for poly in net_wires.get(net, []):
            for i in range(len(poly) - 1):
                (x1, y1), (x2, y2) = poly[i], poly[i + 1]
                ln = math.hypot(x2 - x1, y2 - y1)
                dr = ((x2 - x1) / ln, (y2 - y1) / ln) if ln else (1.0, 0.0)
                pts.append((((x1 + x2) / 2.0, (y1 + y2) / 2.0), dr))
                pts.append(((x1, y1), dr))
                pts.append(((x2, y2), dr))
        return pts

    def longest_mid(net):
        """(midpoint, unit direction) of the net's longest segment."""
        best, bdir, blen = None, (1.0, 0.0), -1.0
        for poly in net_wires.get(net, []):
            for i in range(len(poly) - 1):
                (x1, y1), (x2, y2) = poly[i], poly[i + 1]
                ln = math.hypot(x2 - x1, y2 - y1)
                if ln > blen:
                    blen = ln
                    best = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
                    bdir = ((x2 - x1) / ln, (y2 - y1) / ln) if ln else bdir
        return best, bdir

    out = [f'<g id="proposed-branches" fill="none" stroke="{C}" '
           f'stroke-width="0.22" stroke-linecap="round">']
    n_drawn = 0
    cap_pitch = math.dist(cap.local("1"), cap.local("2"))
    res_pitch = math.dist(res.local("1"), res.local("2")) if res else 0.0
    def dash(p, q):
        out.append(f'<path d="M {p[0]:.2f} {p[1]:.2f} '
                   f'L {q[0]:.2f} {q[1]:.2f}" '
                   f'stroke-dasharray="1.1,0.7"/>')

    def dot(p):
        out.append(f'<circle cx="{p[0]:.2f}" cy="{p[1]:.2f}" r="0.55" '
                   f'fill="{C}" stroke="none"/>')

    for net_a, net_b, text, with_r in branches:
        body = cap_pitch + (res_pitch if with_r else 0.0)
        if net_b is None:
            pa, sdir = longest_mid(net_a)
            if pa is None:
                continue
            # a branch to ground JOGS off the tapped wire first -- one
            # dropped straight down a vertical wire drew ON it -- then
            # the body runs down in clear space, on the side where the
            # column stays OUT of the symbols, label on the free side
            if abs(sdir[1]) > abs(sdir[0]):        # vertical wire
                options = [(pa[0] + 6.0, pa[1]), (pa[0] - 6.0, pa[1])]
            else:                                   # horizontal: below
                options = [(pa[0], pa[1] + 3.0)]

            def col_score(pc):
                bot = (pc[0], pc[1] + body + 5.0)
                return (_seg_block(pa, pc, rects)
                        + 2.0 * _seg_block(pc, bot, rects))

            pc = min(options, key=col_score)
            dash(pa, pc)
            dot(pa)
            t = pc
            if with_r:
                out.append(_proposed_body(res, "1", "2", t, math.pi / 2))
                t = (t[0], t[1] + res_pitch)
            out.append(_proposed_body(cap, "1", "2", t, math.pi / 2))
            t = (t[0], t[1] + cap_pitch)
            out.append(_proposed_body(gnd, "1", None, t, math.pi / 2))
            side = -1.0 if pc[0] < pa[0] else 1.0
            out.append(f'<text x="{pc[0] + side * 3.0:.2f}" '
                       f'y="{pc[1] + body / 2.0:.2f}" font-size="2.4" '
                       f'fill="{C}" stroke="none" text-anchor='
                       f'"{"start" if side > 0 else "end"}" '
                       f'dominant-baseline="middle">{text}</text>')
            n_drawn += 1
            continue
        ca, cb = points_of(net_a), points_of(net_b)
        if not ca or not cb:
            continue
        (pa, da), (pb, db) = min(
            ((a, b) for a in ca for b in cb),
            key=lambda q: math.hypot(q[0][0][0] - q[1][0][0],
                                     q[0][0][1] - q[1][0][1]))

        def jog(p, sd, sign):
            # perpendicular off the tapped segment
            return (p[0] - 6.0 * sd[1] * sign, p[1] + 6.0 * sd[0] * sign)

        # Manhattan routes between the jogs (both sides of each wire,
        # both L corners; nearly-collinear jogs take the straight
        # run), each carrying the body on its LONGEST leg. Scored by
        # how much of the route -- the body leg doubly -- lies inside
        # symbol rectangles: the first cut ran the branch straight
        # through the OTA block. Ties break on total length. The
        # direct diagonal is the last resort of a cramped page.
        cands = []
        for sa in (1.0, -1.0):
            ja = jog(pa, da, sa)
            for sb in (1.0, -1.0):
                jb = jog(pb, db, sb)
                if abs(ja[0] - jb[0]) < 1.0 or abs(ja[1] - jb[1]) < 1.0:
                    routes = [([pa, ja, jb, pb], (ja, jb))]
                else:
                    routes = []
                    for m in ((jb[0], ja[1]), (ja[0], jb[1])):
                        legs = ((ja, m), (m, jb))
                        routes.append(
                            ([pa, ja, m, jb, pb],
                             max(legs,
                                 key=lambda s: math.dist(s[0], s[1]))))
                if rects and abs(ja[0] - jb[0]) >= body + 1.0:
                    # Z detours over/under every symbol: when the L
                    # corners are walled in, going around beats going
                    # through -- the length term keeps them honest
                    for Y in (min(r[1] for r in rects) - 4.0,
                              max(r[3] for r in rects) + 4.0):
                        routes.append(
                            ([pa, ja, (ja[0], Y), (jb[0], Y), jb, pb],
                             ((ja[0], Y), (jb[0], Y))))
                if rects and abs(ja[1] - jb[1]) >= body + 1.0:
                    for X in (min(r[0] for r in rects) - 4.0,
                              max(r[2] for r in rects) + 4.0):
                        routes.append(
                            ([pa, ja, (X, ja[1]), (X, jb[1]), jb, pb],
                             ((X, ja[1]), (X, jb[1]))))
                for path, leg in routes:
                    if math.dist(leg[0], leg[1]) < body + 1.0:
                        continue
                    score = sum(_seg_block(p, q, rects)
                                for p, q in zip(path, path[1:]))
                    score += 2.0 * _seg_block(leg[0], leg[1], rects)
                    length = sum(math.dist(p, q)
                                 for p, q in zip(path, path[1:]))
                    cands.append((round(score, 1), length, path, leg))
        if cands:
            _s, _l, path, (s0, s1) = min(cands,
                                         key=lambda c: (c[0], c[1]))
        else:
            path, (s0, s1) = [pa, pb], (pa, pb)   # the direct diagonal
        dx, dy = s1[0] - s0[0], s1[1] - s0[1]
        d = math.hypot(dx, dy) or 1.0
        ux, uy = dx / d, dy / d              # along the body's run
        vx, vy = -uy, ux                     # across it
        theta = math.atan2(uy, ux)
        cx_, cy_ = (s0[0] + s1[0]) / 2.0, (s0[1] + s1[1]) / 2.0

        def at(t, w=0.0):
            return (cx_ + t * ux + w * vx, cy_ + t * uy + w * vy)

        # the LIBRARY's own artwork, centered on the body's run: R
        # then C for a series-RC, value pitch straight from the anchors
        t0 = -body / 2.0
        if with_r:
            out.append(_proposed_body(res, "1", "2", at(t0), theta))
            t0 += res_pitch
        out.append(_proposed_body(cap, "1", "2", at(t0), theta))
        # dashed leads: the route with the body's span cut out of its run
        b0, b1 = at(-body / 2.0), at(body / 2.0)
        pts = list(path)
        i0 = pts.index(s0)
        pts[i0 + 1:i0 + 1] = [b0, b1]
        for p, q in zip(pts, pts[1:]):
            if (p, q) == (b0, b1):
                continue                     # the body sits here
            if math.dist(p, q) > 0.05:
                dash(p, q)
        dot(pa)
        dot(pb)
        lx, ly = at(0.0, 3.0)
        # grow the text AWAY from the body: sideways-displaced labels
        # anchor at their near edge
        anchor = ("middle" if abs(vx) <= abs(vy)
                  else "start" if vx > 0 else "end")
        out.append(f'<text x="{lx:.2f}" y="{ly:.2f}" font-size="2.4" '
                   f'fill="{C}" stroke="none" text-anchor="{anchor}" '
                   f'dominant-baseline="middle">{text}</text>')
        n_drawn += 1
    out.append("</g>")
    return "\n".join(out), n_drawn


def _overlay_svg(flat, blocks, placed, raw_wires, mm) -> str:
    """The harvest as recorded, drawn over the composition."""
    V, B = "#D55E00", "#0072B2"
    out = ['<g id="hints-overlay" opacity="0.85">']
    for pts in raw_wires:                      # unsnapped polylines
        d = "M " + " L ".join(f"{x:.3f} {y:.3f}" for x, y in pts)
        out.append(f'<path d="{d}" fill="none" stroke="{B}" '
                   f'stroke-width="0.3"/>')
        for x, y in (pts[0], pts[-1]):
            out.append(f'<rect x="{x - 0.4:.3f}" y="{y - 0.4:.3f}" '
                       f'width="0.8" height="0.8" fill="{B}"/>')
    for inst in [p_[0] for p_ in placed] + list(blocks):
        # the box is the TERMINAL extent (+1 mm), not Virtuoso's
        # label-inclusive bBox -- the faithful footprint
        ps = [mm(v) for v in inst.terms.values()] or [mm(inst.xy)]
        xs_ = [q[0] for q in ps]
        ys_ = [q[1] for q in ps]
        q0 = (min(xs_) - 1.0, min(ys_) - 1.0)
        q1 = (max(xs_) + 1.0, max(ys_) + 1.0)
        out.append(f'<rect x="{q0[0]:.2f}" y="{q0[1]:.2f}" '
                   f'width="{q1[0] - q0[0]:.2f}" '
                   f'height="{q1[1] - q0[1]:.2f}" fill="none" '
                   f'stroke="{V}" stroke-width="0.3" '
                   f'stroke-dasharray="1.2,0.8"/>')
        ox, oy = mm(inst.xy)                   # the instance origin
        out.append(f'<path d="M {ox - 1.2:.2f} {oy:.2f} H {ox + 1.2:.2f} '
                   f'M {ox:.2f} {oy - 1.2:.2f} V {oy + 1.2:.2f}" '
                   f'stroke="{V}" stroke-width="0.35"/>')
        out.append(f'<text x="{q0[0]:.2f}" y="{q0[1] - 0.4:.2f}" '
                   f'font-size="1.8" fill="{V}">{inst.name} '
                   f'{inst.orient}</text>')
        for tname, txy in inst.terms.items():  # harvested pin points
            tx, ty = mm(txy)
            out.append(f'<circle cx="{tx:.3f}" cy="{ty:.3f}" r="0.5" '
                       f'fill="none" stroke="{V}" stroke-width="0.35"/>')
            out.append(f'<text x="{tx + 0.7:.2f}" y="{ty - 0.5:.2f}" '
                       f'font-size="1.6" fill="{V}">{tname}</text>')
    out.append("</g>")
    return "\n".join(out)
