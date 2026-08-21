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
from .symmap import (CELL_SYMBOLS, SymbolSpec, anchor_for, resolve,
                     resolve_cell)

CELL_GLYPHS = frozenset(CELL_SYMBOLS)

#: the library's terminal-stub style, verbatim but in page units: the
#: stubs carry stroke-width 0.566929 in the symbols' px-scaled frame,
#: which is 0.15 mm on the 1 mm grid (rule 17); drawing wires at
#: 0.567 MM made them four times heavier than the stubs they meet
WIRE_STROKE = 0.15               # mm
WIRE_STYLE = (f'fill="none" stroke="#000000" stroke-width="{WIRE_STROKE}" '
              f'stroke-linecap="round" stroke-linejoin="miter"')
DOT_R = 0.53                     # mm (~2 px)

#: analysis-state styling on the SAME layout (the simplified view):
#: instance tags and net tags the GUI derives from the session
STYLE_REMOVED = "removed"        # dead source / deleted: faded
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


def library_root(explicit=None) -> Path:
    root = explicit or os.environ.get("CIN_SYMLIB")
    if not root:
        raise ComposeError(
            "no symbol library: set CIN_SYMLIB to the library-rescaled "
            "directory (the library is personal and does not ship with "
            "CircuitInsight)")
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
                     origin=origin)
        self._cache[filename] = sym
        return sym


#: Cadence port glyph cells: their label is the pin name
PORT_CELLS = frozenset({"ipin", "opin", "iopin"})


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


FONT_MM = 3.0                    # instance / net label size
FONT_W = 0.56                    # mean glyph advance, em fraction


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


def _place(text, size, candidates, obstacles):
    """The candidate (x, y, anchor, baseline) covering the least
    obstacle area; the first candidate wins ties (it is the natural
    spot). Returns (x, y, anchor, baseline, box)."""
    best = None
    for i, (x, y, an, bl) in enumerate(candidates):
        box = _text_box(x, y, text, size, an, bl)
        cost = sum(_overlap(box, o) for o in obstacles)
        key = (round(cost, 3), i)
        if best is None or key < best[0]:
            best = (key, (x, y, an, bl, box))
        if cost == 0:
            break
    return best[1]


def _symbol_rect(cx, cy, sym, m):
    """World-axis extent of a placed symbol canvas under orientation m."""
    a, b = m[0]
    c, d = m[1]
    ew = abs(a) * sym.w + abs(b) * sym.h
    eh = abs(c) * sym.w + abs(d) * sym.h
    return (cx - ew / 2, cy - eh / 2, cx + ew / 2, cy + eh / 2)


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
            styles: dict | None = None, net_styles: dict | None = None
            ) -> str:
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
    """
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
             f'font-size="3.0">']

    # ---- confirmed block symbols become placed symbols. The harvested
    # orient placed the PDK symbol; the library symbol may order its
    # anchors differently, so the orientation is chosen by geometry:
    # the one of the eight with the smallest registration residual,
    # the harvested orient winning ties.
    stubbed = []                      # (inst, sym, spec, stubs)
    for inst in list(blocks):
        bs = block_symbols.get(inst.cell)
        if not bs or not bs.confirmed:
            continue
        sym = lib.load(bs.symbol)
        spec = SymbolSpec(bs.symbol, dict(bs.pins))
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
            k = min(max(min(ks), 1.0), BLOCK_MAX_SCALE)
            sym = _scaled(sym, k)
        best = None
        for name, m in ORIENTS.items():
            off = _offset_fn(sym, m)
            _cx, _cy, res = _register([(h, off(an)) for h, an in pairs],
                                      mode=False)
            key = (round(res, 6), 0 if m == inst.m else 1)
            if best is None or key < best[0]:
                best = (key, m)
        blocks.remove(inst)
        inst2 = replace(inst, m=best[1])
        placed.append((inst2, sym, spec))
        stubbed.append((inst2, sym, spec,
                        {r: bs.stubs.get(r, "auto") for r in inst.terms
                         if anchor_for(spec, r) not in sym.anchors}))

    block_names = {s[0].name for s in stubbed}

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
            if math.dist(pin, an) < 1e-6:
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
            parts.append(f'<text x="{tx:.2f}" y="{ty:.2f}" '
                         f'font-size="2.2" text-anchor="{ta}" '
                         f'dominant-baseline="central">{r}</text>')
    for pts in bridges:
        d = "M " + " L ".join(f"{x:.3f} {y:.3f}" for x, y in pts)
        parts.append(f'<path d="{d}" {WIRE_STYLE}/>')

    raw_wires = [[mm_raw(p) for p in w.points] for w in flat.wires]
    earthed: set = set()
    segs: list = []                   # (net, (x0, y0), (x1, y1)) lanes
    wire_ends: list = []              # (net, point)
    for w in flat.wires:
        pts = [mm(p) for p in w.points]
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
        cands = [(x, y, "start", "alphabetic")]
        cands += [(x, y + dy, "start", "alphabetic")
                  for dy in (-2.0, 2.0, -4.0, 4.0)]
        # the designer's spot is kept unless it sits on a symbol body
        # or an earlier label; wires alone do not move it
        px, py, an, bl, box = _place(txt, FONT_MM, cands,
                                     sym_rects + label_boxes)
        net_label_pos.append((px, py, txt))
        label_boxes.append(box)
        obstacles.append(box)

    inst_label_pos: dict = {}
    for inst, sym, spec in placed:
        if sym is None or inst.name not in anchor_world:
            continue
        if inst.cell in PORT_CELLS or inst.cell in block_symbols:
            continue
        cx_, cy_, _ = anchor_world[inst.name]
        r = _symbol_rect(cx_, cy_, sym, inst.m)
        gap = 1.0
        cands = [
            (r[2] + gap, cy_ + 0.35 * FONT_MM, "start", "alphabetic"),
            (r[0] - gap, cy_ + 0.35 * FONT_MM, "end", "alphabetic"),
            (cx_, r[1] - gap, "middle", "alphabetic"),
            (cx_, r[3] + gap, "middle", "hanging"),
        ]
        px, py, an, bl, box = _place(inst.name, FONT_MM, cands, obstacles)
        inst_label_pos[inst.name] = (px, py, an, bl)
        obstacles.append(box)

    # every instance sits in <g id="inst:NAME"> so a viewer can hit-test
    # and highlight it (QSvgRenderer.boundsOnElement); class says what
    # a click may do: "block" descends, "device" cross-probes
    instances: dict = {}

    # ---- subcircuit blocks: the harvested symbol box + pin stubs
    for inst in blocks:
        q0, q1 = _term_box(inst)
        instances[inst.name] = inst.cell
        parts.append(f'<g id="{_xml_id(inst.name)}" class="block">')
        parts.append(
            f'<rect x="{q0[0]:.2f}" y="{q0[1]:.2f}" '
            f'width="{q1[0] - q0[0]:.2f}" height="{q1[1] - q0[1]:.2f}" '
            f'fill="white" stroke="#000000" stroke-width="{WIRE_STROKE}"/>')
        cxb = (q0[0] + q1[0]) / 2.0
        parts.append(f'<text x="{cxb:.2f}" y="{q0[1] - 1.2:.2f}" '
                     f'text-anchor="middle">{inst.name} '
                     f'({inst.cell})</text>')
        for pname, pxy in inst.terms.items():
            px, py = anchor_world[inst.name][2][pname]
            parts.append(f'<circle cx="{px:.3f}" cy="{py:.3f}" r="0.45" '
                         f'fill="white" stroke="#000000" '
                         f'stroke-width="{WIRE_STROKE}"/>')
            # pin name just inside the box edge nearest the pin
            inside_x = px + (1.0 if px < cxb else -1.0)
            anchor = "start" if px < cxb else "end"
            parts.append(f'<text x="{inside_x:.2f}" y="{py + 1.0:.2f}" '
                         f'font-size="2.2" text-anchor="{anchor}">'
                         f'{pname}</text>')
        parts.append("</g>")

    # ---- symbols
    for inst, sym, spec in placed:
        kind = ("block" if inst.cell in block_symbols else
                "glyph" if inst.cell in CELL_GLYPHS else "device")
        instances[inst.name] = inst.cell
        tag = styles.get(inst.name)
        extra = ' opacity="0.25"' if tag == STYLE_REMOVED else ""
        parts.append(f'<g id="{_xml_id(inst.name)}" class="{kind}"{extra}>')
        # in the formula: bold; folded into a number: gray. Blue stays
        # the harvested net labels' color, nothing else
        lab_attr = (' font-weight="bold"' if tag == STYLE_SYMBOLIC else
                    ' fill="#8a8a8a"' if tag == STYLE_NUMERIC else "")
        badge = (f'<tspan fill="{EMPH}" font-size="2.4"> ĝ</tspan>'
                 if tag == STYLE_LUMPED else "")
        if sym is None:
            q0 = mm((inst.bbox[0], inst.bbox[3]))
            q1 = mm((inst.bbox[2], inst.bbox[1]))
            parts.append(
                f'<rect x="{q0[0]:.2f}" y="{q0[1]:.2f}" '
                f'width="{q1[0] - q0[0]:.2f}" '
                f'height="{q1[1] - q0[1]:.2f}" fill="none" '
                f'stroke="#a80000" stroke-dasharray="1,1" '
                f'stroke-width="{WIRE_STROKE}"/>'
                f'<text x="{q0[0]:.2f}" y="{q0[1] - 1:.2f}" '
                f'fill="#a80000">{inst.name} ({inst.cell})</text>')
            parts.append("</g>")
            continue
        cx, cy, _ = anchor_world[inst.name]
        # a plain scaled <g>, not a nested <svg>: QtSvg (SVG Tiny)
        # ignores nested svg elements entirely — measured, not assumed
        vb = [float(v) for v in sym.viewbox.split()]
        sx = sym.w / (vb[2] or 1.0)
        sy = sym.h / (vb[3] or 1.0)
        parts.append(
            f'<g transform="translate({cx:.3f}, {cy:.3f}) '
            f'{_orient_svg(inst.m)} '
            f'translate({-sym.w / 2.0:.3f}, {-sym.h / 2.0:.3f}) '
            f'scale({sx:.6f}, {sy:.6f}) '
            f'translate({-vb[0]:.3f}, {-vb[1]:.3f})">'
            f'{sym.inner}</g>')
        if inst.cell in PORT_CELLS:
            # a port shows its PIN NAME where the source draws it (the
            # harvested pin figure), never the Cadence instance name
            pn = _pin_named(inst, flat.pins)
            if pn:
                x, y = mm_raw(pn["xy"])
                parts.append(f'<text x="{x:.2f}" y="{y:.2f}" '
                             f'text-anchor="middle" '
                             f'dominant-baseline="central">'
                             f'{pn["name"]}</text>')
        elif inst.cell in block_symbols:
            # under the body, centered: the sides carry the bridges
            ext = max(sym.w, sym.h) / 2.0
            parts.append(f'<text x="{cx:.2f}" y="{cy + ext + 1.5:.2f}" '
                         f'text-anchor="middle" '
                         f'dominant-baseline="hanging"{lab_attr}>'
                         f'{inst.name}{badge}</text>')
            parts.append(f'<text x="{cx:.2f}" y="{cy + ext + 5.2:.2f}" '
                         f'text-anchor="middle" '
                         f'dominant-baseline="hanging" font-size="2.2">'
                         f'({inst.cell})</text>')
        else:
            px, py, an, bl = inst_label_pos.get(
                inst.name, (cx + sym.w / 2 + 1, cy, "start", "alphabetic"))
            parts.append(f'<text x="{px:.2f}" y="{py:.2f}" '
                         f'text-anchor="{an}" dominant-baseline="{bl}"'
                         f'{lab_attr}>{inst.name}{badge}</text>')
        parts.append("</g>")
    compose.last_instances = instances

    for x, y, txt in net_label_pos:
        parts.append(f'<text x="{x:.2f}" y="{y:.2f}" '
                     f'fill="#1a466b">{txt}</text>')

    if overlay:
        parts.append(_overlay_svg(flat, blocks, placed, raw_wires, mm_raw))

    parts.append("</svg>")
    return "\n".join(parts)


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
