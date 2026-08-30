"""The Schematic pane: the circuit the human drew, inside CircuitInsight.

Fed by the layout hints the Virtuoso harvester (`CInHintsExport`) writes
beside the CIN (`<stem>.hints.json`) and the personal SVG symbol
library (settings key `symlib`, else `CIN_SYMLIB`). The picture is
hierarchical like its source: one page per definition, a breadcrumb to
descend into a block (double-click) and come back.

Cross-probing: a click on a device selects it in the Instances tree
(its dot-joined path, the same name the engine uses); selecting in the
tree highlights it here. Nothing is computed in this pane; it only
shows the composer's SVG and maps hits through the instance ids the
composer emits (`<g id="inst:NAME" class="device|block|glyph">`).
"""
from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QPainter, QPen
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtSvgWidgets import QGraphicsSvgItem
from PySide6.QtWidgets import (QFileDialog, QGraphicsRectItem,
                               QGraphicsScene, QGraphicsView, QHBoxLayout,
                               QLabel, QMenu, QToolButton,
                               QVBoxLayout, QWidget)

from ...schematic import (ComposeError, SymbolLibrary, compose,
                          confirmed_only, load_block_symbols, load_hints,
                          page, sidecar_path)
from ...schematic.blocks import propose, save_block_symbols
from ...schematic.compose import (NET_ACGROUND, STYLE_LUMPED,
                                  STYLE_NUMERIC, STYLE_PASSIVE,
                                  STYLE_REMOVED, STYLE_SYMBOLIC, _xml_id)
from .. import theme


def hints_path(cin_path) -> Path:
    """``tb_fc.cin.json`` -> ``tb_fc.hints.json`` beside it."""
    p = Path(cin_path)
    stem = p.name[:-len(".cin.json")] if p.name.endswith(".cin.json") \
        else p.stem
    return p.with_name(stem + ".hints.json")


def _svg_darken(svg: str, sch: dict) -> str:
    """The composed page re-inked BY ROLE for the dark ground, at
    DISPLAY time: compose keeps emitting the canonical black-on-white
    document (the export artifact), and the view recolors it the way
    Virtuoso's own schematic editor draws -- green device artwork,
    blue wires with their junction dots, wire-family net names,
    yellow instance and pin labels. The roles are told apart by the
    exact markup compose emits: wires carry WIRE_STYLE verbatim, net
    names their blue fill, mathtext labels fill+stroke="none";
    whatever #000000 remains is symbol artwork. Contributor-colored
    strokes are already non-black and pass through untouched."""
    from ...schematic.compose import WIRE_STYLE

    s = svg.replace(WIRE_STYLE,
                    WIRE_STYLE.replace("#000000", sch["wire"]))
    s = s.replace('fill="black"', f'fill="{sch["wire"]}"')   # junction dots
    s = s.replace('fill="#1a466b"', f'fill="{sch["net"]}"')  # net names
    s = s.replace('fill="#000000" stroke="none"',            # mathtext labels
                  f'fill="{sch["label"]}" stroke="none"')
    s = s.replace('#000000', sch["device"])                  # symbol artwork
    s = s.replace('"#000"', f'"{sch["device"]}"')
    # white fills follow the canvas: block plates AND the library's
    # own filled shapes (the switch contacts wear fill:#ffffff)
    s = s.replace('fill="white"', f'fill="{sch["paper"]}"')
    s = s.replace('fill:#ffffff', f'fill:{sch["paper"]}')
    s = s.replace('fill="#ffffff"', f'fill="{sch["paper"]}"')
    # default-black text inherits the label ink from the root
    return s.replace("<svg ", f'<svg fill="{sch["label"]}" ', 1)


_BODY_MEASURE_CACHE: dict = {}


def _measure_body(sym, inner) -> tuple | None:
    """The symbol ARTWORK's tight local bbox in symbol mm, measured
    from rendered pixels: wrap the stripped inner in the symbol's own
    viewBox, render offscreen, and take the alpha bounding box. The
    one source that is honest about what is actually DRAWN --
    QSvgRenderer.boundsOnElement pads with heuristics and ignores a
    group's mirror transform (measured: the isource body reported
    wider than its own canvas). Cached per symbol artwork (the inner
    markup is the identity: variants are distinct strings)."""
    key = (sym.viewbox, sym.w, sym.h, hash(sym.inner))
    if key in _BODY_MEASURE_CACHE:
        return _BODY_MEASURE_CACHE[key]
    out = None
    try:
        import numpy as np
        from PySide6.QtGui import QImage, QPainter

        vb = [float(v) for v in sym.viewbox.split()]
        if vb[2] > 0 and vb[3] > 0:
            scale = 256.0 / max(vb[2], vb[3])
            w = max(2, int(round(vb[2] * scale)))
            h = max(2, int(round(vb[3] * scale)))
            doc = (f'<svg xmlns="http://www.w3.org/2000/svg" '
                   f'viewBox="{sym.viewbox}">{inner}</svg>')
            r = QSvgRenderer()
            if r.load(doc.encode("utf-8")):
                img = QImage(w, h, QImage.Format_ARGB32)
                img.fill(0)
                p = QPainter(img)
                r.render(p)
                p.end()
                buf = np.frombuffer(bytes(img.constBits()), dtype=np.uint8)
                alpha = buf.reshape(h, w, 4)[:, :, 3]
                ys, xs = np.nonzero(alpha)
                if xs.size:
                    lx0 = vb[0] + xs.min() / scale
                    lx1 = vb[0] + (xs.max() + 1) / scale
                    ly0 = vb[1] + ys.min() / scale
                    ly1 = vb[1] + (ys.max() + 1) / scale
                    # viewBox px -> symbol mm
                    fx = sym.w / vb[2]
                    fy = sym.h / vb[3]
                    out = (float((lx0 - vb[0]) * fx),
                           float((ly0 - vb[1]) * fy),
                           float((lx1 - vb[0]) * fx),
                           float((ly1 - vb[1]) * fy))
    except Exception:                             # noqa: BLE001
        out = None                                # canvas-frame fallback
    _BODY_MEASURE_CACHE[key] = out
    return out


def _symbol_variant(device_type: str, params: dict):
    """The symbol-variant slot of the composer's devtypes join. For a
    switch it is the resolved AC state, by the solver's own rule -- the
    schematic shows the configuration the small-signal circuit is in
    (fc: S1 open, S0 closed), never the drawn-in-Virtuoso one."""
    if device_type == "switch":
        from ...models.small_signal import switch_ac_state
        return {0: "open", 1: "closed"}.get(switch_ac_state(params))
    return params.get("polarity")


class SchematicView(QGraphicsView):
    """Zoomable, pannable SVG with instance hit-testing."""
    instanceClicked = Signal(str, str)      # name, kind
    instanceActivated = Signal(str, str)    # double-click
    backgroundClicked = Signal()            # click on nothing: deselect
    netClicked = Signal(str)                # click on a wire: its net
    netMenu = Signal(str, object)           # right-click on a wire
    instanceMenu = Signal(str, str, object)  # right-click: name, kind, QPoint

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHints(QPainter.Antialiasing
                            | QPainter.SmoothPixmapTransform
                            | QPainter.TextAntialiasing)
        # left button is SELECTION only; the middle button pans (a
        # left-drag pan stole every click that moved a pixel)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setFocusPolicy(Qt.ClickFocus)   # a click arms the "f" fit key
        self._pan_from = None
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        # a paper drawing: black ink on white in the light theme; the
        # dark theme re-inks at display time (_svg_darken) and the
        # render path sets the matching background
        self.setBackgroundBrush(QBrush(QColor("#ffffff")))
        # the item OWNS its renderer: a renderer parented elsewhere
        # and shared into the scene crashed at teardown (undefined
        # destruction order between view children and scene items)
        self._renderer: QSvgRenderer | None = None
        self._item: QGraphicsSvgItem | None = None
        self._bounds: dict = {}          # name -> (QRectF item coords, kind)
        self._hl: QGraphicsRectItem | None = None
        self._net_hl = None              # the selected net's wire run
        self._net_wires: dict = {}       # net -> [polyline (mm)], from
        self._sx = self._sy = 1.0        # the composer

    # ----------------------------------------------------------- content
    def load_svg(self, svg: str, instances: dict,
                 net_wires: dict | None = None,
                 keep_view: bool = False,
                 bodies: dict | None = None) -> None:
        """instances: name -> kind (device/block/glyph); net_wires:
        net -> [polyline in SVG user units], for the net highlight.
        bodies: name -> (x0, y0, x1, y1) artwork rect in SVG user
        units (compose.last_bodies) -- the selection bounds of
        choice; boundsOnElement pads with stroke extents and
        exceeded the drawn symbol (field report).
        keep_view: a decoration-only re-render (cross-light, styles)
        must not throw away the user's zoom and pan."""
        saved = None
        if keep_view and self._item is not None:
            saved = (self.transform(),
                     self.horizontalScrollBar().value(),
                     self.verticalScrollBar().value())
        self._scene.clear()
        self._hl = None
        self._hl_set = []
        self._hl_wires = []
        self._net_hl = None
        self._net_wires = dict(net_wires or {})
        self._bounds = {}
        self._item = QGraphicsSvgItem()
        self._renderer = self._item.renderer()
        self._renderer.load(svg.encode("utf-8"))
        self._item.setElementId("")          # re-read size after load
        self._scene.addItem(self._item)
        vb = self._renderer.viewBoxF()
        br = self._item.boundingRect()
        self._sx = br.width() / vb.width() if vb.width() else 1.0
        self._sy = br.height() / vb.height() if vb.height() else 1.0
        for name, kind in instances.items():
            # the selection box hugs the ARTWORK: the composer's own
            # body rect is the geometric truth; boundsOnElement (the
            # fallback for anything without one) unions stroke
            # extents and runs larger than the drawn symbol
            body = (bodies or {}).get(name)
            if body is not None:
                x0, y0, x1, y1 = body
                r = QRectF((x0 - vb.x()) * self._sx,
                           (y0 - vb.y()) * self._sy,
                           (x1 - x0) * self._sx, (y1 - y0) * self._sy)
                self._bounds[name] = (r, kind)
                continue
            eid = _xml_id(name)
            if not self._renderer.elementExists(eid):
                continue
            b = self._renderer.boundsOnElement(eid)
            r = QRectF((b.x() - vb.x()) * self._sx,
                       (b.y() - vb.y()) * self._sy,
                       b.width() * self._sx, b.height() * self._sy)
            self._bounds[name] = (r, kind)
        # a full page of pan headroom on every side: the scroll range
        # never collapses, so the pan works even when the whole
        # schematic fits in the viewport
        m = max(br.width(), br.height())
        self._scene.setSceneRect(br.adjusted(-m, -m, m, m))
        if saved is not None:
            t, hx, vx = saved
            self.setTransform(t)
            self.horizontalScrollBar().setValue(hx)
            self.verticalScrollBar().setValue(vx)
        else:
            self.fit()

    def fit(self) -> None:
        if self._item is not None:
            self.fitInView(self._item, Qt.KeepAspectRatio)

    def zoom(self, factor: float) -> None:
        """Keyboard zoom about the view center (Virtuoso's ]/[)."""
        self.scale(factor, factor)

    def pan_step(self, dx: int, dy: int) -> None:
        """Pan one keyboard step: a quarter of the viewport in the
        given direction, the SAME fraction on both axes (the raw
        scrollbar single-steps drift apart -- horizontal outran
        vertical)."""
        h, v = self.horizontalScrollBar(), self.verticalScrollBar()
        h.setValue(h.value() + dx * max(1, self.viewport().width() // 4))
        v.setValue(v.value() + dy * max(1, self.viewport().height() // 4))

    def hit(self, view_pos):
        """(name, kind) of the smallest instance under a view point."""
        if self._item is None:
            return None
        p = self._item.mapFromScene(self.mapToScene(view_pos))
        best = None
        for name, (r, kind) in self._bounds.items():
            if r.adjusted(-1, -1, 1, 1).contains(p):
                a = r.width() * r.height()
                if best is None or a < best[0]:
                    best = (a, name, kind)
        return (best[1], best[2]) if best else None

    def highlight(self, name: str | None) -> None:
        if self._hl is not None:
            self._scene.removeItem(self._hl)
            self._hl = None
        if name is None or name not in self._bounds or self._item is None:
            return
        r, _ = self._bounds[name]
        sr = self._item.mapRectToScene(r.adjusted(-1.5, -1.5, 1.5, 1.5))
        # the SELECTION mark follows Virtuoso: white on the dark
        # canvas (selected objects turn white there), the accent on
        # the light paper where white would vanish
        c = QColor(theme.SCH["select"])
        fill = QColor(c)
        fill.setAlpha(40)
        self._hl = self._scene.addRect(sr, QPen(c, 0.6), QBrush(fill))
        self._hl.setZValue(10)

    def scene_center(self, name: str):
        r = self._bounds.get(name, (None, None))[0]
        if r is None or self._item is None:
            return None
        return self._item.mapRectToScene(r).center()

    def highlight_many(self, names) -> None:
        """The WEIGHTED cross-light of a ledger row: entries are names
        or (name, weight 0..1) pairs -- the strongest contributors
        read most emphasized, the weak ones fade -- independent of the
        single tree-driven highlight."""
        for it in getattr(self, "_hl_set", []):
            try:
                self._scene.removeItem(it)
            except RuntimeError:              # scene already rebuilt
                pass
        self._hl_set = []
        if self._item is None:
            return
        for entry in names:
            name, w = entry if isinstance(entry, tuple) else (entry, 1.0)
            if name not in self._bounds:
                continue
            w = max(0.05, min(1.0, float(w)))
            r, _ = self._bounds[name]
            sr = self._item.mapRectToScene(r.adjusted(-1.5, -1.5, 1.5, 1.5))
            it = self._scene.addRect(
                sr, QPen(QColor(213, 94, 0, int(60 + 180 * w)),
                         0.3 + 0.6 * w),
                QBrush(QColor(213, 94, 0, int(12 + 95 * w))))
            it.setZValue(10)
            self._hl_set.append(it)

    @staticmethod
    def _net_join_segments(polys, attach_mm):
        """The minimal run of a net's segments JOINING the attach
        points: union of BFS paths over the segment graph from the
        first anchor to every other. A net keeps flowing past its
        contributors (the pair's tail net runs on to the mirrors);
        painting the WHOLE run gradient-tinted read as if those
        bystanders were on the map (field report: PM7/PM9). None on
        any trouble -- the caller falls back to the full run."""
        from collections import deque

        def key(p):
            return (round(p[0], 3), round(p[1], 3))

        raw = []
        nodes = set()
        for pts in polys:
            for a, b in zip(pts, pts[1:]):
                ka, kb = key(a), key(b)
                if ka != kb:
                    raw.append((ka, kb))
                    nodes.add(ka)
                    nodes.add(kb)
        if not raw or len(attach_mm) < 2:
            return None
        # a wire ATTACHING MID-SEGMENT (the junction dots) puts a
        # node on another segment's interior: without splitting there
        # the graph disconnects at every T and the joining run came
        # out with gaps (field report). Split each segment at every
        # node lying on it.
        adj: dict = {}

        def _link(u, v):
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)

        for a, b in raw:
            dx, dy = b[0] - a[0], b[1] - a[1]
            ln2 = dx * dx + dy * dy
            on = []
            for p in nodes:
                if p == a or p == b:
                    continue
                px, py = p[0] - a[0], p[1] - a[1]
                t = (px * dx + py * dy) / ln2 if ln2 else -1.0
                if not 0.0 < t < 1.0:
                    continue
                cross = px * dy - py * dx
                if cross * cross <= 1e-6 * ln2:
                    on.append((t, p))
            chain = [a] + [p for _, p in sorted(on)] + [b]
            for u, v in zip(chain, chain[1:]):
                _link(u, v)
        if not adj:
            return None
        anchors = [min(adj, key=lambda k: (k[0] - p[0]) ** 2
                       + (k[1] - p[1]) ** 2) for p in attach_mm]
        base = anchors[0]
        parent = {base: None}
        dq = deque([base])
        while dq:
            u = dq.popleft()
            for v in adj[u]:
                if v not in parent:
                    parent[v] = u
                    dq.append(v)
        edges = set()
        for a in anchors[1:]:
            if a not in parent:
                continue
            v = a
            while parent[v] is not None:
                u = parent[v]
                edges.add((u, v) if u <= v else (v, u))
                v = u
        return [[u, v] for u, v in sorted(edges)] or None

    def highlight_wires(self, wires) -> None:
        """Contributor-JOINING wires, each drawn with a linear-gradient
        pen running from its strongest attached contributor toward its
        weakest. wires: [(net, w_strong, w_weak, p_strong, p_weak)] or
        with a sixth element: the contributors' scene points, which
        restrict the drawing to the segments that JOIN them (the rest
        of the net's run stays unlit)."""
        from PySide6.QtGui import QLinearGradient, QPainterPath

        for it in getattr(self, "_hl_wires", []):
            try:
                self._scene.removeItem(it)
            except RuntimeError:
                pass
        self._hl_wires = []
        if self._item is None:
            return
        vb = self._renderer.viewBoxF()
        for entry in wires:
            net, ws, ww, ps, pw = entry[:5]
            attach = entry[5] if len(entry) > 5 else None
            polys = self._net_wires.get(net)
            if not polys:
                continue
            if attach:
                mmpts = []
                for sp in attach:
                    if sp is None:
                        continue
                    ip = self._item.mapFromScene(sp)
                    mmpts.append((ip.x() / self._sx + vb.x(),
                                  ip.y() / self._sy + vb.y()))
                segs = (self._net_join_segments(polys, mmpts)
                        if len(mmpts) >= 2 else None)
                if segs:
                    polys = segs
            path = QPainterPath()
            for pts in polys:
                x0, y0 = pts[0]
                path.moveTo((x0 - vb.x()) * self._sx,
                            (y0 - vb.y()) * self._sy)
                for x, y in pts[1:]:
                    path.lineTo((x - vb.x()) * self._sx,
                                (y - vb.y()) * self._sy)
            spath = self._item.mapToScene(path)
            if ps is not None and pw is not None and ps != pw:
                grad = QLinearGradient(ps, pw)
                grad.setColorAt(0.0, QColor(213, 94, 0,
                                            int(60 + 180 * ws)))
                grad.setColorAt(1.0, QColor(213, 94, 0,
                                            int(30 + 120 * ww)))
                pen = QPen(QBrush(grad), 0.9)
            else:
                pen = QPen(QColor(213, 94, 0, int(50 + 150 * ws)), 0.9)
            pen.setCapStyle(Qt.RoundCap)
            it = self._scene.addPath(spath, pen)
            it.setZValue(9)
            self._hl_wires.append(it)

    def bounds_of(self, name: str):
        return self._bounds.get(name, (None, None))[0]

    def hit_net(self, view_pos):
        """The net whose wire runs nearest the view point, within a
        lane's width -- None when nothing is close. Instances win over
        wires (hit() is asked first), so this only fires on the open
        wire between symbols."""
        if self._item is None or not self._net_wires:
            return None
        p = self._item.mapFromScene(self.mapToScene(view_pos))
        vb = self._renderer.viewBoxF()
        qx = p.x() / self._sx + vb.x()
        qy = p.y() / self._sy + vb.y()
        tol = 0.8                          # mm: most of a 1 mm lane
        best = None
        for net, polys in self._net_wires.items():
            for pts in polys:
                for (ax, ay), (bx, by) in zip(pts, pts[1:]):
                    dx, dy = bx - ax, by - ay
                    L2 = dx * dx + dy * dy
                    t = 0.0 if L2 == 0 else max(
                        0.0, min(1.0, ((qx - ax) * dx + (qy - ay) * dy)
                                 / L2))
                    ex, ey = ax + t * dx - qx, ay + t * dy - qy
                    d = (ex * ex + ey * ey) ** 0.5
                    if d <= tol and (best is None or d < best[0]):
                        best = (d, net)
        return best[1] if best else None

    def highlight_net(self, net: str | None) -> None:
        """The net's own wire run, overdrawn -- a bounding rect around
        a whole net would cover half the page."""
        if self._net_hl is not None:
            self._scene.removeItem(self._net_hl)
            self._net_hl = None
        polys = self._net_wires.get(net) if net else None
        if not polys or self._item is None:
            return
        from PySide6.QtGui import QPainterPath

        vb = self._renderer.viewBoxF()
        path = QPainterPath()
        for pts in polys:
            x0, y0 = pts[0]
            path.moveTo((x0 - vb.x()) * self._sx,
                        (y0 - vb.y()) * self._sy)
            for x, y in pts[1:]:
                path.lineTo((x - vb.x()) * self._sx,
                            (y - vb.y()) * self._sy)
        # a selected net follows the selection ink (white on dark)
        c = QColor(theme.SCH["select"])
        c.setAlpha(200)
        pen = QPen(c, 1.0)
        pen.setCapStyle(Qt.RoundCap)
        item = self._scene.addPath(self._item.mapToScene(path), pen)
        item.setZValue(10)
        self._net_hl = item

    # ------------------------------------------------------------ events
    def wheelEvent(self, ev):
        f = 1.15 if ev.angleDelta().y() > 0 else 1 / 1.15
        self.scale(f, f)

    def mousePressEvent(self, ev):
        if ev.button() == Qt.MiddleButton:
            self._pan_from = ev.position()
            self.setCursor(Qt.ClosedHandCursor)
            ev.accept()
            return
        if ev.button() == Qt.LeftButton:
            h = self.hit(ev.position().toPoint())
            if h:
                self.instanceClicked.emit(*h)
            else:
                net = self.hit_net(ev.position().toPoint())
                if net:
                    self.netClicked.emit(net)
                else:
                    self.backgroundClicked.emit()
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        if self._pan_from is not None and (ev.buttons() & Qt.MiddleButton):
            d = ev.position() - self._pan_from
            self._pan_from = ev.position()
            h = self.horizontalScrollBar()
            v = self.verticalScrollBar()
            h.setValue(int(h.value() - d.x()))
            v.setValue(int(v.value() - d.y()))
            ev.accept()
            return
        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev):
        if ev.button() == Qt.MiddleButton:
            self._pan_from = None
            self.unsetCursor()
            ev.accept()
            return
        super().mouseReleaseEvent(ev)

    def keyPressEvent(self, ev):
        """Focused, the view answers the canvas keys itself -- through
        the same table the window's fall-through uses, so the arrows
        pan uniformly instead of QGraphicsView's per-axis scrollbar
        steps (horizontal outran vertical)."""
        if not (ev.modifiers() & ~Qt.KeypadModifier):
            fn = SchematicMixin._CANVAS_KEYS.get(ev.key())
            if fn is not None:
                fn(self)
                ev.accept()
                return
        super().keyPressEvent(ev)

    def mouseDoubleClickEvent(self, ev):
        h = self.hit(ev.position().toPoint())
        if h:
            self.instanceActivated.emit(*h)
        super().mouseDoubleClickEvent(ev)

    def contextMenuEvent(self, ev):
        h = self.hit(ev.pos())
        if h:
            self.instanceMenu.emit(h[0], h[1], ev.globalPos())
            return
        net = self.hit_net(ev.pos())
        if net:
            self.netMenu.emit(net, ev.globalPos())
            return
        super().contextMenuEvent(ev)


class SchematicMixin:
    """MainWindow part: the Schematic tab."""

    def _schematic_page(self):
        pagew = QWidget()
        v = QVBoxLayout(pagew)
        row = QHBoxLayout()
        self._sch_crumb = QHBoxLayout()
        row.addLayout(self._sch_crumb)
        row.addStretch(1)
        self._sch_status = QLabel("")
        self._sch_status.setWordWrap(True)
        row.addWidget(self._sch_status, 1)
        # icon toggles + the fit verb, drawn from the CircuitInsight
        # icon set (assets/*.svg, re-inked per theme by theme.icon);
        # the Library... chooser lives in the Options menu now
        self._sch_state_chk = QToolButton()
        self._sch_state_chk.setCheckable(True)
        self._sch_state_chk.setChecked(True)
        self._sch_state_chk.setAutoRaise(True)
        self._sch_state_chk.setToolTip(
            "Analysis state -- draw the session's state on the layout: "
            "AC-grounded nets gray-dashed with an earth, removed sources "
            "faded, ĝm-lumped devices badged, devices whose symbol is in "
            "the shown formula emphasized. Off: the plain drawing.")
        self._sch_state_chk.toggled.connect(
            lambda _on: self._schematic_render())
        row.addWidget(self._sch_state_chk)
        self._sch_match_chk = QToolButton()
        self._sch_match_chk.setCheckable(True)
        self._sch_match_chk.setChecked(True)
        self._sch_match_chk.setAutoRaise(True)
        self._sch_match_chk.setToolTip(
            "Matches -- tint matched device groups (the shared-symbol "
            "pairs) in their group colors. Off: no group colors -- the "
            "analysis state marks keep their own toggle.")
        self._sch_match_chk.toggled.connect(
            lambda _on: self._schematic_render())
        row.addWidget(self._sch_match_chk)
        fitb = QToolButton()
        fitb.setAutoRaise(True)
        fitb.setToolTip(
            "Fit the page in the view (also the F key; wheel zooms, "
            "drag pans)")
        fitb.clicked.connect(lambda: self.sch_view.fit())
        row.addWidget(fitb)
        self._sch_fit_btn = fitb
        self._sch_refresh_icons()
        v.addLayout(row)
        self.sch_view = SchematicView()
        self.sch_view.instanceClicked.connect(self._schematic_clicked)
        self.sch_view.instanceActivated.connect(self._schematic_activated)
        self.sch_view.instanceMenu.connect(self._schematic_menu)
        self.sch_view.backgroundClicked.connect(self._schematic_deselect)
        self.sch_view.netClicked.connect(self._schematic_net_clicked)
        self.sch_view.netMenu.connect(self._schematic_net_menu)
        v.addWidget(self.sch_view, 1)
        self._sch_hints = None
        self._sch_cin = None
        self._sch_lib = None
        self._sch_blocks = {}
        self._sch_mapping = {}           # the whole sidecar, confirmed or not
        self._sch_proposals: dict = {}   # cell -> BlockSymbol proposal
        self._sch_path: list = []        # [(instance name, defname)]
        self._sch_synced = False
        # f / ] / [ ride WINDOW SHORTCUTS: they must beat the item
        # views' type-to-find (which would eat any printable key),
        # and text-entry widgets still claim their characters through
        # Qt's shortcut-override, so typing in a filter stays safe.
        # The arrows instead FALL THROUGH (keyPressEvent below): their
        # owners -- trees, tables, combos -- consume them first, and
        # only an unconsumed press pans the canvas.
        from PySide6.QtGui import QKeySequence, QShortcut
        for key in (Qt.Key_F, Qt.Key_BracketRight, Qt.Key_BracketLeft):
            s = QShortcut(QKeySequence(key), self)
            s.setContext(Qt.WindowShortcut)
            s.activated.connect(
                lambda k=key: self._CANVAS_KEYS[k](self.sch_view)
                if self.sch_view.isVisible() else None)
        return pagew

    #: Virtuoso's canvas keys: what an UNCONSUMED press does to the view
    _CANVAS_KEYS = {
        Qt.Key_F: lambda v: v.fit(),
        Qt.Key_BracketRight: lambda v: v.zoom(2.0),
        Qt.Key_BracketLeft: lambda v: v.zoom(0.5),
        Qt.Key_Left: lambda v: v.pan_step(-1, 0),
        Qt.Key_Right: lambda v: v.pan_step(1, 0),
        Qt.Key_Up: lambda v: v.pan_step(0, -1),
        Qt.Key_Down: lambda v: v.pan_step(0, 1),
    }

    def keyPressEvent(self, ev):
        """The canvas keys, by FALL-THROUGH: a key press the focused
        widget does not consume bubbles up the parent chain and lands
        here -- so f, ]/[ and the arrows drive the schematic from
        anywhere, while a filter box keeps its letters and a tree its
        arrow navigation, with no focus bookkeeping at all. (An
        app-wide event filter did this before and segfaulted libpyside
        on <sim-host>; window shortcuts fired BEFORE the focused widget
        instead of after it.)"""
        view = getattr(self, "sch_view", None)
        if (view is not None and view.isVisible()
                and not (ev.modifiers() & ~Qt.KeypadModifier)):
            fn = self._CANVAS_KEYS.get(ev.key())
            if fn is not None:
                fn(view)
                ev.accept()
                return
        super().keyPressEvent(ev)

    def _sch_refresh_icons(self):
        """(Re-)hand the header buttons their themed icons; called at
        build time and from apply_theme (theme.icon caches per theme,
        but a button keeps the QIcon it was handed)."""
        from .. import theme
        self._sch_state_chk.setIcon(theme.icon("state"))
        self._sch_match_chk.setIcon(theme.icon("match"))
        self._sch_fit_btn.setIcon(theme.icon("fit"))

    # ------------------------------------------------------------ loading
    def _schematic_library_root(self):
        """The GUI setting, else CIN_SYMLIB, else the symbols submodule
        of a source checkout (compose.library_root's own order)."""
        from ...schematic.compose import library_root

        root = str(self._settings().value("symlib", "") or "")
        if root:
            return root
        try:
            return str(library_root(None))
        except ComposeError:
            return None

    def _schematic_pick_library(self):
        d = QFileDialog.getExistingDirectory(
            self, "SVG symbol library (directory with terminals.csv)",
            self._schematic_library_root() or "")
        if d:
            self._settings().setValue("symlib", d)
            self._schematic_load()

    def _schematic_load(self):
        """Find the hints beside the CIN, the library, the block symbols;
        open the top page. Every miss is a sentence in the status, never
        an exception in the open path."""
        self._sch_hints = self._sch_cin = self._sch_lib = None
        self._sch_last_render_key = None
        self._sch_last_page = None
        self._sch_blocks = {}
        self._sch_path = []
        self.sch_view.load_svg('<svg xmlns="http://www.w3.org/2000/svg" '
                               'width="1" height="1"/>', {})
        cin = getattr(self, "_cin", None)
        if not cin:
            self._sch_status.setText("")
            return
        hp = hints_path(cin)
        if not hp.exists():
            self._sch_status.setText(
                f"no layout hints beside the CIN ({hp.name}): in Virtuoso "
                f"run CInHintsExport(lib cell \"schematic\" path) and save "
                f"it next to the CIN")
            self._schematic_crumbs()
            return
        root = self._schematic_library_root()
        if not root:
            self._sch_status.setText(
                "no symbol library: choose it with Library… (or set "
                "CIN_SYMLIB; a source checkout has it as the symbols "
                "submodule: git submodule update --init)")
            self._schematic_crumbs()
            return
        try:
            self._sch_hints = load_hints(hp)
            self._sch_cin = json.loads(Path(cin).read_text(encoding="utf-8"))
            self._sch_lib = SymbolLibrary(root)
            self._sch_mapping = load_block_symbols(sidecar_path(cin))
            self._sch_blocks = confirmed_only(self._sch_mapping)
            self._sch_proposals = {}
        except Exception as e:                   # noqa: BLE001
            self._sch_status.setText(f"schematic unavailable: {e}")
            self._schematic_crumbs()
            return
        top = self._sch_hints.top
        self._sch_path = [(None, top)]
        if not self._sch_synced:
            self.devices.itemSelectionChanged.connect(
                self._schematic_follow_tree)
            self.nets_tree.itemSelectionChanged.connect(
                self._schematic_follow_nets)
            self._sch_synced = True
        self._schematic_render()

    def _schematic_prefix(self) -> str:
        names = [n for n, _ in self._sch_path if n]
        return ".".join(names) + "." if names else ""

    def _sch_localize_net(self, full: str) -> str | None:
        """SESSION net name -> this page's local net, walking the
        breadcrumb: an instance's own nets lose the prefix; a parent
        net entering the block becomes the PORT net inside (through
        the instance's terminals); global rails (name!) pass through.
        None when the net has no name on this page."""
        defs = (self._sch_cin or {}).get("definitions", {})
        name, parent = full, self._sch_path[0][1]
        for inst_name, defname in self._sch_path[1:]:
            pre = inst_name + "."
            if name.startswith(pre):
                name = name[len(pre):]
            elif name.endswith("!"):
                pass                          # vdd! and friends
            else:
                inst = next((i for i in defs.get(parent,
                                                 {}).get("instances", [])
                             if i.get("name") == inst_name), None)
                inv = {v: k for k, v in
                       ((inst or {}).get("terminals") or {}).items()}
                if name not in inv:
                    return None
                name = inv[name]
            parent = defname
        return name if "." not in name else None

    def _schematic_render(self):
        if not self._sch_hints or not self._sch_path:
            return
        defname = self._sch_path[-1][1]
        cdef = (self._sch_cin.get("definitions", {}).get(defname)
                or {"instances": []})
        devtypes = {i["name"]: (i["device_type"],
                                _symbol_variant(i["device_type"],
                                                i.get("params") or {}))
                    for i in cdef["instances"] if "device_type" in i}
        # Virtuoso-deactivated instances: in the harvest (drawn X-ed,
        # like the source editor), never in the model
        deact = frozenset(i["name"] for i in cdef["instances"]
                          if i.get("deactivated"))
        subckts = frozenset(self._sch_cin.get("definitions", {}))
        styles, net_styles = ({}, {})
        if self._sch_state_chk.isChecked():
            try:
                styles, net_styles = self._schematic_styles()
            except Exception:                     # noqa: BLE001
                styles, net_styles = ({}, {})     # decoration only
        # the match-group tints are STRUCTURAL session state with
        # their own toggle -- one can read the reduction marks without
        # the group colors, and vice versa
        tints = (self._schematic_tints()
                 if self._sch_match_chk.isChecked() else {})
        aliases = self._schematic_aliases()
        # the selected compensation suggestion, localized to this page
        # (entries whose nets have no name here are dropped; the
        # composer additionally skips nets without wires)
        proposed = []
        for na, nb, text, with_r in getattr(self, "_sch_proposed", ()):
            la = self._sch_localize_net(na)
            lb = self._sch_localize_net(nb) if nb is not None else None
            if la is not None and (nb is None or lb is not None):
                proposed.append((la, lb, text, with_r))
        proposed = tuple(proposed)
        # a solve restyles the analysis state on every _show, but most
        # of the time nothing the drawing encodes has changed: same
        # page, same styles, same tints, same aliases, same blocks --
        # skip the whole re-compose then, and the view (zoom, pan,
        # highlights) is untouched by construction
        # the contribution map holds FULL instance names and localizes
        # PER PAGE at render time: a contributor deeper than this page
        # lights its enclosing block (the OTA symbol contains
        # contributors), a page's own instance lights itself -- so the
        # cross-light follows navigation up and down
        contrib: dict[str, float] = {}
        pre_c = self._schematic_prefix()
        for full, w in (getattr(self, "_sch_contrib", None)
                        or {}).items():
            if not full.startswith(pre_c):
                continue
            head = full[len(pre_c):].split(".", 1)[0]
            if head:
                contrib[head] = max(contrib.get(head, 0.0), float(w))
        dark = theme.current == "dark"
        key = (defname, self._sch_state_chk.isChecked(), styles,
               net_styles, tints, aliases, self._sch_blocks, proposed,
               tuple(sorted(contrib.items())), theme.current,
               tuple(sorted(deact)),
               str(self._sch_lib.root) if self._sch_lib else "")
        if key == getattr(self, "_sch_last_render_key", None):
            return
        try:
            pg = page(self._sch_hints, defname)
            svg = compose(pg, devtypes, self._sch_lib, subckts=subckts,
                          block_symbols=self._sch_blocks,
                          styles=styles, net_styles=net_styles,
                          proposed_cells=self._schematic_proposed(pg,
                                                                  subckts),
                          tints=tints, tex_labels=True,
                          aliases=aliases, proposed=proposed,
                          contrib=contrib or None,
                          contrib_base=(theme.SCH["device"] if dark
                                        else "#000000"),
                          port_base=(theme.SCH["port"] if dark
                                     else None),
                          deactivated=deact,
                          body_measure=_measure_body)
        except (ComposeError, KeyError, ValueError) as e:
            self._sch_status.setText(f"{defname}: {e}")
            self._schematic_crumbs()
            return
        if dark:
            # display re-ink only: exports keep the canonical paper
            svg = _svg_darken(svg, theme.SCH)
        self.sch_view.setBackgroundBrush(QBrush(QColor(theme.SCH["paper"])))
        insts = dict(getattr(compose, "last_instances", {}))
        kinds = {n: ("block" if c in subckts else "device")
                 for n, c in insts.items()}
        # a SAME-PAGE re-render is a restyle: the user's zoom and pan
        # survive it; only a page change re-fits
        same_page = getattr(self, "_sch_last_page", None) == defname
        if same_page:
            tr = self.sch_view.transform()
            hv = self.sch_view.horizontalScrollBar().value()
            vv = self.sch_view.verticalScrollBar().value()
        self.sch_view.load_svg(
            svg, kinds,
            net_wires=dict(getattr(compose, "last_net_wires", {})),
            bodies=dict(getattr(compose, "last_bodies", {})))
        if same_page:
            self.sch_view.setTransform(tr)
            self.sch_view.horizontalScrollBar().setValue(hv)
            self.sch_view.verticalScrollBar().setValue(vv)
        self._sch_last_page = defname
        self._sch_last_render_key = key
        # the sensitivity map's wire gradients are OVERLAY items and
        # died with the scene rebuild -- re-apply them from the
        # stored light state (margins mixin)
        if getattr(self, "_sch_contrib", None):
            try:
                self._ml_apply_wires()
            except Exception:                 # noqa: BLE001
                pass                          # decoration only
        # no instance/wire counts here -- the drawing shows them, the
        # label keeps the analysis-state summary and the diagnostics
        bits = []
        n_acg = sum(1 for v in net_styles.values() if v == NET_ACGROUND)
        n_rm = sum(1 for v in styles.values() if v == STYLE_REMOVED)
        n_pv = sum(1 for v in styles.values() if v == STYLE_PASSIVE)
        n_sym = sum(1 for v in styles.values() if v == STYLE_SYMBOLIC)
        n_num = sum(1 for v in styles.values() if v == STYLE_NUMERIC)
        n_lump = sum(1 for v in styles.values() if v == STYLE_LUMPED)
        state = [f"{n} {what}" for n, what in (
            (n_acg, "net(s) AC-grounded"), (n_rm, "removed"),
            (n_pv, "turned into loads (gm dead)"),
            (n_lump, "ĝm-lumped"), (n_sym, "in the formula"),
            (n_num, "numeric")) if n]
        if state:
            bits.append("; ".join(state))
        self._sch_status.setText(
            f"{defname}: " + ", ".join(bits) if bits else "")
        self._schematic_crumbs()
        # a re-render (a solve restyles the analysis state) rebuilds
        # the scene and drops the highlight overlays; the trees still
        # hold the selection, so re-apply it -- the marks survive the
        # solve instead of silently vanishing (field report)
        self._schematic_follow_tree()
        self._schematic_follow_nets()

    # ------------------------------------------------- analysis state
    def _schematic_styles(self):
        """(instance styles, net styles) for the CURRENT page from the
        session: the reduction's AC-grounded nodes and dead sources,
        the ĝm lumping, and the devices whose symbol is in the shown
        formula. Names are the engine's dot-joined paths; this page's
        prefix is stripped, deeper names are not shown."""
        c = getattr(self, "controller", None)
        if c is None:
            return {}, {}
        pre = self._schematic_prefix()

        def local(full):
            if pre:
                return full[len(pre):] if full.startswith(pre) else None
            return full if "." not in full else None

        styles: dict = {}
        nets: dict = {}
        summ = c.reduction_summary()
        if summ:
            for n in summ.get("nodes", []):
                ln = local(n)
                if ln:
                    nets[ln] = NET_ACGROUND
            # faded = nothing of the device is left; a device whose
            # transconductance died but whose gds/caps remain is a
            # load and stays drawn (the input pair lost only its gmbs)
            gone = summ.get("devices_silenced")
            if gone is None:                         # older summaries
                gone = summ.get("dead_sources", [])
            for d in gone:
                ld = local(d)
                if ld:
                    styles[ld] = STYLE_REMOVED
            # an explicit element deleted by the removal scan's Apply
            # is gone outright (equivalents are not drawn; local() skips
            # them)
            for d in summ.get("removed", []):
                ld = local(d)
                if ld:
                    styles[ld] = STYLE_REMOVED
            # a device whose gm died is a load now (a mirror with its
            # gate grounded); one that lost only gmbs is untouched
            for prim in summ.get("dead_primitives", []):
                if prim.startswith("gm_"):
                    ld = local(prim[len("gm_"):])
                    if ld and ld not in styles:
                        styles[ld] = STYLE_PASSIVE
        try:
            for d, how in c.lumped_gmb().items():
                ld = local(d)
                if ld and how == "lumped":
                    styles.setdefault(ld, STYLE_LUMPED)
        except Exception:                         # noqa: BLE001
            pass
        r = getattr(self, "result", None)
        tf = getattr(r, "tf", None)
        expr = getattr(tf, "expr", None)
        if expr is not None and hasattr(expr, "free_symbols"):
            present = {str(x) for x in expr.free_symbols}
            for d in c.devices:
                ld = local(d.name)
                if not ld or styles.get(ld) in (STYLE_REMOVED,
                                                STYLE_PASSIVE):
                    continue
                key = d.name.replace(".", "_")
                if d.name in present or any(
                        s.endswith("_" + key) for s in present):
                    styles[ld] = STYLE_SYMBOLIC
                elif ld not in styles:
                    styles[ld] = STYLE_NUMERIC
        return styles, nets

    def _schematic_aliases(self) -> dict:
        """Device-level LaTeX aliases (the Instances tree's 'Set LaTeX
        alias') for this page, by local name: M_1 is typeset as math
        on the schematic, as in the expressions."""
        c = getattr(self, "controller", None)
        if c is None:
            return {}
        pre = self._schematic_prefix()
        names = {d.name for d in c.devices}
        out = {}
        for full, tex in (getattr(c, "sym_aliases", {}) or {}).items():
            if full not in names:
                continue                       # a per-symbol alias
            if pre:
                if full.startswith(pre):
                    out[full[len(pre):]] = tex
            elif "." not in full:
                out[full] = tex
        return out

    def _schematic_tints(self) -> dict:
        """Matched devices on this page: name -> (tint, '🔗n'), the
        same tint and group number the Instances tree shows, so a
        matched pair reads as one across the two views."""
        from ..devtree import LINK

        tints = theme.GROUP_TINTS
        pre = self._schematic_prefix()
        out = {}
        for gi, group in enumerate(getattr(self, "_match_groups", [])):
            fill = tints[gi % len(tints)]
            for full in group:
                if pre:
                    if not full.startswith(pre):
                        continue
                    local = full[len(pre):]
                elif "." in full:
                    continue
                else:
                    local = full
                out[local] = (fill, f"{LINK}{gi + 1}")
        return out

    def _schematic_restyle(self):
        """Re-draw the current page with the session's state; cheap
        enough to call after every result or reduction change."""
        if getattr(self, "_sch_hints", None) and self._sch_path:
            self._schematic_render()

    def _schematic_crumbs(self):
        while self._sch_crumb.count():
            it = self._sch_crumb.takeAt(0)
            w = it.widget()
            if w is not None:
                w.setParent(None)            # off the canvas now, not later
                w.deleteLater()
        for i, (name, defname) in enumerate(self._sch_path):
            txt = defname if name is None else f"{name} ({defname})"
            last = i == len(self._sch_path) - 1
            if last:
                # the CURRENT level: highlighted, not a dimmed disabled
                # button (palette text follows the theme; bold carries
                # the emphasis)
                lbl = QLabel(f"<b>{txt}</b>")
                self._sch_crumb.addWidget(lbl)
            else:
                b = QToolButton()
                b.setText(txt)
                b.setAutoRaise(True)
                b.clicked.connect(
                    lambda _=False, k=i: self._schematic_up(k))
                self._sch_crumb.addWidget(b)
                self._sch_crumb.addWidget(QLabel("›"))

    # ------------------------------------------------------ block symbols
    def _schematic_proposed(self, pg, subckts) -> set:
        """Cells on this page with a proposal but no confirmation."""
        out = set()
        for inst in pg.instances:
            if inst.cell not in subckts or inst.cell in self._sch_blocks:
                continue
            if self._schematic_proposal(inst.cell, inst.terms) is not None:
                out.add(inst.cell)
        return out

    def _schematic_proposal(self, cell: str, terms=None):
        """The sidecar's unconfirmed entry, else a fresh proposal
        (cached per cell), else None."""
        if cell in self._sch_mapping:
            return self._sch_mapping[cell]
        if cell not in self._sch_proposals:
            try:
                avail = set(self._sch_lib.block_candidates())
                self._sch_proposals[cell] = propose(self._sch_cin, cell,
                                                    avail, terms)
            except Exception:                     # noqa: BLE001
                self._sch_proposals[cell] = None
        return self._sch_proposals[cell]

    def _schematic_menu(self, name: str, kind: str, gpos):
        menu = QMenu(self)
        if kind == "block":
            menu.addAction("Symbol…").triggered.connect(
                lambda: self._schematic_symbol_dialog(name))
            menu.addAction("Descend").triggered.connect(
                lambda: self._schematic_activated(name, "block"))
        else:
            menu.addAction("Select in Instances").triggered.connect(
                lambda: self._schematic_clicked(name, kind))
        menu.exec(gpos)

    def _schematic_symbol_dialog(self, name: str, dialog_cls=None):
        """Confirm (or drop) the symbol for the block's cell; the
        sidecar is the record, the page re-renders from it.
        dialog_cls: a test double for the dialog."""
        from ..blockdialog import BlockSymbolDialog

        BlockSymbolDialog = dialog_cls or BlockSymbolDialog
        if not self._sch_hints or self._sch_lib is None:
            return
        defname = self._sch_path[-1][1]
        dd = self._sch_hints.definitions.get(defname)
        inst = next((i for i in dd.instances if i.name == name), None) \
            if dd else None
        if inst is None:
            return
        cell = inst.cell
        cdef = (self._sch_cin.get("definitions", {}).get(cell) or {})
        ports = list(cdef.get("ports") or inst.terms.keys())
        proposal = self._schematic_proposal(cell, inst.terms)
        dlg = BlockSymbolDialog(cell, ports, self._sch_lib, proposal, self)
        dlg.exec()
        # the dialog's own outcome, not exec()'s return value: in
        # PySide6 that int never compares equal to the DialogCode enum,
        # which made every OK read as Cancel
        if dlg.result_mapping is None and not dlg.draw_as_box:
            return
        if dlg.draw_as_box:
            if cell in self._sch_mapping:
                self._sch_mapping[cell].confirmed = False
        elif dlg.result_mapping is not None:
            self._sch_mapping[cell] = dlg.result_mapping
        side = sidecar_path(self._cin)
        try:
            save_block_symbols(side, self._sch_mapping)
        except OSError as e:
            self._sch_status.setText(f"could not write {side.name}: {e}")
            return
        self._sch_blocks = confirmed_only(self._sch_mapping)
        self._schematic_render()
        self.statusBar().showMessage(
            f"{cell}: " + ("drawn as a box" if dlg.draw_as_box else
                           f"drawn as {dlg.result_mapping.symbol}")
            + f" — saved to {side.name}")

    # --------------------------------------------------------- navigation
    def _schematic_up(self, k: int):
        self._sch_path = self._sch_path[:k + 1]
        self._schematic_render()

    def _schematic_activated(self, name: str, kind: str):
        if kind != "block" or not self._sch_hints:
            return
        defname = self._sch_path[-1][1]
        dd = self._sch_hints.definitions.get(defname)
        cell = next((i.cell for i in dd.instances if i.name == name), None) \
            if dd else None
        if cell and cell in self._sch_hints.definitions:
            self._sch_path.append((name, cell))
            self._schematic_render()

    def _schematic_clicked(self, name: str, kind: str):
        self.sch_view.highlight_net(None)
        self.sch_view.highlight(name)
        if kind == "device":
            self._goto_instance(self._schematic_prefix() + name)

    def _schematic_net_clicked(self, local: str):
        """A click on a wire selects its net in the Nets tree -- the
        selection then lights the run, the exact mirror of the
        instance click. Passthrough names (supplies) have no page
        prefix, so the bare name is the fallback."""
        for full in (self._schematic_prefix() + local, local):
            it = self.nets_tree.item_for(full)
            if it is not None:
                self.dev_tabs.setCurrentWidget(self.nets_tree)
                self.nets_tree.setCurrentItem(it)
                self.nets_tree.scrollToItem(it)
                # setCurrentItem on the ALREADY-current item fires no
                # selectionChanged -- follow explicitly, so clicking
                # the selected net's wire re-lights it after a render
                self._schematic_follow_nets()
                self.statusBar().showMessage(f"net: {full}")
                return

    def _schematic_net_actions(self, full: str, local: str,
                               others=()) -> list:
        """The wire's verbs: the same ones the Nets tree offers.
        [(label, callable)], empty for ground. `others`: nets already
        selected when the wire was clicked -- exactly one adds the
        two-net verb, Z between the pair."""
        gnd = set(self.controller.ground) | {"0"} if self.controller \
            else {"0"}
        if full in gnd:
            return []
        acts = [
            ("Set as output", lambda: self._set_output_net(full)),
            ("AC ground (measure cost)…",
             lambda: self._acg_from_net(full)),
            (f"Set as impedance target ({local} to gnd)",
             lambda: self._impedance_at_nets(full, "")),
        ]
        others = [o for o in others if o != full and o not in gnd]
        if len(others) == 1:
            o = others[0]
            acts.append((f"Set as impedance target (between {local} "
                         f"and {o.split('.')[-1]})",
                         lambda: self._impedance_at_nets(full, o)))
        return acts

    def _schematic_net_menu(self, local: str, gpos):
        """Right-click on a wire: select the net (so the mark shows
        what the menu applies to), then the Nets tree's own verbs. A
        net selected beforehand -- click one wire, right-click the
        other -- stays selected and adds Z between the pair."""
        full = next((c for c in (self._schematic_prefix() + local, local)
                     if self.nets_tree.item_for(c) is not None), None)
        if full is None:
            return
        others = [n for n in self.nets_tree.selected_nets()
                  if n != full]
        self._schematic_net_clicked(local)
        if len(others) == 1:
            prev = self.nets_tree.item_for(others[0])
            if prev is not None:
                prev.setSelected(True)        # the pair stays visible
        actions = self._schematic_net_actions(full, local, others)
        if not actions:
            return
        from PySide6.QtWidgets import QMenu

        menu = QMenu(self)
        for i, (label, fn) in enumerate(actions):
            if i == 2:
                menu.addSeparator()          # impedance verbs together
            menu.addAction(label, fn)
        menu.exec(gpos)

    def _schematic_deselect(self):
        """A click on the background clears the selection, here and in
        the Instances tree (the two are one selection)."""
        self.sch_view.highlight(None)
        self.sch_view.highlight_net(None)
        self.devices.clearSelection()
        self.nets_tree.clearSelection()

    def _schematic_follow_tree(self):
        """Tree selection -> highlight, when the device is on this page."""
        if not self._sch_hints:
            return
        names = self.devices.selected_names()
        pre = self._schematic_prefix()
        for full in names:
            if full.startswith(pre):
                local = full[len(pre):]
                if self.sch_view.bounds_of(local) is not None:
                    self.sch_view.highlight_net(None)
                    self.sch_view.highlight(local)
                    return
        self.sch_view.highlight(None)

    def _schematic_follow_nets(self):
        """Nets-tree selection -> the net's wires light up, when the
        net is on this page: the same cross-probe an instance gets.
        Passthrough names (supplies, globals) are never prefixed, so
        they match on any page."""
        if not self._sch_hints:
            return
        pre = self._schematic_prefix()
        for full in self.nets_tree.selected_nets():
            if full.startswith(pre) and len(full) > len(pre):
                local = full[len(pre):]
            elif "." not in full:
                local = full                  # vdd! and friends
            else:
                continue
            if "." not in local and self.sch_view._net_wires.get(local):
                self.sch_view.highlight(None)
                self.sch_view.highlight_net(local)
                return
        self.sch_view.highlight_net(None)
