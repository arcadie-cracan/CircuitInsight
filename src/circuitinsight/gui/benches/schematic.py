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
import os
from pathlib import Path

from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QPainter, QPen
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtSvgWidgets import QGraphicsSvgItem
from PySide6.QtWidgets import (QCheckBox, QFileDialog, QGraphicsRectItem,
                               QGraphicsScene, QGraphicsView, QHBoxLayout,
                               QLabel, QPushButton, QToolButton,
                               QVBoxLayout, QWidget)

from ...schematic import (ComposeError, SymbolLibrary, compose,
                          confirmed_only, load_block_symbols, load_hints,
                          page, sidecar_path)
from ...schematic.compose import (NET_ACGROUND, STYLE_LUMPED,
                                  STYLE_NUMERIC, STYLE_REMOVED,
                                  STYLE_SYMBOLIC, _xml_id)


def hints_path(cin_path) -> Path:
    """``tb_fc.cin.json`` -> ``tb_fc.hints.json`` beside it."""
    p = Path(cin_path)
    stem = p.name[:-len(".cin.json")] if p.name.endswith(".cin.json") \
        else p.stem
    return p.with_name(stem + ".hints.json")


class SchematicView(QGraphicsView):
    """Zoomable, pannable SVG with instance hit-testing."""
    instanceClicked = Signal(str, str)      # name, kind
    instanceActivated = Signal(str, str)    # double-click

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHints(QPainter.Antialiasing
                            | QPainter.SmoothPixmapTransform
                            | QPainter.TextAntialiasing)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        # the schematic is a paper drawing: black ink on white, in
        # every theme (the symbol library is drawn for white)
        self.setBackgroundBrush(QBrush(QColor("#ffffff")))
        # the item OWNS its renderer: a renderer parented elsewhere
        # and shared into the scene crashed at teardown (undefined
        # destruction order between view children and scene items)
        self._renderer: QSvgRenderer | None = None
        self._item: QGraphicsSvgItem | None = None
        self._bounds: dict = {}          # name -> (QRectF item coords, kind)
        self._hl: QGraphicsRectItem | None = None
        self._sx = self._sy = 1.0

    # ----------------------------------------------------------- content
    def load_svg(self, svg: str, instances: dict) -> None:
        """instances: name -> kind (device/block/glyph)."""
        self._scene.clear()
        self._hl = None
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
            eid = _xml_id(name)
            if not self._renderer.elementExists(eid):
                continue
            b = self._renderer.boundsOnElement(eid)
            r = QRectF((b.x() - vb.x()) * self._sx,
                       (b.y() - vb.y()) * self._sy,
                       b.width() * self._sx, b.height() * self._sy)
            self._bounds[name] = (r, kind)
        self._scene.setSceneRect(br.adjusted(-10, -10, 10, 10))
        self.fit()

    def fit(self) -> None:
        if self._item is not None:
            self.fitInView(self._item, Qt.KeepAspectRatio)

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
        self._hl = self._scene.addRect(
            sr, QPen(QColor("#d55e00"), 0.6),
            QBrush(QColor(213, 94, 0, 40)))
        self._hl.setZValue(10)

    def bounds_of(self, name: str):
        return self._bounds.get(name, (None, None))[0]

    # ------------------------------------------------------------ events
    def wheelEvent(self, ev):
        f = 1.15 if ev.angleDelta().y() > 0 else 1 / 1.15
        self.scale(f, f)

    def mousePressEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            h = self.hit(ev.position().toPoint())
            if h:
                self.instanceClicked.emit(*h)
        super().mousePressEvent(ev)

    def mouseDoubleClickEvent(self, ev):
        h = self.hit(ev.position().toPoint())
        if h:
            self.instanceActivated.emit(*h)
        super().mouseDoubleClickEvent(ev)


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
        self._sch_state_chk = QCheckBox("analysis state")
        self._sch_state_chk.setChecked(True)
        self._sch_state_chk.setToolTip(
            "Draw the session's state on the layout: AC-grounded nets "
            "gray-dashed with an earth, removed sources faded, ĝm-lumped "
            "devices badged, devices whose symbol is in the shown formula "
            "emphasized. Off: the plain drawing.")
        self._sch_state_chk.toggled.connect(
            lambda _on: self._schematic_render())
        row.addWidget(self._sch_state_chk)
        fitb = QPushButton("Fit")
        fitb.setToolTip("Fit the page in the view (wheel zooms, drag pans)")
        fitb.clicked.connect(lambda: self.sch_view.fit())
        row.addWidget(fitb)
        libb = QPushButton("Library…")
        libb.setToolTip("Choose the SVG symbol library directory (the one "
                        "with terminals.csv); remembered across sessions")
        libb.clicked.connect(self._schematic_pick_library)
        row.addWidget(libb)
        v.addLayout(row)
        self.sch_view = SchematicView()
        self.sch_view.instanceClicked.connect(self._schematic_clicked)
        self.sch_view.instanceActivated.connect(self._schematic_activated)
        v.addWidget(self.sch_view, 1)
        self._sch_hints = None
        self._sch_cin = None
        self._sch_lib = None
        self._sch_blocks = {}
        self._sch_path: list = []        # [(instance name, defname)]
        self._sch_synced = False
        return pagew

    # ------------------------------------------------------------ loading
    def _schematic_library_root(self):
        root = str(self._settings().value("symlib", "") or "")
        return root or os.environ.get("CIN_SYMLIB") or None

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
                "CIN_SYMLIB)")
            self._schematic_crumbs()
            return
        try:
            self._sch_hints = load_hints(hp)
            self._sch_cin = json.loads(Path(cin).read_text(encoding="utf-8"))
            self._sch_lib = SymbolLibrary(root)
            self._sch_blocks = confirmed_only(
                load_block_symbols(sidecar_path(cin)))
        except Exception as e:                   # noqa: BLE001
            self._sch_status.setText(f"schematic unavailable: {e}")
            self._schematic_crumbs()
            return
        top = self._sch_hints.top
        self._sch_path = [(None, top)]
        if not self._sch_synced:
            self.devices.itemSelectionChanged.connect(
                self._schematic_follow_tree)
            self._sch_synced = True
        self._schematic_render()

    def _schematic_prefix(self) -> str:
        names = [n for n, _ in self._sch_path if n]
        return ".".join(names) + "." if names else ""

    def _schematic_render(self):
        if not self._sch_hints or not self._sch_path:
            return
        defname = self._sch_path[-1][1]
        cdef = (self._sch_cin.get("definitions", {}).get(defname)
                or {"instances": []})
        devtypes = {i["name"]: (i["device_type"],
                                (i.get("params") or {}).get("polarity"))
                    for i in cdef["instances"] if "device_type" in i}
        subckts = frozenset(self._sch_cin.get("definitions", {}))
        styles, net_styles = ({}, {})
        if self._sch_state_chk.isChecked():
            try:
                styles, net_styles = self._schematic_styles()
            except Exception:                     # noqa: BLE001
                styles, net_styles = ({}, {})     # decoration only
        try:
            pg = page(self._sch_hints, defname)
            svg = compose(pg, devtypes, self._sch_lib, subckts=subckts,
                          block_symbols=self._sch_blocks,
                          styles=styles, net_styles=net_styles)
        except (ComposeError, KeyError, ValueError) as e:
            self._sch_status.setText(f"{defname}: {e}")
            self._schematic_crumbs()
            return
        insts = dict(getattr(compose, "last_instances", {}))
        kinds = {n: ("block" if c in subckts else "device")
                 for n, c in insts.items()}
        self.sch_view.load_svg(svg, kinds)
        n_blk = sum(1 for k in kinds.values() if k == "block")
        bits = [f"{len(insts)} instances, {len(pg.wires)} wires"]
        if n_blk:
            bits.append(f"{n_blk} block(s) — double-click to descend")
        n_acg = sum(1 for v in net_styles.values() if v == NET_ACGROUND)
        n_rm = sum(1 for v in styles.values() if v == STYLE_REMOVED)
        n_sym = sum(1 for v in styles.values() if v == STYLE_SYMBOLIC)
        n_num = sum(1 for v in styles.values() if v == STYLE_NUMERIC)
        n_lump = sum(1 for v in styles.values() if v == STYLE_LUMPED)
        state = [f"{n} {what}" for n, what in (
            (n_acg, "net(s) AC-grounded"), (n_rm, "removed"),
            (n_lump, "ĝm-lumped"), (n_sym, "in the formula"),
            (n_num, "numeric")) if n]
        if state:
            bits.append("; ".join(state))
        self._sch_status.setText(f"{defname}: " + ", ".join(bits))
        self._schematic_crumbs()

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
            for d in summ.get("dead_sources", []):
                ld = local(d)
                if ld:
                    styles[ld] = STYLE_REMOVED
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
                if not ld or styles.get(ld) == STYLE_REMOVED:
                    continue
                key = d.name.replace(".", "_")
                if d.name in present or any(
                        s.endswith("_" + key) for s in present):
                    styles[ld] = STYLE_SYMBOLIC
                elif ld not in styles:
                    styles[ld] = STYLE_NUMERIC
        return styles, nets

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
            b = QToolButton()
            b.setText(defname if name is None else f"{name} ({defname})")
            b.setAutoRaise(True)
            last = i == len(self._sch_path) - 1
            b.setEnabled(not last)
            b.clicked.connect(lambda _=False, k=i: self._schematic_up(k))
            self._sch_crumb.addWidget(b)
            if not last:
                self._sch_crumb.addWidget(QLabel("›"))

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
        self.sch_view.highlight(name)
        if kind == "device":
            self._goto_instance(self._schematic_prefix() + name)

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
                    self.sch_view.highlight(local)
                    return
        self.sch_view.highlight(None)
