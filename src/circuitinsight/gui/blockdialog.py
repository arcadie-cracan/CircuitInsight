"""The block-symbol dialog: confirm how a subcircuit cell is drawn.

Opened from the Schematic pane on a block (right-click → Symbol…).
Shows the library's block-capable symbols with a preview, a table of
the cell's ports with the proposal's evidence per row and an anchor
chooser per port (an anchor of the chosen symbol, or a stub edge), and
validates inline: an anchor used twice, or no anchor at all, blocks OK
with the reason. OK writes the mapping CONFIRMED to the sidecar beside
the CIN; "Draw as box" removes the confirmation. Nothing is applied
until the user presses OK -- the proposal is a suggestion on screen.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtSvgWidgets import QSvgWidget
from PySide6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox,
                               QHBoxLayout, QLabel, QPushButton,
                               QTableWidget, QTableWidgetItem, QVBoxLayout)

from ..schematic.blocks import BlockSymbol
from ..schematic.compose import _strip_equations

STUB_EDGES = ("auto", "top", "bottom", "left", "right")


class BlockSymbolDialog(QDialog):
    """Returns, after exec(): self.result_mapping (BlockSymbol with
    confirmed=True) or None; self.draw_as_box True when the user asked
    for the plain box."""

    def __init__(self, cell: str, ports: list, lib, proposal=None,
                 parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Symbol for {cell}")
        self.cell = cell
        self.ports = list(ports)
        self.lib = lib
        self.proposal = proposal
        self.result_mapping = None
        self.draw_as_box = False

        v = QVBoxLayout(self)
        head = QLabel(
            f"<b>{cell}</b> — every instance of this cell draws the "
            f"same way. Proposed from the netlist; nothing is applied "
            f"until OK.")
        head.setWordWrap(True)
        v.addWidget(head)

        row = QHBoxLayout()
        row.addWidget(QLabel("symbol:"))
        self.sym_combo = QComboBox()
        for f in lib.block_candidates():
            self.sym_combo.addItem(f)
        row.addWidget(self.sym_combo, 1)
        v.addLayout(row)
        prow = QHBoxLayout()
        self.preview = QSvgWidget()
        self.preview.setFixedHeight(120)
        prow.addStretch(1)
        prow.addWidget(self.preview)
        prow.addStretch(1)
        v.addLayout(prow)

        self.table = QTableWidget(len(self.ports), 3)
        self.table.setHorizontalHeaderLabels(["port", "anchor", "evidence"])
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setStretchLastSection(True)
        self._combos: dict = {}
        for i, port in enumerate(self.ports):
            it = QTableWidgetItem(port)
            it.setFlags(it.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(i, 0, it)
            cb = QComboBox()
            self._combos[port] = cb
            self.table.setCellWidget(i, 1, cb)
            ev = "; ".join((proposal.evidence.get(port, [])
                            if proposal else []))
            e = QTableWidgetItem(ev)
            e.setFlags(e.flags() & ~Qt.ItemIsEditable)
            e.setToolTip(ev)
            self.table.setItem(i, 2, e)
        v.addWidget(self.table, 1)

        self.verdict = QLabel("")
        self.verdict.setWordWrap(True)
        v.addWidget(self.verdict)

        btns = QDialogButtonBox(QDialogButtonBox.Ok
                                | QDialogButtonBox.Cancel)
        self.box_btn = QPushButton("Draw as box")
        self.box_btn.setToolTip("Remove the confirmation: the cell draws "
                                "as the plain harvested box again")
        btns.addButton(self.box_btn, QDialogButtonBox.ResetRole)
        self.box_btn.clicked.connect(self._as_box)
        btns.accepted.connect(self._accept)
        btns.rejected.connect(self.reject)
        self.ok_btn = btns.button(QDialogButtonBox.Ok)
        v.addWidget(btns)

        self.sym_combo.currentTextChanged.connect(self._symbol_changed)
        if proposal and proposal.symbol:
            ix = self.sym_combo.findText(proposal.symbol)
            if ix >= 0:
                self.sym_combo.setCurrentIndex(ix)
        self._symbol_changed(self.sym_combo.currentText())

    # ---------------------------------------------------------- filling
    def _symbol_changed(self, filename: str):
        if not filename:
            return
        path = Path(self.lib.root) / filename
        if path.exists():
            # the drawing without the library's designator placeholder,
            # at the symbol's own aspect (QSvgWidget stretches to fit)
            data = _strip_equations(path.read_text(encoding="utf-8"))
            self.preview.load(data.encode("utf-8"))
            ds = self.preview.renderer().defaultSize()
            if ds.height() > 0:
                self.preview.setFixedSize(
                    max(40, int(120 * ds.width() / ds.height())), 120)
        anchors = list(self.lib.anchors_of(filename))
        for port, cb in self._combos.items():
            cb.blockSignals(True)
            cb.clear()
            for a in anchors:
                cb.addItem(a)
            for edge in STUB_EDGES:
                cb.addItem(f"stub: {edge}")
            want = None
            if self.proposal and filename == self.proposal.symbol:
                want = self.proposal.pins.get(port)
                if want is None and port in self.proposal.stubs:
                    want = f"stub: {self.proposal.stubs[port]}"
            if want is None:
                want = "stub: auto"
            ix = cb.findText(want)
            cb.setCurrentIndex(ix if ix >= 0 else cb.count() - 5)
            cb.blockSignals(False)
            cb.currentTextChanged.connect(lambda _t: self._validate())
        self._validate()

    def current(self) -> BlockSymbol:
        pins, stubs = {}, {}
        for port, cb in self._combos.items():
            txt = cb.currentText()
            if txt.startswith("stub: "):
                stubs[port] = txt[len("stub: "):]
            else:
                pins[port] = txt
        ev = dict(self.proposal.evidence) if self.proposal else {}
        return BlockSymbol(symbol=self.sym_combo.currentText(), pins=pins,
                           stubs=stubs, confirmed=True, evidence=ev)

    def _validate(self):
        bs = self.current()
        problems = []
        used: dict = {}
        for port, an in bs.pins.items():
            used.setdefault(an, []).append(port)
        for an, ps in used.items():
            if len(ps) > 1:
                problems.append(f"anchor {an} used by {', '.join(ps)}")
        if not bs.pins:
            problems.append("no port is on an anchor — that is a box")
        free = [a for a in self.lib.anchors_of(bs.symbol)
                if a not in used]
        note = f"unused anchors: {', '.join(free)}" if free else ""
        if problems:
            self.verdict.setText("✗ " + "; ".join(problems))
            self.ok_btn.setEnabled(False)
        else:
            self.verdict.setText("✓ " + (note or "every anchor has a port")
                                 + f"; {len(bs.stubs)} stub(s)")
            self.ok_btn.setEnabled(True)

    # ----------------------------------------------------------- actions
    def _accept(self):
        self.result_mapping = self.current()
        self.accept()

    def _as_box(self):
        self.draw_as_box = True
        self.result_mapping = None
        self.accept()
