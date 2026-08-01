"""The circuit as two trees: Instances and Nets.

The flat device table hid the one thing a hierarchy makes obvious --
which subcircuit owns what -- and it had no place for the nets at all.
The Instances tree groups devices under their subcircuit path and wears
the analysis state as decorations: a match set is a link mark with its
group number on every member, an impact-ionization device keeps its red
warning. Hovering an instance shows the operating-point glance; the
full record stays on double-click. The Nets tree is the circuit's other
half -- every net with its connections -- and tells the reduction truth:
an AC-grounded net carries the earth mark, the input and output nets
carry their arrows. Both trees act through signals only; nothing here
mutates the session.
"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QAbstractItemView, QMenu, QTreeWidget, QTreeWidgetItem,
)

from . import view

LINK = "\U0001F517"          # match-set decoration
EARTH = "⏚"             # AC-grounded net
IN_MARK = "▶"           # input source's net
OUT_MARK = "◀"          # output net

#: light, colorblind-friendly member tints for match groups
GROUP_TINTS = ("#dbe9f6", "#fbe4d5", "#e2efda", "#ece1f0",
               "#fff2cc", "#dbeef4")

_ROLE_NAME = Qt.UserRole             # full device / net name
_ROLE_KIND = Qt.UserRole + 1         # "device" | "net" | "conn"

#: the OP parameters worth a glance, in display order; everything else
#: stays in the double-click record
_GLANCE = ("gm", "gds", "gmbs", "ids", "vgs", "vds", "vdsat", "vth")
_UNITS = {"v": "V", "i": "A", "g": "S", "c": "F", "q": "C", "r": "Ω"}


def _branch_style(widget) -> None:
    """Draw the actual branch lines. Fusion paints expand arrows but no
    connectors; the classic Windows style draws the dotted tree lines,
    and QStyleFactory ships it on every platform Qt6 supports."""
    from PySide6.QtWidgets import QStyleFactory

    st = QStyleFactory.create("windows")
    if st is not None:
        st.setParent(widget)                  # keep it alive with the tree
        widget.setStyle(st)


class InstanceTree(QTreeWidget):
    """Devices grouped by subcircuit path, decorated with match sets.

    One column: the hierarchy itself. Everything else -- type, region,
    gm/gds, terminals -- lives in the hover glance; the full record on
    double-click. Emits instead of acting: `deviceActivated(name)` on
    double-click, `aliasRequested(name)` / `unmatchRequested(gi)` /
    `representativeRequested(gi)` / `matchRequested()` from the
    context menu."""

    aliasRequested = Signal(str)
    deviceActivated = Signal(str)
    unmatchRequested = Signal(int)
    representativeRequested = Signal(int)
    matchRequested = Signal()

    NAME = 0
    _ROLE_INFO = Qt.UserRole + 2
    _ROLE_GLANCE = Qt.UserRole + 3

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setColumnCount(1)
        self.setHeaderLabels(["instance"])
        _branch_style(self)
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.setMouseTracking(True)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._menu)
        self.itemEntered.connect(self._hover)
        self.itemDoubleClicked.connect(self._dclick)
        self._filling = False
        self._groups: list[tuple[str, ...]] = []
        self._by_name: dict[str, QTreeWidgetItem] = {}
        self._op_fetch = None

    # ----------------------------------------------------------- population
    def populate(self, rows, op_fetch=None) -> None:
        """rows: dicts with name, type, region, gm, gds, alias, terminals
        and an optional ii_note -- kept as the item's info record (see
        `info()`) and shown in the hover glance; op_fetch: name -> OP
        record dict, fetched lazily on first hover."""
        self._filling = True
        try:
            self.clear()
            self._by_name.clear()
            self._op_fetch = op_fetch
            parents: dict[tuple, QTreeWidgetItem] = {}

            def parent_for(path: tuple):
                if not path:
                    return self.invisibleRootItem()
                if path not in parents:
                    up = parent_for(path[:-1])
                    it = QTreeWidgetItem(up, [path[-1]])
                    it.setFlags(Qt.ItemIsEnabled)      # container, not device
                    it.setData(0, _ROLE_KIND, "sub")
                    parents[path] = it
                return parents[path]

            for r in rows:
                parts = r["name"].split(".")
                it = QTreeWidgetItem(parent_for(tuple(parts[:-1])))
                it.setText(self.NAME, parts[-1])
                it.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                it.setData(0, _ROLE_NAME, r["name"])
                it.setData(0, _ROLE_KIND, "device")
                it.setData(0, self._ROLE_INFO, dict(r))
                if r.get("ii_note"):
                    it.setForeground(self.NAME, QBrush(QColor("#8a1c12")))
                    it.setToolTip(self.NAME, r["ii_note"])
                elif r.get("region") and r["region"] != "sat":
                    # outside saturation: worth a mark even with the
                    # region column gone -- the glance names the region
                    it.setForeground(self.NAME, QBrush(QColor("#b06000")))
                self._by_name[r["name"]] = it
            self.expandAll()
        finally:
            self._filling = False

    # ------------------------------------------------------------ accessors
    def leaf_items(self) -> list:
        out = []

        def walk(it):
            for i in range(it.childCount()):
                ch = it.child(i)
                if ch.data(0, _ROLE_KIND) == "device":
                    out.append(ch)
                walk(ch)
        walk(self.invisibleRootItem())
        return out

    def item_for(self, name: str):
        return self._by_name.get(name)

    def device_name(self, item) -> str | None:
        return item.data(0, _ROLE_NAME) if item is not None else None

    def selected_names(self) -> list[str]:
        return [n for n in (self.device_name(it)
                            for it in self.selectedItems()) if n]

    def group_of(self, name: str) -> int | None:
        for gi, g in enumerate(self._groups):
            if name in g:
                return gi
        return None

    # ---------------------------------------------------------- decorations
    def set_groups(self, groups) -> None:
        """Match sets as decorations: every member wears the link mark
        with its group number, plus the group tint on all columns."""
        self._groups = [tuple(g) for g in groups]
        self._filling = True
        try:
            for name, it in self._by_name.items():
                gi = self.group_of(name)
                leaf = name.split(".")[-1]
                if gi is None:
                    it.setText(self.NAME, leaf)
                    brush = QBrush()
                else:
                    it.setText(self.NAME, f"{leaf} {LINK}{gi + 1}")
                    brush = QBrush(QColor(
                        GROUP_TINTS[gi % len(GROUP_TINTS)]))
                for col in range(self.columnCount()):
                    it.setBackground(col, brush)
        finally:
            self._filling = False

    def info(self, item) -> dict:
        """The populate-time record: type, region, gm, gds, alias,
        terminals, ii_note. The columns this tree no longer has."""
        return dict(item.data(0, self._ROLE_INFO) or {})

    # ----------------------------------------------------------- interaction
    def _dclick(self, item, _col):
        name = self.device_name(item)
        if name:
            self.deviceActivated.emit(name)

    def _hover(self, item, _col):
        """The OP glance, fetched on first hover and cached on the item."""
        name = self.device_name(item)
        if not name or self._op_fetch is None:
            return
        if item.data(0, self._ROLE_GLANCE):
            return                                    # already built
        info = self.info(item)
        try:
            rec = dict(self._op_fetch(name) or {})
        except Exception:
            rec = {}
        lines = [f"<b>{name}</b> — {info.get('type', '')}"
                 + (f", {view.region_name(rec['region'])}"
                    if "region" in rec else "")]
        terms = info.get("terminals") or {}
        if terms:
            lines.append("  ".join(f"{t}→{n}" for t, n in terms.items()))
        vals = []
        for key in _GLANCE:
            if key in rec:
                try:
                    unit = _UNITS.get(key[:1].lower(), "")
                    vals.append(f"{key} = {view.eng(float(rec[key]), unit)}")
                except (TypeError, ValueError):
                    pass
        for i in range(0, len(vals), 4):
            lines.append("   ".join(vals[i:i + 4]))
        tip = "<br>".join(lines)
        ii = info.get("ii_note", "")
        if ii:
            tip += f"<br><i>{ii}</i>"
        item.setToolTip(self.NAME, tip)
        item.setData(0, self._ROLE_GLANCE, True)

    def _menu(self, pos):
        item = self.itemAt(pos)
        name = self.device_name(item)
        menu = QMenu(self)
        if len(self.selected_names()) >= 2:
            menu.addAction("Match selected devices",
                           lambda: self.matchRequested.emit())
        if name is not None:
            menu.addAction(
                "Set LaTeX alias…",
                lambda name=name: self.aliasRequested.emit(name))
            gi = self.group_of(name)
            if gi is not None:
                menu.addAction(
                    f"Unmatch {LINK}{gi + 1}",
                    lambda gi=gi: self.unmatchRequested.emit(gi))
                menu.addAction(
                    f"Choose representative for {LINK}{gi + 1}…",
                    lambda gi=gi: self.representativeRequested.emit(gi))
        if menu.actions():
            menu.exec(self.viewport().mapToGlobal(pos))


class NetTree(QTreeWidget):
    """Every net with its connections; decorations tell the truth about
    the working circuit: the earth mark on AC-grounded nets, arrows on
    the input source's net and the output net. Double-click a net to
    make it the output; the context menu routes AC-grounding through
    the measured Reduce flow."""

    outputRequested = Signal(str)
    acgroundRequested = Signal(str)
    connectionActivated = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setColumnCount(1)
        self.setHeaderLabels(["net / connections"])
        _branch_style(self)
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._menu)
        self.itemDoubleClicked.connect(self._dclick)
        self._by_net: dict[str, QTreeWidgetItem] = {}
        self._ground: set[str] = set()
        self._decor = {"acg": set(), "inp": None, "out": None}

    def populate(self, connections: dict, ground=()) -> None:
        """connections: net -> [(instance, terminal)]. Hierarchical like
        the Instances tree: the testbench's own nets (vout, vin_dm)
        come first at the root, then each subcircuit is a container
        node (I0) holding its internal nets (net19), ground grayed at
        the bottom so students see where 0 is."""
        self.clear()
        self._by_net.clear()
        self._ground = set(ground)
        signal = sorted((n for n in connections if n not in self._ground),
                        key=lambda n: ("." in n, n))
        gnd = sorted(n for n in connections if n in self._ground)
        parents: dict[tuple, QTreeWidgetItem] = {}

        def parent_for(path: tuple):
            if not path:
                return self.invisibleRootItem()
            if path not in parents:
                up = parent_for(path[:-1])
                it = QTreeWidgetItem(up, [path[-1]])
                it.setFlags(Qt.ItemIsEnabled)          # container, not a net
                it.setData(0, _ROLE_KIND, "sub")
                parents[path] = it
            return parents[path]

        for net in signal + gnd:
            parts = net.split(".")
            it = QTreeWidgetItem(parent_for(tuple(parts[:-1])))
            it.setText(0, parts[-1])
            it.setData(0, _ROLE_NAME, net)
            it.setData(0, _ROLE_KIND, "net")
            if net in self._ground:
                it.setForeground(0, QBrush(QColor("#999999")))
            for inst, term in sorted(connections[net]):
                ch = QTreeWidgetItem(it, [f"{inst} · {term}"])
                ch.setData(0, _ROLE_NAME, inst)
                ch.setData(0, _ROLE_KIND, "conn")
                ch.setForeground(0, QBrush(QColor("#666666")))
            self._by_net[net] = it
        for it in parents.values():
            it.setExpanded(True)          # show each subcircuit's nets;
        self._apply_decor()               # connections stay folded

    def item_for(self, net: str):
        return self._by_net.get(net)

    def net_for(self, item) -> str | None:
        if item is not None and item.data(0, _ROLE_KIND) == "net":
            return item.data(0, _ROLE_NAME)
        return None

    def set_decorations(self, *, acg=(), inp=None, out=None) -> None:
        self._decor = {"acg": set(acg), "inp": inp, "out": out}
        self._apply_decor()

    def _apply_decor(self):
        d = self._decor
        for net, it in self._by_net.items():
            marks = []
            if net in d["acg"]:
                marks.append(EARTH)
            if net == d["inp"]:
                marks.append(IN_MARK)
            if net == d["out"]:
                marks.append(OUT_MARK)
            leaf = net.split(".")[-1]
            it.setText(0, (" ".join(marks) + " " if marks else "") + leaf)
            if net in d["acg"]:
                it.setToolTip(0, "AC-grounded by the active reduction — "
                              "every analysis runs on the reduced circuit")
            elif marks:
                it.setToolTip(0, "input source net" if net == d["inp"]
                              else "output net")
            else:
                it.setToolTip(0, "")

    def _dclick(self, item, _col):
        kind = item.data(0, _ROLE_KIND)
        if kind == "conn":
            self.connectionActivated.emit(item.data(0, _ROLE_NAME))
        elif kind == "net" and item.data(0, _ROLE_NAME) not in self._ground:
            self.outputRequested.emit(item.data(0, _ROLE_NAME))

    def _menu(self, pos):
        item = self.itemAt(pos)
        net = self.net_for(item)
        if net is None or net in self._ground:
            return
        menu = QMenu(self)
        menu.addAction("Set as output",
                       lambda: self.outputRequested.emit(net))
        menu.addAction("AC ground (measure cost)…",
                       lambda: self.acgroundRequested.emit(net))
        menu.exec(self.viewport().mapToGlobal(pos))
