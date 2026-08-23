"""The operating-point card: one device, read the way a designer reads it.

Follows the current device from every view (Instances tree, schematic,
expression symbol, keep-table row) and shows, persistently, what the
hover glance and the double-click record only offered in passing:
region with the margins that matter (V_DS - V_DSAT, V_OV), the bias
point, the small-signal parameters WITH the derived figures (gm/I_D,
r_o, intrinsic gain, f_T), and what the working model actually uses for
each symbol after matching, lumping and reduction. The full dcOpInfo
record stays one click away.
"""
from __future__ import annotations

import math

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QGroupBox, QHBoxLayout, QLabel, QPushButton,
                               QVBoxLayout, QWidget)

from . import theme
from .view import eng, op_unit, region_name

THIN_MARGIN_V = 0.05          # V_DS - V_DSAT under this reads amber


def _f(v, unit=""):
    return eng(v, unit) if v is not None else "—"


def op_card_rows(controller, name: str, result=None, groups=(),
                 policy: str = "") -> list:
    """[(section, [(label, value, note)])] for the device `name`, or []
    when the session does not know it. Pure data; the widget renders
    it, the tests read it."""
    dev = next((d for d in controller.devices if d.name == name), None)
    if dev is None:
        return []
    op = controller.device_op(name) or {}
    model = controller.model_values(name)
    key = name.replace(".", "_")
    present = set()
    tf = getattr(result, "tf", None)
    expr = getattr(tf, "expr", None)
    if expr is not None and hasattr(expr, "free_symbols"):
        present = {str(s) for s in expr.free_symbols}
    sections = []

    # identity
    ident = [("type", dev.device_type, "")]
    grp = next((g for g in groups if name in g), None)
    if grp:
        gi = list(groups).index(grp) + 1
        others = [m for m in grp if m != name]
        ident.append((f"match 🔗{gi}", ", ".join(others),
                      f"values: {policy}" if policy else ""))
    try:
        lump = controller.lumped_gmb().get(name)
    except Exception:                            # noqa: BLE001
        lump = None
    if lump:
        ident.append(("ĝm", lump, ""))
    sections.append(("device", ident))

    if dev.device_type in ("mosfet", "bjt") and op:
        # region and margins
        reg = region_name(op.get("region"))
        vds, vdsat = op.get("vds"), op.get("vdsat")
        vgs, vth = op.get("vgs"), op.get("vth")
        rows = [("region", reg or "—", "")]
        if vds is not None and vdsat is not None:
            m = abs(vds) - abs(vdsat)
            rows.append(("V_DS − V_DSAT", _f(m, "V"),
                         "thin" if m < THIN_MARGIN_V else ""))
        if vgs is not None and vth is not None:
            rows.append(("V_OV = V_GS − V_TH", _f(abs(vgs) - abs(vth), "V"),
                         ""))
        sections.append(("region", rows))
        # bias
        sections.append(("bias", [
            ("I_D", _f(op.get("ids"), "A"), ""),
            ("V_GS", _f(op.get("vgs"), "V"), ""),
            ("V_DS", _f(op.get("vds"), "V"), ""),
            ("V_TH", _f(op.get("vth"), "V"), ""),
            ("V_DSAT", _f(op.get("vdsat"), "V"), ""),
        ]))
        # small signal with the derived figures
        gm, gds, gmbs, ids = (op.get("gm"), op.get("gds"), op.get("gmbs"),
                              op.get("ids"))
        ss = [("g_m", _f(gm, "S"), ""), ("g_ds", _f(gds, "S"), ""),
              ("g_mbs", _f(gmbs, "S"), "")]
        if gm and ids:
            ss.append(("g_m / I_D", f"{gm / abs(ids):.3g} /V", ""))
        if gds:
            ss.append(("r_o = 1/g_ds", _f(1.0 / gds, "Ω"), ""))
        if gm and gds:
            ss.append(("g_m / g_ds", f"{gm / gds:.3g}",
                       f"{20 * math.log10(gm / gds):.1f} dB"))
        cgs, cgd = op.get("cgs"), op.get("cgd")
        if gm and cgs is not None:
            ct = abs(cgs) + abs(cgd or 0.0)
            if ct > 0:
                ss.append(("f_T ≈ g_m/2π(c_gs+c_gd)",
                           _f(gm / (2 * math.pi * ct), "Hz"), ""))
        sections.append(("small-signal", ss))

    # what the model uses: per symbol, the value in the working model
    # against the device's own, and whether it is in the shown formula
    used = []
    folded = []                     # capacitances not in the formula
    order = {"gm": 0, "gmhat": 0, "gds": 1, "gmbs": 2, "gpi": 1, "go": 1}
    for param, v in sorted(model.items(),
                           key=lambda kv: (order.get(kv[0], 9), kv[0])):
        if dev.device_type in ("mosfet", "bjt"):
            sym = f"{param}_{key}"
            unit = op_unit(sym)
        else:
            sym = key                       # a passive's symbol is itself
            unit = {"c": "F", "r": "Ω", "g": "S", "l": "H"}.get(param, "")
        in_formula = sym in present
        first_order = param in order or dev.device_type not in ("mosfet",
                                                                 "bjt")
        if not in_formula and not first_order:
            folded.append((param, v))
            continue
        own = op.get(param)
        note = ""
        if own not in (None, 0) and v is not None and abs(v / own - 1) > 0.02:
            note = f"device's own {_f(own, unit)} ({v / own:.2f}×, match policy)"
        status = ("in the formula" if in_formula
                  else "numeric" if result is not None else "")
        used.append((sym, _f(v, unit),
                     " — ".join(x for x in (status, note) if x)))
    if folded:
        # the charge matrix / junction caps in one line: their count
        # and the two that matter most (the largest gate-side ones)
        top = sorted(folded, key=lambda kv: -abs(kv[1]))[:2]
        used.append((f"{len(folded)} capacitances",
                     ", ".join(f"{k} {_f(v, 'F')}" for k, v in top) + ", …",
                     "numeric" if result is not None else ""))
    if not model and dev.device_type not in ("vsource", "isource"):
        used.append(("(no live primitive)",
                     "lumped into an equivalent or reduced away", ""))
    sections.append(("as the model sees it", used))
    return sections


class OpCard(QGroupBox):
    """The card widget: a rich-text label that follows the selection."""

    def __init__(self, parent=None):
        super().__init__("Operating point", parent)
        # collapsed by default: the title carries the essentials
        # (device, region), one click opens the card; the trees above
        # keep their room
        self._inner = QWidget()
        v = QVBoxLayout(self._inner)
        v.setContentsMargins(4, 2, 4, 4)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(2, 2, 2, 2)
        outer.addWidget(self._inner)
        self.setCheckable(True)
        self.setChecked(False)
        self._inner.setVisible(False)
        self.toggled.connect(self._inner.setVisible)
        self.body = QLabel("select a device")
        self.body.setTextFormat(Qt.RichText)
        self.body.setWordWrap(True)
        self.body.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.body.setStyleSheet(f"color: {theme.MUTED};")
        self.body.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        # the card must not starve the trees above it: a capped,
        # scrolling body
        from PySide6.QtWidgets import QScrollArea
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setWidget(self.body)
        scroll.setMaximumHeight(240)
        v.addWidget(scroll)
        row = QHBoxLayout()
        row.addStretch(1)
        self.full_btn = QPushButton("full record…")
        self.full_btn.setFlat(True)
        self.full_btn.setEnabled(False)
        row.addWidget(self.full_btn)
        self.copy_btn = QPushButton("copy")
        self.copy_btn.setFlat(True)
        self.copy_btn.setEnabled(False)
        row.addWidget(self.copy_btn)
        v.addLayout(row)
        self._text = ""
        self.name = None
        self.copy_btn.clicked.connect(self._copy)

    def clear(self):
        self.name = None
        self._text = ""
        self.setTitle("Operating point")
        self.body.setText("select a device")
        self.body.setStyleSheet(f"color: {theme.MUTED};")
        self.full_btn.setEnabled(False)
        self.copy_btn.setEnabled(False)

    def show_sections(self, name: str, sections: list):
        self.name = name
        if not sections:
            self.clear()
            return
        # the title says what a glance needs: the device and its region
        reg = next((v for title, rows in sections if title == "region"
                    for lab, v, _ in rows if lab == "region"), "")
        self.setTitle(f"Operating point — {name}" + (f", {reg}" if reg
                                                      else ""))
        html = [f"<b>{name}</b>"]
        text = [name]
        for title, rows in sections:
            if not rows:
                continue
            html.append(f'<div style="color:{theme.INFO}; margin-top:3px">'
                        f'<b>{title}</b></div>')
            html.append('<table cellspacing="0" cellpadding="1">')
            text.append(f"[{title}]")
            for label, value, note in rows:
                col = theme.WARN if note == "thin" else theme.MUTED
                note_txt = ("thin margin" if note == "thin" else note)
                html.append(
                    f'<tr><td style="padding-right:8px">{label}</td>'
                    f'<td style="font-family:monospace; padding-right:8px">'
                    f'{value}</td>'
                    f'<td style="color:{col}">{note_txt}</td></tr>')
                text.append(f"  {label}: {value}"
                            + (f"  ({note_txt})" if note_txt else ""))
            html.append("</table>")
        self.body.setStyleSheet("")
        self.body.setText("".join(html))
        self._text = "\n".join(text)
        self.full_btn.setEnabled(True)
        self.copy_btn.setEnabled(True)

    def _copy(self):
        from PySide6.QtWidgets import QApplication
        QApplication.clipboard().setText(self._text)
