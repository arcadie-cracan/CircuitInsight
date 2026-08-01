"""A two-handle range slider over a log-frequency axis.

Qt ships no double slider, and the band a budgeted solve is certified
for wants choosing the way engineers think about it -- graphically, in
decades, right above the Bode plot it applies to. The widget works in
log10(Hz) internally; its API speaks Hz. Each cursor carries its
frequency printed right above it, so no separate readout is needed.
"""
from __future__ import annotations

import math

from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

#: handle radius and groove thickness, px
_R = 7
_GROOVE = 4
#: room reserved above the groove for the per-cursor frequency labels
_LABEL_H = 14

#: exposed for layouts that align the groove with a plot axis: the
#: groove's active span is inset this many px from each widget edge
HANDLE_R = _R


def fmt_hz(hz: float) -> str:
    """Engineering-notation frequency, the way an axis tick reads."""
    for div, suffix in ((1e9, "GHz"), (1e6, "MHz"), (1e3, "kHz"),
                        (1.0, "Hz")):
        if hz >= div:
            return f"{hz / div:.3g} {suffix}"
    return f"{hz * 1e3:.3g} mHz"


class RangeSlider(QWidget):
    """Two cursors on one groove; the span between them is the selection.

    valuesChanged(fmin_hz, fmax_hz) fires on every user drag and on
    programmatic setValues, so one slot keeps plot overlays live."""

    valuesChanged = Signal(float, float)

    def __init__(self, fmin: float = 1.0, fmax: float = 1e10, parent=None):
        super().__init__(parent)
        self._lo_log = math.log10(fmin)
        self._hi_log = math.log10(fmax)
        self._a = self._lo_log            # handle positions, log10 Hz
        self._b = self._hi_log
        self._drag = None                 # "a" | "b" | None
        self.setMinimumHeight(2 * _R + _LABEL_H + 4)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setCursor(Qt.PointingHandCursor)

    # ------------------------------------------------------------- API (Hz)
    def setRange(self, fmin: float, fmax: float) -> None:
        self._lo_log = math.log10(fmin)
        self._hi_log = math.log10(fmax)
        self._a = min(max(self._a, self._lo_log), self._hi_log)
        self._b = min(max(self._b, self._lo_log), self._hi_log)
        self.update()

    def range(self) -> tuple[float, float]:
        return 10.0 ** self._lo_log, 10.0 ** self._hi_log

    def setValues(self, fmin: float, fmax: float) -> None:
        a = min(max(math.log10(fmin), self._lo_log), self._hi_log)
        b = min(max(math.log10(fmax), self._lo_log), self._hi_log)
        self._a, self._b = min(a, b), max(a, b)
        self.update()
        self.valuesChanged.emit(*self.values())

    def values(self) -> tuple[float, float]:
        return 10.0 ** self._a, 10.0 ** self._b

    def labels(self) -> tuple[str, str]:
        """The two cursor captions as painted, for tests and tooltips."""
        return fmt_hz(10.0 ** self._a), fmt_hz(10.0 ** self._b)

    # ------------------------------------------------------------ geometry
    def _y(self) -> float:
        return _LABEL_H + _R + 1          # groove centerline

    def _x_of(self, v: float) -> float:
        w = self.width() - 2 * _R
        span = self._hi_log - self._lo_log
        return _R + w * (v - self._lo_log) / span

    def _v_of(self, x: float) -> float:
        w = max(1, self.width() - 2 * _R)
        span = self._hi_log - self._lo_log
        v = self._lo_log + span * (x - _R) / w
        return min(max(v, self._lo_log), self._hi_log)

    # ------------------------------------------------------------- painting
    def paintEvent(self, _ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        y = self._y()
        xa, xb = self._x_of(self._a), self._x_of(self._b)

        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#c9c9c9"))                       # groove
        p.drawRoundedRect(QRectF(_R, y - _GROOVE / 2,
                                 self.width() - 2 * _R, _GROOVE), 2, 2)
        p.setBrush(QColor("#4a78a8"))                       # selected band
        p.drawRoundedRect(QRectF(xa, y - _GROOVE / 2, xb - xa, _GROOVE),
                          2, 2)
        p.setPen(QPen(QColor("#2d5379"), 1.2))
        p.setBrush(QColor("#f4f7fa"))
        for x in (xa, xb):                                  # handles
            p.drawEllipse(QRectF(x - _R, y - _R, 2 * _R, 2 * _R))

        # the frequencies, right above their cursors; nudged apart when
        # the handles meet so both stay readable
        f = p.font()
        f.setPointSizeF(max(7.0, f.pointSizeF() - 1.5))
        p.setFont(f)
        fm = p.fontMetrics()
        la, lb = self.labels()
        wa, wb = fm.horizontalAdvance(la), fm.horizontalAdvance(lb)
        xla = min(max(xa - wa / 2, 0.0), self.width() - wa)
        xlb = min(max(xb - wb / 2, 0.0), self.width() - wb)
        if xla + wa + 4 > xlb:
            xla = max(0.0, xlb - wa - 4)
        p.setPen(QColor("#2d5379"))
        p.drawText(QRectF(xla, 0, wa + 2, _LABEL_H), Qt.AlignLeft, la)
        p.drawText(QRectF(xlb, 0, wb + 2, _LABEL_H), Qt.AlignLeft, lb)

    # ---------------------------------------------------------------- mouse
    def mousePressEvent(self, ev):
        x = ev.position().x()
        self._drag = ("a" if abs(x - self._x_of(self._a))
                      <= abs(x - self._x_of(self._b)) else "b")
        self.mouseMoveEvent(ev)

    def mouseMoveEvent(self, ev):
        if self._drag is None:
            return
        v = self._v_of(ev.position().x())
        if self._drag == "a":
            self._a = min(v, self._b)
        else:
            self._b = max(v, self._a)
        self.update()
        self.valuesChanged.emit(*self.values())

    def mouseReleaseEvent(self, _ev):
        self._drag = None
