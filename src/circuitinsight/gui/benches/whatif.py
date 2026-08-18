"""The What-if bench: per-symbol value sliders re-evaluating the shown
model live."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QLabel, QVBoxLayout, QWidget)
from .. import theme
from .. import view

class WhatIfMixin:
    # ------------------------------------------------------------- what-if
    def _whatif_page(self):
        from PySide6.QtWidgets import QGridLayout

        page = QWidget()
        v = QVBoxLayout(page)
        self._wf_hint = QLabel(
            "Solve with a keep set to get sliders: each kept symbol can be "
            "swept ×0.25…"
            "×4 while the rest of the circuit stays "
            "EXACT at the operating point.")
        self._wf_hint.setWordWrap(True)
        self._wf_hint_default = self._wf_hint.text()
        v.addWidget(self._wf_hint)
        self._wf_grid_host = QWidget()
        self._wf_grid = QGridLayout(self._wf_grid_host)
        self._wf_grid.setContentsMargins(0, 0, 0, 0)
        v.addWidget(self._wf_grid_host)
        self._wf_pm = QLabel("")
        v.addWidget(self._wf_pm)
        v.addStretch(1)
        self._wf_sliders = {}
        self._wf_eval = None
        self._whatif_tab = page
        return page

    def _rebuild_whatif(self, result):
        from PySide6.QtWidgets import QSlider

        while self._wf_grid.count():
            item = self._wf_grid.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        self._wf_sliders = {}
        self._wf_pm.setText("")
        self._wf_hint.setText(self._wf_hint_default)
        if getattr(result, "reduced_order", False):
            # a lowest-order model is certified for ONE band at ONE
            # operating point; a slider excursion moves the operating
            # point, and the dropped dynamics that would react are no
            # longer in the model to react. Refusing with the reason
            # teaches the contract; the fix is one form-selector click.
            self._wf_hint.setText(
                "What-if is disabled for lowest-order results: moving a "
                "parameter can silently leave the band and operating "
                "point the reduction was certified for — the dropped "
                "poles cannot respond. Solve as Simplified · full order "
                "to explore parameters.")
            self._wf_hint.setVisible(True)
            self._wf_eval = None
            return
        wf = view.whatif_fn(result)
        self._wf_eval = wf
        self._wf_hint.setVisible(wf is None)
        if wf is None:
            return
        names, _ = wf
        for row, name in enumerate(names):
            lbl = QLabel(name)
            sl = QSlider(Qt.Horizontal)
            sl.setRange(0, 100)
            sl.setValue(50)
            sl.valueChanged.connect(self._on_whatif_changed)
            val = QLabel("×1.00")
            self._wf_grid.addWidget(lbl, row, 0)
            self._wf_grid.addWidget(sl, row, 1)
            self._wf_grid.addWidget(val, row, 2)
            self._wf_sliders[name] = (sl, val)

    def whatif_factors(self) -> dict:
        out = {}
        for name, (sl, _val) in self._wf_sliders.items():
            out[name] = 4.0 ** ((sl.value() - 50) / 50.0)
        return out

    def _on_whatif_changed(self, _v=None):
        if self._wf_eval is None or self.result is None:
            return
        import numpy as np

        factors = self.whatif_factors()
        for name, (sl, val) in self._wf_sliders.items():
            val.setText(f"×{factors[name]:.2f}")
        names, ev = self._wf_eval
        f = np.asarray(self.result.freqs, dtype=float)
        h = ev(f, factors)
        view.bode_figure(self.result, self.canvas.figure)
        ax1, ax2 = self.canvas.figure.axes[:2]
        ax1.semilogx(f, 20 * np.log10(np.abs(h)), color=theme.ORANGE,
                     lw=1.3, ls="--", label="what-if")
        ax2.semilogx(f, np.degrees(np.unwrap(np.angle(h))),
                     color=theme.ORANGE, lw=1.3, ls="--")
        view.figure_legend(self.canvas.figure, ax1)
        theme.style_figure(self.canvas.figure)
        self.canvas.draw_idle()
        if self.result.out.startswith("T@"):
            from ...analysis.sensitivity import loop_margins
            pm, fpm, gm, _ = loop_margins(f, h)
            self._wf_pm.setText(
                f"what-if margins:  PM {pm:.1f}°"
                f" @ {view.eng(fpm, 'Hz')}" +
                (f",  GM {gm:.1f} dB" if gm is not None else "")
                if pm is not None else "what-if: no unity crossing")
