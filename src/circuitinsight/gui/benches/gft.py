"""The GFT bench: error terms of one loop, the exact quartet."""
from __future__ import annotations

from PySide6.QtWidgets import (
    QComboBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget)
from .. import theme, view

class GFTBenchMixin:
    # ----------------------------------------------------------------- gft
    def _gft_page(self):
        page = QWidget()
        v = QVBoxLayout(page)
        row = QHBoxLayout()
        row.addWidget(QLabel("error ref:"))
        self.gft_ref_combo = QComboBox()
        self.gft_ref_combo.setMinimumWidth(120)
        row.addWidget(self.gft_ref_combo)
        self.gft_c_combo = QComboBox()
        self.gft_c_combo.addItems(["follower (c = −1)",
                                   "inverting (c = +1)"])
        row.addWidget(self.gft_c_combo)
        self.gft_btn = QPushButton("Dissect")
        self.gft_btn.clicked.connect(self.run_gft)
        row.addWidget(self.gft_btn)
        row.addStretch(1)
        v.addLayout(row)
        self.gft_lbl = QLabel(
            "The GFT quartet at the designated probe: H, the ideal Hinf "
            "(error nulled), the loop part Hinf·T/(1+T) and the "
            "feedthrough part H0/(1+T). The identity is checked in EXACT "
            "rational arithmetic.")
        self.gft_lbl.setWordWrap(True)
        v.addWidget(self.gft_lbl)
        v.addStretch(1)
        self._gft_tab = page
        return page

    def run_gft(self):
        if self.controller is None:
            return
        probe = self.probe_combo.currentText()
        inp, out = self._io()
        ref = self.gft_ref_combo.currentText()
        c = -1 if self.gft_c_combo.currentIndex() == 0 else +1
        if not probe or not ref:
            self.statusBar().showMessage("GFT needs a probe and an error ref")
            return
        # the form ladder, GFT flavor: the bench is numeric, so full
        # order (expression pruning) has nothing to act on and its
        # entry is disabled in this mode; the lowest order computes
        # the quartet OF THE REDUCED MODEL -- the identity is then
        # exact on that model, and the reduction is the one priced
        # approximation
        reduce_kw = None
        if self.form_combo.currentText() == "Simplified · lowest order":
            fl, fh = self.band_slider.values()
            reduce_kw = dict(strategy=self._strategy(),
                             strategy_opts=self._strategy_opts(),
                             fmin=fl, fmax=fh)

        def fn(_cb):
            return self._gft_compute(probe, inp, out, ref, c,
                                     reduce_kw=reduce_kw)

        self._launch(fn, f"GFT dissection at {probe}…",
                     on_done=self._on_gft_done)

    def gft_sync(self, probe, inp, out, ref, c, reduce_kw=None):
        payload = self._gft_compute(probe, inp, out, ref, c,
                                    reduce_kw=reduce_kw)
        self._on_gft_done(payload)
        return payload

    def _gft_compute(self, probe, inp, out, ref, c, reduce_kw=None):
        # the quartet lives in SessionController now (it is analysis,
        # and it used to take the analyzer raw -- plan E-A)
        return self.controller.gft_quartet(probe, inp, out, ref, c,
                                           reduce_kw=reduce_kw)

    def _on_gft_done(self, payload):
        import numpy as np

        self._tick.stop()
        self._t0 = None
        self.progress.hide()
        self.cancel_btn.hide()
        self._job_finished(self.gft_btn)
        self._show_result_tab()
        f = payload["freqs"]
        q = payload["q"]
        T = q["T"]
        loop_part = q["Hinf"] * T / (1 + T)
        ft_part = q["H0"] / (1 + T)
        fig = self.canvas.figure
        fig.clear()
        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)
        db = lambda z: 20 * np.log10(np.maximum(np.abs(z), 1e-300))
        for arr, lab, color, ls in (
                (q["H"], "H", theme.BLUE, "-"),
                (q["Hinf"], "H∞", theme.GREEN, "--"),
                (loop_part, "loop part", theme.ORANGE, "-"),
                (ft_part, "feedthrough", theme.PINK, ":")):
            ax1.semilogx(f, db(arr), color=color, lw=1.2, ls=ls, label=lab)
        try:                       # ac truth for the closed-loop H
            fr, packed = self.controller.ac_reference(payload["inp"],
                                                      payload["out"])
            if packed is not None:
                ax1.semilogx(fr, db(packed[0]), color="k", ls="--", lw=0.9,
                             label="AC sim")
        except Exception:
            pass                    # no ac truth here -- not a show-stopper
        dev = np.abs(q["H"] / q["Hinf"] - 1)
        ax2.loglog(f, np.maximum(dev, 1e-16), color=theme.VERMILION, lw=1.2)
        ax1.set_ylabel("(dB)")
        ax2.set_ylabel("|H/H∞ − 1|")
        ax2.set_xlabel("frequency (Hz)")
        for ax in (ax1, ax2):
            ax.grid(True, which="both", alpha=0.25, lw=0.4)
        fig.tight_layout(rect=(0, 0.06, 1, 1))
        view.figure_legend(fig, ax1)      # one legend home: below the
        theme.style_figure(fig)           # frequency axis, everywhere
        self.canvas.draw_idle()
        res = payload["residual"]
        sign = "+" if payload["c"] > 0 else "−"
        head = payload.get("headline")
        tail = f" — {head}" if head else ""
        if res == 0.0:
            self._set_strip("GFT identity EXACT (rational residual 0.0) at "
                            f"probe {payload['probe']}, error "
                            f"v({payload['ref']}) {sign} v(p)" + tail,
                            "ok")
        else:
            self._set_strip(f"GFT identity residual {res:.2e} -- designation "
                            "does not straddle the probe?", "error")
