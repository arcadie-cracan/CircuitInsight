"""The GFT bench: error terms of one loop, the exact quartet."""
from __future__ import annotations

from PySide6.QtWidgets import (
    QComboBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget)
from .. import theme

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

        def fn(_cb):
            return self._gft_compute(probe, inp, out, ref, c)

        self._launch(fn, f"GFT dissection at {probe}…",
                     on_done=self._on_gft_done)

    def gft_sync(self, probe, inp, out, ref, c):
        payload = self._gft_compute(probe, inp, out, ref, c)
        self._on_gft_done(payload)
        return payload

    def _gft_compute(self, probe, inp, out, ref, c):
        import numpy as np
        import sympy as sp

        from ...analysis import nested_gft
        from ...analysis.gft import _probe_indices
        from ...engine.mna import S

        an = self.controller._analyzer_ready()
        sys_in = an.system(inp)
        sys_pr = an.system(probe)
        A = nested_gft._exact_A(sys_pr)
        pr = _probe_indices(sys_pr, probe)
        err = (nested_gft._node(sys_pr, ref), int(c))
        io = nested_gft._node(sys_pr, out)
        fn_A = sp.lambdify(S, A, "numpy")
        z_in = np.asarray(sys_in.z, dtype=complex).ravel()
        z_pr = np.asarray(sys_pr.z, dtype=complex).ravel()
        freqs = np.geomspace(1.0, 1e10, 240)
        qs = [nested_gft._num_quartet(np.asarray(fn_A(2j * np.pi * f), complex),
                                z_in, z_pr, io, pr, err) for f in freqs]
        pack = {k: np.array([q[k] for q in qs]) for k in qs[0]}
        worst = 0.0
        for sv in (2, 3):
            A0 = A.xreplace({S: sp.Rational(sv)})
            q = nested_gft._point_quartet(A0, sys_in.z, sys_pr.z, io, pr, err)
            r = nested_gft._residual_of(q)
            if r != 0:
                worst = max(worst, abs(float((r / q["H"]).evalf())))
        return {"freqs": freqs, "q": pack, "residual": worst,
                "probe": probe, "ref": ref, "c": c, "inp": inp, "out": out}

    def _on_gft_done(self, payload):
        import numpy as np

        self._tick.stop()
        self._t0 = None
        self.progress.hide()
        self.cancel_btn.hide()
        self._job_finished(self.gft_btn)
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
            fr, packed = self.controller._reference(payload["inp"],
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
        ax1.legend(fontsize=8, frameon=False, loc="lower left", ncols=2)
        fig.tight_layout()
        theme.style_figure(fig)
        self.canvas.draw_idle()
        res = payload["residual"]
        sign = "+" if payload["c"] > 0 else "−"
        if res == 0.0:
            self._set_strip("GFT identity EXACT (rational residual 0.0) at "
                            f"probe {payload['probe']}, error "
                            f"v({payload['ref']}) {sign} v(p)",
                            "ok")
        else:
            self._set_strip(f"GFT identity residual {res:.2e} -- designation "
                            "does not straddle the probe?", "error")
