"""The Modes bench: the DM/CM 2x2 loop matrix and eigenloci."""
from __future__ import annotations

from .. import theme

class ModesBenchMixin:
    # ---------------------------------------------------------------- modes
    def run_modes(self):
        if self.controller is None:
            return
        pa, pb = self.probe_combo.currentText(), self.probe2_combo.currentText()
        if not pa or not pb or pa == pb:
            self.statusBar().showMessage("Modes needs two distinct probes")
            return

        def fn(_cb):
            an = self.controller._analyzer_ready()
            return an.mode_loop(pa, pb)

        self._launch(fn, f"mode loop matrix at ({pa}, {pb})…",
                     on_done=self._on_modes_done)

    def modes_sync(self, pa, pb):
        an = self.controller._analyzer_ready()
        rep = an.mode_loop(pa, pb)
        self._on_modes_done(rep)
        return rep

    def _on_modes_done(self, rep):
        import numpy as np

        self._tick.stop()
        self._t0 = None
        self.progress.hide()
        self.cancel_btn.hide()
        self._job_finished()
        fig = self.canvas.figure
        fig.clear()
        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)
        colors = (theme.BLUE, theme.VERMILION)
        for k in range(rep.loci.shape[1]):
            lam = rep.loci[:, k]
            pm, fu, gm = rep.margins[k]
            lab = rep.labels[k].split(".")[-1]
            if pm is not None:
                lab += f"  (PM {pm:.1f}°)"
            ax1.semilogx(rep.freqs, 20 * np.log10(np.abs(lam)),
                         color=colors[k % 2], lw=1.3, label=lab)
            ax2.semilogx(rep.freqs,
                         np.degrees(np.unwrap(np.angle(lam))),
                         color=colors[k % 2], lw=1.3)
            if fu:
                for ax in (ax1, ax2):
                    ax.axvline(fu, color=colors[k % 2], lw=0.6, ls="--",
                               alpha=0.6)
        try:                       # the run's own stb, on its locus
            stb = self.controller._run.stb()
            sp_probe = self.controller.stb_probe()
            if sp_probe in rep.probes:
                k = list(rep.probes).index(sp_probe)
                ax1.semilogx(stb.freq,
                             20 * np.log10(np.abs(stb.loop_gain)),
                             color="k", ls="--", lw=0.9,
                             label=f"Spectre stb ({sp_probe.split('.')[-1]})")
                ax2.semilogx(stb.freq,
                             np.degrees(np.unwrap(np.angle(stb.loop_gain))),
                             color="k", ls="--", lw=0.9)
        except Exception:
            pass                    # no stb truth here -- not a show-stopper
        ax1.axhline(0.0, color="k", lw=0.5, ls=":", alpha=0.6)
        ax1.set_ylabel("|λ| (dB)")
        ax2.set_ylabel("phase (deg)")
        ax2.set_xlabel("frequency (Hz)")
        for ax in (ax1, ax2):
            ax.grid(True, which="both", alpha=0.25, lw=0.4)
        ax1.legend(fontsize=8, frameon=False, loc="lower left")
        fig.tight_layout()
        theme.style_figure(fig)
        self.canvas.draw_idle()
        sev = "ok"
        if any(m[0] is not None and m[0] < 45 for m in rep.margins):
            sev = "warn"
        self._set_strip("modes: " + rep.summary()
                        + f"  |  Schur certificate {rep.schur_residual:.1e}",
                        sev)
        self.summary.setPlainText(
            "Mode loop matrix (eigenloci)\n" + rep.summary()
            + f"\nmax cross-mode coupling r = {rep.max_coupling:.3g}"
            + f"\nSchur certificate {rep.schur_residual:.2e}")
