"""The Compensate bench: suggestion table, branch preview, the what-if
mirror."""
from __future__ import annotations

from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QHBoxLayout, QLabel, QLineEdit, QPushButton, QSpinBox, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)
from .. import theme
from .. import view

class CompensateBenchMixin:
    def _comp_page(self):
        page = QWidget()
        v = QVBoxLayout(page)
        row = QHBoxLayout()
        row.addWidget(QLabel("goal:"))
        self.goal_combo = QComboBox()
        self.goal_combo.addItems(["mfm", "pm", "spec"])
        self.goal_combo.setToolTip(
            "mfm: place the dominant closed-loop pair at Butterworth damping\n"
            "pm: meet a phase-margin floor\n"
            "spec: hold the peak sensitivity Ms = max|1/(1-T)| (Middlebrook's\n"
            "      discrepancy target; Ms 1.3 ~ PM 50°, 1.2 ~ PM 60°)")
        self.goal_combo.currentTextChanged.connect(self._on_goal_changed)
        row.addWidget(self.goal_combo)
        self.pm_lbl = QLabel("PM target:")
        row.addWidget(self.pm_lbl)
        self.comp_pm_spin = self._spin(60.0, 30.0, 85.0, 1.0, " °")
        row.addWidget(self.comp_pm_spin)
        self.ms_lbl = QLabel("Ms target:")
        row.addWidget(self.ms_lbl)
        self.ms_spin = self._spin(1.3, 1.0, 3.0, 0.05, "")
        self.ms_spin.setDecimals(2)
        row.addWidget(self.ms_spin)
        row.addWidget(QLabel("branches:"))
        self.kmax_spin = QSpinBox()
        self.kmax_spin.setRange(1, 3)
        self.kmax_spin.setValue(1)
        self.kmax_spin.setToolTip(
            "1: rank single OP-invariant branches by area.\n"
            ">1: grow a nested (NMC) network one branch at a time, stopping\n"
            "when the goal is met or a further branch does not pay its area.")
        row.addWidget(self.kmax_spin)
        row.addWidget(QLabel("strip:"))
        self.exclude_edit = QLineEdit()
        self.exclude_edit.setPlaceholderText("I0.Cc, I0.Rz")
        self.exclude_edit.setMaximumWidth(140)
        self.exclude_edit.setToolTip(
            "Existing compensation instances to remove before suggesting:\n"
            "the re-compensate workflow. Removing a C or series-RC branch is\n"
            "operating-point invariant, so the reconstruction stays exact and\n"
            "one DC solve still spans the whole design space.")
        row.addWidget(self.exclude_edit)
        self.mirror_chk = QCheckBox("mirrored")
        self.mirror_chk.setToolTip(
            "Fully-differential: install each branch as a matched symmetric\n"
            "pair (itself plus its mirror image, same value). The map is\n"
            "derived from p/n node names; self-symmetric positions such as\n"
            "CMFB or tail stay single-ended.")
        row.addWidget(self.mirror_chk)
        self.suggest_btn = QPushButton("Suggest compensation")
        self.suggest_btn.clicked.connect(self.suggest_comp)
        row.addWidget(self.suggest_btn)
        row.addStretch(1)
        v.addLayout(row)
        self.comp_tbl = QTableWidget(0, 8)
        self.comp_tbl.setHorizontalHeaderLabels(
            ["pair", "network", "C", "R", "area", "ζ",
             "PM", "ok"])
        self.comp_tbl.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.comp_tbl.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.comp_tbl.itemSelectionChanged.connect(self._on_comp_selected)
        v.addWidget(self.comp_tbl, 1)
        self._comp_hint = QLabel(
            "Select a row to preview its loop gain instantly (rank-one "
            "update, no re-solve).")
        self._comp_hint.setWordWrap(True)
        v.addWidget(self._comp_hint)
        self._comp_steps = QLabel("")
        self._comp_steps.setWordWrap(True)
        v.addWidget(self._comp_steps)
        self._comp_suggestions = []
        self._comp_multi = None
        self._comp_exclude = ()
        self._comp_probe = None
        self._comp_upd = None
        self._comp_tab = page
        self._on_goal_changed(self.goal_combo.currentText())
        return page

    def _on_goal_changed(self, goal):
        """PM and Ms targets are alternatives; mfm needs neither."""
        for w in (self.pm_lbl, self.comp_pm_spin):
            w.setVisible(goal == "pm")
        for w in (self.ms_lbl, self.ms_spin):
            w.setVisible(goal == "spec")

    def _comp_mirror(self):
        """The p/n mirror map for this design, or None when unchecked."""
        if not self.mirror_chk.isChecked() or self.controller is None:
            return None
        an = self.controller._analyzer_ready()
        return view.mirror_map(an.system(self._comp_probe).node_index) or None

    def _comp_kw(self):
        """Goal-specific keywords, so an unused target never reaches the
        session cache key."""
        goal = self.goal_combo.currentText()
        kw = {"goal": goal}
        if goal == "pm":
            kw["pm_target"] = self.comp_pm_spin.value()
        elif goal == "spec":
            kw["ms_target"] = self.ms_spin.value()
        return kw

    def suggest_comp(self):
        if self.controller is None:
            return
        probe = self.probe_combo.currentText()
        if not probe:
            self.statusBar().showMessage("no loop probe in this design")
            return
        self._comp_probe = probe
        kw = self._comp_kw()
        exclude = tuple(s.strip() for s in self.exclude_edit.text().split(",")
                        if s.strip())
        self._comp_exclude = exclude
        if exclude:
            kw["exclude"] = exclude
        k_max = self.kmax_spin.value()
        mirror = self._comp_mirror()
        if mirror is not None:
            kw["mirror"] = mirror
        what = "network" if k_max > 1 else "branches"

        def fn(_cb):
            baseline = self.controller.loop_gain(probe)
            if k_max > 1:
                res = self.controller.suggest_multi_compensation(
                    probe, k_max=k_max, **kw)
            else:
                res = self.controller.suggest_compensation(probe, **kw)
            return (baseline, res)

        self._launch(fn, f"searching compensation {what} at {probe} "
                         f"({kw['goal']})…", on_done=self._on_comp_done)

    def suggest_sync(self, probe, *, k_max=1, **kw):
        """Blocking search, for tests and scripting. k_max > 1 grows a
        multi-branch (NMC) network instead of ranking single branches."""
        self._comp_probe = probe
        self._comp_exclude = tuple(kw.get("exclude", ()))
        baseline = self.controller.loop_gain(probe)
        if k_max > 1:
            res = self.controller.suggest_multi_compensation(
                probe, k_max=k_max, **kw)
        else:
            res = self.controller.suggest_compensation(probe, **kw)
        self._on_comp_done((baseline, res))
        return self._comp_multi if k_max > 1 else self._comp_suggestions

    def _on_comp_done(self, payload):
        self._tick.stop()
        self._t0 = None
        self.progress.hide()
        self.cancel_btn.hide()
        self._job_finished(self.suggest_btn)
        baseline, res = payload
        self._comp_baseline = baseline
        self._comp_upd = None                     # rebuilt lazily on select
        if not self._comp_probe:
            self._comp_probe = self.probe_combo.currentText()
        self._show(baseline)
        tbl = self.comp_tbl

        if hasattr(res, "branches"):              # MultiSuggestion (NMC)
            self._comp_multi = res
            self._comp_suggestions = []
            rows = res.branches
            tbl.setRowCount(len(rows))
            for i, b in enumerate(rows):
                pair = f"{b.node_a} ↔ {b.node_b or 'gnd'}"
                if b.twin is not None:
                    pair += f"  (+{b.twin[0]} ↔ {b.twin[1] or 'gnd'})"
                cells = (pair, b.network, view.eng(b.C, "F"),
                         view.eng(b.R, "Ω"),
                         f"{b.mult * (b.C / 1e-12 + 0.05 * b.R / 1e3):.1f}",
                         "—", "—", "—")
                for j, text in enumerate(cells):
                    tbl.setItem(i, j, QTableWidgetItem(text))
            self._comp_steps.setText("  ·  ".join(res.steps))
            ms = (f", Ms {res.spec_dev:.2f}" if res.spec_dev is not None
                  else "")
            pm = (f", PM {res.pm_deg:.1f}°" if res.pm_deg is not None else "")
            verdict = "goal met" if res.achieved else "goal NOT met"
            self._comp_hint.setText(
                f"{len(rows)}-branch network, area {res.area:.1f}, "
                f"ζ {res.zeta:.3f}{pm}{ms} — {verdict}. "
                f"Select any row to preview the whole network.")
            self.tabs.setCurrentWidget(self._comp_tab)
            self.statusBar().showMessage(
                f"{len(rows)}-branch network at {self._comp_probe} "
                f"({verdict})")
            return

        self._comp_multi = None
        self._comp_suggestions = list(res)
        self._comp_steps.setText("")
        tbl.setRowCount(len(self._comp_suggestions))
        for i, sg in enumerate(self._comp_suggestions):
            pair = f"{sg.candidate.node_a} ↔ "                    f"{sg.candidate.node_b or 'gnd'}"
            cells = (pair, sg.network, view.eng(sg.C, "F"),
                     view.eng(sg.R, "Ω"),
                     f"{sg.area:.1f}", f"{sg.zeta:.3f}",
                     f"{sg.pm_deg:.1f}°" if sg.pm_deg else "—",
                     "✓" if sg.achieved else "✗")
            for j, text in enumerate(cells):
                tbl.setItem(i, j, QTableWidgetItem(text))
        self.tabs.setCurrentWidget(self._comp_tab)
        self.statusBar().showMessage(
            f"{len(self._comp_suggestions)} suggestions at "
            f"{self._comp_probe}; select one to preview")

    def _comp_updater(self):
        """Preview updater on the SAME system the search ran on. When the
        search excluded existing compensation, the preview must exclude it
        too, or it would stack the suggestion on top of the branches the
        search had removed and report a margin nobody designed."""
        if self._comp_upd is None:
            import numpy as np
            from ...analysis.compensate import LoopGainUpdater
            from ...engine.mna import build_mna

            an = self.controller._analyzer_ready()
            drop = set(self._comp_exclude)
            if drop:
                system = build_mna(
                    [p for p in an.primitives if p.inst not in drop],
                    an.flat.ground, self._comp_probe, an._alias)
            else:
                system = an.system(self._comp_probe)
            self._comp_upd = LoopGainUpdater(
                system, self._comp_probe, np.geomspace(1.0, 1e10, 300))
        return self._comp_upd

    @staticmethod
    def _admittance(C, R):
        """Y(s) of a series R-C branch (R = 0 is a plain capacitor)."""
        return lambda s: s * C / (1 + s * R * C)

    def _preview_branches(self, row):
        """The physical (node_a, node_b, Y) branches the selected row
        installs. For a multi-branch network this is the WHOLE network, since
        the intermediate states are not designs the tool proposes; for a
        single suggestion it is that branch, plus its mirror image when the
        search ran mirrored."""
        import re

        if self._comp_multi is not None:
            out = []
            for b in self._comp_multi.branches:
                Y = self._admittance(b.C, b.R)
                out.extend((na, nb, Y) for na, nb in b.physical())
            return out
        sg = self._comp_suggestions[row]
        Y = self._admittance(sg.C, sg.R)
        out = [(sg.candidate.node_a, sg.candidate.node_b, Y)]
        # the single-branch suggester records a mirrored twin in its rationale
        m = re.search(r"\[symmetric pair with \(([^,]+), ([^)]+)\)\]",
                      sg.candidate.rationale)
        if m:
            a, b = (x.strip() for x in m.groups())
            out.append((a, None if b in ("None", "gnd") else b, Y))
        return out

    def _on_comp_selected(self):
        import numpy as np

        rows = {i.row() for i in self.comp_tbl.selectedIndexes()}
        if not rows or not (self._comp_suggestions or self._comp_multi):
            return
        upd = self._comp_updater()
        branches = self._preview_branches(min(rows))
        T = (upd.with_branches(branches) if len(branches) > 1
             else upd.with_branch(*branches[0]))
        f = upd.freqs
        view.bode_figure(self._comp_baseline, self.canvas.figure)
        ax1, ax2 = self.canvas.figure.axes[:2]
        ax1.semilogx(f, 20 * np.log10(np.abs(T)), color=theme.ORANGE, lw=1.3,
                     ls="--", label="preview")
        ax2.semilogx(f, np.degrees(np.unwrap(np.angle(T))), color=theme.ORANGE,
                     lw=1.3, ls="--")
        ax1.legend(fontsize=8, frameon=False, loc="lower left")
        theme.style_figure(self.canvas.figure)
        self.canvas.draw_idle()
        from ...analysis.sensitivity import loop_margins
        pm, fpm, gm, _ = loop_margins(f, T)
        note = (f"preview PM {pm:.1f}° @ {view.eng(fpm, 'Hz')}"
                if pm is not None else "preview: no crossing")
        if gm is not None:
            note += f",  GM {gm:.1f} dB"
        if len(branches) > 1:
            note += f"  ({len(branches)} branches installed)"
        self._comp_hint.setText(note)
