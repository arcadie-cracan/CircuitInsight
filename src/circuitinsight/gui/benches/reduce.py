"""The Reduce-circuit bench: AC-ground scan, removal scan, pole
attribution, numeral explanation, apply/revert of the reduced netlist."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView, QComboBox, QGroupBox, QHBoxLayout, QLabel,
    QPushButton, QTableWidget, QTableWidgetItem, QTextEdit,
    QVBoxLayout, QWidget)
from .. import theme
from .. import view

class ReduceBenchMixin:
    # --------------------------------------------------------- compensation
    def _reduce_candidates_col(self):
        """The whole Reduce bench as ONE full-height column in the left
        side's second slot (the symbols column's, idle here): the
        two candidate tabs -- AC grounds, Device removals -- each with
        its scan verb above its tick table, the ticked set's joint
        price with Apply, the applied headline right below the button
        that produced it with the verb that undoes it, and the
        evidence. The tables are declarative: Apply re-derives
        the working circuit from whatever is ticked in both tabs, and
        grounding and outright deletion are the only approximations in
        the chain -- everything shown after them is exact for the
        rewritten circuit. No bottom bench tab: the plot keeps the
        room."""
        from PySide6.QtWidgets import QTabWidget

        col = QWidget()
        v = QVBoxLayout(col)
        v.setContentsMargins(2, 2, 2, 2)
        v.setSpacing(4)

        # the contract's RESPONSE: an stb bench must preserve the loop
        # gain, an ac bench its transfer -- the scans, the joint price,
        # the plot and Apply all follow this choice
        arow = QHBoxLayout()
        arow.addWidget(QLabel("price against:"))
        self.reduce_against = QComboBox()
        self.reduce_against.addItems(["Transfer (in → out)", "Loop gain"])
        self.reduce_against.setToolTip(
            "The response every reduction is priced on.\n"
            "Transfer: tf(in → out), the ac bench's contract.\n"
            "Loop gain: the Tian loop gain at the stb probe -- what a\n"
            "reduction must preserve on a stability bench.")
        self.reduce_against.currentIndexChanged.connect(
            self._on_reduce_against)
        arow.addWidget(self.reduce_against, 1)
        v.addLayout(arow)

        tabs = QTabWidget()
        tabs.setObjectName("reduce_cands")

        pg = QWidget()
        pv = QVBoxLayout(pg)
        pv.setContentsMargins(2, 2, 2, 2)
        pv.setSpacing(2)
        scanb = QPushButton("Scan AC grounds")
        scanb.setToolTip(
            "Rank the mirror/bias nodes by the EXACT error grounding each "
            "would cause in the current in→out transfer (one matrix inverse "
            "per frequency prices every node)")
        scanb.clicked.connect(self.run_acg_scan)
        self.acg_scan_btn = scanb
        pv.addWidget(scanb)
        self.acg_tbl = QTableWidget(0, 5)
        self.acg_tbl.setHorizontalHeaderLabels(
            ["node", "cost", "phase", "kind", "gates"])
        self.acg_tbl.horizontalHeader().setStretchLastSection(True)
        self.acg_tbl.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.acg_tbl.itemChanged.connect(self._on_acg_toggled)
        pv.addWidget(self.acg_tbl, 1)
        tabs.addTab(pg, "AC grounds")

        pg2 = QWidget()
        rv = QVBoxLayout(pg2)
        rv.setContentsMargins(2, 2, 2, 2)
        rv.setSpacing(2)
        self.removal_btn = QPushButton("Scan removals")
        self.removal_btn.setToolTip(
            "Which explicit elements can simply be DELETED: every netlist "
            "passive priced by the exact response shift its removal would "
            "cause. Tick the priced elements below; Apply deletes them "
            "from the working circuit.")
        self.removal_btn.clicked.connect(self.run_removal_scan)
        rv.addWidget(self.removal_btn)
        self.rm_tbl = QTableWidget(0, 5)
        self.rm_tbl.setHorizontalHeaderLabels(
            ["element", "value", "cost", "phase", "kind"])
        self.rm_tbl.horizontalHeader().setStretchLastSection(True)
        self.rm_tbl.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.rm_tbl.itemChanged.connect(self._on_acg_toggled)
        rv.addWidget(self.rm_tbl, 1)
        tabs.addTab(pg2, "Device removals")
        self.reduce_tabs = tabs
        v.addWidget(tabs, 1)

        # the TICKED set's joint price and the verb that enacts it.
        # NO local budget knob: the scans gate under the toolbar's
        # tolerance strategy -- one contract for every approximation
        # (a second dB spin here would be the pm_spin class of dead
        # knob, or worse, a live one that silently disagrees)
        self.acg_joint_lbl = QLabel("")
        self.acg_joint_lbl.setWordWrap(True)
        v.addWidget(self.acg_joint_lbl)
        self.acg_apply = QPushButton("Apply reduction")
        self.acg_apply.setEnabled(False)
        self.acg_apply.setToolTip(
            "Rewrite the working circuit: ground the ticked nodes, remove "
            "the dead sources, lump, and delete the ticked elements. "
            "Measures the true end-to-end cost. Every bench then analyses "
            "the reduced circuit.")
        self.acg_apply.clicked.connect(self.apply_reduction)
        v.addWidget(self.acg_apply)

        # the APPLIED state with its measured cost, right below the
        # button that produced it -- where the eye lands on click --
        # and the one verb that undoes it
        self.red_banner = QLabel("circuit: as imported")
        self.red_banner.setWordWrap(True)
        v.addWidget(self.red_banner)
        self.acg_revert = QPushButton("Revert to as-imported")
        self.acg_revert.setEnabled(False)
        self.acg_revert.clicked.connect(self.revert_reduction)
        v.addWidget(self.acg_revert)

        # the evidence: what Apply does, collapsed by default
        self.acg_preview = QTextEdit()
        self.acg_preview.setReadOnly(True)
        self.acg_preview.setPlaceholderText(
            "Tick nodes above to see what grounding them unlocks: the "
            "controlled sources that die (exact) and the passives that "
            "lump into C_node / G_node symbols (exact).")
        self.acg_preview.setMaximumHeight(110)
        self.acg_evidence = QGroupBox("what Apply does")
        _ev = QVBoxLayout()
        _ev.setContentsMargins(2, 2, 2, 2)
        _ev.addWidget(self.acg_preview)
        self._collapsible(self.acg_evidence, _ev, collapsed=True)
        v.addWidget(self.acg_evidence, 0)
        self._acg_report = None
        self._reduce_plot_base = None
        return col

    def explain_per_numeral(self):
        """Analysis menu: the deep pass — every collapsed numeral of the
        current expression resolved individually. Needs the result's
        symbolic keep set (the numerals ARE the collapsed complements of
        those letters)."""
        if self.controller is None or self.result is None:
            return
        keep = self.result.keep if isinstance(self.result.keep, list) else []
        if not keep:
            self._set_strip("per-numeral attribution needs a symbolic "
                            "keep set — solve with kept letters first",
                            "info")
            return
        inp, out = self.result.inp, self.result.out

        def run(cb):
            # the float64 circle kernel (~20x): slots and values stay
            # exact from the cached solve; unconfirmed slots arrive
            # flagged approx and render with a leading ≈. Runs FIRST so
            # its progress (which includes a hidden plain solve when the
            # display is a lowest-order form) starts moving immediately
            deep = self.controller.explain_per_numeral(
                inp, out, keep=keep, progress=cb, fast=True)
            # the coefficient-level stories ride along (seconds next to
            # the deep sweep): they fill the A0/p1/z1 line hovers, which
            # are ratio attributions over whole coefficients -- without
            # them the formula-line numerals kept showing the run-me
            # prompt even after the deep pass
            self.controller.explain_numerals(inp, out, keep=keep)
            return deep

        # (the deep handler re-renders the CURRENT result; its keep is
        # the result's own, so display and stories agree by build)

        self._launch(
            run,
            f"resolving each numeral of {inp} → {out} (hybrid-grid "
            f"derivative sweep) …",
            on_done=self._on_explain_deep_done, est_s=None)

    def _shown_numerals(self) -> dict:
        """(part, k) -> the collapsed numeral the expression displays for
        that coefficient, when it has exactly one — the bridge from the
        numbers on screen to their attributions."""
        shown = {}
        try:
            import sympy as sp

            from ...units import eng

            npoly, dpoly = self.result.tf.num_den
            for part, poly in (("num", npoly), ("den", dpoly)):
                for powers, coeff in poly.as_dict().items():
                    fl = list(view.round_expr(coeff).atoms(sp.Float))
                    if len(fl) == 1:
                        shown[(part, powers[0])] = eng(float(fl[0]), sig=4)
        except Exception:
            pass
        return shown

    def _on_explain_deep_done(self, stories):
        self._tick.stop()
        self._t0 = None
        self.progress.hide()
        self.cancel_btn.hide()
        self._job_finished()
        lines = []
        # the coefficient stories were computed alongside the deep pass;
        # lead with the DISPLAYED numerals before the full per-numeral list
        if self.controller is not None and self.result is not None:
            rkeep = (self.result.keep
                     if isinstance(self.result.keep, list) else ())
            coarse = self.controller.cached_numerals(
                self.result.inp, self.result.out, keep=rkeep)
            if coarse:
                from ...analysis.explain import ratio_lines
                lines += ["ratio attribution — the DISPLAYED numerals "
                          "(shares subtract in a ratio):"]
                lines += ["  " + ln
                          for ln in ratio_lines(
                              coarse, shown=self._shown_numerals())]
                lines += [""]
        lines += ["per-numeral attribution (each collapsed numeral of the "
                  "expression, kept letters excluded):"]
        lines += ["  " + st.describe() for st in stories
                  if st.contributors]
        self.summary.setPlainText(self.summary.toPlainText()
                                  + "\n\n" + "\n".join(lines))
        self.tabs.setCurrentIndex(0)             # Summary
        self._render_expr()                      # fine hovers go live
        self.statusBar().showMessage(
            f"numerals resolved: {len(stories)} (see Summary — and hover "
            f"the numerals in the Expression tab)")

    def run_removal_scan(self):
        if self.controller is None:
            return
        inp, out = self._io()
        kw = self._contract_kw()
        target = (f"the loop gain at {kw['loop_probe']}"
                  if "loop_probe" in kw else f"{inp} → {out}")
        self._launch(
            lambda cb: self.controller.scan_removals(inp, out, **kw),
            f"pricing element removals against {target} …",
            on_done=self._on_removal_scan_done, est_s=None)

    def _on_removal_scan_done(self, rep):
        self._tick.stop()
        self._t0 = None
        self.progress.hide()
        self.cancel_btn.hide()
        self._job_finished()
        recommended = set(rep.recommended)
        units = {"c": "F", "r": "Ω", "g": "S", "l": "H"}
        self._filling = True
        try:
            self.rm_tbl.setRowCount(len(rep.candidates))
            ro = Qt.ItemIsEnabled | Qt.ItemIsSelectable
            for i, c in enumerate(rep.candidates):
                el_it = QTableWidgetItem(c.inst)
                el_it.setFlags(ro | Qt.ItemIsUserCheckable)
                el_it.setCheckState(Qt.Checked if c.inst in recommended
                                    else Qt.Unchecked)
                val_it = QTableWidgetItem(
                    view.eng(c.value, units.get(c.kind, ""), sig=4)
                    if c.value is not None else "—")
                cost_it = QTableWidgetItem(
                    f"{c.score:.2f}×" if c.score is not None
                    else f"{c.worst_db:.3g} dB")
                cost_it.setToolTip(f"{c.worst_db:.3g} dB / "
                                   f"{c.worst_deg:.3g}° worst; × "
                                   f"is the toolbar contract's budget")
                deg_it = QTableWidgetItem(f"{c.worst_deg:.3g}°")
                kind_it = QTableWidgetItem(c.kind)
                if not c.within_budget:
                    from PySide6.QtGui import QBrush, QColor
                    cost_it.setForeground(QBrush(QColor(theme.BAD)))
                for col, it in enumerate((el_it, val_it, cost_it, deg_it,
                                          kind_it)):
                    if col:
                        it.setFlags(ro)
                    self.rm_tbl.setItem(i, col, it)
        finally:
            self._filling = False
        self._refresh_applied_marks()
        self.statusBar().showMessage(
            f"removal scan: {len(rep.recommended)} element(s) deletable "
            f"within the budget" if rep.recommended
            else "removal scan: every element earns its place")
        self._refresh_acg_choice()

    def attribute_poles(self):
        """Analysis menu: which element establishes which pole, verified.
        On demand rather than per solve -- it costs a few seconds."""
        if self.controller is None:
            return
        inp, _ = self._io()
        self._launch(
            lambda cb: self.controller.pole_attribution(inp),
            f"attributing poles of {inp}'s network …",
            on_done=self._on_attribution_done, est_s=None)

    def _on_attribution_done(self, atts):
        self._tick.stop()
        self._t0 = None
        self.progress.hide()
        self.cancel_btn.hide()
        self._job_finished()
        lines = ["pole attribution (nudge-verified):"]
        lines += ["  " + a.describe() for a in atts]
        self.summary.setPlainText(self.summary.toPlainText()
                                  + "\n\n" + "\n".join(lines))
        self.tabs.setCurrentIndex(0)             # Summary
        self.statusBar().showMessage(
            f"poles attributed: {len(atts)} (see Summary)")

    def explain_numbers(self):
        """Analysis menu: rank the collapsed parameters behind each
        numeral of H(s). Kept symbols are excluded — those are already
        letters in the expression.

        Everything keys off the DISPLAYED result, not the panel: right
        after opening, the panel is ahead of the display by design
        (first light is pre-match, the auto plan pre-ticks the table),
        and stories computed for the panel's keep could never attach to
        the shown numerals — the hover kept its run-me prompt after the
        run (field report). The worker also re-solves the displayed
        (in, out, keep) under the CURRENT configuration — a cache hit
        when nothing drifted; when matches did drift, the refreshed
        display is what makes the value-keyed hovers land."""
        if self.controller is None or self.result is None:
            return
        r = self.result
        inp, out = r.inp, r.out
        keep = r.keep if isinstance(r.keep, list) else []

        def run(cb):
            cur = self.controller.attach_template(
                self.controller.solve(inp, out, keep, progress=cb))
            stories = self.controller.explain_numerals(inp, out, keep=keep,
                                                       progress=cb)
            return cur, stories

        self._launch(
            run,
            f"explaining the numbers of {inp} → {out} …",
            on_done=self._on_explain_done, est_s=None)

    def _on_explain_done(self, payload):
        cur, stories = payload
        self._tick.stop()
        self._t0 = None
        self.progress.hide()
        self.cancel_btn.hide()
        self._job_finished()
        if cur is not self.result:
            # the configuration moved since the shown solve (auto
            # matches landing is the common case): show the aligned
            # result so the numerals ARE the ones the stories explain.
            # _show composes the summary WITH the (now cached) stories
            self._show(cur, push_history=False)
        else:
            # APPEND, never recompose: other advisories (scan, pole
            # attribution) live in the summary as appended blocks and
            # a recompose would drop them -- the composer covers the
            # re-render paths, this covers the in-place one
            txt = self.summary.toPlainText()
            if "the numbers, explained" not in txt:
                self.summary.setPlainText(
                    txt + "\n\n" + self._stories_text(stories))
        self.tabs.setCurrentIndex(0)             # Summary
        self._render_expr()                      # numeral hovers go live
        self.statusBar().showMessage(
            f"numbers explained: {len(stories)} coefficients (see Summary "
            f"— and hover the numerals in the Expression tab)")

    def _stories_text(self, stories) -> str:
        """The 'numbers, explained' summary block; composed by
        _summary_with_extras wherever the summary renders, so the
        stories survive later async re-renders."""
        from ...analysis.explain import ratio_lines
        lines = ["the numbers, explained (kept symbols excluded):",
                 "ratio attribution — what shapes each DISPLAYED numeral "
                 "(shares subtract in a ratio, so the common gm chain "
                 "cancels):"]
        lines += ["  " + ln
                  for ln in ratio_lines(stories,
                                        shown=self._shown_numerals())]
        lines += ["per-coefficient shares (the raw material):"]
        lines += ["  " + st.describe() for st in stories]
        return "\n".join(lines)

    def _contract_kw(self) -> dict:
        """The toolbar's tolerance contract, as the scans consume it:
        strategy + budgets + the user's band, and the contract's
        RESPONSE (the loop probe when pricing against the loop gain).
        THE one approximation criterion for this bench."""
        lo, hi = self.band_slider.values()
        kw = dict(strategy=self._strategy(),
                  strategy_opts=self._strategy_opts(),
                  fmin=float(lo), fmax=float(hi))
        lp = self._reduce_loop_probe()
        if lp is not None:
            kw["loop_probe"] = lp
        return kw

    def _reduce_loop_probe(self):
        """The probe of the loop contract, or None in transfer mode."""
        combo = getattr(self, "reduce_against", None)
        if combo is None or combo.currentIndex() != 1 \
                or self.controller is None:
            return None
        return (self.controller.stb_probe()
                or (self.controller.probes[0]
                    if self.controller.probes else None))

    def _on_reduce_against(self, i: int):
        """Contract switch: prices in the tables describe the OTHER
        response now -- clear them; the plot follows the new contract."""
        if i == 1 and self._reduce_loop_probe() is None:
            self.statusBar().showMessage(
                "no loop probe in this run — pricing against the "
                "transfer")
            self.reduce_against.setCurrentIndex(0)
            return
        self.acg_tbl.setRowCount(0)
        self.rm_tbl.setRowCount(0)
        self.acg_joint_lbl.setText("")
        self.acg_apply.setEnabled(False)
        lp = self._reduce_loop_probe()
        if lp is not None:
            self.statusBar().showMessage(
                f"reductions priced against the loop gain at {lp}")
        if self.mode_combo.currentText() == "Reduce circuit":
            self._plot_reduction([])

    @staticmethod
    def _cost_phrase(score, worst_db, worst_deg=None) -> str:
        """The one sentence shape for a budgeted step's price."""
        native = (f"{worst_db:.3g} dB" if worst_deg is None
                  else f"{worst_db:.3g} dB / {worst_deg:.3g}°")
        if score is None:
            return native
        return f"{score:.2f}× budget ({native})"

    def run_acg_scan(self):
        if self.controller is None:
            return
        inp, out = self._io()
        kw = self._contract_kw()
        # a Nets-tree wish is ALWAYS priced: include carries it past the
        # structural filter, so the request cannot vanish silently
        kw["include"] = tuple(sorted(self._acg_pending))
        target = (f"the loop gain at {kw['loop_probe']}"
                  if "loop_probe" in kw else f"{inp} → {out}")
        self._launch(
            lambda cb: self.controller.scan_ac_grounds(inp, out, **kw),
            f"scanning AC-ground candidates against {target} …",
            on_done=self._on_acg_scan_done, est_s=None)

    def _on_acg_scan_done(self, rep):
        self._tick.stop()
        self._t0 = None
        self.progress.hide()
        self.cancel_btn.hide()
        self._job_finished()
        self._acg_report = rep
        recommended = set(rep.recommended)
        self._filling = True
        try:
            self.acg_tbl.setRowCount(len(rep.candidates))
            ro = Qt.ItemIsEnabled | Qt.ItemIsSelectable
            for i, c in enumerate(rep.candidates):
                node_it = QTableWidgetItem(c.node)
                node_it.setFlags(ro | Qt.ItemIsUserCheckable)
                node_it.setCheckState(Qt.Checked if c.node in recommended
                                      else Qt.Unchecked)
                cost_it = QTableWidgetItem(
                    f"{c.score:.2f}×" if c.score is not None
                    else f"{c.worst_db:.3g} dB")
                cost_it.setToolTip(f"{c.worst_db:.3g} dB / "
                                   f"{c.worst_deg:.3g}° worst; × is the "
                                   f"toolbar contract's budget")
                deg_it = QTableWidgetItem(f"{c.worst_deg:.3g}°")
                kind_it = QTableWidgetItem(c.kind)
                gates_it = QTableWidgetItem(", ".join(c.controls[:6]))
                if not c.within_budget:
                    from PySide6.QtGui import QBrush, QColor
                    cost_it.setForeground(QBrush(QColor(theme.BAD)))
                for col, it in enumerate((node_it, cost_it, deg_it,
                                          kind_it, gates_it)):
                    if col:
                        it.setFlags(ro)
                    self.acg_tbl.setItem(i, col, it)
        finally:
            self._filling = False
        if self._acg_pending:
            # a Nets-tree wish arrived before the scan: tick it now that
            # it is priced -- the scan's `include` guarantees it is in
            # the table, structural candidate or not
            self._filling = True
            try:
                for i in range(self.acg_tbl.rowCount()):
                    it = self.acg_tbl.item(i, 0)
                    if it is not None and it.text() in self._acg_pending:
                        it.setCheckState(Qt.Checked)
            finally:
                self._filling = False
            self._acg_pending.clear()
        self._refresh_applied_marks()
        # one cost grammar, in one place: the ticked line above the
        # table (refreshed below) -- not a dB-only copy in the status bar
        self.statusBar().showMessage(
            f"scan: {len(rep.recommended)} node(s) groundable within the "
            f"budget" if rep.recommended
            else "scan: no node is groundable within the budget")
        self._refresh_acg_choice()

    # ------------------------------------------------------------ nets tree
    def _input_net(self) -> str | None:
        """The net the input source drives: the source instance's first
        non-ground terminal."""
        if self.controller is None:
            return None
        inp = self.in_combo.currentText()
        gnd = set(self.controller.ground) | {"0"}
        for d in self.controller.devices:
            if d.name == inp:
                for net in d.terminals.values():
                    if net not in gnd:
                        return net
        return None

    def _refresh_net_decor(self):
        """The Nets tree tells the truth about the working circuit: ⏚ on
        the nets the active reduction AC-grounded, arrows on the input
        source's net and the output net."""
        if self.controller is None or not hasattr(self, "nets_tree"):
            return
        summ = self.controller.reduction_summary()
        self.nets_tree.set_decorations(
            acg=(summ or {}).get("nodes", ()),
            inp=self._input_net(), out=self.out_combo.currentText())

    def _set_output_net(self, net: str):
        self.out_combo.setCurrentText(net)
        self._refresh_net_decor()
        self.statusBar().showMessage(f"output: {net}")

    def _acg_from_net(self, net: str):
        """Route an AC-ground request from the Nets tree through the
        measured Reduce flow: never ground silently — the scan prices
        the node, the user applies."""
        if self.controller is None:
            return
        self.mode_combo.setCurrentText("Reduce circuit")
        self.reduce_tabs.setCurrentIndex(0)
        for i in range(self.acg_tbl.rowCount()):
            it = self.acg_tbl.item(i, 0)
            if it is not None and it.text() == net:
                it.setCheckState(Qt.Checked)
                self.statusBar().showMessage(
                    f"{net} ticked — Apply reduction to ground it "
                    f"(cost shown above)")
                return
        # not scanned yet: remember the wish, scan, tick when priced
        self._acg_pending.add(net)
        self.run_acg_scan()

    def _goto_instance(self, name: str):
        """A connection in the Nets tree jumps to its instance."""
        it = self.devices.item_for(name)
        if it is None:
            return
        self.dev_tabs.setCurrentWidget(self.devices)
        self.devices.setCurrentItem(it)
        self.devices.scrollToItem(it)

    def checked_acg_nodes(self) -> list[str]:
        out = []
        for i in range(self.acg_tbl.rowCount()):
            it = self.acg_tbl.item(i, 0)
            if it is not None and it.checkState() == Qt.Checked:
                out.append(it.text())
        return out

    def _refresh_applied_marks(self):
        """The tables are declarative -- ticks ARE the target state, and
        Apply re-derives the circuit from whatever is ticked -- so a row
        whose tick is already in force must read differently from a
        pending suggestion: the name turns green with an 'applied'
        tooltip. Refreshed after every scan refill, apply and revert."""
        from PySide6.QtGui import QBrush, QColor

        summ = (self.controller.reduction_summary()
                if self.controller is not None else None) or {}
        marks = ((self.acg_tbl, set(summ.get("nodes") or ()),
                  "grounded"),
                 (self.rm_tbl, set(summ.get("removed") or ()),
                  "deleted"))
        was = self._filling
        self._filling = True
        try:
            for tbl, applied, verb in marks:
                for i in range(tbl.rowCount()):
                    it = tbl.item(i, 0)
                    if it is None:
                        continue
                    if it.text() in applied:
                        it.setForeground(QBrush(QColor(theme.GOOD)))
                        it.setToolTip(f"applied — {verb} in the working "
                                      f"circuit; untick and Apply to "
                                      f"undo")
                    else:
                        it.setForeground(QBrush())
                        it.setToolTip("")
        finally:
            self._filling = was

    def checked_removals(self) -> list[str]:
        out = []
        for i in range(self.rm_tbl.rowCount()):
            it = self.rm_tbl.item(i, 0)
            if it is not None and it.checkState() == Qt.Checked:
                out.append(it.text())
        return out

    def _on_acg_toggled(self, item):
        if self._filling or item.column() != 0:
            return
        self._refresh_acg_choice()

    def _refresh_acg_choice(self):
        """Price the ticked set and preview its exact follow-on. Both are
        sub-second (one inverse per frequency; a pure-python rewrite), so
        this runs inline on every toggle."""
        if self.controller is None:
            return
        nodes = self.checked_acg_nodes()
        rms = self.checked_removals()
        self.acg_apply.setEnabled(bool(nodes or rms))
        if not nodes and not rms:
            self.acg_joint_lbl.setText("")
            self.acg_preview.clear()
            self._plot_reduction([])
            return
        inp, out = self._io()
        try:
            jm = self.controller.reduction_joint(inp, out, nodes, rms,
                                                 **self._contract_kw())
            pv = self.controller.preview_reduction(nodes, rms)
        except Exception as exc:
            self.acg_joint_lbl.setText(f"({type(exc).__name__}: {exc})")
            return
        cost = self._cost_phrase(jm["score"], jm["worst_db"],
                                 jm["worst_deg"])
        summ = self.controller.reduction_summary() or {}
        if summ and set(nodes) == set(summ.get("nodes") or ()) \
                and set(rms) == set(summ.get("removals") or ()):
            self.acg_joint_lbl.setText(f"ticked = applied state ({cost})")
        else:
            self.acg_joint_lbl.setText(f"ticked: {cost} together")
        self._plot_reduction(nodes, rms)
        lines = [f"{pv['prims_before']} → {pv['prims_after']} primitives:"]
        if pv["dead_sources"]:
            dp = pv.get("dead_primitives") or pv["dead_sources"]
            lines.append(
                f"  {len(dp)} controlled sources contribute exactly "
                f"zero — removed (exact): " + ", ".join(dp[:8])
                + ("…" if len(dp) > 8 else ""))
            gone = pv.get("devices_silenced") or []
            if gone:
                lines.append("    devices left with nothing live: "
                             + ", ".join(gone[:8])
                             + ("…" if len(gone) > 8 else ""))
            part = pv.get("devices_passive") or []
            if part:
                lines.append("    devices reduced to loads (gds/caps "
                             "stay): " + ", ".join(part[:8])
                             + ("…" if len(part) > 8 else ""))
        for g in pv["lump_groups"]:
            lines.append("  lump (exact): " + g)
        if pv["symbols_saved"]:
            lines.append(f"  {pv['symbols_saved']} symbols saved for the "
                         f"solver grid")
        if pv.get("removed"):
            lines.append("  deleted outright (priced by the removal "
                         "scan): " + ", ".join(pv["removed"]))
        missing = set(rms) - set(pv.get("removed") or ())
        if missing:
            lines.append("  not in the rewritten circuit (nothing to "
                         "delete): " + ", ".join(sorted(missing)))
        self.acg_preview.setPlainText("\n".join(lines))

    def apply_reduction(self):
        if self.controller is None:
            return
        nodes = self.checked_acg_nodes()
        rms = self.checked_removals()
        if not nodes and not rms:
            return
        inp, out = self._io()
        self._launch(
            lambda cb: self.controller.apply_reduction(
                nodes, inp=inp, out=out, removals=rms,
                **self._contract_kw()),
            f"applying reduction ({', '.join(nodes + rms)}) …",
            on_done=self._on_reduction_applied, est_s=None)

    def _on_reduction_applied(self, summ):
        self._tick.stop()
        self._t0 = None
        self.progress.hide()
        self.cancel_btn.hide()
        self._job_finished()
        self._refresh_reduction_banner()
        self._refresh_net_decor()
        # the reduction REWRITES the circuit: dead sources removed,
        # passives lumped into Ceq/Geq node symbols -- the old ranking
        # and its ticks name symbols that may no longer exist. Re-rank
        # the reduced circuit in the background and carry the ticks
        # over: a ticked symbol that survived stays ticked, a ticked
        # member of a lump ticks the equivalent that absorbed it.
        self._rerank_after_rewrite(
            "ranking the reduced circuit's symbols …",
            eqs=summ.get("equivalents") or [])
        cost = self._cost_phrase(summ.get("score"), summ["worst_db"],
                                 summ["worst_deg"])
        # the headline carries the applied state; the strip would be
        # a third copy of the same sentence
        self.log(f"circuit reduced: {summ['prims_before']} → "
                 f"{summ['prims_after']} primitives at a measured {cost} "
                 f"({summ['inp']} → {summ['out']}); every bench now "
                 f"analyses the reduced circuit")
        self._plot_reduction([])
        # the ĝm composition: grounding a bias net can make the exact
        # gm+gmb criterion hold on the reduced circuit — say which
        # devices earned their hat through this reduction
        if getattr(self, "mos_model", "separate") == "lumped-gmb":
            info = self.controller.lumped_gmb()
            composed = sorted(n for n, how in info.items()
                              if how == "lumped (reduced)")
            if composed:
                self.log(f"ĝm = gm+gmb now holds on the reduced circuit "
                         f"for {len(composed)} device(s): "
                         f"{', '.join(composed)} — the grounding above "
                         f"is the only approximation, the bundle itself "
                         f"is exact")

    def revert_reduction(self):
        if self.controller is None:
            return
        self.controller.revert_reduction()
        self._refresh_reduction_banner()
        self._refresh_net_decor()
        self._rerank_after_rewrite("ranking the restored symbols …")
        self._set_strip("circuit: back to as-imported", "info")

    def _rerank_after_rewrite(self, status: str, eqs=(), ranker=None):
        """The keep table after the circuit changed under it: rank the
        working circuit off the GUI thread and refill, keeping every
        tick whose symbol still exists and mapping a ticked lump
        member onto its equivalent. Until it lands the table is empty
        and the estimate says why."""
        c = self.controller
        # a later rerank supersedes this one: without the sequence, a
        # fast mode cycle lands a stale ranking (e.g. Z's) after the
        # bench moved on
        seq = getattr(self, "_rank_seq", 0) + 1
        self._rank_seq = seq
        was = set(self.checked_keep())
        absorbed = {}
        for e in eqs:
            for m in e.get("members", []):
                absorbed[m.replace(".", "_")] = e["name"]
        self.keep_tbl.setRowCount(0)
        self.estimate_lbl.setText(f"estimate: — ({status})")
        try:
            inp, out = self._io()
        except Exception:
            return

        def land(ranking):
            if self.controller is not c or ranking is None                     or seq != getattr(self, "_rank_seq", 0):
                return
            names = {r[0] if isinstance(r, (tuple, list)) else
                     getattr(r, "name", r) for r in ranking}
            keep = set()
            for s in was:
                if s in names:
                    keep.add(s)
                elif absorbed.get(s) in names:
                    keep.add(absorbed[s])
            self._fill_keep_table(ranking, checked=sorted(keep))
            self._update_estimate()

        # rank over the band the plot shows (see _rerank_for_z: the
        # feature-spanned default can collapse to a single resonance)
        lo, hi = self.band_slider.values()
        self._run_bg(ranker or (lambda: c.rank_symbols(inp, out,
                                                       fmin=lo, fmax=hi)),
                     land)

    def _plot_reduction(self, nodes, removals=()):
        """The bench's plot: the as-imported response (solid) against
        the ticked reduction (dashed) inside the tolerance tube -- the
        scan's cost as a curve. Computed off the GUI thread; the last
        plot stays until the new one lands."""
        if self.controller is None or \
                self.mode_combo.currentText() != "Reduce circuit":
            return
        c = self.controller
        try:
            inp, out = self._io()
        except Exception:
            return
        nodes = list(nodes)
        removals = list(removals)
        seq = getattr(self, "_redplot_seq", 0) + 1
        self._redplot_seq = seq
        lp = self._reduce_loop_probe()

        def compute():
            if lp is not None:
                import numpy as np

                ref = c.loop_reference()   # the stb truth rides along
                f = (np.asarray(ref[0], dtype=float) if ref is not None
                     else np.geomspace(1.0, 1e10, 201))
                hb, hr = c.reduction_responses(nodes, "", "", f,
                                               removals=removals,
                                               loop_probe=lp)
                return ("loop", f, hb, hr, ref)
            r0 = c.solve(inp, out, [])            # freqs + the sim reference
            hb, hr = c.reduction_responses(nodes, inp, out, r0.freqs,
                                           removals=removals)
            return ("tf", r0, hb, hr)

        def land(res):
            if res is None or seq != self._redplot_seq \
                    or self.controller is not c \
                    or self.mode_combo.currentText() != "Reduce circuit":
                return
            import types
            import numpy as np
            from dataclasses import replace

            if res[0] == "loop":
                _tag, f, hb, hr, ref = res
                base = types.SimpleNamespace(
                    inp=lp, out="T", freqs=f, h=np.asarray(hb),
                    h_ref=(np.asarray(ref[1]) if ref is not None
                           else None),
                    ref_label=(ref[2] if ref is not None else None),
                    poles_hz=np.array([]), zeros_hz=np.array([]),
                    simplified=False, pm_deg=None, gm_db=None,
                    band_fmin=None, band_fmax=None, warnings=[])
                red = types.SimpleNamespace(
                    **{**base.__dict__, "h": np.asarray(hr),
                       "h_ref": None})
            else:
                _tag, r0, hb, hr = res
                base = replace(r0, h=np.asarray(hb),
                               poles_hz=np.array([]),
                               zeros_hz=np.array([]), simplified=False,
                               pm_deg=None, gm_db=None, warnings=[])
                red = replace(base, h=np.asarray(hr), h_ref=None)
            bits = ([f"{len(nodes)} net(s)"] if nodes else []) + \
                ([f"{len(removals)} removal(s)"] if removals else [])
            any_tick = bool(nodes or removals)
            labels = ("as imported T" if lp is not None else "as imported",
                      [f"ticked reduction ({', '.join(bits)})"]
                      if any_tick else [])
            self._reduce_plot_base = base
            view.bode_figure(base, self.canvas.figure,
                             overlays=[red] if any_tick else (),
                             labels=labels)
            theme.style_figure(self.canvas.figure)
            self._update_tol_bands()
            self.canvas.draw_idle()

        self._run_bg(compute, land)

    def _refresh_reduction_banner(self):
        summ = (self.controller.reduction_summary()
                if self.controller is not None else None)
        try:
            self._schematic_restyle()
            self._refresh_solve_hint()
        except Exception:                         # noqa: BLE001
            pass                                  # decoration only
        if summ is None:
            self.red_banner.setText("applied: none — the circuit is as "
                                    "imported")
            self.red_banner.setStyleSheet(f"color: {theme.MUTED};")
            self.acg_revert.setEnabled(False)
        else:
            cost = self._cost_phrase(summ.get("score"), summ["worst_db"],
                                     summ["worst_deg"])
            bits = []
            if summ["nodes"]:
                bits.append(f"{len(summ['nodes'])} net(s) grounded "
                            f"({', '.join(summ['nodes'])})")
            rmv = summ.get("removed") or []
            if rmv:
                shown = ", ".join(rmv[:8]) + ("…" if len(rmv) > 8
                                              else "")
                bits.append(f"{len(rmv)} element(s) removed ({shown})")
            self.red_banner.setText(
                f"applied: {'; '.join(bits)}; {summ['prims_before']} → "
                f"{summ['prims_after']} primitives; {cost}, measured "
                f"end to end")
            self.red_banner.setStyleSheet(f"color: {theme.GOOD};")
            self.acg_revert.setEnabled(True)
        self._refresh_applied_marks()
