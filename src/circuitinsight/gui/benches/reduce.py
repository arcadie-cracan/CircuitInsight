"""The Reduce-circuit bench: AC-ground scan, removal scan, pole
attribution, numeral explanation, apply/revert of the reduced netlist."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QTableWidget, QTableWidgetItem, QTextEdit, QVBoxLayout, QWidget)
from .. import theme
from .. import view

class ReduceBenchMixin:
    # --------------------------------------------------------- compensation
    def _reduce_page(self):
        """The Reduce-circuit bench (gui-ux-plan.md U-C): scan the bias
        nodes, tick a set, watch the measured joint cost and the exact
        follow-on (dead sources, lumping) update, then Apply — the reduced
        circuit becomes THE working circuit for every bench, revertibly.
        Grounding is the only approximation in the chain; everything shown
        after it is exact for the rewritten circuit."""
        page = QWidget()
        v = QVBoxLayout(page)
        row = QHBoxLayout()
        self.removal_btn = QPushButton("Scan removals")
        self.removal_btn.setToolTip(
            "Which explicit elements can simply be DELETED: every netlist "
            "passive priced by the exact response shift its removal would "
            "cause. Advisory only -- edit the schematic to act on it.")
        self.removal_btn.clicked.connect(self.run_removal_scan)
        scanb = QPushButton("Scan AC grounds")
        scanb.setToolTip(
            "Rank the mirror/bias nodes by the EXACT error grounding each "
            "would cause in the current in→out transfer (one matrix inverse "
            "per frequency prices every node)")
        scanb.clicked.connect(self.run_acg_scan)
        row.addWidget(scanb)
        row.addWidget(self.removal_btn)
        # NO local budget knob: the scans gate under the toolbar's
        # tolerance strategy -- one contract for every approximation
        # (a second dB spin here would be the pm_spin class of dead
        # knob, or worse, a live one that silently disagrees)
        self.acg_joint_lbl = QLabel("")
        self.acg_joint_lbl.setSizePolicy(QSizePolicy.Ignored,
                                         QSizePolicy.Preferred)
        row.addWidget(self.acg_joint_lbl, 1)
        v.addLayout(row)

        self.acg_tbl = QTableWidget(0, 5)
        self.acg_tbl.setHorizontalHeaderLabels(
            ["node", "cost", "phase", "kind", "gates"])
        self.acg_tbl.horizontalHeader().setStretchLastSection(True)
        self.acg_tbl.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.acg_tbl.itemChanged.connect(self._on_acg_toggled)
        v.addWidget(self.acg_tbl, 2)

        self.acg_preview = QTextEdit()
        self.acg_preview.setReadOnly(True)
        self.acg_preview.setPlaceholderText(
            "Tick nodes above to see what grounding them unlocks: the "
            "controlled sources that die (exact) and the passives that "
            "lump into C_node / R_node symbols (exact).")
        v.addWidget(self.acg_preview, 1)

        arow = QHBoxLayout()
        self.acg_apply = QPushButton("Apply reduction")
        self.acg_apply.setEnabled(False)
        self.acg_apply.setToolTip(
            "Rewrite the working circuit: ground the ticked nodes, remove "
            "the dead sources, lump. Measures the true end-to-end cost and "
            "banners it. Every bench then analyses the reduced circuit.")
        self.acg_apply.clicked.connect(self.apply_reduction)
        arow.addWidget(self.acg_apply)
        self.acg_revert = QPushButton("Revert to as-imported")
        self.acg_revert.setEnabled(False)
        self.acg_revert.clicked.connect(self.revert_reduction)
        arow.addWidget(self.acg_revert)
        self.red_banner = QLabel("circuit: as imported")
        self.red_banner.setSizePolicy(QSizePolicy.Ignored,
                                      QSizePolicy.Preferred)
        arow.addWidget(self.red_banner, 1)
        v.addLayout(arow)
        self._acg_report = None
        return page

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
        self._launch(
            lambda cb: self.controller.scan_removals(inp, out, **kw),
            f"pricing element removals for {inp} → {out} …",
            on_done=self._on_removal_scan_done, est_s=None)

    def _on_removal_scan_done(self, rep):
        self._tick.stop()
        self._t0 = None
        self.progress.hide()
        self.cancel_btn.hide()
        self._job_finished()
        self.acg_preview.setPlainText(rep.describe())
        if rep.recommended:
            cost = self._cost_phrase(rep.joint_score, rep.joint_db)
            self.statusBar().showMessage(
                f"removal scan: {', '.join(rep.recommended)} deletable "
                f"together at {cost}")
        else:
            self.statusBar().showMessage(
                "removal scan: every element earns its place")

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
            # result so the numerals ARE the ones the stories explain
            self._show(cur, push_history=False)
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
        self.summary.setPlainText(self.summary.toPlainText()
                                  + "\n\n" + "\n".join(lines))
        self.tabs.setCurrentIndex(0)             # Summary
        self._render_expr()                      # numeral hovers go live
        self.statusBar().showMessage(
            f"numbers explained: {len(stories)} coefficients (see Summary "
            f"— and hover the numerals in the Expression tab)")

    def _contract_kw(self) -> dict:
        """The toolbar's tolerance contract, as the scans consume it:
        strategy + budgets + the user's band. THE one approximation
        criterion for this bench."""
        lo, hi = self.band_slider.values()
        return dict(strategy=self._strategy(),
                    strategy_opts=self._strategy_opts(),
                    fmin=float(lo), fmax=float(hi))

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
        self._launch(
            lambda cb: self.controller.scan_ac_grounds(inp, out, **kw),
            f"scanning AC-ground candidates for {inp} → {out} …",
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
        if rep.recommended:
            self.statusBar().showMessage(
                f"scan: ground {', '.join(rep.recommended)} together for "
                f"{rep.joint_db:.3g} dB")
        else:
            self.statusBar().showMessage(
                "scan: no node is groundable within the budget")
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
        self.acg_apply.setEnabled(bool(nodes))
        if not nodes:
            self.acg_joint_lbl.setText("")
            self.acg_preview.clear()
            return
        inp, out = self._io()
        try:
            jm = self.controller.acground_joint(inp, out, nodes,
                                                **self._contract_kw())
            pv = self.controller.preview_reduction(nodes)
        except Exception as exc:
            self.acg_joint_lbl.setText(f"({type(exc).__name__}: {exc})")
            return
        cost = self._cost_phrase(jm["score"], jm["worst_db"],
                                 jm["worst_deg"])
        self.acg_joint_lbl.setText(
            f"ticked set: {cost} together — the only approximation")
        lines = [f"{pv['prims_before']} → {pv['prims_after']} primitives:"]
        if pv["dead_sources"]:
            lines.append(
                f"  {len(pv['dead_sources'])} controlled sources contribute "
                f"exactly zero — removed (exact): "
                + ", ".join(pv["dead_sources"][:8])
                + ("…" if len(pv["dead_sources"]) > 8 else ""))
        for g in pv["lump_groups"]:
            lines.append("  lump (exact): " + g)
        if pv["symbols_saved"]:
            lines.append(f"  {pv['symbols_saved']} symbols saved for the "
                         f"solver grid")
        self.acg_preview.setPlainText("\n".join(lines))

    def apply_reduction(self):
        if self.controller is None:
            return
        nodes = self.checked_acg_nodes()
        if not nodes:
            return
        inp, out = self._io()
        self._launch(
            lambda cb: self.controller.apply_reduction(nodes, inp=inp, out=out,
                **self._contract_kw()),
            f"applying reduction ({', '.join(nodes)}) …",
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
        # passives lumped into Ceq/Req node symbols -- the old ranking
        # and its ticks name symbols that may no longer exist, and a
        # solve with them fails. Invalidate, like the matches path does.
        self.keep_tbl.setRowCount(0)
        self.estimate_lbl.setText(
            "estimate: — (re-Rank: the reduced circuit has lumped/removed "
            "symbols)")
        cost = self._cost_phrase(summ.get("score"), summ["worst_db"],
                                 summ["worst_deg"])
        self._set_strip(
            f"circuit reduced: {summ['prims_before']} → "
            f"{summ['prims_after']} primitives at a measured {cost} "
            f"({summ['inp']} → {summ['out']}); grounding was the only "
            f"approximation — every bench now analyses the reduced circuit",
            "ok")
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
        self.keep_tbl.setRowCount(0)             # reduced-circuit ranking
        self.estimate_lbl.setText("estimate: — (re-Rank)")
        self._set_strip("circuit: back to as-imported — re-Rank for the "
                        "restored symbols", "info")

    def _refresh_reduction_banner(self):
        summ = (self.controller.reduction_summary()
                if self.controller is not None else None)
        try:
            self._schematic_restyle()
        except Exception:                         # noqa: BLE001
            pass                                  # decoration only
        if summ is None:
            self.red_banner.setText("circuit: as imported")
            self.red_banner.setStyleSheet(f"color: {theme.MUTED};")
            self.acg_revert.setEnabled(False)
        else:
            self.red_banner.setText(
                f"circuit: REDUCED — grounded {', '.join(summ['nodes'])}; "
                f"{summ['prims_before']} → {summ['prims_after']} primitives, "
                f"≤ {summ['worst_db']:.3g} dB measured")
            self.red_banner.setStyleSheet(f"color: {theme.GOOD};")
            self.acg_revert.setEnabled(True)
