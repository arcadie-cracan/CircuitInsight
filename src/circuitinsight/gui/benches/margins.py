"""The Margins readout: the ledger that says WHAT dynamics set PM/GM
and WHICH element moves them (L-D surface over analysis.marginledger)."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView, QLabel, QMenu, QPushButton, QSplitter,
    QTableWidget, QTableWidgetItem, QTreeWidget, QTreeWidgetItem,
    QVBoxLayout, QWidget)

from ...analysis.marginledger import rebias_exponent
from .. import view


class MarginsBenchMixin:
    def _margins_page(self):
        page = QWidget()
        v = QVBoxLayout(page)
        # the bench's verbs live in the MAIN toolbar, beside Solve --
        # their acts show in the Loop-gain bench only (managed with the
        # tab visibility); the page keeps the readout itself
        tb = self.toolbar
        self._ml_sep_act = tb.addSeparator()
        self.ledger_btn = QPushButton("Compute margin ledger")
        self.ledger_btn.setToolTip(
            "Attribute PM and GM exactly: each singularity's share of\n"
            "the phase at the unity crossing and of the gain at the\n"
            "critical-phase crossing (the shares SUM to the measured\n"
            "response -- the certificate), plus dPM/dGM per percent for\n"
            "every element through the rank-one machinery.")
        self.ledger_btn.clicked.connect(self.compute_margins)
        self._ml_btn_act = tb.addWidget(self.ledger_btn)
        self.ledger_cmp_btn = QPushButton("Compare corners")
        self.ledger_cmp_btn.setToolTip(
            "The ledger at EVERY operating point of the set, each\n"
            "corner's shares matched onto the bound corner's rows:\n"
            "the term whose drift explains the margin's spread tops\n"
            "the table (already-computed points serve from cache).")
        self.ledger_cmp_btn.clicked.connect(self.compare_margins)
        # shown only over a point set AND in the bench: the session
        # sets the availability, the mode gate ANDs it in
        self._ml_cmp_avail = False
        self._ml_cmp_act = tb.addWidget(self.ledger_cmp_btn)
        self.ml_whatif_btn = QPushButton("What-if the top knobs")
        self.ml_whatif_btn.setToolTip(
            "Tick the strongest margin knobs into the keep set and\n"
            "run the hybrid loop solve: the What-if sliders become\n"
            "live PM knobs for exactly the parameters that matter.")
        self.ml_whatif_btn.clicked.connect(self.margins_to_whatif)
        self.ml_whatif_btn.setEnabled(False)
        self._ml_whatif_act = tb.addWidget(self.ml_whatif_btn)
        from PySide6.QtWidgets import QComboBox
        self.ml_cplx = QComboBox()
        self.ml_cplx.addItems(["simple", "detailed", "exhaustive"])
        self.ml_cplx.setToolTip(
            "How rich a named formula may get.\n"
            "simple: two branches, the fewest symbols in the error\n"
            "        band win\n"
            "detailed: a wider mover pool, up to three branches and\n"
            "        four summed capacitances\n"
            "exhaustive: the most accurate formula the 25% gate\n"
            "        admits, whatever its size")
        self.ml_cplx.currentIndexChanged.connect(
            lambda _i: self._ml_rows and self.compute_margins())
        self._ml_cplx_act = tb.addWidget(self.ml_cplx)
        for a in (self._ml_sep_act, self._ml_btn_act, self._ml_cmp_act,
                  self._ml_whatif_act, self._ml_cplx_act):
            a.setVisible(False)
        self._ml_head = QLabel("")
        self._ml_head.setWordWrap(True)
        v.addWidget(self._ml_head)

        # L-I: THE MARGIN EQUATION of this amplifier -- the folded
        # terms as arctans with each frequency's certified formula
        # substituted, KaTeX when WebEngine is present
        from .. import exprweb
        if exprweb.WEBENGINE:                     # pragma: no cover
            self._ml_eq = exprweb.ExprWebView()
        else:
            self._ml_eq = QLabel("")
            self._ml_eq.setWordWrap(True)
            self._ml_eq.setStyleSheet(
                "font-family: 'Cascadia Mono', Consolas, monospace;")
        self._ml_eq.setMinimumHeight(80)
        self._ml_eq.setVisible(False)
        self._ml_eq_lines: list = []
        # a theta LINK in the PM sum selects its root's table row --
        # the full cross-light (gradient, contributors) from the
        # equation itself; the web shell also scrolls to the term's
        # definition line
        if hasattr(self._ml_eq, "bridge"):        # pragma: no cover
            self._ml_eq.bridge.termClicked.connect(self._ml_term_clicked)
            # an f_p/f_z click is the same ROW SELECTION -- the
            # gradient, the contributor list and the plot cursor are
            # the selection's three synchronized views
            self._ml_eq.bridge.frootClicked.connect(self._ml_term_clicked)
            # a G_eff/C_eff is the one thing in that line carrying an
            # action, and it is where the reader is already looking
            self._ml_eq.bridge.aggClicked.connect(self._ml_agg_clicked)

        self.ml_tbl = QTableWidget(0, 7)
        self.ml_tbl.setHorizontalHeaderLabels(
            ["term", "f0", "ζ", "node", "root sens",
             "φ at f_u", "dB at f_180"])
        self.ml_tbl.horizontalHeaderItem(1).setToolTip(
            "The exact root frequency; its certified formula, when one"
            " exists, lives in the equation above. The cell tooltip"
            " carries the capacitor time constants (Euler: the tau_i"
            " sum to -1/s) with their seen resistances.")
        self.ml_tbl.horizontalHeaderItem(4).setToolTip(
            "The top element movers of THIS root (attribution by\n"
            "sensitivity: d|root|/|root| in percent per percent of the\n"
            "element) -- works for the minor-combination poles that\n"
            "have no modal identity. Full list in the cell tooltip.")
        self.ml_tbl.horizontalHeaderItem(3).setToolTip(
            "Modal owner, named only where the root IS a natural\n"
            "frequency (the cancelling doublets); the true loop-gain\n"
            "poles are minor-combination roots -- modes of no single\n"
            "network -- and honestly carry no node")
        # L-J: the element ledger grouped per DEVICE, split by
        # actionability -- a netlist component's advice is exact
        # (OP-invariant), a device parameter's is directional
        # (re-bias to apply)
        self.ms_tree = QTreeWidget()
        self.ms_tree.setColumnCount(5)
        self.ms_tree.setHeaderLabels(
            ["element", "class", "value", "dPM °/%", "dGM dB/%"])
        self.ms_tree.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.ms_tree.headerItem().setToolTip(
            1, "component: a netlist R/C/L -- an OP-invariant knob,\n"
               "the advice is exact (size additions in Compensate).\n"
               "re-bias: a device model parameter -- exact at THIS\n"
               "operating point, but gm tracks bias current and caps\n"
               "track geometry: re-simulate to confirm the move.")
        self.ml_tbl.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.ml_tbl.setSelectionBehavior(QAbstractItemView.SelectRows)
        # L-K cross-light: a selected row lights its owners on the
        # Schematic pane
        self.ml_tbl.itemSelectionChanged.connect(self._ml_cross_light)
        # clicking the already-selected row again RELEASES it: the
        # right panel returns to the margin knobs, the lights go out
        self.ml_tbl.cellClicked.connect(self._ml_cell_clicked)
        # the two DEEP readings are on demand, by right-click: each
        # costs seconds of determinant work where this table must stay
        # interactive, so neither can ride along with the ledger
        self.ml_tbl.setContextMenuPolicy(Qt.CustomContextMenu)
        self.ml_tbl.customContextMenuRequested.connect(self._ml_menu)
        self.ms_tree.itemSelectionChanged.connect(self._ms_cross_light)
        # the per-corner comparison, shown once computed: rows are the
        # bound corner's terms, columns the corners, ranked by drift
        self.mlc_tbl = QTableWidget(0, 1)
        self.mlc_tbl.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.mlc_tbl.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.mlc_tbl.setVisible(False)
        # Margins keeps the DESCRIPTIVE surfaces -- the equation and
        # the corner comparison -- on a vertical splitter at full
        # height; the roots table moved to the Sensitivities tab
        # beside the element tree (field request: selecting a root
        # reads its contributors right next to it)
        self._ml_vsplit = QSplitter(Qt.Vertical)
        self._ml_vsplit.addWidget(self._ml_eq)
        self._ml_vsplit.addWidget(self.mlc_tbl)
        self._ml_vsplit.setStretchFactor(0, 1)
        self._ml_vsplit.setStretchFactor(1, 1)
        v.addWidget(self._ml_vsplit, 1)
        self._ml_tab = page
        return page

    def _sens_page(self):
        """The sensitivity WORKBENCH: the roots table beside the
        dual-mode element tree (margin knobs, or the selected
        singularity's contributors) -- a root selected on the left
        details on the right, the schematic lights up above."""
        page = QWidget()
        v = QVBoxLayout(page)
        split = QSplitter()
        split.addWidget(self.ml_tbl)
        split.addWidget(self.ms_tree)
        v.addWidget(split)
        self._ms_tab = page
        return page

    def _ml_hand_size(self, idx: int, want: int):
        """A splitter child shown for the first time inherits size 0
        -- the section then exists but cannot be seen (and a fixed
        height instead left the tables stuck): hand it room from the
        largest other section, once."""
        szs = self._ml_vsplit.sizes()
        if len(szs) <= idx or szs[idx] >= 80:
            return
        total = sum(szs) or 620
        give = min(want, total // 2)
        j = max((k for k in range(len(szs)) if k != idx),
                key=lambda k: szs[k])
        szs[j] = max(120, szs[j] - give)
        szs[idx] = give
        self._ml_vsplit.setSizes(szs)

    def _ml_row_of(self, nm: str) -> int:
        """The table row wearing the equation's term name, or -1."""
        tbl = self.ml_tbl
        for r in range(tbl.rowCount()):
            it = tbl.item(r, 0)
            if it is not None and it.text().split("  ·")[0] == nm:
                return r
        return -1

    def _ml_term_clicked(self, nm: str):
        """A theta link in the equation: select the root's row in the
        roots table, which fires the standard cross-light."""
        r = self._ml_row_of(nm)
        if r >= 0:
            self.ml_tbl.selectRow(r)

    def _ml_agg_clicked(self, nm: str):
        """A RIGHT-click on the equation's G_eff/C_eff: the SAME menu
        the row offers, opened where the symbol is printed.

        A reader looking at the equation is looking at the symbol, not
        at the table. What they get is a menu, not an immediate launch:
        the reading costs seconds, and firing on button-down leaves no
        chance to see what was about to happen, or to pick the ray test
        instead (field report)."""
        from PySide6.QtGui import QCursor

        r = self._ml_row_of(nm)
        if r < 0:
            return
        self.ml_tbl.selectRow(r)          # say WHICH root is being read
        self._ml_root_menu(r, QCursor.pos(), parent=self._ml_eq)

    @staticmethod
    def _ml_formula_tip(e) -> str:
        """What the jet read for this root: the expression when one
        certifies, and otherwise WHY it does not. A PARTIAL reading is
        worth showing -- naming half of a pole's structure beats
        naming none of it -- so the reason and the residuals travel
        with it rather than being swallowed into a blank cell."""
        rec = getattr(e, "reconstruction", None)
        if rec is None:
            if not getattr(e, "shared", ()):
                return ""
            return ("no low-entropy formula: the root is SHARED —\n"
                    + "\n".join(f"{s}  {v:+.2f} %/%"
                                for s, v in e.shared))
        out = []
        if rec.verdict == "formula":
            out.append(f"f₀ ≈ {rec.text}")
        elif rec.verdict == "partial":
            out.append(f"partly read: {rec.text}")
        else:
            out.append("no low-entropy formula: the root is SHARED")
        if rec.value:
            dev = (rec.value - e.f0_hz) / e.f0_hz if e.f0_hz else 0.0
            out.append(f"the named structure gives "
                       f"{view.eng(rec.value, 'Hz')} "
                       f"({view.eng(e.f0_hz, 'Hz')} {dev:+.0%})")
        if rec.coverage:
            out.append(
                f"coverage: reactive {rec.coverage.get('reactive', 0):+.2f}"
                f", conductive {rec.coverage.get('conductive', 0):+.2f}")
        if rec.reason:
            out.append(rec.reason)
        if rec.verdict == "shared" and rec.movers:
            out.append("\n".join(f"{s}  {v:+.2f} %/%"
                                 for s, v in rec.movers))
        ze = getattr(e, "zeta_estimate", None)
        if ze:
            out.append(f"ζ ≈ {ze['text']} = {ze['zeta_est']:.2f} "
                       f"({ze['err']:+.0%})")
        # an on-demand reading, once asked for, travels with the row:
        # the aggregate the text above names as G_eff is only a SYMBOL
        # until the port is computed, and this is where the answer to
        # that question belongs
        pr = getattr(e, "port_reading", None)
        if pr is not None:
            out.append(MarginsBenchMixin._ml_port_text(pr))
        rr = getattr(e, "ray_reading", None)
        if rr is not None:
            out.append(MarginsBenchMixin._ml_ray_text(rr))
        return "\n".join(out)

    @staticmethod
    def _ml_term(e) -> str:
        # the RHP side belongs on EVERY row that can carry it, not
        # only the unfolded ones. A RHP pole paired with a near RHP
        # zero used to read as a plain doublet here, which put the
        # table at odds with the advisor's UNSTABLE verdict
        side = "RHP " if e.rhp else ""
        if e.kind == "gain":
            return "T0"
        if e.kind == "rest":
            return f"{side}{e.count} smaller terms"
        if e.kind == "band":
            return f"{side}band ({e.count} terms)"
        if e.kind == "doublet":
            return side + ("cancelling doublet" if e.spacing < 1e-4
                           else f"doublet ({e.spacing:.0%} apart)")
        return side + e.kind + (" pair" if e.pair else "")

    def _ml_clear(self):
        self._ml_mark_freq(None)
        self.ml_tbl.setRowCount(0)
        self.ms_tree.clear()
        self.mlc_tbl.setRowCount(0)
        self.mlc_tbl.setVisible(False)
        self._ml_eq.setVisible(False)
        self._ml_eq_lines = []
        self._ml_head.setText("")
        self._ml_rows = []
        self._ms_sens = []
        self._ms_map = {}
        self.ml_whatif_btn.setEnabled(False)
        view_ = getattr(self, "sch_view", None)
        if view_ is not None:
            view_.highlight_many([])
            view_.highlight_wires([])
        if getattr(self, "_sch_contrib", None):
            self._sch_contrib = {}
            try:
                self._schematic_render()
            except Exception:                 # noqa: BLE001
                pass                          # decoration only

    def _ml_light(self, weights, net=None):
        """Light the owning instances with their WEIGHTS (strongest
        most emphasized, fading to the weakest), plus the wires that
        JOIN two or more contributors -- each wire wearing a gradient
        from its strongest attached device toward its weakest. Page-
        local rules as the trees use; `weights` maps full instance
        names to 0..1."""
        view_ = getattr(self, "sch_view", None)
        if view_ is None or not getattr(self, "_sch_hints", None):
            return
        # FULL names go to the renderer, which localizes per page --
        # the cross-light then follows navigation: a block lights when
        # it CONTAINS contributors, a page instance lights itself
        self._sch_contrib = dict(weights)
        self._ml_light_net = net
        try:
            self._schematic_render()
        except Exception:                     # noqa: BLE001
            pass                              # decoration only
        # the render's tail re-applies the overlays after any scene
        # rebuild; when the render early-returns (unchanged key) the
        # scene kept them -- this call covers that path
        self._ml_apply_wires()

    def _ml_apply_wires(self):
        """The wire-gradient OVERLAYS of the sensitivity map, rebuilt
        from the stored light state. Overlay items die with every
        scene rebuild (matches toggle, analysis state, theme, page),
        so the schematic render calls this at its tail (field report:
        toggling the match tints off vanished the gradients)."""
        view_ = getattr(self, "sch_view", None)
        if view_ is None:
            return
        weights = getattr(self, "_sch_contrib", None) or {}
        net = getattr(self, "_ml_light_net", None)
        if not weights:
            view_.highlight_wires([])
            return
        pre = self._schematic_prefix()
        local: dict[str, float] = {}
        for full, w in weights.items():
            if full.startswith(pre):
                head = full[len(pre):].split(".", 1)[0]
                if head:
                    local[head] = max(local.get(head, 0.0), float(w))
        local = {n: w for n, w in local.items()
                 if view_.bounds_of(n) is not None}
        wires = []
        defname = (self._sch_path[-1][1]
                   if getattr(self, "_sch_path", None) else None)
        defs = (getattr(self, "_sch_cin", None) or {}).get(
            "definitions", {})
        by_net: dict[str, list] = {}
        for inst in (defs.get(defname) or {}).get("instances", []):
            nm = inst.get("name")
            if nm not in local:
                continue
            for nnet in (inst.get("terminals") or {}).values():
                # global rails join everything and would light the
                # whole page
                if isinstance(nnet, str) and not nnet.endswith("!"):
                    by_net.setdefault(nnet, []).append(nm)
        for nnet, members in by_net.items():
            ms = sorted(set(members), key=lambda m: -local[m])
            if len(ms) < 2:
                continue
            # ALL contributors' scene points ride along: the view
            # restricts the drawing to the segments that join them,
            # so the net's onward run to bystanders stays unlit
            wires.append((nnet, local[ms[0]], local[ms[-1]],
                          view_.scene_center(ms[0]),
                          view_.scene_center(ms[-1]),
                          [view_.scene_center(m) for m in ms]))
        view_.highlight_wires(wires)
        if net and not local:
            loc = self._sch_localize_net(net)
            if loc is not None and view_._net_wires.get(loc):
                view_.highlight_net(loc)

    def _ml_cell_clicked(self, row, _col):
        if row == getattr(self, "_ml_last_click", None):
            self.ml_tbl.clearSelection()
            self.ml_tbl.setCurrentCell(-1, -1)
            self._ml_last_click = None
        else:
            self._ml_last_click = row

    def _ml_render_equation(self, led):
        """Build and show the hand equation. Separate from the ledger
        handler because it is rendered TWICE: once when the ledger
        lands, and again when a port expansion arrives and the bare
        G_eff in it can be replaced by the algebra behind it."""
        from ...analysis.marginledger import (margin_equation,
                                              margin_equation_latex,
                                              margin_equation_text)
        try:
            eq = margin_equation(led)
        except ValueError:
            return None
        self._ml_eq_lines = margin_equation_text(eq)
        if hasattr(self._ml_eq, "set_payload"):      # pragma: no cover
            aliases = getattr(self.controller, "sym_aliases", {})
            self._ml_eq.set_payload({
                "lines": margin_equation_latex(
                    eq, texer=lambda n: view.symbol_tex(
                        n, aliases=aliases)),
                "values": {}})
        else:
            self._ml_eq.setText("\n".join(self._ml_eq_lines))
        self._ml_eq.setVisible(True)
        self._ml_hand_size(0, 240)
        return eq

    # ---- the two on-demand readings -----------------------------
    # Both answer a question the ledger RAISES but deliberately does
    # not settle: what the aggregate G_eff actually is, and why a root
    # reads as SHARED. Each costs seconds -- port determinants, a ray
    # sweep -- so they run through the job runner like a solve instead
    # of blocking the bench, and the answer is attached to the entry
    # so the formula tooltip carries it from then on.

    def _ml_root_menu(self, row, where, parent=None):
        """The on-demand menu for ONE root, at a global point.

        Shared by the table row and by a right-click on the equation's
        G_eff: the same two readings, worded and gated once. A menu
        rather than a direct launch -- each costs seconds, and an
        action that fires the instant a button goes down gives no
        chance to see what it was going to do."""
        rows = getattr(self, "_ml_rows", ())
        if row < 0 or row >= len(rows) or self.controller is None:
            return
        e = rows[row]
        menu = QMenu(parent or self.ml_tbl)
        rec = getattr(e, "reconstruction", None)
        agg = rec.aggregate if rec is not None else None
        ok, why = self._ml_port_gate(e)
        a_port = menu.addAction(
            f"Expand {agg['symbol']} at its port" if agg
            else "Expand the aggregate at its port")
        a_port.setEnabled(ok)
        if why:
            a_port.setToolTip(why)
        a_ray = menu.addAction("Assess this root (ray test)")
        a_ray.setEnabled(bool(getattr(e, "roots", ())))
        if not getattr(e, "roots", ()):
            a_ray.setToolTip("this term has no root to follow")
        act = menu.exec(where)
        if act is a_port and a_port.isEnabled():
            self.ml_port_expand(row)
        elif act is a_ray and a_ray.isEnabled():
            self.ml_assess(row)

    def _ml_menu(self, pos):
        row = self.ml_tbl.rowAt(pos.y())
        self._ml_root_menu(row, self.ml_tbl.viewport().mapToGlobal(pos))

    @staticmethod
    def _ml_port_gate(e):
        """Whether this root's aggregate can be expanded at a port,
        and if not, why. Only a CONDUCTANCE has the driving-point
        route: the port is read from two determinants of the s = 0
        matrix, which is purely conductive. A C_eff names a
        capacitance and has no dual here, so the action says so rather
        than offering a computation that returns nothing."""
        rec = getattr(e, "reconstruction", None)
        agg = rec.aggregate if rec is not None else None
        if not agg:
            return False, "this root's reading names no aggregate"
        if agg["unit"] != "S":
            return False, (f"{agg['symbol']} is a capacitance; the "
                           "port route reads conductances only")
        return True, ""

    def _ml_on_demand(self, row, what):
        """Shared launch path: the two readings take the same (probe,
        f0) and differ only in the controller call they make."""
        rows = getattr(self, "_ml_rows", ())
        if self.controller is None or row < 0 or row >= len(rows):
            return None
        probe = self.probe_combo.currentText()
        if not probe:
            self.statusBar().showMessage("no loop probe in this design")
            return None
        e = rows[row]
        # keep 3, not the API's 5. On the folded cascode's 20 kHz pole
        # the two agree exactly -- same value, same ratio, same
        # verdict -- but cost 22 s against 562 s. The extra symbols
        # buy nothing an interactive action can spend 9 minutes on.
        call = ((lambda pr, f0: self.controller.port_expression(
            pr, f0, max_keep=3)) if what == "port"
            else self.controller.assess_root)
        label = ("the port conductance at " if what == "port"
                 else "the ray test at ") + view.eng(e.f0_hz, "Hz")

        def fn(_cb):
            return {"row": row, "what": what,
                    "out": call(probe, e.f0_hz)}

        return fn, label

    def ml_port_expand(self, row):
        got = self._ml_on_demand(row, "port")
        if got is not None:
            fn, label = got
            self._launch(fn, label + "…", est_s=None,
                         on_done=self._on_ml_reading)

    def ml_assess(self, row):
        got = self._ml_on_demand(row, "ray")
        if got is not None:
            fn, label = got
            self._launch(fn, label + "…", est_s=None,
                         on_done=self._on_ml_reading)

    def ml_reading_sync(self, row, what):
        """Blocking, for tests and scripting."""
        got = self._ml_on_demand(row, what)
        if got is None:
            return None
        fn, _label = got
        out = fn(None)
        self._on_ml_reading(out)
        return out

    def _on_ml_reading(self, res):
        self._tick.stop()
        self._t0 = None
        self.progress.hide()
        self.cancel_btn.hide()
        self._job_finished()
        rows = getattr(self, "_ml_rows", ())
        row, what, out = res["row"], res["what"], res["out"]
        if row >= len(rows):
            return                   # the ledger was refilled under us
        e = rows[row]
        rewrote = False
        if what == "port":
            e.port_reading = out
            self.log(self._ml_port_text(out))
            # the expansion belongs in the EQUATION, where the bare
            # G_eff is printed and where a reader looks for it -- the
            # log scrolls away
            aliases = getattr(self.controller, "sym_aliases", {})
            exp = self._ml_expansion(
                out, lambda n: view.symbol_tex(n, aliases=aliases))
            rec = getattr(e, "reconstruction", None)
            if exp and rec is not None:
                rec.expansion = exp
                if getattr(self, "_ml_led", None) is not None:
                    self._ml_render_equation(self._ml_led)
                    rewrote = True
        else:
            e.ray_reading = out
            self.log(self._ml_ray_text(out))
        it = self.ml_tbl.item(row, 1)
        if it is not None:
            it.setToolTip(self._ml_formula_tip(e))
        # _launch brings the Log forward while the job runs; the
        # completion returns to wherever the ANSWER now is. A port
        # reading that rewrote the equation belongs on the bench --
        # but a ray verdict, and a port reading whose branches could
        # not all be named, exist only in the log, and dropping the
        # user on an unchanged pane would look like nothing happened.
        self.tabs.setCurrentWidget(self._ml_tab if rewrote
                                   else self.logview)
        if rewrote:
            self.ml_tbl.selectRow(row)

    @staticmethod
    def _ml_tex_expr(s, texer):
        """LaTeX for one branch's printed form.

        branch_form emits a simple grammar -- groups joined by '*',
        at most one '/' -- so the two sides are lifted into a real
        fraction, and every identifier goes through the SAME texer the
        rest of the equation uses. Pasting the raw text in would put
        gds\\_I0\\_NM14 beside the aliased names in the same line."""
        import re

        def tok(part):
            # an engineering value with its unit -- "15.6 uS", thin
            # space, from units.eng -- is STASHED before the identifier
            # pass, which would otherwise read "uS" as a symbol name and
            # hand it to the texer. Typeset as 15.6\,\mu\mathrm{S}.
            kept = []

            def stash(m):
                num, pre, unit = m.groups()
                pre = {"u": r"\mu"}.get(pre, pre)
                pre = rf"\mathrm{{{pre}}}" if pre and pre != r"\mu" \
                    else pre
                kept.append(rf"{num}\,{pre}\mathrm{{{unit}}}")
                return f"\x00{len(kept) - 1}\x00"

            part = re.sub(r"(\d+(?:\.\d+)?) ([afpnumkMGT]?)"
                          r"(Hz|S|F|V|A|W|Ohm)\b", stash, part)
            # not preceded by a word char or a dot, so the 'e' of
            # 1.54e-05 is left alone
            part = re.sub(r"(?<![\w.])[A-Za-z_][A-Za-z_0-9]*",
                          lambda m: texer(m.group(0)), part)
            part = re.sub(r"\x00(\d+)\x00",
                          lambda m: kept[int(m.group(1))], part)
            return part.replace("*", r"\cdot ")

        if "/" in s:
            top, bot = s.split("/", 1)
            return rf"\frac{{{tok(top)}}}{{{tok(bot)}}}"
        return tok(s)

    @classmethod
    def _ml_expansion(cls, out, texer=None):
        """The branch decomposition as one printable sum, or None.

        Only when EVERY branch was named: a sum with a hole in it is
        not the expression, and printing a partly-named one in the
        equation would state as fact something the port did not
        establish. The unnamed case still reaches the log, where it is
        reported as unnamed."""
        br = out.get("branches") or ()
        if not out.get("matches") or not br:
            return None
        if any(not b.get("text") for b in br):
            return None
        tx = texer or (lambda n: n.replace("_", r"\_"))
        parts = [b["text"] for b in br]
        return {"text": " + ".join(parts),
                "tex": " + ".join(cls._ml_tex_expr(p, tx)
                                  for p in parts)}

    @staticmethod
    def _ml_port_text(out) -> str:
        """The port answer in words. `matches` is the whole point: it
        says the pole really IS that node's time constant, and where
        it is false the algebra is reported as not applying rather
        than shown as though it did."""
        agg = out.get("aggregate") or {}
        sym = agg.get("symbol", "the aggregate")
        fixed = view.eng(agg.get("value", 0.0), "S")
        lines = [f"{sym} at the {out['port']} port, in "
                 + ", ".join(out["keep"])
                 + f"  [{out['seconds']:.1f}s]"]
        if out["matches"]:
            lines.append(f"  {sym} = {view.eng(out['value'], 'S')}, the "
                         f"identity fixed {fixed} ({out['ratio']:.2f}x)"
                         " — the pole IS this node's time constant")
            # the branch decomposition, NOT the raw num/den: those come
            # back uncancelled and read as hundred-digit integers, which
            # is not something to put in front of a designer
            br = out.get("branches") or ()
            if br:
                lines.append(f"  {len(br)} parallel branches:")
                for b in br:
                    lines.append(
                        f"    {b['text'] or '<not named>'}"
                        f"   {view.eng(b['conductance'], 'S')}")
        else:
            lines.append(
                f"  {sym} = {view.eng(out['value'], 'S')} against "
                f"{fixed} ({out['ratio']:.2f}x): the port conductance "
                "does NOT match, so this root is not that node's time "
                "constant and the aggregate stands unexpanded")
        return "\n".join(lines)

    @staticmethod
    def _ml_ray_text(v) -> str:
        """The ray verdict in words. The verdict says itself in one
        sentence; what the log adds under it is the winning ray -- the
        walk that gathers the root, which is where compensation
        belongs."""
        lines = [v.describe()]
        for p in getattr(v, "path", ()) or ():
            lines.append(f"  x{p.scale:g}: {view.eng(p.f0_hz, 'Hz')}, "
                         f"L {p.localization:.0%} at {p.node}")
        return "\n".join(lines)

    def _ml_mark_freq(self, f_hz, label: str | None = None):
        """A vertical cursor at the selected root's EXACT frequency,
        on every Bode axis, wearing the term's name (f_p1, f_z2) at
        the frequency axis -- the third synchronized view of a root
        selection (schematic gradient, contributor list, plot
        cursor). None clears."""
        for ln in getattr(self, "_ml_fmarks", ()):
            try:
                ln.remove()
            except Exception:                 # noqa: BLE001
                pass                          # axes were re-plotted
        self._ml_fmarks = []
        canvas = getattr(self, "canvas", None)
        if canvas is None:
            return
        if f_hz:
            from .. import theme
            axes = canvas.figure.axes
            for ax in axes:
                self._ml_fmarks.append(
                    ax.axvline(float(f_hz), color=theme.VERMILION,
                               lw=0.8, ls="--", alpha=0.85))
            if label and axes:
                sub = label[0], label[1:]
                self._ml_fmarks.append(axes[-1].annotate(
                    rf"$f_{{{sub[0]}{sub[1]}}}$",
                    xy=(float(f_hz), 0.0),
                    xycoords=("data", "axes fraction"),
                    xytext=(3, 3), textcoords="offset points",
                    fontsize=7, color=theme.VERMILION))
        canvas.draw_idle()

    def _ml_cross_light(self):
        rows = getattr(self, "_ml_rows", [])
        if not self.ml_tbl.selectedItems():
            # deselection: the right panel returns to the margin
            # knobs, the schematic light goes out, the plot cursor
            # leaves
            self._ml_last_click = None
            if getattr(self, "_ms_sens", None):
                self._ms_fill(self._ms_sens)
            self._ml_light({}, net=None)
            self._ml_mark_freq(None)
            return
        r = self.ml_tbl.currentRow()
        if not (0 <= r < len(rows)):
            return
        e = rows[r]
        it0 = self.ml_tbl.item(r, 0)
        nm = (it0.text().split("  ·")[0] if it0 else "")
        self._ml_mark_freq(getattr(e, "f0_hz", None),
                           label=nm if nm[:1] in ("p", "z")
                           and nm[1:].isdigit() else None)
        # the right panel becomes THIS singularity's contributor list
        # -- the detailed, investigable twin of the schematic gradient
        if e.root_sens:
            it = self.ml_tbl.item(r, 0)
            self._ms_fill_root(e, it.text() if it else None)
        # ALL stored root sensitivities light, weighted by relative strength --
        # the gradient replaces any cutoff: the strongest contributors
        # read most emphasized and the tail fades instead of being
        # sliced (a fixed top-3 once lit PM4 while leaving its own
        # cascode PM11 dark on an exact tie). Only the HOT members of
        # a matched symbol carry their symbol's weight -- lighting the
        # whole match group was misleading.
        # The map shows CONTRIBUTION -- the same gross the contributor
        # list ranks by: per instance, the sum of |df0| over the
        # symbols acting through it, each split by its share. One
        # number, so the brightest device on the page is the top row
        # of the list. It used to take the max |S| per symbol and fold
        # capacitor tau-shares in, which ranked the page by a third
        # quantity neither view displayed.
        weights = self._ml_contrib_weights(e, self._ms_map)
        self._ml_light(weights, net=e.node or None)

    @staticmethod
    def _ml_contrib_weights(e, ms_map, *, floor: float = 0.03) -> dict:
        """{instance: 0..1} for the schematic gradient -- each device's
        CONTRIBUTION, the sum of |df0| over the symbols acting through
        it split by their shares, normalised to the strongest.

        Pure, so it is testable without a schematic: the miller
        fixture carries no hints, and a guard that only checked the
        map once it was lit was checking nothing there."""
        weights: dict[str, float] = {}
        for m in e.root_sens:
            insts = m.get("insts") or ms_map.get(m["symbol"], ())
            if not insts:
                continue
            shares = m.get("shares") or {i: 1.0 / len(insts)
                                         for i in insts}
            for i in insts:
                weights[i] = weights.get(i, 0.0) + abs(
                    m["df0"]) * float(shares.get(i, 0.0))
        w0 = max(weights.values(), default=1.0) or 1.0
        weights = {i: w / w0 for i, w in weights.items()}
        # a floor keeps the deep tail from speckling the page with
        # imperceptible boxes; everything above it fades honestly
        return {i: w for i, w in weights.items() if w >= floor}

    def _ms_cross_light(self):
        if getattr(self, "_ms_filling", False):
            return
        it = self.ms_tree.currentItem()
        if it is None:
            return
        name = it.text(0)
        if it.text(1) != "device":
            insts = self._ms_map.get(name, ())
            if insts:
                name = insts[0]
            elif "_" in name:
                # a re-spelled row names its acting device directly
                name = self._sym_device(name).replace("_", ".")
            else:
                name = None
        if name:
            # the SAME selection a click in the Instances tree makes:
            # the tree row selects (and scrolls into view), the
            # standard single-device highlight draws, the OP card
            # follows -- the gradient context stays underneath
            self._goto_instance(name)

    def margins_to_whatif(self, *, solve: bool = True, top: int = 3):
        """L-K bridge: tick the strongest margin movers into the keep
        set and run the hybrid loop solve -- the What-if sliders then
        move exactly the parameters that move the margin, with live
        PM."""
        sens = getattr(self, "_ms_sens", None) or []
        if not sens or self.controller is None:
            self.statusBar().showMessage("compute the margin ledger first")
            return []
        want = {r["symbol"] for r in sens[:top]}
        self._clear_keep_ticks()
        self._filling = True
        try:
            found = []
            for i in range(self.keep_tbl.rowCount()):
                it = self.keep_tbl.item(i, 0)
                if it is not None and it.text() in want:
                    it.setCheckState(Qt.Checked)
                    found.append(it.text())
        finally:
            self._filling = False
        self._apply_keep_filter()
        self._update_estimate()
        self._refresh_solve_hint()
        if not found:
            self.statusBar().showMessage(
                "none of the top knobs are in the keep table")
            return []
        self.statusBar().showMessage(
            "modeling " + ", ".join(found) + " — hybrid loop solve")
        if solve:
            self.solve()
        return found

    def compute_margins(self):
        if self.controller is None:
            return
        probe = self.probe_combo.currentText()
        if not probe:
            self.statusBar().showMessage("no loop probe in this design")
            return

        cplx = self.ml_cplx.currentIndex() + 1

        def fn(_cb):
            return self.controller.margin_ledger(probe,
                                                 complexity=cplx)

        self._launch(fn, f"margin ledger at {probe}…", est_s=None,
                     on_done=self._on_margins_done)

    def margins_sync(self, probe, complexity: int = 1):
        """Blocking, for tests and scripting."""
        out = self.controller.margin_ledger(probe,
                                            complexity=complexity)
        self._on_margins_done(out)
        return out

    def _on_margins_done(self, out):
        self._tick.stop()
        self._t0 = None
        self.progress.hide()
        self.cancel_btn.hide()
        self._job_finished(self.ledger_btn)
        led, sens = out["ledger"], out["sensitivities"]
        head = []
        if led.pm_deg is not None:
            head.append(f"PM {led.pm_deg:.1f}° at "
                        f"{view.eng(led.f_unity_hz, 'Hz')}")
        if led.gm_db is not None:
            head.append(f"GM {led.gm_db:.1f} dB at "
                        f"{view.eng(led.f_180_hz, 'Hz')}")
        res = [f"{r:.1e}" for r in (led.phase_residual_deg,
                                    led.mag_residual_db) if r is not None]
        stab = getattr(led, "stability", None)
        self._ml_head.setText(
            "  /  ".join(head)
            + (f"   (certificate residuals {', '.join(res)})" if res
               else "  no crossing in band")
            + (f"\n{stab['verdict'].upper()}: {stab['reason']}"
               if stab else ""))

        from ...analysis.marginledger import margin_equation_names

        self._ml_led = led            # kept: a port expansion re-renders
        eq = self._ml_render_equation(led)

        # the anchor T0 lives in the equation above with the other
        # definitions; the table keeps the singularities only, in the
        # equation's ascending-frequency order, remainder rows last
        entries = [e for e in led.fold() if e.kind != "gain"]

        def _order(e):
            if e.kind in ("pole", "zero"):
                return (1, e.f0_hz)
            if e.kind == "doublet":
                return (2, e.f0_hz)
            return (3, 0.0)                       # rest, band


        entries = sorted(entries, key=_order)
        self._ml_rows = entries
        self._ms_sens = sens
        self._ms_map = {r["symbol"]: r["insts"] for r in sens}
        self._ms_knob = {r["symbol"]: r["knob"] for r in sens}
        self._ms_val = {r["symbol"]: r["value"] for r in sens}
        self.ml_whatif_btn.setEnabled(True)
        # the SAME term names the equation uses (p1, z1, ... in
        # frequency order), keyed by kind and exact frequency, so a
        # theta line above finds its row below at a glance
        name_of = {}
        if eq is not None:
            for nm, tt in zip(margin_equation_names(eq), eq["terms"]):
                name_of[(tt["kind"], tt["f0_hz"])] = nm
        tbl = self.ml_tbl
        tbl.setRowCount(len(entries))
        # the modal-owner column carries a name only for entries whose
        # root IS a natural frequency (the ghost doublets) -- on a
        # folded view those usually sink into the remainder, so an
        # all-empty column hides rather than sitting as clutter
        tbl.setColumnHidden(3, not any(e.node for e in entries))
        for i, e in enumerate(entries):
            term = self._ml_term(e)
            nm = name_of.get((e.kind, e.f0_hz))
            if nm:
                term = f"{nm}  ·  {term}"
            elif e.kind in ("doublet", "rest", "band"):
                # these sum into the equation's remainder constant
                term += "  (in remainder)"
            node = f"{e.node} ({e.share:.0%})" if e.node else ""
            f0txt = view.eng(e.f0_hz, "Hz") if e.f0_hz else ""
            if e.kind == "doublet" and e.spacing >= 1e-4:
                f0txt += f" → {view.eng(e.f0b_hz, 'Hz')}"
            moved = " / ".join(f"{m['symbol']} {m['df0']:+.2f}"
                               for m in e.root_sens[:2])
            cells = (term,
                     f0txt,
                     f"{e.zeta:.2f}" if e.zeta is not None else "",
                     node,
                     moved,
                     f"{e.phase_deg:+.1f}°"
                     if e.phase_deg is not None else "—",
                     f"{e.mag_db:+.1f}" if e.mag_db is not None else "—")
            for j, text in enumerate(cells):
                it = QTableWidgetItem(text)
                if j == 1:
                    # what the jet read here, verdict and residuals --
                    # a formula, the part of one, or the sharers named
                    # instead of a fit
                    tip = self._ml_formula_tip(e)
                    if tip:
                        it.setToolTip(tip)
                if j == 0 and e.kind == "rest" and e.members:
                    # enumerate WHO was folded, so this row reconciles
                    # against the Expression tab's factored form
                    it.setToolTip("folded (net "
                                  + (f"{e.phase_deg:+.1f}°"
                                     if e.phase_deg is not None else "—")
                                  + "):\n" + "\n".join(
                        self._ml_term(m) + "  " + view.eng(m.f0_hz, "Hz")
                        + (f" → {view.eng(m.f0b_hz, 'Hz')}"
                           if m.kind == "doublet" and m.spacing >= 1e-4
                           else "")
                        + (f"  ({m.phase_deg:+.2f}°)"
                           if m.phase_deg is not None else "")
                        for m in e.members))
                if j == 1 and e.taus:
                    tt_txt = "\n".join(
                        f"{tt['symbol']}: {tt['share']:+.0%} of tau, "
                        f"R_eff {view.eng(tt['r_eff'], 'Ω')}"
                        for tt in e.taus)
                    prev = it.toolTip()           # the shared-root note
                    it.setToolTip(f"{prev}\n\n{tt_txt}" if prev else tt_txt)
                if j == 4 and e.root_sens:
                    it.setToolTip("\n".join(
                        f"{m['symbol']}: df0 {m['df0']:+.3f} %/%, "
                        f"dζ {m['dzeta']:+.4f} per %"
                        for m in e.root_sens[:8]))
                tbl.setItem(i, j, it)

        self._ms_fill(sens)
        self.tabs.setCurrentWidget(self._ml_tab)
        self.statusBar().showMessage(
            f"margin ledger: {len(entries)} terms, "
            f"{len(sens)} element rows")

    def _ms_local_name(self, symbol: str, under: str | None) -> str:
        """The symbol as the DEVICE IT ACTS THROUGH spells it: a
        matched group's shared symbol is named after one member
        (keep-name), and under another member's header that name is a
        misnomer -- the sensitivity views show physical origin, so
        the parameter is re-spelled onto the hot device (field
        request: no match-group indirection in the map or the list;
        formulas keep the shared symbol)."""
        if not under:
            return symbol
        own = self._sym_device(symbol).replace("_", ".")
        if own == under or "_" not in symbol:
            return symbol
        head = symbol[: len(symbol) - len(self._sym_device(symbol)) - 1]
        return f"{head}_{under.replace('.', '_')}"

    @staticmethod
    def _ms_cells(label, r):
        dpm = r["dpm_deg_per_pct"]
        dgm = r["dgm_db_per_pct"]
        return [label,
                "component" if r["knob"] == "component" else "re-bias",
                view.eng(r["value"], view.op_unit(r["symbol"]), sig=4),
                f"{dpm:+.4f}" if dpm is not None else "—",
                f"{dgm:+.4f}" if dgm is not None else "—"]

    def _ms_fill_root(self, e, label=None):
        """The right panel as ONE singularity's contributor list: its
        root sensitivities (semi-relative, x ds/dx) grouped per
        device, the detailed twin of the schematic's gradient. df0 is
        the percent shift of |root| per percent of the element, dzeta
        the damping shift per percent; a capacitor row's tooltip adds
        its tau share and seen resistance."""
        self._ms_filling = True
        try:
            tree = self.ms_tree
            tree.clear()
            tree.setHeaderLabels(
                [f"contributors to {label or self._ml_term(e)} "
                 f"({view.eng(e.f0_hz, 'Hz')})", "class", "value",
                 "df0 %/%", "dζ per %", "re-bias %/%"])
            hdr = tree.headerItem()
            hdr.setToolTip(
                0, "Click the selected singularity again (or "
                   "Ctrl+click it) to return to the margin-knob "
                   "view (dPM/dGM per element).")
            # Two questions, two columns, and the list is RANKED by the
            # first. df0 on a device row is its CONTRIBUTION -- the sum
            # of |df0| over its parameters, how much of the root runs
            # through it. re-bias is its LEVERAGE -- the same
            # sensitivities weighted by d ln(param)/d ln(bias), i.e.
            # what re-biasing the device 1% actually does. They differ
            # exactly when the device's parameters oppose each other:
            # a cascode whose gds sits in the numerator and gm in the
            # denominator is central to the formula (high
            # contribution) yet nearly inert to re-bias. Both are
            # true; neither hides the other.
            hdr.setToolTip(
                3, "device rows: sum of |df0| over the device's "
                   "parameters -- how much of the root runs through "
                   "it. The list is ranked by this.")
            hdr.setToolTip(
                5, "device rows: the same sensitivities weighted by "
                   "d ln(param)/d ln(bias) -- what re-biasing this "
                   "device by 1% does to the root. Near zero with a "
                   "large contribution means its parameters cancel.")
            taus = {t["symbol"]: t for t in e.taus}
            groups: dict[str, list] = {}
            singles = []
            for r in e.root_sens:
                insts = r.get("insts") or ()
                knob = self._ms_knob.get(r["symbol"], "device")
                if knob == "component" or not insts:
                    singles.append((r, 1.0))
                else:
                    # a shared symbol goes under EVERY hot member, by
                    # its share -- the map lights them the same way
                    shares = r.get("shares") or {
                        i: 1.0 / len(insts) for i in insts}
                    for inst in insts:
                        groups.setdefault(inst, []).append(
                            (r, float(shares.get(inst, 0.0))))

            def _w(r, under):
                knob = self._ms_knob.get(r["symbol"], "device")
                region = None
                if under and self.controller is not None:
                    try:
                        region = self.controller.device_op(
                            under).get("region")
                    except Exception:                # noqa: BLE001
                        region = None
                return rebias_exponent(r["symbol"], region,
                                       explicit=(knob == "component"))

            def _row(r, share, under=None):
                knob = self._ms_knob.get(r["symbol"], "device")
                val = self._ms_val.get(r["symbol"])
                label = self._ms_local_name(r["symbol"], under)
                d = r["df0"] * share
                it = QTreeWidgetItem(
                    [label,
                     "component" if knob == "component" else "re-bias",
                     view.eng(val, view.op_unit(r["symbol"]), sig=4)
                     if val is not None else "",
                     f"{d:+.3f}", f"{r['dzeta'] * share:+.4f}",
                     f"{_w(r, under) * d:+.3f}"])
                tips = []
                if label != r["symbol"]:
                    tips.append(f"in formulas: the matched group's "
                                f"shared symbol {r['symbol']}")
                if share < 0.999:
                    tips.append(f"{share:.0%} of that symbol's "
                                "sensitivity acts through this device")
                tt = taus.get(r["symbol"])
                if tt is not None:
                    tips.append(f"tau share {tt['share']:+.0%}, "
                                f"R_eff {view.eng(tt['r_eff'], 'Ω')}")
                if tips:
                    it.setToolTip(0, "\n".join(tips))
                return it

            tops = []
            for r, share in singles:
                tops.append((abs(r["df0"]), _row(r, share)))
            for inst, rows in groups.items():
                rows.sort(key=lambda rs: -abs(rs[0]["df0"] * rs[1]))
                gross = sum(abs(r["df0"] * s) for r, s in rows)
                lever = sum(_w(r, inst) * r["df0"] * s for r, s in rows)
                head = QTreeWidgetItem(
                    [inst, "device", "", f"{gross:.3f}",
                     f"{sum(r['dzeta'] * s for r, s in rows):+.4f}",
                     f"{lever:+.3f}"])
                head.setToolTip(3, "contribution: sum of |df0| over "
                                   "this device's parameters")
                head.setToolTip(5, "leverage: re-biasing this device "
                                   "1% moves the root by this")
                for r, s in rows:
                    head.addChild(_row(r, s, under=inst))
                tops.append((gross, head))
            tops.sort(key=lambda t: -t[0])
            for k, (_, it) in enumerate(tops[:14]):
                tree.addTopLevelItem(it)
                if k < 3 and it.childCount():
                    it.setExpanded(True)
        finally:
            self._ms_filling = False

    def _ms_fill(self, sens):
        """The element rows grouped per device: components (exact,
        OP-invariant knobs) stand alone; a device gathers its
        parameters under one header whose dPM/dGM is the SIGNED SUM --
        the effect of every parameter of that device up one percent
        together. Ranked by the strongest single handle inside."""
        self._ms_filling = True
        try:
            self._ms_fill_margin(sens)
        finally:
            self._ms_filling = False

    def _ms_fill_margin(self, sens):
        tree = self.ms_tree
        tree.clear()
        tree.setHeaderLabels(
            ["element", "class", "value", "dPM °/%", "dGM dB/%"])
        groups: dict[str, list] = {}
        singles = []
        for r in sens[:40]:
            if r["knob"] == "component" or not r["insts"]:
                singles.append(r)
            else:
                groups.setdefault(r["insts"][0], []).append(r)
        tops = []
        for r in singles:
            it = QTreeWidgetItem(self._ms_cells(r["symbol"], r))
            tops.append((abs(r["dpm_deg_per_pct"] or 0.0), it))
        for inst, rows in groups.items():
            rows.sort(key=lambda r: -abs(r["dpm_deg_per_pct"] or 0.0))
            tot_pm = sum(r["dpm_deg_per_pct"] or 0.0 for r in rows)
            tot_gm = sum(r["dgm_db_per_pct"] or 0.0 for r in rows)
            head = QTreeWidgetItem(
                [inst, "device", "", f"{tot_pm:+.4f}", f"{tot_gm:+.4f}"])
            head.setToolTip(
                3, "every parameter of this device up 1% together")
            for r in rows:
                it = QTreeWidgetItem(
                    self._ms_cells(self._ms_local_name(r["symbol"],
                                                       inst), r))
                if it.text(0) != r["symbol"]:
                    it.setToolTip(0, f"in formulas: the matched "
                                     f"group's shared symbol "
                                     f"{r['symbol']}")
                head.addChild(it)
            tops.append(
                (abs(rows[0]["dpm_deg_per_pct"] or 0.0), head))
        tops.sort(key=lambda t: -t[0])
        for k, (_, it) in enumerate(tops[:14]):
            tree.addTopLevelItem(it)
            if k < 3 and it.childCount():
                it.setExpanded(True)

    def compare_margins(self):
        if self.controller is None or not self.controller.op_points:
            return
        probe = self.probe_combo.currentText()
        if not probe:
            self.statusBar().showMessage("no loop probe in this design")
            return
        n = len(self.controller.op_points)

        def fn(_cb):
            def prog(msg, done=None, total=None):
                self.worker_note(msg)
                if done is not None and total:
                    _cb(done, total)

            out = self.controller.margin_ledger_set(probe, progress=prog)
            from ...analysis.marginledger import compare_ledgers
            return compare_ledgers(out["ledgers"], out["labels"],
                                   ref_index=out["ref_index"])

        self._launch(fn, f"margin ledgers over {n} operating points…",
                     est_s=None, on_done=self._on_margins_cmp)

    def margins_cmp_sync(self, probe):
        """Blocking, for tests and scripting."""
        from ...analysis.marginledger import compare_ledgers

        out = self.controller.margin_ledger_set(probe)
        cmp_ = compare_ledgers(out["ledgers"], out["labels"],
                               ref_index=out["ref_index"])
        self._on_margins_cmp(cmp_)
        return cmp_

    def _on_margins_cmp(self, cmp_):
        self._tick.stop()
        self._t0 = None
        self.progress.hide()
        self.cancel_btn.hide()
        self._job_finished(self.ledger_cmp_btn)
        labels = cmp_["labels"]
        n = len(labels)
        tbl = self.mlc_tbl
        tbl.setColumnCount(n + 2)
        tbl.setHorizontalHeaderLabels(["term"] + labels + ["Δ"])
        head = [("PM (°)", cmp_["pm"], "{:+.1f}"),
                ("f_u", cmp_["fu"], None),
                ("GM (dB)", cmp_["gm"], "{:+.1f}")]
        rows = cmp_["rows"]
        tbl.setRowCount(len(head) + len(rows) + 1)
        r = 0
        for name, vals, fmt in head:
            tbl.setItem(r, 0, QTableWidgetItem(name))
            for k, vv in enumerate(vals):
                text = ("—" if vv is None
                        else view.eng(vv, "Hz") if fmt is None
                        else fmt.format(vv))
                tbl.setItem(r, 1 + k, QTableWidgetItem(text))
            vs = [vv for vv in vals if vv is not None]
            if fmt and len(vs) > 1:
                tbl.setItem(r, n + 1,
                            QTableWidgetItem(f"{max(vs) - min(vs):.1f}"))
            r += 1
        for row in rows:
            e = row["term"]
            it = QTableWidgetItem(self._ml_term(e))
            if e.node:
                it.setToolTip(f"{e.node} ({e.share:.0%})")
            tbl.setItem(r, 0, it)
            for k, c in enumerate(row["cells"]):
                it = QTableWidgetItem("—" if c is None else f"{c:+.1f}")
                if row["f0s"][k]:
                    it.setToolTip(view.eng(row["f0s"][k], "Hz"))
                tbl.setItem(r, 1 + k, it)
            tbl.setItem(r, n + 1,
                        QTableWidgetItem(f"{row['delta']:.1f}"))
            r += 1
        tbl.setItem(r, 0, QTableWidgetItem("other terms"))
        for k, o in enumerate(cmp_["other"]):
            tbl.setItem(r, 1 + k, QTableWidgetItem(f"{o:+.1f}"))
        os_ = cmp_["other"]
        tbl.setItem(r, n + 1,
                    QTableWidgetItem(f"{max(os_) - min(os_):.1f}"))
        tbl.setVisible(True)
        self._ml_hand_size(1, 220)
        self.tabs.setCurrentWidget(self._ml_tab)
        self.statusBar().showMessage(
            f"per-corner ledger: {len(rows)} terms over {n} points; "
            "rows ranked by drift")
