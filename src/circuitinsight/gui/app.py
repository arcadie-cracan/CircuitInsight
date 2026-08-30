"""PySide6 desktop front end. A thin window over `SessionController` + `view`:
open a CIN + psf, declare matched pairs, pick input/output, choose a keep-set
(ranked by band sensitivity, gated by a solve-time estimate), solve or
error-budget-simplify (in a worker thread), and export a report.
Requires `circuitinsight[gui]` (PySide6).
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from PySide6.QtCore import QEvent, Qt, QTimer
from PySide6.QtGui import QActionGroup, QKeySequence
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QCheckBox, QComboBox, QDoubleSpinBox,
    QFileDialog, QFrame, QGroupBox, QHBoxLayout, QLabel, QMenu, QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton, QProgressBar, QScrollArea, QSizePolicy,
    QSplitter, QSplitterHandle, QTableWidget,
    QTableWidgetItem, QTabWidget, QTextEdit, QToolBar, QVBoxLayout, QWidget,
)
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg, NavigationToolbar2QT)
from matplotlib.figure import Figure

from ..session import SessionController
from . import devtree, exprweb, summaryweb, theme, view
from .benches import (CompensateBenchMixin, GFTBenchMixin,
                      MarginsBenchMixin, ModesBenchMixin,
                      ReduceBenchMixin, SchematicMixin, WhatIfMixin)
from .jobs import JobRunnerMixin, _Worker
from .persistence import PersistenceMixin
from .report import ReportMixin
from .rangeslider import RangeSlider


#: (name, tooltip, requirement) — one row per tool. The Tool dropdown
#: shows only the rows the loaded run can GROUND in simulator data:
#: "loop" needs stb results in the run (or the user's explicit
#: declaration that the AC data is a return-ratio capture), "modes"
#: additionally needs two declared iprobes for the 2x2 matrix, "" only
#: the AC/OP import. A reconstructed loop quantity is never offered
#: without simulator truth to check it against.
_TOOLS = (
    ("Transfer", "H(s): exact, budgeted, or textbook", ""),
    ("Loop gain", "T(s), margins, probe adequacy — needs stb results "
     "(or a declared return-ratio AC)", "loop"),
    ("Compensate", "suggest Cc / Rz / multi-branch networks — needs stb "
     "results (or a declared return-ratio AC)", "loop"),
    ("Modes", "DM/CM 2x2 loop matrix — needs stb results and two "
     "iprobes", "modes"),
    ("GFT", "error terms of one loop — needs stb results (or a declared "
     "return-ratio AC)", "loop"),
    ("Impedance", "Zin/Zout at a port", ""),
    ("Reduce circuit", "AC grounds, dead sources, lumping", ""),
)


class _RestoringHandle(QSplitterHandle):
    """A splitter handle that brings a collapsed pane back on
    double-click: the pane dragged to nothing is the one the handle
    sits against, and a double-click is the gesture everyone tries."""

    def mouseDoubleClickEvent(self, ev):
        sp = self.splitter()
        sizes = sp.sizes()
        if sizes and 0 in sizes:
            default = getattr(sp, "_default_sizes", None) or                 [1] * len(sizes)
            total = sum(sizes) or sum(default)
            share = [d / sum(default) for d in default]
            sp.setSizes([max(1, int(total * s)) for s in share])
            ev.accept()
            return
        super().mouseDoubleClickEvent(ev)


class RestoringSplitter(QSplitter):
    """QSplitter whose handles restore a collapsed pane on double-click;
    `_default_sizes` holds the proportions to come back to."""

    def createHandle(self):
        return _RestoringHandle(self.orientation(), self)


class MainWindow(JobRunnerMixin, PersistenceMixin, ReportMixin,
                 WhatIfMixin, ReduceBenchMixin,
                 CompensateBenchMixin, ModesBenchMixin,
                 MarginsBenchMixin, GFTBenchMixin, SchematicMixin,
                 QMainWindow):
    #: tests point this at a temp .ini so they never touch the registry
    settings_path: str | None = None

    def __init__(self):
        super().__init__()
        self.setWindowTitle("CircuitInsight")
        self.controller: SessionController | None = None
        self.result = None
        self._thread: _Worker | None = None
        self._filling = False
        self._splitters_restored = False
        self._report_sections: list[str] = []
        self._match_groups: list[tuple[str, ...]] = []
        self._acg_pending: set[str] = set()   # nets awaiting scan pricing
        # progress state: a solve that cannot report units still ticks an
        # elapsed clock, so "working" is distinguishable from "wedged"
        self._t0: float | None = None
        self._phase = ""
        self._phase_units: tuple = (None, None)
        self._est_s: float | None = None      # keep-set label
        self._run_est: float | None = None    # the RUNNING job
        self._live_est: float | None = None   # estimate refined mid-solve
        self._calib_thread = None             # one-shot calibration
        self._phase_t0: float | None = None   # current phase started
        self._phase_totals: dict = {}         # phase -> seconds
        self._phase_runs: dict = {}           # phase -> times entered
        self._tick = QTimer(self)
        self._tick.setInterval(500)
        self._tick.timeout.connect(self._refresh_progress)
        # cap model persists across sessions; read before _build so the
        # Model-menu check reflects it
        self.cap_model = str(self._settings().value("cap_model", "matrix"))
        self.mos_model = str(self._settings().value("mos_model",
                                                    "separate"))
        self._build()
        self._build_backend_menu()
        self._restore_settings()

    # --------------------------------------------------------------- layout
    def _build(self):
        # Open/Export live in the File menu only: the toolbar buttons
        # duplicated them and cost row width the in/out combos need.
        self.in_combo = QComboBox()
        self.out_combo = QComboBox()
        self.solve_btn = QPushButton("Solve")
        self.solve_btn.setEnabled(False)
        self.solve_btn.clicked.connect(self.solve)
        # ONE Solve action, THREE result forms. Solve/Simplify/Reduce as
        # three adjacent buttons stated no contract -- a student could not
        # tell they are three output forms of the same analysis. The form
        # selector says what you get; the error-budget spins appear only
        # when the chosen form uses them.
        self.form_combo = QComboBox()
        for label, tip in (
            ("Exact", "H(s) with every kept symbol — no approximation"),
            ("Simplified · full order", "SHORTENS THE FORMULA, keeps the "
             "shape: every pole and zero stays, but terms inside the "
             "coefficients are pruned while the response stays within the "
             "tolerance. The working model — safe to manipulate, valid at "
             "every frequency."),
            ("Simplified · lowest order", "THE FEWEST POLES your band and "
             "tolerance allow: reactances are removed and added back by "
             "matching pursuit until the smallest certified set remains — "
             "e.g. a 17-pole cascode becomes A0/(1+s/p1). The whiteboard "
             "model, certified ONLY over the band you select below."),
        ):
            self.form_combo.addItem(label)
            self.form_combo.setItemData(self.form_combo.count() - 1, tip,
                                        Qt.ToolTipRole)
        self.form_combo.setToolTip("What Solve returns")
        self.form_combo.currentTextChanged.connect(self._on_form_changed)
        self.mag_spin = self._spin(1.0, 0.0, 20.0, 0.1, " dB")
        self.phase_spin = self._spin(5.0, 0.0, 90.0, 0.5, " °")
        # the tolerance tube tracks the spins live
        self.mag_spin.valueChanged.connect(
            lambda _v: self._update_tol_bands())
        self.phase_spin.valueChanged.connect(
            lambda _v: self._update_tol_bands())

        # A QToolBar, not a raw QHBoxLayout: overflow chevron for free at
        # narrow widths, and the matplotlib navigation bar (itself a
        # QToolBar) shares the row instead of costing a second one.
        self.in_combo.setMinimumWidth(130)
        self.out_combo.setMinimumWidth(130)
        # the Nets tree mirrors the in/out choice with ▶/◀ arrows
        self.in_combo.currentTextChanged.connect(
            lambda _t: (self._refresh_net_decor(),
                        self._refresh_solve_hint(),
                        self._rerank_on_io()))
        self.out_combo.currentTextChanged.connect(
            lambda _t: (self._refresh_net_decor(),
                        self._refresh_solve_hint(),
                        self._rerank_on_io()))
        tb = QToolBar("main")
        tb.setObjectName("main_toolbar")
        tb.setMovable(False)
        # the mode combo is the STATE object -- every internal path and test
        # drives it -- but it is no longer shown: the Tool dropdown is its
        # face (the bench list it replaces cost a whole side column).
        self.mode_combo = QComboBox()
        self.mode_combo.addItems([name for name, _, _ in _TOOLS])
        self.mode_combo.currentTextChanged.connect(self._on_mode_changed)
        tb.addWidget(QLabel(" tool: "))
        self.tool_combo = QComboBox()
        self.tool_combo.setObjectName("tool_combo")
        for name, tip, _ in _TOOLS:
            self.tool_combo.addItem(name)
            self.tool_combo.setItemData(self.tool_combo.count() - 1, tip,
                                        Qt.ToolTipRole)
        self.tool_combo.currentTextChanged.connect(self._on_tool_selected)
        tb.addWidget(self.tool_combo)
        # the operating-point selector: a maestro corner/parametric run
        # opens as ONE session over many OPs (same topology); this picks
        # the active one, and every analysis re-derives on it
        self.op_lbl = QLabel(" OP: ")
        self.op_combo = QComboBox()
        self.op_combo.setMinimumWidth(110)
        self.op_combo.setToolTip(
            "The active operating point of a corner / parametric run "
            "(#maeCorner labels). Matches, reductions and ticks are "
            "structural and survive the switch; results re-derive.")
        self.op_combo.currentIndexChanged.connect(self._on_op_point)
        self._op_lbl_act = tb.addWidget(self.op_lbl)
        self._op_act = tb.addWidget(self.op_combo)
        self._op_lbl_act.setVisible(False)
        self._op_act.setVisible(False)
        self._in_lbl_act = tb.addWidget(QLabel(" in: "))
        self._in_act = tb.addWidget(self.in_combo)
        self._out_lbl_act = tb.addWidget(QLabel(" out: "))
        self._out_act = tb.addWidget(self.out_combo)
        self.probe_lbl = QLabel(" probe: ")
        self.probe_combo = QComboBox()
        self.probe_combo.setMinimumWidth(110)
        self._probe_lbl_act = tb.addWidget(self.probe_lbl)
        self._probe_act = tb.addWidget(self.probe_combo)
        self.probe2_combo = QComboBox()
        self.probe2_combo.setMinimumWidth(110)
        self._probe2_act = tb.addWidget(self.probe2_combo)
        self._probe_lbl_act.setVisible(False)
        self._probe_act.setVisible(False)
        self._probe2_act.setVisible(False)
        # the Impedance tool's TARGET: no dropdown -- a label saying
        # what Solve will compute (seen by I0 / from node smpl to gnd
        # / between nodes reg and fdbk), set from the trees or a wire
        # click; the ranking follows the choice immediately, the
        # impedance itself waits for Solve (a port dropdown here twice
        # showed a ranking that was not the port's)
        self.imp_target_lbl = QLabel("")
        self._imp_tpre_act = tb.addWidget(QLabel(" target: "))
        self._imp_target_act = tb.addWidget(self.imp_target_lbl)
        self._imp_tpre_act.setVisible(False)
        self._imp_target_act.setVisible(False)
        tb.addSeparator()
        self._solve_act = tb.addWidget(self.solve_btn)
        self._form_act = tb.addWidget(self.form_combo)
        self._budget_lbl = QLabel(" error budget: ")
        self._budget_lbl_act = tb.addWidget(self._budget_lbl)
        self._mag_act = tb.addWidget(self.mag_spin)
        self._phase_act = tb.addWidget(self.phase_spin)
        # the LOWEST-ORDER form's contract: a TOLERANCE STRATEGY, each
        # gating the pursuit on the quantity its designer actually reads
        # (field lesson: a band-wide criterion turned a generous cursor
        # drag into a useless 26-element model). Every strategy is
        # capped at a readable order and diagnoses instead of dumping.
        self.strategy_combo = QComboBox()
        for label, tip in (
            ("Gain & phase", "Spec-sheet contract: |H| within the dB "
             "budget AND phase within the degree budget at every band "
             "point. Trusts the cursor literally — a band deep into the "
             "rolloff demands fidelity there too."),
            ("Stability (margins)", "The reduced model must reproduce "
             "the full model's PHASE MARGIN (within the ° budget), GAIN "
             "MARGIN (within the dB budget) and unity-crossing "
             "frequency. The band's only job is to contain the "
             "crossover; everything else is forgiven."),
            ("Rejection (dB)", "For CMRR/PSRR studies: the dB curve "
             "tracks the full model within the budget at every band "
             "point, phase unconstrained — dB error IS relative error "
             "of the small quantity."),
        ):
            self.strategy_combo.addItem(label)
            self.strategy_combo.setItemData(
                self.strategy_combo.count() - 1, tip, Qt.ToolTipRole)
        self.strategy_combo.setToolTip("How the reduction tolerance is "
                                       "judged")
        self.strategy_combo.currentTextChanged.connect(
            lambda _t: self._on_strategy_changed())
        self._strategy_act = tb.addWidget(self.strategy_combo)
        # strat_* names, deliberately: the Compensate page also builds a
        # PM spin, and a bare self.pm_spin was BOUND TWICE — the page,
        # built later, silently won, so the Stability strategy ran with
        # the compensation TARGET (60°, clamped 30–85) instead of the
        # 5° tolerance and the visible toolbar spin changed nothing
        self.strat_pm_spin = self._spin(5.0, 0.5, 45.0, 1.0, " ° PM")
        self.strat_pm_spin.setToolTip("Phase margin reproduced within this")
        self.strat_gm_spin = self._spin(2.0, 0.5, 20.0, 0.5, " dB GM")
        self.strat_gm_spin.setToolTip("Gain margin reproduced within this")
        self.strat_rej_spin = self._spin(3.0, 0.1, 40.0, 0.5, " dB track")
        self.strat_rej_spin.setToolTip("|H| tracked within this many dB")
        self._pm_act = tb.addWidget(self.strat_pm_spin)
        self._gm_act = tb.addWidget(self.strat_gm_spin)
        self._rej_act = tb.addWidget(self.strat_rej_spin)
        for s in (self.strat_pm_spin, self.strat_gm_spin,
                  self.strat_rej_spin):
            s.valueChanged.connect(lambda _v: self._on_strategy_changed())
        # budget spins only exist for the budgeted forms
        self._budget_lbl_act.setVisible(False)
        self._mag_act.setVisible(False)
        self._phase_act.setVisible(False)
        for a in (self._strategy_act, self._pm_act, self._gm_act,
                  self._rej_act):
            a.setVisible(False)
        self.toolbar = tb

        # vertical room is the scarce one on a full-screen window:
        # the left side is TWO full-height columns side by side -- the
        # circuit (trees, the operating-point card) and, beside it, the
        # symbols (keep table over History, both about solves) -- not
        # three groups stacked
        self._devices_box = self._devices_group()
        self._keep_box = self._keepset_group()
        self._hist_box = self._history_group()
        keep_col = RestoringSplitter(Qt.Vertical)
        keep_col.setObjectName("keep_col")
        keep_col._default_sizes = [520, 140]
        keep_col.addWidget(self._keep_box)
        keep_col.addWidget(self._hist_box)
        keep_col.setSizes([520, 140])
        self._keep_col = keep_col
        # a QGroupBox's minimumSizeHint (title + indicator) is ~214 px
        # whatever its content needs; an explicit minimum lets the
        # column be dragged down to a symbol name's width
        for b in (self._keep_box, self._hist_box, self._devices_box):
            b.setMinimumWidth(120)
        left = RestoringSplitter(Qt.Horizontal)
        left.setObjectName("left_cols")
        left._default_sizes = [300, 330]
        left.addWidget(self._devices_box)
        left.addWidget(keep_col)
        # the Reduce bench's candidate tables take the symbols column's
        # slot while that bench is active (tall lists, idle column)
        self.reduce_col = self._reduce_candidates_col()
        self.reduce_col.setMinimumWidth(120)
        self.reduce_col.setVisible(False)
        left.addWidget(self.reduce_col)
        # the Compensate bench's controls take the same slot: the
        # suggester never reads the keep set, so the symbols column
        # would only invite ticking into a pointless hybrid Solve
        self.comp_col = self._comp_controls_col()
        self.comp_col.setMinimumWidth(120)
        self.comp_col.setVisible(False)
        left.addWidget(self.comp_col)
        left.setSizes([300, 330])
        self.left_split = left

        self.canvas = FigureCanvasQTAgg(Figure(figsize=(5.2, 4.0)))
        # typeset HTML summary (falls back to a QTextEdit without the
        # WebEngine addon); same setPlainText/toPlainText API either way
        self.summary = summaryweb.make()
        # Expression surface: the KaTeX web view when QtWebEngine is present --
        # crisp vector math, hover = OP value, click = symbolClicked. Falls
        # back to the matplotlib mathtext canvas on a PySide6 install
        # without the WebEngine addon.
        self.exprweb = None
        if exprweb.WEBENGINE:
            try:
                self.exprweb = exprweb.ExprWebView()
                self.exprweb.bridge.symbolClicked.connect(self._select_keep_symbol)
            except Exception:
                self.exprweb = None            # broken GL/WebEngine -> fall back
        if self.exprweb is not None:
            self.expr_canvas = None
            expr_surface = self.exprweb
        else:
            # The factored H(s) runs to several lines. Shrinking them to fit a
            # short panel renders ~6pt mush, so let the panel scroll and size the
            # canvas to its content instead (see _render_expr).
            self.expr_canvas = FigureCanvasQTAgg(Figure(figsize=(5.2, 1.6)))
            self.expr_scroll = QScrollArea()
            self.expr_scroll.setWidgetResizable(True)
            self.expr_scroll.setWidget(self.expr_canvas)
            # A matplotlib canvas eats wheel events (it turns them into
            # scroll_event for plot callbacks), so they never reach the
            # QScrollArea and the tab won't scroll. Intercept the wheel and
            # drive the scrollbar ourselves.
            self.expr_canvas.installEventFilter(self)
            expr_surface = self.expr_scroll
        # names default to the readable leaf form (g_{m,MN1}); tick to expand the
        # full hierarchy (g_{m,I0.MN1}) when a leaf name is ambiguous
        self.fullnames_chk = QCheckBox("Full names")
        self.fullnames_chk.setToolTip(
            "Show the full instance hierarchy (I0.MN1) instead of the leaf (MN1)")
        self.fullnames_chk.toggled.connect(lambda _=False: self._render_expr())
        # a reduction's lumped equivalents (G_eq,net8) are definitions;
        # unticked they are listed under the expressions, ticked they
        # are substituted by their member sums throughout A0, N, D,
        # p1, z1 -- the formula in the original device symbols
        self.expand_eq_chk = QCheckBox("Expand equivalents")
        self.expand_eq_chk.setToolTip(
            "Replace every lumped equivalent (G_eq, C_eq from an AC-ground "
            "reduction) by the sum of the elements it stands for, in every "
            "expression shown. Unticked: the equivalents are defined below "
            "the expressions.")
        self.expand_eq_chk.setEnabled(False)
        self.expand_eq_chk.toggled.connect(lambda _=False: self._render_expr())
        _hdr = QHBoxLayout()
        _hdr.setContentsMargins(0, 0, 0, 0)
        _hdr.addWidget(self.fullnames_chk)
        _hdr.addWidget(self.expand_eq_chk)
        _hdr.addStretch(1)
        expr_tab = QWidget()
        self._expr_tab = expr_tab
        _ev = QVBoxLayout(expr_tab)
        _ev.setContentsMargins(0, 0, 0, 0)
        _ev.setSpacing(2)
        _ev.addLayout(_hdr)
        _ev.addWidget(expr_surface)
        # Two overlapping Bode curves cannot show a 0.5 dB residual. The Error
        # tab is where "how good is this model" actually becomes readable.
        self.err_canvas = FigureCanvasQTAgg(Figure(figsize=(5.2, 3.0)))
        # a fresh Figure's face is WHITE until the first solve draws
        # into it -- on dark the empty Error tab flashed a white page
        theme.style_figure(self.err_canvas.figure)
        # the Error tab is often first SHOWN long after the solve, and
        # its pane resizes independently: re-align on its own draws too
        self.err_canvas.mpl_connect("draw_event",
                                    lambda _e: self._align_error_axes())
        tabs = QTabWidget()
        # a live operations log: what ran, in which phase, how long it
        # took and how that compared with the estimate. The status bar
        # shows only the current line, so the moment a run ends its
        # history is gone -- and that history is exactly what a bug
        # report needs. Append-only, copyable, capped.
        self.logview = QTextEdit()
        self.logview.setReadOnly(True)
        self.logview.setLineWrapMode(QTextEdit.NoWrap)
        self.logview.setStyleSheet(
            "font-family: 'Cascadia Mono', Consolas, monospace;"
            "font-size: 9pt;")
        self._log_t0 = time.monotonic()
        tabs.addTab(self.summary, "Summary")
        tabs.addTab(expr_tab, "Expression")
        tabs.addTab(self.err_canvas, "Error")
        self._whatif_tab = self._whatif_page()
        tabs.addTab(self._whatif_tab, "What-if")
        tabs.addTab(self._comp_page(), "Suggestions")
        tabs.addTab(self._margins_page(), "Margins")
        tabs.addTab(self._sens_page(), "Sensitivities")
        tabs.addTab(self._gft_page(), "GFT")
        tabs.addTab(self.logview, "Log")
        self.tabs = tabs

        # the certification band, chosen where it is judged: a two-cursor
        # slider above the Bode, its span mirrored on the plot. Budgeted
        # results are certified FOR THIS BAND -- making it draggable and
        # visible is what teaches that contract.
        self.band_slider = RangeSlider(1.0, 1e10)
        self.band_slider.setValues(1e3, 1e9)
        self.band_slider.valuesChanged.connect(self._on_band_changed)
        self._band_spans: list = []
        self._tol_bands: list = []
        # the cursors print their own frequencies, so the row is just the
        # slider; its margins are re-synced to the Bode's axis extent on
        # every draw (_sync_band_row) so groove and frequency axis align
        band_row = QWidget()
        bl = QVBoxLayout(band_row)
        bl.setContentsMargins(6, 2, 6, 0)
        bl.setSpacing(0)
        # the slider gets the full row: its groove must align with the
        # Bode's frequency axis (_sync_band_row), so nothing shares its
        # line -- the certificate hint sits UNDER it
        bl.addWidget(self.band_slider)
        # the live order certificate: what the band DEMANDS at the set
        # tolerance, before any solve (Loewner rank, ~50 ms, debounced)
        self.cert_lbl = QLabel("")
        self.cert_lbl.setStyleSheet(f"color: {theme.MUTED}; font-size: 8pt;")
        self.cert_lbl.setAlignment(Qt.AlignHCenter)
        self.cert_lbl.setVisible(False)
        bl.addWidget(self.cert_lbl)
        self._cert_timer = QTimer(self)
        self._cert_timer.setSingleShot(True)
        self._cert_timer.setInterval(250)
        self._cert_timer.timeout.connect(self._refresh_certificate)
        # the ranking's scores and peaks are measured over the chosen
        # band, so a band drag re-ranks the keep table (debounced
        # longer than the certificate: a rank is a numeric sweep) --
        # a list computed for another band is the stale-list trap
        self._rank_band_timer = QTimer(self)
        self._rank_band_timer.setSingleShot(True)
        self._rank_band_timer.setInterval(700)
        self._rank_band_timer.timeout.connect(self._rerank_on_band)
        # the engine child's stderr goes to a file (host.py dup2);
        # this tail brings its warnings into the Log where they belong
        self._engine_log_timer = QTimer(self)
        self._engine_log_timer.setInterval(2000)
        self._engine_log_timer.timeout.connect(self._tail_engine_log)
        band_row.setToolTip(
            "The frequency band a budgeted result is certified for: the "
            "error budget is enforced HERE, and a reduced-order model is "
            "meaningless outside it. Drag the cursors; the span is "
            "mirrored on the plot.")
        self.band_row = band_row
        self.band_row.setVisible(False)
        plot_col = QWidget()
        pv = QVBoxLayout(plot_col)
        pv.setContentsMargins(0, 0, 0, 0)
        pv.setSpacing(0)
        pv.addWidget(band_row)
        pv.addWidget(self.canvas, 1)
        # plot interaction lives beside the plots, not in the window
        # toolbar: a vertical strip keeps the full width for the Bode
        self.nav = NavigationToolbar2QT(self.canvas, self,
                                        coordinates=False)
        self.nav.setOrientation(Qt.Vertical)
        # hidden on request: the buttons duplicated what the mouse and
        # the report/export verbs already do, and the strip stole plot
        # width. The toolbar object stays alive -- matplotlib's canvas
        # shortcuts and any programmatic pan/zoom still route through it
        self.nav.setVisible(False)
        plot_area = QWidget()
        pa = QHBoxLayout(plot_area)
        pa.setContentsMargins(0, 0, 0, 0)
        pa.setSpacing(0)
        pa.addWidget(plot_col, 1)
        pa.addWidget(self.nav, 0, Qt.AlignTop)
        self.canvas.mpl_connect("draw_event",
                                lambda _ev: self._sync_band_row())
        # margins are FRACTIONS frozen at render-time width; re-fit
        # them whenever the canvas actually changes size (first light
        # renders into a pane that has not been laid out yet)
        self.canvas.mpl_connect(
            "resize_event", lambda _ev: view.relayout(self.canvas.figure))
        self.err_canvas.mpl_connect(
            "resize_event",
            lambda _ev: view.relayout(self.err_canvas.figure))

        # the right column: TWO pages up top (the analysis plots and
        # the schematic, each needing the full width to be read) over
        # ONE shared strip of result tabs -- the Log, Suggestions or
        # Expression concern the run, not the page you happen to view
        self._schematic_tab = self._schematic_page()
        self.main_tabs = QTabWidget()
        # a plain tab widget, like the result strip below: document
        # mode draws tabs with no pane frame -- on dark they floated
        # detached over the bench (and its tab-bar base line is pure
        # white, a glare there). Dark flattens the pane bevel to a
        # 1px border (theme.py, by name): Fusion's inner highlight
        # read as a raised button face.
        self.main_tabs.setObjectName("main_tabs")
        self.main_tabs.addTab(plot_area, "Analysis")
        self.main_tabs.addTab(self._schematic_tab, "Schematic")
        self.main_tabs.currentChanged.connect(self._on_main_tab)

        right = RestoringSplitter(Qt.Vertical)
        right.setObjectName("right_split")
        right._default_sizes = [500, 240]
        right.addWidget(self.main_tabs)
        right.addWidget(tabs)
        right.setSizes([500, 240])
        self.right_split = right
        # the result strip reads bottom-up: its tab bar sits at the
        # BOTTOM edge, and each pane's corner carries a maximize
        # toggle -- press to take the full height, press again (or
        # double-click the splitter bar) to come back
        tabs.setTabPosition(QTabWidget.South)
        self._right_max: str | None = None

        def _mk_max(which, host):
            from PySide6.QtWidgets import QToolButton
            b = QToolButton()
            b.setAutoRaise(True)
            b.setIcon(theme.icon("expand"))
            b.setToolTip(
                "Full height for this pane; press again to restore\n"
                "(double-clicking the splitter bar also restores)")
            b.clicked.connect(lambda: self._toggle_right_max(which))
            host.setCornerWidget(b, Qt.TopRightCorner)
            return b

        self._max_top_btn = _mk_max("top", self.main_tabs)
        self._max_bot_btn = _mk_max("bottom", tabs)

        split = RestoringSplitter(Qt.Horizontal)
        split.setObjectName("h_split")
        split._default_sizes = [640, 900]
        split.addWidget(left)
        split.addWidget(right)
        # panes may collapse to nothing (a wide plot is worth it), so
        # the handle must stay findable when one has: wider. The LOOK
        # comes from the app QSS (theme._qss: chrome background,
        # three-dot grip, hover lift) -- a widget-level stylesheet
        # here overrode it with a hardcoded light bar that glared on
        # the dark theme (field report)
        for sp in (split, left, keep_col):
            sp.setHandleWidth(7)
        # setSizes before show() is clobbered by the first layout pass (the
        # tables' size hints won, squeezing the plots to a sliver): stretch
        # factors carry the intent through layout, and showEvent re-applies
        # the sizes once the window is real.
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        self.h_split = split

        self._set_tab_visibility(self.mode_combo.currentText())
        self.addToolBar(tb)

        # Persistent, color-coded message strip: advisories and failures
        # live here instead of interrupting with modal dialogs.
        self.msg_strip = QLabel()
        self.msg_strip.setWordWrap(True)
        self.msg_strip.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.msg_strip.hide()

        # the workflow breadcrumb (gui-ux-plan.md U-E): the intended path
        # Open -> Match -> Choose symbols -> Solve made visible, first
        # incomplete step highlighted, every step a click target
        self.crumb = QLabel()
        self.crumb.setTextFormat(Qt.RichText)
        self.crumb.setTextInteractionFlags(Qt.TextBrowserInteraction)
        self.crumb.linkActivated.connect(self._on_crumb_clicked)
        self._update_crumb()

        outer = QVBoxLayout()
        outer.setContentsMargins(0, 0, 0, 0)
        # the Virtuoso red rule, as a REAL widget under the toolbar area:
        # a QSS border-bottom on the toolbar is painted under its own
        # buttons (Qt lays toolbar children over stylesheet borders,
        # padding or not), so the rule lives where nothing can overlap it
        from PySide6.QtWidgets import QFrame
        rule = QFrame()
        rule.setFixedHeight(2)
        rule.setStyleSheet(f"background: {theme.ACCENT};")
        outer.addWidget(rule)
        inner = QVBoxLayout()                 # the rule spans full width;
        inner.setContentsMargins(9, 4, 9, 9)  # content keeps its margins
        inner.addWidget(self.crumb)
        inner.addWidget(self.msg_strip)
        inner.addWidget(split, 1)
        outer.addLayout(inner, 1)
        central = QWidget()
        central.setLayout(outer)
        self.setCentralWidget(central)
        self._build_menus()

        # Solve progress lives in the status bar: a hybrid solve's cost IS the
        # grid, whose size is known up front, so this is real progress.
        self.progress = QProgressBar()
        self.progress.setMaximumWidth(220)
        self.progress.setMaximumHeight(14)
        self.progress.setTextVisible(True)
        self._tick.stop()
        self._t0 = None
        self.progress.hide()
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setMaximumHeight(18)
        self.cancel_btn.hide()
        self.cancel_btn.clicked.connect(self._cancel_solve)
        self.statusBar().addPermanentWidget(self.progress)
        self.statusBar().addPermanentWidget(self.cancel_btn)

        self.resize(1120, 780)
        self.statusBar().showMessage("Open a CIN + psf results directory to begin.")

    def showEvent(self, event):
        super().showEvent(event)
        if not self._splitters_restored:
            self._splitters_restored = True
            s = self._settings()
            restored = False
            # new keys for the two-column left side: a state saved for
            # the old vertical stack must not be restored into it
            for name, sp in (("h_split2", self.h_split),
                             ("left_cols", self.left_split),
                             ("keep_col", self._keep_col),
                             ("right_split", self.right_split)):
                state = s.value("splitters/" + name)
                if state is not None:
                    sp.restoreState(state)
                    restored = True
            if not restored:
                self._reset_layout()

    def _reset_layout(self):
        """The default proportions: two left columns, the plots wide."""
        w = max(self.width(), 1200)
        self.h_split.setSizes([640, w - 660])
        self.left_split.setSizes([300, 330])
        self._keep_col.setSizes([520, 140])
        self.right_split.setSizes([500, 240])
        self._right_max = None
        self._sync_max_icons()

    def _sync_max_icons(self):
        # CircuitInsight's own themed glyphs: the Fusion titlebar
        # icons drew in a dark shade (invisible on dark) and meant
        # WINDOW maximize, not pane expand
        for which, btn in (("top", self._max_top_btn),
                           ("bottom", self._max_bot_btn)):
            active = self._right_max == which
            btn.setIcon(theme.icon("restore" if active else "expand"))

    def _toggle_right_max(self, which: str):
        """Maximize the plots/schematic pane or the result strip: the
        other pane collapses; a second press (or a double-click on
        the splitter bar) restores the default proportions."""
        sp = self.right_split
        sizes = sp.sizes()
        total = sum(sizes) or 740
        # the truth is the SIZES, not a flag: a drag or a double-click
        # on the bar restores behind the button's back
        maxed = (sizes[1] == 0 if which == "top" else sizes[0] == 0) \
            if len(sizes) >= 2 else False
        if maxed:
            d = sp._default_sizes
            share = [x / sum(d) for x in d]
            sp.setSizes([max(1, int(total * s)) for s in share])
            self._right_max = None
        else:
            sp.setSizes([total, 0] if which == "top" else [0, total])
            self._right_max = which
        self._sync_max_icons()

    def _toggle_left_panes(self):
        """The keyboard way home: any pane dragged to nothing, in any
        of the splitters (the keep column inside the left side too),
        comes back at its default share; when nothing is collapsed the
        left side collapses for a wide plot."""
        defaults = ((self.h_split, [640, max(self.width(), 1200) - 660]),
                    (self.left_split, [300, 330]),
                    (self._keep_col, [520, 140]),
                    (self.right_split, [500, 240]))
        restored = False
        for sp, default in defaults:
            sizes = sp.sizes()
            vis = [i for i in range(sp.count())
                   if not sp.widget(i).isHidden()]
            if vis and any(sizes[i] == 0 for i in vis):
                total = sum(sizes[i] for i in vis) or sum(default)
                share = [d / sum(default) for d in default]
                new = list(sizes)
                for i, s in zip(vis, share):
                    new[i] = max(1, int(total * s))
                sp.setSizes(new)
                restored = True
        if restored:
            return
        sizes = self.h_split.sizes()
        self._left_saved = sizes[0]
        self.h_split.setSizes([0, sum(sizes)])

    def eventFilter(self, obj, event):
        """Forward wheel scrolls on the (event-swallowing) expression canvas to
        its scroll area, so the tab scrolls like any other.

        getattr, not self.expr_canvas: Qt can deliver a last event to a still-
        registered filter during teardown, after the attribute is gone.
        """
        if (event.type() == QEvent.Wheel
                and obj is getattr(self, "expr_canvas", None)):
            sb = self.expr_scroll.verticalScrollBar()
            sb.setValue(sb.value() - event.angleDelta().y())
            return True                          # consumed
        return super().eventFilter(obj, event)

    def _build_menus(self):
        """A Virtuoso-style menu bar. The actions already exist as buttons; a
        menu bar is what makes the window read as part of the toolchain rather
        than a panel of loose controls."""
        mb = self.menuBar()

        m_file = mb.addMenu("&File")
        m_file.setTearOffEnabled(True)          # Virtuoso menus tear off
        a_open = m_file.addAction("&Open CIN + psf…")
        a_open.setShortcut(QKeySequence.Open)                 # Ctrl+O
        a_open.triggered.connect(self.open_dialog)
        self.m_recent = m_file.addMenu("Open &Recent")
        self.a_export = m_file.addAction("&Export report…")
        self.a_export.setShortcut("Ctrl+E")
        self.a_export.triggered.connect(self.export)
        self.a_export.setEnabled(False)
        self.a_copy_tex = m_file.addAction("&Copy H(s) as LaTeX")
        self.a_copy_tex.setShortcut("Ctrl+L")
        self.a_copy_tex.triggered.connect(self.copy_latex)
        self.a_copy_tex.setEnabled(False)
        m_file.addSeparator()
        self.a_add_report = m_file.addAction("&Add view to report")
        self.a_add_report.setShortcut("Ctrl+D")
        self.a_add_report.setToolTip(
            "Append the current plot + summary to the session report "
            "(lab-notebook style)")
        self.a_add_report.triggered.connect(self.add_to_report)
        self.a_add_report.setEnabled(False)
        self.a_export_session = m_file.addAction(
            "Export &session report…")
        self.a_export_session.triggered.connect(self.export_session_report)
        self.a_export_session.setEnabled(False)
        self.a_export_csv = m_file.addAction(
            "Export &traces (CSV)…")
        self.a_export_csv.triggered.connect(self.export_csv)
        self.a_export_csv.setEnabled(False)
        m_file.addSeparator()
        # session states: the user's accumulated work (ticks, matches,
        # band, strategy) as a recoverable object, with the computed
        # solution riding along fingerprint-gated. Autosaved as
        # <cin>.last.cistate after every shown result and on close.
        self.a_save_state = m_file.addAction("Save s&tate as…")
        self.a_save_state.triggered.connect(self.save_state_dialog)
        self.a_save_state.setEnabled(False)
        self.a_load_state = m_file.addAction("&Load state…")
        self.a_load_state.triggered.connect(self.load_state_dialog)
        self.a_load_state.setEnabled(False)
        self.a_restore_last = m_file.addAction("&Restore last state")
        self.a_restore_last.triggered.connect(
            lambda: self._load_state_file(None))
        self.a_restore_last.setEnabled(False)
        m_file.addSeparator()
        m_file.addAction("&Quit").triggered.connect(self.close)

        m_an = mb.addMenu("&Analysis")
        m_an.setTearOffEnabled(True)
        self.a_solve = m_an.addAction("&Solve")
        self.a_solve.setShortcut("Ctrl+Return")
        self.a_solve.triggered.connect(self.solve)
        self.a_solve.setEnabled(False)
        self.a_simplify = m_an.addAction("Sim&plify — full order")
        self.a_simplify.setShortcut("Ctrl+Shift+Return")
        self.a_simplify.triggered.connect(self.simplify)
        self.a_simplify.setEnabled(False)
        self.a_reduce = m_an.addAction("Simplify — lowest o&rder")
        self.a_reduce.setShortcut("Ctrl+R")
        self.a_reduce.triggered.connect(self.reduce)
        self.a_reduce.setEnabled(False)
        m_an.addSeparator()
        self.a_rank = m_an.addAction("&Rank symbols")
        self.a_rank.setShortcut("F5")
        self.a_rank.triggered.connect(self._rank_async)
        self.a_suggest = m_an.addAction("Suggest keep-set ≤ &budget…")
        self.a_suggest.triggered.connect(self._suggest_keep_dialog)
        self.a_split = m_an.addAction("Split ad&vice")
        self.a_split.setToolTip("Rank the tearing cuts for this solve and, "
                                "when none pays, name the AC ground that "
                                "would fix it")
        self.a_split.triggered.connect(self._split_advice)
        # the solver backend: set once, rarely -- a radio submenu driving
        # the (hidden) combo that holds the state
        self._backend_menu = m_an.addMenu("Solver &backend")
        self._backend_group = QActionGroup(self)
        self._backend_group.setExclusive(True)
        self._backend_actions = []
        m_an.addSeparator()
        self.a_attr = m_an.addAction("&Attribute poles")
        self.a_attr.setToolTip(
            "Which element establishes which pole — computed from exact "
            "sensitivities, then VERIFIED by nudging the owner and "
            "re-rooting. Lands in the Summary.")
        self.a_attr.triggered.connect(self.attribute_poles)
        self.a_explain = m_an.addAction("&Explain the numbers")
        self.a_explain.setToolTip(
            "Which collapsed parameters carry each numeral of H(s): the "
            "unkept symbols are substituted before the solve, so the "
            "numbers in a simplified expression are sums of products "
            "whose names are gone — this ranks them back, per "
            "coefficient. Lands in the Summary.")
        self.a_explain.triggered.connect(self.explain_numbers)
        self.a_explain_deep = m_an.addAction("Explain per &numeral (deep)")
        self.a_explain_deep.setToolTip(
            "Resolve EACH collapsed numeral of the current symbolic "
            "expression individually — a derivative sweep over the hybrid "
            "grid separates the kept-monomials the operating-point sweep "
            "cannot. Costs a few multiples of the solve; needs a keep set.")
        self.a_explain_deep.triggered.connect(self.explain_per_numeral)
        m_an.addSeparator()
        self.a_declare_rr = m_an.addAction("Declare AC as return ra&tio…")
        self.a_declare_rr.setToolTip(
            "State that this run's AC sweep IS the loop gain: pick the "
            "net whose v(net)/v(input) is the return ratio. Opens the "
            "loop benches when the run carries no stb results — the "
            "declared trace becomes their ground-truth overlay.")
        self.a_declare_rr.triggered.connect(self.declare_return_ratio)

        m_dev = mb.addMenu("&Devices")
        m_dev.setTearOffEnabled(True)
        m_dev.addAction("Suggest &matched pairs").triggered.connect(
            self.suggest_matches)
        m_dev.addAction("Match &selection").triggered.connect(self.match_selected)
        m_dev.addAction("&Unmatch selection").triggered.connect(
            self.unmatch_selected)
        m_dev.addSeparator()
        m_dev.addAction("&Clear matches").triggered.connect(self.clear_matches)
        m_dev.addSeparator()
        # how a match group's shared symbols get their numbers: a
        # modelling choice set once, like the cap and MOS models below
        m_val = m_dev.addMenu("Match &values")
        self._policy_group = QActionGroup(self)
        self._policy_group.setExclusive(True)
        self._policy_actions = {}
        for label, tip in (
            ("representative", "one member's values stand for the group "
             "(default: whichever the netlist reaches first; right-click "
             "a 🔗 member to choose)"),
            ("average", "the arithmetic mean of the members' values"),
            ("weighted", "the mean weighted by each member's band "
             "sensitivity — the member that shapes the response most "
             "pulls hardest (measured best on both benches)"),
        ):
            a = m_val.addAction(label)
            a.setCheckable(True)
            a.setStatusTip(tip)
            a.setToolTip(tip)
            self._policy_group.addAction(a)
            self._policy_actions[label] = a
            a.triggered.connect(
                lambda _c=False, label=label: self._on_matchval_changed(label))
        self._policy_actions["weighted"].setChecked(True)
        m_dev.addSeparator()
        # cap model: charge-matrix (accurate, exact transcapacitances) vs the
        # five-cap lumped model -- the textbook contrast. Toggling re-opens
        # the run (the model is baked into the reconstruction) and re-solves.
        self.a_matrix_caps = m_dev.addAction("Charge-&matrix caps")
        self.a_matrix_caps.setCheckable(True)
        self.a_matrix_caps.setChecked(getattr(self, "cap_model", "matrix")
                                      == "matrix")
        self.a_matrix_caps.setToolTip(
            "Exact transcapacitance (charge) matrix vs the five-cap lumped "
            "model. On the non-reciprocal SKY130 gate-drain the lumped model "
            "drifts up to ~1 dB at 10 GHz (5T ~28 dB; follower Zout 0.70 dB "
            "vs matrix 0.004 dB near its peak). Re-opens the run.")
        self.a_matrix_caps.toggled.connect(self._on_cap_model_toggled)
        # exact per-device gmb bundle: where gate and bulk sit at the
        # same AC potential, one hat symbol carries gm+gmb; a
        # bulk-tied-to-source gmbs is dropped as inert. Re-opens the run.
        self.a_lump_gmb = m_dev.addAction("Lump ĝm = gm + gmb")
        self.a_lump_gmb.setCheckable(True)
        self.a_lump_gmb.setChecked(getattr(self, "mos_model", "separate")
                                   == "lumped-gmb")
        self.a_lump_gmb.setToolTip(
            "EXACT per-device bundle: devices whose gate and bulk sit at "
            "the same AC potential (same net, or both held by DC "
            "sources) get one ĝm = gm + gmb symbol; a "
            "bulk-tied-to-source gmbs is dropped as inert. Devices with "
            "a signal-driven gate keep their separate gmb. Re-opens "
            "the run.")
        self.a_lump_gmb.toggled.connect(self._on_mos_model_toggled)

        # Options: DESKTOP preferences -- theme and layout. The
        # modeling choices (cap model, gmb lump, match policy) stay in
        # Devices: they change the model, not the app.
        m_opt = mb.addMenu("&Options")
        m_opt.setTearOffEnabled(True)
        self.a_dark = m_opt.addAction("&Dark theme")
        self.a_dark.setCheckable(True)
        self.a_dark.setChecked(theme.current == "dark")
        self.a_dark.toggled.connect(
            lambda on: self.apply_theme("dark" if on else "light"))
        m_opt.addSeparator()
        # a once-per-machine setting, not a schematic verb: the header
        # button it replaces spent permanent toolbar room on it
        a_lib = m_opt.addAction("Schematic symbol &library…")
        a_lib.setToolTip("Choose the SVG symbol library directory (the "
                         "one with terminals.csv); remembered across "
                         "sessions")
        a_lib.triggered.connect(self._schematic_pick_library)
        a_flush = m_opt.addAction("Flush solve-time &calibration…")
        a_flush.setToolTip(
            "Forget everything the solve-time estimator has learned on "
            "this machine (the measured model and the per-path "
            "corrections) and re-measure from scratch. Use it after a "
            "hardware change, or when training data was polluted by a "
            "loaded server.")
        a_flush.triggered.connect(self._flush_calibration)
        m_opt.addSeparator()
        a_tog = m_opt.addAction("Restore collapsed panes / collapse &left")
        a_tog.setShortcut("Ctrl+Shift+L")
        a_tog.triggered.connect(self._toggle_left_panes)
        m_opt.addAction("Reset la&yout").triggered.connect(self._reset_layout)

        m_help = mb.addMenu("&Help")
        m_help.setTearOffEnabled(True)
        self.a_manual = m_help.addAction("&User guide")
        self.a_manual.setShortcut("F1")
        self.a_manual.triggered.connect(self.show_manual)
        m_help.addSeparator()
        m_help.addAction("&About CircuitInsight").triggered.connect(
            self._show_about)

    def _spin(self, val, lo, hi, step, suffix):
        s = QDoubleSpinBox()
        s.setRange(lo, hi)
        s.setSingleStep(step)
        s.setValue(val)
        s.setSuffix(suffix)
        return s

    def _devices_group(self):
        box = QGroupBox("Circuit — instances / nets")
        # the circuit as two trees: Instances (hierarchy + match sets as
        # link decorations + OP glance on hover) and Nets (connections,
        # with the earth mark on AC-grounded nets and in/out arrows)
        self.devices = devtree.InstanceTree()
        self.devices.itemSelectionChanged.connect(self._on_devices_selected)
        self.devices.deviceActivated.connect(self._show_device_op_for)
        self.devices.aliasRequested.connect(self._edit_device_alias)
        self.devices.unmatchRequested.connect(self._unmatch_group)
        self.devices.representativeRequested.connect(
            self._pick_representative)
        self.devices.matchRequested.connect(self.match_selected)

        self.nets_tree = devtree.NetTree()
        self.nets_tree.outputRequested.connect(self._set_output_net)
        self.nets_tree.acgroundRequested.connect(self._acg_from_net)
        self.nets_tree.impedanceRequested.connect(self._impedance_at_nets)
        self.nets_tree.itemSelectionChanged.connect(
            self._impedance_target_from_selection)
        self.devices.impedanceSeenRequested.connect(self._impedance_seen_by)
        self.probe_combo.currentTextChanged.connect(self._on_port_text)
        self.nets_tree.connectionActivated.connect(self._goto_instance)

        self.dev_tabs = QTabWidget()
        # the dark theme rules THIS pane's border explicitly (theme.py):
        # its page is a BASE-dark tree that swallows the derived outline
        self.dev_tabs.setObjectName("dev_tabs")
        # the tab pane frames its page already: a tree keeping its own
        # frame doubles the line on light and stacks two dark lines
        # into a mush at the bottom on dark
        self.devices.setFrameShape(QFrame.NoFrame)
        self.nets_tree.setFrameShape(QFrame.NoFrame)
        self.dev_tabs.addTab(self.devices, "Instances")
        self.dev_tabs.addTab(self.nets_tree, "Nets")

        # the match verbs live in the tree's context menu and in the
        # Devices menu (the values policy as a radio group there); a
        # button row under the trees was a third copy of the same verbs
        self.devices.suggestRequested.connect(self.suggest_matches)
        self.devices.unmatchSelectionRequested.connect(self.unmatch_selected)

        # the operating-point card: the selected device, read the way a
        # designer reads it, persistent -- follows the selection from
        # the tree, the schematic, an expression symbol, a keep row.
        # Trees and card share a SPLITTER, so the card's height is the
        # user's to drag (it was a fixed 240 px cap)
        from .opcard import OpCard
        self.op_card = OpCard()
        self.op_card.full_btn.clicked.connect(
            lambda: self._show_device_op_for(self.op_card.name)
            if self.op_card.name else None)
        dev_split = RestoringSplitter(Qt.Vertical)
        dev_split.setObjectName("dev_split")
        dev_split._default_sizes = [420, 200]
        dev_split.addWidget(self.dev_tabs)
        dev_split.addWidget(self.op_card)
        dev_split.setStretchFactor(0, 1)
        dev_split.setStretchFactor(1, 0)
        dev_split.setCollapsible(0, False)
        v = QVBoxLayout()
        v.addWidget(dev_split, 1)
        return self._collapsible(box, v)

    def _keepset_group(self):
        # NB: a QGroupBox's minimumSizeHint includes its TITLE width — a
        # sentence-long title here once forced the whole left pane to
        # 1226 px minimum and squeezed the plots to a sliver. Short title,
        # sentence in the tooltip.
        box = QGroupBox("Keep symbolic")
        # no group-wide tooltip: it covered every row with the same
        # sentence; each symbol carries its own (dcOp, score, peak)
        self.keep_tbl = QTableWidget(0, 5)
        self.keep_tbl.setHorizontalHeaderLabels(
            ["symbol", "dcOp", "score", "peaks", "LaTeX"])
        self.keep_tbl.horizontalHeader().setStretchLastSection(True)
        # three visible elements: the filter, the table with what the
        # next solve needs (tick, symbol, dcOp, score), the estimate.
        # peaks and LaTeX are a header right-click and a row right-click
        # away; the verbs (Rank, Suggest, Split, backend, grouping) live
        # in the Analysis menu and the context menus -- the trees above
        # get the room
        # a LIST with checks: only the symbol column shows, no headers;
        # dcOp, score and peak live in each row's tooltip, and the
        # columns come back from the header/row context menu for
        # anyone who wants them side by side
        for col in (1, 2, 3, 4):
            self.keep_tbl.setColumnHidden(col, True)
        self.keep_tbl.horizontalHeader().setVisible(False)
        self.keep_tbl.verticalHeader().setVisible(False)
        self.keep_tbl.setShowGrid(False)
        self.keep_tbl.horizontalHeader().setContextMenuPolicy(
            Qt.CustomContextMenu)
        self.keep_tbl.horizontalHeader().customContextMenuRequested.connect(
            self._keep_header_menu)
        self.keep_tbl.setContextMenuPolicy(Qt.CustomContextMenu)
        self.keep_tbl.customContextMenuRequested.connect(self._keep_menu)
        # only the LaTeX column edits; the AllEditTriggers + per-item
        # ItemIsEditable flag (set in _fill_keep_table) confines editing there
        self.keep_tbl.setEditTriggers(QAbstractItemView.AllEditTriggers)
        self.keep_tbl.itemChanged.connect(self._on_keep_changed)
        self.keep_tbl.itemSelectionChanged.connect(self._on_keep_row_selected)


        self.keep_filter = QLineEdit()
        self.keep_filter.setPlaceholderText(
            "filter symbols… (e.g. gm_ or MN1)")
        self.keep_filter.textChanged.connect(self._apply_keep_filter)
        self.group_chk = QCheckBox("group by device")
        self.group_chk.setToolTip(
            "Sort the ranking by owning device (alternating tint) instead "
            "of by score")
        self.group_chk.toggled.connect(self._on_group_toggled)

        rankb = QPushButton("Rank")
        rankb.clicked.connect(self._rank_async)
        self.budget_spin = self._spin(5.0, 0.1, 600.0, 1.0, " s")
        self.budget_spin.setToolTip(
            "SOLVE-TIME budget in seconds — how long a symbolic solve may "
            "take. Unrelated to the error budget in the toolbar, which is "
            "accuracy in dB/°.")
        suggestb = QPushButton("Suggest ≤ budget")
        suggestb.clicked.connect(self._suggest_keep)
        ctl = QHBoxLayout()
        ctl.addWidget(rankb)
        ctl.addWidget(self.group_chk)
        ctl.addWidget(QLabel("time budget:"))
        ctl.addWidget(self.budget_spin)
        ctl.addWidget(suggestb)

        self.estimate_lbl = QLabel("estimate: —")
        # a wrapped label's sizeHint is its full text width; that was
        # the keep column's minimum width (~370 px). Let it take the
        # column's width and wrap instead of dictating it.
        self.estimate_lbl.setWordWrap(True)
        self.estimate_lbl.setMinimumWidth(1)
        self.estimate_lbl.setSizePolicy(QSizePolicy.Ignored,
                                        QSizePolicy.Preferred)
        self.keep_filter.setMinimumWidth(1)
        # split advisory: whether tearing this solve pays, and which
        # AC-ground designation would make it pay (analysis/tearing.py).
        # Off by default -- it costs a graph scan plus numeric solves.
        self.split_lbl = QLabel("")
        self.split_lbl.setWordWrap(True)
        self.split_lbl.setStyleSheet(f"color: {theme.MUTED};")
        # a wrapped label's sizeHint is WIDE: left unchecked it inflates the
        # keep pane's minimum and flips the main splitter's proportions
        # (caught by test_splitter_gives_plots_the_width). Let it take
        # whatever width the pane has instead of asking for its own.
        self.split_lbl.setMinimumWidth(1)
        self.split_lbl.setSizePolicy(QSizePolicy.Ignored,
                                     QSizePolicy.Preferred)
        splitb = QPushButton("Split")
        splitb.setToolTip("Rank the tearing cuts for this solve and, when "
                          "none pays, name the AC ground that would fix it")
        splitb.clicked.connect(self._split_advice)
        # its OWN row: appending to `ctl` grows that row's minimum width,
        # which pushes the keep pane wider and steals the plot pane's space
        # (test_splitter_gives_plots_the_width)
        # solver backend: auto by default, re-resolved on every keep-set
        # change (the crossover means one more symbol can pick the sparse
        # path and run FASTER), with an expert override. Session-scoped
        # on purpose -- a persisted override would silently outlive the
        # circuit it was right for.
        self.backend_combo = QComboBox()
        for label, tip in (
            ("auto", "let the selector choose by problem size: the exact "
                     "dense grid below the crossover, the sparse "
                     "Ben-Or/Tiwari path above it"),
            ("qq — dense exact", "exact rational grid; every cell of the "
                                 "coefficient tensor"),
            ("bot — sparse", "Ben-Or/Tiwari: probes until the support is "
                             "pinned, so cost scales with the number of "
                             "NONZERO terms, not the grid"),
            ("zp — mod-p dense", "dense grid in machine-word modular "
                                 "arithmetic, CRT-lifted"),
            ("ratfun — rational", "sparse rational reconstruction"),
        ):
            self.backend_combo.addItem(label)
            self.backend_combo.setItemData(self.backend_combo.count() - 1,
                                           tip, Qt.ToolTipRole)
        self.backend_combo.setToolTip(
            "Which solver runs. Leave on auto unless you are testing: the "
            "estimate line names the backend the current keep set resolves "
            "to, and a forced backend that fails its self-check falls back "
            "to the dense grid.")
        self.backend_combo.currentIndexChanged.connect(
            self._on_backend_changed)
        brow = QHBoxLayout()
        brow.addWidget(QLabel("backend:"))
        brow.addWidget(self.backend_combo, 1)

        srow = QHBoxLayout()
        srow.addWidget(self.estimate_lbl)
        srow.addWidget(splitb)
        srow.addWidget(self.split_lbl, 1)

        # the controls stay alive as STATE (the budget, the grouping,
        # the backend, the split verdict) but out of the layout
        for w in (rankb, suggestb, splitb, self.group_chk,
                  self.budget_spin, self.backend_combo, self.split_lbl):
            # parented to the WINDOW, not the box: even hidden, a child
            # of the group counted in its minimum width (214 px)
            w.setParent(self)
            w.setVisible(False)
        # ticks survive reranks by design (they are stated intent),
        # but a rerank can sink them below the fold -- the field report
        # solved with 5 symbols while believing in 2. The count is
        # always visible, the filter shows exactly what is ticked, and
        # clear starts over.
        self.ticked_chk = QCheckBox("ticked only (0)")
        self.ticked_chk.setToolTip(
            "Show only the ticked symbols — wherever the rerank sorted "
            "them")
        self.ticked_chk.toggled.connect(
            lambda _on: self._apply_keep_filter())
        clearb = QPushButton("clear ticks")
        clearb.setFlat(True)
        clearb.setToolTip("Untick every symbol: the next solve is "
                          "numeric until you tick again")
        clearb.clicked.connect(self._clear_keep_ticks)
        trow = QHBoxLayout()
        trow.addWidget(self.ticked_chk)
        trow.addStretch(1)
        trow.addWidget(clearb)
        v = QVBoxLayout()
        v.addWidget(self.keep_filter)
        v.addLayout(trow)
        v.addWidget(self.keep_tbl, 1)
        v.addWidget(self.estimate_lbl)
        return self._collapsible(box, v)

    def _z_rank_target(self):
        """The Impedance tool's active target spec, or None outside
        the tool (or before one is chosen)."""
        if self.mode_combo.currentText() != "Impedance":
            return None
        return getattr(self, "_imp_spec", None)

    def _rerank_for_z(self):
        """The keep table's contributors follow the response on the
        bench: in the Impedance tool they are Z's own band
        sensitivities (probe-injected), not the transfer's. A spec
        the probe cannot serve (between two nets) keeps the transfer
        ranking -- better than an empty table."""
        if self.controller is None or self._filling:
            return
        if not getattr(self, "_keep_ranked", False):
            return                     # the open chain fills it first
        t = self._z_rank_target()
        if t is None:
            return
        c = self.controller
        desc = self._imp_desc(t)
        try:
            io = self._io()
        except Exception:
            io = None
        # the ranking samples the band the PLOT shows -- without the
        # bound the sampler spans only the reactive features, and a
        # circuit whose features cluster at one resonance was scored
        # at that single frequency (the DC plateau never sampled)
        lo, hi = self.band_slider.values()
        # resolve the target NOW, on the GUI thread: when Z cannot
        # rank (a port between two internal nets), the table falls
        # back to the transfer ranking -- which misled twice in the
        # field while it fell back silently. The status bar now says
        # WHOSE ranking the table carries.
        if not c.can_rank_z(t):
            if not io:
                return
            self.statusBar().showMessage(
                f"{desc} is not a ground-referenced port — the "
                f"symbols table shows the {io[0]} → {io[1]} TRANSFER "
                f"ranking, not Z's")
            self._rerank_after_rewrite(
                f"ranking for {io[0]} → {io[1]} (transfer) …",
                ranker=lambda: c.rank_symbols(*io, fmin=lo, fmax=hi))
            return

        def rank():
            try:
                return c.rank_symbols_z(t, fmin=lo, fmax=hi)
            except Exception:                     # noqa: BLE001
                return None

        self._rerank_after_rewrite(f"ranking for {desc} …", ranker=rank)

    def _rerank_on_io(self):
        """A changed in/out pair re-ranks the table by itself (as after a
        reduction), ticks carried over -- the table is never stale and
        Rank needs no button."""
        if self.controller is None or self._filling or                 getattr(self, "_io_rerank_guard", False):
            return
        if not getattr(self, "_keep_ranked", False):
            return                         # nothing ranked yet: the open
        try:                               # chain fills it
            inp, out = self._io()
        except Exception:
            return
        if not inp or not out:
            return
        self._io_rerank_guard = True
        try:
            self._rerank_after_rewrite(f"ranking for {inp} → {out} …")
        finally:
            self._io_rerank_guard = False

    def _keep_header_menu(self, pos):
        """View options of the keep table: the peaks column, grouping."""

        menu = QMenu(self)
        for col, label in ((1, "Show dcOp column"), (2, "Show score column"),
                           (3, "Show peaks column")):
            a = menu.addAction(label)
            a.setCheckable(True)
            a.setChecked(not self.keep_tbl.isColumnHidden(col))
            a.toggled.connect(
                lambda on, col=col: self._keep_show_column(col, on))
        g = menu.addAction("Group by device")
        g.setCheckable(True)
        g.setChecked(self.group_chk.isChecked())
        g.toggled.connect(self.group_chk.setChecked)
        menu.exec(self.keep_tbl.horizontalHeader().mapToGlobal(pos))

    def _keep_show_column(self, col: int, on: bool):
        self.keep_tbl.setColumnHidden(col, not on)
        any_extra = any(not self.keep_tbl.isColumnHidden(c) for c in (1, 2, 3))
        self.keep_tbl.horizontalHeader().setVisible(any_extra)

    def _keep_menu(self, pos):
        """Row verbs: the symbol's LaTeX alias, its device's alias, the
        planner, the ranking -- what the button row used to carry."""
        from PySide6.QtWidgets import QInputDialog

        it = self.keep_tbl.itemAt(pos)
        menu = QMenu(self)
        if it is not None:
            row = it.row()
            name = self.keep_tbl.item(row, 0).text()

            def edit_alias(row=row, name=name):
                cell = self.keep_tbl.item(row, 4)
                cur = cell.text() if cell is not None else ""
                text, ok = QInputDialog.getText(
                    self, "LaTeX alias", f"LaTeX for {name} (e.g. g_{{m1}}; "
                    f"blank = default):", text=cur)
                if ok and cell is not None:
                    cell.setText(text.strip())    # itemChanged does the rest
            menu.addAction(f"LaTeX alias for {name}…", edit_alias)
            dev = self._device_of_symbol(name)
            if dev:
                menu.addAction(f"LaTeX alias for device {dev}…",
                               lambda dev=dev: self._edit_device_alias(dev))
            menu.addSeparator()
        menu.addAction("Rank symbols", self._rank_async)
        menu.addAction("Suggest keep set ≤ time budget…",
                       self._suggest_keep_dialog)
        menu.addAction("Split advice", self._split_advice)
        menu.addSeparator()
        for col, label in ((1, "Show dcOp column"), (2, "Show score column"),
                           (3, "Show peaks column")):
            a = menu.addAction(label)
            a.setCheckable(True)
            a.setChecked(not self.keep_tbl.isColumnHidden(col))
            a.toggled.connect(
                lambda on, col=col: self._keep_show_column(col, on))
        menu.exec(self.keep_tbl.viewport().mapToGlobal(pos))

    def _device_of_symbol(self, name: str):
        """The device a symbol belongs to (gm_I0_MN1 -> I0.MN1; a
        passive's value symbol IS its device), or None."""
        if self.controller is None:
            return None
        names = {d.name for d in self.controller.devices}
        if name in names:
            return name
        for dev in names:
            if name.endswith("_" + dev.replace(".", "_")):
                return dev
        return None

    def _build_backend_menu(self):
        """Analysis -> Solver backend mirrors the hidden combo both ways."""
        # the menu is HELD from where it was created; it used to be
        # re-found by matching its own title text, which quietly tied
        # the wiring to a label a rename would have broken
        menu = getattr(self, "_backend_menu", None)
        if menu is None:
            return
        for i in range(self.backend_combo.count()):
            label = self.backend_combo.itemText(i)
            a = menu.addAction(label)
            a.setCheckable(True)
            a.setStatusTip(self.backend_combo.itemData(i, Qt.ToolTipRole) or "")
            a.setChecked(i == self.backend_combo.currentIndex())
            a.triggered.connect(
                lambda _c=False, i=i: self.backend_combo.setCurrentIndex(i))
            self._backend_group.addAction(a)
            self._backend_actions.append(a)
        self.backend_combo.currentIndexChanged.connect(
            lambda i: self._backend_actions[i].setChecked(True)
            if 0 <= i < len(self._backend_actions) else None)

    def _crumb_steps(self):
        """(label, action, done) per step. "Done" is judged by what exists,
        so the crumb needs no bookkeeping calls sprinkled through the app —
        callers just _update_crumb() after anything user-visible."""
        if self.mode_combo.currentText() == "Compensate":
            # no symbols to choose, no Solve: the bench's verb is Suggest
            return [
                ("Open", "open", self.controller is not None),
                ("Match", "match", bool(self._match_groups)),
                ("Suggest", "suggest",
                 bool(getattr(self, "_comp_suggestions", [])
                      or getattr(self, "_comp_multi", None))),
            ]
        if self.mode_combo.currentText() == "Impedance":
            # the tool's own step: Solve has nothing to do until a
            # target is chosen from the trees or a wire
            return [
                ("Open", "open", self.controller is not None),
                ("Match", "match", bool(self._match_groups)),
                ("Set target", "ztarget",
                 getattr(self, "_imp_spec", None) is not None),
                ("Choose symbols", "keep",
                 self.controller is not None and bool(self.checked_keep())),
                ("Solve", "solve", self.history.count() > 0),
            ]
        return [
            ("Open", "open", self.controller is not None),
            ("Match", "match", bool(self._match_groups)),
            ("Choose symbols", "keep",
             self.controller is not None and bool(self.checked_keep())),
            ("Solve", "solve", self.history.count() > 0),
        ]

    def _update_crumb(self):
        parts = []
        current_seen = False
        for i, (label, action, done) in enumerate(self._crumb_steps(), 1):
            if done:
                color, weight = theme.GOOD, "normal"     # done: quiet green
                mark = "✓ "
            elif not current_seen:
                color, weight = theme.INFO, "bold"       # the next step
                mark = ""
                current_seen = True
            else:
                color, weight = theme.PENDING, "normal"  # not yet
                mark = ""
            parts.append(
                f'<a href="{action}" style="color:{color}; '
                f'font-weight:{weight}; text-decoration:none;">'
                f'{mark}{i} {label}</a>')
        self.crumb.setText(
            "&nbsp;&nbsp;" + " &nbsp;·&nbsp; ".join(parts))

    def _on_crumb_clicked(self, action: str):
        if action == "open":
            self.open_dialog()
        elif action == "match" and self.controller is not None:
            self.suggest_matches()
            self._update_crumb()
        elif action == "ztarget" and self.controller is not None:
            # bring the Nets tree forward -- targets are chosen from
            # the circuit, and this is where they live
            self.dev_tabs.setCurrentWidget(self.nets_tree)
            self.statusBar().showMessage(
                "impedance: set a target — right-click a net or a "
                "two-terminal element (Set as impedance target), "
                "or click a wire")
        elif action == "keep" and self.controller is not None:
            self._rank()
            self._update_crumb()
        elif action == "solve" and self.controller is not None:
            self.solve()
        elif action == "suggest" and self.controller is not None:
            self.suggest_comp()

    def _surface_match_conflicts(self, baseline=None):
        """Say OUT LOUD when matches overwrite reality, with the price
        MEASURED. Matching shares the first member's values; the engine
        warns per symbol at build time, but those are Python warnings a
        GUI user never sees -- this is the folded-cascode lesson, where
        auto-matches silently cost dB of DC gain and the only symptom was
        the plot disagreeing with the sim.

        `baseline` is the pre-match numeric Result (first light): with it,
        the strip leads with the measured response shift, which is the
        number that decides whether the matches are acceptable -- no
        parameter-ratio threshold can (the ota5t's TRUE pair differs 23%
        on gds at ~0 dB cost; the cascode strangers differ 25% at 1.6 dB)."""
        if self.controller is None:
            return
        try:
            conf = self.controller.match_conflicts()
        except Exception:
            return
        if not conf:
            return
        delta = None
        if baseline is not None:
            try:
                import numpy as np
                inp, out = self._io()
                r = self.controller.solve(inp, out, [])
                with np.errstate(divide="ignore", invalid="ignore"):
                    delta = float(np.nanmax(np.abs(
                        20 * np.log10(np.abs(r.h / baseline.h)))))
            except Exception:
                delta = None
        param, kept, other, ratio = conf[0]
        more = f" (+{len(conf) - 1} more)" if len(conf) > 1 else ""
        cost = (f"moved the model by {delta:.2g} dB (measured): "
                if delta is not None else "")
        severity = "warn" if (delta is None or delta > 0.1) else "info"
        self._set_strip(
            f"⚠ matches {cost}{param} differs "
            f"{ratio:.1f}× between {kept} and "
            f"{other}{more} -- the model follows {kept}. Unmatch the group "
            f"(or Clear) if this is not intended.", severity)

    def _first_light(self):
        """Solve numerically the moment a session opens — the plot pane is
        never empty, and symbolic work then reads as refinement of a result
        the user already sees. Sub-second since the all-numeric s-sweep;
        synchronous on purpose (a worker racing the user's first click is
        worse than a half-second open). NOT pushed to History: History
        records the user's solves, and this is scaffolding."""
        inp, out = self._io()
        r = self.controller.attach_template(
            self.controller.solve(inp, out, []))
        # the first light IS the transfer: shown only when the session
        # opens INTO the Transfer tool. A restored Impedance session
        # once opened onto a transfer Bode it never asked for (field
        # report); other tools start with an empty plot, and the
        # computed result waits for a switch to Transfer.
        self._first_light_result = r
        mode = self.mode_combo.currentText()
        if mode == "Transfer":
            self._show(r, push_history=False)
            self.statusBar().showMessage(
                f"first light: {inp} → {out} numeric, "
                f"{r.dc_gain_db:.2f} dB — tick symbols and Solve to refine")
        else:
            self.statusBar().showMessage(
                f"{mode}: pick a target and Solve — the plot fills with "
                f"this tool's own result")

    @staticmethod
    def _collapsible(box, body_layout, collapsed=False):
        """Give a group box a collapse toggle: the checkbox in its title
        shows/hides the body. Collapsing is layout relief, not state -- the
        widgets keep working while hidden."""
        body = QWidget()
        body.setLayout(body_layout)
        outer = QVBoxLayout()
        outer.setContentsMargins(2, 2, 2, 2)
        outer.addWidget(body)
        box.setLayout(outer)
        box.setCheckable(True)
        box.setChecked(not collapsed)
        body.setVisible(not collapsed)
        box.toggled.connect(body.setVisible)
        return box

    def _split_advice(self):
        self._split_advice_run()

    def _split_advice_run(self):
        """Ask whether this solve is worth tearing, and show the verdict."""
        if self.controller is None:
            return
        inp, out = self._io()
        self.split_lbl.setText("split: analysing…")
        try:
            adv = self.controller.advise_split(inp, out, self.checked_keep())
        except Exception as exc:
            self.split_lbl.setText(f"split: — ({type(exc).__name__}: {exc})")
            return
        best = adv.cuts[0] if adv.cuts else None
        self.split_lbl.setStyleSheet(
            f"color: {theme.GOOD};" if best and best["pays"]
            else f"color: {theme.MUTED};")
        self.split_lbl.setText("split: " + adv.verdict())
        # the label is hidden state now; the verdict reads in the strip
        self._set_strip("split: " + adv.verdict(), "info")

    def _apply_keep_filter(self, text: str | None = None):
        if text is None:
            text = self.keep_filter.text()
        text = text.strip().lower()
        only = getattr(self, "ticked_chk", None) is not None \
            and self.ticked_chk.isChecked()
        for i in range(self.keep_tbl.rowCount()):
            it = self.keep_tbl.item(i, 0)
            hide = (bool(text) and text not in it.text().lower()) \
                or (only and it.checkState() != Qt.Checked)
            self.keep_tbl.setRowHidden(i, hide)
        self._refresh_tick_count()

    def _refresh_tick_count(self):
        chk = getattr(self, "ticked_chk", None)
        if chk is not None:
            chk.setText(f"ticked only ({len(self.checked_keep())})")

    def _clear_keep_ticks(self):
        self._filling = True
        try:
            for i in range(self.keep_tbl.rowCount()):
                it = self.keep_tbl.item(i, 0)
                if it is not None:
                    it.setCheckState(Qt.Unchecked)
        finally:
            self._filling = False
        self._apply_keep_filter()
        self._update_estimate()
        self._refresh_solve_hint()

    def _history_group(self):
        from PySide6.QtWidgets import QListWidget

        box = QGroupBox("History")
        box.setToolTip("Every solve of this session. Click to re-show; "
                       "select several to overlay them on one Bode.")
        self.history = QListWidget()
        self.history.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.history.itemSelectionChanged.connect(self._on_history_selected)
        self._history_results = []
        self._showing_from_history = False
        v = QVBoxLayout()
        v.addWidget(self.history)
        # collapsed by default: History earns its space only once there is
        # history, and the click that expands it is the click that wants it
        return self._collapsible(box, v, collapsed=True)

    def _history_label(self, r) -> str:
        keep = r.keep
        try:
            nkeep = "ALL" if not isinstance(keep, list) else str(len(keep))
        except Exception:
            nkeep = "?"
        extra = (f"  PM {r.pm_deg:.1f}°" if r.pm_deg is not None
                 else f"  {r.dc_gain_db:.1f} dB")
        red = ("  [reduced]"
               if getattr(r, "circuit_state", "") == "reduced" else "")
        return f"{r.inp} → {r.out}  [keep {nkeep}]{extra}{red}"

    def _push_history(self, result):
        if any(r is result for r in self._history_results):
            return
        self._history_results.append(result)
        self.history.addItem(self._history_label(result))
        self.history.scrollToBottom()

    def _on_history_selected(self):
        rows = sorted(i.row() for i in self.history.selectedIndexes())
        if not rows:
            return
        picked = [self._history_results[r] for r in rows]
        self._showing_from_history = True
        try:
            self._show(picked[0], overlays=picked[1:])
        finally:
            self._showing_from_history = False

    def _set_strip(self, text: str, severity: str = "info"):
        self._strip_sev = severity      # a theme switch re-tints it
        colors = theme.SEVERITY
        bg, fg = colors.get(severity, colors["info"])
        self.msg_strip.setStyleSheet(
            f"QLabel {{ background: {bg}; color: {fg}; padding: 3px 8px;"
            f" border-radius: 2px; }}")
        self.msg_strip.setText(text)
        self.msg_strip.setVisible(bool(text))

    def _clear_strip(self):
        self.msg_strip.hide()
        self.msg_strip.setText("")

    def _imp_desc(self, spec) -> str:
        """The target label's language: what Solve will compute."""
        if "seen_by" in spec:
            return f"seen by {spec['seen_by']}"
        if "port" in spec:
            return f"at marker {spec['port']}"
        if spec.get("ref"):
            return f"between nodes {spec['node']} and {spec['ref']}"
        return f"from node {spec['node']} to gnd"

    def _imp_update_target_lbl(self):
        spec = getattr(self, "_imp_spec", None)
        if spec is None:
            self.imp_target_lbl.setText(
                "none — right-click a net or element, or click a wire")
            self.imp_target_lbl.setStyleSheet(f"color: {theme.MUTED};")
        else:
            self.imp_target_lbl.setText(self._imp_desc(spec))
            self.imp_target_lbl.setStyleSheet("")

    def _impedance_at_nets(self, net: str, ref: str = ""):
        """A Nets-tree pick: one net is Z to ground, two are Z between
        them. Sets the TARGET only; Solve computes."""
        spec = {"node": net, "ref": ref} if ref else {"node": net}
        self._impedance_user_spec(spec)

    def _impedance_seen_by(self, name: str):
        """An Instances-tree pick on a two-terminal element: the
        impedance the element sees at its terminals -- the element
        removed, sources zeroed (Thevenin looking outward). Sets the
        TARGET only; Solve computes."""
        self._impedance_user_spec({"seen_by": name})

    def _impedance_user_spec(self, spec, solve=False):
        """THE target setter: stores the spec, says it in the toolbar
        label, re-ranks the keep table for it. The impedance itself
        is computed by Solve alone (a selection is not a command)."""
        if self.controller is None:
            return
        self.mode_combo.setCurrentText("Impedance")
        self._imp_spec = spec
        self._imp_update_target_lbl()
        self._update_crumb()          # the Set-target step just landed
        desc = self._imp_desc(spec)
        # shown BEFORE the rerank, whose fallback notice must survive
        self.statusBar().showMessage(
            f"impedance target: {desc} — Solve to compute")
        self._rerank_for_z()
        if not solve:
            return
        keep = self.checked_keep()
        self._launch(lambda cb: self.controller.impedance_result(
                         spec, keep=keep),
                     f"impedance {desc} …")

    def _impedance_target_from_selection(self):
        """In the Impedance tool, SELECTING a net -- Nets tree or a
        wire click -- chooses the target: the toolbar label says it
        and the keep ranking re-ranks for that Z, but nothing solves
        until Solve (field request). One net is Z to ground, two are
        Z between them."""
        if self.controller is None                 or self.mode_combo.currentText() != "Impedance":
            return
        sel = self.nets_tree.selected_nets()
        if not sel or len(sel) > 2:
            return
        spec = ({"node": sel[0]} if len(sel) == 1
                else {"node": sel[0], "ref": sel[1]})
        if spec == getattr(self, "_imp_spec", None):
            return
        self._impedance_user_spec(spec)

    def _on_port_text(self, text: str):
        """Loop-gain vocabulary only: the Impedance target lives in
        _imp_spec with its own toolbar label and setters."""
        if (text and self.controller is not None
                and self.mode_combo.currentText() == "Loop gain"):
            self._update_estimate()

    def _show_loop_reference(self) -> bool:
        """Loop-bench first light: the run's ground-truth loop gain
        (stb results, or the declared-AC curve) plotted alone -- it is
        already in the run, so it shows without a model solve. Draws
        nothing and returns False when the run carries no truth."""
        c = self.controller
        ref = c.loop_reference() if c is not None else None
        if ref is None:
            return False
        freqs, h, label, margins = ref
        view.reference_figure(freqs, h, label, margins,
                              fig=self.canvas.figure)
        theme.style_figure(self.canvas.figure)
        self.canvas.draw_idle()
        # the figure no longer shows self.result (e.g. the transfer
        # first light): remember, so re-entering Transfer restores it
        self._ref_on_screen = True
        return True

    def _on_mode_changed(self, mode: str):
        # Impedance is NOT probed: its target is a spec with its own
        # toolbar label, never the probe combo (a port dropdown twice
        # showed a ranking that was not the port's)
        probed = mode in ("Loop gain", "Compensate", "Modes", "GFT")
        self._refresh_solve_hint()
        if self.controller is not None and probed:
            want = self.controller.probes
            have = [self.probe_combo.itemText(i)
                    for i in range(self.probe_combo.count())]
            if have != list(want):
                self.probe_combo.clear()
                self.probe_combo.addItems(want)
        imp = mode == "Impedance"
        self._imp_tpre_act.setVisible(imp)
        self._imp_target_act.setVisible(imp)
        if imp:
            self._imp_update_target_lbl()
        # Compensate owns the probe in ITS panel (the toolbar combo
        # stays the state, populated above; the panel mirrors it)
        show_probe = probed and mode != "Compensate"
        self._probe_lbl_act.setVisible(show_probe)
        self._probe_act.setVisible(show_probe)
        if mode == "Compensate":
            self._comp_sync_probe(self.probe_combo.currentText())
        self._update_crumb()          # the steps are the MODE's steps
        self._probe2_act.setVisible(mode == "Modes")
        # GFT needs the input/output designations alongside the probe;
        # the Reduce bench prices grounding against the in->out transfer
        io_on = mode in ("Transfer", "GFT", "Reduce circuit")
        self.out_combo.setEnabled(io_on)
        self.in_combo.setEnabled(io_on)
        # tools that never read the pair HIDE it instead of graying it
        # (field report: Impedance showing in/out invited reading Z
        # against a transfer pair it ignores)
        for a in (self._in_lbl_act, self._in_act,
                  self._out_lbl_act, self._out_act):
            a.setVisible(io_on)
        # a session that opened into another tool kept its first light
        # off screen; entering Transfer with nothing solved shows it
        if mode == "Transfer" and self.result is None \
                and getattr(self, "_first_light_result", None) is not None:
            self._show(self._first_light_result, push_history=False)
        elif mode == "Transfer" and self.result is not None \
                and getattr(self, "_ref_on_screen", False):
            # the loop reference borrowed the plot: give it back
            self._show(self.result, push_history=False)
        elif mode in ("Loop gain", "Compensate") and \
                (self.result is None
                 or getattr(self.result, "pm_deg", None) is None):
            # these benches have their own first light: the run's
            # ground-truth loop gain, no model solve needed -- shown
            # whenever the plot holds no loop result (a transfer on a
            # loop bench says nothing about the loop)
            self._show_loop_reference()
        # the result-form selector serves the two tools that solve a
        # response: Transfer and Impedance (parity -- one contract
        # ladder, wherever a response is solved)
        transfer = mode == "Transfer"
        self._form_act.setVisible(
            transfer or mode in ("Impedance", "Loop gain", "GFT"))
        # the GFT bench is numeric: full order (expression pruning)
        # has nothing to act on there -- the entry grays out, and a
        # stranded selection falls back to Exact
        fo = self.form_combo.findText("Simplified · full order")
        it = self.form_combo.model().item(fo) if fo >= 0 else None
        if it is not None:
            it.setEnabled(mode != "GFT")
        if mode == "GFT" and                 self.form_combo.currentText() == "Simplified · full order":
            self.form_combo.setCurrentText("Exact")
        self._refresh_form_ui()
        for b in (self.a_simplify, self.a_reduce):
            b.setEnabled(transfer and self.controller is not None)
        if probed and self.controller is not None \
                and self.probe_combo.count() == 0:
            self.probe_combo.addItems(self.controller.probes)
        if mode == "Modes" and self.controller is not None \
                and self.probe2_combo.count() == 0:
            self.probe2_combo.addItems(self.controller.probes)
            if self.probe2_combo.count() > 1:
                self.probe2_combo.setCurrentIndex(1)
        # keep the Tool dropdown honest under programmatic mode changes:
        # a mode driven from code or tests may be filtered out of the
        # dropdown -- reinstate it rather than show a lie
        if self.tool_combo.currentText() != mode:
            self.tool_combo.blockSignals(True)
            ix = self.tool_combo.findText(mode)
            if ix < 0:
                self.tool_combo.addItem(mode)
                ix = self.tool_combo.count() - 1
            self.tool_combo.setCurrentIndex(ix)
            self.tool_combo.blockSignals(False)
        # each bench shows its own views; the shared result views stay
        self._set_tab_visibility(mode)
        # the keep ranking follows the bench's response: Z's own
        # sensitivities inside the Impedance tool, the transfer's
        # everywhere else (restored on the way out)
        prev = getattr(self, "_last_mode", None)
        self._last_mode = mode
        if mode == "Impedance":
            # a net already selected when the tool opens becomes the
            # target -- the flow 'select the net, then pick the tool'
            self._impedance_target_from_selection()
            self._rerank_for_z()
            if getattr(self, "_imp_spec", None) is None:
                self.statusBar().showMessage(
                    "impedance: set a target — right-click a net or a "
                    "two-terminal element (Set as impedance target), "
                    "or click a wire")
        elif prev == "Impedance":
            self._rerank_on_io()
        if self.controller is not None:
            self._update_estimate()
        if mode == "Compensate":
            self.tabs.setCurrentWidget(self._comp_tab)
        if mode == "GFT":
            self.tabs.setCurrentWidget(self._gft_tab)
        if mode == "Reduce circuit":
            # the bench's own plot: as-imported vs the ticked reduction
            if self.controller is not None:
                self._plot_reduction(self.checked_acg_nodes())
        elif getattr(self, "_reduce_plot_base", None) is not None:
            # leaving the bench: the result's plot comes back
            self._reduce_plot_base = None
            if self.result is not None:
                self._show(self.result, push_history=False)

    def _on_main_tab(self, idx: int):
        """Switching to the Schematic page fits it once it has a size
        (a page first shown after the window is laid out was fitted to
        a zero-size view)."""
        if self.main_tabs.widget(idx) is self._schematic_tab:
            QTimer.singleShot(0, self.sch_view.fit)

    def _set_tab_visibility(self, mode: str):
        """Summary/Expression/Error are views of any result and stay put;
        the bench-specific tabs appear only in their bench, which is what
        makes 'bench' mean something.

        Reduce circuit is a STATE tool, not a result tool: it picks a
        node set and applies it. Everything that neither informs nor
        acts on that is clutter there -- the keep panel (the scans are
        numeric, no tick changes a cost), the match controls (editing
        matches mid-scan silently invalidates it), History, the Summary
        and Expression views (they describe a solve), and Solve itself
        (it was a hidden alias of Scan AC grounds). The Nets tree comes
        to the front: it is where a net is AC-grounded by hand and
        where the applied state shows its earth marks."""
        reduce = mode == "Reduce circuit"
        comp = mode == "Compensate"
        vis = {
            # What-if serves BOTH response benches: kept-symbolic
            # sliders with live PM on a loop-gain result (the L-K
            # margins bridge ticks the top knobs into the keep set)
            self._whatif_tab: mode in ("Transfer", "Loop gain"),
            self._comp_tab: comp,
            self._ml_tab: mode == "Loop gain",
            self._ms_tab: mode == "Loop gain",
            self._gft_tab: mode == "GFT",
            # Summary/Expression/Error are views of a SOLVED response;
            # Compensate has no Solve any more, so they say nothing
            # about what this bench does
            self.summary: not reduce and not comp,
            self._expr_tab: not reduce and not comp,
            self.err_canvas: not comp,
        }
        for w, on in vis.items():
            idx = self.tabs.indexOf(w)
            if idx >= 0:
                self.tabs.setTabVisible(idx, on)
        # Compensate is a SUGGESTION tool: its search is numeric (no
        # tick changes it) and Suggest is its one verb -- the keep
        # column and Solve both invited a 10-minute hybrid that buys
        # the bench nothing, so its own controls take the column
        self._solve_act.setVisible(not reduce and not comp)
        self._keep_col.setVisible(not reduce and not comp)
        self.reduce_col.setVisible(reduce)        # the candidates take over
        self.comp_col.setVisible(comp)
        if not comp:
            self._comp_clear_proposed()           # overlay leaves with it
        # the Margins verbs sit in the main toolbar and follow their
        # bench; Compare corners also needs a point set
        lg = mode == "Loop gain"
        for a in (self._ml_sep_act, self._ml_btn_act,
                  self._ml_whatif_act, self._ml_cplx_act):
            a.setVisible(lg)
        self._ml_cmp_act.setVisible(lg and self._ml_cmp_avail)
        if not lg:
            # the sensitivity map is the Loop-gain bench's overlay:
            # leaving the bench releases the selected singularity,
            # which takes the gradient and its wires off the schematic
            tbl = getattr(self, "ml_tbl", None)
            if tbl is not None and tbl.selectedItems():
                tbl.clearSelection()
                tbl.setCurrentCell(-1, -1)
        # a splitter child shown for the first time inherits size 0 --
        # the column then exists but cannot be seen; hand it the hidden
        # symbols column's room
        shown = self.reduce_col if reduce else self.comp_col if comp else None
        if shown is not None:
            sizes = self.left_split.sizes()
            i = self.left_split.indexOf(shown)
            if 0 <= i < len(sizes) and sizes[i] < 120:
                want = 280
                j = max((k for k in range(len(sizes)) if k != i),
                        key=lambda k: sizes[k])
                give = min(want, max(0, sizes[j] - 120))
                sizes[j] -= give
                sizes[i] = max(want, give)
                self.left_split.setSizes(sizes)
        if reduce:
            self.dev_tabs.setCurrentWidget(self.nets_tree)

    def _rerank_on_band(self):
        """The band slider settled on a new range: re-rank the keep
        table for it -- Z's own ranking in the Impedance tool, the
        in -> out response's elsewhere. The benches without a live
        keep panel (Compensate) skip."""
        if self.controller is None:
            return
        if not getattr(self, "_keep_ranked", False):
            return
        mode = self.mode_combo.currentText()
        if mode == "Impedance":
            self._rerank_for_z()
        elif mode in ("Transfer", "Loop gain", "GFT", "Reduce circuit"):
            self._rerank_on_io()

    def _on_tool_selected(self, name: str):
        if name and self.mode_combo.currentText() != name:
            self.mode_combo.setCurrentText(name)

    def _on_op_point(self, i: int):
        """Bind the session to another discovered operating point. The
        structure survives (topology, matches, reduction, ticks); every
        result now describes the OTHER point and re-derives -- the
        Solve * marker says so through the circuit key."""
        c = self.controller
        if c is None or not c.op_points or i < 0 or i == c.op_index:
            return
        c.set_op_point(i)
        label = c.op_point_labels[i]
        self._ml_clear()                   # the ledger described the
        self._refresh_tools()              # other point's OP
        if self.controller is not None:
            self._update_estimate()
        if self.mode_combo.currentText() in ("Loop gain", "Compensate"):
            self._show_loop_reference()    # this point's own truth
        try:
            self._schematic_render()
        except Exception:                  # noqa: BLE001
            pass                           # decoration only
        self._refresh_solve_hint()
        self._set_strip(
            f"operating point: {label} — analyses re-derive on this "
            f"point's OP", "info")

    def declare_return_ratio(self):
        """Analysis menu: the user states that the run's AC data is a
        return-ratio capture. A declaration, not a computation — it is
        the one way the loop benches open without stb results."""
        if self.controller is None:
            return
        from PySide6.QtWidgets import QInputDialog

        c = self.controller
        none_ = "(no declaration)"
        items = [none_] + list(c.nets)
        cur = items.index(c.ac_loop_gain) if c.ac_loop_gain in items else 0
        if cur == 0:
            # the obvious candidate: the out net the user is looking at
            # -- declaring "my current transfer IS the loop gain" is the
            # expected gesture, so it is the preselection, not a hunt
            # through the net list
            _, out = self._io()
            if out in items:
                cur = items.index(out)
        try:
            inp = c.suggested_input()    # the ratio's denominator: the
        except Exception:                # run's AC source, not a choice
            inp = "input"
        choice, ok = QInputDialog.getItem(
            self, "Declare AC as return ratio",
            f"v(net)/v({inp}) in this run's AC sweep IS the loop gain:",
            items, cur, False)
        if not ok:
            return
        c.declare_ac_loop_gain(None if choice == none_ else choice)
        self._refresh_tools()
        if c.ac_loop_gain:
            self._set_strip(
                f"AC declared as return ratio at v({c.ac_loop_gain}) — "
                f"loop benches use it as ground truth", "info")
        else:
            self._set_strip("return-ratio declaration withdrawn", "info")

    def _tool_ok(self, requirement: str) -> bool:
        c = self.controller
        if not requirement:
            return True
        if c is None:
            return False
        loop = c.has_stb or bool(c.ac_loop_gain)
        if requirement == "loop":
            return loop
        return c.has_stb and len(c.tagged_probes) >= 2     # "modes"

    def _refresh_tools(self):
        """Rebuild the Tool dropdown from what the loaded run can GROUND
        (per _TOOLS requirements); the hidden mode combo keeps every
        entry as the state axis, with unavailable ones disabled. Falls
        back to Transfer when the current tool loses its data."""
        cur = self.mode_combo.currentText()
        self.tool_combo.blockSignals(True)
        self.tool_combo.clear()
        for name, tip, req in _TOOLS:
            if self._tool_ok(req):
                self.tool_combo.addItem(name)
                self.tool_combo.setItemData(self.tool_combo.count() - 1,
                                            tip, Qt.ToolTipRole)
        for i in range(self.mode_combo.count()):
            req = next((rq for nm, _, rq in _TOOLS
                        if nm == self.mode_combo.itemText(i)), "")
            self.mode_combo.model().item(i).setEnabled(self._tool_ok(req))
        ok = self.tool_combo.findText(cur) >= 0
        if ok:
            self.tool_combo.setCurrentText(cur)
        self.tool_combo.blockSignals(False)
        if not ok:
            self.mode_combo.setCurrentText("Transfer")
            self.mode_combo.setToolTip(
                f"{cur} needs stb results in the run — or declare the "
                f"AC data as a return ratio (Analysis menu)")
        else:
            self.mode_combo.setToolTip("")

    def _start_advisor(self, probe: str):
        """Second worker: the probe-adequacy verdict arrives a few seconds
        after the loop gain itself -- the strip says so meanwhile."""
        self._set_strip(f"advisor: grading probe {probe}…",
                        "info")

        def fn(_cb):
            return self.controller.assess_probe(probe)

        self._advisor_thread = _Worker(fn)
        self._track(self._advisor_thread)
        self._advisor_thread.done.connect(self._on_advisor_done)
        self._advisor_thread.failed.connect(
            lambda msg: self._set_strip(f"advisor failed: {msg}", "warn"))
        self._advisor_thread.start()

    def _on_advisor_done(self, report):
        verdict = report.verdict()
        sev = "ok"
        if "UNSTABLE" in verdict or "MISLEADING" in verdict:
            sev = "error"
        elif "unobserved" in verdict or "deviates" in verdict:
            sev = "warn"
        self._set_strip("advisor: " + verdict, sev)

    # ----------------------------------------------------------------- open
    @staticmethod
    def _open_error_text(exc) -> str:
        """The open-failure dialog's text, with the diagnosis the raw
        join error only implies (field case: an edited schematic
        Analyzed against the pre-edit simulation)."""
        msg = f"{type(exc).__name__}: {exc}"
        if "no OP record" in str(exc):
            msg += ("\n\nDevices in the schematic have no simulated "
                    "OP: the simulation predates the netlist (re-run "
                    "it, then Analyze again), or the devices are "
                    "DEACTIVATED in Virtuoso -- the netlister skips "
                    "those, and the export now skips them too, so a "
                    "fresh Analyze clears it.")
        return msg

    def open_dialog(self):
        cin, _ = QFileDialog.getOpenFileName(
            self, "Open CIN topology", "", "CIN (*.cin.json *.json)")
        if not cin:
            return
        psf = QFileDialog.getExistingDirectory(
            self, "Select psf results directory", str(Path(cin).parent))
        if not psf:
            return
        try:
            self.open_session(cin, psf)
        except Exception as exc:
            QMessageBox.critical(self, "Open failed",
                                 self._open_error_text(exc))

    def _on_cap_model_toggled(self, matrix_on: bool):
        self.cap_model = "matrix" if matrix_on else "lumped"
        self._save_cap_model()
        self._reopen_with_model(f"cap model: {self.cap_model}")

    def _on_mos_model_toggled(self, lump_on: bool):
        self.mos_model = "lumped-gmb" if lump_on else "separate"
        self._save_cap_model()
        self._reopen_with_model(
            "ĝm lumping: " + ("on" if lump_on else "off"))
        if self.controller is None or not lump_on:
            return
        # say per device what the toggle actually did -- the criterion
        # is exact and therefore selective, and "nothing qualified" must
        # not read as "it worked everywhere"
        info = self.controller.lumped_gmb()
        lumped = sorted(n for n, how in info.items() if how == "lumped")
        dropped = sorted(n for n, how in info.items() if how != "lumped")
        bits = []
        if lumped:
            bits.append(f"ĝm = gm+gmb on {len(lumped)} device(s): "
                        f"{', '.join(lumped)}")
        if dropped:
            bits.append(f"inert gmbs dropped (bulk=source) on "
                        f"{len(dropped)}: {', '.join(dropped)}")
        if not bits:
            bits.append("no device qualifies -- no gate/bulk pair at "
                        "the same AC potential, every gmb stays "
                        "separate")
        self.log("ĝm lumping: " + "; ".join(bits))

    def _reopen_with_model(self, label: str):
        """The shared model-toggle flow: the model is baked into the
        reconstruction, so re-open the same run -- preserving the
        user's in/out, matches and keep ticks -- and re-run whatever
        was last shown so the change is visible immediately."""
        if self.controller is None:
            return
        inp, out = self._io()
        mode = self.mode_combo.currentText()
        groups = list(self._match_groups)
        checked = self.checked_keep()
        had_result = self.result is not None
        with_probe = self.probe_combo.currentText()
        self.statusBar().showMessage(f"re-opening ({label}) …")
        QApplication.setOverrideCursor(Qt.WaitCursor)
        # keep the user's result on screen through the re-open: the
        # first-light numeric solve would contradict the keep panel
        self._suppress_first_light = had_result
        try:
            self.open_session(self._cin, self._psf)
        except Exception as exc:
            self._set_strip(f"re-open failed: {exc}", "error")
            return
        finally:
            QApplication.restoreOverrideCursor()
            self._suppress_first_light = False
        if inp:
            self.in_combo.setCurrentText(inp)
        if out:
            self.out_combo.setCurrentText(out)
        if groups:
            self._match_groups = groups
            self._apply_matches()
        self.mode_combo.setCurrentText(mode)
        if with_probe and self.probe_combo.findText(with_probe) >= 0:
            self.probe_combo.setCurrentText(with_probe)
        # restore the ticks LAST: the mode restore refills the table with
        # the new model's auto-plan, which used to clobber the user's
        # carefully built set with suggested cjd_* additions. Symbols
        # that do not exist under the new cap model (matrix k** vs
        # lumped c**) cannot survive; everything that does, does.
        if checked:
            try:
                ranking = self.controller.rank_symbols(
                    *self._io(), fmin=self.band_slider.values()[0],
                    fmax=self.band_slider.values()[1])
                self._fill_keep_table(ranking, checked=checked)
            except Exception as exc:
                self._set_strip(f"keep ticks could not be restored after "
                                f"the cap-model switch "
                                f"({type(exc).__name__}) — re-Rank and "
                                f"re-tick", "warn")
        self.statusBar().showMessage(label)
        self.log(f"{label} (run re-opened)")
        # re-run the last shown analysis WITH THE USER'S OWN KEEP TICKS,
        # async with progress and Cancel. The previous result stays on
        # screen until the new one lands, so display and keep panel
        # never disagree — and the keep set itself is never touched
        # (it is accumulated work; losing it on a toggle is a surprise
        # nobody asked for).
        if had_result and mode in ("Transfer", "Loop gain"):
            self._set_strip(
                f"{label} — showing the previous model's result until "
                f"the re-solve with your keep set lands", "info")
            try:
                self.solve()
            except Exception as exc:
                self._set_strip(f"{label} — the promised re-solve "
                                f"failed to launch "
                                f"({type(exc).__name__}); press Solve",
                                "warn")
        elif had_result:
            self._set_strip(
                f"{label} — the shown result still uses the previous "
                f"model; press Solve to recompute with your keep set",
                "warn")

    def _flush_calibration(self):
        """Options verb: forget the learned solve-time model everywhere
        it lives -- this process, the engine child, the file -- then
        re-measure the machine in the background."""
        if QMessageBox.question(
                self, "Flush calibration",
                "Forget everything the solve-time estimator has learned "
                "on this machine?\n\nThe measured model and every "
                "per-path correction are deleted; the machine is "
                "re-measured in the background and estimates re-learn "
                "from your next solves.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No) != QMessageBox.Yes:
            return
        from ..analysis import estimate as _est
        _est.flush_calibration()                  # this process + file
        if self.controller is not None:
            try:
                self.controller.flush_calibration()   # the engine child
            except Exception:                     # noqa: BLE001
                pass
        self.log("solve-time model: calibration flushed by request")
        self.statusBar().showMessage("calibration flushed — re-measuring")
        self._ensure_calibration()                # fresh measurement

    def _tail_engine_log(self):
        """Bring the engine child's stderr file into the Log, new lines
        only; quiet on any filesystem hiccup. Doubles as the idle
        watchdog: a death BETWEEN jobs would otherwise go unnoticed
        until the next call raised."""
        c = self.controller
        idle = self._thread is None or not self._thread.isRunning()
        if (idle and hasattr(c, "engine_alive") and not c.engine_alive()
                and not getattr(self, "_engine_down_handled", False)):
            self._engine_died("noticed while idle")
        try:
            with open(self._engine_log, "r", encoding="utf-8",
                      errors="replace") as f:
                f.seek(self._engine_log_pos)
                chunk = f.read()
                self._engine_log_pos = f.tell()
        except OSError:
            return
        for line in chunk.splitlines():
            line = line.strip()
            if line:
                self.log(f"engine: {line}")

    def open_session(self, cin, psf, probe=None, async_open=False):
        # cap model chosen in the Model menu; matrix is the accurate one on
        # non-reciprocal processes (SKY130 CM loops shift ~6 deg vs lumped),
        # and the GUI default -- lumped is offered for the textbook contrast
        # where the session lives is open_engine's decision: in-process
        # by default (tests, headless), the spawned engine child when
        # CIN_ENGINE=proc (launch.py sets it) -- the GIL-residue fix
        from ..engine.remote import RemoteSession, open_engine
        import tempfile
        if hasattr(self.controller, "close_engine"):
            self._engine_down_handled = True      # deliberate: no dialog
            self._engine_log_timer.stop()
            self.controller.close_engine()        # re-open: no orphan
        self._engine_down_handled = False         # the NEW engine is armed
        self._engine_log = str(Path(tempfile.gettempdir())
                               / f"cin-engine-{os.getpid()}.log")
        self._engine_log_pos = 0
        self.controller = open_engine(
            cin, psf, cap_model=getattr(self, "cap_model", "matrix"),
            mos_model=getattr(self, "mos_model", "separate"),
            stderr_path=self._engine_log)
        if isinstance(self.controller, RemoteSession):
            self._engine_log_timer.start()
        self._cin, self._psf = str(cin), str(psf)
        # the previous session's tick set, stashed BEFORE anything in
        # this open can autosave over the state file; the guard holds
        # autosave off until the open flow has filled the table (the
        # timer is the belt-and-braces release if the chain dies)
        self._reopen_keep = self._read_saved_keep()
        self._opening_guard = True
        QTimer.singleShot(30000, lambda: setattr(
            self, "_opening_guard", False))
        self._match_groups = []
        self._auto_token = None      # invalidate any in-flight auto chain
        self._open_async = bool(async_open)
        self._push_recent(str(cin), str(psf))
        self._load_aliases()
        self._populate()
        # A run that carries an stb analysis opens in the Loop-gain bench
        # with ITS designated probe preselected (--probe overrides the
        # discovery; any vsource qualifies).
        pts = self.controller.op_point_labels
        self.op_combo.blockSignals(True)
        self.op_combo.clear()
        self.op_combo.addItems(pts)
        self.op_combo.setCurrentIndex(0)
        self.op_combo.blockSignals(False)
        self._op_lbl_act.setVisible(bool(pts))
        self._op_act.setVisible(bool(pts))
        # worst-corner sizing only makes sense over a point set; on one
        # it IS the wanted default -- the corners were run to be held
        self.robust_chk.setVisible(bool(pts))
        self.robust_chk.setChecked(bool(pts))
        self._ml_cmp_avail = bool(pts)
        self._ml_cmp_act.setVisible(
            bool(pts) and self.mode_combo.currentText() == "Loop gain")
        if pts:
            self.statusBar().showMessage(
                f"operating-point set: {len(pts)} points discovered — "
                f"active: {pts[0]}")
        probe = probe or self.controller.stb_probe()
        if probe and self.probe_combo.findText(probe) >= 0:
            self.mode_combo.setCurrentText("Loop gain")
            self.probe_combo.setCurrentText(probe)
            # the bench's own first light: the stb truth is already in
            # the run, so it plots immediately -- the model joins it
            # after the first Solve (field report: an stb-only run
            # opened into Loop gain onto an empty plot)
            shown = self._show_loop_reference()
            self.statusBar().showMessage(
                f"stb analysis found: probe {probe} preselected — "
                + ("stb loop gain plotted; Solve adds the model"
                   if shown else "Solve for the loop gain"))
        if self.controller.reductions:
            # NON-modal: a modal .information() blocks on its own event loop
            # until the user clicks OK, which hangs any headless/automated open.
            box = QMessageBox(
                QMessageBox.Icon.Information, "Netlist reduced",
                "The simulator pruned components with no OP data (0-valued); "
                "CircuitInsight folded them out:\n\n  "
                + "\n  ".join(self.controller.reductions), parent=self)
            box.setModal(False)
            box.show()
            self._reduce_box = box                # keep a ref so it isn't GC'd

    def _populate(self):
        c = self.controller
        for combo, items in ((self.in_combo, c.input_ports()),
                             (self.out_combo, c.output_nets())):
            combo.clear()
            combo.addItems(items)
        si = c.suggested_input()
        if si:
            self.in_combo.setCurrentText(si)
        so = c.suggested_output()
        if so:
            self.out_combo.setCurrentText(so)
        self._filling = True
        self.probe_combo.clear()
        self.probe2_combo.clear()
        probes = c.probes
        self.probe_combo.addItems(probes)
        self.gft_ref_combo.clear()
        self.gft_ref_combo.addItems(c.nets)
        self._refresh_tools()
        self.history.clear()
        self._history_results = []
        self._report_sections = []
        self.a_export_session.setEnabled(False)
        self._clear_strip()
        # states are per-run: enable the actions, and offer the rolling
        # last-state when one exists — the truth (first light) shows
        # first, the restore is one deliberate click away
        for a in (self.a_save_state, self.a_load_state):
            a.setEnabled(True)
        from . import state as st

        has_last = st.state_path(self._cin).exists() \
            if getattr(self, "_cin", None) else False
        self.a_restore_last.setEnabled(has_last)
        if has_last and not getattr(self, "_suppress_first_light", False):
            self.statusBar().showMessage(
                "a previous session state exists — File → Restore last "
                "state brings back your selections (and the solution, "
                "when still valid)")
        devs = c.devices
        try:
            opv = c.op_values()
        except Exception:
            opv = {}
        try:
            self._ii = dict(c.impact_ionization_devices())
        except Exception:
            self._ii = {}
        rows = []
        for d in devs:
            reg = ""
            if d.device_type == "mosfet":
                reg = view.region_name(c.device_op(d.name).get("region"))
            key = d.name.replace(".", "_")
            gm = opv.get(f"gm_{key}")
            gds = opv.get(f"gds_{key}")
            ii_note = ""
            if d.name in self._ii:                # II active, no gii modeled
                ii_note = (f"impact ionization active: isub/ids = "
                           f"{self._ii[d.name]:.2%}, but no gii modeled -- "
                           f"the first-order gm/gds/gmbs reconstruction is "
                           f"incomplete here (identify gii by AC injection; "
                           f"see r2r)")
            rows.append({
                "name": d.name, "type": d.device_type, "region": reg,
                "gm": view.eng(gm, "S") if gm is not None else "",
                "gds": view.eng(gds, "S") if gds is not None else "",
                "alias": c.sym_aliases.get(d.name, ""),
                "terminals": dict(d.terminals), "ii_note": ii_note,
            })
        self.devices.populate(rows, op_fetch=c.device_op)
        conns: dict[str, list] = {}
        for d in devs:
            for term, net in d.terminals.items():
                conns.setdefault(net, []).append((d.name, term))
        self.nets_tree.populate(conns, ground=c.ground)
        self._refresh_net_decor()
        self._filling = False
        self.keep_tbl.setRowCount(0)
        # a forced backend belongs to the circuit it was chosen for
        c.set_probe_backend(None)
        was, self._filling = self._filling, True
        try:
            self.backend_combo.setCurrentIndex(0)
        finally:
            self._filling = was
        self.estimate_lbl.setText(
            "Rank scores each parameter's effect on the band — start there")
        inv = {v: k for k, v in self._POLICY_MAP.items()}
        self._filling = True
        try:
            self._set_policy_label(inv.get(c.match_value_policy, "weighted"))
        finally:
            self._filling = False
        self.acg_tbl.setRowCount(0)
        self.acg_preview.clear()
        self.acg_joint_lbl.setText("")
        self.acg_apply.setEnabled(False)
        self._refresh_reduction_banner()
        self._refresh_matches_label()
        self._ensure_calibration()   # measure this machine once
        for b in (self.solve_btn,
                  self.a_solve, self.a_simplify,
                  self.a_reduce):
            b.setEnabled(True)
        try:
            self._schematic_load()
        except Exception as e:                   # noqa: BLE001
            self._sch_status.setText(f"schematic unavailable: {e}")
        self._auto_setup(getattr(self, "_open_async", False))

    def _auto_setup(self, async_open: bool = False):
        """Make the tool SYMBOLIC by default. Left alone, the keep table opens
        empty -> keep=[] -> a purely numeric solve, so a symbolic analyzer's
        first result shows no symbols but `s`. Instead: apply the suggested
        matched pairs and pre-select a budget-fit keep set, so the first Solve
        already reads gm/(gds_n+gds_p). All heuristic and all reversible (Clear
        matches, untick symbols); guarded so a hiccup never blocks opening.

        async_open (the launcher's path): first light runs in the solve
        worker and the rest of the chain in background advisors, so the
        WINDOW shows in about a second instead of hiding 8+ seconds of
        session work behind the splash (measured on fc: first light
        2.4 s, keep plan 4.1 s, ranking 0.8 s, conflict solve 0.9 s).
        The synchronous path remains for tests and model re-opens.
        """
        c = self.controller
        name = Path(str(c.cin_path)).name
        n = len(c.devices)
        carries = c.analyses()
        probe = c.stb_probe()
        if carries:
            txt = " ".join(carries)
            if probe:
                txt += f" (stb probe {probe})"
            self.mode_combo.setToolTip("simulator truth in this run: " + txt)
        # A model re-open (cap/mos toggle) owns its own restoration: the
        # handler brings back the user's matches and ticks and re-solves.
        # Running the auto plan under it was exactly the pollution the
        # correlation work removed -- skip the whole chain.
        if getattr(self, "_suppress_first_light", False):
            self._update_crumb()
            return
        if async_open:
            self._auto_setup_async(name, n, carries)
            self._update_crumb()
            return
        try:
            inp, out = self._io()
            # FIRST LIGHT BEFORE MATCHES: the first thing on screen is the
            # honest unmatched model against the sim reference. Auto-matches
            # apply after -- so if they move the model, the user has seen
            # the truth once, and the conflicts strip below says why.
            self._first_light()
            groups = [tuple(g) for g in c.suggest_matches()]
            if groups:
                self._match_groups = groups
                self.controller.set_matches(*groups)
                self._refresh_matches_label()
                self._surface_match_conflicts(baseline=self.result)
            plan = c.suggest_keep(inp, out, self.budget_spin.value())
            lo, hi = self.band_slider.values()
            ranking = c.rank_symbols(inp, out, fmin=lo, fmax=hi)
            saved = self._restored_ticks(ranking, plan.keep)
            self._fill_keep_table(
                ranking,
                checked=saved if saved is not None else list(plan.keep))
            self._update_estimate()          # the empty-state hint yields
            if saved is None:
                self._auto_status(name, n, carries, plan.keep)
            self._ii_strip()
        except Exception as exc:
            # fall back to the manual flow -- never let auto-setup break opening
            self.statusBar().showMessage(
                f"{name}: {n} devices — Suggest matches, Rank, Solve "
                f"(auto-setup skipped: {type(exc).__name__}).")
        self._opening_guard = False          # autosave may write again
        self._update_crumb()

    def _read_saved_keep(self):
        """The last-state file's tick list for the CIN being opened, or
        None; called ONCE at open start, before this open can touch
        the file."""
        import json
        import zipfile

        from . import state as st
        try:
            p = st.state_path(self._cin)
            if not p.exists():
                return None
            with zipfile.ZipFile(p) as z:
                manifest = json.loads(z.read("manifest.json"))
            saved = manifest.get("keep")
            return None if saved is None else [str(s) for s in saved]
        except Exception:                         # noqa: BLE001
            return None

    def _restored_ticks(self, ranking, plan_keep):
        """The last session's tick set for this CIN, validated against
        the CURRENT model, or None (no state file, unreadable file, or
        a saved set that matches nothing here -- a mangled or
        different-model state yields to the auto plan, and says so).
        Restoring the SELECTIONS by default is policy (user decision
        2026-08-29: the tick count is visible, so a forgotten
        low-ranked symbol is not silently restored); the SOLUTION
        stays behind File -> Restore last state, fingerprint-gated.
        Reads the stash taken at open_session start -- the file itself
        may be re-written the moment this open shows a result. Logs
        its own story; the caller only fills the table."""
        saved = getattr(self, "_reopen_keep", None)
        if saved is None:
            return None
        have = {n for n, _s, _p in ranking}
        alive = [s for s in saved if s in have]
        if saved and not alive:
            self.log(f"last session's {len(saved)} ticked symbol(s) "
                     f"match nothing in this model (a different cap "
                     f"model, or a stale state) — using the auto plan")
            return None
        if len(alive) < len(saved):
            self.log(f"restored {len(alive)} of {len(saved)} ticked "
                     f"symbol(s) from the last session (the rest no "
                     f"longer exist under this model; the auto plan "
                     f"suggested {len(plan_keep)})")
        else:
            self.log(f"restored {len(alive)} ticked symbol(s) from "
                     f"the last session (the auto plan suggested "
                     f"{len(plan_keep)})")
        self.statusBar().showMessage(
            f"restored your {len(alive)} ticked symbols from the last "
            f"session — Solve, or File → Restore last state for the "
            f"full state (band, matches, solution)")
        return alive

    def _auto_status(self, name, n, carries, keep) -> None:
        kept = ", ".join(keep) if keep else "none within budget"
        ref_note = (" | run carries: " + " ".join(carries)
                    if carries else "")
        self.statusBar().showMessage(
            f"{name}: {n} devices — auto keep-set [{kept}]; "
            f"Solve for a symbolic result, or edit the ticks.{ref_note}")

    def _ii_strip(self) -> None:
        ii = getattr(self, "_ii", {})
        if ii:
            worst = sorted(ii, key=lambda k: -ii[k])[:4]
            self._set_strip(
                "\N{WARNING SIGN} impact ionization active without a gii "
                "model on " + ", ".join(f"{k} ({ii[k]:.1%})" for k in worst)
                + " -- the first-order model is incomplete here (identify "
                "gii by AC injection; see the r2r case)", "warn")

    def _auto_setup_async(self, name, n, carries) -> None:
        """The launcher's open chain: first light in the solve worker
        (progress + Cancel), then matches -> conflicts + keep plan +
        ranking in background advisors, each landing on the GUI thread
        as it completes. A token guards every landing: a re-open, a
        model toggle or a restored state invalidates the chain, and the
        keep table is only auto-filled while it is still empty -- the
        user's (or a state's) ticks are never clobbered by a late
        plan."""
        c = self.controller
        self._auto_token = tok = object()
        try:
            inp, out = self._io()
        except Exception:
            self.statusBar().showMessage(f"{name}: {n} devices — pick "
                                         f"in/out and Solve")
            return

        def live() -> bool:
            return self._auto_token is tok and self.controller is c

        def fl(_cb):
            return c.attach_template(c.solve(inp, out, []))

        def fl_done(r):
            # the FIRST LIGHT closing line is _launch's (verdict=)
            self._tick.stop()
            self._t0 = None
            self.progress.hide()
            self.cancel_btn.hide()
            self._job_finished()
            if not live():
                return
            if self.mode_combo.currentText() == "Transfer":
                self._show_result_tab()   # the shown first light's tab
            self._first_light_result = r
            if self.mode_combo.currentText() == "Transfer":
                self._show(r, push_history=False)
                self.statusBar().showMessage(
                    f"first light: {inp} → {out} numeric, "
                    f"{r.dc_gain_db:.2f} dB — matches and keep plan on "
                    f"the way …")
            else:
                self.statusBar().showMessage(
                    f"{self.mode_combo.currentText()}: pick a target and "
                    f"Solve — matches and keep plan on the way …")
            self._run_bg(lambda: [tuple(g) for g in c.suggest_matches()],
                         land_matches)

        def land_matches(groups):
            if not live():
                return
            if groups:
                self._match_groups = list(groups)
                c.set_matches(*groups)
                self._refresh_matches_label()
            # the pre-match baseline is the COMPUTED first light, not
            # the displayed result -- a session opened into another
            # tool computes it without showing it
            baseline = getattr(self, "_first_light_result", None) \
                or self.result
            band = self.band_slider.values()

            def compute():
                plan = c.suggest_keep(inp, out, self.budget_spin.value())
                ranking = c.rank_symbols(inp, out, fmin=band[0],
                                         fmax=band[1])
                matched = None
                if groups:
                    # the matched model's own first light: it warms the
                    # conflicts measure AND replaces the pre-match plot,
                    # so the screen describes the circuit the matches
                    # made (a plot from before them read as stale)
                    matched = c.attach_template(c.solve(inp, out, []))
                return plan, ranking, matched

            def land_plan(res):
                if not live():
                    return
                plan, ranking, matched = res
                if groups:
                    if matched is not None and self.result is baseline:
                        self._show(matched, push_history=False)
                    self._surface_match_conflicts(baseline=baseline)
                saved = None
                if not self.keep_tbl.rowCount():
                    saved = self._restored_ticks(ranking, plan.keep)
                    self._fill_keep_table(
                        ranking,
                        checked=(saved if saved is not None
                                 else list(plan.keep)))
                    self._update_estimate()
                if saved is None:
                    self._auto_status(name, n, carries, plan.keep)
                self._opening_guard = False  # autosave may write again
                self._ii_strip()
                self._update_crumb()

            self._run_bg(compute, land_plan)

        def start():
            if live():
                self._launch(fl, f"first light: {inp} → {out} …",
                             on_done=fl_done, est_s=None,
                             verdict="FIRST LIGHT")

        # the launcher defers the chain past the window's first paint
        # (a sympy-holding worker thread starves the first draw of the
        # GIL); a programmatic open starts it at once
        deferred = getattr(self, "_auto_deferred", None)
        if deferred is not None:
            deferred.append(start)
        else:
            start()

    # ------------------------------------------------------------- matches
    def _apply_matches(self):
        self.controller.set_matches(*self._match_groups)
        self._refresh_matches_label()
        self._surface_match_conflicts()
        self.keep_tbl.setRowCount(0)                       # ranking is now stale
        self.estimate_lbl.setText("estimate: — (re-Rank)")

    def _refresh_matches_label(self):
        """Match sets live on the tree itself: every member wears 🔗n and
        its group tint (devtree.set_groups). Each group is passed with
        its representative (the name bearer) first, so the tree orders
        the group's block by that device."""
        groups = []
        for g in self._match_groups:
            try:
                lead = self.controller.name_bearer(g) if self.controller                     else g[0]
            except Exception:                     # noqa: BLE001
                lead = g[0]
            groups.append((lead, *[m for m in g if m != lead]))
        self.devices.set_groups(groups)
        self._refresh_solve_hint()
        try:
            self._schematic_restyle()
        except Exception:                         # noqa: BLE001
            pass                                  # decoration only

    _POLICY_MAP = {"representative": "representative", "average": "mean",
                   "weighted": "weighted"}

    def _policy_label(self) -> str:
        for label, a in self._policy_actions.items():
            if a.isChecked():
                return label
        return "weighted"

    def _set_policy_label(self, label: str) -> None:
        a = self._policy_actions.get(label)
        if a is not None:
            a.setChecked(True)

    def _on_matchval_changed(self, label: str):
        if self.controller is None or self._filling:
            return
        inp, out = self._io()
        try:
            self.controller.set_match_value_policy(
                self._POLICY_MAP.get(label, "representative"), inp, out)
        except Exception as exc:
            self.statusBar().showMessage(f"match values: {exc}")
            return
        self._surface_match_conflicts()
        self.estimate_lbl.setText("estimate: — (re-Rank)")
        self.statusBar().showMessage(
            f"match values: {label} — re-Solve to see the effect")

    def _pick_representative(self, gi: int):
        """Context menu on a 🔗 member: choose which member's values stand
        for the group (and switch the policy to representative)."""
        if self.controller is None or not 0 <= gi < len(self._match_groups):
            return
        group = list(self._match_groups[gi])
        from PySide6.QtWidgets import QInputDialog

        current = self.controller.match_representative(group)
        idx = group.index(current) if current in group else 0
        name, ok = QInputDialog.getItem(
            self, "Representative device",
            "This member's values stand for the whole group:",
            group, idx, False)
        if not ok or not name:
            return
        self.controller.set_match_representative(name)
        self._filling = True
        try:
            self._set_policy_label("representative")
        finally:
            self._filling = False
        self._surface_match_conflicts()
        self.estimate_lbl.setText("estimate: — (re-Rank)")
        self.statusBar().showMessage(
            f"match values: {name} represents its group — re-Solve to "
            f"see the effect")

    def _unmatch_group(self, gi: int):
        if 0 <= gi < len(self._match_groups):
            del self._match_groups[gi]
            self._apply_matches()

    def unmatch_selected(self):
        gis = sorted({gi for gi in (self.devices.group_of(n)
                                    for n in self.devices.selected_names())
                      if gi is not None}, reverse=True)
        if not gis:
            self._set_strip("Unmatch dissolves the group of the SELECTED "
                            "🔗 device — click a linked member first, or "
                            "use Clear to remove all matches", "info")
            return
        for gi in gis:
            del self._match_groups[gi]
        self._apply_matches()

    def suggest_matches(self):
        if self.controller is None:
            return
        self._match_groups = [tuple(g) for g in self.controller.suggest_matches()]
        self._apply_matches()

    def match_selected(self):
        names = tuple(self.devices.selected_names())
        if len(names) < 2:
            self.statusBar().showMessage("select two or more devices to match")
            return
        self._match_groups.append(names)
        self._apply_matches()

    def clear_matches(self):
        self._match_groups = []
        self._apply_matches()

    # ------------------------------------------------------------- keep-set
    def _io(self):
        return self.in_combo.currentText(), self.out_combo.currentText()

    def _rank(self):
        if self.controller is None:
            return
        inp, out = self._io()
        lo, hi = self.band_slider.values()
        try:
            ranking = self.controller.rank_symbols(inp, out,
                                                   fmin=lo, fmax=hi)
        except Exception as exc:
            QMessageBox.warning(self, "Rank failed", f"{type(exc).__name__}: {exc}")
            return
        # re-ranking REORDERS the symbols; it must not silently discard
        # the user's selection (a stray F5 mid-solve emptied the keep
        # set, and the estimate then re-costed the empty one)
        self._fill_keep_table(ranking, checked=self.checked_keep())

    def _rank_async(self):
        """The Rank button's path: the band-sensitivity sweep (~0.8 s on
        fc) runs off the GUI thread; internal callers that need the
        table filled before their next line keep the synchronous
        _rank."""
        if self.controller is None:
            return
        inp, out = self._io()
        c = self.controller
        checked = self.checked_keep()
        self.statusBar().showMessage("ranking …")

        def done(ranking):
            if self.controller is not c:
                return
            if isinstance(ranking, Exception):
                QMessageBox.warning(
                    self, "Rank failed",
                    f"{type(ranking).__name__}: {ranking}")
                return
            self._fill_keep_table(ranking, checked=checked)
            self.statusBar().showMessage("ranked", 3000)

        band = self.band_slider.values()

        def compute():
            try:
                return c.rank_symbols(inp, out, fmin=band[0], fmax=band[1])
            except Exception as exc:
                return exc

        self._run_bg(compute, done)

    @staticmethod
    def _sym_device(name: str) -> str:
        """Owning-device key of a symbol name (gm_I0_MN1 -> I0_MN1)."""
        return name.split("_", 1)[1] if "_" in name else name

    def _fill_keep_table(self, ranking, checked=()):
        self._keep_ranked = True
        self._filling = True
        try:
            values = self.controller.op_values()
        except Exception:
            values = {}
        self._last_ranking = list(ranking)
        rows = view.ranking_rows(ranking, values)
        checked = set(checked)
        grouped = self.group_chk.isChecked()
        if grouped:
            order = {}
            for name, *_ in rows:
                order.setdefault(self._sym_device(name), len(order))
            rows = sorted(rows, key=lambda r: (order[self._sym_device(r[0])],))
        # the list stays in PURE rank order: floating ticked rows to
        # the top misread as importance (field report: CL led the list
        # for Z(net1) and looked like a dominant contributor). The
        # count on the estimate line and the 'ticked only' filter are
        # how carried ticks stay findable.
        from PySide6.QtGui import QBrush, QColor
        tints = (QBrush(), QBrush(QColor(theme.WARN_TINT)))
        self.keep_tbl.setRowCount(len(rows))
        prev_dev, band = None, 0
        for i, (name, opval, score, peak) in enumerate(rows):
            if grouped:
                dev = self._sym_device(name)
                if dev != prev_dev:
                    band, prev_dev = 1 - band, dev
            sym = QTableWidgetItem(name)
            sym.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            sym.setToolTip(f"{name}\n  dcOp: {opval or '—'}\n  score: "
                           f"{score}\n  peaks: {peak or '—'}")
            sym.setCheckState(Qt.Checked if name in checked else Qt.Unchecked)
            ro = Qt.ItemIsEnabled | Qt.ItemIsSelectable
            info = [QTableWidgetItem(opval), QTableWidgetItem(score),
                    QTableWidgetItem(peak)]
            for it in info:
                it.setFlags(ro)                       # non-editable info cells
            al = QTableWidgetItem(
                self.controller.sym_aliases.get(name, "")
                if self.controller else "")
            al.setFlags(ro | Qt.ItemIsEditable)       # only this cell edits
            al.setToolTip("LaTeX for THIS symbol; overrides the device "
                          "alias (e.g. g_{m1}). Blank = default.")
            cells = [sym, *info, al]
            for j, it in enumerate(cells):
                if grouped:
                    it.setBackground(tints[band])
                self.keep_tbl.setItem(i, j, it)
        self._filling = False
        self._apply_keep_filter(self.keep_filter.text())
        self._update_estimate()

    def _edit_device_alias(self, name: str):
        """Context menu on an instance: the device-level LaTeX subscript
        (renames every symbol of the device; per-symbol LaTeX in the
        keep table overrides it)."""
        if self.controller is None:
            return
        from PySide6.QtWidgets import QInputDialog

        current = self.controller.sym_aliases.get(name, "")
        text, ok = QInputDialog.getText(
            self, "LaTeX alias", f"Subscript for {name} (e.g. M_1, "
            f"M_{{in}}; blank = default):", text=current)
        if ok:
            self._apply_device_alias(name, text.strip())

    def _apply_device_alias(self, name: str, text: str):
        if self.controller is None:
            return
        self.controller.set_alias(name, text)
        self._save_aliases()
        self._render_expr()
        try:
            self._schematic_restyle()             # the label is typeset
        except Exception:                         # noqa: BLE001
            pass

    def _on_group_toggled(self, _on):
        if getattr(self, "_last_ranking", None):
            self._fill_keep_table(self._last_ranking,
                                  checked=self.checked_keep())

    def checked_keep(self) -> list[str]:
        out = []
        for i in range(self.keep_tbl.rowCount()):
            it = self.keep_tbl.item(i, 0)
            if it is not None and it.checkState() == Qt.Checked:
                out.append(it.text())
        return out

    def _circuit_key(self) -> dict:
        """Everything the next solve depends on besides the keep ticks:
        the working circuit (reduction), the matches, the models, the
        in/out pair and the bench. Captured with every shown result;
        a difference means the screen no longer describes the circuit."""
        c = self.controller
        summ = c.reduction_summary() if c is not None else None
        try:
            inp, out = self._io()
        except Exception:
            inp = out = None
        return {
            "reduction": ((tuple(summ["nodes"]),
                           tuple(summ.get("removals") or ()))
                          if summ else ()),
            "matches": tuple(sorted(tuple(g) for g in self._match_groups)),
            "cap_model": getattr(self, "cap_model", None),
            "mos_model": getattr(self, "mos_model", None),
            "io": (inp, out),
            "bench": self.mode_combo.currentText(),
            "op": (getattr(c, "op_index", 0) if c is not None else 0),
        }

    _KEY_WORDS = {"reduction": "the working circuit (reduction)",
                  "matches": "the matched pairs", "cap_model": "the cap model",
                  "mos_model": "the MOS model", "io": "the in/out pair",
                  "bench": "the bench", "op": "the operating point"}

    def _refresh_solve_hint(self):
        """The panel is the NEXT solve's input; the plot shows the LAST
        result. When they differ -- keep ticks edited, a reduction
        applied or reverted, matches changed, the in/out pair moved --
        the Solve button turns amber and says what changed, instead of
        letting the screen describe a circuit that no longer exists."""
        r = self.result
        reasons = []
        if r is not None:
            then = getattr(r, "_circuit_key", None)
            if then is not None:
                now = self._circuit_key()
                for k, v in now.items():
                    if then.get(k) != v:
                        reasons.append(self._KEY_WORDS.get(k, k))
        # the keep ticks are the NEXT solve's options, not the circuit's
        # state: at start-up the numeric first light sits under an
        # auto-ticked plan by design, and that must not read as stale.
        # Mentioned in the tooltip, never starred or coloured.
        ticks = ""
        if r is not None and self.mode_combo.currentText() == "Transfer":
            shown = r.keep if isinstance(r.keep, list) else None
            if shown is not None and set(shown) != set(self.checked_keep()):
                ticks = ("the ticked keep set differs from the result on "
                         "screen -- Solve recomputes with the ticks")
        stale = bool(reasons)
        self.solve_btn.setText("Solve *" if stale else "Solve")
        self.solve_btn.setStyleSheet(
            f"QPushButton {{ background: {theme.WARN_TINT}; "
            f"color: {theme.WARN}; "
            f"font-weight: bold; }}" if stale else "")
        tip = ("the result on screen is from a previous circuit state -- "
               "changed since: " + ", ".join(reasons) + " -- Solve recomputes"
               if stale else "")
        self.solve_btn.setToolTip((tip + "\n" if tip and ticks else tip)
                                  + ticks)

    def _on_keep_row_selected(self):
        """A keep row names a symbol; its device goes on the card."""
        rows = {it.row() for it in self.keep_tbl.selectedItems()}
        if len(rows) != 1 or self.controller is None:
            return
        it = self.keep_tbl.item(next(iter(rows)), 0)
        if it is None:
            return
        sym = it.text()
        names = {d.name for d in self.controller.devices}
        dev = sym if sym in names else next(
            (d for d in names if sym.endswith("_" + d.replace(".", "_"))),
            None)
        if dev:
            self._show_op_card(dev)

    def _on_keep_changed(self, item):
        if self._filling or self.controller is None:
            return
        self._update_crumb()
        self._refresh_solve_hint()
        if item.column() == 4:                    # per-symbol LaTeX override
            name = self.keep_tbl.item(item.row(), 0).text()
            self.controller.set_alias(name, item.text().strip())
            self._save_aliases()
            self._render_expr()
            return
        # a checkbox toggled: the count follows, and under 'ticked
        # only' an unticked row leaves the view
        if getattr(self, "ticked_chk", None) is not None \
                and self.ticked_chk.isChecked():
            self._apply_keep_filter()
        else:
            self._refresh_tick_count()
        self._update_estimate()

    #: combo index -> engine.interp.PROBE_BACKEND value (None = auto)
    _BACKENDS = (None, "qq", "bot", "zp", "ratfun")

    def _on_backend_changed(self, idx: int):
        """Expert override of the solver auto-selector. Routed through
        the SESSION -- the estimator and the solve both consult the
        engine-side global, which in engine-process mode lives in the
        child, where a GUI-side module write never lands."""
        if self._filling:
            return
        name = self._BACKENDS[idx] \
            if 0 <= idx < len(self._BACKENDS) else None
        if self.controller is not None:
            self.controller.set_probe_backend(name)
        self._update_estimate()

    def _show_auto_backend(self, resolved) -> None:
        """Name what 'auto' resolves to for the CURRENT keep set. The
        crossover is the whole point of the control: one more symbol can
        flip the dense grid to the sparse path and run faster, and that
        is only actionable if the selector's choice is visible."""
        if self.backend_combo.currentIndex() != 0:
            return                              # an override is in force
        label = "auto"
        if resolved:
            label = f"auto → {str(resolved).split('-')[0]}"
        was, self._filling = self._filling, True
        try:
            self.backend_combo.setItemText(0, label)
        finally:
            self._filling = was

    def _update_estimate(self):
        if self.controller is None:
            return
        if self._t0 is not None:
            # a solve is IN FLIGHT: its progress line is costed against
            # the keep set it started with. Re-costing from the current
            # controls would hand the running solve a foreign estimate
            # (an emptied keep table once turned a 772 s solve into
            # "~0s"). The next launch re-estimates.
            return
        inp, out = self._io()
        keep = self.checked_keep()
        c = self.controller
        # the label costs the response the bench will solve: the Tian
        # hybrid at the probe (4x), the probe-injected Z, or the
        # transfer -- one language, the right subject
        mode = self.mode_combo.currentText()
        probe = (self.probe_combo.currentText()
                 if mode == "Loop gain" else None)
        ztarget = self._z_rank_target() if mode == "Impedance" else None
        seq = getattr(self, "_est_seq", 0) + 1
        self._est_seq = seq
        self.estimate_lbl.setText("estimate: …")

        # the re-cost probes real determinants (~0.3 s on fc) and ran on
        # the GUI thread at every tick change — a stall exactly when the
        # user is clicking. Off-thread, sequenced, stale results dropped.
        def done(est):
            if (seq != self._est_seq or self.controller is not c
                    or self._t0 is not None):
                return
            if isinstance(est, Exception):
                self.estimate_lbl.setStyleSheet("")
                self.estimate_lbl.setText(
                    f"estimate: — ({type(est).__name__})")
                return
            n = len(keep)
            tick_note = f"  —  {n} ticked" if n else ""
            self.estimate_lbl.setText(f"estimate: {est}{tick_note}")
            self._show_auto_backend(getattr(est, "backend", None))
            secs = getattr(est, "seconds", None)
            # an unmodelled path's seconds are an extrapolation the
            # estimator itself refuses to print -- the launch line must
            # not quote them either (field report: '[estimate ~3s]'
            # against a 45 s direct determinant)
            if getattr(est, "path", None) not in getattr(
                    est, "MODELLED", ()):
                secs = None
            # the progress bar shows elapsed against this, so a wrong
            # estimate is visible while you wait, not only afterwards
            self._est_s = secs
            budget = self.budget_spin.value()
            color = theme.GOOD                       # green: within budget
            if secs is None or secs > budget:
                color = theme.BAD                    # red: over / unknown
            elif secs > 0.7 * budget:
                color = theme.WARN                   # amber: close
            self.estimate_lbl.setStyleSheet(f"color: {color};")

        def compute():
            try:
                if probe:
                    return c.estimate_loop(probe, keep)
                if ztarget is not None:
                    try:
                        return c.estimate_z(ztarget, keep)
                    except Exception:     # between-nets: transfer cost
                        pass
                return c.estimate(inp, out, keep)
            except Exception as exc:
                return exc

        self._run_bg(compute, done)

    def _suggest_keep_dialog(self):
        """The menu entry: ask for the solve-time budget (the hidden spin
        holds it), then plan. _suggest_keep itself stays non-modal for
        programmatic callers."""
        from PySide6.QtWidgets import QInputDialog

        if self.controller is None:
            return
        val, ok = QInputDialog.getDouble(
            self, "Suggest keep set", "SOLVE-TIME budget in seconds -- how "
            "long a symbolic solve may take (unrelated to the error "
            "budget, which is accuracy):", self.budget_spin.value(),
            0.1, 600.0, 1)
        if not ok:
            return
        self.budget_spin.setValue(val)
        self._suggest_keep()

    def _suggest_keep(self):
        """Form-aware: for a lowest-order solve the right keeps are the
        LETTERS OF THE STORY — the reactances the pursuit will keep plus
        the conductances that set A0 and the dominant pole — not the
        band-sensitivity/time-budget plan, whose large sets multiply the
        grid while the reduction collapses the coefficients they live in."""
        if self.controller is None:
            return
        inp, out = self._io()
        lowest = "lowest" in self.form_combo.currentText()
        try:
            if lowest:
                fmin, fmax = self.band_slider.values()
                keep = self.controller.suggest_story_keep(
                    inp, out, fmin=fmin, fmax=fmax,
                    tol_db=self.mag_spin.value())
            else:
                keep = list(self.controller.suggest_keep(
                    inp, out, self.budget_spin.value()).keep)
            lo, hi = self.band_slider.values()
            ranking = self.controller.rank_symbols(inp, out,
                                                   fmin=lo, fmax=hi)
        except Exception as exc:
            QMessageBox.warning(self, "Suggest failed", f"{type(exc).__name__}: {exc}")
            return
        self._fill_keep_table(ranking, checked=keep)
        if lowest:
            self.statusBar().showMessage(
                f"story keep for lowest order ({len(keep)}): {keep} — the "
                f"letters of A0 and the dominant pole over your band")
        else:
            self.statusBar().showMessage(
                f"suggested keep-set ({len(keep)} symbols): {keep}")

    # ---------------------------------------------------------------- solve
    #: sentinel: "use the keep-set estimate" (a solve), vs an explicit
    #: value or None for jobs the keep-set estimator does not describe

    def solve(self):
        if self.controller is None:
            return
        # ONE pre-flight for every tool's Solve: an estimate over the
        # time budget asks Continue up front instead of failing later
        # (or, on the unbudgeted paths, silently running long)
        budget = self._confirm_over_budget()
        if budget is False:
            return
        self._insist_budget = budget          # loop gain passes it on
        if self.mode_combo.currentText() == "Compensate":
            self.suggest_comp()
            return
        if self.mode_combo.currentText() == "Modes":
            self.run_modes()
            return
        if self.mode_combo.currentText() == "Impedance":
            spec = getattr(self, "_imp_spec", None)
            if spec is None:
                self.statusBar().showMessage(
                    "no impedance target — right-click a net or a "
                    "two-terminal element (Set as impedance target), "
                    "or click a wire")
                return
            desc = self._imp_desc(spec)
            keep = self.checked_keep()
            kw, phrase = self._impedance_form_kw()
            self._launch(lambda cb: self.controller.impedance_result(
                             spec, keep=keep, progress=cb, **kw),
                         f"impedance {desc}{phrase}"
                         f"{f' [{len(keep)} kept]' if keep else ''} …")
            return
        if self.mode_combo.currentText() == "GFT":
            self.run_gft()
            return
        if self.mode_combo.currentText() == "Reduce circuit":
            self.run_acg_scan()
            return
        if self.mode_combo.currentText() == "Loop gain":
            probe = self.probe_combo.currentText()
            if not probe:
                self.statusBar().showMessage("no loop probe in this design")
                return
            keep = self.checked_keep()
            kw, phrase = self._loopgain_form_kw()
            self._launch(lambda cb: self.controller.loop_gain(
                             probe, keep=keep, progress=cb,
                             max_seconds=self._insist_budget,
                             note=lambda m: self.worker_note(m), **kw),
                         f"loop gain at {probe}{phrase}"
                         f"{f' [{len(keep)} kept]' if keep else ''} …")
            return
        # Transfer: ONE Solve action; the form selector picks the contract
        form = self.form_combo.currentText()
        if form == "Simplified · full order":
            self._launch_simplify()
            return
        if form == "Simplified · lowest order":
            self._launch_reduce()
            return
        inp, out, keep = *self._io(), self.checked_keep()
        self._launch(lambda cb: self.controller.attach_template(
                         self.controller.solve(inp, out, keep, progress=cb)),
                     f"solving {inp} → {out}"
                     f"{f' [{len(keep)} kept]' if keep else ''} …")

    def _loopgain_form_kw(self) -> tuple[dict, str]:
        """The Loop-gain solve's contract from the form selector --
        the same ladder as Transfer and Impedance. The stability
        strategy is the loop's native contract: it gates on the
        margins themselves."""
        form = self.form_combo.currentText()
        if form == "Simplified · full order":
            mag, ph = self.mag_spin.value(), self.phase_spin.value()
            return (dict(form="simplified", mag_db=mag, phase_deg=ph,
                         band_fmin=1e3, band_fmax=1e9),
                    f" (simplified, {mag} dB / {ph}°)")
        if form == "Simplified · lowest order":
            fl, fh = self.band_slider.values()
            strat = self._strategy()
            return (dict(form="lowest", strategy=strat,
                         strategy_opts=self._strategy_opts(),
                         band_fmin=fl, band_fmax=fh),
                    f" (lowest order, {strat})")
        return {}, ""

    def _impedance_form_kw(self) -> tuple[dict, str]:
        """The Impedance solve's contract from the form selector --
        the same ladder the Transfer tool reads."""
        form = self.form_combo.currentText()
        if form == "Simplified · full order":
            mag, ph = self.mag_spin.value(), self.phase_spin.value()
            return (dict(form="simplified", mag_db=mag, phase_deg=ph,
                         fmin=1e3, fmax=1e9),
                    f" (simplified, {mag} dB / {ph}°)")
        if form == "Simplified · lowest order":
            fl, fh = self.band_slider.values()
            strat = self._strategy()
            return (dict(form="lowest", strategy=strat,
                         strategy_opts=self._strategy_opts(),
                         fmin=fl, fmax=fh),
                    f" (lowest order, {strat})")
        return {}, ""

    def _on_form_changed(self, form: str):
        self._refresh_form_ui()

    def _refresh_form_ui(self):
        """Every control appears exactly when it means something (gated on
        the mode too -- the menu shortcuts set the form from any bench):

          Exact                    -- no tolerance, no band, no overlays;
          Simplified, either kind  -- tolerance spins + the tolerance TUBE
                                      around the model trace;
          Simplified lowest order  -- additionally the band slider, its
                                      span highlighted: the certification
                                      region of the reduced model.
        """
        mode = self.mode_combo.currentText()
        formmode = mode in ("Transfer", "Impedance", "Loop gain", "GFT")
        form = self.form_combo.currentText()
        budgeted = formmode and form.startswith("Simplified")
        lowest = budgeted and "lowest" in form
        strat = self._strategy()
        # the CONTRACT surface (strategy dropdown, its budget spins, the
        # band slider) shows wherever the contract governs something on
        # screen: the lowest-order solve, and the Reduce-circuit tool --
        # its AC-ground and removal scans price and gate under exactly
        # these controls, and a criterion set by invisible knobs would
        # be the dead-knob bug inverted (field report: the tool showed
        # neither the band nor the strategy it was gating with).
        contract = lowest or mode == "Reduce circuit"
        # full order: dB/deg collapse budgets. Under the contract, the
        # plain strategy reuses the same dB/deg pair as its budgets.
        show_dbdeg = ((budgeted and not lowest)
                      or (contract and strat == "plain"))
        self._budget_lbl_act.setVisible(show_dbdeg)
        self._mag_act.setVisible(show_dbdeg)
        self._phase_act.setVisible(show_dbdeg)
        self._strategy_act.setVisible(contract)
        self._pm_act.setVisible(contract and strat == "stability")
        self._gm_act.setVisible(contract and strat == "stability")
        self._rej_act.setVisible(contract and strat == "rejection")
        self.band_row.setVisible(contract)
        self._on_band_changed(*self.band_slider.values())

    def _strategy(self) -> str:
        return {"Gain & phase": "plain",
                "Stability (margins)": "stability",
                "Rejection (dB)": "rejection"}.get(
            self.strategy_combo.currentText(), "plain")

    def _strategy_opts(self) -> dict:
        s = self._strategy()
        if s == "stability":
            return {"pm_deg": self.strat_pm_spin.value(),
                    "gm_db": self.strat_gm_spin.value()}
        if s == "rejection":
            return {"rej_db": self.strat_rej_spin.value()}
        return {"gain_db": self.mag_spin.value(),
                "phase_deg": self.phase_spin.value()}

    def _strategy_eps_eq(self) -> float:
        """The strategy budget as a comparable relative tolerance, for
        the order-certificate label — the criterion's own mapping, not
        a GUI-side copy of it."""
        from ..analysis.criteria import make_criterion

        return make_criterion(strategy=self._strategy(),
                              strategy_opts=self._strategy_opts()
                              ).eps_equivalent()

    def _on_strategy_changed(self):
        self._refresh_form_ui()
        self._cert_timer.start()

    def _refresh_certificate(self):
        """The band's demand at the set tolerance, printed beside the
        slider while dragging: 'lowpass band needs order 2 at 5%', plus
        the doublet caveat. Computed OFF the GUI thread — the first
        call costs a numeric sweep (~1 s) that used to stall the drag."""
        if (self.controller is None
                or not self.band_row.isVisibleTo(self)
                or self.mode_combo.currentText() == "Reduce circuit"):
            # the order verdict belongs to the lowest-order solve; the
            # Reduce bench shows the band for its own scans
            self.cert_lbl.setText("")
            self.cert_lbl.setVisible(False)
            return
        inp, out = self._io()
        fmin, fmax = self.band_slider.values()
        eps_eq = self._strategy_eps_eq()
        c = self.controller
        seq = getattr(self, "_cert_seq", 0) + 1
        self._cert_seq = seq

        def done(cert):
            if seq != self._cert_seq or self.controller is not c:
                return                      # superseded by a newer drag
            try:
                self.cert_lbl.setText(cert.describe(eps_eq))
                self.cert_lbl.setVisible(True)
            except Exception:
                self.cert_lbl.setText("")
                self.cert_lbl.setVisible(False)

        self._run_bg(lambda: c.order_certificate(inp, out, fmin, fmax),
                     done)

    def _update_tol_bands(self):
        """The tolerance made visible, per strategy. Gain & phase: a
        ±gain-dB tube on the magnitude axis and a ±phase-deg tube on the
        phase axis, over the whole band. Rejection: the magnitude tube
        only. Stability: a ±PM-deg phase tube hugging the unity
        crossover — the only place the criterion looks."""
        for artist in self._tol_bands:
            try:
                artist.remove()
            except Exception:
                pass
        self._tol_bands = []
        form = self.form_combo.currentText()
        budgeted = (self.mode_combo.currentText() == "Transfer"
                    and form.startswith("Simplified"))
        r = self.result
        if self.mode_combo.currentText() == "Reduce circuit":
            # the comparison plot: the tube hugs the as-imported
            # response, the ticked reduction must stay inside it
            r = getattr(self, "_reduce_plot_base", None)
            budgeted = r is not None
            form = "Simplified · lowest order"
        axes = self.canvas.figure.axes
        if not budgeted or r is None or not axes:
            self.canvas.draw_idle()
            return
        import numpy as np

        f = np.asarray(r.freqs, dtype=float)
        h = np.asarray(r.h)
        with np.errstate(divide="ignore", invalid="ignore"):
            mag = 20 * np.log10(np.abs(h))
            ph = np.degrees(np.unwrap(np.angle(h)))
        if "lowest" in form:
            # the tube renders THE STRATEGY'S OWN promise: plain shades
            # both budgets over the band; stability shades only the
            # phase around the unity crossing (that is all it checks);
            # rejection shades the magnitude alone.
            strat = self._strategy()
            fmin, fmax = self.band_slider.values()
            mask = (f >= fmin) & (f <= fmax)
            if not mask.any():
                self.canvas.draw_idle()
                return
            if strat == "plain":
                gdb = self.mag_spin.value()
                pdeg = self.phase_spin.value()
                self._tol_bands.append(axes[0].fill_between(
                    f[mask], (mag - gdb)[mask], (mag + gdb)[mask],
                    color=theme.TOL_BAND, alpha=0.18, zorder=0))
                if len(axes) > 1:
                    self._tol_bands.append(axes[1].fill_between(
                        f[mask], (ph - pdeg)[mask], (ph + pdeg)[mask],
                        color=theme.TOL_BAND, alpha=0.18, zorder=0))
            elif strat == "rejection":
                rej = self.strat_rej_spin.value()
                self._tol_bands.append(axes[0].fill_between(
                    f[mask], (mag - rej)[mask], (mag + rej)[mask],
                    color=theme.TOL_BAND, alpha=0.18, zorder=0))
            else:                                # stability
                absh = np.abs(h)
                cross = np.where(np.diff(np.sign(
                    20 * np.log10(np.maximum(absh, 1e-300))))
                    != 0)[0]
                if cross.size:
                    fc0 = float(f[cross[0]])
                    near = mask & (f >= fc0 / 2.5) & (f <= fc0 * 2.5)
                    pmd = self.strat_pm_spin.value()
                    self._tol_bands.append(axes[0].plot(
                        [fc0, fc0], [mag[mask].min(), mag[mask].max()],
                        color="0.35", lw=0.8, ls=":", zorder=1)[0])
                    if len(axes) > 1 and near.any():
                        self._tol_bands.append(axes[1].fill_between(
                            f[near], (ph - pmd)[near], (ph + pmd)[near],
                            color=theme.TOL_BAND, alpha=0.18, zorder=0))
            self.canvas.draw_idle()
            return
        m = self.mag_spin.value()
        self._tol_bands.append(axes[0].fill_between(
            f, mag - m, mag + m,
            color=theme.TOL_BAND, alpha=0.18, zorder=0))
        if len(axes) > 1:
            pd = self.phase_spin.value()
            self._tol_bands.append(axes[1].fill_between(
                f, ph - pd, ph + pd,
                color=theme.TOL_BAND, alpha=0.18, zorder=0))
        self.canvas.draw_idle()

    def _align_error_axes(self):
        """Put the Error plots' frequency axis at the same WINDOW x range
        as the Bode's. Both figures run tight_layout, but the residual's
        y labels are wider, so equal fractions are not equal pixels --
        and the two canvases have different widths anyway. Map through
        window coordinates instead, and only move when it actually
        differs so the draw loop converges."""
        from PySide6.QtCore import QPoint

        mains = self.canvas.figure.axes
        errs = self.err_canvas.figure.axes
        if not mains or not errs or self.err_canvas.width() < 50:
            return
        mpos = mains[0].get_position()
        mx0 = self.canvas.mapTo(self, QPoint(0, 0)).x()
        mw = self.canvas.width()
        left_px = mx0 + mpos.x0 * mw
        right_px = mx0 + mpos.x1 * mw
        ex0 = self.err_canvas.mapTo(self, QPoint(0, 0)).x()
        ew = max(1, self.err_canvas.width())
        left = (left_px - ex0) / ew
        right = (right_px - ex0) / ew
        if not (0.0 <= left < right <= 1.0):
            return                              # off-canvas: leave it be
        cur = errs[0].get_position()
        if abs(cur.x0 - left) < 0.002 and abs(cur.x1 - right) < 0.002:
            return                              # already aligned
        for ax in errs:
            p = ax.get_position()
            ax.set_position([left, p.y0, right - left, p.height])
        self.err_canvas.draw_idle()

    def _sync_band_row(self):
        """Keep the slider groove visually aligned with the Bode's
        frequency axis: on every canvas draw, the row's margins are set
        so the groove's active span matches the axes' x-extent."""
        axes = self.canvas.figure.axes
        if not axes:
            return
        from .rangeslider import HANDLE_R

        pos = axes[0].get_position()
        w = self.canvas.width()
        left = max(0, round(pos.x0 * w) - HANDLE_R)
        right = max(0, round((1.0 - pos.x1) * w) - HANDLE_R)
        lay = self.band_row.layout()
        m = lay.contentsMargins()
        if (left, right) != (m.left(), m.right()):
            lay.setContentsMargins(left, m.top(), right, m.bottom())
        # the legend's measured anchor also drifts with canvas height
        view.refresh_legend(self.canvas.figure)
        self._align_error_axes()      # the residual tracks the Bode

    def _on_band_changed(self, fmin: float, fmax: float):
        """Mirror the chosen band onto every axis of the Bode — the
        highlighted span IS the certification region, drawn while the
        user drags so the contract is visible before the solve."""
        for artist in self._band_spans:
            try:
                artist.remove()
            except Exception:
                pass
        self._band_spans = []
        if not self.band_row.isVisibleTo(self):
            # no band highlight -- but the tolerance tube has its own
            # visibility rules (any Simplified form), so hand over
            self._update_tol_bands()
            return
        for ax in self.canvas.figure.axes:
            self._band_spans.append(
                ax.axvspan(fmin, fmax, color=theme.BAND_SPAN, alpha=0.10,
                           zorder=0))
        self._update_tol_bands()
        self._cert_timer.start()          # the band's demand, debounced
        self._rank_band_timer.start()     # ...and the band's ranking

    def simplify(self):
        """Menu/shortcut entry: select the form so the toolbar agrees, then
        run it."""
        self.form_combo.setCurrentText("Simplified · full order")
        self._launch_simplify()

    def _launch_simplify(self):
        if self.controller is None:
            return
        inp, out, keep = *self._io(), self.checked_keep()
        mag, ph = self.mag_spin.value(), self.phase_spin.value()
        # full order keeps every pole, so it is certified over the wide
        # default band -- the slider belongs to the lowest-order form
        fmin, fmax = 1e3, 1e9
        self._launch(
            lambda cb: self.controller.attach_template(
                self.controller.simplify(inp, out, keep, mag_db=mag,
                                         phase_deg=ph, fmin=fmin,
                                         fmax=fmax, progress=cb)),
            f"simplifying {inp} → {out} within {mag} dB / {ph}° over "
            f"{view.eng(fmin, 'Hz')}–{view.eng(fmax, 'Hz')} …")

    def reduce(self):
        """Menu/shortcut entry: select the form so the toolbar agrees, then
        run it."""
        self.form_combo.setCurrentText("Simplified · lowest order")
        self._launch_reduce()

    def _launch_reduce(self):
        if self.controller is None:
            return
        inp, out, keep = *self._io(), self.checked_keep()
        strat, opts = self._strategy(), self._strategy_opts()
        fmin, fmax = self.band_slider.values()
        knobs = ", ".join(f"{k.split('_')[0]} {v:g}"
                          for k, v in opts.items())
        self._launch(
            lambda cb: self.controller.attach_template(
                self.controller.reduce_solve(
                    inp, out, keep, strategy=strat, strategy_opts=opts,
                    fmin=fmin, fmax=fmax,
                    progress=cb, note=lambda m: self.worker_note(m))),
            f"reducing {inp} → {out} ({strat}: {knobs}) over "
            f"{view.eng(fmin, 'Hz')}–{view.eng(fmax, 'Hz')} …",
            growth_reason=("the pursuit accepted a reactance and re-tries "
                           "the remaining candidates — see the round "
                           "notes above"))

    def solve_sync(self):
        inp, out = self._io()
        self._show(self.controller.attach_template(
            self.controller.solve(inp, out, self.checked_keep())))
        return self.result

    def impedance_sync(self, port, **kw):
        self._show(self.controller.impedance_result(
            port, keep=self.checked_keep(), **kw))
        return self.result

    def loop_gain_sync(self, probe=None, **kw):
        probe = probe or self.probe_combo.currentText()
        self._show(self.controller.loop_gain(probe,
                                             keep=self.checked_keep(),
                                             **kw))
        return self.result

    def simplify_sync(self):
        inp, out = self._io()
        fmin, fmax = 1e3, 1e9                 # full order: wide default
        self._show(self.controller.attach_template(self.controller.simplify(
            inp, out, self.checked_keep(),
            mag_db=self.mag_spin.value(), phase_deg=self.phase_spin.value(),
            fmin=fmin, fmax=fmax)))
        return self.result

    def reduce_sync(self):
        inp, out = self._io()
        fmin, fmax = self.band_slider.values()
        self._show(self.controller.attach_template(self.controller.reduce_solve(
            inp, out, self.checked_keep(), strategy=self._strategy(),
            strategy_opts=self._strategy_opts(), fmin=fmin, fmax=fmax)))
        return self.result

    #: keep the log bounded -- it is a session record, not a data store
    _LOG_MAX_LINES = 2000

    def _show_result_tab(self):
        """A finished computation surfaces its result tab (the Log held
        the stage while it ran); a tool whose result tab is hidden
        (Reduce) stays on the Log, which carries its findings."""
        w = {"Compensate": self._comp_tab,
             "GFT": self._gft_tab}.get(self.mode_combo.currentText(),
                                       self._expr_tab)
        idx = self.tabs.indexOf(w)
        if idx >= 0 and self.tabs.isTabVisible(idx):
            self.tabs.setCurrentWidget(w)

    def _on_done(self, result):
        # the closing DONE line is logged centrally by _launch's
        # done wrapper before any handler runs
        self._show_result_tab()
        self._learn_from_solve()
        self._tick.stop()
        self._t0 = None
        self.progress.hide()
        self.cancel_btn.hide()
        self._show(result)
        self._update_crumb()
        if result.out.startswith("T@") and self.controller is not None:
            self._start_advisor(result.inp)
        for b in (self.solve_btn,
                  self.a_solve, self.a_simplify, self.a_reduce):
            b.setEnabled(True)
        self.statusBar().showMessage(self._result_status(result))

    def _result_status(self, result) -> str:
        """One line stating the CONSEQUENCE of the chosen form — the order
        change is the fact that makes full-order vs lowest-order click, so
        it is said on every budgeted solve, not buried in a tab."""
        base = (f"{result.inp} → {result.out}:  "
                f"{result.dc_gain_db:.2f} dB, {result.n_terms} terms")
        if result.simplified:
            n_poles = len(result.poles_hz)
            pole_s = "pole" if n_poles == 1 else "poles"
            err = (f"{result.mag_err_db:.2g} dB"
                   if result.mag_err_db is not None else "?")
            if getattr(result, "reduced_order", False):
                base += (f" — LOWERED to {n_poles} {pole_s}, within {err} "
                         f"over the band")
            else:
                base += f" — all {n_poles} {pole_s} kept, within {err}"
        return base + self._backend_note()

    def _backend_note(self) -> str:
        """Which solver actually ran (S-D telemetry). The auto-selector
        silently routes large solves to the sparse backend; saying so keeps
        that visible instead of magic, and names a fallback when one
        happened."""
        try:
            tl = (self.controller.last_solve()
                  if self.controller is not None else None)
        except Exception:                        # pragma: no cover
            return ""
        if not tl:
            return ""
        note = f"  [{tl.get('backend', '?')}"
        wall = tl.get("wall_s")
        if wall:
            note += f", {wall:.1f} s"
        if tl.get("fell_back"):
            # a fallback means the whole grid ran TWICE -- the reason is
            # the difference between "tune the selector" and "this
            # circuit defeats the sparse path", so it must survive
            note += ", fell back: " + (tl.get("fallback_reason")
                                       or "reason not recorded")
        return note + "]"

    def _confirm_over_budget(self):
        """Pre-flight for a budgeted launch: when the estimate already
        says the solve will blow the time budget, ASK -- a dialog with
        Continue beats a refusal that sends the user hunting for the
        budget spin (field request). Returns the max_seconds to pass:
        the budget (within it), None (the user insisted -- no engine
        refusal either), or False (do not run)."""
        budget = self.budget_spin.value()
        est = self._est_s
        if not (est and budget and est > budget):
            return budget
        # the budget "spin" is hidden state (retired from the keep
        # pane); pointing the user at it was a lie -- the dialog
        # itself offers the raise instead
        import math
        new = min(600.0, math.ceil(est * 1.2 / 10.0) * 10.0)
        box = QMessageBox(self)
        box.setWindowTitle("Over the time budget")
        box.setIcon(QMessageBox.Warning)
        box.setText(f"This solve is estimated at ~{est:.0f} s — over "
                    f"the {budget:.0f} s solve-time budget.")
        once = box.addButton("Continue this once", QMessageBox.AcceptRole)
        raise_b = box.addButton(f"Raise budget to {new:.0f} s",
                                QMessageBox.AcceptRole)
        box.addButton(QMessageBox.Cancel)
        box.setDefaultButton(once)
        box.exec()
        if box.clickedButton() is once:
            self.log(f"over-budget solve confirmed by the user "
                     f"(~{est:.0f}s vs {budget:.0f}s budget)")
            return None
        if box.clickedButton() is raise_b:
            self.budget_spin.setValue(new)    # persisted with the rest
            self.log(f"solve-time budget raised to {new:.0f}s "
                     f"(estimate ~{est:.0f}s)")
            return new
        self.statusBar().showMessage("solve not started (over the "
                                     "time budget)")
        return False

    def _on_failed(self, msg):
        self._tick.stop()
        self._t0 = None
        self.progress.hide()
        self.cancel_btn.hide()
        for b in (self.solve_btn,
                  self.a_solve, self.a_simplify, self.a_reduce):
            b.setEnabled(True)
        self._log_finish("FAILED")
        self.tabs.setCurrentWidget(self.logview)  # the error is HERE
        self.statusBar().showMessage("solve failed")
        # the refusal's remedies are written for the API; the GUI user
        # holds a SPIN, not a kwarg (field report: the honest estimate
        # crossed the 100 s default for the first time and the strip
        # said "pass max_seconds=None")
        if "SolveTooLarge" in msg:
            msg = msg.replace(
                "pass max_seconds=None to insist",
                f"raise the time budget (now "
                f"{self.budget_spin.value():.0f} s, under Keep "
                f"symbolic) to insist")
            self._set_strip("solve refused: " + msg, "warn")
        else:
            self._set_strip("solve failed: " + msg, "error")
        if msg.startswith("EngineDied"):
            self._engine_died(msg)

    def _engine_died(self, detail: str) -> None:
        """The engine process died UNINVITED (a crash, a kill from
        outside -- not a cancel). Crash isolation is the design's
        promise: the window survives, the shown results survive, and
        one dialog offers the same respawn-and-replay the hard cancel
        uses (plan E-H)."""
        if getattr(self, "_engine_down_handled", False):
            return
        if not hasattr(self.controller, "engine_restart"):
            return
        self._engine_down_handled = True
        self.log(f"the engine process died: {detail}")
        restart = QMessageBox.question(
            self, "Engine died",
            "The analysis engine process died unexpectedly.\n\n"
            "Your shown results are intact. Restart the engine and "
            "replay the session state (matches, reductions, aliases)?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes) == QMessageBox.Yes
        if not restart:
            self._set_strip(
                "engine down — results remain readable; re-open the "
                "session to compute again", "warn")
            return
        try:
            self.controller.engine_restart()
            self._engine_down_handled = False
            self.log("engine restarted; session state replayed")
            self.statusBar().showMessage("engine restarted")
            self._clear_strip()
        except Exception as exc:                  # noqa: BLE001
            self._set_strip(f"engine restart failed: {exc} — re-open "
                            f"the session", "error")

    def _show(self, result, overlays=(), push_history=True):
        self.result = result
        self._ref_on_screen = False
        card = getattr(self, "op_card", None)
        if card is not None and card.name:
            self._show_op_card(card.name)
        try:
            object.__setattr__(result, "_circuit_key", self._circuit_key())
        except Exception:                         # noqa: BLE001
            pass
        try:
            self._schematic_restyle()
        except Exception:                         # noqa: BLE001
            pass                                  # decoration only
        view.bode_figure(result, self.canvas.figure, overlays=overlays)
        theme.style_figure(self.canvas.figure)
        self.canvas.draw_idle()
        if not self._showing_from_history and push_history:
            self._push_history(result)
            if result.warnings:
                self._set_strip("⚠ "
                                + "   ".join(result.warnings), "warn")
            elif not result.out.startswith("T@"):
                self._clear_strip()
        # the slider covers exactly what the data covers: its range
        # follows the result's frequency grid, which is the simulator's
        # own AC sweep whenever a reference rode along
        import numpy as np

        f = np.asarray(result.freqs, dtype=float)
        if f.size and f[0] > 0:
            self.band_slider.setRange(float(f.min()), float(f.max()))
        # the plot rebuild cleared the overlay artists; re-apply both
        self._band_spans = []
        self._tol_bands = []
        self._on_band_changed(*self.band_slider.values())
        self._refresh_solve_hint()
        if not self._showing_from_history:
            self._autosave_state()      # every shown result checkpoints
        self.summary.setPlainText(self._summary_with_extras(result))
        self._refresh_ledger(result)
        try:
            view.error_figure(result, self.err_canvas.figure)
            theme.style_figure(self.err_canvas.figure)
            self._align_error_axes()
            self.err_canvas.draw_idle()
        except Exception:
            self.err_canvas.figure.clear()
            self.err_canvas.draw_idle()
        self._render_expr()
        self._rebuild_whatif(result)
        for b in (self.a_export, self.a_copy_tex,
                  self.a_add_report, self.a_export_csv):
            b.setEnabled(True)

    def _refresh_ledger(self, result):
        """Append the approximation ledger to the Summary, computed off
        the GUI thread: every approximation between the imported circuit
        and the shown response, priced under the toolbar contract, with
        the MEASURED totals (never summed). Advisory -- failures drop
        silently like every advisor, and a stale result is discarded."""
        if (self.controller is None or result is None
                or not result.inp or result.out.startswith("T@")):
            return
        kw = self._contract_kw()

        def compute():
            return self.controller.approximation_report(
                result.inp, result.out, result=result, **kw)

        def done(rep):
            if self.result is not result or isinstance(rep, Exception):
                return
            eqs = (self.controller.equivalent_elements()
                   if self.controller is not None else [])
            self.summary.setPlainText(
                self._summary_with_extras(result, rep, eqs))

        self._run_bg(compute, done)

    def _summary_with_extras(self, result, ledger=None, eqs=()) -> str:
        """EVERY summary (re)render goes through here, so what attaches
        to the shown result -- the cost ledger, cached numeral stories
        -- survives any later async re-render instead of being
        clobbered by whichever advisory lands last (field case: the
        approximation report landed after Explain's stories and wiped
        them; the engine process stretched the window)."""
        txt = view.summary_text(result)
        if ledger is not None:
            txt += "\n\n" + self._ledger_text(ledger, eqs)
        try:
            keep = result.keep if isinstance(result.keep, list) else []
            stories = (self.controller.cached_numerals(
                result.inp, result.out, keep=keep)
                if self.controller is not None else None)
        except Exception:                         # noqa: BLE001
            stories = None
        if stories:
            txt += "\n\n" + self._stories_text(stories)
        return txt

    @staticmethod
    def _ledger_text(rep, eqs=()) -> str:
        """One contract, one unit, one sentence shape -- the totals are
        measured end to end, and an over-budget total says so even when
        every step individually passed."""
        lo, hi = rep["band"]
        lines = [f"approximation ledger — contract: {rep['criterion']}, "
                 f"band {view.eng(lo, 'Hz')}–{view.eng(hi, 'Hz')} "
                 f"(≤ 1.00× budget means within it)"]
        for e in rep["entries"]:
            if e["exact"]:
                lines.append(f"  {e['step']} — exact, no budget spent")
            else:
                lines.append(f"  {e['step']}: {e['score']:.2f}× budget")
        if not rep["entries"]:
            lines.append("  no circuit-level approximations — the "
                         "working circuit IS the imported one")
        lines.append(f"  working circuit vs imported, measured end to "
                     f"end: {rep['circuit_score']:.2f}× budget")
        if rep.get("grand_score") is not None:
            sv = rep.get("solve_score")
            solve = (f" (the solve's own score: {sv:.2f}×)"
                     if sv is not None else "")
            lines.append(f"  SHOWN RESULT vs imported circuit, measured: "
                         f"{rep['grand_score']:.2f}× budget{solve}")
        worst = rep.get("grand_score", rep["circuit_score"])
        if worst is not None and worst > 1.0:
            lines.append("  OVER BUDGET: the composition exceeds the "
                         "contract even though each step may have "
                         "passed alone — revert a step or relax the "
                         "budget")
        # the lumped equivalents, spelled out: a Geq_net8 in H(s) is
        # a definition the reader must be able to look up
        if eqs:
            lines.append("equivalent elements (exact parallel lumps at "
                         "AC-grounded nodes):")
            unit = {"c": "F", "g": "S", "r": "Ω"}
            for e in eqs:
                val = (view.eng(e["value"], unit.get(e["kind"], ""))
                       if e["value"] is not None else "?")
                lines.append(f"  {e['name']} = "
                             + " + ".join(e["members"]) + f" = {val}")
        return chr(10).join(lines)

    def _render_expr(self):
        """(Re)draw the Expression tab for the current result, honouring the
        Full-names toggle. Split from _show so the checkbox can re-render without
        re-solving."""
        if self.result is None:
            return
        base = not self.fullnames_chk.isChecked()
        aliases = self.controller.sym_aliases if self.controller else None
        # the lumped equivalents: defined below the expressions, or
        # expanded into them when asked
        defs = (self.controller.equivalent_definitions()
                if self.controller is not None else [])
        # only the equivalents that actually appear in the shown H(s):
        # a reduction lumps many nodes, the formula mentions a few
        try:
            present = {str(s) for s in self.result.tf.expr.free_symbols}
            defs = [d for d in defs if d[0] in present]
        except Exception:                         # noqa: BLE001
            pass
        self.expand_eq_chk.setEnabled(bool(defs))
        shown = self.result
        if defs and self.expand_eq_chk.isChecked():
            try:
                shown = self.controller.expand_equivalents(self.result)
            except Exception as exc:                  # noqa: BLE001
                self.log(f"expand equivalents failed: {exc}")
                shown = self.result
            defs = []
        if self.exprweb is not None:                      # KaTeX web view
            try:
                numerals = {}
                if self.controller is not None:
                    rkeep = (self.result.keep
                             if isinstance(self.result.keep, list) else ())
                    stories = self.controller.cached_numerals(
                        self.result.inp, self.result.out, keep=rkeep)
                    deep = self.controller.cached_per_numeral(
                        self.result.inp, self.result.out, keep=rkeep)
                    if stories or deep:
                        numerals = view.numeral_tips(stories or [],
                                                     deep=deep)
                payload = view.expr_katex(
                    shown, base=base, aliases=aliases,
                    numerals=numerals,
                    numhint="collapsed operating-point products — run "
                            "<b>Analysis → Explain the numbers</b> for the "
                            "ranked contributors",
                    equivalents=defs)
                # a lumped equivalent says what it stands for on hover
                if self.controller is not None:
                    for e in self.controller.equivalent_elements():
                        n = e["name"]
                        if n in payload["values"]:
                            payload["values"][n] = (
                                f"{payload['values'][n]} — exact parallel "
                                f"lump at {e['node']}: "
                                + " + ".join(e["members"]))
                self.exprweb.set_payload(payload)
            except Exception:
                pass
            return
        try:                                              # matplotlib fallback
            n = len(view._expr_lines(shown, base=base))
            # size the canvas to the content and let the scroll area handle the
            # overflow — rather than squeezing N lines into a fixed short panel
            self.expr_canvas.setMinimumHeight(int(34 * n + 24))
            view.expr_figure(shown, self.expr_canvas.figure,
                             base=base, aliases=aliases)
            theme.style_figure(self.expr_canvas.figure)
            self.expr_canvas.draw()                       # parse mathtext now
        except Exception:
            self.expr_canvas.figure.clear()
            self.expr_canvas.draw_idle()

    def _select_keep_symbol(self, name: str):
        """A click on an expression symbol lands on its keep-table row -- the
        \\htmlData tag carries the raw join-key name, which is exactly the
        table's identifier -- and cross-probes to the schematic."""
        for r in range(self.keep_tbl.rowCount()):
            it = self.keep_tbl.item(r, 0)
            if it is not None and it.text() == name:
                self.keep_tbl.selectRow(r)
                self.keep_tbl.scrollToItem(it)
                break

    def apply_theme(self, name: str):
        """Switch the LIVE theme: swap the token palettes (chrome and
        figure data colors), re-apply the Qt palette and stylesheet,
        and repaint the surfaces holding baked color. The choice
        persists in QSettings (ui/theme) and seeds the next launch."""
        from PySide6.QtWidgets import QApplication

        if theme.current == name:
            return
        theme.set_theme(name)
        view.set_fig_theme(name)
        app = QApplication.instance()
        if app is not None:
            theme.apply(app)
        # figures bake their colors into artists: re-plot what is on
        # screen, restyle the rest for their next draw
        if self.result is not None:
            self._show(self.result, push_history=False)
        elif getattr(self, "_ref_on_screen", False):
            self._show_loop_reference()
        elif getattr(self, "_first_light_result", None) is not None \
                and self.mode_combo.currentText() == "Transfer":
            self._show(self._first_light_result, push_history=False)
        else:
            theme.style_figure(self.canvas.figure)
            self.canvas.draw_idle()
        theme.style_figure(self.err_canvas.figure)
        self.err_canvas.draw_idle()
        if getattr(self, "expr_canvas", None) is not None:
            theme.style_figure(self.expr_canvas.figure)
            self.expr_canvas.draw_idle()
        # the trees' proxy style reads its colors at paint time and
        # follows the switch on its own; a repaint suffices
        for tree in (getattr(self, "devices", None),
                     getattr(self, "nets_tree", None)):
            if tree is not None:
                tree.viewport().update()
                QWidget.update(tree)      # the frame; the item-view
                # overload shadows the argless QWidget one
        # the schematic header's icons re-ink to the new theme's text
        if hasattr(self, "_sch_fit_btn"):
            self._sch_refresh_icons()
        if hasattr(self, "_max_top_btn"):
            self._sync_max_icons()        # the pane expand/restore pair
        # the web shells swap their CSS tokens; the summary re-sets
        # its kept text through the theme-aware stylesheet
        if getattr(self, "exprweb", None) is not None:
            self.exprweb.set_theme(name)
        eq = getattr(self, "_ml_eq", None)
        if eq is not None and hasattr(eq, "set_theme"):
            eq.set_theme(name)
        self.summary.setPlainText(self.summary.toPlainText())
        # the OP card re-renders its rich text with the new tokens
        card = getattr(self, "op_card", None)
        if card is not None and card.name:
            self._show_op_card(card.name)
        # the schematic re-inks (its render key carries the theme)
        try:
            self._schematic_render()
        except Exception:                     # noqa: BLE001
            pass                              # decoration only
        # surfaces that BAKED the old tokens into rich text or
        # brushes: the flow crumb, the message strip, the match-tint
        # rows of the Instances tree (field report: all three kept
        # their light colors on a dark ground)
        self._update_crumb()
        if self.msg_strip.isVisible() and self.msg_strip.text():
            self._set_strip(self.msg_strip.text(),
                            getattr(self, "_strip_sev", "info"))
        if getattr(self, "_match_groups", None):
            try:
                self.devices._set_groups_decorate(self._match_groups)
            except Exception:                 # noqa: BLE001
                pass                          # decoration only
        if self.a_dark.isChecked() != (name == "dark"):
            self.a_dark.blockSignals(True)
            self.a_dark.setChecked(name == "dark")
            self.a_dark.blockSignals(False)
        s = self._settings()
        s.setValue("ui/theme", name)
        s.sync()

    def show_manual(self):
        """The user guide, shipped inside the package (gui/assets) and
        rendered by a QTextBrowser -- self-contained, anchors navigable,
        no browser dependency. One instance, raised on repeat calls."""
        dlg = getattr(self, "_manual_dlg", None)
        if dlg is not None:
            dlg.show()
            dlg.raise_()
            return
        from PySide6.QtWidgets import QDialog, QTextBrowser

        path = Path(__file__).resolve().parent / "assets" / "manual.html"
        dlg = QDialog(self)
        dlg.setWindowTitle("CircuitInsight — user guide")
        browser = QTextBrowser(dlg)
        browser.setOpenExternalLinks(False)
        browser.setHtml(path.read_text(encoding="utf-8"))
        lay = QVBoxLayout(dlg)
        lay.addWidget(browser)
        dlg.resize(760, 640)
        self._manual_dlg = dlg
        self._manual_browser = browser           # tests reach it here
        dlg.show()

    def _show_about(self):
        from importlib.metadata import PackageNotFoundError, version
        try:
            v = version("circuitinsight")
        except PackageNotFoundError:
            v = "development tree"
        QMessageBox.about(
            self, "About CircuitInsight",
            f"<b>CircuitInsight</b> {v}<br>"
            f"Symbolic circuit analysis driven by the simulator's own "
            f"operating point.<br>"
            f"<a href='https://github.com/arcadie-cracan/CircuitInsight'>"
            f"github.com/arcadie-cracan/CircuitInsight</a>")

    def _show_device_op(self, item):
        """Full OP record of the double-clicked device — every parameter the
        simulator reported, not just the gm/gds summary columns. Read-only,
        selectable, so values copy straight into a notebook."""
        self._show_device_op_for(self.devices.device_name(item))

    def _show_device_op_for(self, name):
        if self.controller is None or not name:
            return
        try:
            rec = self.controller.device_op(name)
        except Exception:
            rec = {}
        if not rec:
            self.statusBar().showMessage(f"no OP record for {name}")
            return
        from PySide6.QtWidgets import QDialog

        dlg = QDialog(self)
        dlg.setWindowTitle(f"OP: {name}")
        tbl = QTableWidget(len(rec), 2, dlg)
        tbl.setHorizontalHeaderLabels(["parameter", "value"])
        tbl.horizontalHeader().setStretchLastSection(True)
        tbl.verticalHeader().setVisible(False)
        tbl.setEditTriggers(QAbstractItemView.NoEditTriggers)
        ro = Qt.ItemIsEnabled | Qt.ItemIsSelectable
        for i, key in enumerate(sorted(rec)):
            val = rec[key]
            if key == "region":
                shown = f"{val}  ({view.region_name(val)})"
            else:
                # units by the simulator's OP naming, not symbol naming:
                # vds/vgs/vdsat -> V, ids/isub -> A, gm/gds/gmbs -> S, c* -> F
                unit = {"v": "V", "i": "A", "g": "S", "c": "F",
                        "q": "C", "r": "Ω"}.get(key[:1].lower(), "")
                try:
                    shown = view.eng(float(val), unit)
                except (TypeError, ValueError):
                    shown = str(val)
            for col, txt in ((0, key), (1, shown)):
                it = QTableWidgetItem(txt)
                it.setFlags(ro)
                tbl.setItem(i, col, it)
        lay = QVBoxLayout(dlg)
        lay.addWidget(tbl)
        dlg.resize(360, min(620, 80 + 22 * len(rec)))
        self._op_dialog = dlg                    # tests reach it here
        dlg.show()

    def _on_devices_selected(self):
        """Selecting devices in the tree shows the first one's
        operating point on the card (the schematic follows through
        its own tree hook)."""
        names = self.devices.selected_names()
        self._show_op_card(names[0] if names else None)

    def _show_op_card(self, name):
        """The card for `name` (a device), or cleared for None."""
        card = getattr(self, "op_card", None)
        if card is None:
            return
        if not name or self.controller is None:
            card.clear()
            return
        from .opcard import op_card_rows
        try:
            rows = op_card_rows(self.controller, name, self.result,
                                self._match_groups, self._policy_label())
        except Exception as exc:                  # noqa: BLE001
            rows = [("device", [("card unavailable", str(exc), "")])]
        card.show_sections(name, rows)


def build_window(cin=None, psf=None, probe=None,
                 async_open=False) -> MainWindow:
    """Construct the window, optionally preloaded with a CIN + psf (the entry
    the Virtuoso SKILL launcher targets). `probe` preselects the loop-gain
    bench on that vsource; without it the run's own stb designation is
    discovered from the psf header / netlist. async_open: show the
    window after the cheap populate and run first light + auto-setup as
    narrated background work (the launcher's path; tests default to the
    synchronous open)."""
    win = MainWindow()
    if cin and psf:
        win.open_session(cin, psf, probe=probe, async_open=async_open)
    return win


def main(argv=None):
    """Compatibility shim: the real entry is gui.launch.main, which shows
    the loading banner before this module's heavy imports are paid for.
    Anyone entering here has already imported them."""
    from .launch import main as _launch

    return _launch(argv)


if __name__ == "__main__":
    sys.exit(main())
