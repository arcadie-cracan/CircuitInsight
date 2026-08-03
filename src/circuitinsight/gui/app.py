"""PySide6 desktop front end. A thin window over `SessionController` + `view`:
open a CIN + psf, declare matched pairs, pick input/output, choose a keep-set
(ranked by band sensitivity, gated by a solve-time estimate), solve or
error-budget-simplify (in a worker thread), and export a report.
Requires `circuitinsight[gui]` (PySide6).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

from PySide6.QtCore import QEvent, QSettings, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QCheckBox, QComboBox, QDoubleSpinBox,
    QFileDialog, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMainWindow,
    QMessageBox,
    QPushButton, QProgressBar, QScrollArea, QSizePolicy, QSpinBox,
    QSplitter, QTableWidget,
    QTableWidgetItem, QTabWidget, QTextEdit, QToolBar, QVBoxLayout, QWidget,
)
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg, NavigationToolbar2QT)
from matplotlib.figure import Figure

from ..session import SessionController
from . import devtree, exprweb, summaryweb, theme, view
from .rangeslider import RangeSlider


class _Cancelled(BaseException):
    """Raised inside the worker's progress callback to abandon a solve.

    A BaseException on purpose, like KeyboardInterrupt: the engine's
    backend machinery wraps its fast paths in `except Exception` to fall
    back to slower ones, and a cancel that subclasses Exception was
    CAUGHT there -- the user's cancel turned into a silent serial re-run
    of the very solve they abandoned."""


class _Worker(QThread):
    """Run any Result-returning callable off the UI thread.

    `fn` is handed a progress callback; it is invoked in THIS thread, so it only
    emits a signal -- Qt queues it to the GUI thread. Touching widgets from here
    would be a crash waiting for a slow solve.

    cancel() cooperates through that same callback: the flag is checked on
    every grid-point report, so cancellation lands within one grid point on
    the interpolation path. A direct-determinant solve reports no progress
    and therefore cannot be interrupted -- the button stays honest by
    switching to "cancelling..." until the solver next yields.
    """
    done = Signal(object)
    failed = Signal(str)
    cancelled = Signal()
    progress = Signal(int, int)          # (done, total) grid points
    note = Signal(str)                   # worker-side narration -> Log

    def __init__(self, fn):
        super().__init__()
        self._fn = fn
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        def cb(done, total):
            if self._cancel:
                raise _Cancelled
            self.progress.emit(done, total)

        try:
            self.done.emit(self._fn(cb))
        except _Cancelled:
            self.cancelled.emit()
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")


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


class MainWindow(QMainWindow):
    #: tests point this at a temp .ini so they never touch the registry
    settings_path: str | None = None

    def __init__(self):
        self._xprobe = None          # Virtuoso cross-probe, off until asked
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
        self._build()
        self._restore_settings()

    def _settings(self) -> QSettings:
        if self.settings_path:
            return QSettings(self.settings_path, QSettings.IniFormat)
        return QSettings("CircuitInsight", "desktop")

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
            lambda _t: self._refresh_net_decor())
        self.out_combo.currentTextChanged.connect(
            lambda _t: self._refresh_net_decor())
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
        tb.addWidget(QLabel(" in: "))
        tb.addWidget(self.in_combo)
        tb.addWidget(QLabel(" out: "))
        tb.addWidget(self.out_combo)
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
        tb.addSeparator()
        tb.addWidget(self.solve_btn)
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
        self.pm_spin = self._spin(5.0, 0.5, 45.0, 1.0, " ° PM")
        self.pm_spin.setToolTip("Phase margin reproduced within this")
        self.gm_spin = self._spin(2.0, 0.5, 20.0, 0.5, " dB GM")
        self.gm_spin.setToolTip("Gain margin reproduced within this")
        self.rej_spin = self._spin(3.0, 0.1, 40.0, 0.5, " dB track")
        self.rej_spin.setToolTip("|H| tracked within this many dB")
        self._pm_act = tb.addWidget(self.pm_spin)
        self._gm_act = tb.addWidget(self.gm_spin)
        self._rej_act = tb.addWidget(self.rej_spin)
        for s in (self.pm_spin, self.gm_spin, self.rej_spin):
            s.valueChanged.connect(lambda _v: self._on_strategy_changed())
        # budget spins only exist for the budgeted forms
        self._budget_lbl_act.setVisible(False)
        self._mag_act.setVisible(False)
        self._phase_act.setVisible(False)
        for a in (self._strategy_act, self._pm_act, self._gm_act,
                  self._rej_act):
            a.setVisible(False)
        self.toolbar = tb

        left = QSplitter(Qt.Vertical)
        left.setObjectName("left_split")
        left.addWidget(self._devices_group())
        left.addWidget(self._keepset_group())
        left.addWidget(self._history_group())
        left.setSizes([280, 340, 120])
        self.left_split = left

        self.canvas = FigureCanvasQTAgg(Figure(figsize=(5.2, 4.0)))
        # typeset HTML summary (falls back to a QTextEdit without the
        # WebEngine addon); same setPlainText/toPlainText API either way
        self.summary = summaryweb.make()
        # Expression surface: the KaTeX web view when QtWebEngine is present --
        # crisp vector math, hover = OP value, click = symbolClicked (the future
        # cross-probe handle). Falls back to the matplotlib mathtext canvas on a
        # PySide6 install without the WebEngine addon.
        self.exprweb = None
        if exprweb.WEBENGINE:
            try:
                self.exprweb = exprweb.ExprWebView()
                self.exprweb.bridge.symbolClicked.connect(self._select_keep_symbol)
                self.exprweb.bridge.selectionChanged.connect(
                    self._xprobe_selection)
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
        expr_tab = QWidget()
        _ev = QVBoxLayout(expr_tab)
        _ev.setContentsMargins(0, 0, 0, 0)
        _ev.setSpacing(2)
        _ev.addWidget(self.fullnames_chk)
        _ev.addWidget(expr_surface)
        # Two overlapping Bode curves cannot show a 0.5 dB residual. The Error
        # tab is where "how good is this model" actually becomes readable.
        self.err_canvas = FigureCanvasQTAgg(Figure(figsize=(5.2, 3.0)))
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
        tabs.addTab(self._comp_page(), "Compensation")
        tabs.addTab(self._gft_page(), "GFT")
        self._reduce_tab = self._reduce_page()
        tabs.addTab(self._reduce_tab, "Reduce")
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
        self.cert_lbl.setStyleSheet("color: #555; font-size: 8pt;")
        self.cert_lbl.setAlignment(Qt.AlignHCenter)
        self.cert_lbl.setVisible(False)
        bl.addWidget(self.cert_lbl)
        self._cert_timer = QTimer(self)
        self._cert_timer.setSingleShot(True)
        self._cert_timer.setInterval(250)
        self._cert_timer.timeout.connect(self._refresh_certificate)
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
        plot_area = QWidget()
        pa = QHBoxLayout(plot_area)
        pa.setContentsMargins(0, 0, 0, 0)
        pa.setSpacing(0)
        pa.addWidget(plot_col, 1)
        pa.addWidget(self.nav, 0, Qt.AlignTop)
        self.canvas.mpl_connect("draw_event",
                                lambda _ev: self._sync_band_row())

        right = QSplitter(Qt.Vertical)
        right.setObjectName("right_split")
        right.addWidget(plot_area)
        right.addWidget(tabs)
        right.setSizes([500, 240])
        self.right_split = right

        split = QSplitter(Qt.Horizontal)
        split.setObjectName("h_split")
        split.addWidget(left)
        split.addWidget(right)
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
            for name, sp in (("h_split", self.h_split),
                             ("left_split", self.left_split),
                             ("right_split", self.right_split)):
                state = s.value("splitters/" + name)
                if state is not None:
                    sp.restoreState(state)
                    restored = True
            if not restored:
                # first run: plots get the width the stretch factors promise
                w = max(self.width(), 900)
                self.h_split.setSizes([340, w - 360])

    def _restore_settings(self):
        s = self._settings()
        geo = s.value("geometry")
        if geo is not None:
            self.restoreGeometry(geo)
        for key, spin in (("budget/mag", self.mag_spin),
                          ("budget/phase", self.phase_spin),
                          ("budget/solve_s", self.budget_spin)):
            v = s.value(key)
            if v is not None:
                try:
                    spin.setValue(float(v))
                except (TypeError, ValueError):
                    pass
        lo, hi = s.value("budget/band_lo"), s.value("budget/band_hi")
        if lo is not None and hi is not None:
            try:
                self.band_slider.setValues(float(lo), float(hi))
            except (TypeError, ValueError):
                pass
        mode = s.value("ui/mode")
        if mode and self.mode_combo.findText(str(mode)) >= 0:
            self.mode_combo.setCurrentText(str(mode))
        # cross-probe: restoring "on" re-attempts the connection, and the
        # toggle handler already un-checks itself with a reason when Virtuoso
        # is not there -- so a remembered "on" is safe with no session running
        if str(s.value("ui/xprobe", "false")).lower() in ("true", "1"):
            self.a_xprobe.setChecked(True)
        self._rebuild_recents()

    def closeEvent(self, event):
        s = self._settings()
        s.setValue("geometry", self.saveGeometry())
        for name, sp in (("h_split", self.h_split),
                         ("left_split", self.left_split),
                         ("right_split", self.right_split)):
            s.setValue("splitters/" + name, sp.saveState())
        s.setValue("budget/mag", self.mag_spin.value())
        s.setValue("budget/phase", self.phase_spin.value())
        s.setValue("budget/solve_s", self.budget_spin.value())
        s.setValue("budget/band_lo", self.band_slider.values()[0])
        s.setValue("budget/band_hi", self.band_slider.values()[1])
        s.setValue("ui/mode", self.mode_combo.currentText())
        s.setValue("ui/xprobe", self.a_xprobe.isChecked())
        s.sync()
        super().closeEvent(event)

    def recents(self) -> list[tuple[str, str]]:
        s = self._settings()
        raw = s.value("recent", []) or []
        if isinstance(raw, str):          # QSettings: 1-element list -> str
            raw = [raw]
        out = []
        for entry in raw:
            parts = str(entry).split("|")
            if len(parts) == 2:
                out.append((parts[0], parts[1]))
        return out

    def _push_recent(self, cin: str, psf: str):
        pairs = [(str(cin), str(psf))]
        pairs += [p for p in self.recents() if p != pairs[0]]
        s = self._settings()
        s.setValue("recent", ["|".join(p) for p in pairs[:6]])
        s.sync()
        self._rebuild_recents()

    def _rebuild_recents(self):
        self.m_recent.clear()
        pairs = self.recents()
        if not pairs:
            self.m_recent.addAction("(empty)").setEnabled(False)
            return
        for cin, psf in pairs:
            act = self.m_recent.addAction(Path(cin).name + "  —  " + cin)
            act.triggered.connect(
                lambda _=False, c=cin, p=psf: self._open_recent(c, p))

    def _open_recent(self, cin, psf):
        try:
            self.open_session(cin, psf)
        except Exception as exc:
            QMessageBox.critical(self, "Open failed",
                                 f"{type(exc).__name__}: {exc}")

    def _cancel_solve(self):
        if self._thread is not None:
            self._thread.cancel()
            self.cancel_btn.setEnabled(False)
            self.progress.setFormat("cancelling…")

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
        self.a_rank.triggered.connect(self._rank)
        self.a_suggest = m_an.addAction("Suggest keep-set ≤ &budget")
        self.a_suggest.triggered.connect(self._suggest_keep)
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
        m_dev.addSeparator()
        m_dev.addAction("&Clear matches").triggered.connect(self.clear_matches)
        m_dev.addSeparator()
        # cross-probe: select here, highlight in the Virtuoso schematic
        self.a_xprobe = m_dev.addAction("Cross-&probe to Virtuoso")
        self.a_xprobe.setCheckable(True)
        self.a_xprobe.setToolTip(
            "Highlight the clicked symbol's device in the Virtuoso schematic.\n"
            "Needs the circuitinsight[virtuoso] extra, a running skillbridge\n"
            "server, and cin_xprobe.il loaded in the CIW.")
        self.a_xprobe.toggled.connect(self._on_xprobe_toggled)
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
        self.nets_tree.connectionActivated.connect(self._goto_instance)

        self.dev_tabs = QTabWidget()
        self.dev_tabs.addTab(self.devices, "Instances")
        self.dev_tabs.addTab(self.nets_tree, "Nets")

        sugg = QPushButton("Suggest")
        sugg.clicked.connect(self.suggest_matches)
        matchsel = QPushButton("Match sel.")
        matchsel.clicked.connect(self.match_selected)
        unmatch = QPushButton("Unmatch")
        unmatch.setToolTip("Dissolve the match group of the selected "
                           "device (or right-click a 🔗 member)")
        unmatch.clicked.connect(self.unmatch_selected)
        clr = QPushButton("Clear")
        clr.clicked.connect(self.clear_matches)
        row = QHBoxLayout()
        for wdg in (sugg, matchsel, unmatch, clr):
            row.addWidget(wdg)

        self.matchval_combo = QComboBox()
        for label, tip in (
            ("representative", "one member's values stand for the group "
             "(default: whichever the netlist reaches first; right-click "
             "a 🔗 member to choose)"),
            ("average", "the arithmetic mean of the members' values"),
            ("weighted", "the mean weighted by each member's band "
             "sensitivity — the member that shapes the response most "
             "pulls hardest (measured best on both benches)"),
        ):
            self.matchval_combo.addItem(label)
            self.matchval_combo.setItemData(self.matchval_combo.count() - 1,
                                            tip, Qt.ToolTipRole)
        self.matchval_combo.setToolTip(
            "How a match group's shared symbols get their numeric values")
        self.matchval_combo.currentTextChanged.connect(
            self._on_matchval_changed)
        mrow = QHBoxLayout()
        mrow.addWidget(QLabel("values:"))
        mrow.addWidget(self.matchval_combo, 1)

        v = QVBoxLayout()
        v.addWidget(self.dev_tabs, 1)
        v.addLayout(row)
        v.addLayout(mrow)
        return self._collapsible(box, v)

    def _keepset_group(self):
        # NB: a QGroupBox's minimumSizeHint includes its TITLE width — a
        # sentence-long title here once forced the whole left pane to
        # 1226 px minimum and squeezed the plots to a sliver. Short title,
        # sentence in the tooltip.
        box = QGroupBox("Keep symbolic")
        box.setToolTip("Which parameters stay as letters — exact either "
                       "way; use Simplify to trade accuracy for size")
        self.keep_tbl = QTableWidget(0, 5)
        self.keep_tbl.setHorizontalHeaderLabels(
            ["symbol", "dcOp", "score", "peaks", "LaTeX"])
        self.keep_tbl.horizontalHeader().setStretchLastSection(True)
        # only the LaTeX column edits; the AllEditTriggers + per-item
        # ItemIsEditable flag (set in _fill_keep_table) confines editing there
        self.keep_tbl.setEditTriggers(QAbstractItemView.AllEditTriggers)
        self.keep_tbl.itemChanged.connect(self._on_keep_changed)

        from PySide6.QtWidgets import QLineEdit

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
        rankb.clicked.connect(self._rank)
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
        # split advisory: whether tearing this solve pays, and which
        # AC-ground designation would make it pay (analysis/tearing.py).
        # Off by default -- it costs a graph scan plus numeric solves.
        self.split_lbl = QLabel("")
        self.split_lbl.setWordWrap(True)
        self.split_lbl.setStyleSheet("color: #555;")
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

        v = QVBoxLayout()
        v.addWidget(self.keep_filter)
        v.addWidget(self.keep_tbl, 1)
        v.addLayout(ctl)
        v.addLayout(brow)
        v.addLayout(srow)
        return self._collapsible(box, v)

    def _crumb_steps(self):
        """(label, action, done) per step. "Done" is judged by what exists,
        so the crumb needs no bookkeeping calls sprinkled through the app —
        callers just _update_crumb() after anything user-visible."""
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
                color, weight = "#1e5c2f", "normal"      # done: quiet green
                mark = "✓ "
            elif not current_seen:
                color, weight = "#1a466b", "bold"        # the next step
                mark = ""
                current_seen = True
            else:
                color, weight = "#888888", "normal"      # not yet
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
        elif action == "keep" and self.controller is not None:
            self._rank()
            self._update_crumb()
        elif action == "solve" and self.controller is not None:
            self.solve()

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
        self._show(r, push_history=False)
        self.statusBar().showMessage(
            f"first light: {inp} → {out} numeric, "
            f"{r.dc_gain_db:.2f} dB — tick symbols and Solve to refine")

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
            "color: #1e5c2f;" if best and best["pays"] else "color: #555;")
        self.split_lbl.setText("split: " + adv.verdict())

    def _apply_keep_filter(self, text: str):
        text = text.strip().lower()
        for i in range(self.keep_tbl.rowCount()):
            it = self.keep_tbl.item(i, 0)
            hide = bool(text) and text not in it.text().lower()
            self.keep_tbl.setRowHidden(i, hide)

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
        colors = {"info": ("#eef4fa", "#1a466b"),
                  "ok": ("#eaf6ec", "#1e5c2f"),
                  "warn": ("#fdf3e0", "#7a5200"),
                  "error": ("#fbe9e7", "#8a1c12")}
        bg, fg = colors.get(severity, colors["info"])
        self.msg_strip.setStyleSheet(
            f"QLabel {{ background: {bg}; color: {fg}; padding: 3px 8px;"
            f" border-radius: 2px; }}")
        self.msg_strip.setText(text)
        self.msg_strip.setVisible(bool(text))

    def _clear_strip(self):
        self.msg_strip.hide()
        self.msg_strip.setText("")

    def _on_mode_changed(self, mode: str):
        probed = mode in ("Loop gain", "Compensate", "Modes", "GFT",
                          "Impedance")
        self.probe_lbl.setText(" port: " if mode == "Impedance"
                               else " probe: ")
        if self.controller is not None and probed:
            want = (self.controller.ports if mode == "Impedance"
                    else self.controller.probes)
            have = [self.probe_combo.itemText(i)
                    for i in range(self.probe_combo.count())]
            if have != list(want):
                self.probe_combo.clear()
                self.probe_combo.addItems(want)
        self._probe_lbl_act.setVisible(probed)
        self._probe_act.setVisible(probed)
        self._probe2_act.setVisible(mode == "Modes")
        # GFT needs the input/output designations alongside the probe;
        # the Reduce bench prices grounding against the in->out transfer
        io_on = mode in ("Transfer", "GFT", "Reduce circuit")
        self.out_combo.setEnabled(io_on)
        self.in_combo.setEnabled(io_on)
        # the result-form selector belongs to the Transfer workflow
        transfer = mode == "Transfer"
        self._form_act.setVisible(transfer)
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
        if mode == "Compensate":
            self.tabs.setCurrentWidget(self._comp_tab)
        if mode == "GFT":
            self.tabs.setCurrentWidget(self._gft_tab)
        if mode == "Reduce circuit":
            self.tabs.setCurrentWidget(self._reduce_tab)

    def _set_tab_visibility(self, mode: str):
        """Summary/Expression/Error are views of any result and stay put;
        the bench-specific tabs appear only in their bench, which is what
        makes 'bench' mean something."""
        vis = {
            self._whatif_tab: mode == "Transfer",
            self._comp_tab: mode == "Compensate",
            self._gft_tab: mode == "GFT",
            self._reduce_tab: mode == "Reduce circuit",
        }
        for w, on in vis.items():
            idx = self.tabs.indexOf(w)
            if idx >= 0:
                self.tabs.setTabVisible(idx, on)

    def _on_tool_selected(self, name: str):
        if name and self.mode_combo.currentText() != name:
            self.mode_combo.setCurrentText(name)

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
        choice, ok = QInputDialog.getItem(
            self, "Declare AC as return ratio",
            "v(net)/v(input) in this run's AC sweep IS the loop gain:",
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
        ax1.semilogx(f, 20 * np.log10(np.abs(h)), color="#E69F00",
                     lw=1.3, ls="--", label="what-if")
        ax2.semilogx(f, np.degrees(np.unwrap(np.angle(h))),
                     color="#E69F00", lw=1.3, ls="--")
        view.figure_legend(self.canvas.figure, ax1)
        theme.style_figure(self.canvas.figure)
        self.canvas.draw_idle()
        if self.result.out.startswith("T@"):
            from ..session import _loop_margins
            pm, fpm, gm, _ = _loop_margins(f, h)
            self._wf_pm.setText(
                f"what-if margins:  PM {pm:.1f}°"
                f" @ {view.eng(fpm, 'Hz')}" +
                (f",  GM {gm:.1f} dB" if gm is not None else "")
                if pm is not None else "what-if: no unity crossing")

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
        row.addWidget(QLabel("budget:"))
        self.acg_budget = self._spin(0.1, 0.001, 3.0, 0.05, " dB")
        self.acg_budget.setDecimals(3)
        row.addWidget(self.acg_budget)
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

            from ..units import eng

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
        for b in (self.solve_btn, self.a_solve, self.a_simplify,
                  self.a_reduce):
            b.setEnabled(True)
        lines = []
        # the coefficient stories were computed alongside the deep pass;
        # lead with the DISPLAYED numerals before the full per-numeral list
        if self.controller is not None and self.result is not None:
            rkeep = (self.result.keep
                     if isinstance(self.result.keep, list) else ())
            coarse = self.controller.cached_numerals(
                self.result.inp, self.result.out, keep=rkeep)
            if coarse:
                from ..analysis.explain import ratio_lines
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
        budget = self.acg_budget.value()
        self._launch(
            lambda cb: self.controller.scan_removals(inp, out,
                                                     budget_db=budget),
            f"pricing element removals for {inp} → {out} …",
            on_done=self._on_removal_scan_done, est_s=None)

    def _on_removal_scan_done(self, rep):
        self._tick.stop()
        self._t0 = None
        self.progress.hide()
        self.cancel_btn.hide()
        for b in (self.solve_btn, self.a_solve, self.a_simplify,
                  self.a_reduce):
            b.setEnabled(True)
        self.acg_preview.setPlainText(rep.describe())
        if rep.recommended:
            self.statusBar().showMessage(
                f"removal scan: {', '.join(rep.recommended)} deletable "
                f"together at {rep.joint_db:.3g} dB")
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
        for b in (self.solve_btn, self.a_solve, self.a_simplify,
                  self.a_reduce):
            b.setEnabled(True)
        lines = ["pole attribution (nudge-verified):"]
        lines += ["  " + a.describe() for a in atts]
        self.summary.setPlainText(self.summary.toPlainText()
                                  + "\n\n" + "\n".join(lines))
        self.tabs.setCurrentIndex(0)             # Summary
        self.statusBar().showMessage(
            f"poles attributed: {len(atts)} (see Summary)")

    def explain_numbers(self):
        """Analysis menu: rank the collapsed parameters behind each
        numeral of H(s). The current keep-set is excluded — those are
        already letters in the expression."""
        if self.controller is None:
            return
        inp, out = self._io()
        keep = self.checked_keep()
        self._launch(
            lambda cb: self.controller.explain_numerals(inp, out, keep=keep,
                                                        progress=cb),
            f"explaining the numbers of {inp} → {out} …",
            on_done=self._on_explain_done, est_s=None)

    def _on_explain_done(self, stories):
        self._tick.stop()
        self._t0 = None
        self.progress.hide()
        self.cancel_btn.hide()
        for b in (self.solve_btn, self.a_solve, self.a_simplify,
                  self.a_reduce):
            b.setEnabled(True)
        from ..analysis.explain import ratio_lines
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

    def run_acg_scan(self):
        if self.controller is None:
            return
        inp, out = self._io()
        budget = self.acg_budget.value()
        self._launch(
            lambda cb: self.controller.scan_ac_grounds(inp, out,
                                                       budget_db=budget),
            f"scanning AC-ground candidates for {inp} → {out} …",
            on_done=self._on_acg_scan_done, est_s=None)

    def _on_acg_scan_done(self, rep):
        self._tick.stop()
        self._t0 = None
        self.progress.hide()
        self.cancel_btn.hide()
        for b in (self.solve_btn, self.a_solve, self.a_simplify,
                  self.a_reduce):
            b.setEnabled(True)
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
                cost_it = QTableWidgetItem(f"{c.worst_db:.3g} dB")
                deg_it = QTableWidgetItem(f"{c.worst_deg:.3g}°")
                kind_it = QTableWidgetItem(c.kind)
                gates_it = QTableWidgetItem(", ".join(c.controls[:6]))
                if not c.within_budget:
                    from PySide6.QtGui import QBrush, QColor
                    cost_it.setForeground(QBrush(QColor("#8a1c12")))
                for col, it in enumerate((node_it, cost_it, deg_it,
                                          kind_it, gates_it)):
                    if col:
                        it.setFlags(ro)
                    self.acg_tbl.setItem(i, col, it)
        finally:
            self._filling = False
        if self._acg_pending:
            # a Nets-tree wish arrived before the scan: tick it now that
            # it is priced (unknown nets simply are not candidates)
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
            joint = self.controller.acground_joint(inp, out, nodes)
            pv = self.controller.preview_reduction(nodes)
        except Exception as exc:
            self.acg_joint_lbl.setText(f"({type(exc).__name__}: {exc})")
            return
        self.acg_joint_lbl.setText(
            f"ticked set: {joint:.3g} dB together — the only approximation")
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
            lambda cb: self.controller.apply_reduction(nodes, inp=inp,
                                                       out=out),
            f"applying reduction ({', '.join(nodes)}) …",
            on_done=self._on_reduction_applied, est_s=None)

    def _on_reduction_applied(self, summ):
        self._tick.stop()
        self._t0 = None
        self.progress.hide()
        self.cancel_btn.hide()
        for b in (self.solve_btn, self.a_solve, self.a_simplify,
                  self.a_reduce):
            b.setEnabled(True)
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
        self._set_strip(
            f"circuit reduced: {summ['prims_before']} → "
            f"{summ['prims_after']} primitives at a measured "
            f"{summ['worst_db']:.3g} dB / {summ['worst_deg']:.3g}° "
            f"({summ['inp']} → {summ['out']}); grounding was the only "
            f"approximation — every bench now analyses the reduced circuit",
            "ok")

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
        if summ is None:
            self.red_banner.setText("circuit: as imported")
            self.red_banner.setStyleSheet("color: #555;")
            self.acg_revert.setEnabled(False)
        else:
            self.red_banner.setText(
                f"circuit: REDUCED — grounded {', '.join(summ['nodes'])}; "
                f"{summ['prims_before']} → {summ['prims_after']} primitives, "
                f"≤ {summ['worst_db']:.3g} dB measured")
            self.red_banner.setStyleSheet("color: #1e5c2f;")
            self.acg_revert.setEnabled(True)

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
        self.pm_spin = self._spin(60.0, 30.0, 85.0, 1.0, " °")
        row.addWidget(self.pm_spin)
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
        for w in (self.pm_lbl, self.pm_spin):
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
            kw["pm_target"] = self.pm_spin.value()
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
        for b in (self.solve_btn, self.a_solve, self.suggest_btn):
            b.setEnabled(True)
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
            from ..analysis.compensate import LoopGainUpdater
            from ..engine.mna import build_mna

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
        ax1.semilogx(f, 20 * np.log10(np.abs(T)), color="#E69F00", lw=1.3,
                     ls="--", label="preview")
        ax2.semilogx(f, np.degrees(np.unwrap(np.angle(T))), color="#E69F00",
                     lw=1.3, ls="--")
        ax1.legend(fontsize=8, frameon=False, loc="lower left")
        theme.style_figure(self.canvas.figure)
        self.canvas.draw_idle()
        from ..session import _loop_margins
        pm, fpm, gm, _ = _loop_margins(f, T)
        note = (f"preview PM {pm:.1f}° @ {view.eng(fpm, 'Hz')}"
                if pm is not None else "preview: no crossing")
        if gm is not None:
            note += f",  GM {gm:.1f} dB"
        if len(branches) > 1:
            note += f"  ({len(branches)} branches installed)"
        self._comp_hint.setText(note)

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
        for b in (self.solve_btn, self.a_solve):
            b.setEnabled(True)
        fig = self.canvas.figure
        fig.clear()
        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)
        colors = ("#0072B2", "#D55E00")
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

        from ..analysis import nested_gft
        from ..analysis.gft import _probe_indices
        from ..engine.mna import S

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
        for b in (self.solve_btn, self.a_solve, self.gft_btn):
            b.setEnabled(True)
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
                (q["H"], "H", "#0072B2", "-"),
                (q["Hinf"], "H∞", "#009E73", "--"),
                (loop_part, "loop part", "#E69F00", "-"),
                (ft_part, "feedthrough", "#CC79A7", ":")):
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
        ax2.loglog(f, np.maximum(dev, 1e-16), color="#D55E00", lw=1.2)
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

    # ----------------------------------------------------------------- open
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
            QMessageBox.critical(self, "Open failed", f"{type(exc).__name__}: {exc}")

    def _on_cap_model_toggled(self, matrix_on: bool):
        self.cap_model = "matrix" if matrix_on else "lumped"
        self._save_cap_model()
        if self.controller is None:
            return
        # the model is baked into the reconstruction, so re-open the same run
        # -- preserving the user's in/out, matches and keep ticks -- and
        # re-run whatever was last shown so the change is visible immediately
        inp, out = self._io()
        mode = self.mode_combo.currentText()
        groups = list(self._match_groups)
        checked = self.checked_keep()
        had_result = self.result is not None
        with_probe = self.probe_combo.currentText()
        self.statusBar().showMessage(
            f"re-opening with {self.cap_model} caps …")
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self.open_session(self._cin, self._psf)
        except Exception as exc:
            self._set_strip(f"re-open failed: {exc}", "error")
            return
        finally:
            QApplication.restoreOverrideCursor()
        if inp:
            self.in_combo.setCurrentText(inp)
        if out:
            self.out_combo.setCurrentText(out)
        if groups:
            self._match_groups = groups
            self._apply_matches()
        if checked:
            try:
                ranking = self.controller.rank_symbols(*self._io())
                self._fill_keep_table(ranking, checked=checked)
            except Exception:
                pass
        self.mode_combo.setCurrentText(mode)
        if with_probe and self.probe_combo.findText(with_probe) >= 0:
            self.probe_combo.setCurrentText(with_probe)
        self.statusBar().showMessage(f"cap model: {self.cap_model}")
        self.log(f"cap model: {self.cap_model} (run re-opened)")
        # re-run the last shown analysis so the model change is visible --
        # through the ASYNC path with progress and Cancel. The synchronous
        # re-run froze the GUI for the whole hybrid solve when a keep set
        # was ticked (minutes on fc): whatever Solve would launch, this
        # must launch the same way.
        if had_result and mode in ("Transfer", "Loop gain"):
            try:
                self.solve()
            except Exception:
                pass

    def _save_cap_model(self):
        s = self._settings()
        s.setValue("cap_model", self.cap_model)
        s.sync()

    def open_session(self, cin, psf, probe=None):
        # cap model chosen in the Model menu; matrix is the accurate one on
        # non-reciprocal processes (SKY130 CM loops shift ~6 deg vs lumped),
        # and the GUI default -- lumped is offered for the textbook contrast
        self.controller = SessionController.open(
            cin, psf, cap_model=getattr(self, "cap_model", "matrix"))
        self._cin, self._psf = str(cin), str(psf)
        self._match_groups = []
        self._push_recent(str(cin), str(psf))
        self._load_aliases()
        self._populate()
        # A run that carries an stb analysis opens in the Loop-gain bench
        # with ITS designated probe preselected (--probe overrides the
        # discovery; any vsource qualifies).
        probe = probe or self.controller.stb_probe()
        if probe and self.probe_combo.findText(probe) >= 0:
            self.mode_combo.setCurrentText("Loop gain")
            self.probe_combo.setCurrentText(probe)
            self.statusBar().showMessage(
                f"stb analysis found: probe {probe} preselected "
                f"— Solve for the loop gain")
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
        from ..engine import interp
        interp.PROBE_BACKEND = None
        was, self._filling = self._filling, True
        try:
            self.backend_combo.setCurrentIndex(0)
        finally:
            self._filling = was
        self.estimate_lbl.setText("estimate: —")
        self.estimate_lbl.setText(
            "Rank scores each parameter's effect on the band — start there")
        inv = {v: k for k, v in self._POLICY_MAP.items()}
        self._filling = True
        try:
            self.matchval_combo.setCurrentText(
                inv.get(c.match_value_policy, "weighted"))
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
                  self.a_reduce, self.a_export):
            b.setEnabled(True)
        for b in (self.a_export,):                        # nothing solved yet
            b.setEnabled(False)
        self._auto_setup()

    def _auto_setup(self):
        """Make the tool SYMBOLIC by default. Left alone, the keep table opens
        empty -> keep=[] -> a purely numeric solve, so a symbolic analyzer's
        first result shows no symbols but `s`. Instead: apply the suggested
        matched pairs and pre-select a budget-fit keep set, so the first Solve
        already reads gm/(gds_n+gds_p). All heuristic and all reversible (Clear
        matches, untick symbols); guarded so a hiccup never blocks opening.
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
            ranking = c.rank_symbols(inp, out)
            self._fill_keep_table(ranking, checked=list(plan.keep))
            self._update_estimate()          # the empty-state hint yields
            kept = ", ".join(plan.keep) if plan.keep else "none within budget"
            ref_note = (" | run carries: " + " ".join(carries)
                        if carries else "")
            self.statusBar().showMessage(
                f"{name}: {n} devices — auto keep-set [{kept}]; "
                f"Solve for a symbolic result, or edit the ticks.{ref_note}")
            ii = getattr(self, "_ii", {})
            if ii:
                worst = sorted(ii, key=lambda k: -ii[k])[:4]
                self._set_strip(
                    "\N{WARNING SIGN} impact ionization active without a gii "
                    "model on " + ", ".join(f"{k} ({ii[k]:.1%})" for k in worst)
                    + " -- the first-order model is incomplete here (identify "
                    "gii by AC injection; see the r2r case)", "warn")
        except Exception as exc:
            # fall back to the manual flow -- never let auto-setup break opening
            self.statusBar().showMessage(
                f"{name}: {n} devices — Suggest matches, Rank, Solve "
                f"(auto-setup skipped: {type(exc).__name__}).")
        self._update_crumb()

    # ------------------------------------------------------------- matches
    def _apply_matches(self):
        self.controller.set_matches(*self._match_groups)
        self._refresh_matches_label()
        self._surface_match_conflicts()
        self.keep_tbl.setRowCount(0)                       # ranking is now stale
        self.estimate_lbl.setText("estimate: — (re-Rank)")

    def _refresh_matches_label(self):
        """Match sets live on the tree itself: every member wears 🔗n and
        its group tint (devtree.set_groups)."""
        self.devices.set_groups(self._match_groups)

    _POLICY_MAP = {"representative": "representative", "average": "mean",
                   "weighted": "weighted"}

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
            self.matchval_combo.setCurrentText("representative")
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
        try:
            ranking = self.controller.rank_symbols(inp, out)
        except Exception as exc:
            QMessageBox.warning(self, "Rank failed", f"{type(exc).__name__}: {exc}")
            return
        # re-ranking REORDERS the symbols; it must not silently discard
        # the user's selection (a stray F5 mid-solve emptied the keep
        # set, and the estimate then re-costed the empty one)
        self._fill_keep_table(ranking, checked=self.checked_keep())

    @staticmethod
    def _sym_device(name: str) -> str:
        """Owning-device key of a symbol name (gm_I0_MN1 -> I0_MN1)."""
        return name.split("_", 1)[1] if "_" in name else name

    def _fill_keep_table(self, ranking, checked=()):
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
        from PySide6.QtGui import QBrush, QColor
        tints = (QBrush(), QBrush(QColor("#f0f4f8")))
        self.keep_tbl.setRowCount(len(rows))
        prev_dev, band = None, 0
        for i, (name, opval, score, peak) in enumerate(rows):
            if grouped:
                dev = self._sym_device(name)
                if dev != prev_dev:
                    band, prev_dev = 1 - band, dev
            sym = QTableWidgetItem(name)
            sym.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
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
        if text:
            self.controller.sym_aliases[name] = text
        else:
            self.controller.sym_aliases.pop(name, None)
        self._save_aliases()
        self._render_expr()

    def _alias_key(self) -> str:
        stem = Path(str(self.controller.cin_path)).name if self.controller \
            else "?"
        return "aliases/" + stem

    def _save_aliases(self):
        import json

        s = self._settings()
        s.setValue(self._alias_key(), json.dumps(self.controller.sym_aliases))
        s.sync()

    def _load_aliases(self):
        import json

        raw = self._settings().value(self._alias_key())
        if raw:
            try:
                self.controller.sym_aliases = dict(json.loads(raw))
            except Exception:
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

    def _on_keep_changed(self, item):
        if self._filling or self.controller is None:
            return
        self._update_crumb()
        if item.column() == 4:                    # per-symbol LaTeX override
            name = self.keep_tbl.item(item.row(), 0).text()
            text = item.text().strip()
            if text:
                self.controller.sym_aliases[name] = text
            else:
                self.controller.sym_aliases.pop(name, None)
            self._save_aliases()
            self._render_expr()
            return
        self._update_estimate()                   # a checkbox toggled

    #: combo index -> engine.interp.PROBE_BACKEND value (None = auto)
    _BACKENDS = (None, "qq", "bot", "zp", "ratfun")

    def _on_backend_changed(self, idx: int):
        """Expert override of the solver auto-selector. Written to the
        engine global the estimator and the solve BOTH consult, so the
        estimate immediately re-costs with the chosen backend."""
        if self._filling:
            return
        from ..engine import interp

        interp.PROBE_BACKEND = self._BACKENDS[idx] \
            if 0 <= idx < len(self._BACKENDS) else None
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
        try:
            est = self.controller.estimate(inp, out, self.checked_keep())
            self.estimate_lbl.setText(f"estimate: {est}")
            self._show_auto_backend(getattr(est, "backend", None))
            secs = getattr(est, "seconds", None)
            # the progress bar shows elapsed against this, so a wrong
            # estimate is visible while you wait, not only afterwards
            self._est_s = secs
            budget = self.budget_spin.value()
            color = "#1e5c2f"                        # green: within budget
            if secs is None or secs > budget:
                color = "#8a1c12"                    # red: over / unknown
            elif secs > 0.7 * budget:
                color = "#7a5200"                    # amber: close
            self.estimate_lbl.setStyleSheet(f"color: {color};")
        except Exception as exc:
            self.estimate_lbl.setStyleSheet("")
            self.estimate_lbl.setText(f"estimate: — ({type(exc).__name__})")

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
            ranking = self.controller.rank_symbols(inp, out)
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
    _KEEP_EST = object()

    def _launch(self, fn, label, on_done=None, est_s=_KEEP_EST,
                growth_reason=None):
        for b in (self.solve_btn,
                  self.a_solve, self.a_simplify, self.a_reduce):
            b.setEnabled(False)
        self.statusBar().showMessage(label)
        self.progress.setRange(0, 0)      # busy until an estimate exists
        self._live_est = None             # fresh launch, fresh refinement
        self._growth_reason = growth_reason
        self._t0 = time.monotonic()
        # an advisory pass is NOT priced by the keep-set estimator; it
        # used to inherit the solve's number and log a promise that
        # belonged to a different analysis entirely
        self._run_est = (self._est_s if est_s is self._KEEP_EST else est_s)
        est = (f"  [estimate ~{self._run_est:.0f}s]" if self._run_est
               else "  [no estimate for this pass]")
        self.log(f"START {label.rstrip(' …')}{est}")
        self._phase_totals, self._phase_runs = {}, {}
        self._phase, self._phase_t0 = None, None
        self._set_phase("preparing")
        self._tick.start()
        self.progress.show()
        self.cancel_btn.setEnabled(True)
        self.cancel_btn.show()
        self._thread = _Worker(fn)
        self._thread.progress.connect(self._on_progress)
        self._thread.done.connect(on_done or self._on_done)
        self._thread.failed.connect(self._on_failed)
        self._thread.cancelled.connect(self._on_cancelled)
        # worker-side narration (pursuit rounds and the like) lands in
        # the Log through the queued signal -- never touch widgets there
        self._thread.note.connect(lambda m: self.log(f"  {m}"))
        self.worker_note = self._thread.note.emit
        self._thread.start()

    def solve(self):
        if self.controller is None:
            return
        if self.mode_combo.currentText() == "Compensate":
            self.suggest_comp()
            return
        if self.mode_combo.currentText() == "Modes":
            self.run_modes()
            return
        if self.mode_combo.currentText() == "Impedance":
            port = self.probe_combo.currentText()
            if not port:
                self.statusBar().showMessage("no port marker in this design")
                return
            keep = self.checked_keep()
            self._launch(lambda cb: self.controller.impedance_result(
                             port, keep=keep),
                         f"impedance at {port} …")
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
            self._launch(lambda cb: self.controller.loop_gain(
                             probe, keep=keep, progress=cb),
                         f"loop gain at {probe} …")
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
                     f"solving {inp} → {out} …")

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
        transfer = self.mode_combo.currentText() == "Transfer"
        form = self.form_combo.currentText()
        budgeted = transfer and form.startswith("Simplified")
        lowest = budgeted and "lowest" in form
        strat = self._strategy()
        # full order: dB/deg collapse budgets. Lowest order: the strategy
        # dropdown plus ITS spins — plain reuses the dB/deg pair.
        self._budget_lbl_act.setVisible(budgeted
                                        and (not lowest
                                             or strat == "plain"))
        self._mag_act.setVisible(budgeted and (not lowest
                                               or strat == "plain"))
        self._phase_act.setVisible(budgeted and (not lowest
                                                 or strat == "plain"))
        self._strategy_act.setVisible(lowest)
        self._pm_act.setVisible(lowest and strat == "stability")
        self._gm_act.setVisible(lowest and strat == "stability")
        self._rej_act.setVisible(lowest and strat == "rejection")
        self.band_row.setVisible(lowest)
        self._on_band_changed(*self.band_slider.values())

    def _strategy(self) -> str:
        return {"Gain & phase": "plain",
                "Stability (margins)": "stability",
                "Rejection (dB)": "rejection"}.get(
            self.strategy_combo.currentText(), "plain")

    def _strategy_opts(self) -> dict:
        s = self._strategy()
        if s == "stability":
            return {"pm_deg": self.pm_spin.value(),
                    "gm_db": self.gm_spin.value()}
        if s == "rejection":
            return {"rej_db": self.rej_spin.value()}
        return {"gain_db": self.mag_spin.value(),
                "phase_deg": self.phase_spin.value()}

    def _strategy_eps_eq(self) -> float:
        """The strategy budget as a comparable relative tolerance, for
        the order-certificate label."""
        s, o = self._strategy(), self._strategy_opts()
        if s == "stability":
            return 0.05
        db = o.get("rej_db", o.get("gain_db", 1.0))
        return 10.0 ** (db / 20.0) - 1.0

    def _on_strategy_changed(self):
        self._refresh_form_ui()
        self._cert_timer.start()

    def _refresh_certificate(self):
        """The band's demand at the set tolerance, printed beside the
        slider while dragging: 'lowpass band needs order 2 at 5%', plus
        the doublet caveat. First call may cost a numeric sweep (~1 s);
        after that ~50 ms."""
        if (self.controller is None
                or not self.band_row.isVisibleTo(self)):
            self.cert_lbl.setText("")
            self.cert_lbl.setVisible(False)
            return
        try:
            inp, out = self._io()
            fmin, fmax = self.band_slider.values()
            cert = self.controller.order_certificate(inp, out, fmin, fmax)
            self.cert_lbl.setText(
                cert.describe(self._strategy_eps_eq()))
            self.cert_lbl.setVisible(True)
        except Exception:
            self.cert_lbl.setText("")
            self.cert_lbl.setVisible(False)

    def _update_tol_bands(self):
        """The tolerance made visible. Full order: a fixed tube of ±mag
        dB / ±phase° around the model traces. Lowest order: the ANCHORED
        tube — |dH| <= eps*(|H| + anchor) rendered exactly, thin and
        relative above the anchor, flaring open below it (where the
        criterion stops caring), with the anchor level drawn dotted."""
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
                    color="#c9962a", alpha=0.18, zorder=0))
                if len(axes) > 1:
                    self._tol_bands.append(axes[1].fill_between(
                        f[mask], (ph - pdeg)[mask], (ph + pdeg)[mask],
                        color="#c9962a", alpha=0.18, zorder=0))
            elif strat == "rejection":
                rej = self.rej_spin.value()
                self._tol_bands.append(axes[0].fill_between(
                    f[mask], (mag - rej)[mask], (mag + rej)[mask],
                    color="#c9962a", alpha=0.18, zorder=0))
            else:                                # stability
                absh = np.abs(h)
                cross = np.where(np.diff(np.sign(
                    20 * np.log10(np.maximum(absh, 1e-300))))
                    != 0)[0]
                if cross.size:
                    fc0 = float(f[cross[0]])
                    near = mask & (f >= fc0 / 2.5) & (f <= fc0 * 2.5)
                    pmd = self.pm_spin.value()
                    self._tol_bands.append(axes[0].plot(
                        [fc0, fc0], [mag[mask].min(), mag[mask].max()],
                        color="0.35", lw=0.8, ls=":", zorder=1)[0])
                    if len(axes) > 1 and near.any():
                        self._tol_bands.append(axes[1].fill_between(
                            f[near], (ph - pmd)[near], (ph + pmd)[near],
                            color="#c9962a", alpha=0.18, zorder=0))
            self.canvas.draw_idle()
            return
        m = self.mag_spin.value()
        self._tol_bands.append(axes[0].fill_between(
            f, mag - m, mag + m,
            color="#c9962a", alpha=0.18, zorder=0))
        if len(axes) > 1:
            pd = self.phase_spin.value()
            self._tol_bands.append(axes[1].fill_between(
                f, ph - pd, ph + pd,
                color="#c9962a", alpha=0.18, zorder=0))
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
                ax.axvspan(fmin, fmax, color="#4a78a8", alpha=0.10,
                           zorder=0))
        self._update_tol_bands()
        self._cert_timer.start()          # the band's demand, debounced

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

    def impedance_sync(self, port):
        self._show(self.controller.impedance_result(
            port, keep=self.checked_keep()))
        return self.result

    def loop_gain_sync(self, probe=None):
        probe = probe or self.probe_combo.currentText()
        self._show(self.controller.loop_gain(probe,
                                             keep=self.checked_keep()))
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

    def log(self, text: str) -> None:
        """Append one timestamped line to the Log tab. Session-relative
        seconds, not wall-clock: what a reader compares is durations
        between events, and a relative clock survives being pasted into
        a report from another timezone."""
        t = time.monotonic() - self._log_t0
        self.logview.append(f"[{t:8.1f}s] {text}")
        doc = self.logview.document()
        if doc.blockCount() > self._LOG_MAX_LINES:
            cur = self.logview.textCursor()
            cur.movePosition(cur.MoveOperation.Start)
            for _ in range(doc.blockCount() - self._LOG_MAX_LINES):
                cur.select(cur.SelectionType.BlockUnderCursor)
                cur.removeSelectedText()
                cur.deleteChar()
        sb = self.logview.verticalScrollBar()
        sb.setValue(sb.maximum())               # follow the tail

    def _close_phase(self) -> float:
        """End the current phase, banking its DURATION. Entry timestamps
        alone made the reader subtract; durations are what a report is
        actually about, and a phase entered twice (a backend fallback
        re-running the grid) accumulates rather than overwrites."""
        now = time.monotonic()
        prev = getattr(self, "_phase", None)
        t0 = getattr(self, "_phase_t0", None)
        dur = 0.0
        if prev and t0 is not None:
            dur = now - t0
            self._phase_totals[prev] = self._phase_totals.get(prev, 0.0) + dur
            self._phase_runs[prev] = self._phase_runs.get(prev, 0) + 1
        self._phase_t0 = now
        return dur

    def _set_phase(self, phase, done=None, total=None):
        prev = getattr(self, "_phase", None)
        if phase != prev:                       # phase transitions only
            dur = self._close_phase()
            self._phase = phase
            self._phase_units = (done, total)
            el = (time.monotonic() - self._t0) if self._t0 else 0.0
            est = self._live_est or self._est_s
            tail = f", est ~{est:.0f}s" if est else ""
            took = f" [{prev} took {dur:.1f}s]" if prev and dur else ""
            self.log(f"  phase: {phase} (at {el:.0f}s{tail}){took}")
            if phase == "reconstructing" and prev == "evaluating" and dur:
                # reconstruction is the DOMINANT phase (measured 90% of a
                # sparse solve) and no cost model prices it. The learned
                # ratio turns the end of evaluation into a real
                # projection instead of a bar creeping on elapsed.
                self._project_from_recon_ratio(dur)
        else:
            # same phase, more work: a growing total is a real event (a
            # pursuit round ended, a prime batch was queued) and used to
            # pass silently -- the 70/74 mystery. Name it in the Log.
            old = self._phase_units[1] if self._phase_units else None
            if total and old and total > old:
                why = getattr(self, "_growth_reason", None) or (
                    "the stop test has not passed, so the phase queued "
                    "more work (another pursuit round or prime batch)")
                self.log(f"  {phase} total {old} -> {total}: {why}")
            self._phase = phase
            self._phase_units = (done, total)
        self._refresh_progress()

    def _phase_breakdown(self) -> str:
        """'evaluating 256.2s (35%, x2) · reconstructing 475.9s (65%)' --
        where the time actually went, and how often each phase ran."""
        tot = sum(self._phase_totals.values())
        if not tot:
            return ""
        parts = []
        for name, secs in sorted(self._phase_totals.items(),
                                 key=lambda kv: -kv[1]):
            runs = self._phase_runs.get(name, 1)
            again = f", x{runs}" if runs > 1 else ""
            parts.append(f"{name} {secs:.1f}s ({secs / tot:.0%}{again})")
        return " · ".join(parts)

    def _refresh_progress(self):
        """The bar moves with TIME against a LIVE estimate; the text keeps
        the units and both clocks.

        The bar's geometry is elapsed / estimate, refreshed every tick, so
        it advances with the seconds even between unit reports (a chunked
        parallel phase can be quiet for a while). The estimate itself is
        cheaply refined as units arrive: elapsed/fraction-done projects
        the total from observation, blended with the pre-solve estimate
        weighted by how much has actually been observed -- so the prior
        rules the first seconds and the measurement takes over. When the
        estimate improves, the bar's position re-derives from it, forward
        or back; an honest bar beats a monotone one that lies. With no
        estimate and no units yet it stays indeterminate, and the rising
        elapsed clock distinguishes "still working" from "hung"."""
        if self._t0 is None:
            return
        el = time.monotonic() - self._t0
        done, total = self._phase_units
        units = f" {done}/{total}" if total else ""
        live = self._live_est or self._run_est  # keep refinements across phases
        if total and done:
            frac = done / total
            observed = el / frac                # projected total from data
            w = frac                            # confidence grows with coverage
            prior = self._run_est
            live = observed if not prior else (1 - w) * prior + w * observed
        elif live and el > live:
            # no units to project from (reconstruction, a direct symbolic
            # determinant): the estimate cannot be refined by observation,
            # but elapsed OVERTAKING it proves it wrong. Grow it as a
            # moving lower bound so the bar keeps creeping and the number
            # stops asserting a finish time that has already passed.
            live = el * 1.15
        self._live_est = live
        est = ""
        if live:
            est = f" / ~{live:.0f}s"
            # "(over)" means the PROMISE was blown, not merely that the
            # live number moved: since the estimate now self-corrects,
            # elapsed rarely overtakes it -- what the user needs to see
            # is the live estimate having run away from the pre-solve one
            blown = (el > 1.2 * live
                     or (self._run_est and live > 1.5 * self._run_est))
            if blown:
                est += " (over)"
            self.progress.setRange(0, 1000)
            self.progress.setValue(int(1000 * min(0.99, el / live)))
        self.progress.setFormat(f"{self._phase}{units} — {el:.0f}s{est}")

    def _on_progress(self, done, total):
        """Evaluation units completed. Queued from the worker thread, so this
        runs on the GUI thread and may touch widgets.

        Units feed the LIVE estimate; the bar itself is time-driven in
        _refresh_progress. The evaluation is not the whole solve -- setup
        precedes it and the reconstruction follows -- so the bar names
        the phase rather than hitting 100% and appearing to hang."""
        if total <= 0:
            return
        if done >= total:                       # evaluation done; rebuild left
            # keep the time-driven bar running against the live estimate
            self._set_phase("reconstructing")
            return
        self._set_phase("evaluating", done, total)

    def _on_cancelled(self):
        self._tick.stop()
        self._t0 = None
        self.progress.hide()
        self.cancel_btn.hide()
        for b in (self.solve_btn,
                  self.a_solve, self.a_simplify, self.a_reduce):
            b.setEnabled(True)
        self.statusBar().showMessage("solve cancelled")
        self.log("CANCELLED")

    def _log_finish(self, verdict: str) -> None:
        """One closing line per run: how long it took, against what was
        promised, and which solver actually ran — the three numbers a
        report about a slow or surprising solve needs."""
        el = (time.monotonic() - self._t0) if self._t0 else 0.0
        est = self._run_est
        acc = ""
        if est:
            acc = f" (estimate ~{est:.0f}s, {el / est:.1f}x)"
        self._close_phase()                      # bank the final phase
        self.log(f"{verdict} after {el:.1f}s{acc}"
                 f"{self._backend_note()}")
        bd = self._phase_breakdown()
        if bd:
            self.log(f"  phases: {bd}")

    def _ensure_calibration(self):
        """Measure this machine ONCE, in the background, when no
        calibration exists. Without it every estimate comes from a
        deliberately pessimistic built-in default (alpha 3.0) that
        over-predicts a real gmpy2 + worker-pool machine by ~20x -- the
        estimates are not wrong by chance, they are un-measured. Bounded
        by calibrate()'s own max_seconds; failure is silent and simply
        leaves the default in place."""
        try:
            from ..analysis import estimate as _est
        except Exception:                        # pragma: no cover
            return
        if _est.get_calibration().platform != "builtin-default":
            return
        if _est.load_calibration() is not None:
            _est.set_calibration(_est.load_calibration())
            self.log("solve-time model: loaded this machine's calibration")
            return
        if getattr(self, "_calib_thread", None) is not None:
            return
        self.log("solve-time model: measuring this machine "
                 "(first run; estimates are the conservative default "
                 "until it finishes) …")

        class _Calib(QThread):
            done = Signal(object)

            def run(self):
                try:
                    self.done.emit(_est.calibrate(max_seconds=2.0))
                except Exception as exc:         # never break the GUI
                    self.done.emit(exc)

        def finished(res):
            self._calib_thread = None
            if isinstance(res, Exception):
                self.log(f"solve-time model: calibration failed ({res})")
                return
            self.log(f"solve-time model: calibrated "
                     f"(alpha_par {res.a_parallel:.3g}, "
                     f"{res.n_samples} samples) — estimates now use this "
                     f"machine's own numbers")
            self._update_estimate()

        self._calib_thread = _Calib()
        self._calib_thread.done.connect(finished)
        self._calib_thread.start()

    def _solve_key(self) -> str:
        try:
            from ..engine import interp
            tl = getattr(interp, "LAST_SOLVE", None) or {}
        except Exception:                        # pragma: no cover
            return "parallel"
        if tl.get("backend") == "bot":
            return "bot"
        return "parallel" if tl.get("n_dense_dets", 0) else "serial"

    def _project_from_recon_ratio(self, eval_s: float):
        try:
            from ..analysis import estimate as _est

            r = getattr(_est.get_calibration(), "r_" + self._solve_key(), 1.0)
        except Exception:                        # pragma: no cover
            return
        total = eval_s * (1.0 + r)
        if total > (self._live_est or 0):
            self._live_est = total
            self.log(f"  projected total ~{total:.0f}s "
                     f"(reconstruction runs ~{r:.1f}x evaluation here)")
        self._refresh_progress()

    def _learn_from_solve(self):
        """Feed the finished solve back into this machine's persistent
        calibration: the pre-solve estimate vs the wall clock. The model
        is fitted on synthetic ladders and covers only the evaluation,
        so on real circuits (reconstruction included) it runs low --
        this is what closes that gap over sessions instead of leaving
        every user with the factory number."""
        if self._t0 is None or not self._run_est:
            return
        actual = time.monotonic() - self._t0
        try:
            from ..analysis import estimate as _est
            from ..engine import interp

            tl = getattr(interp, "LAST_SOLVE", None) or {}
            bk = tl.get("backend")
            # learn on the key that PRICED this solve: the sparse path
            # has its own cost model, so mixing its samples into the
            # dense correction taught both the wrong thing
            key = "bot" if bk == "bot" else (
                "parallel" if tl.get("n_dense_dets", 0) else "serial")
            _est.observe(self._run_est, actual, parallel=(key != "serial"),
                         key=key)
            ev = self._phase_totals.get("evaluating", 0.0)
            rc = self._phase_totals.get("reconstructing", 0.0)
            if ev > 0 and rc > 0:
                _est.observe_phases(ev, rc, key=key)
        except Exception:
            pass                                # never break a finished solve

    def _on_done(self, result):
        self._log_finish("DONE")
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
            from ..engine import interp
        except Exception:                        # pragma: no cover
            return ""
        tl = getattr(interp, "LAST_SOLVE", None)
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

    def _on_failed(self, msg):
        self._tick.stop()
        self._t0 = None
        self.progress.hide()
        self.cancel_btn.hide()
        for b in (self.solve_btn,
                  self.a_solve, self.a_simplify, self.a_reduce):
            b.setEnabled(True)
        self._log_finish("FAILED")
        self.statusBar().showMessage("solve failed")
        self._set_strip("solve failed: " + msg, "error")

    def _show(self, result, overlays=(), push_history=True):
        self.result = result
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
        self.summary.setPlainText(view.summary_text(result))
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

    def _render_expr(self):
        """(Re)draw the Expression tab for the current result, honouring the
        Full-names toggle. Split from _show so the checkbox can re-render without
        re-solving."""
        if self.result is None:
            return
        base = not self.fullnames_chk.isChecked()
        aliases = self.controller.sym_aliases if self.controller else None
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
                self.exprweb.set_payload(view.expr_katex(
                    self.result, base=base, aliases=aliases,
                    numerals=numerals,
                    numhint="collapsed operating-point products — run "
                            "<b>Analysis → Explain the numbers</b> for the "
                            "ranked contributors"))
            except Exception:
                pass
            return
        try:                                              # matplotlib fallback
            n = len(view._expr_lines(self.result, base=base))
            # size the canvas to the content and let the scroll area handle the
            # overflow — rather than squeezing N lines into a fixed short panel
            self.expr_canvas.setMinimumHeight(int(34 * n + 24))
            view.expr_figure(self.result, self.expr_canvas.figure,
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

    # ------------------------------------------------------- cross-probe
    def _instances(self):
        """Instance names of the reconstruction, for symbol resolution."""
        if self.controller is None:
            return []
        try:
            an = self.controller._analyzer_ready()
        except Exception:
            return []
        return sorted({p.inst for p in an.primitives})

    def _on_xprobe_toggled(self, on: bool):
        from ..virtuoso import xprobe

        if not on:
            if self._xprobe is not None:
                self._xprobe.close()
                self._xprobe = None
            self._set_strip("cross-probe off")
            return
        probe, why = xprobe.CrossProbe.connect()
        self._xprobe = probe
        if probe is None:
            # a failed connection must not leave the menu claiming it is on
            self.a_xprobe.blockSignals(True)
            self.a_xprobe.setChecked(False)
            self.a_xprobe.blockSignals(False)
            self._set_strip(f"cross-probe unavailable: {why}", "warn")
        else:
            self._set_strip("cross-probe on: clicking a symbol or a device "
                            "row highlights it in the schematic")

    def _xprobe_selection(self, names):
        """Mirror the expression view's pinned SET into the schematic, so
        ctrl-clicking several symbols lights all their devices at once.

        The set is the single source of truth for what the schematic shows;
        symbolClicked stays a pure keep-table navigation signal, so the two
        never fight over the selection."""
        if self._xprobe is None:
            return
        from ..virtuoso.xprobe import instance_for_symbol

        known = self._instances()
        insts = [i for i in (instance_for_symbol(n, known) for n in names) if i]
        if insts:
            self.xprobe_instances(insts)
        else:
            self._xprobe.clear()

    def xprobe_symbol(self, name: str) -> bool:
        """Highlight the device a keep-set symbol belongs to."""
        from ..virtuoso.xprobe import instance_for_symbol

        inst = instance_for_symbol(name, self._instances())
        return self.xprobe_instances([inst] if inst else [])

    def xprobe_instances(self, instances) -> bool:
        """Highlight instances in the schematic; a no-op when probing is off."""
        if self._xprobe is None or not instances:
            return False
        ok = self._xprobe.highlight(instances)
        if not ok:
            self._set_strip(
                f"cross-probe: {', '.join(instances)} not found in the "
                f"schematic window", "warn")
        return ok

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
        """Selecting devices in the tree cross-probes them as a set."""
        if self._xprobe is None:
            return
        self.xprobe_instances(self.devices.selected_names())

    def add_to_report(self):
        """Snapshot the CURRENT view (whatever bench drew it) into the
        session report."""
        if self.controller is None:
            return
        n = len(self._report_sections) + 1
        mode = self.mode_combo.currentText()
        label = ""
        if self.result is not None:
            label = f"{self.result.inp} → {self.result.out}"
        title = f"{n}. {mode}" + (f" — {label}" if label else "")
        text = self.summary.toPlainText()
        strip = self.msg_strip.text()
        if strip:
            text += "\n\n" + strip
        self._report_sections.append(
            view.report_section(title, self.canvas.figure, text))
        self.a_export_session.setEnabled(True)
        self.statusBar().showMessage(
            f"added section {n} to the session report")

    def export_session_report(self):
        if not self._report_sections:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export session report", "circuitinsight_session.html",
            "HTML (*.html)")
        if not path:
            return
        name = Path(str(self.controller.cin_path)).name \
            if self.controller else "session"
        Path(path).write_text(
            view.session_report(f"CircuitInsight — {name}",
                                self._report_sections),
            encoding="utf-8")
        self.statusBar().showMessage(
            f"session report: {len(self._report_sections)} section(s) "
            f"→ {Path(path).name}")

    def export_csv(self):
        if self.result is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export traces", "circuitinsight_traces.csv",
            "CSV (*.csv)")
        if not path:
            return
        Path(path).write_text(view.traces_csv(self.result),
                              encoding="utf-8")
        self.statusBar().showMessage(f"traces → "
                                     f"{Path(path).name}")

    def copy_latex(self):
        """Put the normalized, rounded H(s) on the clipboard as LaTeX --
        the paper-writing verb. (The exact 60-digit form stays on the
        Result for provenance.)"""
        if self.result is None:
            return
        QApplication.clipboard().setText("H(s) = " + view.tf_latex(self.result))
        self.statusBar().showMessage("H(s) copied to the clipboard as LaTeX")

    # --------------------------------------------------------------- export
    def export(self):
        if self.result is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export report", "circuitinsight_report.html",
            "HTML (*.html);;Markdown (*.md)")
        if not path:
            return
        try:
            p = self._write_report(Path(path))
        except Exception as exc:
            QMessageBox.warning(self, "Export failed", f"{type(exc).__name__}: {exc}")
            return
        extra = "" if p.suffix == ".html" else \
            f" and {p.with_suffix('.png').name}"
        self.statusBar().showMessage(f"exported {p.name}{extra}")

    def _write_report(self, p: Path) -> Path:
        """Write the report: single-file HTML (embedded plots), or
        Markdown + Bode PNG."""
        if p.suffix.lower() == ".html":
            p.write_text(view.html_report(self.result), encoding="utf-8")
            return p
        p.write_text(view.markdown_report(self.result), encoding="utf-8")
        self.canvas.figure.savefig(p.with_suffix(".png"), dpi=200,
                                   bbox_inches="tight")
        return p


def build_window(cin=None, psf=None, probe=None) -> MainWindow:
    """Construct the window, optionally preloaded with a CIN + psf (the entry
    the Virtuoso SKILL launcher targets). `probe` preselects the loop-gain
    bench on that vsource; without it the run's own stb designation is
    discovered from the psf header / netlist."""
    win = MainWindow()
    if cin and psf:
        win.open_session(cin, psf, probe=probe)
    return win


def main(argv=None):
    """Compatibility shim: the real entry is gui.launch.main, which shows
    the loading banner before this module's heavy imports are paid for.
    Anyone entering here has already imported them."""
    from .launch import main as _launch

    return _launch(argv)


if __name__ == "__main__":
    sys.exit(main())
