"""PySide6 desktop front end. A thin window over `SessionController` + `view`:
open a CIN + psf, declare matched pairs, pick input/output, choose a keep-set
(ranked by band sensitivity, gated by a solve-time estimate), solve or
error-budget-simplify (in a worker thread), and export a report.
Requires `circuitinsight[gui]` (PySide6).
"""
from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import QEvent, QSettings, Qt, QThread, Signal
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QCheckBox, QComboBox, QDoubleSpinBox,
    QFileDialog, QGroupBox, QHBoxLayout, QLabel, QMainWindow, QMessageBox,
    QPushButton, QProgressBar, QScrollArea, QSplitter, QTableWidget,
    QTableWidgetItem, QTabWidget, QTextEdit, QToolBar, QVBoxLayout, QWidget,
)
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg, NavigationToolbar2QT)
from matplotlib.figure import Figure

from ..session import SessionController
from . import exprweb, theme, view


class _Cancelled(Exception):
    """Raised inside the worker's progress callback to abandon a solve."""


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


class MainWindow(QMainWindow):
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
        openb = QPushButton("Open CIN + psf…")
        openb.clicked.connect(self.open_dialog)
        self.exportb = QPushButton("Export…")
        self.exportb.setEnabled(False)
        self.exportb.clicked.connect(self.export)
        self.in_combo = QComboBox()
        self.out_combo = QComboBox()
        self.solve_btn = QPushButton("Solve")
        self.solve_btn.setEnabled(False)
        self.solve_btn.clicked.connect(self.solve)
        self.simplify_btn = QPushButton("Simplify")
        self.simplify_btn.setEnabled(False)
        self.simplify_btn.clicked.connect(self.simplify)
        self.reduce_btn = QPushButton("Reduce")
        self.reduce_btn.setToolTip(
            "Reduce model ORDER: keep only the reactances that shape H(s) within "
            "the dB budget (in-band), then simplify -> the textbook low-order form")
        self.reduce_btn.setEnabled(False)
        self.reduce_btn.clicked.connect(self.reduce)
        self.mag_spin = self._spin(1.0, 0.0, 20.0, 0.1, " dB")
        self.phase_spin = self._spin(5.0, 0.0, 90.0, 0.5, " °")

        # A QToolBar, not a raw QHBoxLayout: overflow chevron for free at
        # narrow widths, and the matplotlib navigation bar (itself a
        # QToolBar) shares the row instead of costing a second one.
        self.in_combo.setMinimumWidth(130)
        self.out_combo.setMinimumWidth(130)
        tb = QToolBar("main")
        tb.setObjectName("main_toolbar")
        tb.setMovable(False)
        for wdg in (openb, self.exportb):
            tb.addWidget(wdg)
        tb.addSeparator()
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(
            ["Transfer", "Loop gain", "Compensate", "Modes", "GFT",
             "Impedance"])
        self.mode_combo.currentTextChanged.connect(self._on_mode_changed)
        tb.addWidget(self.mode_combo)
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
        tb.addWidget(self.simplify_btn)
        tb.addWidget(self.reduce_btn)
        tb.addSeparator()
        tb.addWidget(QLabel(" budget: "))
        tb.addWidget(self.mag_spin)
        tb.addWidget(self.phase_spin)
        self.toolbar = tb

        left = QSplitter(Qt.Vertical)
        left.setObjectName("left_split")
        left.addWidget(self._devices_group())
        left.addWidget(self._keepset_group())
        left.addWidget(self._history_group())
        left.setSizes([280, 340, 120])
        self.left_split = left

        self.canvas = FigureCanvasQTAgg(Figure(figsize=(5.2, 4.0)))
        self.summary = QTextEdit()
        self.summary.setReadOnly(True)
        self.summary.setLineWrapMode(QTextEdit.NoWrap)
        # Expression surface: the KaTeX web view when QtWebEngine is present --
        # crisp vector math, hover = OP value, click = symbolClicked (the future
        # cross-probe handle). Falls back to the matplotlib mathtext canvas on a
        # PySide6 install without the WebEngine addon.
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
        expr_tab = QWidget()
        _ev = QVBoxLayout(expr_tab)
        _ev.setContentsMargins(0, 0, 0, 0)
        _ev.setSpacing(2)
        _ev.addWidget(self.fullnames_chk)
        _ev.addWidget(expr_surface)
        # Two overlapping Bode curves cannot show a 0.5 dB residual. The Error
        # tab is where "how good is this model" actually becomes readable.
        self.err_canvas = FigureCanvasQTAgg(Figure(figsize=(5.2, 3.0)))
        tabs = QTabWidget()
        tabs.addTab(self.summary, "Summary")
        tabs.addTab(expr_tab, "Expression")
        tabs.addTab(self.err_canvas, "Error")
        tabs.addTab(self._whatif_page(), "What-if")
        tabs.addTab(self._comp_page(), "Compensation")
        tabs.addTab(self._gft_page(), "GFT")
        self.tabs = tabs

        right = QSplitter(Qt.Vertical)
        right.setObjectName("right_split")
        right.addWidget(self.canvas)
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

        self.addToolBar(tb)
        self.addToolBar(NavigationToolbar2QT(self.canvas, self))

        # Persistent, color-coded message strip: advisories and failures
        # live here instead of interrupting with modal dialogs.
        self.msg_strip = QLabel()
        self.msg_strip.setWordWrap(True)
        self.msg_strip.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.msg_strip.hide()

        outer = QVBoxLayout()
        outer.addWidget(self.msg_strip)
        outer.addWidget(split, 1)
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
        mode = s.value("ui/mode")
        if mode and self.mode_combo.findText(str(mode)) >= 0:
            self.mode_combo.setCurrentText(str(mode))
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
        s.setValue("ui/mode", self.mode_combo.currentText())
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
        self.a_simplify = m_an.addAction("Sim&plify")
        self.a_simplify.setShortcut("Ctrl+Shift+Return")
        self.a_simplify.triggered.connect(self.simplify)
        self.a_simplify.setEnabled(False)
        self.a_reduce = m_an.addAction("&Reduce order")
        self.a_reduce.setShortcut("Ctrl+R")
        self.a_reduce.triggered.connect(self.reduce)
        self.a_reduce.setEnabled(False)
        m_an.addSeparator()
        self.a_rank = m_an.addAction("&Rank symbols")
        self.a_rank.setShortcut("F5")
        self.a_rank.triggered.connect(self._rank)
        self.a_suggest = m_an.addAction("Suggest keep-set ≤ &budget")
        self.a_suggest.triggered.connect(self._suggest_keep)

        m_dev = mb.addMenu("&Devices")
        m_dev.setTearOffEnabled(True)
        m_dev.addAction("Suggest &matched pairs").triggered.connect(
            self.suggest_matches)
        m_dev.addAction("Match &selection").triggered.connect(self.match_selected)
        m_dev.addSeparator()
        m_dev.addAction("&Clear matches").triggered.connect(self.clear_matches)
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

    def _spin(self, val, lo, hi, step, suffix):
        s = QDoubleSpinBox()
        s.setRange(lo, hi)
        s.setSingleStep(step)
        s.setValue(val)
        s.setSuffix(suffix)
        return s

    def _devices_group(self):
        box = QGroupBox("Devices / matched pairs")
        self.devices = QTableWidget(0, 6)
        self.devices.setHorizontalHeaderLabels(
            ["device", "type", "region", "gm", "gds", "LaTeX"])
        self.devices.itemChanged.connect(self._on_alias_edited)
        self.devices.horizontalHeader().setStretchLastSection(True)
        # editable per item: only the LaTeX cell carries ItemIsEditable
        # (set in _populate); all other cells stay read-only
        self.devices.setEditTriggers(QAbstractItemView.AllEditTriggers)
        self.devices.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.devices.setSelectionMode(QAbstractItemView.ExtendedSelection)

        sugg = QPushButton("Suggest")
        sugg.clicked.connect(self.suggest_matches)
        matchsel = QPushButton("Match sel.")
        matchsel.clicked.connect(self.match_selected)
        unmatch = QPushButton("Unmatch")
        unmatch.setToolTip("Remove the selected match group from the list")
        unmatch.clicked.connect(self.unmatch_selected)
        clr = QPushButton("Clear")
        clr.clicked.connect(self.clear_matches)
        row = QHBoxLayout()
        for wdg in (sugg, matchsel, unmatch, clr):
            row.addWidget(wdg)
        from PySide6.QtWidgets import QListWidget

        self.matches_list = QListWidget()
        self.matches_list.setMaximumHeight(64)
        self.matches_list.setToolTip(
            "Matched groups share one symbol; select a group and Unmatch "
            "to dissolve it")

        v = QVBoxLayout()
        v.addWidget(self.devices, 1)
        v.addLayout(row)
        v.addWidget(self.matches_list)
        box.setLayout(v)
        return box

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
        suggestb = QPushButton("Suggest ≤ budget")
        suggestb.clicked.connect(self._suggest_keep)
        ctl = QHBoxLayout()
        ctl.addWidget(rankb)
        ctl.addWidget(self.group_chk)
        ctl.addWidget(QLabel("budget:"))
        ctl.addWidget(self.budget_spin)
        ctl.addWidget(suggestb)

        self.estimate_lbl = QLabel("estimate: —")

        v = QVBoxLayout()
        v.addWidget(self.keep_filter)
        v.addWidget(self.keep_tbl, 1)
        v.addLayout(ctl)
        v.addWidget(self.estimate_lbl)
        box.setLayout(v)
        return box

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
        box.setLayout(v)
        return box

    def _history_label(self, r) -> str:
        keep = r.keep
        try:
            nkeep = "ALL" if not isinstance(keep, list) else str(len(keep))
        except Exception:
            nkeep = "?"
        extra = (f"  PM {r.pm_deg:.1f}°" if r.pm_deg is not None
                 else f"  {r.dc_gain_db:.1f} dB")
        return f"{r.inp} → {r.out}  [keep {nkeep}]{extra}"

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
        # GFT needs the input/output designations alongside the probe
        io_on = mode in ("Transfer", "GFT")
        self.out_combo.setEnabled(io_on)
        self.in_combo.setEnabled(io_on)
        # Simplify/Reduce belong to the Transfer workflow
        for b in (self.simplify_btn, self.reduce_btn,
                  self.a_simplify, self.a_reduce):
            b.setEnabled(mode == "Transfer" and self.controller is not None)
        if probed and self.controller is not None \
                and self.probe_combo.count() == 0:
            self.probe_combo.addItems(self.controller.probes)
        if mode == "Modes" and self.controller is not None \
                and self.probe2_combo.count() == 0:
            self.probe2_combo.addItems(self.controller.probes)
            if self.probe2_combo.count() > 1:
                self.probe2_combo.setCurrentIndex(1)
        if mode == "Compensate":
            self.tabs.setCurrentWidget(self._comp_tab)
        if mode == "GFT":
            self.tabs.setCurrentWidget(self._gft_tab)

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
        ax1.legend(fontsize=8, frameon=False, loc="lower left")
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
    def _comp_page(self):
        page = QWidget()
        v = QVBoxLayout(page)
        row = QHBoxLayout()
        row.addWidget(QLabel("goal:"))
        self.goal_combo = QComboBox()
        self.goal_combo.addItems(["mfm", "pm"])
        row.addWidget(self.goal_combo)
        row.addWidget(QLabel("PM target:"))
        self.pm_spin = self._spin(60.0, 30.0, 85.0, 1.0, " °")
        row.addWidget(self.pm_spin)
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
        v.addWidget(self._comp_hint)
        self._comp_suggestions = []
        self._comp_probe = None
        self._comp_upd = None
        self._comp_tab = page
        return page

    def suggest_comp(self):
        if self.controller is None:
            return
        probe = self.probe_combo.currentText()
        if not probe:
            self.statusBar().showMessage("no loop probe in this design")
            return
        goal = self.goal_combo.currentText()
        pm_t = self.pm_spin.value()

        def fn(_cb):
            baseline = self.controller.loop_gain(probe)
            sugg = self.controller.suggest_compensation(
                probe, goal=goal, pm_target=pm_t)
            return (baseline, sugg)

        self._launch(fn, f"searching compensation at {probe} ({goal})…",
                     on_done=self._on_comp_done)

    def suggest_sync(self, probe, **kw):
        baseline = self.controller.loop_gain(probe)
        sugg = self.controller.suggest_compensation(probe, **kw)
        self._on_comp_done((baseline, sugg))
        return self._comp_suggestions

    def _on_comp_done(self, payload):
        self.progress.hide()
        self.cancel_btn.hide()
        for b in (self.solve_btn, self.a_solve, self.suggest_btn):
            b.setEnabled(True)
        baseline, sugg = payload
        self._comp_baseline = baseline
        self._comp_probe = self.probe_combo.currentText()
        self._comp_upd = None                     # rebuilt lazily on select
        self._comp_suggestions = list(sugg)
        self._show(baseline)
        tbl = self.comp_tbl
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
        if self._comp_upd is None:
            import numpy as np
            from ..analysis.compensate import LoopGainUpdater

            an = self.controller._analyzer_ready()
            self._comp_upd = LoopGainUpdater(
                an.system(self._comp_probe), self._comp_probe,
                np.geomspace(1.0, 1e10, 300))
        return self._comp_upd

    def _on_comp_selected(self):
        import re

        import numpy as np

        rows = {i.row() for i in self.comp_tbl.selectedIndexes()}
        if not rows or not self._comp_suggestions:
            return
        sg = self._comp_suggestions[min(rows)]
        upd = self._comp_updater()

        C, R = sg.C, sg.R

        def Y(s):
            return s * C / (1 + s * R * C)

        branches = [(sg.candidate.node_a, sg.candidate.node_b, Y)]
        m = re.search(r"\[symmetric pair with \(([^,]+), ([^)]+)\)\]",
                      sg.candidate.rationale)
        if m:
            branches.append((m.group(1).strip(), m.group(2).strip(), Y))
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
        try:
            self.open_session(self._cin, self._psf)
        except Exception as exc:
            self._set_strip(f"re-open failed: {exc}", "error")
            return
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
        if had_result:
            try:
                if mode == "Loop gain":
                    self.loop_gain_sync(with_probe)
                elif mode in ("Transfer",):
                    self.solve_sync()
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
        idx = self.mode_combo.findText("Loop gain")
        self.mode_combo.model().item(idx).setEnabled(bool(probes))
        if not probes:
            self.mode_combo.setCurrentText("Transfer")
            self.mode_combo.setToolTip("Loop gain needs an iprobe in the "
                                       "design; none found in this CIN")
        else:
            self.mode_combo.setToolTip("")
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
        # info cells stay read-only under AllEditTriggers; only col 5 edits
        ro = Qt.ItemIsEnabled | Qt.ItemIsSelectable
        self.devices.setRowCount(len(devs))
        from PySide6.QtGui import QBrush, QColor
        for i, d in enumerate(devs):
            name_it = QTableWidgetItem(d.name); name_it.setFlags(ro)
            type_it = QTableWidgetItem(d.device_type); type_it.setFlags(ro)
            if d.name in self._ii:                # II active, no gii modeled
                name_it.setForeground(QBrush(QColor("#8a1c12")))
                name_it.setToolTip(
                    f"impact ionization active: isub/ids = "
                    f"{self._ii[d.name]:.2%}, but no gii modeled -- the "
                    f"first-order gm/gds/gmbs reconstruction is incomplete "
                    f"here (identify gii by AC injection; see r2r)")
            self.devices.setItem(i, 0, name_it)
            self.devices.setItem(i, 1, type_it)
            reg = ""
            if d.device_type == "mosfet":
                reg = view.region_name(
                    c.device_op(d.name).get("region"))
            it = QTableWidgetItem(reg)
            it.setFlags(ro)
            if reg and reg != "sat":
                from PySide6.QtGui import QBrush, QColor
                it.setForeground(QBrush(QColor("#8a1c12")))
            self.devices.setItem(i, 2, it)
            key = d.name.replace(".", "_")
            gm = opv.get(f"gm_{key}")
            gds = opv.get(f"gds_{key}")
            gm_it = QTableWidgetItem(view.eng(gm, "S") if gm is not None else "")
            gds_it = QTableWidgetItem(
                view.eng(gds, "S") if gds is not None else "")
            gm_it.setFlags(ro); gds_it.setFlags(ro)
            self.devices.setItem(i, 3, gm_it)
            self.devices.setItem(i, 4, gds_it)
            al = QTableWidgetItem(c.sym_aliases.get(d.name, ""))
            al.setFlags(ro | Qt.ItemIsEditable)
            al.setToolTip("LaTeX subscript for this device (e.g. M_1, "
                          "M_{in}); blank = default. Passives take the whole "
                          "symbol (e.g. R_S). A per-symbol LaTeX (keep table) "
                          "overrides this.")
            self.devices.setItem(i, 5, al)
        self._filling = False
        self.keep_tbl.setRowCount(0)
        self.estimate_lbl.setText("estimate: —")
        self._refresh_matches_label()
        for b in (self.solve_btn, self.simplify_btn, self.reduce_btn,
                  self.exportb, self.a_solve, self.a_simplify,
                  self.a_reduce, self.a_export):
            b.setEnabled(True)
        for b in (self.exportb, self.a_export):           # nothing solved yet
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
            groups = [tuple(g) for g in c.suggest_matches()]
            if groups:
                self._match_groups = groups
                self.controller.set_matches(*groups)
                self._refresh_matches_label()
            plan = c.suggest_keep(inp, out, self.budget_spin.value())
            ranking = c.rank_symbols(inp, out)
            self._fill_keep_table(ranking, checked=list(plan.keep))
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

    # ------------------------------------------------------------- matches
    def _apply_matches(self):
        self.controller.set_matches(*self._match_groups)
        self._refresh_matches_label()
        self.keep_tbl.setRowCount(0)                       # ranking is now stale
        self.estimate_lbl.setText("estimate: — (re-Rank)")

    #: light, colorblind-friendly row tints for match groups
    _GROUP_TINTS = ("#dbe9f6", "#fbe4d5", "#e2efda", "#ece1f0",
                    "#fff2cc", "#dbeef4")

    def _refresh_matches_label(self):
        from PySide6.QtGui import QBrush, QColor

        self.matches_list.clear()
        for g in self._match_groups:
            self.matches_list.addItem(" = ".join(g))
        by_dev = {}
        for gi, g in enumerate(self._match_groups):
            for name in g:
                by_dev[name] = gi
        was, self._filling = self._filling, True   # setBackground fires itemChanged
        try:
            for i in range(self.devices.rowCount()):
                name = self.devices.item(i, 0).text()
                gi = by_dev.get(name)
                brush = (QBrush(QColor(self._GROUP_TINTS[gi %
                                                         len(self._GROUP_TINTS)]))
                         if gi is not None else QBrush())
                for jcol in range(self.devices.columnCount()):
                    it = self.devices.item(i, jcol)
                    if it is not None:
                        it.setBackground(brush)
        finally:
            self._filling = was

    def unmatch_selected(self):
        rows = sorted({i.row() for i in self.matches_list.selectedIndexes()},
                      reverse=True)
        if not rows:
            self.statusBar().showMessage("select a match group to remove")
            return
        for r in rows:
            if 0 <= r < len(self._match_groups):
                del self._match_groups[r]
        self._apply_matches()

    def suggest_matches(self):
        if self.controller is None:
            return
        self._match_groups = [tuple(g) for g in self.controller.suggest_matches()]
        self._apply_matches()

    def match_selected(self):
        rows = sorted({i.row() for i in self.devices.selectedIndexes()})
        names = tuple(self.devices.item(r, 0).text() for r in rows)
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
        self._fill_keep_table(ranking)

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

    def _on_alias_edited(self, item):
        if self._filling or item.column() != 5 or self.controller is None:
            return
        name = self.devices.item(item.row(), 0).text()
        text = item.text().strip()
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

    def _update_estimate(self):
        if self.controller is None:
            return
        inp, out = self._io()
        try:
            est = self.controller.estimate(inp, out, self.checked_keep())
            self.estimate_lbl.setText(f"estimate: {est}")
            secs = getattr(est, "seconds", None)
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
        if self.controller is None:
            return
        inp, out = self._io()
        try:
            plan = self.controller.suggest_keep(inp, out, self.budget_spin.value())
            ranking = self.controller.rank_symbols(inp, out)
        except Exception as exc:
            QMessageBox.warning(self, "Suggest failed", f"{type(exc).__name__}: {exc}")
            return
        self._fill_keep_table(ranking, checked=list(plan.keep))
        self.statusBar().showMessage(
            f"suggested keep-set ({len(list(plan.keep))} symbols): {list(plan.keep)}")

    # ---------------------------------------------------------------- solve
    def _launch(self, fn, label, on_done=None):
        for b in (self.solve_btn, self.simplify_btn, self.reduce_btn,
                  self.a_solve, self.a_simplify, self.a_reduce):
            b.setEnabled(False)
        self.statusBar().showMessage(label)
        self.progress.setRange(0, 0)      # busy until the solver reports a size
        self.progress.setFormat("preparing…")
        self.progress.show()
        self.cancel_btn.setEnabled(True)
        self.cancel_btn.show()
        self._thread = _Worker(fn)
        self._thread.progress.connect(self._on_progress)
        self._thread.done.connect(on_done or self._on_done)
        self._thread.failed.connect(self._on_failed)
        self._thread.cancelled.connect(self._on_cancelled)
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
        if self.mode_combo.currentText() == "Loop gain":
            probe = self.probe_combo.currentText()
            if not probe:
                self.statusBar().showMessage("no loop probe in this design")
                return
            keep = self.checked_keep()
            # session.loop_gain has no progress plumbing (the numeric path
            # is fast); the busy bar covers the hybrid case
            self._launch(lambda cb: self.controller.loop_gain(probe,
                                                              keep=keep),
                         f"loop gain at {probe} …")
            return
        inp, out, keep = *self._io(), self.checked_keep()
        self._launch(lambda cb: self.controller.solve(inp, out, keep,
                                                      progress=cb),
                     f"solving {inp} → {out} …")

    def simplify(self):
        if self.controller is None:
            return
        inp, out, keep = *self._io(), self.checked_keep()
        mag, ph = self.mag_spin.value(), self.phase_spin.value()
        self._launch(
            lambda cb: self.controller.simplify(inp, out, keep, mag_db=mag,
                                                phase_deg=ph),
            f"simplifying {inp} → {out} within {mag} dB / {ph}° …")

    def reduce(self):
        if self.controller is None:
            return
        inp, out, keep = *self._io(), self.checked_keep()
        mag, ph = self.mag_spin.value(), self.phase_spin.value()
        # the dB budget doubles as the reactance-reduction tolerance
        self._launch(
            lambda cb: self.controller.reduce_solve(
                inp, out, keep, tol_db=mag, mag_db=mag, phase_deg=ph),
            f"reducing {inp} → {out} within {mag} dB (in-band) …")

    def solve_sync(self):
        inp, out = self._io()
        self._show(self.controller.solve(inp, out, self.checked_keep()))
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
        self._show(self.controller.simplify(
            inp, out, self.checked_keep(),
            mag_db=self.mag_spin.value(), phase_deg=self.phase_spin.value()))
        return self.result

    def reduce_sync(self):
        inp, out = self._io()
        self._show(self.controller.reduce_solve(
            inp, out, self.checked_keep(),
            tol_db=self.mag_spin.value(), mag_db=self.mag_spin.value(),
            phase_deg=self.phase_spin.value()))
        return self.result

    def _on_progress(self, done, total):
        """Grid points evaluated. Queued from the worker thread, so this runs on
        the GUI thread and may touch widgets.

        The grid is not the whole solve -- setup precedes it and the tensor
        reconstruction follows -- so the bar is honest about which phase it is
        in rather than hitting 100% and then appearing to hang. The grid's share
        grows with the keep set (10% at 4 symbols, 55% at 7), i.e. it dominates
        exactly the solves slow enough to need a progress bar.
        """
        if total <= 0:
            return
        if done >= total:                       # grid done; reconstruction left
            self.progress.setRange(0, 0)        # back to busy
            self.progress.setFormat("reconstructing…")
            return
        self.progress.setRange(0, total)
        self.progress.setValue(done)
        self.progress.setFormat(f"grid %p%  ({done}/{total})")

    def _on_cancelled(self):
        self.progress.hide()
        self.cancel_btn.hide()
        for b in (self.solve_btn, self.simplify_btn, self.reduce_btn,
                  self.a_solve, self.a_simplify, self.a_reduce):
            b.setEnabled(True)
        self.statusBar().showMessage("solve cancelled")

    def _on_done(self, result):
        self.progress.hide()
        self.cancel_btn.hide()
        self._show(result)
        if result.out.startswith("T@") and self.controller is not None:
            self._start_advisor(result.inp)
        for b in (self.solve_btn, self.simplify_btn, self.reduce_btn,
                  self.a_solve, self.a_simplify, self.a_reduce):
            b.setEnabled(True)
        self.statusBar().showMessage(
            f"{result.inp} → {result.out}:  {result.dc_gain_db:.2f} dB, "
            f"{result.n_terms} terms")

    def _on_failed(self, msg):
        self.progress.hide()
        self.cancel_btn.hide()
        for b in (self.solve_btn, self.simplify_btn, self.reduce_btn,
                  self.a_solve, self.a_simplify, self.a_reduce):
            b.setEnabled(True)
        self.statusBar().showMessage("solve failed")
        self._set_strip("solve failed: " + msg, "error")

    def _show(self, result, overlays=()):
        self.result = result
        view.bode_figure(result, self.canvas.figure, overlays=overlays)
        theme.style_figure(self.canvas.figure)
        self.canvas.draw_idle()
        if not self._showing_from_history:
            self._push_history(result)
            if result.warnings:
                self._set_strip("⚠ "
                                + "   ".join(result.warnings), "warn")
            elif not result.out.startswith("T@"):
                self._clear_strip()
        # The expanded H(s), rounded. The exact form is a ratio of 60-digit
        # integers -- kept on the Result for provenance, never shown raw.
        self.summary.setPlainText(
            view.summary_text(result) + "\n\nH(s):\n" + view.tf_latex(result))
        try:
            view.error_figure(result, self.err_canvas.figure)
            theme.style_figure(self.err_canvas.figure)
            self.err_canvas.draw_idle()
        except Exception:
            self.err_canvas.figure.clear()
            self.err_canvas.draw_idle()
        self._render_expr()
        self._rebuild_whatif(result)
        for b in (self.exportb, self.a_export, self.a_copy_tex,
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
                self.exprweb.set_payload(
                    view.expr_katex(self.result, base=base, aliases=aliases))
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
        table's identifier. (Also the handle a schematic cross-probe will use.)"""
        for r in range(self.keep_tbl.rowCount()):
            it = self.keep_tbl.item(r, 0)
            if it is not None and it.text() == name:
                self.keep_tbl.selectRow(r)
                self.keep_tbl.scrollToItem(it)
                break

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
    import argparse

    raw = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(
        prog="circuitinsight-gui",
        description="CircuitInsight desktop app. With --cin/--psf it opens "
                    "preloaded (used by the Virtuoso one-click launcher).")
    ap.add_argument("--cin", help="CIN topology file to open on startup")
    ap.add_argument("--psf", help="psf results directory to pair with --cin")
    ap.add_argument("--probe", help="preselect the loop-gain bench on this "
                                    "vsource (default: the run's own stb "
                                    "designation, when discoverable)")
    ap.add_argument("--theme", choices=("cadence", "native"), default="cadence",
                    help="'cadence' blends with Virtuoso's windows (default); "
                         "'native' leaves your desktop's own Qt style alone")
    args, _ = ap.parse_known_args(raw)

    app = QApplication.instance() or QApplication(sys.argv[:1])
    if args.theme == "cadence":
        theme.apply(app)
    win = build_window(args.cin, args.psf, probe=args.probe)
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
