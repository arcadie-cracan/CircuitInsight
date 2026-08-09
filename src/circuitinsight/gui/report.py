"""Session-report and export verbs, factored off the main window.

Pure delegation to gui.view's renderers: these methods own only the
file dialogs, the status-bar acknowledgements and the report-section
list. They read the window's `controller`, `result`, `summary`,
`msg_strip`, `canvas` and `_report_sections` attributes and are mixed
into MainWindow — no state of their own.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox

from . import view


class ReportMixin:
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
