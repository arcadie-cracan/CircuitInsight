"""Everything the main window remembers across restarts and sessions,
factored off MainWindow: QSettings (geometry, budgets, toolbar modes,
recents, per-CIN aliases, the cap model) and the .cistate session
states (rolling autosave + named checkpoints, fingerprint-gated
solutions — see gui/state.py for the on-disk format).

Mixed into MainWindow; owns no widgets and no state of its own beyond
the one `_autosave_warned` latch. `settings_path` stays a MainWindow
class attribute so tests can point the whole family at a temp ini.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QFileDialog, QMessageBox


class PersistenceMixin:
    # ------------------------------------------------------- QSettings
    def _settings(self) -> QSettings:
        if self.settings_path:
            return QSettings(self.settings_path, QSettings.IniFormat)
        return QSettings("CircuitInsight", "desktop")

    def _restore_settings(self):
        s = self._settings()
        geo = s.value("geometry")
        if geo is not None:
            self.restoreGeometry(geo)
        for key, spin in (("budget/mag", self.mag_spin),
                          ("budget/phase", self.phase_spin),
                          ("budget/solve_s", self.budget_spin),
                          ("budget/pm", self.strat_pm_spin),
                          ("budget/gm", self.strat_gm_spin),
                          ("budget/rej", self.strat_rej_spin)):
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
        # the newer toolbar controls were NOT persisted while the older
        # mag/phase pair was -- the two persistence layers had drifted
        for key, combo in (("ui/form", self.form_combo),
                           ("ui/strategy", self.strategy_combo)):
            v = s.value(key)
            if v and combo.findText(str(v)) >= 0:
                combo.setCurrentText(str(v))
        # cross-probe: restoring "on" re-attempts the connection, and the
        # toggle handler already un-checks itself with a reason when Virtuoso
        # is not there -- so a remembered "on" is safe with no session running
        if str(s.value("ui/xprobe", "false")).lower() in ("true", "1"):
            self.a_xprobe.setChecked(True)
        self._rebuild_recents()

    def closeEvent(self, event):
        self._autosave_state()          # the last-state survives the close
        s = self._settings()
        s.setValue("geometry", self.saveGeometry())
        for name, sp in (("h_split", self.h_split),
                         ("left_split", self.left_split),
                         ("right_split", self.right_split)):
            s.setValue("splitters/" + name, sp.saveState())
        s.setValue("budget/mag", self.mag_spin.value())
        s.setValue("budget/phase", self.phase_spin.value())
        s.setValue("budget/solve_s", self.budget_spin.value())
        s.setValue("budget/pm", self.strat_pm_spin.value())
        s.setValue("budget/gm", self.strat_gm_spin.value())
        s.setValue("budget/rej", self.strat_rej_spin.value())
        s.setValue("ui/form", self.form_combo.currentText())
        s.setValue("ui/strategy", self.strategy_combo.currentText())
        s.setValue("budget/band_lo", self.band_slider.values()[0])
        s.setValue("budget/band_hi", self.band_slider.values()[1])
        s.setValue("ui/mode", self.mode_combo.currentText())
        s.setValue("ui/xprobe", self.a_xprobe.isChecked())
        s.sync()
        super().closeEvent(event)

    # --------------------------------------------------------- recents
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

    # ---------------------------------------------------------- states
    def _state_manifest(self) -> dict:
        from . import state as st

        c = self.controller
        return {
            "cin": self._cin, "psf": self._psf,
            "cap_model": self.cap_model,
            "matches": [list(g) for g in self._match_groups],
            "circuit_state": getattr(c, "circuit_state", "as imported"),
            "in": self.in_combo.currentText(),
            "out": self.out_combo.currentText(),
            "mode": self.mode_combo.currentText(),
            "probe": self.probe_combo.currentText(),
            "form": self.form_combo.currentText(),
            "strategy": self.strategy_combo.currentText(),
            "mag_db": self.mag_spin.value(),
            "phase_deg": self.phase_spin.value(),
            "pm_deg": self.strat_pm_spin.value(),
            "gm_db": self.strat_gm_spin.value(),
            "rej_db": self.strat_rej_spin.value(),
            "band": list(self.band_slider.values()),
            "keep": self.checked_keep(),
            "aliases": dict(getattr(c, "sym_aliases", {}) or {}),
            "fingerprint": st.fingerprint(
                self._cin, self._psf, self.cap_model,
                self._match_groups,
                getattr(c, "circuit_state", "as imported")),
        }

    def _autosave_state(self):
        """The rolling last-state, beside the CIN. Fired after every
        shown result and on close — losing a carefully built keep set
        to a crash or an absent-minded close is the failure this
        prevents. Must never break the session itself."""
        if self.controller is None or not getattr(self, "_cin", None):
            return
        try:
            from . import state as st

            st.save_state(st.state_path(self._cin),
                          self._state_manifest(), self.result)
            self._autosave_warned = False
        except Exception as exc:
            # the whole point of this path is "never lose the user's
            # work" -- a silent permanent failure is the one outcome
            # worse than no autosave at all. Say it once, keep working.
            if not getattr(self, "_autosave_warned", False):
                self._autosave_warned = True
                self.log(f"autosave FAILED: {type(exc).__name__}: {exc}")
                self._set_strip("state autosave failed — File → Save "
                                "state as… still works; see the Log",
                                "warn")

    def save_state_dialog(self):
        if self.controller is None:
            return
        from . import state as st

        start = str(st.state_path(self._cin, "checkpoint"))
        path, _ = QFileDialog.getSaveFileName(
            self, "Save session state", start,
            "CircuitInsight state (*.cistate)")
        if not path:
            return
        st.save_state(path, self._state_manifest(), self.result)
        self._set_strip(f"state saved: {Path(path).name}", "info")

    def load_state_dialog(self):
        if self.controller is None:
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Load session state",
            str(Path(self._cin).parent),
            "CircuitInsight state (*.cistate)")
        if path:
            self._load_state_file(path)

    def _load_state_file(self, path=None):
        """Restore a state: the selections always; the stored solution
        only when its fingerprint matches the run now open — a stale
        solution is declared stale, never displayed as current."""
        if self.controller is None:
            return
        from . import state as st

        if path is None:
            path = st.state_path(self._cin)
        try:
            fp = st.fingerprint(
                self._cin, self._psf, self.cap_model,
                self._match_groups,
                getattr(self.controller, "circuit_state", "as imported"))
            manifest, result, stale = st.load_state(path, fp)
        except Exception as exc:
            self._set_strip(f"state load failed: {exc}", "error")
            return
        # selections, in dependency order: matches change the symbol
        # set, mode/form refill tables, ticks land last
        groups = [tuple(g) for g in manifest.get("matches", [])]
        if groups:
            self._match_groups = groups
            self._apply_matches()
        for combo, key in ((self.in_combo, "in"),
                           (self.out_combo, "out")):
            v = manifest.get(key)
            if v:
                combo.setCurrentText(v)
        self.mode_combo.setCurrentText(manifest.get("mode", "Transfer"))
        pv = manifest.get("probe")
        if pv and self.probe_combo.findText(pv) >= 0:
            self.probe_combo.setCurrentText(pv)
        self.form_combo.setCurrentText(manifest.get("form", "Exact"))
        sv = manifest.get("strategy")
        if sv and self.strategy_combo.findText(sv) >= 0:
            self.strategy_combo.setCurrentText(sv)
        for spin, key in ((self.mag_spin, "mag_db"),
                          (self.phase_spin, "phase_deg"),
                          (self.strat_pm_spin, "pm_deg"),
                          (self.strat_gm_spin, "gm_db"),
                          (self.strat_rej_spin, "rej_db")):
            if key in manifest:
                spin.setValue(float(manifest[key]))
        band = manifest.get("band")
        if band and len(band) == 2:
            self.band_slider.setValues(float(band[0]), float(band[1]))
        aliases = manifest.get("aliases") or {}
        if aliases and hasattr(self.controller, "sym_aliases"):
            self.controller.sym_aliases.update(aliases)
        checked = list(manifest.get("keep", []))
        keep_ok = True
        if checked:
            try:
                ranking = self.controller.rank_symbols(*self._io())
                self._fill_keep_table(ranking, checked=checked)
            except Exception as exc:
                keep_ok = False
                self.log(f"state restore: keep table failed "
                         f"({type(exc).__name__}: {exc})")
        name = Path(path).name
        if result is not None:
            self._show(result)
            partial = ("" if keep_ok else
                       " — KEEP TICKS NOT RESTORED (see the Log)")
            self._set_strip(f"state restored from {name} — selections "
                            f"AND the computed solution (fingerprint "
                            f"matched){partial}",
                            "info" if keep_ok else "warn")
            self.log(f"state restored: {name} (with solution)")
        else:
            why = ("solution stale — the run, cap model, matches or "
                   "version changed" if stale else "no solution stored")
            self._set_strip(f"state restored from {name} — selections "
                            f"only ({why}); press Solve", "warn")
            self.log(f"state restored: {name} (selections only)")
            self._refresh_solve_hint()

    # ------------------------------------------------ cap model, aliases
    def _save_cap_model(self):
        s = self._settings()
        s.setValue("cap_model", self.cap_model)
        s.sync()

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
                # a corrupt alias entry silently reverting every LaTeX
                # override would be invisible -- say so once in the Log
                self.log("saved LaTeX aliases could not be parsed; "
                         "using defaults")
