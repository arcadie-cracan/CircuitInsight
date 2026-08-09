"""The worker-thread machinery, factored off MainWindow: the cancel
exception, the generic result worker, and the JobRunnerMixin that owns
launching, progress, phases, the Log, cancellation, and the solve-time
calibration loop.

The mixin reads MainWindow's widgets (progress bar, cancel button, the
Log tab, the action buttons) and its estimate state (`_est_s`,
`_live_est`, `_run_est`); display handlers (`_on_done`, `_on_failed`,
`_backend_note`, `_update_estimate`) stay on the window and resolve
through self.
"""
from __future__ import annotations

import time

from PySide6.QtCore import QThread, Signal

from . import exprweb, view


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


class JobRunnerMixin:
    #: sentinel: "price this launch with the keep-set estimate"
    _KEEP_EST = object()

    def _cancel_solve(self):
        if self._thread is not None:
            self._thread.cancel()
            self.cancel_btn.setEnabled(False)
            self.progress.setFormat("cancelling…")

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
        # warm the display transforms IN THE WORKER: rendering the
        # expression lines runs cancel/nsimplify over the whole result
        # (measured 3 s at 1040 terms, tens of seconds on big hybrids)
        # and used to freeze the GUI the moment the solve delivered
        base = not self.fullnames_chk.isChecked()
        wrap = bool(getattr(exprweb, "WEBENGINE", False))
        aliases = dict(getattr(self.controller, "sym_aliases", {}) or {})

        def _prepped(cb, _fn=fn):
            res = _fn(cb)
            view.prepare_display(res, base=base, wrap=wrap,
                                 aliases=aliases)
            return res

        self._thread = _Worker(_prepped)
        self._thread.progress.connect(self._on_progress)
        self._thread.done.connect(on_done or self._on_done)
        self._thread.failed.connect(self._on_failed)
        self._thread.cancelled.connect(self._on_cancelled)
        # worker-side narration (pursuit rounds and the like) lands in
        # the Log through the queued signal -- never touch widgets there
        self._thread.note.connect(lambda m: self.log(f"  {m}"))
        self.worker_note = self._thread.note.emit
        self._thread.start()

    #: every launch disables these; every completion path re-enables ALL
    #: of them. Four handlers each re-enabled their own subset, so a
    #: Modes/Compensate/GFT run left Simplify and Reduce greyed until an
    #: unrelated mode change happened to fix them.
    def _job_finished(self, *extra):
        for b in (self.solve_btn, self.a_solve, self.a_simplify,
                  self.a_reduce, *extra):
            b.setEnabled(True)

    def _run_bg(self, fn, on_done):
        """Advisor-pattern short worker: controller math off the GUI
        thread (estimates, rank, the certificate — each measured
        0.3–1 s of main-thread stall per interaction), result delivered
        queued. Failures are dropped: advisories never break a session."""
        w = _Worker(lambda _cb: fn())
        if not hasattr(self, "_bg_workers"):
            self._bg_workers = []
        self._bg_workers.append(w)

        def _done(res, _w=w):
            try:
                on_done(res)
            finally:
                if _w in self._bg_workers:
                    self._bg_workers.remove(_w)

        w.done.connect(_done)
        w.failed.connect(lambda _m, _w=w: (
            self._bg_workers.remove(_w)
            if _w in self._bg_workers else None))
        w.start()

    # ------------------------------------------------------ log, phases
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

    # ----------------------------------------------- solve-time learning
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
