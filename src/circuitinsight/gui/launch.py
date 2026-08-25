"""The fast path in: a loading banner before the heavy imports.

Entering through gui.app pays for matplotlib + sympy + the session
layer before anything is on screen, and a preloaded open adds matches,
ranking and the first-light solve on top -- several blank seconds. This
module imports NOTHING heavy at module level (not even Qt: everything
is inside the functions), so the console script reaches main() almost
instantly, shows the banner, and only then pulls the machinery in,
narrating the stages as it goes.
"""
from __future__ import annotations

import sys


def _version() -> str:
    """The running build's version, cheaply: the installed distribution's
    the package constant first -- it lives in the tree that is
    actually RUNNING, so an editable install shows what it executes
    (dist-info is stamped at pip-install time and once kept a splash
    at v0.1.1 while the dev tree ran 0.2.0); pip's metadata is the
    fallback. Never raises -- a banner must not be the thing that
    fails to start."""
    try:
        import circuitinsight

        return "v" + circuitinsight.__version__
    except Exception:
        pass
    try:
        from importlib.metadata import version

        return "v" + version("circuitinsight")
    except Exception:
        return ""


def _splash():
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QColor, QFont, QPainter, QPixmap
    from PySide6.QtWidgets import QSplashScreen

    pm = QPixmap(440, 200)
    pm.fill(QColor("#f7f7f7"))
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    # explicit BASELINES, not nested rects: the version must sit on the
    # wordmark's own baseline, and rect-relative alignment put it a line
    # lower. Which build is running is the first question a bug report
    # needs, and the banner is the screen every session starts on.
    f = QFont()
    f.setPointSize(21)
    f.setBold(True)
    p.setFont(f)
    p.setPen(QColor("#1d3a56"))
    base_y = 78
    p.drawText(26, base_y, "CircuitInsight")
    x_ver = 26 + p.fontMetrics().horizontalAdvance("CircuitInsight") + 10
    fv = QFont()
    fv.setPointSize(11)
    p.setFont(fv)
    p.setPen(QColor("#7a7a7a"))
    p.drawText(x_ver, base_y, _version())
    f2 = QFont()
    f2.setPointSize(9)
    p.setFont(f2)
    p.setPen(QColor("#555555"))
    p.drawText(26, base_y + 30,
               "operating-point-driven symbolic circuit analysis")
    p.setPen(Qt.NoPen)
    p.setBrush(QColor("#a80000"))                # the Virtuoso red rule
    p.drawRect(0, pm.height() - 3, pm.width(), 3)
    # a thin dark frame: a splash has no window chrome, so on a light
    # desktop the pale banner otherwise melts into whatever is behind it
    p.setBrush(Qt.NoBrush)
    p.setPen(QColor("#3c3c3c"))
    p.drawRect(0, 0, pm.width() - 1, pm.height() - 1)
    p.end()
    sp = QSplashScreen(pm, Qt.WindowStaysOnTopHint)
    sp._align = Qt.AlignBottom | Qt.AlignLeft    # reused by _stage
    return sp


def _stage(app, splash, text: str) -> None:
    from PySide6.QtGui import QColor

    splash.showMessage("  " + text, splash._align, QColor("#555555"))
    app.processEvents()


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
    ap.add_argument("--theme", choices=("cadence", "native"),
                    default="cadence",
                    help="'cadence' blends with Virtuoso's windows "
                         "(default); 'native' leaves your desktop's own Qt "
                         "style alone")
    args, _ = ap.parse_known_args(raw)

    # ONE BLAS thread in the GUI process. OpenBLAS/MKL start a spinning
    # thread per core at import; on a 32-core shared server those 32
    # threads starved the drawing thread so badly that a 0.3 s Bode draw
    # took 18 s and the window stayed blank for as long (measured on
    # <sim-host>; OPENBLAS_NUM_THREADS=1 brought first light from 19 s to
    # 2.5 s). The GUI's own numpy work is 3x3 transforms and small
    # solves; the heavy solves run exact arithmetic in worker processes.
    import os
    for var in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    # and a quicker GIL handoff: with a compute thread (an advisor's
    # lambdified sweep, a sympy pass) holding the GIL in 5 ms slices,
    # the drawing thread got 0.3 s of CPU in a 1.9 s stall; 1 ms slices
    # cost the compute threads a few percent and give the GUI its turn
    sys.setswitchinterval(0.001)

    # stage clock: every splash phase is timed and written to the Log
    # tab once the window is up, so a slow start on a given machine
    # (X11 over VNC, NFS-hosted interpreter, a license checkout) says
    # WHICH stage cost the seconds instead of being a guess
    import time as _time
    t_start = _time.perf_counter()
    marks = []

    def _mark(label):
        marks.append((label, _time.perf_counter()))

    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication(sys.argv[:1])
    _mark("Qt")
    splash = _splash()
    splash.show()
    _stage(app, splash, "loading the analysis engine …")

    # the banner's own latency is pure Qt start-up -- nothing shows a
    # pixel before the toolkit loads -- but everything AFTER it can
    # overlap: the heavy imports run in a background thread while the
    # splash stays responsive, instead of freezing it for their sum
    import threading

    heavy_done = threading.Event()

    def _preload():
        try:
            import matplotlib          # noqa: F401  (core, no backend)
            import numpy               # noqa: F401
            import sympy               # noqa: F401

            import circuitinsight.session  # noqa: F401  (engine, Qt-free)
        except Exception:
            pass                       # the real import below re-raises
        finally:
            heavy_done.set()

    threading.Thread(target=_preload, daemon=True,
                     name="ci-preload").start()

    from . import theme                          # Qt-only, cheap
    if args.theme == "cadence":
        theme.apply(app)
    while not heavy_done.wait(0.03):             # splash keeps painting
        app.processEvents()
    _mark("engine imports")
    from .app import build_window                # deps cached: fast now
    _mark("GUI imports")

    _stage(app, splash, "building the workbench …")
    from .app import MainWindow
    win = MainWindow()
    _mark("window")
    if args.cin and args.psf:
        from pathlib import Path
        _stage(app, splash, f"opening {Path(args.cin).name} …")
        # async_open: the window appears after the cheap populate;
        # first light and the auto-setup narrate themselves in the
        # status bar instead of hiding 8+ measured seconds behind
        # this splash
        # the first-light worker must not overlap the first paint: a
        # sympy-holding worker thread starved the draw of the GIL
        # (0.3 s of CPU in a 1.6 s stall). The chain starts after the
        # first paint instead.
        win._auto_deferred = []
        win.open_session(args.cin, args.psf, probe=args.probe,
                         async_open=True)
        _mark("open")
    win.show()
    splash.finish(win)
    _mark("show")

    # show() only SCHEDULES the paint: the window is seen when the
    # event loop first paints it, and anything that blocks the GUI
    # thread before that (a web view's GPU probe on a display without
    # GL, a render process start-up) is invisible to the marks above.
    # Catch the first paint and log the whole breakdown then.
    from PySide6.QtCore import QEvent, QObject, QTimer

    class _FirstPaint(QObject):
        def eventFilter(self, obj, ev):
            if ev.type() == QEvent.Paint and obj is win:
                _mark("first paint")
                win.removeEventFilter(self)
                deferred = getattr(win, "_auto_deferred", None) or []
                win._auto_deferred = None
                for thunk in deferred:
                    QTimer.singleShot(0, thunk)
                total = marks[-1][1] - t_start
                prev = t_start
                bits = []
                for label, at in marks:
                    bits.append(f"{label} {at - prev:.1f}")
                    prev = at
                try:
                    win.log(f"start-up {total:.1f} s to the first paint: "
                            + ", ".join(bits))
                except Exception:                # noqa: BLE001
                    pass                         # the clock is advisory
            return False

    win._first_paint_filter = _FirstPaint(win)
    win.installEventFilter(win._first_paint_filter)
    _install_stall_watch(app, win)
    del build_window
    return app.exec()


def _install_stall_watch(app, win, limit_s: float = 1.0):
    """A watchdog for the GUI thread: a 100 ms heartbeat timer on the
    main thread, a daemon thread that notices when the heartbeat stops
    for more than limit_s and SAMPLES THE MAIN THREAD'S STACK while it
    is stuck; once the thread runs again the stall (duration + stack)
    goes to the Log. Turns "the window froze for a while" into the
    name of the function that froze it."""
    import os
    import sys
    import threading
    import traceback

    from PySide6.QtCore import QTimer

    main_ident = threading.main_thread().ident
    main_tid = threading.get_native_id()
    beat = {"t": __import__("time").perf_counter()}
    pending = []

    def _main_cpu():
        """The MAIN THREAD's own CPU seconds (Linux: /proc task stat);
        None elsewhere. CPU ~ wall over a stall means it computed;
        CPU << wall means it was starved (GIL or cores)."""
        try:
            with open(f"/proc/self/task/{main_tid}/stat") as f:
                fields = f.read().rsplit(")", 1)[1].split()
            return (int(fields[11]) + int(fields[12])) / os.sysconf(
                "SC_CLK_TCK")
        except Exception:                        # noqa: BLE001
            return None

    def _others():
        """Top frame of every OTHER busy Python thread, compact: who
        holds the GIL. The watchdog itself and idle waiters are left out."""
        names = {th.ident: th.name for th in threading.enumerate()}
        out = []
        for ident, frame in sys._current_frames().items():
            if ident == main_ident:
                continue
            fn = frame.f_code.co_name
            if names.get(ident, "").startswith("ci-stall") or fn in (
                    "wait", "_wait", "select", "get", "sleep",
                    "_bootstrap_inner"):
                continue
            base = os.path.basename(frame.f_code.co_filename)
            out.append(f"{fn} ({base}:{frame.f_lineno})")
        return ", ".join(out)

    def _where(frame):
        """The innermost CircuitInsight frame of the stalled main thread,
        else the innermost frame with its package: 'draw (matplotlib)'."""
        chain = []
        f = frame
        while f is not None:
            chain.append(f)
            f = f.f_back
        for f in chain:
            path = f.f_code.co_filename
            if "circuitinsight" in path:
                return (f"{f.f_code.co_name} "
                        f"({os.path.basename(path)}:{f.f_lineno})")
        f = chain[0]
        path = f.f_code.co_filename.replace(os.sep, "/")
        pkg = path.split("site-packages/")
        pkg = pkg[1].split("/")[0] if len(pkg) > 1 else os.path.basename(path)
        return f"{f.f_code.co_name} ({pkg})"

    def _heartbeat():
        import time as _t
        beat["t"] = _t.perf_counter()
        while pending:
            dur, stack, cpu0, cpu1, others, where = pending.pop(0)
            cpu = (f", main thread CPU {cpu1 - cpu0:.1f} s"
                   if cpu0 is not None and cpu1 is not None else "")
            try:
                line = (f"GUI thread stalled {dur:.1f} s{cpu} in {where}"
                        + (f"; other threads: {others}" if others else ""))
                if dur >= 5.0:                   # a long one earns its stack
                    line += chr(10) + stack
                win.log(line)
            except Exception:                    # noqa: BLE001
                pass

    hb = QTimer(win)
    hb.setInterval(100)
    hb.timeout.connect(_heartbeat)
    hb.start()
    win._stall_heartbeat = hb

    def _watch():
        import time as _t
        stuck_since = None
        sample = ""
        while True:
            _t.sleep(0.25)
            age = _t.perf_counter() - beat["t"]
            if age > limit_s:
                if stuck_since is None:
                    stuck_since = beat["t"]
                    cpu0 = _main_cpu()
                    others = _others()
                    frame = sys._current_frames().get(main_ident)
                    where = "the event loop"
                    if frame is not None:
                        sample = "".join(traceback.format_stack(frame)[-8:])
                        where = _where(frame)
            elif stuck_since is not None:
                pending.append((beat["t"] - stuck_since, sample, cpu0,
                                _main_cpu(), others, where))
                stuck_since = None

    threading.Thread(target=_watch, daemon=True, name="ci-stall").start()


if __name__ == "__main__":
    sys.exit(main())
