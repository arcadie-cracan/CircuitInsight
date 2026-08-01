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
    metadata first (what pip actually put there), the package constant
    as the fallback for a source checkout. Never raises -- a banner must
    not be the thing that fails to start."""
    try:
        from importlib.metadata import version

        return "v" + version("circuitinsight")
    except Exception:
        pass
    try:
        import circuitinsight

        return "v" + circuitinsight.__version__
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

    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication(sys.argv[:1])
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
    from .app import build_window                # deps cached: fast now

    _stage(app, splash, "building the workbench …")
    if args.cin and args.psf:
        from pathlib import Path
        _stage(app, splash,
               f"opening {Path(args.cin).name} — matches, rank, "
               f"first light …")
    win = build_window(args.cin, args.psf, probe=args.probe)
    win.show()
    splash.finish(win)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
