"""Where does opening a run spend its time? Phase timings for the
session open and the GUI populate, plus the top cumulative functions.

Usage: python scripts/profile_open.py <cin.json> <psf-dir>
"""
import cProfile
import os
import pstats
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.simplefilter("ignore")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

cin, psf = sys.argv[1], sys.argv[2]
t0 = time.perf_counter()
import circuitinsight.gui.exprweb  # noqa: E402,F401
from PySide6.QtWidgets import QApplication  # noqa: E402
app = QApplication([])
from circuitinsight.gui.app import MainWindow  # noqa: E402
print(f"imports + QApplication      {time.perf_counter() - t0:6.2f} s")

t1 = time.perf_counter()
win = MainWindow()
print(f"MainWindow()                {time.perf_counter() - t1:6.2f} s")

from circuitinsight.session import SessionController  # noqa: E402
t2 = time.perf_counter()
pr = cProfile.Profile()
pr.enable()
c = SessionController.open(cin, psf, cap_model="matrix")
pr.disable()
print(f"SessionController.open      {time.perf_counter() - t2:6.2f} s")
pstats.Stats(pr).sort_stats("cumulative").print_stats(14)

t3 = time.perf_counter()
pr2 = cProfile.Profile()
pr2.enable()
win.open_session(cin, psf, async_open=True)
pr2.disable()
print(f"open_session(async)         {time.perf_counter() - t3:6.2f} s "
      f"(includes a second SessionController.open)")
pstats.Stats(pr2).sort_stats("cumulative").print_stats(14)
