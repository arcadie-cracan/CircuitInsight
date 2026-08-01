"""Launch the CircuitInsight desktop app preloaded with the checked-in 5T OTA
fixture — a one-command "try it".

    pip install -e .[gui]
    python examples/gui_demo.py

Or run the app empty and open your own CIN + psf from the toolbar:

    python -m circuitinsight.gui        # (or the `circuitinsight-gui` command)
"""
import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

from circuitinsight.gui.app import MainWindow

FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "spectre" / "ota5t"


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.open_session(str(FIXTURE / "tb_ota5t.cin.json"), str(FIXTURE / "psf"))
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
