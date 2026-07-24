"""Front ends for CircuitInsight, over the headless `SessionController`.

- `view`     — pure presentation (Result -> Matplotlib figure + strings); no Qt.
- `app`      — the PySide6 desktop application (needs `circuitinsight[gui]`).
- `notebook` — the ipywidgets teaching UI (needs `circuitinsight[notebook]`).

The desktop app and the notebook are thin adapters over the same `view` helpers
and `SessionController`; neither holds analysis or state logic. Nothing here is
imported by the core (`circuitinsight/__init__.py`), so importing the base package
never pulls in Qt.
"""
