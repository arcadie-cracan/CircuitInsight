"""ipywidgets DEMO front end over `SessionController` + `view`.

STATUS: intentionally minimal -- a Solve/Simplify demonstrator for
notebooks. The desktop app (gui.app) is the primary surface and is
where the benches live (loop gain, compensation, modes, GFT, what-if
sliders, session reports); this module does not track it. Script the
SessionController directly for anything beyond a quick look.

Usage in a notebook (needs `circuitinsight[notebook]`):

    from circuitinsight.gui.notebook import build_ui
    build_ui(cin="ota5t.cin.json", psf="psf/")     # or build_ui(controller=ctrl)
"""
from __future__ import annotations

from ..session import SessionController
from . import view


def build_ui(controller: SessionController | None = None, *, cin=None, psf=None,
             simulator: str = "spectre"):
    """Return an ipywidgets UI: in/out dropdowns, a band-ranked keep-set
    multiselect, and Solve / Simplify. Pass a ready `controller`, or `cin`/`psf`."""
    import ipywidgets as w
    import matplotlib.pyplot as plt
    from IPython.display import display

    if controller is None:
        if cin is None or psf is None:
            raise ValueError("build_ui: pass a controller, or both cin and psf")
        controller = SessionController.open(cin, psf, simulator=simulator)

    in_dd = w.Dropdown(options=controller.input_ports(), description="in:")
    suggested = controller.suggested_input()
    if suggested:
        in_dd.value = suggested
    out_dd = w.Dropdown(options=controller.output_nets(), description="out:")

    match_btn = w.Button(description="Suggest matches")
    match_lbl = w.Label(value="matched: none")

    def _suggest_matches(_):
        groups = controller.suggest_matches()
        controller.set_matches(*groups)
        match_lbl.value = ("matched: " + "; ".join("=".join(g) for g in groups)
                           if groups else "matched: none")
    match_btn.on_click(_suggest_matches)

    rank_btn = w.Button(description="Rank")
    keep = w.SelectMultiple(options=[], rows=8, description="keep:",
                            layout=w.Layout(width="360px"))
    mag = w.BoundedFloatText(value=1.0, min=0.0, max=20.0, step=0.1,
                             description="mag dB:", layout=w.Layout(width="150px"))
    phase = w.BoundedFloatText(value=5.0, min=0.0, max=90.0, step=0.5,
                               description="phase °:", layout=w.Layout(width="150px"))
    solve_btn = w.Button(description="Solve", button_style="primary")
    simp_btn = w.Button(description="Simplify")
    out = w.Output()

    def _rank(_):
        try:
            ranking = controller.rank_symbols(in_dd.value, out_dd.value)
        except Exception as exc:
            with out:
                out.clear_output(wait=True)
                print(f"rank failed: {type(exc).__name__}: {exc}")
            return
        keep.options = [f"{n}   ({s:.3g}, @{view.eng(pk, 'Hz')})"
                        for n, s, pk in ranking]
        keep._names = [n for n, _, _ in ranking]     # display -> name

    def _selected_keep():
        names = getattr(keep, "_names", [])
        opts = list(keep.options)
        return [names[opts.index(v)] for v in keep.value] if names else []

    def _render(get_result):
        with out:
            out.clear_output(wait=True)
            try:
                r = get_result()
            except Exception as exc:
                print(f"failed: {type(exc).__name__}: {exc}")
                return
            print(view.summary_text(r))
            fig = view.bode_figure(r)
            display(fig)
            plt.close(fig)

    rank_btn.on_click(_rank)
    solve_btn.on_click(lambda _: _render(
        lambda: controller.solve(in_dd.value, out_dd.value, _selected_keep())))
    simp_btn.on_click(lambda _: _render(
        lambda: controller.simplify(in_dd.value, out_dd.value, _selected_keep(),
                                    mag_db=mag.value, phase_deg=phase.value)))

    return w.VBox([
        w.HBox([in_dd, out_dd]),
        w.HBox([match_btn, match_lbl]),
        w.HBox([rank_btn, keep]),
        w.HBox([mag, phase, solve_btn, simp_btn]),
        out,
    ])
