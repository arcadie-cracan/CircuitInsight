"""Text, Markdown, HTML and CSV report generation."""
from __future__ import annotations
import warnings
from ...keep import is_all
import numpy as np
from ...units import eng  # noqa: E402,F401  (re-exported: view.eng)
from .figures import _fig_b64, bode_figure, error_figure, expr_figure, fidelity
from .format import _fmt_root
from .simplify import _expr_lines, tf_latex


def summary_text(result) -> str:
    """Human-readable one-block summary of a solve."""
    # keep is ALL (fully symbolic) or a list; [] means fully numeric. These
    # are opposites — reporting both as "numeric" is what the old falsy test
    # did back when fully-symbolic was spelled None.
    if is_all(result.keep):
        mode = "   (fully symbolic)"
    elif result.keep:
        mode = f"   keep: {', '.join(result.keep)}"
    else:
        mode = "   (numeric — no symbols kept)"
    lines = [
        f"{result.inp} → {result.out}{mode}",
    ]
    # designer numbers FIRST: A0, GBW, one formula per root. This is the
    # form a designer actually reads a result in; the raw pole/zero list
    # below stays as the exact record.
    tpl = getattr(result, "template_text", None)
    if tpl:
        lines += [tpl, ""]
    lines.append(
        f"DC gain : {eng(abs(result.dc_gain), sig=4)} "
        f" ({result.dc_gain_db:.2f} dB)")
    poles = list(result.poles_hz)
    zeros = list(result.zeros_hz)
    if poles:
        lines.append("poles   : " + ", ".join(_fmt_root(p) for p in poles[:8])
                     + (" …" if len(poles) > 8 else ""))
    if zeros:
        lines.append("zeros   : " + ", ".join(_fmt_root(z) for z in zeros[:8])
                     + (" …" if len(zeros) > 8 else ""))
    # Two errors against two DIFFERENT baselines. Reporting one while plotting the
    # other is what made these results impossible to interpret.
    if result.simplified:
        lines.append(
            f"terms   : {result.n_terms} (from {result.n_terms_full}) — pruned "
            f"within {result.mag_err_db:.3f} dB / {result.phase_err_deg:.2f}° "
            f"of the FULL MODEL")
    else:
        lines.append(f"terms   : {result.n_terms}")

    fid = fidelity(result)
    if fid is not None:
        lines.append(f"vs sim  : {fid[0]:.3f} dB / {fid[1]:.2f}° max "
                     f"(model fidelity — independent of the keep set)")

    # The keep set is the most misread control in the tool: it selects which
    # parameters stay as letters, and cannot trade accuracy, because the rest
    # become the EXACT rationals of their OP values.
    if not is_all(result.keep) and result.keep:
        lines.append(f"note    : keep chooses which symbols survive — every keep "
                     f"set is exact. Simplify (budget) is the accuracy knob.")
    if getattr(result, "circuit_state", "as imported") != "as imported":
        lines.append("circuit : REDUCED — AC-grounded bias nodes; the "
                     "grounding cost was measured at apply time (Reduce "
                     "bench), and this result is exact for that circuit")
    for w in result.warnings:
        lines.append(f"⚠ {w}")
    det = getattr(result, "details", None)
    if det:
        lines.append("")
        lines.append("reduction detail")
        for d in det:
            lines.append(f"  · {d}")
    return "\n".join(lines)

def markdown_report(result) -> str:
    """A self-contained Markdown report of the current solve."""
    md = [f"# CircuitInsight — {result.inp} → {result.out}", "",
          "```", summary_text(result), "```", "",
          "## Expressions", ""]
    md += [f"$${label.strip()}{tex}$$" for label, tex in _expr_lines(result)]
    md += ["", "## Transfer function (expanded)", "",
           f"$$H(s) = {tf_latex(result)}$$", ""]
    return "\n".join(md)

def report_section(title: str, fig, text: str) -> str:
    """One lab-notebook entry: heading, the CURRENT figure, the summary
    block. Appended to a session report by the GUI's Add-to-report."""
    b64 = _fig_b64(fig)
    return (f"<h2>{title}</h2>\n<pre>{text}</pre>\n"
            f"<img alt='{title}' src='data:image/png;base64,{b64}'>")

def session_report(title: str, sections: list[str]) -> str:
    """The accumulated session report: SLiCAP-style, the report IS the
    artifact -- every Add-to-report click appends a section."""
    head = ("<meta charset='utf-8'>"
            f"<title>{title}</title>"
            "<style>body{font-family:sans-serif;max-width:900px;"
            "margin:2em auto;padding:0 1em}pre{background:#f4f4f4;"
            "padding:1em;overflow-x:auto}img{max-width:100%}"
            "h1{font-size:1.4em}h2{font-size:1.1em;border-top:1px solid "
            "#ddd;padding-top:0.8em}</style>"
            f"<h1>{title}</h1>")
    return head + "\n" + "\n".join(sections)

def traces_csv(result) -> str:
    """The current curves as CSV: frequency, model magnitude/phase, and
    the sim reference when present."""
    import io

    buf = io.StringIO()
    f = np.asarray(result.freqs, dtype=float)
    h = np.asarray(result.h)
    cols = ["freq_hz", "model_db", "model_deg"]
    data = [f, 20 * np.log10(np.abs(h)),
            np.degrees(np.unwrap(np.angle(h)))]
    if result.h_ref is not None:
        hr = np.asarray(result.h_ref)
        cols += ["ref_db", "ref_deg"]
        data += [20 * np.log10(np.abs(hr)),
                 np.degrees(np.unwrap(np.angle(hr)))]
    buf.write(",".join(cols) + "\n")
    for row in zip(*data):
        buf.write(",".join(f"{x:.10g}" for x in row) + "\n")
    return buf.getvalue()

def html_report(result) -> str:
    """A single-file HTML report: summary, Bode, expressions (rendered by
    matplotlib mathtext -- no JS, opens anywhere), and the error view.
    All images embedded as base64."""
    imgs = [("Bode", _fig_b64(bode_figure(result)))]
    try:
        imgs.append(("Expressions", _fig_b64(expr_figure(result))))
    except Exception:
        pass
    if result.h_ref is not None:
        imgs.append(("Model − AC sim", _fig_b64(error_figure(result))))

    parts = [
        "<meta charset='utf-8'>",
        "<title>CircuitInsight — "
        f"{result.inp} → {result.out}</title>",
        "<style>body{font-family:sans-serif;max-width:900px;margin:2em auto;"
        "padding:0 1em}pre{background:#f4f4f4;padding:1em;overflow-x:auto}"
        "img{max-width:100%}h1{font-size:1.4em}h2{font-size:1.1em}</style>",
        f"<h1>CircuitInsight — {result.inp} "
        f"→ {result.out}</h1>",
        "<pre>" + summary_text(result) + "</pre>",
    ]
    for title, b64 in imgs:
        parts.append(f"<h2>{title}</h2>")
        parts.append(f"<img alt='{title}' src='data:image/png;base64,{b64}'>")
    return "\n".join(parts)
