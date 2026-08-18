"""Matplotlib figures for both front ends (desktop and notebook).
No Qt imports, ever."""
from __future__ import annotations
import warnings
from ...keep import is_all
import numpy as np
from ...units import eng  # noqa: E402,F401  (re-exported: view.eng)
from .simplify import _expr_lines


#: overlay palette after the primary blue (Okabe-Ito, colorblind-safe)
_OVERLAY_COLORS = ("#D55E00", "#009E73", "#CC79A7", "#E69F00")

def _annotate_margins(ax1, ax2, result):
    """PM/GM markers for a loop-gain Result (pm_deg set by
    session.loop_gain); silently nothing for ordinary transfers."""
    pm = getattr(result, "pm_deg", None)
    if pm is None:
        return
    ax1.axhline(0.0, color="k", lw=0.5, ls=":", alpha=0.6)
    if result.pm_freq_hz:
        for ax in (ax1, ax2):
            ax.axvline(result.pm_freq_hz, color="#009E73", lw=0.7, ls="--",
                       alpha=0.8)
        ax2.annotate(f"PM {pm:.1f}° @ {eng(result.pm_freq_hz, 'Hz')}",
                     xy=(result.pm_freq_hz, pm - 180.0),
                     xytext=(4, 4), textcoords="offset points",
                     fontsize=7, color="#009E73")
    gm = getattr(result, "gm_db", None)
    if gm is not None and result.gm_freq_hz:
        ax1.axvline(result.gm_freq_hz, color="#CC79A7", lw=0.7, ls=":",
                    alpha=0.8)
        ax1.annotate(f"GM {gm:.1f} dB", xy=(result.gm_freq_hz, -gm),
                     xytext=(4, 4), textcoords="offset points",
                     fontsize=7, color="#CC79A7")

def _pole_zero_ticks(ax1, result, f):
    """Small markers along the top edge of the magnitude axis at the
    pole/zero magnitudes (x = poles, o = zeros; red = RHP)."""
    lo, hi = float(np.min(f)), float(np.max(f))
    ymin, ymax = ax1.get_ylim()
    y = ymax - 0.04 * (ymax - ymin)
    for roots, marker in ((result.poles_hz, "x"), (result.zeros_hz, "o")):
        for r in np.atleast_1d(roots):
            fr = abs(complex(r))
            if not (lo <= fr <= hi) or fr == 0:
                continue
            rhp = complex(r).real > 0
            ax1.plot([fr], [y], marker=marker, ms=4,
                     color=("#D00000" if rhp else "#666666"),
                     mew=1.0, ls="none", clip_on=False)

def bode_figure(result, fig=None, overlays=()):
    """Magnitude/phase Bode of the model, with the AC-sim overlay if present,
    PM/GM annotations for loop-gain results, and pole/zero tick markers.
    `overlays`: additional Results drawn for comparison (history multi-select).
    Pass an existing Figure (e.g. a Qt canvas's) to draw into it."""
    from matplotlib.figure import Figure

    fig = fig if fig is not None else Figure(figsize=(5.2, 4.0))
    # Clearing a figure that already holds sharex-linked LOG axes makes
    # matplotlib reset them to (0, 1) and warn about the non-positive 0. It
    # discards that limit anyway and we set proper ones below, so the warning
    # is pure noise on every redraw into an existing canvas.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*non-positive.*",
                                category=UserWarning)
        # zero-margin shared axes can also collapse to identical xlims
        # during the clear; both limits are discarded and reset below
        warnings.filterwarnings("ignore", message=".*identical low and high.*",
                                category=UserWarning)
        fig.clear()
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)

    f = np.asarray(result.freqs, dtype=float)
    h = np.asarray(result.h)
    label = f"{result.inp} → {result.out}" if not overlays         else f"{result.out}"
    ax1.semilogx(f, 20 * np.log10(np.abs(h)), color="#0072B2", lw=1.4,
                 label=label if overlays else "model")
    ax2.semilogx(f, np.degrees(np.unwrap(np.angle(h))), color="#0072B2", lw=1.4)
    if result.h_ref is not None:
        hr = np.asarray(result.h_ref)
        ax1.semilogx(f, 20 * np.log10(np.abs(hr)), color="k", ls="--", lw=1.0,
                     label=result.ref_label or "sim")
        ax2.semilogx(f, np.degrees(np.unwrap(np.angle(hr))), color="k",
                     ls="--", lw=1.0)
    for i, other in enumerate(overlays):
        fo = np.asarray(other.freqs, dtype=float)
        ho = np.asarray(other.h)
        c = _OVERLAY_COLORS[i % len(_OVERLAY_COLORS)]
        ax1.semilogx(fo, 20 * np.log10(np.abs(ho)), color=c, lw=1.1,
                     label=f"{other.out}")
        ax2.semilogx(fo, np.degrees(np.unwrap(np.angle(ho))), color=c, lw=1.1)

    if getattr(result, "simplified", False) and \
            getattr(result, "band_fmin", None) is not None:
        for ax in (ax1, ax2):
            ax.axvspan(result.band_fmin, result.band_fmax,
                       color="#0072B2", alpha=0.05, lw=0)
    ax1.set_ylabel("|H| (dB)")
    ax2.set_ylabel("phase (deg)")
    ax2.set_xlabel("frequency (Hz)")
    # one shared frequency axis: labels only under the phase plot, and
    # no x padding -- the data spans the full axes width, so the band
    # slider above aligns with what it highlights
    ax1.tick_params(labelbottom=False)
    for ax in (ax1, ax2):
        ax.grid(True, which="both", alpha=0.25, lw=0.4)
        ax.margins(x=0)
    _annotate_margins(ax1, ax2, result)
    _pole_zero_ticks(ax1, result, f)
    # layout FIRST: the legend anchors to the axes' final position, so
    # it stays centered on the frequency axis, not on the figure.
    # pad trimmed from the 1.08 default — the default costs ~15 px per
    # side; the right edge then stretches to just half an x tick label,
    # since no artist lives right of the axes
    fig.tight_layout(rect=(0, 0.06, 1, 1), pad=0.35)
    fig.subplots_adjust(right=0.985)
    figure_legend(fig, ax1)
    return fig

def figure_legend(fig, ax1):
    """The legend on ONE line right below the frequency-axis label,
    center-aligned to the axis (the axes are offset right by the y
    labels, so figure-center would sit visibly left of the axis).
    Rebuilt from ax1's labeled lines -- the what-if overlay re-calls
    this after adding its dashed trace.

    The vertical anchor is MEASURED from the xlabel's laid-out extent:
    a fixed figure-fraction offset scales with window height and put
    the legend on top of the label on short canvases."""
    handles, labels = ax1.get_legend_handles_labels()
    if not handles:
        return          # never delete a legend we cannot rebuild: a call
    #                     at a label-less moment used to kill it for good
    for lg in list(fig.legends):
        lg.remove()
    axb = fig.axes[1] if len(fig.axes) > 1 else ax1
    pos = axb.get_position()
    y = _legend_anchor_y(fig, axb)
    # the box hangs DOWNWARD from the anchor; clamp it inside the figure
    # or a canvas that shrank (certificate hint appearing, short panel)
    # pushes it below the edge — clipped is indistinguishable from gone
    h_px = fig.get_size_inches()[1] * fig.dpi
    y = max(y, 24.0 / max(h_px, 1.0))    # anchor is the box TOP; ~21 px tall
    fig._ci_legend_y = y                         # refresh_legend compares
    fig.legend(handles, labels, loc="upper center",
               bbox_to_anchor=((pos.x0 + pos.x1) / 2, y),
               ncol=max(1, len(labels)), frameon=False, fontsize=8,
               borderaxespad=0.0, columnspacing=1.4, handlelength=1.6)

def _legend_anchor_y(fig, axb) -> float:
    h_px = fig.get_size_inches()[1] * fig.dpi
    y = axb.get_position().y0 - 46.0 / max(h_px, 1)   # fallback: ticks+label
    try:
        ren = fig.canvas.get_renderer()
        bb = axb.xaxis.label.get_window_extent(renderer=ren)
        y = bb.transformed(fig.transFigure.inverted()).y0 - 4.0 / h_px
    except Exception:
        pass
    return y

def refresh_legend(fig) -> None:
    """Re-anchor the bottom legend after a canvas resize: the xlabel's
    figure-fraction position shifts with height (its pad is in points),
    so a legend placed for one size collides at another. Called from the
    canvas draw hook; rebuilds only when the anchor actually moved, so
    the draw loop converges."""
    if len(fig.axes) < 2:
        return
    if not fig.legends:
        # a redraw path dropped it (or a label-less moment refused to
        # build one): resurrect instead of only re-anchoring survivors
        figure_legend(fig, fig.axes[0])
        return
    axb = fig.axes[1]
    y = _legend_anchor_y(fig, axb)
    if abs(y - getattr(fig, "_ci_legend_y", 1e9)) < 0.004:
        return
    figure_legend(fig, fig.axes[0])

def whatif_fn(result):
    """Compile the kept-symbolic TF once: returns (names, f(freqs, factors))
    where factors maps kept-symbol name -> multiplier on its OP value.
    The rest of the circuit stays the EXACT rationals of the operating
    point -- that is the whole point: what-if on one knob, exact
    everywhere else. None when the result has no (finite) keep set."""
    import sympy as sp

    keep = result.keep
    if not isinstance(keep, list) or not keep:
        return None
    tf = result.tf
    # the keep table stores full symbol names (gm_I0_MN1); instance-suffix
    # keeps (hybrid_split semantics) expand to every matching symbol
    names = []
    for n in tf.symbols:
        if n in tf.values and (n in keep
                               or any(n.endswith("_" + k) for k in keep)):
            names.append(n)
    if not names or len(names) > 12:
        return None
    syms = [tf.symbols[n] for n in names]
    s = sp.Symbol("s")
    fn = sp.lambdify((s, *syms), tf.expr, "numpy")
    fn_mp = None

    def evaluate(freqs, factors):
        nonlocal fn_mp
        vals = [tf.values[n] * float(factors.get(n, 1.0)) for n in names]
        w = 2j * np.pi * np.asarray(freqs, dtype=float)
        try:
            out = np.broadcast_to(fn(w, *vals), w.shape).astype(complex)
            if not np.all(np.isfinite(out[w != 0])):
                raise OverflowError
        except (OverflowError, TypeError, ValueError):
            # the non-kept parameters are exact rationals, so on a large
            # circuit the coefficients overflow float64 even though the
            # ratio does not -- same fallback as TransferFunction.numeric
            import mpmath

            if fn_mp is None:
                fn_mp = sp.lambdify((s, *syms), tf.expr, "mpmath")
            with mpmath.workdps(50):
                # the KEPT values must be mpf too: a huge int times a plain
                # float overflows before mpmath ever sees the product
                mv = [mpmath.mpf(v) for v in vals]
                out = np.array([complex(fn_mp(mpmath.mpc(0, wi.imag), *mv))
                                for wi in w], dtype=complex)
        return out

    return names, evaluate

def fidelity(result):
    """(max |dB| error, max |deg| error) of the model against the AC sim, or None.

    This is the model-vs-SIMULATOR gap: the small-signal reconstruction (hybrid-pi,
    lumped caps) versus the simulator's own device models. It is a property of the
    modelling, and is INDEPENDENT of the keep set -- non-kept parameters become the
    exact rationals of their OP values, so every keep set is exact and reproduces
    this same curve.

    Do not confuse it with `simplify()`'s error, which is measured against the FULL
    SYMBOLIC MODEL, not against the simulator. Two errors, two baselines; reporting
    one while plotting the other is what made results impossible to interpret.
    """
    if result.h_ref is None:
        return None
    h, hr = np.asarray(result.h), np.asarray(result.h_ref)
    dmag = np.abs(20 * np.log10(np.abs(h)) - 20 * np.log10(np.abs(hr)))
    dph = np.abs(np.degrees(np.unwrap(np.angle(h)))
                 - np.degrees(np.unwrap(np.angle(hr))))
    return float(np.max(dmag)), float(np.max(dph))

def error_figure(result, fig=None):
    """Residual against the AC sim, which two overlapping Bode curves cannot show.

    Also draws simplify()'s budget, so the two errors are visibly distinct: the
    residual is model-vs-simulator, the budget is simplified-vs-full-model.
    """
    from matplotlib.figure import Figure

    fig = fig if fig is not None else Figure(figsize=(5.2, 3.0))
    fig.clear()
    if result.h_ref is None:
        ax = fig.add_subplot(1, 1, 1)
        ax.axis("off")
        ax.text(0.5, 0.5, "no AC reference in this run", ha="center",
                va="center", fontsize=9)
        return fig

    f = np.asarray(result.freqs, dtype=float)
    h, hr = np.asarray(result.h), np.asarray(result.h_ref)
    dmag = 20 * np.log10(np.abs(h)) - 20 * np.log10(np.abs(hr))
    dph = (np.degrees(np.unwrap(np.angle(h)))
           - np.degrees(np.unwrap(np.angle(hr))))

    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)
    ax1.semilogx(f, dmag, color="#D55E00", lw=1.2)
    ax2.semilogx(f, dph, color="#D55E00", lw=1.2)
    for ax, budget in ((ax1, result.mag_err_db), (ax2, result.phase_err_deg)):
        ax.axhline(0.0, color="k", lw=0.6, ls=":")
        if result.simplified and budget is not None:
            for sign in (+1, -1):
                ax.axhline(sign * budget, color="#0072B2", lw=0.8, ls="--")
    ax1.set_ylabel("Δ|H| (dB)")
    ax2.set_ylabel("Δphase (deg)")
    ax2.set_xlabel("frequency (Hz)")
    ax1.set_title("model − AC sim   (blue: simplify budget, vs the full model)",
                  fontsize=8)
    # same recipe as the Bode above it: one shared frequency axis and
    # no x padding, so the residual sits visually aligned under the
    # main plots
    ax1.tick_params(labelbottom=False)
    for ax in (ax1, ax2):
        ax.grid(True, which="both", alpha=0.25, lw=0.4)
        ax.margins(x=0)
    fig.tight_layout()
    return fig

def expr_figure(result, fig=None, fontsize: float = 11.0, base: bool = True,
                aliases: dict | None = None):
    """Render the readable expressions as mathtext, one line per entry.

    ``base`` picks leaf device names (g_{m,MN1}) over the full hierarchy
    (g_{m,I0.MN1})."""
    from matplotlib.figure import Figure

    lines = _expr_lines(result, base=base, aliases=aliases)
    fig = fig if fig is not None else Figure(figsize=(7.0, 0.42 * len(lines) + 0.3))
    fig.clear()
    ax = fig.add_axes([0.01, 0.0, 0.98, 1.0])
    ax.axis("off")

    n = max(len(lines), 1)
    step = 1.0 / (n + 0.5)
    y = 1.0 - 0.6 * step
    for label, tex in lines:
        # continuation of a wrapped product (empty label): indent, so it reads as
        # a continuation rather than a new statement
        x = 0.0 if label else 0.05
        ax.text(x, y, f"${label}{tex}$", fontsize=fontsize, va="center",
                ha="left")
        y -= step
    return fig

def _fig_b64(fig) -> str:
    """Render a Figure to a base64 PNG for the self-contained report."""
    import base64
    import io

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    return base64.b64encode(buf.getvalue()).decode("ascii")
