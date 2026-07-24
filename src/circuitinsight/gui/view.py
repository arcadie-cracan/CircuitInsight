"""Pure presentation helpers: turn a `Result` into a Matplotlib figure and
display strings. No Qt, no ipywidgets — both front ends (and the tests) reuse
this with the plain Agg backend.
"""
from __future__ import annotations
from ..keep import is_all

import numpy as np

__all__ = ["bode_figure", "summary_text", "poles_table", "eng"]


def eng(x: float, unit: str = "") -> str:
    """Engineering notation, e.g. 1.24 k, 12.9 M."""
    x = float(x)
    if x == 0 or not np.isfinite(x):
        return f"{x:g}{unit}"
    # down to atto: device caps are fF-pF, so nano was far too coarse
    # (63.8 fF printed as "6.38e-05 nF").
    prefixes = {-18: "a", -15: "f", -12: "p", -9: "n", -6: "u", -3: "m",
                0: "", 3: "k", 6: "M", 9: "G", 12: "T"}
    exp = int(np.floor(np.log10(abs(x)) / 3) * 3)
    exp = max(min(exp, 12), -18)
    return f"{x / 10 ** exp:.3g} {prefixes[exp]}{unit}"


_ENG_PREFIX_TEX = {-18: r"\mathrm{a}", -15: r"\mathrm{f}", -12: r"\mathrm{p}",
                   -9: r"\mathrm{n}", -6: r"\mu", -3: r"\mathrm{m}", 0: "",
                   3: r"\mathrm{k}", 6: r"\mathrm{M}", 9: r"\mathrm{G}",
                   12: r"\mathrm{T}"}


def _eng_coeff_tex(x: float, sig: int = 4) -> str:
    """A numeric coefficient in engineering notation for LaTeX: 5.936e-5 ->
    '59.36\\,\\mu', 2.94e7 -> '29.4\\,\\mathrm{M}'. No unit -- the prefix alone
    makes it comparable at a glance with the dcOp column (gm = 364 uS)."""
    x = float(x)
    if x == 0 or not np.isfinite(x):
        return f"{x:g}"
    sign = "-" if x < 0 else ""
    ax = abs(x)
    if 0.1 <= ax < 1000:                    # comfortable range: plain reads better
        return f"{sign}{ax:.{sig}g}"        # (0.5 -> 0.5, not 500m)
    exp = int(np.floor(np.log10(ax) / 3) * 3)
    exp = max(min(exp, 12), -18)
    mant = f"{ax / 10.0 ** exp:.{sig}g}"
    p = _ENG_PREFIX_TEX[exp]
    return f"{sign}{mant}" if not p else rf"{sign}{mant}\,{p}"


def _tok(i: int) -> str:
    """Fixed-width, digit-free, collision-free placeholder name for a coefficient
    (so sympy won't subscript it and no token is a substring of another).

    The AA prefix makes the token sort BEFORE any device symbol (RSP, CL,
    g_...) in sympy's Mul ordering, so coefficients print first -- without
    it the r2r denominator rendered as 'RSP445 p' (math mode swallows the
    single space sympy leaves between factors)."""
    a, b, c = i // 676 % 26, i // 26 % 26, i % 26
    return "AAeng" + chr(97 + a) + chr(97 + b) + chr(97 + c)


# small-signal quantity prefixes that split off the front of a symbol name;
# anything else is a passive value symbol keyed by its instance name
_QTY = frozenset((
    "gm", "gds", "gmb", "go", "gpi", "gmu",                 # conductances
    "cgd", "cgs", "cgb", "cdb", "csb", "cds", "cdg", "csg",  # MOS caps
    "cbd", "cbs", "cbg", "cpi", "cmu", "cjs", "cjd", "ccs",  # + bipolar/junction
    "csub",                                                  # bjt substrate cap
    "kdd", "kdg", "kdb", "kgd", "kgg", "kgb",                # charge matrix
    "kbd", "kbg", "kbb",
))
# params whose join-key spelling is longer than the conventional subscript,
# or carries its own underscore (checked BEFORE the generic split; the
# body-effect key is gmbs but the textbook symbol is g_mb, and the
# impact-ionization pair gii_d/gii_m would otherwise partition at the
# wrong underscore and render as a bare instance name)
_SPECIAL = {"gmbs": ("g", "mb"),
            "gii_d": ("g", r"ii\,d"), "gii_m": ("g", r"ii\,m")}
_GREEK = {"pi": r"\pi", "mu": r"\mu"}


def _inst_sub(rest: str, base: bool, aliases: dict) -> str:
    """Subscript for an instance path (join-key underscores). A user alias
    for the full instance (I0.MN0) or its leaf (MN0) wins and is inserted
    VERBATIM as LaTeX; otherwise the leaf (base) or full path, upright."""
    full = rest.replace("_", ".")
    leaf = rest.split("_")[-1]
    if full in aliases:
        return aliases[full]
    if leaf in aliases:
        return aliases[leaf]
    return rf"\mathrm{{{leaf if base else full}}}"


def symbol_tex(name: str, base: bool = True, aliases: dict | None = None) -> str:
    """LaTeX for a device symbol name (the raw join-key stays as it is).

    A quantity prefix is typeset as g_m, g_{ds}, c_{gd}; the instance path
    becomes a subscript -- its leaf only in ``base`` mode (g_{m,MN1}), the full
    hierarchy otherwise (g_{m,I0.MN1}). A passive value symbol carries no
    quantity, so it renders as the plain device name (base: Cc; full: I0.Cc) --
    which also stops the I0_ prefix from reading as a current.

    ``aliases`` maps a device instance (by full path or leaf) OR a whole
    symbol name to a LaTeX string: an instance alias remaps the subscript
    of every symbol of that device (MN0 -> M_1 gives g_{m,M_1}, g_{ds,M_1}
    ...), a whole-symbol alias overrides the render outright (RSP -> R_S)."""
    aliases = aliases or {}
    if name in aliases:                         # whole-symbol override
        return aliases[name]
    for pref, (letter, sub) in _SPECIAL.items():
        if name.startswith(pref + "_"):
            return rf"{letter}_{{{sub},{_inst_sub(name[len(pref) + 1:], base, aliases)}}}"
    head, sep, rest = name.partition("_")
    if sep and head in _QTY:
        sub = _GREEK.get(head[1:], head[1:])
        return rf"{head[0]}_{{{sub},{_inst_sub(rest, base, aliases)}}}"
    full = name.replace("_", ".")               # passive: whole = instance
    leaf = name.split("_")[-1]
    if full in aliases:
        return aliases[full]
    if leaf in aliases:
        return aliases[leaf]
    return rf"\mathrm{{{leaf if base else full}}}"


def latex_eng(e, base: bool = True, wrap: bool = False,
              aliases: dict | None = None) -> str:
    """sympy.latex, but every numeric coefficient in engineering notation and
    every device symbol typeset via ``symbol_tex`` (``base`` picks leaf vs full
    instance names).

    With ``wrap`` each symbol is additionally tagged with its raw join-key name
    via KaTeX ``\\htmlData{sym=...}{...}`` -- an identity handle for the web
    view's hover/click, ignored by matplotlib mathtext (so it is off there).

    sympy has no eng-format option, so swap each Float for a placeholder symbol,
    render, then substitute the engineering string back. Integer exponents (s^2)
    are Integers, not Floats, so they're left untouched."""
    import sympy as sp

    if not hasattr(e, "atoms"):
        return sp.latex(e)
    floats = list(e.atoms(sp.Float))
    subs, repl = {}, {}
    for i, f in enumerate(floats):
        t = _tok(i)
        subs[f] = sp.Symbol(t)
        repl[t] = _eng_coeff_tex(f)
    expr = e.xreplace(subs) if subs else e

    def _name(sym):
        tex = symbol_tex(sym.name, base, aliases)
        return rf"\htmlData{{sym={sym.name}}}{{{tex}}}" if wrap else tex

    names = {sym: _name(sym) for sym in expr.free_symbols
             if sym.name not in repl and sym.name != "s"}   # keep s, tokens raw
    s = sp.latex(expr, symbol_names=names)
    import re as _re

    for t, val in repl.items():
        # an explicit \cdot between a coefficient and an adjacent symbol
        # factor: math mode ignores the lone space sympy emits, so
        # '25.76 u g_m' would otherwise render glued as micrograms
        sep = val.replace("\\", "\\\\")          # literal for re.sub repl
        s = _re.sub(_re.escape(t) + r"\s+(?=[\\A-Za-z])",
                    sep + r" \\cdot ", s)
        s = _re.sub(r"(?<=[}a-zA-Z])\s+" + _re.escape(t),
                    r" \\cdot " + sep, s)
        s = s.replace(t, val)
    return s


def _fmt_root(c: complex) -> str:
    c = complex(c)
    if c.imag == 0 or abs(c.imag) < 1e-4 * abs(c.real):
        sign = "−" if c.real < 0 else ""      # LHP negative, RHP positive
        return f"{sign}{eng(abs(c.real), 'Hz')}"
    ang = np.degrees(np.angle(c))
    return f"{eng(abs(c), 'Hz')} ∠{ang:.0f}°"


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
    for ax in (ax1, ax2):
        ax.grid(True, which="both", alpha=0.25, lw=0.4)
    _annotate_margins(ax1, ax2, result)
    _pole_zero_ticks(ax1, result, f)
    ax1.legend(fontsize=8, frameon=False, loc="lower left")
    fig.tight_layout()
    return fig


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

    def evaluate(freqs, factors):
        vals = [tf.values[n] * float(factors.get(n, 1.0)) for n in names]
        w = 2j * np.pi * np.asarray(freqs, dtype=float)
        out = fn(w, *vals)
        return np.broadcast_to(out, w.shape).astype(complex)

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
    for ax in (ax1, ax2):
        ax.grid(True, which="both", alpha=0.25, lw=0.4)
    fig.tight_layout()
    return fig


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
        f"DC gain : {abs(result.dc_gain):.5g}  ({result.dc_gain_db:.2f} dB)",
    ]
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
    for w in result.warnings:
        lines.append(f"⚠ {w}")
    return "\n".join(lines)


def op_unit(name: str) -> str:
    """SI unit for a device symbol, inferred from its name.

    Two naming schemes coexist: an intrinsic parameter carries the quantity as a
    prefix (``gm_I0_MN1``, ``cgd_I0_MP2``), while a passive's value symbol is just
    its (possibly hierarchical) instance name (``I0_Cc``, ``CL``, ``I0_Rz``). For
    the latter the element type is the leading letter of the *leaf* segment, not
    of the whole name -- keying off the whole name lets the ``I0_`` prefix
    masquerade as a current (``I0_Cc`` -> ``14.7 pA``)."""
    n = name.lower()
    if n.startswith(("gm", "gds", "gmb", "go", "gpi", "gmu")):
        return "S"                    # trans/output conductances
    if n.startswith("c"):
        return "F"                    # intrinsic caps (cgd/cgs/...) and C-named caps
    # passive value keyed by instance name: type = first letter of the leaf
    leaf = n.replace(".", "_").split("_")[-1]
    if leaf.startswith("c"):
        return "F"                    # capacitance, e.g. I0_Cc
    if leaf.startswith("r"):
        return "Ω"                    # resistance, e.g. I0_Rz
    if leaf.startswith("l"):
        return "H"                    # inductance
    return ""


def ranking_rows(ranking, values=None):
    """(name, opval, score, 'peaks @ …') rows from rank_symbols().

    `values`: optional name->OP-value map (SessionController.op_values()); the
    dcOp column shows what each symbol actually IS, in engineering units."""
    values = values or {}
    rows = []
    for n, s, pk in ranking:
        ov = eng(values[n], op_unit(n)) if n in values else ""
        rows.append((n, ov, f"{s:.3g}", f"@ {eng(pk, 'Hz')}"))
    return rows


SIG = 4                      # significant digits for numeric coefficients


def round_expr(expr, sig: int = SIG, factored: bool = False):
    """Numbers to `sig` significant digits; symbols and integer exponents intact.

    The engine solves in exact rational arithmetic, so a fully-numeric result is
    a ratio of 60-digit integers — correct and unreadable. sympy's N() rounds the
    coefficients while leaving s**4 an integer power and any kept symbols alone.

    Rounding alone is not enough for a SYMBOLIC ratio. Numerator and denominator
    routinely carry an enormous common factor (exact rationals over big
    denominators), so A_0 prints as
        gm*(2.4e109*gm + 1.4e105) / (2.4e109*gds*gm + 1.4e105*gds + ...)
    Divide both by their polynomial content and the same expression is `gm/gds` --
    the textbook 5T gain. The formula was always there, buried under a factor
    nobody cancelled.

    Finally, N() leaves unit coefficients as `1.0` (`1.0*gds_MN1 + 1.0*gds_MP1`).
    Note sympy's Float(1.0) == 1 is False, so the fold must compare in floats.
    """
    import sympy as sp

    e = sp.cancel(sp.together(expr))

    # 1. Round, then cancel in an EXACT domain.
    # A common factor like (gm + eps) carries an eps differing in the 15th digit
    # between numerator and denominator, so cancel() cannot see it. Rounding makes
    # the two identical -- but cancel() will not do polynomial GCD over floats, so
    # feed it the rounded values as exact rationals. A_0 then collapses from a
    # four-term ratio to gm/(gds_n + gds_p): the textbook 5T gain. A display
    # transform within the rounding tolerance; the exact value is shown alongside.
    try:
        e = sp.cancel(sp.nsimplify(sp.N(e, sig), rational=True))
    except (sp.PolynomialError, ValueError, TypeError):
        pass

    # 2. THEN scale. Cancelling re-derives the coefficients, so normalizing before
    # this step is undone by it -- the scale must be the last thing applied.
    # Divide both sides by the denominator's largest coefficient: polynomial
    # *content* (their GCD) does not help, because the denominator carries extra
    # terms and nothing cancels exactly. Scaling leaves the ratio untouched and
    # brings 1e25-size coefficients onto a human range.
    num, den = sp.fraction(sp.together(e))
    syms = sorted(e.free_symbols, key=str)
    if syms and den != 1:
        try:
            coeffs = [abs(c) for c in sp.Poly(den, *syms).coeffs()]
            scale = max(coeffs) if coeffs else sp.Integer(1)
            if scale not in (0, 1):
                e = sp.expand(num / scale) / sp.expand(den / scale)
        except (sp.PolynomialError, ZeroDivisionError):
            pass

    e = sp.N(e, sig)

    # 3. Factor, if asked. Numerator and denominator can't cancel exactly (a
    # small term blocks the common factor -- that is what Simplify would prune),
    # but factoring each side EXPOSES the near-common factor, so you can see what
    # relaxing the budget would buy: A_0 -> gm*(gm + eps) / (gds_n*(gm+eps) + ...)
    # makes it obvious the (gm+eps) almost cancels. Exact, no accuracy traded.
    if factored:
        try:
            num, den = sp.fraction(sp.together(e))
            # factoring a huge polynomial is slow and the result is unreadable
            # anyway (that is the un-simplified case); only factor tidy ones.
            nterms = len(sp.Add.make_args(num)) + len(sp.Add.make_args(den))
            if nterms <= 40:
                e = sp.factor(num) / sp.factor(den)
        except (sp.PolynomialError, sp.GeneratorsNeeded):
            pass

    e = _drop_common_scale(e, sig)

    ones = {a: sp.Integer(1) for a in e.atoms(sp.Float) if float(a) == 1.0}
    return e.xreplace(ones) if ones else e


def _rgcd(a, b):
    """True rational GCD (sympy's gcd treats rationals as units over QQ and
    returns 1). gcd of numerators over lcm of denominators."""
    import sympy as sp

    a, b = sp.Rational(a), sp.Rational(b)
    if a == 0:
        return b
    if b == 0:
        return a
    return sp.Rational(sp.igcd(a.p, b.p), sp.ilcm(a.q, b.q))


def _dominant_scale(p, rel: float = 1e-6):
    """The GCD of a polynomial's numeric coefficients, taken only over the
    terms within `rel` of the largest |coeff| so a tiny parasitic term can
    not shrink it. None when there is nothing numeric."""
    import sympy as sp
    from functools import reduce

    cs = [abs(t.as_coeff_Mul()[0]) for t in sp.Add.make_args(sp.expand(p))
          if t.as_coeff_Mul()[0].is_number and t.as_coeff_Mul()[0] != 0]
    if not cs:
        return None
    top = max(float(c) for c in cs)
    keep = [sp.nsimplify(c, rational=True) for c in cs if float(c) >= rel * top]
    return reduce(_rgcd, keep) if keep else None


def _drop_common_scale(e, sig: int):
    """Remove a common numeric factor shared by numerator and denominator
    that sympy's float cancel()/factor() miss -- e.g. a balun's 0.5 on
    every term of a large A_0. Conservative: acts ONLY when both sides
    carry the SAME dominant scalar (gn == gd != 1), which the uniform
    balun factor satisfies but a genuine mixed-coefficient ratio does not,
    so no working expression is rescaled."""
    import sympy as sp

    try:
        num, den = sp.fraction(sp.together(e))
        gn, gd = _dominant_scale(num), _dominant_scale(den)
        if gn is None or gd is None or gn != gd or gn in (0, 1):
            return e
        return sp.N(sp.expand(num / gn) / sp.expand(den / gn), sig)
    except Exception:
        return e


def _eng_tex(x_hz: float, unit: str = "Hz") -> str:
    """'4.91 MHz' -> '4.91\\,\\mathrm{MHz}' for mathtext.

    eng() separates the mantissa from the prefix with a THIN SPACE (U+2009), not
    an ASCII space -- so split on whitespace generally. Partitioning on " " never
    matched, and dropped a raw U+2009 into the LaTeX.
    """
    parts = eng(abs(float(x_hz)), unit).split()      # str.split() handles U+2009
    if len(parts) == 2:
        mant, suffix = parts
        return rf"{mant}\,\mathrm{{{suffix}}}"
    return parts[0] if parts else "0"


def _pair_roots(roots):
    """Group roots into ('real', f) and ('pair', f) — conjugates share a factor."""
    roots = [complex(r) for r in roots]
    used = [False] * len(roots)
    out = []
    for i, r in enumerate(roots):
        if used[i]:
            continue
        used[i] = True
        if abs(r.imag) <= 1e-6 * abs(r.real):
            out.append(("real", r))
            continue
        for j in range(i + 1, len(roots)):
            if not used[j] and abs(complex(roots[j]) - r.conjugate()) <= 1e-6 * abs(r):
                used[j] = True
                break
        out.append(("pair", r))
    return out


def _factor_tex(kind, r, sig: int = 3) -> str:
    """One factor of the pole/zero product, normalized to 1 at DC.

    Real root at f:      (1 - s/2*pi*f)   -> LHP roots read as (1 + s/2*pi*|f|).
    Conjugate pair:      (1 + s/2*pi*Q*f0 + (s/2*pi*f0)^2), Q from the real part.
    """
    if kind == "real":
        f = r.real
        sign = "+" if f < 0 else "-"         # LHP -> +, RHP -> -
        return (rf"\left(1 {sign} \frac{{s}}{{2\pi\cdot {_eng_tex(abs(f))}}}"
                rf"\right)")

    f0 = abs(r)                              # |root|, Hz
    if abs(r.real) < 1e-30 * f0:             # purely imaginary: no s term
        return rf"\left(1 + \left(\frac{{s}}{{2\pi\cdot {_eng_tex(f0)}}}\right)^2\right)"
    q = f0 / (2 * abs(r.real))               # Q of the pair
    sign = "+" if r.real < 0 else "-"
    return (rf"\left(1 {sign} \frac{{s}}{{2\pi\cdot {_eng_tex(q * f0)}}} + "
            rf"\left(\frac{{s}}{{2\pi\cdot {_eng_tex(f0)}}}\right)^2\right)")


def _product_tex(roots) -> str:
    facs = [_factor_tex(k, r) for k, r in _pair_roots(roots)]
    return "".join(facs) if facs else "1"


_FACTORS_PER_LINE = 2      # a 4-zero/4-pole product does not fit on one line


def _wrapped_product(label, roots):
    """(label, latex) lines for a factor product, wrapped so it stays legible.

    Shrinking a 4x4 factored form to fit one line renders it at ~6pt — technically
    present, actually unreadable. Wrap instead, and keep the type size.
    """
    facs = [_factor_tex(k, r) for k, r in _pair_roots(roots)]
    if not facs:
        return [(label, "1")]
    out = []
    for i in range(0, len(facs), _FACTORS_PER_LINE):
        chunk = "".join(facs[i:i + _FACTORS_PER_LINE])
        out.append((label if i == 0 else "", chunk))
    return out


def _raw_tf_lines(result, max_terms: int = 14, base: bool = True,
                  wrap: bool = False, aliases: dict | None = None):
    """H(s) = N(s)/D(s) with SYMBOLIC coefficients — the expanded textbook form
    (paper eq. 4), returned only when the expansion is compact enough to read.

    The reduced / low-order solve has clean coefficients (products like
    C_C C_L, C_C g_m, G_o1 G_o2); the full solve is a ratio of many-digit
    integers, so past ``max_terms`` this returns [] and the caller falls back to
    the numeric-root corners. Each denominator coefficient is factored on its own,
    so a sum of output conductances groups back into G_o1 G_o2 as written by hand.
    Returns a list of (label, latex) pairs, or [] when not applicable."""
    import sympy as sp

    try:
        npoly, dpoly = result.tf.num_den
    except Exception:
        return []
    s = result.tf.s
    ne, de = npoly.as_expr(), dpoly.as_expr()
    if not ((ne.free_symbols | de.free_symbols) - {s}):
        return []                                     # nothing symbolic to show
    if len(sp.Add.make_args(ne)) + len(sp.Add.make_args(de)) > max_terms:
        return []                                     # too big -> numeric corners

    def _per_coeff_factored(poly):
        # factor each s-power coefficient separately so a conductance sum groups
        # into a product (G_o1 G_o2) instead of expanding across the polynomial
        expr = sp.Integer(0)
        for powers, coeff in poly.as_dict().items():
            expr += sp.factor(round_expr(coeff)) * s ** powers[0]
        return latex_eng(expr, base, wrap, aliases)

    n_tex = latex_eng(round_expr(sp.factor(ne), factored=True), base, wrap,
                      aliases)
    return [("N(s) = ", n_tex), ("D(s) = ", _per_coeff_factored(dpoly))]


def _expr_lines(result, base: bool = True, wrap: bool = False,
                aliases: dict | None = None):
    """(label, latex) pairs — the readable form, not the raw expression.

    H(s) is given in factored pole/zero form: A0 times a product of corner-
    frequency factors. That is the textbook form the tool exists to produce, and
    it stays readable where the expanded polynomial — a ratio of 60-digit exact
    integers — does not. A right-half-plane root shows up as (1 - s/...), so its
    excess phase lag is visible in the form itself.
    """
    import sympy as sp

    a0 = result.dc_gain.real if hasattr(result.dc_gain, "real") else result.dc_gain
    numeric_a0 = (rf"{float(a0):.4g}\quad({result.dc_gain_db:.2f}\,"
                  rf"\mathrm{{dB}})")

    # When symbols are kept, A_0 IS the point — a ratio in gm/gds, not a number.
    # Round it (the exact form carries 40-digit integer coefficients), but do not
    # replace it with its value: that would throw away the answer the user asked
    # for. Fall back to the number only when nothing symbolic survives.
    lines = []
    a0_sym = None
    try:
        e = round_expr(result.tf.dc_gain(), factored=True)
        if e.free_symbols:
            a0_sym = latex_eng(e, base, wrap, aliases)
    except Exception:
        a0_sym = None

    if a0_sym:
        lines.append(("A_0 = ", a0_sym))
        lines.append(("", rf"= {numeric_a0}"))
    else:
        lines.append(("A_0 = ", numeric_a0))

    # When the expanded H(s) is compact -- the reduced / low-order solve -- show it
    # with SYMBOLIC coefficients: that IS the textbook expression (paper eq. 4),
    # g_m1(g_m5 - C_C s)/(C_C C_L s^2 + C_C g_m5 s + G_o1 G_o2). For the full solve
    # the expansion is a ratio of many-digit integers, so fall back to the
    # numeric-root factored corners instead.
    raw = _raw_tf_lines(result, base=base, wrap=wrap, aliases=aliases)
    poles, zeros = list(result.poles_hz), list(result.zeros_hz)
    if raw:
        lines.append(("H(s) = ", r"\dfrac{N(s)}{D(s)}"))
        lines += raw
    elif poles:
        lines.append(("H(s) = ", r"A_0\,\frac{N(s)}{D(s)}"))
        lines += _wrapped_product("N(s) = ", zeros)
        lines += _wrapped_product("D(s) = ", poles)

    # Dominant pole/zero as closed-form s-plane ROOTS in rad/s (paper eq. 6). The
    # root of a first-order edge is -a0/a1, so p1,z1 = -edge: an LHP pole comes out
    # negative and an RHP zero positive, matching the factored corners and
    # _fmt_root. Left in rad/s -- angular frequency is the natural home of an
    # s-plane root and drops the 1/2pi clutter. Shown on ANY solve, not only after
    # Simplify -- that is where a kept capacitance finally reads as a letter.
    tf = result.tf
    shown_symbolic = False
    for label, poly in ((r"p_1 = ", _num_den(tf)[1]),
                        (r"z_1 = ", _num_den(tf)[0])):
        edge = _edge_ratio(poly)
        if edge is None:
            continue
        e = round_expr(-edge, factored=True)         # -edge = s-plane root (rad/s)
        if getattr(e, "free_symbols", set()):        # symbolic => worth showing
            lines.append((label, latex_eng(e, base, wrap, aliases)))
            shown_symbolic = True
    if shown_symbolic:
        note = (r"\mathrm{N,D:\ exact\ symbolic;\ }p_1,z_1\mathrm{\ in\ rad/s.}"
                if raw else
                r"\mathrm{N,D:\ numeric\ corners.\quad}"
                r"p_1,z_1\mathrm{:\ symbolic,\ rad/s.}")
        lines.append(("", note))
    return lines


def _num_den(tf):
    """(numerator, denominator) Polys in s of a TransferFunction."""
    return tf.num_den


def _edge_ratio(poly):
    """a0/a1 of a Poly in s -- the dominant root's magnitude (rad/s) when the
    roots are well separated. This is what makes a kept capacitance appear: a0 is
    the DC coefficient (caps open at DC, so C-free), a1 is the first-order one
    (built FROM the caps), so a0/a1 is a ratio with the kept C in the bottom."""
    a = list(reversed(poly.all_coeffs()))            # ascending powers of s
    if len(a) >= 2 and a[0] != 0 and a[1] != 0:
        return a[0] / a[1]
    return None


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


def expr_value_map(result) -> dict:
    """Operating-point value of every symbol that appears in H(s), formatted for
    the hover tooltip: name -> '360 uS'. Names are the raw join keys -- the same
    identity the web view's \\htmlData tags carry."""
    vals = getattr(result.tf, "values", {}) or {}
    present = {str(x) for x in result.tf.expr.free_symbols}
    return {n: eng(v, op_unit(n)) for n, v in vals.items() if n in present}


def expr_katex(result, base: bool = True,
               aliases: dict | None = None) -> dict:
    """Payload for the KaTeX web view: the readable expression lines with every
    device symbol identity-tagged (hover/click handles), plus the value map.

    ``{"lines": [latex, ...], "values": {name: "360 uS", ...}}``"""
    lines = [f"{label}{tex}"
             for label, tex in _expr_lines(result, base=base, wrap=True,
                                           aliases=aliases)]
    return {"lines": lines, "values": expr_value_map(result)}


def markdown_report(result) -> str:
    """A self-contained Markdown report of the current solve."""
    md = [f"# CircuitInsight — {result.inp} → {result.out}", "",
          "```", summary_text(result), "```", "",
          "## Expressions", ""]
    md += [f"$${label.strip()}{tex}$$" for label, tex in _expr_lines(result)]
    md += ["", "## Transfer function (expanded)", "",
           f"$$H(s) = {tf_latex(result)}$$", ""]
    return "\n".join(md)


def _fig_b64(fig) -> str:
    """Render a Figure to a base64 PNG for the self-contained report."""
    import base64
    import io

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    return base64.b64encode(buf.getvalue()).decode("ascii")


#: Spectre's MOSFET region codes (dcOpInfo `region` parameter)
_REGIONS = {0: "off", 1: "triode", 2: "sat", 3: "subth", 4: "break"}


def region_name(code) -> str:
    try:
        return _REGIONS.get(int(code), str(code))
    except (TypeError, ValueError):
        return ""


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


def tf_latex(result, sig: int = SIG) -> str:
    """The expanded H(s): normalized, with coefficients rounded to `sig` digits.

    `result.tf_latex` is the exact expression — a ratio of 60-digit integers once
    everything is numeric. Keep it for provenance; never show it.

    Rounding alone is not enough: the raw coefficients span 10^65..10^101, which
    is just as unreadable in scientific notation. Normalizing so the denominator's
    constant term is 1 puts H(0) = A_0 in plain sight and brings the rest onto a
    human scale.
    """
    import sympy as sp

    s = sp.Symbol("s")
    num, den = sp.fraction(sp.cancel(result.tf.expr))
    d0 = den.subs(s, 0)
    if d0 != 0:
        num, den = sp.expand(num / d0), sp.expand(den / d0)
    return sp.latex(round_expr(num / den, sig))


def poles_table(result):
    """(kind, freq_Hz_str, note) rows for a table widget."""
    rows = []
    for p in result.poles_hz:
        note = "RHP" if p.real > 0 else ""
        rows.append(("pole", _fmt_root(p), note))
    for z in result.zeros_hz:
        note = "RHP" if z.real > 0 else ""
        rows.append(("zero", _fmt_root(z), note))
    return rows
