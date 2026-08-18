"""Sympy display transforms: rounding/factoring for presentation, the
memoized expression lines, KaTeX and clipboard LaTeX. Depends on
format only."""
from __future__ import annotations
import warnings
from ...keep import is_all
import numpy as np
from ...units import eng  # noqa: E402,F401  (re-exported: view.eng)
from .format import SIG, _wrapped_product, latex_eng, op_unit


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
                # fold factor's monic 1.0*(...) wrapper FIRST, or the
                # grouping sees a Mul where the Add hides
                e = (_best_grouping(_fold_unit_floats(sp.factor(num)))
                     / _best_grouping(_fold_unit_floats(sp.factor(den))))
        except (sp.PolynomialError, sp.GeneratorsNeeded):
            pass

    e = _drop_common_scale(e, sig)
    return _fold_unit_floats(e)

def _best_grouping(side):
    """Partial structure sp.factor cannot see: when a side stays a raw
    Add (irreducible as a whole), collect() by the symbol shared by
    most terms exposes the near-common factor — the gm carried by four
    of five denominator terms in a cascode A0. Tries the few most-shared
    symbols and keeps the most compact form by count_ops; exact
    rearrangement, no value change."""
    from collections import Counter

    import sympy as sp

    terms = sp.Add.make_args(side)
    if len(terms) < 4:
        return side
    best, score = side, sp.count_ops(side)
    cnt = Counter(s for t in terms for s in t.free_symbols)
    for sym, n in cnt.most_common(6):
        if n < 2:
            break
        try:
            c = sp.collect(side, sym)
        except Exception:
            continue
        sc = sp.count_ops(c)
        if sc < score:
            best, score = c, sc
    return best

def _fold_unit_floats(e):
    """Float(±1.0) coefficients become Integers so sympy drops them as
    factors. Two producers: N() leaves unit coefficients as 1.0, and
    sp.factor over float coefficients is monic-with-a-Float — its RR
    content extraction wraps the result in an explicit 1.0*(...). Both
    rendered as a noise '1 ·' in the expression view."""
    import sympy as sp

    m = {}
    for a in e.atoms(sp.Float):
        if float(a) == 1.0:
            m[a] = sp.Integer(1)
        elif float(a) == -1.0:
            m[a] = sp.Integer(-1)
    return e.xreplace(m) if m else e

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

    def _term_tags(poly, part):
        # each numeral tagged at its FINEST identity: part:power:monomial
        # (the per-numeral pass resolves exactly this); numerals of a
        # kept-monomial-free coefficient carry mono '1'. Floats are
        # matched by VALUE (sympy Floats hash by value), so the tags
        # survive the factored rendering wherever factoring kept the
        # rounded coefficients intact.
        from ...analysis.explain import mono_key

        tags = {}
        for powers, coeff in poly.as_dict().items():
            rc = sp.expand(round_expr(coeff))
            for term in sp.Add.make_args(rc):
                c, rest = term.as_coeff_Mul()
                fl = ([c] if isinstance(c, sp.Float)
                      else list(term.atoms(sp.Float)))
                if not fl:
                    continue
                pw = rest.as_powers_dict() if rest != 1 else {}
                key = mono_key({str(y): int(e) for y, e in pw.items()
                                if str(y) != "s"})
                for fa in fl:
                    tags[fa] = f"{part}:{powers[0]}:{key}"
        return tags

    def _per_coeff_factored(poly, part):
        # factor each s-power coefficient separately so a conductance sum groups
        # into a product (G_o1 G_o2) instead of expanding across the polynomial;
        # factor over floats is monic-with-a-Float, so fold its 1.0's out
        expr = sp.Integer(0)
        for powers, coeff in poly.as_dict().items():
            expr += (_fold_unit_floats(sp.factor(round_expr(coeff)))
                     * s ** powers[0])
        return latex_eng(expr, base, wrap, aliases,
                         num_tags=_term_tags(poly, part))

    n_tex = latex_eng(round_expr(sp.factor(ne), factored=True), base, wrap,
                      aliases, num_tags=_term_tags(npoly, "num"))
    return [("N(s) = ", n_tex), ("D(s) = ", _per_coeff_factored(dpoly, "den"))]

def prepare_display(result, base: bool = True, wrap: bool = False,
                    aliases: dict | None = None) -> None:
    """Warm the expensive display transforms OFF the GUI thread.

    The expression lines run round_expr — cancel/nsimplify over the
    full result, measured 3 s at a mere 1040 terms and tens of seconds
    on big hybrids — and used to freeze the app the moment a solve
    delivered. The worker calls this before emitting done; the GUI's
    own _expr_lines call then hits the per-result cache."""
    if getattr(result, "tf", None) is None:
        return
    try:
        _expr_lines(result, base=base, wrap=wrap, aliases=aliases)
    except Exception:
        pass

def _expr_lines(result, base: bool = True, wrap: bool = False,
                aliases: dict | None = None):
    """(label, latex) pairs — the readable form, not the raw expression.

    H(s) is given in factored pole/zero form: A0 times a product of corner-
    frequency factors. That is the textbook form the tool exists to produce, and
    it stays readable where the expanded polynomial — a ratio of 60-digit exact
    integers — does not. A right-half-plane root shows up as (1 - s/...), so its
    excess phase lag is visible in the form itself.

    Memoized per result: the transform is pure (same result, same
    flags, same lines) and expensive, and the worker pre-warms it so
    the GUI-thread call after a solve is a dictionary lookup.
    """
    import sympy as sp

    key = (base, wrap, tuple(sorted((aliases or {}).items())))
    cache = getattr(result, "_expr_lines_cache", None)
    if cache is not None and key in cache:
        return cache[key]

    def _memo(lines):
        try:
            c = getattr(result, "_expr_lines_cache", None)
            if c is None:
                c = {}
                object.__setattr__(result, "_expr_lines_cache", c)
            c[key] = lines
        except Exception:
            pass
        return lines

    a0 = result.dc_gain.real if hasattr(result.dc_gain, "real") else result.dc_gain
    numeric_a0 = (rf"{float(a0):.4g}\quad({result.dc_gain_db:.2f}\,"
                  rf"\mathrm{{dB}})")
    if wrap:
        numeric_a0 = rf"\htmlData{{num=A0}}{{{numeric_a0}}}"

    # When symbols are kept, A_0 IS the point — a ratio in gm/gds, not a number.
    # Round it (the exact form carries 40-digit integer coefficients), but do not
    # replace it with its value: that would throw away the answer the user asked
    # for. Fall back to the number only when nothing symbolic survives.
    lines = []
    a0_sym = None
    try:
        e = round_expr(result.tf.dc_gain(), factored=True)
        if e.free_symbols:
            a0_sym = latex_eng(e, base, wrap, aliases, num_tags="A0")
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
            tag = "p1" if label.startswith(r"p_") else "z1"
            lines.append((label, latex_eng(e, base, wrap, aliases,
                                           num_tags=tag)))
            shown_symbolic = True
    if shown_symbolic:
        note = (r"\mathrm{N,D:\ exact\ symbolic;\ }p_1,z_1\mathrm{\ in\ rad/s.}"
                if raw else
                r"\mathrm{N,D:\ numeric\ corners.\quad}"
                r"p_1,z_1\mathrm{:\ symbolic,\ rad/s.}")
        lines.append(("", note))
    return _memo(lines)

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

def expr_value_map(result) -> dict:
    """Operating-point value of every symbol that appears in H(s), formatted for
    the hover tooltip: name -> '360 uS'. Names are the raw join keys -- the same
    identity the web view's \\htmlData tags carry."""
    vals = getattr(result.tf, "values", {}) or {}
    present = {str(x) for x in result.tf.expr.free_symbols}
    return {n: eng(v, op_unit(n)) for n, v in vals.items() if n in present}

def expr_katex(result, base: bool = True, aliases: dict | None = None,
               numerals: dict | None = None, numhint: str = "") -> dict:
    """Payload for the KaTeX web view: the readable expression lines with every
    device symbol identity-tagged (hover/click handles), plus the value map.
    ``numerals`` maps a numeral tag (num:k / den:k / A0 / p1 / z1) to its
    hover text from Explain the numbers; ``numhint`` is what a numeral
    hover shows when the explanation has not been computed yet.

    ``{"lines": [...], "values": {...}, "numerals": {...}, "numhint": ...}``"""
    lines = [f"{label}{tex}"
             for label, tex in _expr_lines(result, base=base, wrap=True,
                                           aliases=aliases)]
    return {"lines": lines, "values": expr_value_map(result),
            "numerals": numerals or {}, "numhint": numhint}

def numeral_tips(stories, deep=None, top: int = 5) -> dict:
    """tag -> hover text for the Expression view's numerals, from the
    Explain-the-numbers stories. Plain coefficients get their top
    shares; the displayed ratios (A0, p1, z1) get the DIFFERENCE
    attribution -- shares subtract in a ratio, so the common gm chain
    cancels and what remains is what shapes the numeral. `deep` adds
    the per-numeral stories (part:k:monomial tags), which resolve each
    numeral individually."""
    from ...analysis.explain import ratio_contributors

    def fmt(pairs):
        return ", ".join(f"{n} {d:+.0%}" for n, d in pairs[:top])

    tips = {}
    for st in stories:
        tips[f"{st.part}:{st.k}"] = (f"<b>{st.part} s^{st.k}</b> — "
                                     + fmt(st.contributors))
    for st in deep or []:
        m = "" if st.mono == "1" else f" · {st.mono}"
        approx = getattr(st, "approx", False)
        tail = ("<br>per-numeral, fast pass — unconfirmed (≈)"
                if approx else "<br>per-numeral (exact for linear stamps)")
        tips[f"{st.part}:{st.k}:{st.mono}"] = (
            f"<b>{st.part} s^{st.k}{m}</b> — "
            + ("≈ " if approx else "")
            + (fmt(st.contributors) or "no collapsed contributor")
            + tail)
    ks = {p: sorted(st.k for st in stories if st.part == p)
          for p in ("num", "den")}
    for tag, label, a, b in (
        ("A0", "A0", ("num", ks["num"][0]) if ks["num"] else None,
         ("den", ks["den"][0]) if ks["den"] else None),
        ("p1", "p1 (den ratio)",
         ("den", ks["den"][0]) if len(ks["den"]) > 1 else None,
         ("den", ks["den"][1]) if len(ks["den"]) > 1 else None),
        ("z1", "z1 (num ratio)",
         ("num", ks["num"][0]) if len(ks["num"]) > 1 else None,
         ("num", ks["num"][1]) if len(ks["num"]) > 1 else None),
    ):
        if a is None or b is None:
            continue
        rc = ratio_contributors(stories, a, b, top=top)
        if rc:
            tips[tag] = (f"<b>{label}</b> — {fmt(rc)}"
                         "<br>ratio attribution: common factors cancel")
            # the starred variant marks a line with SEVERAL numerals:
            # they share this line-level attribution, the split between
            # them is not resolved at the operating point
            tips[tag + "*"] = (tips[tag]
                               + "<br><i>this line's numerals share one "
                                 "attribution; the per-numeral split is "
                                 "not resolved</i>")
    return tips

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
