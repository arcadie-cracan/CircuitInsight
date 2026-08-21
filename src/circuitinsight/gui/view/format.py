"""Symbol/LaTeX/engineering formatting: names to TeX, coefficients to
engineering notation, root and region tables. No sympy transforms, no
figures -- the leaf module of the view package."""
from __future__ import annotations
import warnings
from ...keep import is_all
import numpy as np
from ...units import eng  # noqa: E402,F401  (re-exported: view.eng)


# every prefix upright: \mu alone is the ITALIC Greek letter, and an
# italic prefix reads as a variable. The unicode micro sign inside
# \mathrm stays roman in BOTH engines -- KaTeX (strict: false) and the
# matplotlib mathtext fallback, which has no \text command at all.
_ENG_PREFIX_TEX = {-18: r"\mathrm{a}", -15: r"\mathrm{f}", -12: r"\mathrm{p}",
                   -9: r"\mathrm{n}", -6: r"\mathrm{µ}", -3: r"\mathrm{m}",
                   0: "",
                   3: r"\mathrm{k}", 6: r"\mathrm{M}", 9: r"\mathrm{G}",
                   12: r"\mathrm{T}"}

def _eng_coeff_tex(x: float, sig: int = 4) -> str:
    """A numeric coefficient in engineering notation for LaTeX: 5.936e-5 ->
    '59.36\\,\\text{µ}', 2.94e7 -> '29.4\\,\\mathrm{M}'. No unit -- the prefix
    makes it comparable at a glance with the dcOp column (gm = 364 uS).

    Outside the prefix table (below atto, above tera -- cap products in
    a zero formula reach 1e-26) the old clamp glued a scientific-notation
    mantissa to the nearest prefix: '1.097e-08\\,a'. Such values render
    as a clean power of ten instead."""
    x = float(x)
    if x == 0 or not np.isfinite(x):
        return f"{x:g}"
    sign = "-" if x < 0 else ""
    ax = abs(x)
    if 0.1 <= ax < 1000:                    # comfortable range: plain reads better
        return f"{sign}{ax:.{sig}g}"        # (0.5 -> 0.5, not 500m)
    exp = int(np.floor(np.log10(ax) / 3) * 3)
    if not -18 <= exp <= 12:
        mant = f"{ax / 10.0 ** exp:.{sig}g}"
        return rf"{sign}{mant}\times 10^{{{exp}}}"
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
    if name.startswith("gmhat_"):
        # the exact gm+gmb bundle (mos_model='lumped-gmb'): the hat is
        # the tent over the two contributions. Deliberately NOT a tilde
        # -- in a tool that flags approximations with ≈, a tilde would
        # claim the opposite of the bundle's exactness.
        return rf"\hat{{g}}_{{m,{_inst_sub(name[6:], base, aliases)}}}"
    for pref, (letter, sub) in _SPECIAL.items():
        if name.startswith(pref + "_"):
            return rf"{letter}_{{{sub},{_inst_sub(name[len(pref) + 1:], base, aliases)}}}"
    if name[:4] in ("Ceq_", "Geq_", "Req_"):
        # a reduction's lumped equivalent at a node: G_{eq,net8}, never
        # the bare node name (the passive rule would show "net8")
        return rf"{name[0]}_{{eq,{_inst_sub(name[4:], base, aliases)}}}"
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
              aliases: dict | None = None, num_tags=None) -> str:
    """sympy.latex, but every numeric coefficient in engineering notation and
    every device symbol typeset via ``symbol_tex`` (``base`` picks leaf vs full
    instance names).

    With ``wrap`` each symbol is additionally tagged with its raw join-key name
    via KaTeX ``\\htmlData{sym=...}{...}`` -- an identity handle for the web
    view's hover/click, ignored by matplotlib mathtext (so it is off there).
    ``num_tags`` does the same for the NUMERALS: a str tags every numeric
    coefficient of this expression (an A0 or root-formula line), a dict maps
    individual Float atoms to their coefficient tag (num:k / den:k) -- the
    handle the Explain-the-numbers hover resolves.

    sympy has no eng-format option, so swap each Float for a placeholder symbol,
    render, then substitute the engineering string back. Integer exponents (s^2)
    are Integers, not Floats, so they're left untouched."""
    import sympy as sp

    if not hasattr(e, "atoms"):
        return sp.latex(e)
    floats = list(e.atoms(sp.Float))
    # a str tag is LINE-level: with several numerals on the line their
    # individual stories are not resolved (that would need per-kept-
    # monomial attribution), so the tag is starred and the tooltip says
    # the attribution is shared
    line_tag = num_tags if isinstance(num_tags, str) else None
    if line_tag and len(floats) > 1:
        line_tag += "*"
    subs, repl = {}, {}
    for i, f in enumerate(floats):
        t = _tok(i)
        subs[f] = sp.Symbol(t)
        val = _eng_coeff_tex(f)
        tag = line_tag if line_tag else (num_tags or {}).get(f)
        if wrap and tag:
            val = rf"\htmlData{{num={tag}}}{{{val}}}"
        repl[t] = val
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

#: Spectre's MOSFET region codes (dcOpInfo `region` parameter)
_REGIONS = {0: "off", 1: "triode", 2: "sat", 3: "subth", 4: "break"}

def region_name(code) -> str:
    try:
        return _REGIONS.get(int(code), str(code))
    except (TypeError, ValueError):
        return ""

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

def mirror_map(nodes):
    """Derive a fully-differential mirror map {p-side: n-side} from node
    names, for the mirrored compensation synthesis.

    Pairs names that are identical except for one character, where the p-side
    has 'p' and the n-side has 'n': outp/outn, I0.netp/I0.netn, n1p/n1n, and
    also the medial case outpi/outni. Only unambiguous pairs are kept, so a
    name that could mirror onto two different partners is dropped rather than
    guessed.

    Returns {} when the design has no such pairs, which is the single-ended
    case. This is a naming heuristic, not a topological one; the caller shows
    the derived pairs so the designer can see what was matched.
    """
    names = sorted(set(nodes))
    hits = {}
    for n in names:
        for i, ch in enumerate(n):
            if ch != "p":
                continue
            twin = n[:i] + "n" + n[i + 1:]
            if twin in names and twin != n:
                hits.setdefault(n, set()).add(twin)
    # keep only unambiguous pairs, and only one direction (p -> n)
    return {n: next(iter(t)) for n, t in hits.items() if len(t) == 1}
