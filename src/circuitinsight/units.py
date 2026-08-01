"""Engineering notation for every human-facing figure.

One formatter for the whole tree: analysis describe() methods and the
GUI both print through this, so a frequency is always "20.5 kHz" and a
capacitance "63.8 fF" -- never a bare 2.046e4. Lives outside gui/ so
the analysis layer can use it without importing Qt or matplotlib.
"""
from __future__ import annotations

import math

_PREFIXES = {-18: "a", -15: "f", -12: "p", -9: "n", -6: "u", -3: "m",
             0: "", 3: "k", 6: "M", 9: "G", 12: "T"}


def eng_expr_str(expr) -> str:
    """sympy sstr with numeric coefficients in engineering notation:
    4.83174808323383e-5*gds -> '48.32 u*gds'. Small integers (powers,
    counts) and values in the comfortable 0.1..1000 range stay as they
    are. The LaTeX path does the same through view._eng_coeff_tex; this
    is its plain-text sibling for describe() lines."""
    import sympy as sp

    reps = {}
    for a in expr.atoms(sp.Float, sp.Rational):
        if a.is_Integer and abs(int(a)) < 1000:
            continue                          # exponents and counts
        try:
            val = float(a)
        except (TypeError, ValueError):
            continue
        if val != 0 and not 0.1 <= abs(val) < 1000:
            reps[a] = sp.Symbol(eng(val, sig=4))
    return sp.sstr(expr.xreplace(reps) if reps else expr)


def eng(x: float, unit: str = "", *, sig: int = 3) -> str:
    """Engineering notation, e.g. '1.24 kHz', '12.9 MHz', '63.8 fF'.
    The separator is a thin space (U+2009), matching the GUI's historic
    output."""
    x = float(x)
    if x == 0 or not math.isfinite(x):
        return f"{x:g}{unit}"
    exp = int(math.floor(math.log10(abs(x)) / 3) * 3)
    if not -18 <= exp <= 12:
        # outside the prefix table (below atto, above tera): a clamped
        # prefix would glue scientific notation to a wrong prefix
        # ('1.097e-08 a'); plain scientific is the honest form
        return f"{x:.{sig}g}{' ' if unit else ''}{unit}"
    suffix = _PREFIXES[exp] + unit
    if not suffix:
        return f"{x / 10 ** exp:.{sig}g}"
    return f"{x / 10 ** exp:.{sig}g} {suffix}"
