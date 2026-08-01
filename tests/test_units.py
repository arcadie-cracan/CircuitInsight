"""One engineering formatter for every human-facing figure.

The summary, the template's designer numbers, pole attribution and the
mode margins all print through circuitinsight.units.eng, so 2.046e4 Hz
is always 20.5 kHz. The formatter must also refuse to lie outside its
prefix table: cap products in a zero formula reach 1e-26, and the old
clamp printed them as '1.097e-08 a' -- scientific notation glued to
the atto prefix.
"""
from circuitinsight.units import eng


def test_eng_basics():
    assert eng(20460, "Hz") == "20.5 kHz"
    assert eng(6.38e-14, "F") == "63.8 fF"
    assert eng(1.27e7, "Hz") == "12.7 MHz"
    assert eng(202.454, sig=4) == "202.5"          # no unit: no separator
    assert eng(-202.454, sig=4) == "-202.5"
    assert eng(0, "Hz") == "0Hz"


def test_eng_refuses_to_lie_outside_the_prefix_table():
    assert "a" not in eng(1.097e-26)               # the '1.097e-08 a' bug
    assert eng(1.097e-26) == "1.1e-26"
    assert eng(3e15, "Hz") == "3e+15 Hz"


def test_tex_coefficients_use_powers_of_ten_outside_the_table():
    from circuitinsight.gui.view import _eng_coeff_tex

    assert _eng_coeff_tex(5.936e-5) == r"59.36\,\mathrm{µ}"  # upright, not \mu
    out = _eng_coeff_tex(1.097e-26)
    assert r"\times 10^{-27}" in out and r"\mathrm{a}" not in out
    assert _eng_coeff_tex(-1.097e-26).startswith("-")


def test_expression_coefficients_read_in_engineering_notation():
    """The template's per-root formulas print their collapsed numerals
    with prefixes: 4.83174808323383e-5*gds reads 48.32 u*gds."""
    import sympy as sp

    from circuitinsight.units import eng_expr_str

    gds, CL, gm, s = sp.symbols("gds CL gm s")
    out = eng_expr_str(sp.Float("4.83174808323383e-5") * gds / (CL * gm))
    assert "48.32" in out and "u" in out and "e-5" not in out
    assert eng_expr_str(sp.Float(20667.5) * gm) .startswith("20.67")
    assert eng_expr_str(s ** 2 + gm) == "gm + s**2"    # exponents untouched


def test_designer_numbers_read_in_engineering_notation():
    """The template and attribution describe() lines the Summary shows."""
    from circuitinsight.analysis.attribution import PoleAttribution, PoleOwner
    from circuitinsight.analysis.template import RootFormula, TemplateForm

    tpl = TemplateForm(dc_gain=None, dc_gain_value=202454.0,
                       gbw_hz=1.27e7,
                       poles=[RootFormula(kind="pole", index=1,
                                          f_hz=20460.0, damping=None,
                                          expr=None, displacement=0.0,
                                          reliable=True)])
    text = tpl.describe()
    assert "202.5 k" in text
    assert "12.7 MHz" in text
    assert "20.5 kHz" in text

    att = PoleAttribution(f_hz=20460.0, damping=None,
                          owners=[PoleOwner("gds_M1", 0.62)], verified=True)
    assert att.describe().startswith("20.5 kHz")
