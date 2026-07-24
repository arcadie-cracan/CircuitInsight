"""Pure presentation helpers (gui/view.py): no Qt, plain Agg backend."""
import warnings
from pathlib import Path

import pytest

# view.py needs matplotlib (an optional [gui] dep). Guard it, or an install
# without the extra fails at COLLECTION and takes the whole suite down with it.
matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from circuitinsight import SessionController
from circuitinsight.gui import view

FIX = Path(__file__).resolve().parent / "fixtures" / "spectre" / "ota5t"
MILLER = Path(__file__).resolve().parent / "fixtures" / "spectre" / "miller"


@pytest.fixture(scope="module")
def session():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return SessionController.open(FIX / "tb_ota5t.cin.json", FIX / "psf")


@pytest.fixture(scope="module")
def result(session):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return session.solve("VIND", "vout", [])


def test_bode_figure(result):
    from matplotlib.figure import Figure

    fig = view.bode_figure(result)
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 2                        # magnitude + phase
    assert len(fig.axes[0].lines) >= 2               # model + AC reference


def test_summary_and_table(result):
    s = view.summary_text(result)
    assert "DC gain" in s and "dB" in s and "poles" in s
    rows = view.poles_table(result)
    assert rows and all(len(r) == 3 for r in rows)


def test_expr_figure_renders(result):
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    fig = view.expr_figure(result)
    FigureCanvasAgg(fig).draw()          # forces mathtext parsing; raises if bad
    assert fig.axes


def test_markdown_report(result):
    md = view.markdown_report(result)
    assert md.startswith("# CircuitInsight")
    assert "A_0" in md and "H(s)" in md


def test_ranking_rows():
    ranking = [("gm_M1", 12.3, 1.2e6), ("Cc", 4.5, 3.8e6)]
    values = {"gm_M1": 3.64e-4, "Cc": 6.38e-14}
    rows = view.ranking_rows(ranking, values)
    # (name, dcOp, score, peaks); eng() uses a thin space, so normalize
    def norm(s):
        return s.replace(" ", " ")
    assert rows[0][0] == "gm_M1"
    assert norm(rows[0][1]) == "364 uS"                    # gm in siemens
    assert norm(rows[1][1]) == "63.8 fF"                   # cap in femtofarad
    assert "M" in rows[0][3]                               # peaks @ 1.2 MHz
    # values optional -> dcOp blank, still 4 columns
    assert view.ranking_rows(ranking)[0][1] == ""


def test_op_unit():
    assert view.op_unit("gm_M1") == "S"
    assert view.op_unit("gds_M2") == "S"
    assert view.op_unit("cdb_M3") == "F"
    assert view.op_unit("Cc") == "F"
    # passive value symbols keyed by a hierarchical instance name: the element
    # type is the leaf, not the I0_ prefix (which used to read as amps)
    assert view.op_unit("I0_Cc") == "F"
    assert view.op_unit("CL") == "F"
    assert view.op_unit("I0_Rz") == "Ω"
    assert view.op_unit("I0.Cc") == "F"


def test_symbol_tex():
    # quantity typeset (g_m, g_ds, c_gd); instance leaf as subscript in base mode
    assert view.symbol_tex("gm_I0_MN1") == r"g_{m,\mathrm{MN1}}"
    assert view.symbol_tex("gds_I0_MP2") == r"g_{ds,\mathrm{MP2}}"
    assert view.symbol_tex("cgd_I0_MP2") == r"c_{gd,\mathrm{MP2}}"
    assert view.symbol_tex("gpi_Q1") == r"g_{\pi,\mathrm{Q1}}"     # greek subscript
    # passive value symbols: plain device name, no I0_ prefix reading as current
    assert view.symbol_tex("I0_Cc") == r"\mathrm{Cc}"
    assert view.symbol_tex("CL") == r"\mathrm{CL}"
    # full mode keeps the hierarchy for disambiguation
    assert view.symbol_tex("gm_I0_MN1", base=False) == r"g_{m,\mathrm{I0.MN1}}"
    assert view.symbol_tex("I0_Cc", base=False) == r"\mathrm{I0.Cc}"


def test_summary_simplified(result):
    import dataclasses

    r2 = dataclasses.replace(result, simplified=True, mag_err_db=0.4,
                             phase_err_deg=1.2, n_terms_full=99)
    s = view.summary_text(r2)
    assert "pruned within" in s and "0.400 dB" in s and "from 99" in s


def test_eng():
    assert view.eng(1235, "Hz").endswith("kHz")
    assert view.eng(12.9e6, "Hz").endswith("MHz")
    assert view.eng(0, "Hz").endswith("Hz")


def test_keep_set_never_changes_accuracy(session):
    """The most misread control in the tool.

    `keep` selects which parameters stay symbolic; the rest become the EXACT
    rationals of their OP values. So every keep set is exact and reproduces the
    same response. Only simplify() trades accuracy. If this ever fails, either
    the hybrid solve stopped being exact, or the keep set started approximating —
    both are serious.
    """
    ranked = [n for n, _, _ in session.rank_symbols("VIND", "vout")]
    fids = []
    for k in (0, 2, 4):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = session.solve("VIND", "vout", ranked[:k])
        fids.append(view.fidelity(r))
        assert r.keep == ranked[:k]

    # identical to well within numerical noise, across a 10x change in term count
    for f in fids[1:]:
        assert f[0] == pytest.approx(fids[0][0], abs=1e-6)
        assert f[1] == pytest.approx(fids[0][1], abs=1e-6)


def test_simplify_is_the_accuracy_knob(session):
    """A bigger budget must actually buy a shorter expression."""
    ranked = [n for n, _, _ in session.rank_symbols("VIND", "vout")][:4]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tight = session.simplify("VIND", "vout", ranked, mag_db=0.1, phase_deg=0.5)
        loose = session.simplify("VIND", "vout", ranked, mag_db=3.0, phase_deg=15.0)
    assert loose.n_terms < tight.n_terms          # brevity bought with error
    assert loose.n_terms < loose.n_terms_full


def test_kept_capacitor_appears_in_dominant_pole(session):
    """A kept capacitance must read as a LETTER somewhere -- the user's complaint
    was keeping a cap and seeing no cap symbol.

    It cannot appear in the factored N(s)/D(s) (numeric roots of a symbolic
    quartic have no closed form), but it MUST appear in the symbolic dominant
    pole/zero, and on a plain Solve, not only after Simplify.
    """
    session.set_matches(*session.suggest_matches())
    keep = ["gm_I0_MN1", "gds_I0_MN1", "gds_I0_MP1", "cdb_I0_MP1"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = session.solve("VIND", "vout", keep)        # plain solve, no simplify
    assert not r.simplified

    lines = view._expr_lines(r)
    labels = {lbl.strip(): tex for lbl, tex in lines}
    assert "p_1 =" in labels, "no symbolic dominant pole shown"
    assert "c_{db" in labels["p_1 ="], "kept capacitance absent from p_1"
    # and A_0, being H(0), must be cap-free (a cap is open at DC)
    assert "c_{db" not in labels.get("A_0 =", "")
    # the full 5th-order solve is too big for the expanded symbolic form -- it
    # must fall back to numeric corners, not print a many-digit-integer ratio
    assert view._raw_tf_lines(r) == []


@pytest.fixture(scope="module")
def miller_reduced():
    """The paper's reduced 2nd-order two-stage solve, shared across tests."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c = SessionController.open(MILLER / "tb_ota2s.cin.json", MILLER / "psf")
        c.set_matches(*c.suggest_matches())
        keep = ["gm_I0_MN1", "gm_I0_MP2", "I0_Cc", "CL",
                "gds_I0_MN1", "gds_I0_MP1", "gds_I0_MP2", "gds_I0_MN3"]
        return c.reduce_solve("VIND", "vout", keep, max_elements=2,
                              fmin=1e3, fmax=1e7)


def test_reduced_solve_shows_symbolic_eq4_and_signs(miller_reduced):
    """The reduced 2nd-order solve is compact, so the Expression tab shows H(s)
    with SYMBOLIC coefficients -- the paper's eq. (4),
    g_m1(g_m5 - C_C s)/(C_C C_L s^2 + C_C g_m5 s + G_o1 G_o2) -- not numeric
    corners; and the dominant pole/zero carry s-plane signs (LHP pole negative,
    RHP zero positive), matching eq. (6) and the (1 +/- s/w) corner form."""
    r = miller_reduced
    lines = view._expr_lines(r)
    labels = {lbl.strip(): tex for lbl, tex in lines}

    # eq (4): symbolic N(s)/D(s) with the kept caps/conductances as letters,
    # typeset in base form (leaf names, g_m/g_ds, plain passive Cc)
    assert "N(s) =" in labels and "D(s) =" in labels
    assert r"\mathrm{Cc}" in labels["N(s) ="] and "g_{m," in labels["N(s) ="]
    assert "s^{2}" in labels["D(s) ="] and "g_{ds," in labels["D(s) ="]
    assert any("exact\\ symbolic" in tex for _, tex in lines)   # not the corner note

    # eq (6) signs mirror the numeric s-plane roots
    assert r.poles_hz[0].real < 0                     # dominant pole in the LHP
    assert max(z.real for z in r.zeros_hz) > 0        # feedforward zero in the RHP
    def _neg(tex):                       # sign may sit outside or inside a frac
        return "-" in tex[:15]           # p_1 negative, z_1 positive
    assert _neg(labels["p_1 ="])          # -> p_1 shown negative (LHP)
    assert not _neg(labels["z_1 ="])      # -> z_1 shown positive (RHP)


def test_expr_katex_payload(miller_reduced):
    """The web-view payload: every device symbol identity-tagged for hover/click,
    values eng-formatted with the right unit -- while the matplotlib path stays
    tag-free (mathtext would choke on \\htmlData)."""
    p = view.expr_katex(miller_reduced)
    assert any(r"\htmlData{sym=gm_I0_MN1}" in ln for ln in p["lines"])
    assert any(r"\htmlData{sym=I0_Cc}" in ln for ln in p["lines"])
    assert p["values"]["I0_Cc"].endswith("pF")            # 14.7 pF, not pA
    assert p["values"]["gm_I0_MN1"].endswith("uS")
    # full-hierarchy variant still tags with the same raw identity
    pf = view.expr_katex(miller_reduced, base=False)
    assert any(r"\htmlData{sym=gm_I0_MN1}{g_{m,\mathrm{I0.MN1}}}" in ln
               for ln in pf["lines"])
    # and the mathtext path must never see the tags
    assert not any("htmlData" in tex
                   for _, tex in view._expr_lines(miller_reduced))


def test_eng_coeff_tex():
    assert view._eng_coeff_tex(5.936e-5) == r"59.36\,\mu"
    assert view._eng_coeff_tex(2.936e7) == r"29.36\,\mathrm{M}"
    assert view._eng_coeff_tex(-6.38e-14) == r"-63.8\,\mathrm{f}"
    assert view._eng_coeff_tex(0.5) == "0.5"              # comfortable range: plain
    assert view._eng_coeff_tex(1764.0) == r"1.764\,\mathrm{k}"


def test_latex_eng_no_scientific_notation():
    """Coefficients render in engineering prefixes, not sympy's a*10^b."""
    import sympy as sp
    gm, gds = sp.symbols("gm gds")
    e = (gm + sp.Float(5.936e-5)) / (gds + sp.Float(6.38e-14))
    s = view.latex_eng(e)
    assert r"\cdot 10^{" not in s                         # no scientific notation
    assert r"\mu" in s and r"\mathrm{f}" in s             # micro and femto present
    assert "engx" not in s                                # no leftover placeholder


def test_symbol_tex_covers_every_param_family():
    """gmbs once fell through to the passive branch and rendered as a bare
    instance name (A0 showed 'gm_MP0 + MP0'); gii_d/gii_m partitioned at
    the wrong underscore. Every join-key family must produce its
    conventional subscripted symbol."""
    from circuitinsight.gui.view import symbol_tex

    assert symbol_tex("gmbs_I0_MN0") == r"g_{mb,\mathrm{MN0}}"
    assert symbol_tex("gii_d_MN0") == r"g_{ii\,d,\mathrm{MN0}}"
    assert symbol_tex("gii_m_MP0") == r"g_{ii\,m,\mathrm{MP0}}"
    assert symbol_tex("kdg_I0_MN2") == r"k_{dg,\mathrm{MN2}}"
    assert symbol_tex("csub_Q3") == r"c_{sub,\mathrm{Q3}}"
    # passives keep rendering as plain device names
    assert symbol_tex("I0_Cc") == r"\mathrm{Cc}"


def test_latex_eng_coefficient_first_with_cdot():
    """The r2r denominator once rendered as 'RSP445 p': the placeholder
    token sorted after uppercase symbols (so the coefficient printed
    second) and math mode swallowed the lone space between factors. Now
    coefficients sort first and adjacent factors get an explicit cdot;
    the mu prefix no longer glues into micrograms."""
    import sympy as sp

    from circuitinsight.gui import view

    RSP, gm = sp.Symbol("RSP"), sp.Symbol("gm_MP0")
    den = RSP * sp.Float("4.45e-10") + sp.Symbol("gm_MN0")
    out = view.latex_eng(den)
    assert r"445\,\mathrm{p} \cdot \mathrm{RSP}" in out
    prod = sp.Float("2.576e-5") * gm
    out2 = view.latex_eng(prod)
    assert out2 == r"25.76\,\mu \cdot g_{m,\mathrm{MP0}}"


def test_round_expr_strips_a_common_balun_half():
    """A large A_0 where a balun's 0.5 multiplies EVERY term of numerator
    and denominator: sympy's float cancel/factor miss the common float,
    so round_expr strips it explicitly (a common factor is not part of
    the answer). A genuine mixed-coefficient ratio is left untouched."""
    import sympy as sp

    from circuitinsight.gui import view

    RSN, RSP = sp.symbols("RSN RSP")
    gens = sp.symbols("a b c d e f g h")
    half = sp.Float("0.5")
    num = sp.Add(*[half * RSN * RSP * x * y for x in gens
                   for y in gens[:4]][:30], half * sp.Float("5.7e-13") * RSP)
    den = sp.Add(*[half * RSN * RSP * x * y for x in gens
                   for y in gens[:5]][:36], half * sp.Symbol("i"))
    out = str(view.round_expr(num / den, factored=True))
    assert "0.5" not in out

    # a real mixed ratio (no shared scalar) must NOT be rescaled: its
    # coefficients survive (2 and 5 not divided down)
    mixed = (2 * gens[0] + 5 * gens[1]) / (3 * gens[2] + gens[3])
    sub = {g: v for g, v in zip(gens[:4], (3.0, 5.0, 2.0, 7.0))}
    assert float(view.round_expr(mixed).xreplace(sub)) == \
        pytest.approx(float(mixed.xreplace(sub)))
    assert "2" in str(view.round_expr(mixed))       # coeff 2 not stripped


def test_drop_common_scale_preserves_value():
    """Stripping the common factor cannot change what the expression
    evaluates to."""
    import sympy as sp

    from circuitinsight.gui import view

    a, b, c = sp.symbols("a b c")
    e = (sp.Rational(1, 2) * a + sp.Rational(1, 2) * b) / (sp.Rational(1, 2) * c)
    subs = {a: 3.0, b: 5.0, c: 2.0}
    assert float(view.round_expr(e).xreplace(subs)) == \
        pytest.approx(float(e.xreplace(subs)))
