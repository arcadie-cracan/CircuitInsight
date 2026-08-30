"""L-A: the margin ledger -- exact per-singularity PM/GM attribution.

Analytic rational loops pin every share against its closed form; the
certificate (shares sum to the direct evaluation) guards the branch
bookkeeping that closed-form continuity requires, in particular the
RHP-pair full-turn restoration; the miller fixture ties the ledger to a
real solved stb loop gain.
"""
import warnings
from pathlib import Path

import numpy as np
import pytest

from circuitinsight.analysis.marginledger import margin_ledger

FIX = Path(__file__).resolve().parent / "fixtures" / "spectre" / "miller"
NMC3 = Path(__file__).resolve().parent / "fixtures" / "spectre" / "nmc3"


class _Rat:
    """A minimal TransferFunction stand-in with known singularities."""

    def __init__(self, K, poles=(), zeros=()):
        self._K = complex(K)
        self._p = [complex(p) for p in poles]
        self._z = [complex(z) for z in zeros]

    def poles(self):
        return np.array(self._p)

    def zeros(self):
        return np.array(self._z)

    def numeric(self, f):
        jf = 1j * np.asarray(f, dtype=float)
        out = np.full(jf.shape, self._K, complex)
        for z in self._z:
            out = out * (jf - z)
        for p in self._p:
            out = out / (jf - p)
        return out


def _stb_gain(A0, poles, zeros=()):
    """K making T(0) = -A0 -- the stb convention's +180 deg anchor."""
    K = -A0
    for p in poles:
        K *= -p
    for z in zeros:
        K /= -z
    return K


def test_two_pole_shares_are_the_arctans():
    """The classic hand decomposition: PM = 180 - sum of arctan(fu/fk),
    each pole's entry carrying exactly its arctan."""
    p1, p2, A0 = -1e3, -1e6, 1e3
    led = margin_ledger(_Rat(_stb_gain(A0, [p1, p2]), poles=[p1, p2]))
    fu = led.f_unity_hz
    assert fu is not None
    want = {abs(p1): -np.degrees(np.arctan(fu / abs(p1))),
            abs(p2): -np.degrees(np.arctan(fu / abs(p2)))}
    got = {e.f0_hz: e.phase_deg for e in led.entries if e.kind == "pole"}
    for f0, ph in want.items():
        assert got[f0] == pytest.approx(ph, abs=1e-9)
    anchor = led.entries[0]
    assert anchor.kind == "gain" and anchor.phase_deg == pytest.approx(180)
    assert led.pm_deg == pytest.approx(180 + sum(want.values()), abs=1e-9)
    assert led.phase_residual_deg < 1e-9
    # a two-pole loop never reaches the critical phase
    assert led.gm_db is None and led.f_180_hz is None


def test_three_pole_gm_decomposes_in_db():
    """At f_180 the gain margin is a sum of per-pole dB climbs plus the
    DC term, each entry exactly -10 log10(1 + (f/fk)^2)."""
    poles, A0 = [-1e3, -1e6, -3e6], 1e3
    led = margin_ledger(_Rat(_stb_gain(A0, poles), poles=poles))
    fg = led.f_180_hz
    assert fg is not None and led.gm_db is not None
    for e in led.entries:
        if e.kind == "pole":
            assert e.mag_db == pytest.approx(
                -10 * np.log10(1 + (fg / e.f0_hz) ** 2), abs=1e-9)
    t0 = led.entries[0].mag_db
    assert t0 == pytest.approx(20 * np.log10(A0), abs=1e-9)
    total = t0 + sum(e.mag_db for e in led.entries if e.kind == "pole")
    assert led.gm_db == pytest.approx(-total, abs=1e-9)
    assert led.mag_residual_db < 1e-9
    # the phase shares RECONSTRUCT the margin: they are shares of
    # arg T at f_u, gain anchor included, so they sum to the phase
    # margin itself. (This computed the sum and asserted nothing,
    # under a comment claiming a different invariant -- shares at
    # f_180 summing to zero -- which is not what the field holds.)
    acc = sum(e.phase_deg for e in led.entries)
    assert acc == pytest.approx(led.pm_deg, abs=1e-9)


def test_rhp_zero_spends_phase():
    """An RHP zero lags like a pole: its share is -arctan(fu/fz),
    flagged rhp, while an LHP zero at the same frequency would give it
    back."""
    p, z, A0 = [-1e3, -1e6], 5e5, 1e3
    led = margin_ledger(_Rat(_stb_gain(A0, p, [z]), poles=p, zeros=[z]))
    fu = led.f_unity_hz
    ze = next(e for e in led.entries if e.kind == "zero")
    assert ze.rhp and not ze.pair
    assert ze.phase_deg == pytest.approx(
        -np.degrees(np.arctan(fu / z)), abs=1e-9)
    assert led.phase_residual_deg < 1e-9


def test_doublet_pair_is_one_entry():
    """A complex pair merges into one (f0, zeta) entry whose share is
    the second-order -atan2(2 zeta u, 1 - u^2) -- the doublet's phase
    bite as a single readable line."""
    f0, zeta, A0 = 2e6, 0.25, 1e3
    pair = [f0 * complex(-zeta, +np.sqrt(1 - zeta ** 2)),
            f0 * complex(-zeta, -np.sqrt(1 - zeta ** 2))]
    poles = [-1e3] + pair
    led = margin_ledger(_Rat(_stb_gain(A0, poles), poles=poles))
    fu = led.f_unity_hz
    pe = next(e for e in led.entries if e.pair)
    assert pe.zeta == pytest.approx(zeta, abs=1e-12)
    assert pe.f0_hz == pytest.approx(f0, rel=1e-12)
    u = fu / f0
    assert pe.phase_deg == pytest.approx(
        -np.degrees(np.arctan2(2 * zeta * u, 1 - u * u)), abs=1e-9)
    assert led.phase_residual_deg < 1e-9


def test_rhp_pair_branch_stays_continuous():
    """An RHP pair whose imaginary part sits BELOW the crossing walks
    the atan2 branch cut; the closed-form turn restoration must keep
    the certificate at zero and the entry inside a physical range."""
    f0, zeta = 1e5, 0.3
    pair = [f0 * complex(+zeta, +np.sqrt(1 - zeta ** 2)),
            f0 * complex(+zeta, -np.sqrt(1 - zeta ** 2))]
    poles = [-1e3, -3e6] + pair
    led = margin_ledger(_Rat(_stb_gain(1e3, poles), poles=poles))
    assert led.f_unity_hz is not None and led.f_unity_hz > f0
    assert led.phase_residual_deg < 1e-9
    pe = next(e for e in led.entries if e.pair)
    assert pe.rhp
    # an RHP pole pair LEADS (positive share), approaching +180 past f0
    assert 90 < pe.phase_deg <= 180


def test_fold_gathers_the_tail_within_budget():
    """fold() keeps the story short: the gathered remainder never
    exceeds its declared budgets, and the kept terms plus the rest
    still sum to the whole."""
    poles = [-1e3, -8e5, -1e6, -2e6, -60e6, -70e6, -80e6, -90e6]
    led = margin_ledger(_Rat(_stb_gain(1e3, poles), poles=poles))
    folded = led.fold(budget_deg=3.0, budget_db=1e9)
    rest = [e for e in folded if e.kind == "rest"]
    assert rest and rest[0].count >= 2
    assert abs(rest[0].phase_deg) <= 3.0
    total = sum(e.phase_deg for e in folded)
    exact = sum(e.phase_deg for e in led.entries)
    assert total == pytest.approx(exact, abs=1e-9)


@pytest.fixture(scope="module")
def miller():
    """The solved miller stb loop: (analyzer, system, T), shared by the
    fixture-level tests."""
    from circuitinsight.adapters.spectre import SpectreRun
    from circuitinsight.analysis.loopgain import loop_gain
    from circuitinsight.engine.mna import build_mna

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run = SpectreRun(FIX / "tb_ota2s_stb.cin.json", FIX / "psf_stb")
        an = run.analyzer(cap_model="matrix")
        system = build_mna(an.primitives, an.flat.ground, "IPRB0",
                           an._alias)
        T = loop_gain(system, "IPRB0", [])
    return an, system, T


def test_element_sensitivities_match_finite_difference(miller):
    """L-C against reality: each row's dPM and dGM per percent must
    match an INDEPENDENT +-1% re-solve through the full chain
    (build_mna, loop_gain, refined crossings) -- the implicit
    crossing-shift term included, or the comparison fails at the first
    decimal."""
    import dataclasses

    from circuitinsight.analysis.loopgain import loop_gain
    from circuitinsight.analysis.marginledger import margin_sensitivities
    from circuitinsight.engine.mna import build_mna, symbol_name

    an, system, T = miller
    led = margin_ledger(T)
    sens = margin_sensitivities(system, an.primitives, "IPRB0", led,
                                alias=an._alias)
    assert sens
    assert all(r["dpm_deg_per_pct"] is not None
               and r["dgm_db_per_pct"] is not None for r in sens)
    # L-J actionability: netlist components are the exact knobs,
    # device parameters are re-bias advice -- and each row knows its
    # owning instance for the device grouping
    by = {r["symbol"]: r for r in sens}
    assert by["CL"]["knob"] == "component"
    assert by["gm_I0_MP2"]["knob"] == "device"
    assert by["gm_I0_MP2"]["insts"] == ("I0.MP2",)
    assert by["gm_I0_MP2"]["param"] == "gm"
    cc = [r["symbol"] for r in sens if "Cc" in r["symbol"]]
    assert cc                       # the Miller cap is on the ledger
    assert by[cc[0]]["knob"] == "component"
    for name in {sens[0]["symbol"], cc[0]}:
        row = next(r for r in sens if r["symbol"] == name)
        pms, gms = [], []
        for k in (-0.01, +0.01):
            prims = [dataclasses.replace(p, value=p.value * (1 + k))
                     if (p.value is not None
                         and symbol_name(p, an._alias) == name) else p
                     for p in an.primitives]
            s2 = build_mna(prims, an.flat.ground, "IPRB0", an._alias)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                led2 = margin_ledger(loop_gain(s2, "IPRB0", []))
            pms.append(led2.pm_deg)
            gms.append(led2.gm_db)
        assert row["dpm_deg_per_pct"] == pytest.approx(
            (pms[1] - pms[0]) / 2.0, rel=0.02, abs=1e-4)
        assert row["dgm_db_per_pct"] == pytest.approx(
            (gms[1] - gms[0]) / 2.0, rel=0.02, abs=1e-4)


def test_root_sensitivities_name_the_textbook_devices(miller):
    """L-G attribution by sensitivity: the miller RHP zero sits at
    ~gm_MP2/Cc, so its relative sensitivities must read ~+1 %/% to the
    second stage's gm and ~-1 %/% to the Miller cap -- the textbook
    expression recovered as measured ownership. The dominant pole (a
    minor-combination root with no modal identity) gets sensitivities too,
    led by the Miller-multiplied Cc."""
    from circuitinsight.analysis.marginledger import (
        attach_root_sensitivities)

    an, system, T = miller
    led = margin_ledger(T, system=system)
    attach_root_sensitivities(led, system, an.primitives, "IPRB0", alias=an._alias)
    z = next(e for e in led.entries if e.kind == "zero" and e.rhp)
    assert z.root_sens
    by = {m["symbol"]: m for m in z.root_sens}
    gm2 = by.get("gm_I0_MP2")
    cc = next((m for s, m in by.items() if "Cc" in s), None)
    assert gm2 is not None and cc is not None
    assert 0.5 < gm2["df0"] < 1.3
    assert -1.3 < cc["df0"] < -0.5
    dom = next(e for e in led.entries
               if e.kind == "pole" and e.phase_deg is not None
               and abs(e.phase_deg + 90) < 2)
    assert dom.root_sens
    assert any("Cc" in m["symbol"] and m["df0"] < -0.5
               for m in dom.root_sens)
    # every sensitivity row names the HOT member instances of its symbol -- the
    # cross-light's answer to matched-group over-lighting
    for e in (z, dom):
        for m in e.root_sens:
            assert m["insts"]
        for t in e.taus:
            assert "insts" in t
    # below the share floor no movers are computed
    assert all(e.root_sens == () for e in led.entries
               if e.phase_deg is not None
               and abs(e.phase_deg) < 2.0
               and (e.mag_db is None or abs(e.mag_db) < 0.5))


def test_tau_ledger_is_exact_and_names_the_zero(miller):
    """L-H: scaling every capacitor by k scales every singularity by
    1/k, so the per-cap time constants tau_i = C_i d(-1/s)/dC_i sum
    EXACTLY to -1/s (Euler) -- residuals at machine precision, not an
    OCT approximation. The miller RHP zero's tau story is 99% Cc
    against R_eff = -1/gm_MP2 (the negative Miller feedforward
    resistance), and the named estimate is the textbook formula
    gm_MP2/(2 pi Cc) with a certified error."""
    from circuitinsight.analysis.marginledger import (
        attach_root_sensitivities)

    an, system, T = miller
    led = margin_ledger(T, system=system)
    attach_root_sensitivities(led, system, an.primitives, "IPRB0", alias=an._alias)
    with_taus = [e for e in led.entries if e.taus]
    assert with_taus
    for e in with_taus:
        assert e.tau_residual < 1e-9          # Euler, machine-exact
        # the stored top-4 is a truncation; only the FULL sum is 1
        assert sum(t["share"] for t in e.taus) <= 1.1
    z = next(e for e in led.entries if e.kind == "zero" and e.rhp)
    lead = z.taus[0]
    assert "Cc" in lead["symbol"] and lead["share"] > 0.9
    gm2 = system.values["gm_I0_MP2"]
    assert lead["r_eff"] < 0                  # RHP: negative seen R
    assert abs(lead["r_eff"]) * gm2 == pytest.approx(1.0, rel=0.15)
    assert z.estimate is not None
    assert "gm_I0_MP2" in z.estimate["text"]
    assert "Cc" in z.estimate["text"]
    assert z.estimate["err"] < 0.1
    assert z.estimate["f_est_hz"] == pytest.approx(z.f0_hz, rel=0.1)
    # the dominant pole's tau is the Miller-multiplied Cc
    dom = next(e for e in led.entries
               if e.kind == "pole" and e.phase_deg is not None
               and abs(e.phase_deg + 90) < 2)
    assert "Cc" in dom.taus[0]["symbol"]
    assert dom.taus[0]["share"] > 0.9
    assert dom.taus[0]["r_eff"] > 1e6         # gm ro ro territory


def test_complexity_knob_sets_how_small_a_named_mover_may_be():
    """There is no search to widen any more: the jet reads one
    expression, and the knob only sets the floor below which a mover
    is left to the coverage residual rather than named. A richer
    setting must therefore never name FEWER elements."""
    from circuitinsight.analysis.marginledger import (
        attach_root_sensitivities)
    from circuitinsight.adapters.spectre import SpectreRun
    from circuitinsight.analysis.loopgain import loop_gain
    from circuitinsight.engine.mna import build_mna

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run = SpectreRun(FIX / "tb_ota2s_stb.cin.json", FIX / "psf_stb")
        an = run.analyzer(cap_model="matrix")
        system = build_mna(an.primitives, an.flat.ground, "IPRB0",
                           an._alias)
        T = loop_gain(system, "IPRB0", [])
    named = {}
    for level in (1, 3):
        led = margin_ledger(T, system=system)
        attach_root_sensitivities(led, system, an.primitives, "IPRB0",
                                  alias=an._alias, complexity=level)
        named[level] = {
            round(e.f0_hz): {n for b in (e.reconstruction.blocks
                                         if e.reconstruction else ())
                             for n in b.members}
            for e in led.entries if e.reconstruction}
    assert named[1] and named[3]
    for f0, low in named[1].items():
        assert low <= named[3].get(f0, set()), f0


def test_margin_equation_certifies_and_predicts(miller):
    """L-I: the margin equation. Structure certificate first -- the
    equation with EXACT frequencies reproduces the ledger's PM to
    numerical precision -- then the designer's version: estimated
    frequencies predict PM within a couple of degrees on miller, the
    RHP zero term carries its gm/2piCc formula, the anchor reads +180
    as the hand equation does, and the GBW relation f_u ~ T0 f_dom
    holds within its gate."""
    from circuitinsight.analysis.marginledger import (
        attach_root_sensitivities, margin_equation,
        margin_equation_latex,
        margin_equation_text)

    an, system, T = miller
    led = margin_ledger(T, system=system)
    attach_root_sensitivities(led, system, an.primitives, "IPRB0", alias=an._alias)
    eq = margin_equation(led)
    assert abs(eq["pm_check_deg"] - eq["pm_exact_deg"]) < 1e-9
    assert eq["pm_err_deg"] < 3.0
    assert eq["anchor_deg"] == pytest.approx(180.0, abs=1e-6)
    # the hand-equation convention: terms and their subscripts run in
    # ascending frequency, poles and zeros counted low-to-high
    fall = [t["f0_hz"] for t in eq["terms"]]
    assert fall == sorted(fall)
    zterm = next(t for t in eq["terms"]
                 if t["kind"] == "zero" and t["rhp"])
    assert zterm["sign"] < 0                  # an RHP zero spends phase
    assert zterm["estimate"] is not None
    assert "gm_I0_MP2" in zterm["estimate"]["text"]
    assert eq["fu_est"] is not None and eq["fu_est"]["err"] < 0.3
    text = margin_equation_text(eq)
    assert text[0].startswith("PM ~ 180")
    assert "(exact" in text[0]
    # every theta references f_u: its line comes right after the sum
    assert text[1].startswith("f_u")
    assert any("gm_I0_MP2" in ln for ln in text)
    assert any("179 Hz" in ln for ln in text)   # no 0.000179 MHz
    # ...and a GHz root never prints as thousands of MHz
    from circuitinsight.analysis.marginledger import _mhz
    assert _mhz(4.34e9) == "4.34 GHz"
    assert _mhz(1.99e10) == "19.9 GHz"
    latex = margin_equation_latex(eq)
    assert latex[0].startswith(r"\mathrm{PM}")
    assert any(r"\arctan" in ln for ln in latex)
    assert any(r"\frac" in ln for ln in latex)
    # a pair line writes the second-order form directly in
    # frequencies -- no auxiliary u -- with zeta subscripted per term
    for ln in latex:
        if r"2\zeta_" in ln:
            assert "f_u f_" in ln and "u_{" not in ln
    # the sum's thetas are LINKS (data-term handles for the web
    # shell) and each definition line carries its anchor
    assert r"\htmlData{term=p1}" in latex[0]
    assert any(r"\htmlData{termdef=p1}" in ln for ln in latex[1:])
    # a named zeta, when one certifies, carries a sane value and its
    # measured error inside the gate
    for tt in eq["terms"]:
        ze = tt.get("zeta_estimate")
        if ze is not None:
            assert tt["pair"]
            assert ze["err"] <= 0.25
            assert 0 < ze["zeta_est"] < 2


def test_root_sensitivities_match_finite_difference(miller):
    """The root-sensitivity figures against independent +-1% re-solves with root
    tracking, for the dominant pole's and the RHP zero's top movers."""
    import dataclasses

    from circuitinsight.analysis.loopgain import loop_gain
    from circuitinsight.analysis.marginledger import (
        attach_root_sensitivities)
    from circuitinsight.engine.mna import build_mna, symbol_name

    an, system, T = miller
    led = margin_ledger(T, system=system)
    attach_root_sensitivities(led, system, an.primitives, "IPRB0", alias=an._alias)
    dom = next(e for e in led.entries
               if e.kind == "pole" and e.phase_deg is not None
               and abs(e.phase_deg + 90) < 2)
    z = next(e for e in led.entries if e.kind == "zero" and e.rhp)
    for entry, kind in ((dom, "poles"), (z, "zeros")):
        m = entry.root_sens[0]
        x0 = max(entry.roots, key=lambda r: r.imag)
        f0s = []
        for k in (-0.01, +0.01):
            prims = [dataclasses.replace(p, value=p.value * (1 + k))
                     if (p.value is not None
                         and symbol_name(p, an._alias) == m["symbol"])
                     else p for p in an.primitives]
            s2 = build_mna(prims, an.flat.ground, "IPRB0", an._alias)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                roots = getattr(loop_gain(s2, "IPRB0", []), kind)()
            f0s.append(abs(min(roots, key=lambda r: abs(r - x0))))
        fd = (f0s[1] - f0s[0]) / (2 * abs(x0)) * 100.0   # %/%
        assert m["df0"] == pytest.approx(fd, rel=0.05, abs=1e-3)


def test_physical_near_doublet_stays_visible():
    """merge_doublets is for ALGEBRAIC ghosts only: a physical
    pole/zero near-pair at percent-level spacing -- the mfm fragility
    case -- must stay two visible lines."""
    poles, zeros = [-1e3, -2e6], [-2.04e6]
    led = margin_ledger(_Rat(_stb_gain(1e3, poles, zeros),
                             poles=poles, zeros=zeros))
    kinds = sorted(e.kind for e in led.entries)
    assert kinds == ["gain", "pole", "pole", "zero"]
    assert not any(e.kind == "doublet" for e in led.entries)


def test_ghost_couples_never_reach_the_ledger():
    """The Tian lift by det(A) rides every closed-loop mode along as a
    coincident pole/zero couple. Those used to arrive here and be
    merged into net-zero 'doublet' entries; since minimal_pz (user
    decision: ghost roots generate noise, not insight -- designers
    investigated sensitivity maps for singularities that do not
    exist), the exact couples cancel at the ROOTING and the ledger
    sees only true singularities."""
    from circuitinsight.adapters.spectre import SpectreRun
    from circuitinsight.analysis.loopgain import loop_gain
    from circuitinsight.engine.mna import build_mna

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run = SpectreRun(FIX / "tb_ota2s_stb.cin.json", FIX / "psf_stb")
        an = run.analyzer(cap_model="matrix")
        system = build_mna(an.primitives, an.flat.ground, "IPRB0",
                           an._alias)
        T = loop_gain(system, "IPRB0", [])
    led = margin_ledger(T, system=system)
    # no entry is a net-zero ghost: every doublet that remains is a
    # GENUINE near pair with a real (if small) phase contribution
    for e in led.entries:
        if e.kind == "doublet":
            assert abs(e.phase_deg) > 1e-6
    # no surviving pole has a zero within ghost distance
    poles = [complex(p) for p in T.poles()]
    zeros = [complex(z) for z in T.zeros()]
    for p in poles:
        for z in zeros:
            assert abs(z - p) / max(abs(p), 1.0) > 1e-6
    # the attribution stays exact without the merge machinery firing
    assert led.phase_residual_deg < 1e-3


def test_fold_pairs_physical_near_doublets():
    """fold() displays a percent-level pole/zero couple as ONE line
    carrying the net share and the spacing; the exact entries stay
    separate, the sums are preserved, and a tighter pair_tol keeps
    the couple visible as two lines."""
    poles, zeros = [-1e3, -2e6], [-2.1e6]
    led = margin_ledger(_Rat(_stb_gain(1e3, poles, zeros),
                             poles=poles, zeros=zeros))
    # the exact entries stay pole + zero
    assert sorted(e.kind for e in led.entries) == \
        ["gain", "pole", "pole", "zero"]
    folded = led.fold(budget_deg=0.0)
    dbl = [e for e in folded if e.kind == "doublet"]
    assert len(dbl) == 1
    d = dbl[0]
    assert d.spacing == pytest.approx(0.05, rel=1e-6)
    assert d.f0_hz == pytest.approx(2e6) and \
        d.f0b_hz == pytest.approx(2.1e6)
    p = next(e for e in led.entries
             if e.kind == "pole" and e.f0_hz == pytest.approx(2e6))
    z = next(e for e in led.entries if e.kind == "zero")
    assert d.phase_deg == pytest.approx(p.phase_deg + z.phase_deg)
    # the fold preserves the whole sum
    assert sum(e.phase_deg for e in folded) == pytest.approx(
        sum(e.phase_deg for e in led.entries), abs=1e-9)
    assert "% apart" in d.describe()
    # a tighter tolerance keeps the couple as two visible lines
    tight = led.fold(budget_deg=0.0, pair_tol=0.01)
    assert not any(e.kind == "doublet" for e in tight)


def test_pairing_respects_structure():
    """A pole PAIR only couples with a zero PAIR; two nearby zeros
    never couple with each other."""
    f0, zeta = 2e6, 0.5
    ppair = [f0 * complex(-zeta, +np.sqrt(1 - zeta ** 2)),
             f0 * complex(-zeta, -np.sqrt(1 - zeta ** 2))]
    zpair = [1.05 * m for m in ppair]
    poles = [-1e3] + ppair
    led = margin_ledger(_Rat(_stb_gain(1e3, poles, zpair),
                             poles=poles, zeros=zpair))
    folded = led.fold(budget_deg=0.0)
    dbl = [e for e in folded if e.kind == "doublet"]
    assert len(dbl) == 1 and dbl[0].pair and dbl[0].count == 4
    assert dbl[0].spacing == pytest.approx(0.05, rel=1e-6)
    # two nearby same-kind singularities never pair
    zz = [-3e6, -3.1e6]
    led2 = margin_ledger(_Rat(_stb_gain(1e3, [-1e3, -50e6], zz),
                              poles=[-1e3, -50e6], zeros=zz))
    assert not any(e.kind == "doublet"
                   for e in led2.fold(budget_deg=0.0))


def test_compare_ledgers_matches_and_ranks():
    """The cross-corner view: the reference's rows carry every corner's
    matched share, drift ranks the rows, a singularity the corner moved
    beyond match_tol lands in that corner's 'other' bucket -- and every
    column still sums to its own corner's folded total."""
    from circuitinsight.analysis.marginledger import compare_ledgers

    ref = margin_ledger(_Rat(_stb_gain(1e3, [-1e3, -2e6]),
                             poles=[-1e3, -2e6]))
    near = margin_ledger(_Rat(_stb_gain(1e3, [-1e3, -2.2e6]),
                              poles=[-1e3, -2.2e6]))
    far = margin_ledger(_Rat(_stb_gain(1e3, [-1e3, -30e6]),
                             poles=[-1e3, -30e6]))
    cmp_ = compare_ledgers([ref, near, far], ["ref", "near", "far"])
    assert cmp_["labels"] == ["ref", "near", "far"]
    assert cmp_["pm"][0] == pytest.approx(ref.pm_deg)
    rows = cmp_["rows"]
    # the anchor row is fold-normalized: +-180 never reads as drift
    assert rows[0]["term"].kind == "gain"
    assert rows[0]["delta"] == pytest.approx(0.0, abs=1e-9)
    # the 2 MHz pole matched at 10% drift, unmatched at 15x
    p2 = next(r for r in rows
              if r["term"].kind == "pole"
              and r["term"].f0_hz == pytest.approx(2e6))
    assert p2["cells"][0] is not None and p2["cells"][1] is not None
    assert p2["cells"][2] is None
    assert p2["delta"] > 0
    # ...its share landed in far's other bucket instead
    assert cmp_["other"][2] != 0.0
    # every column sums exactly to its own corner's total
    for k, led in enumerate([ref, near, far]):
        total = sum(r["cells"][k] for r in rows[1:]
                    if r["cells"][k] is not None) + cmp_["other"][k]
        want = sum(e.phase_deg for e in led.entries[1:]
                   if e.phase_deg is not None)
        assert total == pytest.approx(want, abs=1e-9)
    # identical ledgers drift nowhere
    same = compare_ledgers([ref, ref], ["a", "b"])
    assert all(r["delta"] == pytest.approx(0.0, abs=1e-12)
               for r in same["rows"])
    assert same["other"][0] == pytest.approx(same["other"][1])


def test_miller_stb_loop_certifies():
    """The real solved loop gain: shares reconstruct the direct
    evaluation, PM agrees with the standard margin readout, and the
    top contributor is a genuine in-band singularity."""
    from circuitinsight.adapters.spectre import SpectreRun
    from circuitinsight.analysis.loopgain import loop_gain
    from circuitinsight.analysis.sensitivity import loop_margins
    from circuitinsight.engine.mna import build_mna

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run = SpectreRun(FIX / "tb_ota2s_stb.cin.json", FIX / "psf_stb")
        an = run.analyzer(cap_model="matrix")
        system = build_mna(an.primitives, an.flat.ground, "IPRB0",
                           an._alias)
        T = loop_gain(system, "IPRB0", [])
    led = margin_ledger(T)
    assert led.phase_residual_deg < 1e-3
    assert led.mag_residual_db < 1e-3
    freqs = np.geomspace(1e0, 1e12, 2001)
    pm, fu, gm, fgm = loop_margins(freqs, T.numeric(freqs))
    assert led.pm_deg == pytest.approx(pm, abs=0.5)
    assert led.gm_db == pytest.approx(gm, abs=0.5)
    top = next(e for e in led.entries[1:] if e.kind != "gain")
    # the miller loop's dominant pole (~180 Hz) spends its quarter turn
    assert 1e1 < top.f0_hz < 1e10
    assert top.phase_deg == pytest.approx(-90.0, abs=1.0)
    # the folded story stays certified
    lines = led.describe()
    assert any("PM" in ln for ln in lines[:1])


def test_the_probe_row_survives_the_level_two_search():
    """The level-two pair search once used `ib` as a loop counter, and
    `ib` is the probe BRANCH ROW bound once for the whole function. So
    every entry processed after a pair read the solve vectors at the
    wrong index -- silently wrong wherever the counter stayed in range,
    and an IndexError on nmc3 where it did not. The ledger's own
    first-order sensitivities are checked here against the analytic
    jet, which computes its indices independently, so a shadowed probe
    row cannot pass unnoticed even when it fails quietly."""
    from circuitinsight.analysis.marginledger import (_collect_stamps,
                                                      _tian_context)
    from circuitinsight.analysis.secondorder import loop_pole_jet
    from circuitinsight.engine.mna import build_mna
    from circuitinsight.session import SessionController

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = SessionController.open(str(FIX / "tb_ota2s_stb.cin.json"),
                                   str(FIX / "psf_stb"))
        s.set_matches(*s.suggest_matches())
        probe = s.stb_probe() or s.probes[0]
        led = s.margin_ledger(probe)["ledger"]
        an = s._analyzer_ready()
        system = build_mna(an.primitives, an.flat.ground, probe,
                           an._alias)
    fn, Q, ib, ip = _tian_context(system, probe)
    stamps, vals, _m = _collect_stamps(system, an.primitives, an._alias)

    entries = [e for e in led.entries
               if e.kind in ("pole", "zero") and e.f0_hz > 0
               and e.root_sens]
    assert len(entries) >= 4, "need entries after the first pair"
    checked = 0
    for e in entries:                      # every one, not just the first
        x = max(e.roots, key=lambda r: r.imag)
        g, _H = loop_pole_jet(fn, Q, ib, ip, stamps, vals,
                              2 * np.pi * complex(x))
        for m in e.root_sens[:4]:
            if abs(m["df0"]) < 0.05:
                continue
            assert abs(g[m["symbol"]].real - m["df0"]) < 1e-3, (
                e.f0_hz, m["symbol"])
            checked += 1
    assert checked >= 8


def test_nmc3_ledger_computes():
    """The three-stage nested-Miller bench crashed outright on the
    shadowed probe row; it must simply produce a ledger."""
    from circuitinsight.session import SessionController

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = SessionController.open(str(NMC3 / "tb_nmc3.cin.json"),
                                   str(NMC3 / "psf"))
        led = s.margin_ledger(s.stb_probe() or s.probes[0])["ledger"]
    assert led.pm_deg == pytest.approx(82.95, abs=0.5)
    assert any(e.root_sens for e in led.entries)


def test_the_equation_shows_a_partial_reading_as_structure():
    """A partial reading must not substitute -- its frequency can be
    fourfold wrong -- but the equation used to go silent about it, so
    a bench whose terms ALL read partially printed nothing but
    "(exact)" and looked as though nothing had been found. The
    structure travels; the value does not. Driven through the session
    with matching, which is the path a designer actually sees."""
    from circuitinsight.analysis.marginledger import (
        margin_equation, margin_equation_text)
    from circuitinsight.session import SessionController

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = SessionController.open(str(FIX / "tb_ota2s_stb.cin.json"),
                                   str(FIX / "psf_stb"))
        s.set_matches(*s.suggest_matches())
        led = s.margin_ledger(s.stb_probe() or s.probes[0])["ledger"]
    eq = margin_equation(led)
    partials = [t for t in eq["terms"] if t.get("partial")]
    assert partials, "the bench must exercise a partial reading"
    for t in partials:
        assert t["estimate"] is None        # structure, never a value
        pa = t["partial"]
        assert (pa["text"].startswith("?")
                or (pa["aggregate"]
                    and pa["aggregate"]["symbol"] in pa["text"]))
        if pa["aggregate"]:
            # an aggregate form is exact by its own identity, so the
            # discarded reading's prefactor must not ride along
            assert pa["prefactor"] is None
    text = " | ".join(margin_equation_text(eq))
    assert "reads " in text
    assert "(exact)" in text                # the value stays exact
    # a fully certified term still substitutes its own value
    assert [t for t in eq["terms"] if t["estimate"]]
