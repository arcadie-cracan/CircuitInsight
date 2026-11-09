"""Reading formulas out of the jet, on roots whose answers are known.

The Miller-compensated two-stage has two singularities no designer
would argue about:

    p1 = 1/(gm2 Ro1 Ro2 Cc)      with Ro = 1/(gds + gds) per stage
    z1 = gm2/(2 pi Cc)           the feedforward right-half-plane zero

Both must come back with no vocabulary offered and no value fitted --
partitioned out of the coupling matrix, each block's exponent read
twice and checked, and the whole thing certified by a prefactor near
one and by the Euler class sums it must reproduce.
"""
import warnings
from pathlib import Path

import numpy as np
import pytest

from circuitinsight.analysis.reconstruct import _CLASS, reconstruct
from circuitinsight.analysis.secondorder import loop_pole_jet

MILLER = Path(__file__).resolve().parent / "fixtures" / "spectre" / "miller"


@pytest.fixture(scope="module")
def bench():
    from circuitinsight.analysis.marginledger import (_collect_stamps,
                                                      _tian_context)
    from circuitinsight.engine.mna import build_mna
    from circuitinsight.session import SessionController

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = SessionController.open(str(MILLER / "tb_ota2s_stb.cin.json"),
                                   str(MILLER / "psf_stb"))
        s.set_matches(*s.suggest_matches())
        probe = s.stb_probe() or s.probes[0]
        an = s._analyzer_ready()
        system = build_mna(an.primitives, an.flat.ground, probe,
                           an._alias)
        led = s.margin_ledger(probe)["ledger"]
    stamps, vals, _meta = _collect_stamps(system, an.primitives,
                                          an._alias)
    return led, _tian_context(system, probe), stamps, vals


def _read(entry, bench):
    _led, (fn, Q, ib, ip), stamps, vals = bench
    x = max(entry.roots, key=lambda r: r.imag)
    g, H = loop_pole_jet(fn, Q, ib, ip, stamps, vals,
                         2 * np.pi * complex(x))
    return reconstruct(g, H, vals, stamps, entry.f0_hz)


def _entry(led, kind, **kw):
    cands = [e for e in led.entries
             if e.kind == kind and e.f0_hz > 0 and e.root_sens
             and all(getattr(e, k) == v for k, v in kw.items())]
    return min(cands, key=lambda e: e.f0_hz)


def test_dominant_pole_comes_back_as_the_textbook_formula(bench):
    led = bench[0]
    r = _read(_entry(led, "pole"), bench)
    assert r.verdict == "formula", r.reason
    # two conductance sums over gm and Cc: 1/(gm2 Ro1 Ro2 Cc)
    assert len(r.blocks) == 4
    sums = {frozenset(b.members) for b in r.blocks if len(b.members) > 1}
    assert frozenset(("gds_I0_MN0", "gds_I0_MP1")) in sums
    assert frozenset(("gds_I0_MN3", "gds_I0_MP2")) in sums
    lone = {b.members[0]: b.exponent for b in r.blocks
            if len(b.members) == 1}
    assert lone == {"I0_Cc": -1.0, "gm_I0_MP2": -1.0}
    for b in r.blocks:
        if len(b.members) > 1:
            assert b.exponent == 1.0
            assert b.plain, "a parallel pair: shares must track values"
    # the constant the structure does not fix, and the class sums it must
    assert abs(r.prefactor - 1.0) < 0.05
    assert abs(r.coverage["reactive"] + 1.0) < 0.05
    assert abs(r.coverage["conductive"] - 1.0) < 0.05


def test_rhp_zero_comes_back_as_gm_over_cc(bench):
    led = bench[0]
    r = _read(_entry(led, "zero", rhp=True), bench)
    assert r.verdict == "formula", r.reason
    assert {b.members[0] for b in r.blocks} == {"gm_I0_MP2", "I0_Cc"}
    signs = {b.members[0]: b.exponent for b in r.blocks}
    assert signs["gm_I0_MP2"] == 1.0 and signs["I0_Cc"] == -1.0
    assert abs(r.prefactor - 1.0) < 0.05


def test_blocks_are_always_dimensionally_homogeneous(bench):
    """A sum of a capacitance and a transconductance is not a formula,
    however well its exponents read. Checked over every root the bench
    has, not only the ones that certify."""
    led = bench[0]
    seen = 0
    for e in led.entries:
        if e.kind not in ("pole", "zero") or e.f0_hz <= 0 \
                or not e.root_sens:
            continue
        r = _read(e, bench)
        stamps = bench[2]
        for b in r.blocks:
            classes = set()
            for n in b.members:
                for k in {st[0] for st in stamps.get(n, ())}:
                    if k in _CLASS:
                        classes.add(_CLASS[k][0])
                        break
            assert len(classes) == 1, (b.members, classes)
            seen += 1
    assert seen >= 4


def test_an_unreadable_root_names_its_movers_instead(bench):
    """Where no partition stands the answer is the mover list and the
    reason, never a formula dressed up to look certified."""
    led = bench[0]
    verdicts = {}
    for e in led.entries:
        if e.kind not in ("pole", "zero") or e.f0_hz <= 0 \
                or not e.root_sens:
            continue
        r = _read(e, bench)
        verdicts[round(e.f0_hz / 1e6, 2)] = r.verdict
        if r.verdict != "formula":
            assert r.reason, e.f0_hz
            assert r.movers
            # the unread part is either an explicit question mark or a
            # NAMED aggregate; what it may never be is silent
            assert ("?" in r.text or not r.text
                    or (r.aggregate
                        and r.aggregate["symbol"] in r.text))
        else:
            assert "?" not in r.text
    # the bench must exercise both outcomes, or this proves nothing
    assert "formula" in verdicts.values()
    assert {"partial", "shared"} & set(verdicts.values())


def test_pair_reads_as_half_exponent_factors(bench):
    """A coalesced pair is omega0 = sqrt(Ga Gb/(Ca Cb)), so its factors
    wear exponent one half. The two-stage non-dominant pair must show
    both transconductances at +1/2 and a gate-node capacitance at
    -1/2 -- the grammar a level-two form is written in."""
    led = bench[0]
    e = next(x for x in led.entries if x.pair)
    r = _read(e, bench)
    halves = {b.members: b.exponent for b in r.blocks}
    assert all(abs(abs(a) - 0.5) < 1e-9 for a in halves.values()), halves
    lone = {b.members[0] for b in r.blocks
            if len(b.members) == 1 and b.exponent > 0}
    assert lone == {"gm_I0_MP1", "gm_I0_MP2"}
    caps = [b for b in r.blocks if b.exponent < 0]
    assert caps and set(caps[0].members) == {"cgs_I0_MP1", "cgb_I0_MP1"}
    assert "^0.5" in r.text


def test_damping_jet_matches_the_ledger_dzeta(bench):
    """zeta is the pair's other invariant, and its first derivatives
    must reproduce the ledger's own dzeta, computed by separate code
    from the same roots."""
    from circuitinsight.analysis.secondorder import loop_pair_jets

    led, (fn, Q, ib, ip), stamps, vals = bench
    e = next(x for x in led.entries if x.pair)
    root = max(e.roots, key=lambda r: r.imag)
    zeta = -root.real / abs(root)
    _w, (gz, _Hz) = loop_pair_jets(fn, Q, ib, ip, stamps, vals,
                                   2 * np.pi * complex(root))
    assert gz, "a complex pair must have a damping jet"
    checked = 0
    for m in e.root_sens[:8]:
        if m["symbol"] not in gz or abs(m["dzeta"]) < 1e-6:
            continue
        # the ledger reports dzeta per one percent; the jet is log-log
        mine = gz[m["symbol"]].real * zeta * 0.01
        assert abs(mine - m["dzeta"]) <= 1e-4 * max(abs(m["dzeta"]), 1e-3)
        checked += 1
    assert checked >= 4


def test_a_real_root_has_no_damping_jet(bench):
    from circuitinsight.analysis.secondorder import loop_pair_jets

    led, (fn, Q, ib, ip), stamps, vals = bench
    e = min((x for x in led.entries
             if x.kind == "pole" and x.f0_hz > 0 and not x.pair),
            key=lambda x: x.f0_hz)
    root = max(e.roots, key=lambda r: r.imag)
    (gw, _), (gz, Hz) = loop_pair_jets(fn, Q, ib, ip, stamps, vals,
                                       2 * np.pi * complex(root))
    assert gw and not gz and not Hz


def test_a_dimensionless_target_carries_no_two_pi(bench):
    """The 2 pi belongs to a frequency. A damping ratio must not be
    rendered with one."""
    from circuitinsight.analysis.secondorder import loop_pair_jets

    led, (fn, Q, ib, ip), stamps, vals = bench
    e = next(x for x in led.entries if x.pair)
    root = max(e.roots, key=lambda r: r.imag)
    _w, (gz, Hz) = loop_pair_jets(fn, Q, ib, ip, stamps, vals,
                                  2 * np.pi * complex(root))
    r = reconstruct(gz, Hz, vals, stamps, -root.real / abs(root),
                    two_pi=False)
    assert "2pi" not in r.text


def test_a_formula_reproduces_its_own_value(bench):
    """The prefactor is the one constant the structure does not fix.
    A form that reads cleanly but reproduces a fraction of its own
    root has a factor missing, and may not wear the certified verdict
    (field case: a differential bench pole read structurally at 0.61
    of its frequency). Checked over every root the bench has."""
    led = bench[0]
    seen = 0
    for e in led.entries:
        if e.kind not in ("pole", "zero") or e.f0_hz <= 0 \
                or not e.root_sens:
            continue
        r = _read(e, bench)
        if r.verdict != "formula":
            continue
        assert r.prefactor is not None
        assert abs(r.prefactor - 1.0) <= 0.25, (e.f0_hz, r.prefactor)
        assert abs(r.coverage["reactive"] + 1.0) <= 0.12
        assert abs(r.coverage["conductive"] - 1.0) <= 0.12
        seen += 1
    assert seen >= 2


def test_an_unbalanced_reading_reports_no_prefactor(bench):
    """Coverage IS the dimensional balance -- each block contributes
    its exponent to its class, so reactive -1 and conductive +1 says
    the product is a frequency. Off that balance the prefactor would
    absorb the missing units, so it is not reported at all."""
    led = bench[0]
    for e in led.entries:
        if e.kind not in ("pole", "zero") or e.f0_hz <= 0 \
                or not e.root_sens:
            continue
        r = _read(e, bench)
        if not r.blocks:
            continue
        balanced = (abs(r.coverage["reactive"] + 1.0) <= 0.12
                    and abs(r.coverage["conductive"] - 1.0) <= 0.12)
        assert (r.prefactor is not None) == balanced, (e.f0_hz,
                                                       r.coverage)


def _synthetic(gvals, rho_pairs, kinds, vals):
    """g/H dicts with prescribed couplings, for the rules that are
    about grammar rather than about any particular circuit."""
    g = {n: complex(v) for n, v in gvals.items()}
    H = {}
    for a in gvals:
        for b in gvals:
            if a == b:
                # a lone factor is a monomial in that element, and
                # ln(x**a) is linear in ln x, so the diagonal vanishes
                H[a, b] = 0j
            else:
                r = rho_pairs.get((a, b), rho_pairs.get((b, a), 0.0))
                H[a, b] = complex(r * gvals[a] * gvals[b])
    stamps = {n: [(k, None, None, "X")] for n, k in kinds.items()}
    return g, H, stamps, vals


def test_a_conductance_and_a_resistance_never_share_a_sum():
    """Both scale the admittance, so the coverage rule counts them in
    one class -- but G + R is not a quantity. Series resistances add,
    parallel conductances add; homogeneity keys on the sign too."""
    g, H, stamps, vals = _synthetic(
        {"G1": 0.5, "R1": 0.5}, {("G1", "R1"): -1.0},
        {"G1": "g", "R1": "r"}, {"G1": 1e-4, "R1": 1e4})
    r = reconstruct(g, H, vals, stamps, 1e6)
    for b in r.blocks:
        assert not {"G1", "R1"} <= set(b.members), b.members


def test_a_damping_ratio_is_checked_against_zero_not_one():
    """Scaling every reactance by k scales a FREQUENCY by 1/k, hence
    the -1 class sum; a damping ratio is dimensionless and the same
    scaling leaves it alone, so its class sums are both ZERO. Judging
    zeta against -1/+1 would condemn every correct reading of it."""
    import math

    vals = {"C1": 2e-12, "C2": 1e-12}
    g, H, stamps, _v = _synthetic(
        {"C1": 0.5, "C2": -0.5}, {}, {"C1": "c", "C2": "c"}, vals)
    target = math.sqrt(vals["C1"] / vals["C2"])
    z = reconstruct(g, H, vals, stamps, target, two_pi=False)
    assert z.verdict == "formula", z.reason
    assert z.prefactor == pytest.approx(1.0, abs=0.02)
    assert "2pi" not in z.text
    # the very same reading, judged as a frequency, must NOT certify
    f = reconstruct(g, H, vals, stamps, target, two_pi=True)
    assert f.verdict != "formula"


def test_an_aggregate_reproduces_the_root_exactly(bench):
    """When one side reads and the whole does not, the rest is named
    as a single quantity -- G_eff or C_eff -- and that is not a fit:
    with the other side exact, the identity FIXES it. So the form must
    reproduce the root to machine precision, which is what entitles it
    to be shown without a question mark."""
    import math

    led = bench[0]
    seen = 0
    for e in led.entries:
        if e.kind not in ("pole", "zero") or e.f0_hz <= 0 \
                or not e.root_sens:
            continue
        r = _read(e, bench)
        if not r.aggregate:
            continue
        ag, vals = r.aggregate, bench[3]
        prod = ag["value"] ** ag["exponent"]
        for b in ag["blocks"]:
            prod *= sum(abs(vals[n]) for n in b.members) ** b.exponent
        assert prod / (2 * math.pi) == pytest.approx(e.f0_hz, rel=1e-9)
        assert ag["unit"] in ("S", "F") and ag["value"] > 0
        assert "?" not in r.text and ag["symbol"] in r.text
        seen += 1
    assert seen >= 1, "the bench must exercise an aggregate"


def test_a_one_sided_partial_is_not_dressed_as_a_frequency():
    """`? sqrt(gm_PM9 + gm_PM11) / 2pi` is not a frequency for any
    reading of "?".

    A partial names some of the structure and that is worth showing --
    but only while it still reads as the quantity it claims to be. A
    frequency needs a factor on BOTH sides. Where one side is entirely
    unread the "?" carries the whole dimensional character, and
    printing it over 2pi asserts a shape the visible content does not
    have. Field report, a 248 MHz pole.

    An aggregate is exempt: G_eff and C_eff are NAMED stand-ins fixed
    exactly by the identity, not unread remainders."""

    from circuitinsight.analysis.reconstruct import reconstruct

    # one conductive block at exponent +0.5 and NOTHING reactive --
    # the shape the field report showed. A sum of exponent a has
    # g_i = a*sigma and rho = -1/a, so for a = 0.5 with two equal
    # members: g = 0.25 each, H_ij = rho*g_i*g_j, H_ii = g_i(1-sigma).
    g = {"gm_A": 0.25 + 0j, "gm_B": 0.25 + 0j}
    H = {("gm_A", "gm_A"): 0.125 + 0j, ("gm_B", "gm_B"): 0.125 + 0j,
         ("gm_A", "gm_B"): -0.125 + 0j, ("gm_B", "gm_A"): -0.125 + 0j}
    stamps = {n: [("g", None, None, "M")] for n in g}
    vals = {"gm_A": 1e-3, "gm_B": 1e-3}
    rec = reconstruct(g, H, vals, stamps, 2.48e8, two_pi=True)
    assert rec.verdict != "partial", (
        "a one-sided reading must not render as an expression")
    assert rec.verdict == "shared"
    assert "no expression can carry the units" in rec.reason
    assert not rec.text, "no expression may be offered"
    # the block DID form -- without this the test would pass by taking
    # an earlier shared path and would not exercise the rule at all
    assert rec.coverage["conductive"] == pytest.approx(0.5, abs=1e-6)
    assert rec.coverage["reactive"] == 0


def _monomial_jet(sums, lone, kinds, vals):
    """g/H of f0 = prod_k S_k**a_k * prod lone x**e, each S_k a SUM OF
    MONOMIALS, built from the signature the module docstring states:
    g_i = a e_i sigma_t, H_ij = a e_i e_j sigma_t (1 - sigma_t) inside
    an addend, -a e_i e_j sigma_t sigma_s across addends, zero across
    factors; a lone factor has g = e and no second derivative.

    `sums` is [(a, [(share, {symbol: exponent}), ...])]; `lone` is
    {symbol: exponent}."""
    g, H = {}, {}
    where = {}                       # symbol -> (factor, addend, e)
    for k, (a, addends) in enumerate(sums):
        for t, (share, members) in enumerate(addends):
            for n, e in members.items():
                g[n] = complex(a * e * share)
                where[n] = (k, t, e, share, a)
    for n, e in lone.items():
        g[n] = complex(e)
    for i in g:
        for j in g:
            if i in lone or j in lone:
                H[i, j] = 0j
                continue
            ki, ti, ei, si, a = where[i]
            kj, tj, ej, sj, _a = where[j]
            if ki != kj:
                H[i, j] = 0j
            elif ti == tj:
                H[i, j] = complex(a * ei * ej * si * (1 - si))
            else:
                H[i, j] = complex(-a * ei * ej * si * sj)
    stamps = {n: [(k, None, None, "X")] for n, k in kinds.items()}
    return g, H, stamps, vals


def test_a_sum_of_monomials_reads_as_the_ldo_pole(monkeypatch):
    """p1 = 1/(2 pi Rout (CL + gm R1 Cc)): the dominant pole of a
    Miller-compensated LDO as its paper writes it, a sum of two
    monomials under a lone factor. CL carries 0.3 of the sum and the
    product 0.7, so the product partners couple to one another at
    (1 - 0.7)/(-1 * 0.7) = -0.43, which is also what each one's own
    diagonal shows, and to CL at -1/a = +1. The reader must spell the
    sum with its addends, at exponent -1, with a prefactor of one and
    the class sums of a frequency."""
    import math

    vals = {"Rout": 1e5, "CL": 3e-12, "gm": 1e-3, "R1": 1e4,
            "Cc": 7e-13}
    g, H, stamps, _v = _monomial_jet(
        [(-1.0, [(0.3, {"CL": 1.0}),
                 (0.7, {"gm": 1.0, "R1": 1.0, "Cc": 1.0})])],
        {"Rout": -1.0},
        {"Rout": "r", "CL": "c", "gm": "vccs", "R1": "r", "Cc": "c"},
        vals)
    f0 = 1.0 / (2 * math.pi * vals["Rout"]
                * (vals["CL"] + vals["gm"] * vals["R1"] * vals["Cc"]))
    r = reconstruct(g, H, vals, stamps, f0)
    assert r.verdict == "formula", r.reason
    # addends in order of share; the anchored reader prints each
    # addend's capacitor first, the time constant's own name
    assert r.text == "1/(2pi*(Cc*gm*R1+CL)*Rout)", r.text
    assert r.tex() == (r"\frac{1}{2\pi\,\left(Cc\,gm\,R1 + CL\right)"
                       r"\,Rout}"), r.tex()
    assert r.prefactor == pytest.approx(1.0, abs=1e-6)
    assert r.coverage["reactive"] == pytest.approx(-1.0, abs=1e-9)
    assert r.coverage["conductive"] == pytest.approx(1.0, abs=1e-9)
    assert r.value == pytest.approx(f0, rel=1e-6)
    blk = next(b for b in r.blocks if b.terms)
    assert blk.exponent == -1.0 and blk.kind == "reactive"
    assert set(blk.members) == {"CL", "gm", "R1", "Cc"}
    shares = {t.text(): t for t in blk.terms}
    assert shares["CL"].share == pytest.approx(0.3, abs=1e-6)
    assert shares["Cc*gm*R1"].share == pytest.approx(0.7, abs=1e-6)
    assert shares["Cc*gm*R1"].share_from_rho == pytest.approx(0.7,
                                                              abs=1e-6)
    assert shares["Cc*gm*R1"].share_value == pytest.approx(0.7,
                                                           abs=1e-6)
    assert blk.plain and blk.spread < 1e-9
    assert not r.residue and not r.reason
    # the same jet with product partners DENIED, and the anchors with
    # them, is the grammar as it stood, and that grammar cannot spell
    # this pole
    from circuitinsight.analysis import reconstruct as mod
    monkeypatch.setattr(mod, "_product_terms", lambda *a, **k: [])
    monkeypatch.setattr(mod, "_anchored_sum", lambda *a, **k: None)
    old = reconstruct(g, H, vals, stamps, f0)
    assert old.verdict != "formula"
    assert not any(b.terms for b in old.blocks)


def test_two_cascode_divides_add_over_the_load():
    """The folded cascode output conductance as a hand analysis
    writes it: gds_a G_a/Gm_a + gds_b gds_c/Gm_b, two ratio addends
    at shares 0.6 and 0.4, over 2 pi CL. Exponent +1 on the sum, so
    partners couple at (1 - sigma)/sigma -- +0.67 and +1.5 -- and the
    addends at -1."""
    import math

    vals = {"CL": 5e-12, "gds_a": 2e-6, "G_a": 5e-5, "Gm_a": 2e-4,
            "gds_b": 4e-6, "gds_c": 1e-5, "Gm_b": 1.2e-4}
    A = vals["gds_a"] * vals["G_a"] / vals["Gm_a"]
    B = vals["gds_b"] * vals["gds_c"] / vals["Gm_b"]
    assert A / (A + B) == pytest.approx(0.6, abs=1e-9)
    g, H, stamps, _v = _monomial_jet(
        [(1.0, [(0.6, {"gds_a": 1.0, "G_a": 1.0, "Gm_a": -1.0}),
                (0.4, {"gds_b": 1.0, "gds_c": 1.0, "Gm_b": -1.0})])],
        {"CL": -1.0},
        {n: ("c" if n == "CL" else "g") for n in vals}, vals)
    f0 = (A + B) / (2 * math.pi * vals["CL"])
    r = reconstruct(g, H, vals, stamps, f0)
    assert r.verdict == "formula", r.reason
    assert r.text == "(gds_a*G_a/Gm_a+gds_b*gds_c/Gm_b)/(2pi*CL)", r.text
    assert r.prefactor == pytest.approx(1.0, abs=1e-6)
    assert r.aggregate is None
    blk = next(b for b in r.blocks if b.terms)
    assert blk.exponent == 1.0 and blk.kind == "conductive"
    assert [t.share for t in blk.terms] == pytest.approx([0.6, 0.4])
    assert r"\frac{gds\_a\,G\_a}{Gm\_a}" in r.tex()


def test_a_partner_with_no_integer_exponent_is_residue_not_member():
    """A symbol that couples like a product partner but whose exponent
    reads 0.2 is not a member of any monomial. It is named beside the
    expression with its reading, whatever the verdict -- never folded
    in, never dropped in silence."""
    import math

    vals = {"CL": 5e-12, "gds_a": 2e-6, "G_a": 5e-5, "Gm_a": 2e-4,
            "gds_b": 4e-6, "gds_c": 1e-5, "Gm_b": 1.2e-4, "gm_x": 3e-5}
    g, H, stamps, _v = _monomial_jet(
        [(1.0, [(0.6, {"gds_a": 1.0, "G_a": 1.0, "Gm_a": -1.0,
                       "gm_x": 0.2}),
                (0.4, {"gds_b": 1.0, "gds_c": 1.0, "Gm_b": -1.0})])],
        {"CL": -1.0},
        {n: ("c" if n == "CL" else "g") for n in vals}, vals)
    A = vals["gds_a"] * vals["G_a"] / vals["Gm_a"]
    B = vals["gds_b"] * vals["gds_c"] / vals["Gm_b"]
    r = reconstruct(g, H, vals, stamps, (A + B) / (2 * math.pi * 5e-12))
    assert r.blocks and any(b.terms for b in r.blocks)
    assert [n for n, _g, _w in r.residue] == ["gm_x"]
    assert "gm_x" in r.reason and "+0.20" in r.reason
    assert "gm_x" not in r.text
    # the addend it fell out of still reads, and the named structure
    # still carries the class sums, so the verdict stands with the
    # residue said beside it
    assert "gds_a*G_a/Gm_a" in r.text
    assert r.coverage["conductive"] == pytest.approx(1.0, abs=1e-6)
    assert r.verdict == "formula"
    assert r.residue[0][1] == pytest.approx(0.12, abs=1e-9)


def test_addends_of_different_dimension_do_not_add():
    """gds/Gm is a number and gds*gds/Gm a conductance; no sum holds
    both however their couplings read. The group must fail as a sum
    of monomials rather than print one."""
    import math

    vals = {"CL": 5e-12, "gds_a": 2e-6, "Gm_a": 2e-4,
            "gds_b": 4e-6, "gds_c": 1e-5, "Gm_b": 1.2e-4}
    g, H, stamps, _v = _monomial_jet(
        [(1.0, [(0.6, {"gds_a": 1.0, "Gm_a": -1.0}),
                (0.4, {"gds_b": 1.0, "gds_c": 1.0, "Gm_b": -1.0})])],
        {"CL": -1.0},
        {n: ("c" if n == "CL" else "g") for n in vals}, vals)
    r = reconstruct(g, H, vals, stamps, 1e5 / (2 * math.pi))
    assert not any(b.terms for b in r.blocks)
    assert r.verdict != "formula"


def _unlump(g, H, vals, stamps, name, parts):
    """The jet with symbol `name` split into the members of `parts`
    ({member: value}) it is the sum of -- the chain rule through a sum
    run backwards, so that the lump pre-pass has something to fold:
    with sigma_i = v_i / sum v, g_i = sigma_i g_L, H_ij = sigma_i
    sigma_j (H_LL - g_L) off the diagonal, H_ii = sigma_i**2 (H_LL -
    g_L) + sigma_i g_L, and every outside coupling H_io = sigma_i
    H_Lo."""
    g, H, vals, stamps = dict(g), dict(H), dict(vals), dict(stamps)
    tot = sum(parts.values())
    gL, HLL = g.pop(name), H.pop((name, name))
    kind = stamps.pop(name)
    vals.pop(name)
    others = [n for n in g]
    row = {o: H.pop((name, o)) for o in others}
    for o in others:
        H.pop((o, name), None)
    for m, v in parts.items():
        s = v / tot
        g[m], vals[m], stamps[m] = s * gL, v, kind
        for o in others:
            H[m, o] = H[o, m] = s * row[o]
    for a, va in parts.items():
        for b, vb in parts.items():
            sa, sb = va / tot, vb / tot
            H[a, b] = sa * sb * (HLL - gL) + (sa * gL if a == b else 0)
    return g, H, vals, stamps


def test_the_lump_pre_pass_folds_what_the_root_sees_as_one_sum():
    """Two cascode divides over CL again, but handed over in device
    symbols: G_a is g1 + g2 and Gm_a is gm_a + gmbs_a. Unfolded, the
    partners of each addend are hidden behind couplings of -1/a - 1/g
    and the reader can only name a G_eff. With `lump` the root's own
    jet says which symbols it sees only as a sum -- one coupling, one
    diagonal shift, one row each -- and the reading comes back as the
    hand form with the lumps spelled, nobody having named them."""
    import math

    vals = {"CL": 5e-12, "gds_a": 2e-6, "G_a": 5e-5, "Gm_a": 2e-4,
            "gds_b": 4e-6, "gds_c": 1e-5, "Gm_b": 1.2e-4}
    g, H, stamps, _v = _monomial_jet(
        [(1.0, [(0.6, {"gds_a": 1.0, "G_a": 1.0, "Gm_a": -1.0}),
                (0.4, {"gds_b": 1.0, "gds_c": 1.0, "Gm_b": -1.0})])],
        {"CL": -1.0},
        {n: ("c" if n == "CL" else "g") for n in vals}, vals)
    g, H, vals, stamps = _unlump(g, H, vals, stamps, "G_a",
                                 {"g1": 3e-5, "g2": 2e-5})
    g, H, vals, stamps = _unlump(g, H, vals, stamps, "Gm_a",
                                 {"gm_a": 1.5e-4, "gmbs_a": 5e-5})
    A = vals["gds_a"] * (vals["g1"] + vals["g2"]) \
        / (vals["gm_a"] + vals["gmbs_a"])
    B = vals["gds_b"] * vals["gds_c"] / vals["Gm_b"]
    f0 = (A + B) / (2 * math.pi * vals["CL"])
    flat = reconstruct(g, H, vals, stamps, f0)
    assert flat.verdict != "formula"
    assert "(g1+g2)" not in flat.text and "gmbs_a)" not in flat.text
    r = reconstruct(g, H, vals, stamps, f0, lump=True)
    assert r.verdict == "formula", r.reason
    assert r.text == ("(gds_a*(g1+g2)/(gm_a+gmbs_a)+gds_b*gds_c/Gm_b)"
                      "/(2pi*CL)"), r.text
    assert r"\frac{gds\_a\,\left(g1 + g2\right)}{gm\_a + gmbs\_a}" \
        in r.tex(), r.tex()
    assert r.prefactor == pytest.approx(1.0, abs=1e-6)
    assert r.lumps == {"g1+g2": ("g1", "g2"),
                       "gm_a+gmbs_a": ("gm_a", "gmbs_a")}
    blk = next(b for b in r.blocks if b.terms)
    assert [t.share for t in blk.terms] == pytest.approx([0.6, 0.4])


def test_a_folded_plain_sum_is_unfolded_in_the_result():
    """(G1 + G2)/(2 pi Cc) reads as a plain block today, and the lump
    pre-pass sees G1 + G2 as a sum too. Folding it must leave the
    result exactly as the plain grammar spelled it: members,
    exponent, shares and no lump on record."""
    import math

    vals = {"G1": 3e-5, "G2": 1e-5, "Cc": 1e-12}
    g = {"G1": 0.75 + 0j, "G2": 0.25 + 0j, "Cc": -1.0 + 0j}
    # a sum at exponent +1: rho = -1 between members, H_ii = g(1 - s)
    H = {("G1", "G1"): complex(0.75 * 0.25),
         ("G2", "G2"): complex(0.25 * 0.75),
         ("G1", "G2"): complex(-0.75 * 0.25),
         ("G2", "G1"): complex(-0.75 * 0.25)}
    for n in g:
        H[n, "Cc"] = H["Cc", n] = 0j
    stamps = {"G1": [("g", None, None, "X")], "G2": [("g", None, None, "X")],
              "Cc": [("c", None, None, "X")]}
    f0 = (vals["G1"] + vals["G2"]) / (2 * math.pi * vals["Cc"])
    plain = reconstruct(g, H, vals, stamps, f0)
    lumped = reconstruct(g, H, vals, stamps, f0, lump=True)
    assert plain.verdict == "formula", plain.reason
    assert lumped.verdict == "formula", lumped.reason
    assert plain.text == lumped.text == "(G1+G2)/(2pi*Cc)"
    # the same blocks; their order is not part of the contract -- the
    # lump and Cc both move this root by exactly one, and tied singles
    # are listed by name
    def by_members(rec):
        return {b.members: b for b in rec.blocks}

    assert by_members(plain).keys() == by_members(lumped).keys()
    for mem, b in by_members(plain).items():
        assert b.shares == pytest.approx(by_members(lumped)[mem].shares)
        assert b.exponent == by_members(lumped)[mem].exponent
    assert not lumped.lumps
    assert lumped.prefactor == pytest.approx(plain.prefactor)


def _divide_jet(a, shares, exps, lone, kinds, vals):
    """g/H of f0 = prod_k x_k**e_k * (sum_k x_k)**a * prod lone x**e,
    from the divide signature the module docstring states: g_k = e_k
    + a sigma_k, H_kl = -a sigma_k sigma_l, H_kk = a sigma_k
    (1 - sigma_k); a lone factor has g = e and no second derivative.
    `shares` is {symbol: sigma_k}, `exps` {symbol: e_k} (missing means
    the member stands only in the sum)."""
    g, H = {}, {}
    for n, s in shares.items():
        g[n] = complex(exps.get(n, 0.0) + a * s)
    for n, e in lone.items():
        g[n] = complex(e)
    for i in g:
        for j in g:
            if i in lone or j in lone:
                H[i, j] = 0j
            elif i == j:
                H[i, j] = complex(a * shares[i] * (1 - shares[i]))
            else:
                H[i, j] = complex(-a * shares[i] * shares[j])
    stamps = {n: [(k, None, None, "X")] for n, k in kinds.items()}
    return g, H, stamps, vals


def test_a_divide_reads_the_member_alone_and_in_its_sum(monkeypatch):
    """The low pole of a one-core capacitance multiplier, 1/(2 pi
    Rout2 M Cplaced) with M = 1 + GmK/Gm, is Gm/(2 pi Rout2 Cplaced
    (Gm + GmK)): Gm stands in the numerator and in the sum. Its
    couplings say a = +1 and its shares say a = 0, so the plain
    partition refuses the pair; the divide production reads it, the
    sum as one block and Gm as a lone factor, at a prefactor of one.
    With that production denied the root is not a formula."""
    import math

    from circuitinsight.analysis import reconstruct as mod

    vals = {"Gm": 1e-3, "GmK": 1e-3, "Rout2": 1e9, "Cplaced": 5e-12}
    g, H, stamps, _v = _divide_jet(
        -1.0, {"Gm": 0.5, "GmK": 0.5}, {"Gm": 1.0},
        {"Rout2": -1.0, "Cplaced": -1.0},
        {"Gm": "vccs", "GmK": "vccs", "Rout2": "r", "Cplaced": "c"}, vals)
    f0 = vals["Gm"] / (2 * math.pi * vals["Rout2"] * vals["Cplaced"]
                       * (vals["Gm"] + vals["GmK"]))
    r = reconstruct(g, H, vals, stamps, f0)
    assert r.verdict == "formula", r.reason
    assert r.text == "Gm/(2pi*Cplaced*Rout2*(Gm+GmK))"
    assert r.prefactor == pytest.approx(1.0, abs=0.02)
    assert r.coverage["reactive"] == pytest.approx(-1.0, abs=0.02)
    assert r.coverage["conductive"] == pytest.approx(1.0, abs=0.02)
    total = {b.members: b for b in r.blocks}
    assert total[("Gm", "GmK")].divide == ("Gm",)
    assert total[("Gm", "GmK")].exponent == -1.0
    assert total[("Gm",)].exponent == 1.0
    assert r.tex() == (r"\frac{Gm}{2\pi\,Cplaced\,Rout2\,"
                       r"\left(Gm + GmK\right)}")

    monkeypatch.setattr(mod, "_divide_groups", lambda *a, **k: [])
    assert reconstruct(g, H, vals, stamps, f0).verdict != "formula"


def test_a_series_combination_is_a_divide_with_no_pure_member():
    """gm1 gm2/(gm1 + gm2) over a capacitance: both members stand
    alone and in the sum, none only in the sum. Read at shares 0.6
    and 0.4 so the two readings of each share are distinct roots."""
    import math

    vals = {"gm1": 6e-4, "gm2": 4e-4, "CL": 1e-12}
    g, H, stamps, _v = _divide_jet(
        -1.0, {"gm1": 0.6, "gm2": 0.4}, {"gm1": 1.0, "gm2": 1.0},
        {"CL": -1.0}, {"gm1": "vccs", "gm2": "vccs", "CL": "c"}, vals)
    f0 = (vals["gm1"] * vals["gm2"] / (vals["gm1"] + vals["gm2"])
          / (2 * math.pi * vals["CL"]))
    r = reconstruct(g, H, vals, stamps, f0)
    assert r.verdict == "formula", r.reason
    assert r.text == "gm2*gm1/(2pi*CL*(gm2+gm1))"
    assert r.prefactor == pytest.approx(1.0, abs=0.02)


def test_a_cascode_divide_reads_without_a_lump():
    """gds G/(gm + G + gds) over CL, the cascode divide with its full
    denominator, at shares a reader can see: every member couples
    above the edge. The two overlapping members come out alone and
    the three-member sum intact, which is the depth the sum-of-
    monomials grammar could only point at."""
    import math

    vals = {"gm": 3.5e-4, "G": 4e-4, "gds": 2.5e-4, "CL": 2e-12}
    g, H, stamps, _v = _divide_jet(
        -1.0, {"gm": 0.35, "G": 0.4, "gds": 0.25},
        {"G": 1.0, "gds": 1.0}, {"CL": -1.0},
        {"gm": "vccs", "G": "g", "gds": "g", "CL": "c"}, vals)
    f0 = (vals["gds"] * vals["G"] / (vals["gm"] + vals["G"] + vals["gds"])
          / (2 * math.pi * vals["CL"]))
    r = reconstruct(g, H, vals, stamps, f0)
    assert r.verdict == "formula", r.reason
    assert r.text == "gds*G/(2pi*CL*(gds+G+gm))"
    assert r.prefactor == pytest.approx(1.0, abs=0.02)
    blk = {b.members: b for b in r.blocks}[("gds", "G", "gm")]
    assert set(blk.divide) == {"gds", "G"}
    assert blk.shares["gm"] == pytest.approx(0.35, abs=0.02)


def test_a_bare_constant_is_part_of_the_formula():
    """zeta = 1/(2Q): the named structure reproduces exactly half the
    value, and a half is a formula's constant, not a missing factor.
    The 2 joins the expression, the raw ratio stays as the prefactor,
    and a ratio near nothing nice is still a partial."""
    import math

    vals = {"C1": 2e-12, "C2": 1e-12}
    g, H, stamps, _v = _synthetic(
        {"C1": 0.5, "C2": -0.5}, {}, {"C1": "c", "C2": "c"}, vals)
    half = 0.5 * math.sqrt(vals["C1"] / vals["C2"])
    z = reconstruct(g, H, vals, stamps, half, two_pi=False)
    assert z.verdict == "formula", z.reason
    assert z.constant == 0.5
    assert z.text == "C1^0.5/(2*C2^0.5)"
    assert z.prefactor == pytest.approx(0.5, abs=0.01)
    assert z.value == pytest.approx(half)
    assert z.tex() == r"\frac{\sqrt{C1}}{2\,\sqrt{C2}}"
    twice = reconstruct(g, H, vals, stamps, 4 * half, two_pi=False)
    assert twice.verdict == "formula" and twice.constant == 2.0
    assert twice.text == "2*C1^0.5/(C2^0.5)"
    odd = reconstruct(g, H, vals, stamps, 1.4 * half, two_pi=False)
    assert odd.verdict != "formula" and odd.constant == 1.0
    assert "a factor is missing" in odd.reason


def test_a_constant_is_not_read_onto_a_partial():
    """The same LDO pole with one coupling inside the product addend
    knocked off, so the addend reports a spread, and a target at half
    the value. The prefactor sits near one half, but the prefactor is
    not the only thing in the way: no 2 may dress a structure that has
    not read, on the raw folded-cascode jet that printed a sqrt and a
    half beside a spread complaint."""
    import math

    vals = {"Rout": 1e5, "CL": 3e-12, "gm": 1e-3, "R1": 1e4,
            "Cc": 7e-13}
    g, H, stamps, _v = _monomial_jet(
        [(-1.0, [(0.3, {"CL": 1.0}),
                 (0.7, {"gm": 1.0, "R1": 1.0, "Cc": 1.0})])],
        {"Rout": -1.0},
        {"Rout": "r", "CL": "c", "gm": "vccs", "R1": "r", "Cc": "c"},
        vals)
    H["gm", "R1"] = H["R1", "gm"] = 1.6 * H["gm", "R1"]
    f0 = 1.0 / (2 * math.pi * vals["Rout"]
                * (vals["CL"] + vals["gm"] * vals["R1"] * vals["Cc"]))
    r = reconstruct(g, H, vals, stamps, 0.5 * f0)
    assert r.verdict == "partial"
    assert "spread" in r.reason
    assert r.constant == 1.0
    assert "2*" not in r.text


def _nested_jet(a, addends, lone, kinds, vals):
    """g/H of f0 = (sum_t A_t)**a * prod lone x**e with each addend
    A_t = prod x_i**e_i * (sum_k x_k)**b_t, from the two-level
    signature the module docstring states: with eps_i = e_i + b tau_i,
    g_i = a sigma_t eps_i; inside an addend H_ij = a [sigma_t (1 -
    sigma_t) eps_i eps_j + sigma_t b tau_i (delta_ij - tau_j)]; across
    addends -a sigma_t sigma_s eps_i eps_j; zero to a lone factor.

    `addends` is [(share, {symbol: e}, b, {symbol: tau})]; a member of
    the inner sum that stands outside the product has e = 0."""
    g, H = {}, {}
    where = {}
    for t, (share, es, b, taus) in enumerate(addends):
        names = set(es) | set(taus)
        for n in names:
            eps = es.get(n, 0.0) + b * taus.get(n, 0.0)
            g[n] = complex(a * share * eps)
            where[n] = (t, eps, share, b, taus.get(n, 0.0))
    for n, e in lone.items():
        g[n] = complex(e)
    for i in g:
        for j in g:
            if i in lone or j in lone:
                H[i, j] = 0j
                continue
            ti, ei, si, b, taui = where[i]
            tj, ej, sj, _b, tauj = where[j]
            if ti != tj:
                H[i, j] = complex(-a * si * sj * ei * ej)
            else:
                d = 1.0 if i == j else 0.0
                H[i, j] = complex(a * (si * (1 - si) * ei * ej
                                       + si * b * taui * (d - tauj)))
    stamps = {n: [(k, None, None, "X")] for n, k in kinds.items()}
    return g, H, stamps, vals


def test_two_cascode_divides_with_full_denominators(monkeypatch):
    """The folded cascode's output conductance as it really is: two
    cascode divides, gds G/(gm + G + gds) each, adding under CL. The
    members of each inner sum read at effective exponents off their
    integers -- G at 1 - tau_G, gm at -tau_gm -- which the monomial
    reading can only report as a spread; the nested divide reads the
    inner shares out of what the monomial part leaves, names the outer
    exponents at their integers, and the whole certifies at a
    prefactor of one. With the nested reading denied it is a partial."""
    import math

    from circuitinsight.analysis import reconstruct as mod

    kinds = {"gds_a": "g", "G_a": "g", "gm_a": "vccs",
             "gds_b": "g", "gds_c": "g", "gm_b": "vccs", "CL": "c"}
    vals = {"gds_a": 2e-6, "G_a": 10e-6, "gm_a": 60e-6,
            "gds_b": 3e-6, "gds_c": 4e-6, "gm_b": 50e-6, "CL": 5e-12}
    # inner shares from the values, as a real divide shows them
    da = vals["gm_a"] + vals["G_a"] + vals["gds_a"]
    db = vals["gm_b"] + vals["gds_b"] + vals["gds_c"]
    A = vals["gds_a"] * vals["G_a"] / da
    B = vals["gds_b"] * vals["gds_c"] / db
    g, H, stamps, _v = _nested_jet(
        1.0,                      # the conductances ADD, over CL
        [(A / (A + B), {"gds_a": 1.0, "G_a": 1.0}, -1.0,
          {"gm_a": vals["gm_a"] / da, "G_a": vals["G_a"] / da,
           "gds_a": vals["gds_a"] / da}),
         (B / (A + B), {"gds_b": 1.0, "gds_c": 1.0}, -1.0,
          {"gm_b": vals["gm_b"] / db, "gds_b": vals["gds_b"] / db,
           "gds_c": vals["gds_c"] / db})],
        {"CL": -1.0}, kinds, vals)
    f0 = (A + B) / (2 * math.pi * vals["CL"])
    r = reconstruct(g, H, vals, stamps, f0)
    assert r.verdict == "formula", r.reason
    assert r.prefactor == pytest.approx(1.0, abs=0.03)
    (blk,) = [b for b in r.blocks if b.terms]
    inner = {frozenset(t.inner) for t in blk.terms}
    assert inner == {frozenset({"gm_a", "G_a", "gds_a"}),
                     frozenset({"gm_b", "gds_b", "gds_c"})}
    for t in blk.terms:
        assert t.inner_exponent == -1.0
        assert all(t.exponents[n] in (0.0, 1.0) for n in t.members)
        assert t.inner_shares[max(t.inner, key=lambda n: vals[n])] \
            == pytest.approx(0.833 if "gm_a" in t.inner else 0.877,
                             abs=0.02)
    # addends by share, members by how much they move the root
    assert r.text == ("(gds_a*G_a/(gds_a+G_a+gm_a)"
                      "+gds_b*gds_c/(gds_b+gds_c+gm_b))/(2pi*CL)")
    assert r.coverage["reactive"] == pytest.approx(-1.0, abs=0.02)
    assert r.coverage["conductive"] == pytest.approx(1.0, abs=0.02)

    monkeypatch.setattr(mod, "_nested_divide", lambda *a, **k: None)
    assert reconstruct(g, H, vals, stamps, f0).verdict != "formula"


def _jet_fixture(name):
    """A captured jet: (g, H, vals, stamps, f0) from tests/fixtures/jets."""
    import json
    from pathlib import Path

    d = json.loads((Path(__file__).resolve().parent / "fixtures" / "jets"
                    / f"{name}.json").read_text())
    g = {n: complex(*v) for n, v in d["g"].items()}
    H = {}
    for n, m, re, im in d["H"]:
        H[n, m] = H[m, n] = complex(re, im)
    stamps = {n: [(k, None, None, n) for k in ks]
              for n, ks in d["kinds"].items()}
    return g, H, d["vals"], stamps, d["f0_hz"]


def test_the_folded_cascode_dominant_pole_reads_as_two_divides():
    """The real thing: the captured jet of the SKY130 folded cascode's
    dominant pole, read as the ledger reads it, with the lumps folded
    first. Its output conductance is two cascode divides adding over
    CL, and the whole must certify with both inner sums spelled: the
    NM branch over its node conductance and its gm + gmbs, the PM
    branch over gm + gmbs with the two gds the plain ratio dropped.
    Without the lumps the reading stays a partial, as it always did."""
    g, H, vals, stamps, f0 = _jet_fixture("fc_dominant_pole")
    r = reconstruct(g, H, vals, stamps, f0, floor=0.012, lump=True)
    assert r.verdict == "formula", r.reason
    assert r.prefactor == pytest.approx(0.98, abs=0.03)
    assert r.text.endswith("/(2pi*CL)")
    (blk,) = [b for b in r.blocks if b.terms]
    assert len(blk.terms) == 2
    nm, pm = sorted(blk.terms, key=lambda t: -t.share)
    assert nm.exponents["gds_I0_NM14"] == 1.0
    assert set(nm.inner) == {"gds_I0_PM11+gds_I0_NM10+gds_I0_NM13",
                             "gm_I0_NM14+gmbs_I0_NM14"}
    assert nm.inner_shares["gm_I0_NM14+gmbs_I0_NM14"] == pytest.approx(
        0.81, abs=0.03)
    assert set(pm.inner) == {"gds_I0_PM12", "gds_I0_PM13",
                             "gm_I0_PM12+gmbs_I0_PM12"}
    assert nm.share == pytest.approx(0.60, abs=0.03)
    raw = reconstruct(g, H, vals, stamps, f0, floor=0.012)
    assert raw.verdict == "partial"


@pytest.mark.parametrize("point, share_pm, prefactor", [
    ("1m_1n", 0.89, 1.01), ("50m_100p", 0.80, 1.06)])
def test_a_dominant_addend_still_reads_as_a_product(point, share_pm,
                                                    prefactor):
    """The ROMJIST 2022 high-PSR LDO's dominant pole on SKY130: the
    OTA's output conductance is the PMOS cascode's divide, carrying 0.8
    to 0.9 of the sum, beside the NMOS cascode's. An addend that
    dominates its sum couples its own members at (1 - sigma)/sigma, 0.12
    to 0.26 here, under the edge that keeps independent factors apart,
    so the product-partner search never seeded it and the reading
    stopped at G_eff/(2 pi C2). Below the edge a clearly nonzero value
    shared by the coupling and both diagonals is still the signature of
    a product, and the cascode's gm + gmbs joins on its couplings alone
    (its diagonal carries the inner sum). The whole then certifies as
    the paper's eq. (13) with the buffer's gain spelled: both divides
    over C2, times gds/gm of the pass device and the two followers'
    gm/(gm + gmbs + ...)."""
    g, H, vals, stamps, f0 = _jet_fixture(f"psrldo_{point}_dominant_pole")
    r = reconstruct(g, H, vals, stamps, f0, floor=0.012, lump=True)
    assert r.verdict == "formula", r.reason
    assert r.prefactor == pytest.approx(prefactor, abs=0.03)
    (blk,) = [b for b in r.blocks if b.terms]
    pm, nm = sorted(blk.terms, key=lambda t: -t.share)
    assert pm.share == pytest.approx(share_pm, abs=0.03)
    assert set(pm.members) == {"gds_M13", "gds_M14", "gm_M14+gmbs_M14"}
    assert set(pm.inner) == {"gds_M13", "gds_M14", "gm_M14+gmbs_M14"}
    assert nm.exponents == {"gds_M16": 1.0, "gm_M15+gmbs_M15": -1.0,
                            "gds_M15": 1.0}
    # the pass device's gds/gm, the two followers' gm over gm + gmbs
    # (+ the load's gds where it shows), over C2 (+ C1 where it shows)
    assert "*gds_MP*" in r.text and "*gm_MP*" in r.text
    assert "(gm_M20+gmbs_M20+gds_M20)" in r.text and "*gm_M20" in r.text
    assert "(gm_M19+gmbs_M19" in r.text and "*gm_M19" in r.text
    assert "/(2pi*C2" in r.text or "/(2pi*(C2+C1)" in r.text


def test_a_degeneration_reads_as_one_plus_gm_r():
    """A pole G/(2 pi C (1 + gm R)): the source resistor's factor is a
    sum with a bare 1 in it, which no symbol stands for. The product
    gm R is dimensionless, nothing else adds to it, its readings give
    (1 - s)/(a s) for the share s it carries beside the 1, and the
    values agree (gm R = 1, so s = 1/2): the reader writes the 1."""
    import math

    vals = {"G": 1e-6, "C": 1e-12, "gm": 1e-4, "R": 1e4}
    x = vals["gm"] * vals["R"]
    s = x / (1.0 + x)
    g = {"G": 1.0, "C": -1.0, "gm": -s, "R": -s}
    H = {}
    for a in vals:
        for b in vals:
            H[a, b] = 0.0
    # d2/(dlog gm dlog R) of -log(1 + x) and the diagonals alike
    for a in ("gm", "R"):
        for b in ("gm", "R"):
            H[a, b] = -s * (1.0 - s)
    stamps = {"G": [("g", None, None, "G")], "C": [("c", None, None, "C")],
              "gm": [("g", None, None, "gm")], "R": [("r", None, None, "R")]}
    f0 = vals["G"] / (2 * math.pi * vals["C"] * (1.0 + x))
    r = reconstruct(g, H, vals, stamps, f0)
    assert r.verdict == "formula", r.reason
    assert r.text in ("G/(2pi*C*(1+gm*R))", "G/(2pi*(1+gm*R)*C)")
    assert r.prefactor == pytest.approx(1.0, abs=0.02)
    (blk,) = [b for b in r.blocks if b.unit]
    assert blk.shares["1"] == pytest.approx(0.5, abs=0.02)
    assert "1 + gm" in r.tex() and "R" in r.tex()


def test_a_sum_of_time_constants_reads_through_its_anchors():
    """p = 1/(2 pi (C1 R1 (1 + K) + C2 R2)), K = gm R1 a loop gain: a
    sum of products of sums, each capacitor in one addend. The anchors
    are the capacitors: shares that sum to one, cross couplings at
    -1/a, and each one's row of second derivatives gives every other
    symbol's exponent inside its own addend. R1 and gm sit in C1's
    addend through the (1 + K) at the fraction K/(1 + K), which the
    values confirm; R2 sits in C2's at one."""
    import math

    vals = {"C1": 2e-12, "R1": 2e4, "gm": 1e-4, "C2": 1e-12, "R2": 5e4}
    K = vals["gm"] * vals["R1"]
    t1 = vals["C1"] * vals["R1"] * (1.0 + K)
    t2 = vals["C2"] * vals["R2"]
    S = t1 + t2
    s1, s2 = t1 / S, t2 / S
    tau = K / (1.0 + K)
    eps1 = {"C1": 1.0, "R1": 1.0 + tau, "gm": tau}
    eps2 = {"C2": 1.0, "R2": 1.0}
    names = list(vals)
    g = {n: 0.0 for n in names}
    for n, e in eps1.items():
        g[n] -= s1 * e
    for n, e in eps2.items():
        g[n] -= s2 * e
    H = {}
    for k in names:
        for l in names:
            pred = 0.0
            for s_t, eps in ((s1, eps1), (s2, eps2)):
                pred += s_t * eps.get(k, 0.0) * eps.get(l, 0.0)
                if eps is eps1 and k in ("R1", "gm") and l in ("R1", "gm"):
                    pred += s_t * tau * (1.0 - tau)
            gk = s1 * eps1.get(k, 0.0) + s2 * eps2.get(k, 0.0)
            gl = s1 * eps1.get(l, 0.0) + s2 * eps2.get(l, 0.0)
            H[k, l] = -(pred - gk * gl)
    stamps = {"C1": [("c", None, None, "C1")], "C2": [("c", None, None, "C2")],
              "R1": [("r", None, None, "R1")], "R2": [("r", None, None, "R2")],
              "gm": [("g", None, None, "gm")]}
    f0 = 1.0 / (2 * math.pi * S)
    r = reconstruct(g, H, vals, stamps, f0)
    assert r.verdict == "formula", r.reason
    assert r.prefactor == pytest.approx(1.0, abs=0.02)
    blk = next(b for b in r.blocks if b.terms)
    t_1, t_2 = sorted(blk.terms, key=lambda t: t.members[0])
    assert t_1.members[0] == "C1" and t_1.exponents["R1"] == 1.0
    assert {frozenset(m): b for m, b in t_1.units} == {frozenset({"gm", "R1"}): 1.0}
    assert t_2.members == ("C2", "R2")
    assert "(1+" in r.text and r.text.startswith("1/(2pi*")
    assert t_1.share == pytest.approx(s1, abs=1e-6)


def test_a_current_divider_reads_as_x_over_one_plus_x():
    """p = 1/(2 pi (C1 R1 K/(1 + K) + C2 R2)), K = gm R3: C1 feeds a
    source node that splits its current between R3 and the device,
    and the fraction reaching the mirror is K/(1 + K). Inside C1's
    addend gm and R3 read at 1 - tau each, tau = K/(1 + K): the
    fraction of a (1 + K)^-1 factor with an integer on top, which the
    values tell apart from (1 + K)^+1 at the fraction 1 - tau. The
    reader writes gm R3/(1 + gm R3)."""
    import math

    vals = {"C1": 2e-12, "R1": 2e4, "gm": 3e-5, "R3": 1e4,
            "C2": 1e-12, "R2": 5e4}
    K = vals["gm"] * vals["R3"]
    tau = K / (1.0 + K)
    t1 = vals["C1"] * vals["R1"] * tau
    t2 = vals["C2"] * vals["R2"]
    S = t1 + t2
    s1, s2 = t1 / S, t2 / S
    eps1 = {"C1": 1.0, "R1": 1.0, "gm": 1.0 - tau, "R3": 1.0 - tau}
    eps2 = {"C2": 1.0, "R2": 1.0}
    names = list(vals)
    g = {n: 0.0 for n in names}
    for n, e in eps1.items():
        g[n] -= s1 * e
    for n, e in eps2.items():
        g[n] -= s2 * e
    H = {}
    for k in names:
        for l in names:
            pred = 0.0
            for s_t, eps in ((s1, eps1), (s2, eps2)):
                pred += s_t * eps.get(k, 0.0) * eps.get(l, 0.0)
                if eps is eps1 and k in ("R3", "gm") and l in ("R3", "gm"):
                    pred -= s_t * tau * (1.0 - tau)
            gk = s1 * eps1.get(k, 0.0) + s2 * eps2.get(k, 0.0)
            gl = s1 * eps1.get(l, 0.0) + s2 * eps2.get(l, 0.0)
            H[k, l] = -(pred - gk * gl)
    stamps = {n: [("c" if n.startswith("C") else "r" if n.startswith("R")
                   else "g", None, None, n)] for n in names}
    f0 = 1.0 / (2 * math.pi * S)
    r = reconstruct(g, H, vals, stamps, f0)
    assert r.verdict == "formula", r.reason
    assert r.prefactor == pytest.approx(1.0, abs=0.02)
    blk = next(b for b in r.blocks if b.terms)
    t_1 = next(t for t in blk.terms if t.members[0] == "C1")
    assert t_1.exponents["gm"] == 1.0 and t_1.exponents["R3"] == 1.0
    assert {frozenset(m): b for m, b in t_1.units} == {frozenset({"gm", "R3"}): -1.0}
    assert "/(1+" in t_1.text() and "gm" in t_1.text()
    assert t_1.share_value == pytest.approx(s1, abs=0.01)


def test_the_fast_ldo_dominant_pole_reads_as_four_time_constants():
    """The captured jet of the IEEE Access 2022 fast LDO's dominant pole
    at 20 uA (SKY130 re-design): a sum of four time constants over the
    gate conductance, read through its anchors C1, C2 and the pass
    device's Cgg and Cgd. Each addend names its own factors: C1 under
    the degeneration resistor and the mirror's gm with the pass device's
    gds/gm; Cgg and Cgd under (1 + Rp2a (gm + gmbs)); C2 under that and
    its own current divider Rn2 gm10'/(1 + Rn2 gm10'), the fraction of
    C2's current the NMOS mirror's source passes on, which read as the
    exponent 0.78 = 1/(1 + Rn2 gm10') until the unit production learnt
    an integer on top of a fraction. The gate conductance's NMOS path
    (gds_M8 at 1.6 % of the root, 22 % of the value) is set beside the
    sum as residue rather than seated as a plain addend; the reading
    certifies at a prefactor within 3 % of one."""
    g, H, vals, stamps, f0 = _jet_fixture("fastldo_20u_1p_dominant_pole")
    r = reconstruct(g, H, vals, stamps, f0, floor=0.012, lump=True)
    assert r.verdict == "formula", r.reason
    assert r.prefactor == pytest.approx(0.97, abs=0.05)
    blk = next(b for b in r.blocks if b.terms and b.kind == "reactive")
    assert blk.exponent == -1.0 and blk.plain
    by = {t.members[0]: t for t in blk.terms}
    assert set(by) == {"C1", "C2", "kgg_MPASS", "kgd_MPASS"}
    assert by["C1"].exponents["Rp2a"] == 1.0
    assert by["C1"].exponents["gm_MPASS"] == 1.0
    assert by["C1"].exponents["gds_MPASS"] == -1.0
    loop = frozenset({"Rp2a", "gm_M4yPa+gmbs_M4yPa"})
    divider = frozenset({"Rn2", "gm_M10yN+gmbs_M10yN"})
    assert {frozenset(m): b for m, b in by["kgg_MPASS"].units} == {loop: 1.0}
    assert "gm_MPASS" not in by["kgg_MPASS"].exponents
    assert {frozenset(m): b for m, b in by["C2"].units} == {loop: 1.0, divider: -1.0}
    assert by["C2"].exponents["Rn2"] == 1.0
    assert by["C2"].exponents["gm_M10yN+gmbs_M10yN"] == 1.0
    assert "/(gds_MPASS*(1+Rn2*(gm_M10yN+gmbs_M10yN)))" in by["C2"].text()
    assert abs(blk.shares[by["C2"].text()] - 0.32) < 0.03
    assert by["C2"].share_value == pytest.approx(by["C2"].share, abs=0.01)
    assert [n for n, _v, _w in r.residue] == ["gds_M8"]
    assert "weighs 0.22 of its sum with gds_M4yPa" in r.reason
