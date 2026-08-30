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
