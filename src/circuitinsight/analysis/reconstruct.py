"""Reading a root's formula out of its sensitivity jet.

The level-2 coupling rho_ij = H_ij/(g_i g_j) is not merely a hint: for
the grammar hand analysis actually uses -- a monomial of sums,
f0 = prod_k S_k**a_k with S_k = sum_{i in k} c_i x_i -- it is
DETERMINED. Writing sigma_i = c_i x_i / S_k for a member's share,

    g_i    = a_k sigma_i          so   sum_{i in k} g_i = a_k
    H_ij   = -a_k sigma_i sigma_j so   rho_ij = -1/a_k       (i != j)
    H_ii   = g_i (1 - sigma_i)
    rho_ij = 0                                    (different factors)

so each block announces its exponent twice over -- once through the
couplings, once through the shares -- and the diagonal announces it a
third time. A single symbol standing alone is the degenerate case
sigma = 1: exponent g_i, vanishing diagonal, coupled to nothing.

This module therefore does not search a vocabulary or fit a value. It
partitions the movers by their couplings, reads each block's exponent,
and CHECKS the three readings against each other. When they agree the
expression is emitted with its residuals; when they disagree -- nested
forms, symbols belonging to two blocks at once -- it says so, and when
no partition stands at all it reports the root as shared, which for a
dominant pole is a design verdict in its own right: the response has
not been conditioned onto anything yet.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# how a symbol's parameter scales its ADMITTANCE, for the Euler
# coverage rules: +1 when Y grows with the parameter, -1 when it is a
# reciprocal (a resistance, an inductance)
_CLASS = {"c": ("reactive", +1), "cx": ("reactive", +1),
          "l": ("reactive", -1), "g": ("conductive", +1),
          "vccs": ("conductive", +1), "r": ("conductive", -1)}
# Exponents a formula is allowed to wear. Halves are in because a
# coalesced pair genuinely carries them; THIRDS are deliberately out --
# a third means a depth-three nesting, which this grammar cannot spell,
# and snapping to one would dress a deeper structure as a formula.
_NICE = (1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 3.0, -3.0, 1.5, -1.5)


@dataclass
class Block:
    """One factor of the reconstructed expression."""
    members: tuple
    exponent: float                  # a_k, snapped to a nice rational
    exponent_raw: float              # before snapping, from the shares
    exponent_from_rho: float | None  # the independent reading
    shares: dict = field(default_factory=dict)
    plain: bool = True               # shares track the VALUES: a real
    #                                  parallel combination, not a
    #                                  weighted sum
    spread: float = 0.0              # worst within-block rho deviation

    def text(self) -> str:
        body = "+".join(self.members)
        if len(self.members) > 1:
            body = f"({body})"
        e = abs(self.exponent)
        return body if abs(e - 1.0) < 1e-9 else f"{body}^{e:g}"


@dataclass
class Reconstruction:
    verdict: str                     # formula | inconclusive | shared
    blocks: tuple = ()
    prefactor: float | None = None   # 2 pi f0 / prod(S_k**a_k): ~1 for
    #                                  a textbook form with unit weights
    coverage: dict = field(default_factory=dict)
    text: str = ""
    reason: str = ""
    movers: tuple = ()
    value: float | None = None       # what the NAMED structure alone
    #                                  predicts, prefactor set aside:
    #                                  the equation prints this beside
    #                                  the exact root
    two_pi: bool = True              # the target was a frequency
    aggregate: dict | None = None    # one side read, the rest named as
    #                                  a single quantity that the
    #                                  identity then fixes exactly:
    #                                  {symbol, unit, value, exponent}
    expansion: dict | None = None    # what that named quantity turned
    #                                  out to BE, once the port was
    #                                  asked: {text, tex}. Filled in on
    #                                  demand -- the port costs seconds
    #                                  -- and once filled the equation
    #                                  prints the algebra in place of
    #                                  the bare G_eff symbol, which is
    #                                  where a reader looks for it

    def tex(self, texer=None) -> str:
        """The expression in LaTeX. Half exponents render as roots,
        which is how a coalesced pair is written by hand."""
        tx = texer or (lambda s: s.replace("_", r"\_"))

        def body(b):
            inner = " + ".join(tx(n) for n in b.members)
            e = abs(b.exponent)
            if abs(e - 0.5) < 1e-9:
                return rf"\sqrt{{{inner}}}"
            if len(b.members) > 1:
                inner = rf"\left({inner}\right)"
            return inner if abs(e - 1.0) < 1e-9 else rf"{inner}^{{{e:g}}}"

        agg = self.aggregate
        blocks = agg["blocks"] if agg else self.blocks
        num = [body(b) for b in blocks if b.exponent > 0]
        den = ([r"2\pi"] if self.two_pi else []) + [
            body(b) for b in blocks if b.exponent < 0]
        if agg:
            # the unread side, named: exact by the identity, so it
            # wears its own symbol rather than a question mark -- or
            # the algebra itself, once the port has been asked
            sym = rf"{agg['symbol'][0]}_{{\mathrm{{eff}}}}"
            if self.expansion and self.expansion.get("tex"):
                sym = rf"\left[{self.expansion['tex']}\right]"
            (num if agg["exponent"] > 0 else den).insert(0, sym)
        elif self.verdict != "formula":
            num = ["?"] + num
        top = r"\,".join(num) or "1"
        if not den:
            return top
        # the join is bound FIRST: a backslash inside an f-string
        # expression is a syntax error before 3.12, and this package
        # runs on 3.11 in the field
        bot = r"\,".join(den)
        return rf"\frac{{{top}}}{{{bot}}}"

    def display_text(self) -> str:
        """`text` with the aggregate replaced by what the port said it
        is. `text` is built at construction, before any port has been
        asked, so it keeps the bare symbol; `tex` is late-bound and
        picks the expansion up on its own."""
        exp = (self.expansion or {}).get("text")
        if not (self.aggregate and exp):
            return self.text
        return self.text.replace(self.aggregate["symbol"],
                                 f"[{exp}]", 1)

    def describe(self) -> str:
        cov = "  ".join(f"{k} {v:+.2f}" for k, v in
                        sorted(self.coverage.items()))
        if self.verdict == "formula":
            return (f"{self.text}   [prefactor {self.prefactor:.3f}, "
                    f"coverage {cov}]")
        if self.verdict == "partial":
            return f"{self.text}   PARTIAL: {self.reason}"
        return f"SHARED: {self.reason}"


def _snap(a: float, tol: float = 0.12):
    """Nearest low-entropy exponent, or None if the reading is not one
    of them -- a fractional exponent that snaps to nothing is itself
    the finding (a deeper nesting), not a number to round away."""
    for cand in _NICE:
        if abs(a - cand) <= tol * max(abs(cand), 1.0):
            return cand
    return None


def _rho_of(rho, a, b):
    return rho.get((a, b), rho.get((b, a)))


def _read_exponent(comp, rs, gr, agree_tol: float):
    """A block's exponent, read twice and required to agree: once from
    the couplings (a = -1/rho) and once from the shares
    (a = sum of g over the members). Returns (a, a_rho, a_raw) or None
    when the two readings disagree or the result is not a low-entropy
    exponent -- either way the group is not a plain sum."""
    a_rho = -len(rs) / sum(rs) if len(comp) > 1 else None
    a_sum = sum(gr[n] for n in comp)
    if a_rho is not None and \
            abs(a_rho - a_sum) > agree_tol * max(abs(a_sum), 0.5):
        return None
    a = _snap(a_sum)
    return None if a is None else (a, a_rho, a_sum)


def _cliques(names, rho, gr, cls, strong, edge: float, tol: float,
             agree_tol: float):
    """Maximal groups whose members ALL couple to one another at ONE
    rho value -- the signature of a plain sum.

    Connected components will not do: a symbol that belongs to two
    sums at once, which is exactly what a nested form looks like,
    chains every block into one and destroys the reading. Blocks are
    therefore grown greedily from the strongest coupled pair, and a
    candidate joins only if it agrees with EVERY member already in.
    Returns (blocks, singletons, entangled): symbols coupled to
    something but fitting no block are entangled, and naming them is
    the honest half of the answer."""
    free = set(names)
    blocks = []
    pairs = sorted(((a, b) for i, a in enumerate(names)
                    for b in names[i + 1:]
                    if _rho_of(rho, a, b) is not None
                    and abs(_rho_of(rho, a, b)) >= edge
                    # a SUM is dimensionally homogeneous: a
                    # capacitance and a transconductance never add
                    and cls.get(a) == cls.get(b) and cls.get(a)
                    # rho divides by both g's, so a reading taken
                    # through a feeble mover is inflated noise: it may
                    # not SEED a block, though it may still join one
                    and (a in strong or b in strong)),
                   key=lambda ab: -abs(gr[ab[0]] * gr[ab[1]]))
    for a, b in pairs:
        if a not in free or b not in free:
            continue
        grp = [a, b]
        rs = [_rho_of(rho, a, b)]

        def fits(c, grp=grp, rs=rs, kind=cls.get(a)):
            if cls.get(c) != kind:
                return None
            got = []
            for m in grp:
                r = _rho_of(rho, c, m)
                if r is None or abs(r) < edge:
                    return None
                got.append(r)
            ref = sum(rs) / len(rs)
            if any(r * ref <= 0 or abs(r - ref) > tol * abs(ref)
                   for r in got):
                return None
            return got

        for c in sorted(free - {a, b}, key=lambda n: -abs(gr[n])):
            got = fits(c)
            if got is not None:
                grp.append(c)
                rs.extend(got)
        def rs_of(members):
            return [_rho_of(rho, x, y) for i2, x in enumerate(members)
                    for y in members[i2 + 1:]]

        # A candidate block only CONSUMES its members once its two
        # exponent readings agree. Greedy growth can overreach -- an
        # element straddling two nodes joins at the tolerance edge and
        # breaks a group that would otherwise certify -- so the worst
        # -agreeing member is dropped and the core retried before the
        # group is abandoned.
        read = _read_exponent(grp, rs, gr, agree_tol)
        while read is None and len(grp) > 2:
            cur = rs_of(grp)
            ref = sum(cur) / len(cur)

            def dev(m, grp=grp, ref=ref):
                rr = [_rho_of(rho, m, o) for o in grp if o != m]
                return abs(sum(rr) / len(rr) - ref)

            grp = [x for x in grp if x != max(grp, key=dev)]
            read = _read_exponent(grp, rs_of(grp), gr, agree_tol)
        if read is None:
            continue
        free -= set(grp)
        blocks.append((sorted(grp), rs_of(grp), read))
    singles, entangled = [], []
    for n in sorted(free, key=lambda x: -abs(gr[x])):
        # judged only against couplings that can be TRUSTED, else a
        # feeble mover's inflated rho would entangle a strong one
        peak = max((abs(_rho_of(rho, n, m) or 0.0)
                    for m in strong if m != n), default=0.0)
        (singles if peak < edge else entangled).append(n)
    return blocks, singles, entangled


def reconstruct(g, H, vals, stamps, target, *, two_pi: bool = True,
                floor: float = 0.02,
                trust: float = 0.08,
                edge: float = 0.30, rho_tol: float = 0.25,
                agree_tol: float = 0.20, plain_tol: float = 0.10,
                prefactor_band: float = 0.25,
                coverage_tol: float = 0.12) -> Reconstruction:
    """Read the expression for a root from its jet.

    `g`, `H` come from secondorder.loop_pole_jet or loop_pair_jets;
    `vals` and `stamps` from marginledger._collect_stamps; `target` is
    the value of the reconstructed quantity, used only to report the
    prefactor. `two_pi` says whether that quantity is a FREQUENCY, so
    the 2 pi belongs in the rendering -- a damping ratio is
    dimensionless and carries none. Symbols moving the root by less
    than `floor` are left out of the formula and counted against its
    coverage instead."""
    gr = {n: complex(v).real for n, v in g.items()}
    names = sorted((n for n, v in gr.items() if abs(v) >= floor),
                   key=lambda n: -abs(gr[n]))
    movers = tuple((n, gr[n]) for n in names[:6])
    if not names:
        return Reconstruction(verdict="shared", movers=movers,
                              reason="no element moves this root")

    rho = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            gg = gr[a] * gr[b]
            if abs(gg) > 1e-9:
                rho[a, b] = complex(H[a, b]).real / gg

    # the Euler coverage rules: scaling every reactance by k scales
    # every root by 1/k, every conductance by k scales it by k, so the
    # class sums are fixed at -1 and +1 whatever the circuit. How much
    # of each the NAMED members carry is the formula's completeness.
    cls, sgn = {}, {}
    for n in names:
        for k in {st[0] for st in stamps.get(n, ())}:
            c, sign = _CLASS.get(k, (None, 0))
            if c:
                cls[n], sgn[n] = c, sign
                break
    # What may be ADDED is finer than what shares a class: a
    # conductance and a resistance both scale the admittance, which is
    # what the coverage rule cares about, but G + R is not a quantity.
    # Series resistances add, parallel conductances add, node
    # capacitances add -- so homogeneity keys on the sign too.
    homo = {n: (cls[n], sgn[n]) for n in cls}
    def class_sums(pool):
        return {k: sum(sgn[n] * gr[n] for n in pool if cls.get(n) == k)
                for k in ("reactive", "conductive")}
    strong = {n for n in names if abs(gr[n]) >= trust}
    raw, singles, entangled = _cliques(names, rho, gr, homo, strong,
                                       edge, rho_tol, agree_tol)
    blocks, problems = [], []
    for comp, rs, read in raw + [([n], [], None) for n in singles]:
        if read is None:                        # a lone factor
            hii = complex(H[comp[0], comp[0]]).real
            if abs(hii) > 0.15 * max(abs(gr[comp[0]]), 1e-9):
                entangled.append(comp[0])
                continue
            read = _read_exponent(comp, rs, gr, agree_tol)
            if read is None:
                entangled.append(comp[0])
                continue
        a, a_rho, a_sum = read
        shares = {n: gr[n] / a for n in comp}
        plain = True
        if len(comp) > 1:
            tot = sum(abs(vals[n]) for n in comp)
            plain = all(abs(shares[n] - abs(vals[n]) / tot) <= plain_tol
                        for n in comp)
        blocks.append(Block(members=tuple(
            sorted(comp, key=lambda n: -abs(gr[n]))),
            exponent=a, exponent_raw=a_sum, exponent_from_rho=a_rho,
            shares=shares, plain=plain,
            spread=(max(rs) - min(rs)) if rs else 0.0))

    # COVERAGE is a property of the EMITTED expression, so it counts
    # only what the blocks actually name. Scaling every reactance by k
    # scales every root by 1/k and every conductance by k scales it by
    # k, so the two class sums are fixed at -1 and +1 whatever the
    # circuit; how much of each the formula carries is exactly how
    # complete it is. Elements left entangled show up here as the
    # shortfall they cause, which is why a negligible one costs
    # nothing and a load-bearing one demotes the verdict.
    # The targets depend on WHAT was read. Scaling every reactance by
    # k scales a FREQUENCY by 1/k and every conductance by k scales it
    # by k, hence -1 and +1. A DAMPING RATIO is dimensionless and
    # those same scalings leave it alone, so its class sums are both
    # zero -- checking zeta against -1/+1 would condemn every correct
    # reading of it.
    t_re, t_co = ((-1.0, +1.0) if two_pi else (0.0, 0.0))
    named = [n for b in blocks for n in b.members]
    coverage = class_sums(named)
    worst_cov = max(abs(coverage["reactive"] - t_re),
                    abs(coverage["conductive"] - t_co))

    import math

    def render(bs, whole: bool):
        num = [b for b in bs if b.exponent > 0]
        den = [b for b in bs if b.exponent < 0]
        top = "*".join(b.text() for b in num) or "1"
        if not whole:                       # an unread factor remains
            top = "?*" + top if num else "?"
        parts = (["2pi"] if two_pi else []) + [b.text() for b in den]
        return f"{top}/({'*'.join(parts)})" if parts else top

    if not blocks:
        if not problems:
            top = sorted(entangled or names,
                         key=lambda n: -abs(gr[n]))[:4]
            problems.append(
                "no group of elements couples as a single sum: "
                + ", ".join(f"{n} ({gr[n]:+.2f})" for n in top))
        return Reconstruction(verdict="shared", movers=movers,
                              coverage=coverage,
                              reason="; ".join(problems[:2]))

    # The prefactor only MEANS anything once the expression has the
    # right dimensions, and the coverage sums are precisely that
    # balance: each block contributes its exponent to its class, so
    # reactive -1 and conductive +1 is the statement that the product
    # is a frequency. Read it off an unbalanced partial and it absorbs
    # the missing units instead -- a pair short one capacitance block
    # reported 2.6e5.
    prefactor = named = None
    if worst_cov <= coverage_tol:
        pref = 1.0
        for b in blocks:
            v = sum(abs(vals[n]) for n in b.members)
            pref *= v ** b.exponent
        scale = 2 * math.pi * target if two_pi else target
        prefactor = scale / pref if pref else None
        named = pref / (2 * math.pi) if two_pi else pref
    # THE VALUE CHECK. The prefactor is the one constant the structure
    # does not fix, and a textbook form with plain unweighted sums
    # lands on one -- measured 0.98, 1.00, 0.99, 0.92, 1.09 across the
    # benches where the answer is known. Far from one means a factor
    # was not read, so the expression may not wear the certified
    # verdict however cleanly its exponents partitioned (field case:
    # a differential bench pole read structurally but reproduced only
    # 61% of its own frequency).
    if prefactor is not None and abs(prefactor - 1.0) > prefactor_band:
        problems.append(
            f"the named structure reproduces {prefactor:.2f} of the "
            f"value, not 1: a factor is missing")
    if worst_cov > coverage_tol:
        miss = sorted(entangled, key=lambda n: -abs(gr[n]))[:3]
        problems.insert(0, (
            f"the named blocks carry {coverage['reactive']:+.2f} / "
            f"{coverage['conductive']:+.2f} of the {t_re:+.0f} / "
            f"{t_co:+.0f} this quantity must show")
            + (", missing " + ", ".join(f"{n} ({gr[n]:+.2f})"
                                        for n in miss) if miss else ""))
    agg = None
    if problems and blocks:
        # ONE SIDE CERTIFIES. A folded cascode's output conductance is
        # a sum of cascode divides -- terms that are themselves
        # products and ratios -- which this grammar cannot spell, so a
        # pole whose CAPACITANCE reads perfectly still fails as a
        # whole. Naming the unread side as a single quantity is not a
        # fit: with the other side exact, it is defined by the
        # identity, and G/(2 pi C) is how the pole is written by hand
        # anyway. The designer gets the form and the number it stands
        # for; what was not decomposed is said plainly.
        for keep_c, other_c, t_keep, t_other in (
                ("reactive", "conductive", t_re, t_co),
                ("conductive", "reactive", t_co, t_re)):
            kept = [b for b in blocks
                    if cls.get(b.members[0]) == keep_c]
            if not kept or abs(t_other) < 1e-9:
                continue
            cov_k = sum(sgn[n] * gr[n] for b in kept for n in b.members)
            if abs(cov_k - t_keep) > coverage_tol:
                continue
            prod = 1.0
            for b in kept:
                prod *= sum(abs(vals[n])
                            for n in b.members) ** b.exponent
            scale = 2 * math.pi * target if two_pi else target
            if prod <= 0 or scale <= 0:
                continue
            agg = {"symbol": "G_eff" if other_c == "conductive"
                   else "C_eff",
                   "unit": "S" if other_c == "conductive" else "F",
                   "value": float((scale / prod) ** (1.0 / t_other)),
                   "exponent": float(t_other),
                   "blocks": tuple(kept)}
            break
    if problems:
        # PARTIAL: what certified is still worth saying, and what did
        # not is named rather than papered over
        text = render(blocks, whole=False)
        if agg is not None:
            num = ([agg["symbol"]] if agg["exponent"] > 0 else []) + [
                b.text() for b in agg["blocks"] if b.exponent > 0]
            den = ((["2pi"] if two_pi else [])
                   + [b.text() for b in agg["blocks"] if b.exponent < 0]
                   + ([agg["symbol"]] if agg["exponent"] < 0 else []))
            text = (("*".join(num) or "1")
                    + (f"/({'*'.join(den)})" if den else ""))
            # the discarded side's prefactor complaint described a
            # form no longer being shown; this one is exact by the
            # identity, so the only thing left to say is WHAT was not
            # decomposed
            other = ("conductance" if agg["unit"] == "S"
                     else "capacitance")
            problems = [f"{agg['symbol']} = {agg['value']:.4g} "
                        f"{agg['unit']} stands for the {other} this "
                        f"grammar could not decompose"]
        return Reconstruction(verdict="partial", movers=movers,
                              blocks=tuple(blocks), coverage=coverage,
                              prefactor=prefactor, value=named,
                              two_pi=two_pi, aggregate=agg,
                              text=text,
                              reason="; ".join(problems[:2]))
    return Reconstruction(verdict="formula", blocks=tuple(blocks),
                          prefactor=prefactor, value=named,
                          two_pi=two_pi,
                          coverage=coverage, movers=movers,
                          text=render(blocks, whole=True))
