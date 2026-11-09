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

A sum whose addends are themselves PRODUCTS -- C_L + M gm R1 Cc, or
the two cascode divides a folded output conductance is made of -- is
the same grammar one level up. Writing S_k = sum_t A_t with each
addend a monomial A_t = prod_{i in t} x_i**e_i and sigma_t = A_t/S_k
the addend's share,

    g_i    = a_k e_i sigma_t
    rho_ij = -1/a_k                       (i, j in different addends)
    rho_ij = (1 - sigma_t)/(a_k sigma_t)  (i, j in the same addend)
    H_ii   = g_i**2 (1 - sigma_t)/(a_k sigma_t)

so two symbols multiplied inside one addend couple to each other at
exactly the value each one's own diagonal shows, and that value names
the addend's share once the exponent is known from the couplings
across addends. A plain sum of symbols is the case where every addend
has one member at e = 1. The reader looks for product partners first,
groups the addends into factors by the -1/a they share, and reads the
member exponents e_i = g_i/(a sigma_t) -- which must land on integers
or halves, or the addend is not a monomial.

A DIVIDE -- a symbol standing alone in the numerator and inside a sum
in the denominator, gm/(gm + gmK), gds G/(gm + G + gds), the series
combination gm1 gm2/(gm1 + gm2) -- is the one form a hand analysis
writes that neither grammar above spells, and it too is determined.
For f0 = prod_k x_k**e_k * (sum_k x_k)**a over one set of symbols,

    g_k    = e_k + a sigma_k
    H_kl   = -a sigma_k sigma_l                      (k != l)
    H_kk   = a sigma_k (1 - sigma_k)

so each member's own diagonal fixes its share up to the two roots of
sigma (1 - sigma) = H_kk/a, the exponent it wears alone follows as
g_k - a sigma_k and must land on an integer, a half or zero, the
shares must sum to one and every pair must show -a sigma_k sigma_l.
Such a group is read as the sum raised to a, times each overlapping
member raised to its own exponent. It is the form the plain partition
refuses -- the couplings of gm/(gm + gmK) say a = +1 and the shares
say a = 0 -- so it is sought exactly where that refusal left symbols
entangled.

The same divide INSIDE AN ADDEND -- a folded output conductance is
the sum of two cascode divides, gds G/(gm + G + gds) each -- is the
two grammars composed. With A_t = prod x_i**e_i (sum_k x_k)**b and
tau_k the inner shares, every member carries the effective exponent
eps_k = e_k + b tau_k in place of e_k: the couplings across addends
keep their -1/a, and within the addend the second derivative is the
monomial part a sigma_t (1 - sigma_t) eps_i eps_j plus a residual
-a sigma_t b tau_i tau_j off the diagonal and a sigma_t b tau_i
(1 - tau_i) on it, which divided by a sigma_t is the divide jet above
in the inner shares. So an addend whose readings do not sit together
is re-read as a divide: the inner shares from its residual, the outer
exponents as eps_k - b tau_k, which must again land on integers,
halves or zero, and the addend's share from the other addends, since
the shares sum to one.

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


def _sym_text(name: str) -> str:
    """A symbol as it prints inside a product. A lump the pre-pass
    made carries its own sum as its name, and a sum inside a product
    needs its parentheses."""
    return f"({name})" if "+" in name else name


def _pow_text(body: str, e: float) -> str:
    e = abs(e)
    return body if abs(e - 1.0) < 1e-9 else f"{body}^{e:g}"


@dataclass
class Term:
    """One ADDEND of a sum of monomials: a product of symbols, each
    at an integer or half exponent, with the share of the sum it
    carries."""
    members: tuple                   # strongest first
    exponents: dict                  # name -> e_i, snapped
    exponents_raw: dict              # name -> g_i/(a sigma_t)
    share: float                     # sigma_t from the shares: sum of
    #                                  g over the addend / (a sum e)
    share_from_rho: float = 0.0      # the independent reading, from
    #                                  the addend's own couplings
    share_value: float | None = None  # sigma_t from the VALUES
    spread: float = 0.0              # worst reading deviation inside
    inner: tuple = ()                # members of the DIVIDE's sum,
    #                                  when the addend is one: a
    #                                  cascode's gm + G + gds
    inner_exponent: float = 0.0      # b, the power that sum wears
    inner_shares: dict = field(default_factory=dict)   # tau_k
    units: tuple = ()                # the (1 + x)^b factors the addend
    #                                  carries, as (members, b): x a
    #                                  dimensionless product (a loop
    #                                  gain, gm R), each member at its
    #                                  own integer in `exponents` on
    #                                  top (gm R/(1 + gm R) is gm and
    #                                  R at one with b = -1)

    def _inner_text(self) -> str:
        # a sum of sums is flat, so a lump inside prints bare
        return _pow_text(f"({'+'.join(self.inner)})", self.inner_exponent)

    @staticmethod
    def _unit_text(members, b) -> str:
        return _pow_text(f"(1+{'*'.join(_sym_text(n) for n in members)})", b)

    def text(self) -> str:
        num = [_pow_text(_sym_text(n), self.exponents[n])
               for n in self.members if self.exponents[n] > 0]
        den = [_pow_text(_sym_text(n), self.exponents[n])
               for n in self.members if self.exponents[n] < 0]
        if self.inner:
            (num if self.inner_exponent > 0 else den).append(
                self._inner_text())
        for um, b in self.units:
            (num if b > 0 else den).append(self._unit_text(um, b))
        top = "*".join(num) or "1"
        if not den:
            return top
        bot = "*".join(den)
        return f"{top}/({bot})" if len(den) > 1 else f"{top}/{bot}"

    def value(self, vals) -> float:
        v = 1.0
        for n in self.members:
            v *= abs(vals[n]) ** self.exponents[n]
        if self.inner:
            v *= sum(abs(vals[n]) for n in self.inner) ** self.inner_exponent
        for um, b in self.units:
            x = 1.0
            for n in um:
                x *= abs(vals[n])
            v *= (1.0 + x) ** b
        return v

    def net(self, sgn: dict, cls: dict) -> tuple:
        """(reactive, conductive) exponent this addend carries: the
        dimension it has, which every addend of one sum must share."""
        out = {"reactive": 0.0, "conductive": 0.0}
        for n in self.members:
            if cls.get(n):
                out[cls[n]] += sgn[n] * self.exponents[n]
        if self.inner and cls.get(self.inner[0]):
            n = self.inner[0]
            out[cls[n]] += sgn[n] * self.inner_exponent
        return (out["reactive"], out["conductive"])


@dataclass
class Block:
    """One factor of the reconstructed expression: a sum of symbols
    raised to a power, or -- when `terms` is set -- a sum of monomials
    raised to one. `members` lists every symbol the factor names
    either way, so a consumer that only walks names sees both."""
    members: tuple
    exponent: float                  # a_k, snapped to a nice rational
    exponent_raw: float              # before snapping, from the shares
    exponent_from_rho: float | None  # the independent reading
    shares: dict = field(default_factory=dict)
    plain: bool = True               # shares track the VALUES: a real
    #                                  parallel combination, not a
    #                                  weighted sum
    spread: float = 0.0              # worst within-block rho deviation
    terms: tuple = ()                # the addends, when they are
    #                                  monomials rather than symbols
    kind: str | None = None          # reactive | conductive | None for
    #                                  a mixed sum of monomials
    divide: tuple = ()               # members of this sum that ALSO
    #                                  stand alone as factors of the
    #                                  expression: a cascode divide, a
    #                                  series combination. Those lone
    #                                  factors are blocks of their own
    unit: bool = False               # the sum carries a bare 1 among
    #                                  its addends: (1 + gm R), the
    #                                  factor a degeneration resistor
    #                                  or a loop gain leaves. Only a
    #                                  dimensionless addend can sit
    #                                  beside a 1

    def text(self) -> str:
        if self.terms:
            body = "+".join((["1"] if self.unit else [])
                            + [t.text() for t in self.terms])
            many = len(self.terms) > 1 or self.unit
        else:
            # a lone member that is itself a lump prints as the sum
            # it stands for, parentheses and all
            body = "+".join(self.members)
            many = len(self.members) > 1 or "+" in body
        return _pow_text(f"({body})" if many else body, self.exponent)

    def value(self, vals) -> float:
        """The factor evaluated at the operating point, exponent and
        all: what the prefactor divides by."""
        if self.terms:
            v = sum(t.value(vals) for t in self.terms) + (
                1.0 if self.unit else 0.0)
        else:
            v = sum(abs(vals[n]) for n in self.members)
        return v ** self.exponent


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
    residue: tuple = ()              # movers a sum of monomials could
    #                                  not seat: (symbol, g, why), kept
    #                                  beside the expression rather
    #                                  than folded into it
    lumps: dict = field(default_factory=dict)  # name -> members, for
    #                                  the sums the pre-pass folded
    #                                  into single symbols
    constant: float = 1.0            # a bare number the expression
    #                                  carries -- the 2 of zeta = 1/(2Q)
    #                                  -- read off a prefactor that
    #                                  landed on it; `prefactor` keeps
    #                                  the raw ratio, `value` includes
    #                                  the constant

    def tex(self, texer=None, agg_tag=None) -> str:
        """The expression in LaTeX. Half exponents render as roots,
        which is how a coalesced pair is written by hand.

        `agg_tag` tags the aggregate symbol with \\htmlData so the
        equation view can make it a target -- G_eff is the one thing
        in that line a reader wants to act on, and the port algebra
        behind it is exactly what the tag triggers. Dropped once the
        expansion has arrived: there is then nothing left to ask for.
        """
        tx = texer or (lambda s: s.replace("_", r"\_"))

        def sym(n, bare=False):
            # a lump carries its sum as its name; inside a product
            # that sum needs its parentheses, alone on one side of a
            # fraction bar it does not
            if "+" in n:
                inner = " + ".join(tx(m) for m in n.split("+"))
                return inner if bare else rf"\left({inner}\right)"
            return tx(n)

        def powered(n, e, bare=False):
            e = abs(e)
            if abs(e - 0.5) < 1e-9:
                return rf"\sqrt{{{sym(n, bare=True)}}}"
            if abs(e - 1.0) < 1e-9:
                return sym(n, bare)
            return rf"{sym(n)}^{{{e:g}}}"

        def term(t):
            ups = [n for n in t.members if t.exponents[n] > 0]
            downs = [n for n in t.members if t.exponents[n] < 0]
            num = [powered(n, t.exponents[n], bare=len(ups) == 1 and
                           bool(downs)) for n in ups]
            den = [powered(n, t.exponents[n], bare=len(downs) == 1)
                   for n in downs]
            if t.inner:
                inner = " + ".join(sym(n, bare=True) for n in t.inner)
                b = abs(t.inner_exponent)
                side = num if t.inner_exponent > 0 else den
                if abs(b - 0.5) < 1e-9:
                    piece = rf"\sqrt{{{inner}}}"
                elif abs(b - 1.0) < 1e-9 and not side:
                    piece = inner          # alone on its side: bare
                else:
                    piece = rf"\left({inner}\right)"
                    if abs(b - 1.0) >= 1e-9:
                        piece += rf"^{{{b:g}}}"
                side.append(piece)
            for um, b in t.units:
                x = r"\,".join(sym(n) for n in um)
                piece = rf"\left(1 + {x}\right)"
                if abs(abs(b) - 1.0) >= 1e-9:
                    piece += rf"^{{{abs(b):g}}}"
                (num if b > 0 else den).append(piece)
            top = r"\,".join(num) or "1"
            if not den:
                return top
            bot = r"\,".join(den)
            return rf"\frac{{{top}}}{{{bot}}}"

        def body(b):
            if b.terms:
                inner = " + ".join((["1"] if b.unit else [])
                                   + [term(t) for t in b.terms])
                many = len(b.terms) > 1 or b.unit
            else:
                inner = " + ".join(sym(n) for n in b.members)
                many = len(b.members) > 1
            e = abs(b.exponent)
            if abs(e - 0.5) < 1e-9:
                return rf"\sqrt{{{inner}}}"
            if many:
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
            elif agg_tag:
                sym = rf"\htmlData{{agg={agg_tag}}}{{{sym}}}"
            (num if agg["exponent"] > 0 else den).insert(0, sym)
        elif self.verdict != "formula":
            num = ["?"] + num
        if self.constant != 1.0:
            c = self.constant
            if c > 1.0:
                num.insert(0, f"{c:g}")
            else:
                den.insert(0, f"{1.0 / c:g}")
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
        res = "".join(f"; {n} ({v:+.2f}) {why}"
                      for n, v, why in self.residue)
        if self.verdict == "formula":
            return (f"{self.text}   [prefactor {self.prefactor:.3f}, "
                    f"coverage {cov}]{res}")
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

        for c in sorted(free - {a, b}, key=lambda n: (-abs(gr[n]), n)):
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
    for n in sorted(free, key=lambda x: (-abs(gr[x]), x)):
        # judged only against couplings that can be TRUSTED, else a
        # feeble mover's inflated rho would entangle a strong one
        peak = max((abs(_rho_of(rho, n, m) or 0.0)
                    for m in strong if m != n), default=0.0)
        (singles if peak < edge else entangled).append(n)
    return blocks, singles, entangled


def _snap_member(e: float, tol: float):
    """The exponent a symbol wears INSIDE an addend. Looser than the
    factor snap, because an addend whose member also sits in a sum
    the grammar does not spell -- a cascode divide written as a plain
    ratio -- reads that member a little under one, and naming it at
    one with the shortfall reported is what a hand analysis does. A
    reading under 0.35 is not a member at all."""
    if abs(e) < 0.35:
        return None
    for cand in _NICE:
        if abs(e - cand) <= tol * max(abs(cand), 1.0):
            return cand
    return None


_WEAK_PAIR = 0.05        # a product pair's coupling may sit under the
#                          edge, but must clear this to count as nonzero


def _product_terms(names, rho, diag, gr, strong, edge: float,
                   term_tol: float):
    """Maximal groups of PRODUCT PARTNERS: symbols multiplied together
    inside one addend of a sum.

    The signature is that a pair's coupling and BOTH members' diagonals
    read the same value, (1 - sigma_t)/(a sigma_t); members of a plain
    sum never show it, since there the coupling and the diagonal carry
    opposite signs. Groups grow greedily from the strongest pair, and
    a candidate joins when every reading it brings -- its diagonal and
    its coupling to each member -- has the sign of the group's and lies
    within `term_tol` of the group's mean. That tolerance is looser
    than a block's on purpose: an addend whose members also sit in a
    sum the grammar does not spell still NAMES correctly, and the
    spread it leaves is reported as the reason the formula does not
    certify rather than used to deny the addend.
    Returns [(members, readings)], readings being every coupling and
    diagonal the group was read from."""
    free = set(names)
    out = []

    def same(x, ref):
        return x is not None and x * ref > 0 \
            and abs(x - ref) <= term_tol * abs(ref)

    pairs = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            r = _rho_of(rho, a, b)
            # an addend that DOMINATES its sum couples its own members
            # weakly: (1 - sigma)/sigma is 0.15 at a share of 0.87,
            # under the edge that keeps independent factors apart. Two
            # independent factors read zero on the coupling AND on both
            # diagonals; a dominant addend reads one small value on all
            # three, so the pair still announces itself under the edge
            # when that value is clearly nonzero (field case: the PMOS
            # cascode's gds13 gds14/(gm14 + gmbs14) carrying 0.87 of an
            # LDO's output conductance beside the NMOS cascode's 0.11,
            # left unread as G_eff)
            if r is None or abs(r) < _WEAK_PAIR \
                    or not (a in strong or b in strong):
                continue
            if same(diag[a], r) and same(diag[b], r):
                pairs.append((a, b))
    pairs.sort(key=lambda ab: -abs(gr[ab[0]] * gr[ab[1]]))
    for a, b in pairs:
        if a not in free or b not in free:
            continue
        grp = [a, b]
        reads = [_rho_of(rho, a, b), diag[a], diag[b]]
        for c in sorted(free - {a, b}, key=lambda n: (-abs(gr[n]), n)):
            ref = sum(reads) / len(reads)
            cross = [_rho_of(rho, c, m) for m in grp]
            got = [diag[c]] + cross
            if all(same(r, ref) for r in got):
                grp.append(c)
                reads.extend(got)
            elif all(same(r, ref) for r in cross) \
                    and diag[c] is not None and diag[c] * ref > 0 \
                    and abs(diag[c]) <= abs(ref):
                # the couplings say partner, the diagonal alone does
                # not: the diagonal is the one reading a member of an
                # inner sum distorts (a cascode's gm + gmbs under its
                # own divide), which the nested reading of the addend
                # then explains or reports. An inner sum REDUCES it;
                # a diagonal larger than the group's belongs to a
                # stronger sum of its own (a degeneration resistor's
                # 1 + gm R beside the pass device's gds/(gds + gm)),
                # and must not be swallowed here
                grp.append(c)
                reads.extend(cross)
        free -= set(grp)
        out.append((grp, reads))
    return out


def _monomial_blocks(terms, names, rho, diag, gr, cls, sgn, vals,
                     edge: float, rho_tol: float, agree_tol: float,
                     plain_tol: float, member_tol: float):
    """Group the addends into factors and read each factor.

    The units are the product terms plus every other symbol as a
    one-member addend. From the strongest term outward, a unit joins
    a factor when every coupling across -- each of its symbols against
    each symbol already in -- lies within `rho_tol` of the factor's
    running mean, the -1/a that addends of one sum share. The factor
    is then read the way a plain block is, twice over: a from that
    mean, snapped; each addend's share first from its own couplings,
    sigma_t = 1/(1 + a r_t), which names the member exponents
    e_i = g_i/(a sigma_t) at the nearest integer or half; then again
    from the shares, sigma_t = sum g_i / (a sum e_i), which is the
    reading the plain grammar takes and the one the exponents are
    finally measured against. The shares must sum to one and the
    addends must carry one dimension, or the group is not a sum; a
    symbol whose exponent lands on nothing is residue, named beside
    the expression. A reading that fails drops its worst-fitting unit
    and tries again, the seed excepted.
    Returns (blocks, problems, residue, consumed)."""
    blocks, problems, residue, consumed = [], [], [], set()
    in_term = {n for mem, _ in terms for n in mem}
    units = [("term", list(mem), list(reads)) for mem, reads in terms]
    units += [("sym", [n], [diag[n]]) for n in names if n not in in_term]
    strength = {id(u): sum(abs(gr[n]) for n in u[1]) for u in units}
    units.sort(key=lambda u: -strength[id(u)])

    def cross(u, members):
        return [_rho_of(rho, x, y) for x in u[1] for y in members]

    def consistent(rs, ref):
        if any(r is None or abs(r) < edge for r in rs):
            return False
        ref = sum(rs) / len(rs) if ref is None else ref
        return all(r * ref > 0 and abs(r - ref) <= rho_tol * abs(ref)
                   for r in rs)

    def readings_of(members):
        if len(members) == 1:
            return [diag[members[0]]]
        return ([_rho_of(rho, x, y) for i, x in enumerate(members)
                 for y in members[i + 1:]]
                + [diag[x] for x in members])

    def unit_sum(u):
        """A lone addend as the x of (1 + x)^a: a DIMENSIONLESS
        product (gm R, a loop gain) whose readings give (1 - s)/(a s)
        for the share s it carries beside the bare 1. Returns a Block
        or None."""
        mem = sorted(u[1], key=lambda n: -abs(gr[n]))
        ref_g = gr[mem[0]]
        es = {n: _snap_member(gr[n] / ref_g, member_tol) for n in mem}
        if any(e is None or e == 0.0 for e in es.values()):
            return None
        r_t = sum(u[2]) / len(u[2])
        best = None
        for a in (-1.0, 1.0, -2.0, 2.0, -0.5, 0.5):
            s = ref_g / a
            if not 0.0 < s < 1.0:
                continue
            pred = (1.0 - s) / (a * s)
            dev = abs(r_t - pred) / abs(pred)
            if dev <= rho_tol and (best is None or dev < best[0]):
                best = (dev, a, s)
        if best is None:
            return None
        dev, a, s = best
        t = Term(members=tuple(mem), exponents=es,
                 exponents_raw={n: gr[n] / (a * s) for n in mem},
                 share=s, share_from_rho=1.0 / (1.0 + a * r_t),
                 spread=max((abs(r - r_t) / abs(r_t) for r in u[2]),
                            default=0.0))
        if any(n not in cls for n in mem) or t.net(sgn, cls) != (0.0, 0.0):
            return None                 # a 1 beside a quantity with units
        # the readings of x beside a bare 1 are those of x beside
        # ANYTHING the grammar did not read, so the 1 is asserted only
        # where the values agree: x/(1 + x) at the operating point is
        # the share the couplings gave. Without that the x of a loop
        # gain the rest of the expression does not spell would print
        # as a degeneration
        tv = t.value(vals)
        sv = tv / (1.0 + tv)
        if abs(s - sv) > plain_tol:
            return None
        return Block(members=tuple(mem), exponent=a, exponent_raw=a,
                     exponent_from_rho=-1.0 / r_t if r_t else None,
                     shares={t.text(): s, "1": 1.0 - s},
                     plain=True, spread=t.spread,
                     terms=(t,), kind=None, unit=True)

    def read(group):
        """One factor read, as (terms, a, r_x, cross spread), or
        ("residue", symbol, exponent, unit), or (None, why)."""
        xs = [r for i, u in enumerate(group)
              for v in group[i + 1:] for r in cross(u, v[1])]
        if not xs:
            return None, "lone addend"
        r_x = sum(xs) / len(xs)
        a = _snap(-1.0 / r_x)
        if a is None:
            return None, f"across-addend coupling {r_x:+.2f} is not -1/a"
        prov = []
        for u in group:
            r_t = sum(u[2]) / len(u[2])
            s_rho = 1.0 / (1.0 + a * r_t)
            if not 0.0 < s_rho < 1.0:
                return None, f"{'*'.join(u[1])} carries no share"
            es_raw = {n: gr[n] / (a * s_rho) for n in u[1]}
            es = {n: _snap_member(es_raw[n], member_tol) for n in u[1]}
            s_sum = s_rho
            if all(e is not None for e in es.values()):
                # the second reading, from the shares: with the
                # exponents named, sum g over the addend is a sigma_t
                # sum e
                e_tot = sum(es.values())
                s_sum = (sum(gr[n] for n in u[1]) / (a * e_tot) if e_tot
                         else s_rho)      # a pure ratio: no second reading
                es_raw = {n: gr[n] / (a * s_sum) for n in u[1]}
                es = {n: _snap_member(es_raw[n], member_tol)
                      for n in u[1]}
            spread = max((abs(r - r_t) / abs(r_t) for r in u[2]),
                         default=0.0) if len(u[1]) > 1 else 0.0
            prov.append([u, es, es_raw, s_rho, s_sum, spread, None])
        # Every addend with partners is tried as a DIVIDE as well: its
        # share taken from the other addends first, since the shares
        # sum to one, then from its own two readings. A plain monomial
        # cannot pass the fit -- its residual gives every member a
        # share of zero or one -- so the divide is kept where it reads
        # at all when the monomial reading did not sit together, and
        # where it sits tighter than the monomial reading otherwise
        for p in prov:
            u, es, es_raw, s_rho, s_sum, spread, _nest = p
            if len(u[1]) < 2:
                continue
            off = any(e is None
                      or abs(es_raw[n] - e) > 0.12 * max(abs(e), 1.0)
                      for n, e in es.items())
            others = sum(q[4] for q in prov if q is not p
                         and all(e is not None for e in q[1].values()))
            cands = []
            for s in ((1.0 - others) if others > 0.0 else None,
                      s_sum, s_rho):
                if s is not None and 0.0 < s < 1.0 \
                        and all(abs(s - c) > 1e-9 for c in cands):
                    cands.append(s)
            got = _nested_divide(u[1], a, cands, rho, diag, gr, rho_tol,
                                 agree_tol)
            if got is None:
                continue
            _sc, b, taus, es2, worst, s, e_raw, back = got
            if spread <= rho_tol and not off and worst >= spread:
                continue
            p[1], p[2], p[3], p[4], p[5] = es2, e_raw, back, s, worst
            p[6] = (b, taus)
        out = []
        for u, es, es_raw, s_rho, s_sum, spread, nest in prov:
            for n in u[1]:
                if es[n] is None:
                    return ("residue", n, es_raw[n], u), None
            mem = tuple(sorted(u[1], key=lambda n: -abs(gr[n])))
            if nest:
                b, taus = nest
                inner = tuple(n for n in mem if taus[n] > 0.0)
                out.append(Term(members=mem, exponents=es,
                                exponents_raw=es_raw, share=s_sum,
                                share_from_rho=s_rho, spread=spread,
                                inner=inner, inner_exponent=b,
                                inner_shares={n: taus[n] for n in inner}))
            else:
                out.append(Term(members=mem, exponents=es,
                                exponents_raw=es_raw, share=s_sum,
                                share_from_rho=s_rho, spread=spread))
        tot = sum(t.share for t in out)
        if abs(tot - 1.0) > agree_tol:
            return None, f"shares sum to {tot:.2f}"
        if len({t.net(sgn, cls) for t in out}) > 1:
            return None, "the addends do not share a dimension"
        return (out, a, r_x, max(abs(r - r_x) / abs(r_x) for r in xs)), None

    for seed in units:
        if seed[0] != "term" or any(n in consumed for n in seed[1]):
            continue
        group = [seed]
        banned = []                 # dropped from this group: a unit
        #                             that joins by its couplings but
        #                             breaks the reading must not
        #                             rejoin on the retry
        while seed[1]:
            # grow: strongest first, each unit judged against every
            # symbol already in
            members = [n for u in group for n in u[1]]
            ref = None
            if len(group) > 1:
                xs = [r for i, u in enumerate(group)
                      for v in group[i + 1:] for r in cross(u, v[1])]
                ref = sum(xs) / len(xs)
            for u in units:
                if u in group or any(u is b for b in banned) \
                        or any(n in consumed for n in u[1]):
                    continue
                if consistent(cross(u, members), ref):
                    group.append(u)
                    members += u[1]
                    if ref is None:
                        xs = cross(u, [n for n in members
                                       if n not in u[1]])
                        ref = sum(xs) / len(xs)
            got, _why = read(group)
            if got is None:
                if len(group) <= 1:
                    # a product nothing else adds to: the x of a
                    # (1 + x)^a, when it is dimensionless and its
                    # readings say so
                    blk = unit_sum(seed) if len(seed[1]) > 1 else None
                    if blk is not None:
                        blocks.append(blk)
                        consumed |= set(blk.members)
                    break
                # drop the worst-fitting unit and retry, the seed
                # excepted: the group is read around it
                def dev(u, ref=ref):
                    rs = cross(u, [n for n in members if n not in u[1]])
                    return max(abs(r - ref) for r in rs if r is not None)

                worst = max(group[1:], key=dev)
                group.remove(worst)
                banned.append(worst)
                continue
            if got[0] == "residue":
                _, n, e_raw, home = got
                residue.append((
                    n, gr[n],
                    f"couples as a partner of {'*'.join(home[1])} at "
                    f"exponent {e_raw:+.2f}, which no monomial carries"))
                consumed.add(n)
                home[1].remove(n)
                if not home[1]:
                    group.remove(home)
                else:
                    home[2][:] = readings_of(home[1])
                continue
            terms_out, a, r_x, x_spread = got
            syms = [n for t in terms_out for n in t.members]
            tot = sum(t.value(vals) for t in terms_out)
            for t in terms_out:
                t.share_value = t.value(vals) / tot if tot else None
            plain = all(t.share_value is not None
                        and abs(t.share - t.share_value) <= plain_tol
                        for t in terms_out)
            net = terms_out[0].net(sgn, cls)
            kind = ("reactive" if net[0] and not net[1] else
                    "conductive" if net[1] and not net[0] else None)
            spread = max([x_spread] + [t.spread for t in terms_out])
            blocks.append(Block(
                members=tuple(sorted(syms, key=lambda n: -abs(gr[n]))),
                exponent=a, exponent_raw=-1.0 / r_x,
                exponent_from_rho=-1.0 / r_x,
                shares={t.text(): t.share for t in terms_out},
                plain=plain, spread=spread, terms=tuple(terms_out),
                kind=kind))
            consumed |= set(syms)
            # what keeps the reading from certifying is said per
            # addend: a spread where one value was expected means a
            # member also sits in a sum the addend does not spell, an
            # exponent off its integer says which member, and two
            # readings of the share that disagree say the addend's
            # weight is not what its couplings claim
            for t in terms_out:
                if t.spread > rho_tol and t.inner:
                    problems.append(
                        f"inside {t.text()} the divide reads with a "
                        f"spread of {t.spread:.2f} across its pairs")
                elif t.spread > rho_tol:
                    lo = min(t.members, key=lambda n: diag[n])
                    hi = max(t.members, key=lambda n: diag[n])
                    problems.append(
                        f"inside {t.text()} the couplings spread "
                        f"{diag[lo]:+.2f}..{diag[hi]:+.2f} where one "
                        f"value was expected: a member also sits in a "
                        f"sum this addend does not spell")
                off = [n for n in t.members
                       if abs(t.exponents_raw[n] - t.exponents[n])
                       > 0.12 * max(abs(t.exponents[n]), 1.0)]
                if off:
                    problems.append(
                        ", ".join(f"{n} reads at exponent "
                                  f"{t.exponents_raw[n]:+.2f}"
                                  for n in off)
                        + " where "
                        + "/".join(f"{t.exponents[n]:+g}" for n in off)
                        + " was written")
                if abs(t.share - t.share_from_rho) \
                        > agree_tol * max(t.share, 0.5):
                    problems.append(
                        f"{t.text()} carries {t.share:.2f} of its sum "
                        f"by the shares but {t.share_from_rho:.2f} by "
                        f"the couplings")
            break
    return blocks, problems, residue, consumed


def _fit_divide(members, g, hdiag, hpair, tol: float, agree_tol: float,
                allow_zero: bool = False):
    """Fit prod x_k**e_k (sum x_k)**a to a jet restricted to `members`:
    `g` the first derivatives, `hdiag` and `hpair` the second. For each
    nice a every member's diagonal gives its share as a root of
    sigma (1 - sigma) = H_kk/a; the exponent it wears alone follows as
    g_k - a sigma_k and must snap to an integer, a half or zero; the
    shares must sum to one and each pair must show -a sigma_k sigma_l
    within `tol`. With `allow_zero` a member may carry no share at all
    -- it then stands outside the sum, a plain factor of the addend --
    which is what the nested reading needs. Returns the best
    (score, a, shares, exponents, worst pair deviation) or None."""
    import itertools
    import math

    best = None
    for a in (-1.0, 1.0, -2.0, 2.0, -0.5, 0.5):
        cands = []
        for n in members:
            disc = 1.0 - 4.0 * hdiag[n] / a
            # a double root sits at disc = 0 and noise takes it a
            # little under; a diagonal that wants more than that is
            # not this a
            if disc < -0.05:
                break
            root = math.sqrt(max(disc, 0.0))
            opts = []
            for s in ((1.0 - root) / 2.0, (1.0 + root) / 2.0):
                if allow_zero and s < 0.01:
                    s = 0.0
                if (s <= 0.0 and not allow_zero) or s < 0.0 \
                        or s > 1.0 + agree_tol:
                    continue
                if s not in opts:
                    opts.append(s)
            if not opts:
                break
            cands.append(opts)
        else:
            for combo in itertools.product(*cands):
                if sum(1 for s in combo if s > 0.0) < 2:
                    continue
                if abs(sum(combo) - 1.0) > agree_tol:
                    continue
                es, raw = {}, {}
                for n, s in zip(members, combo):
                    e = g[n] - a * s
                    raw[n] = e
                    es[n] = 0.0 if abs(e) <= 0.12 else _snap(e)
                    if es[n] is None:
                        break
                else:
                    if not any(es.values()) and not allow_zero:
                        continue            # a plain sum: not a divide
                    worst = 0.0
                    for i, k in enumerate(members):
                        for j in range(i + 1, len(members)):
                            m = members[j]
                            pred = -a * combo[i] * combo[j]
                            act = hpair[k, m]
                            worst = max(worst, abs(act - pred)
                                        / max(abs(pred), 0.03))
                    if worst > tol:
                        continue
                    score = worst + sum(abs(raw[n] - es[n])
                                        for n in members)
                    if best is None or score < best[0]:
                        best = (score, a, dict(zip(members, combo)), es,
                                worst)
    return best


def _nested_divide(members, a, sigmas, rho, diag, gr, tol: float,
                   agree_tol: float):
    """Re-read one addend of a sum of monomials as a divide.

    With the addend's share sigma assumed, every member's effective
    exponent is eps_k = g_k/(a sigma), and what is left of the second
    derivatives once the monomial part a sigma (1 - sigma) eps_i eps_j
    is taken out, divided by a sigma, is the divide jet in the inner
    shares. Each sigma offered is tried; the fit must give back the
    share it was handed, re-read from the sum of g with the exponents
    named, within `agree_tol`. Returns the best (score, b, inner
    shares, outer exponents, worst pair deviation, sigma, raw outer
    exponents, sigma re-read) or None."""
    best = None
    for s in sigmas:
        if not 0.0 < s < 1.0:
            continue
        k = a * s
        eps = {n: gr[n] / k for n in members}
        hdiag, hpair = {}, {}
        for n in members:
            hii = diag[n] * gr[n] ** 2
            hdiag[n] = (hii - k * (1.0 - s) * eps[n] ** 2) / k
        for i, n in enumerate(members):
            for m in members[i + 1:]:
                r = _rho_of(rho, n, m)
                if r is None:
                    return None
                hij = r * gr[n] * gr[m]
                hpair[n, m] = hpair[m, n] = \
                    (hij - k * (1.0 - s) * eps[n] * eps[m]) / k
        got = _fit_divide(members, eps, hdiag, hpair, tol, agree_tol,
                          allow_zero=True)
        if got is None:
            continue
        score, b, taus, es, worst = got
        e_tot = sum(es.values()) + b
        back = sum(gr[n] for n in members) / (a * e_tot) if e_tot else s
        if abs(back - s) > agree_tol * max(s, 0.5):
            continue
        score += abs(back - s)
        if best is None or score < best[0]:
            best = (score, b, taus, es, worst, s,
                    {n: eps[n] - b * taus[n] for n in members}, back)
    return best


def _divide_groups(pool, rho, gr, H, homo, vals, edge: float, tol: float,
                   agree_tol: float):
    """Groups that read as prod x_k**e_k * (sum x_k)**a with at least
    one member standing in both -- the divide of the module docstring.

    The pool is what the plain partition left entangled. It is cut into
    components by coupling, and each component (smallest first among
    its members to drop, when the whole does not fit) is fitted to the
    signature: for each nice a, every member's diagonal gives its share
    as a root of sigma (1 - sigma) = H_kk/a, the exponent it wears
    alone follows as g_k - a sigma_k and must snap to an integer, a
    half or zero, the shares must sum to one and each pair must show
    H_kl = -a sigma_k sigma_l within `tol`. At least one exponent must
    be non-zero, else the group is a plain sum the partition would have
    taken. Returns [(members, a, shares, exponents, spread)]."""
    free = sorted(pool, key=lambda n: (-abs(gr[n]), n))

    def linked(a, b):
        r = _rho_of(rho, a, b)
        return r is not None and abs(r) >= edge \
            and homo.get(a) == homo.get(b) and homo.get(a)

    comps, seen = [], set()
    for n in free:
        if n in seen:
            continue
        comp, stack = [], [n]
        seen.add(n)
        while stack:
            x = stack.pop()
            comp.append(x)
            for y in free:
                if y not in seen and linked(x, y):
                    seen.add(y)
                    stack.append(y)
        if len(comp) > 1:
            comps.append(sorted(comp, key=lambda m: (-abs(gr[m]), m)))

    def fit(members):
        return _fit_divide(
            members, gr, {n: complex(H[n, n]).real for n in members},
            {(k, m): complex(H[k, m]).real for k in members for m in members},
            tol, agree_tol)

    out = []
    for comp in comps:
        got = fit(comp) if len(comp) <= 6 else None
        members = comp
        while got is None and len(members) > 2:
            # the weakest mover is the likeliest intruder
            members = members[:-1]
            got = fit(members)
        if got is None:
            continue
        _score, a, shares, es, spread = got
        out.append((tuple(members), a, shares, es, spread))
    return out


def _lump_cliques(names, rho, diag, gr, homo, strong, edge: float,
                  tol: float):
    """Groups of symbols the root depends on ONLY THROUGH THEIR SUM.

    If f depends on x_i and x_j through L = x_i + x_j alone, the chain
    rule fixes rho_ij = d_L - 1/g_L whatever f does with L, each
    member's own diagonal at d_i = rho_ij + 1/g_i, and every member's
    coupling to an OUTSIDE symbol at the lump's own, rho_io = rho_Lo,
    the same row for all of them. So a group whose couplings all read
    one value r, whose diagonals all read r + 1/g_i and whose rows
    agree is a sum the root sees as one quantity -- a node
    conductance, a node capacitance, a gm with its gmbs -- and
    folding it into one symbol is exact for this root. The rows are
    what tells it from two addends of a sum that happen to shift
    alike: those couple to the rest of the circuit differently. A plain
    power-of-sum factor shows the same signature with the extra
    property that -1/r is a nice exponent; it is folded too, and
    unfolded again once read, so nothing the plain grammar spelled
    changes. Members must be addable: one class and one sign."""
    free = set(names)
    out = []

    def shift(n):
        return diag[n] - 1.0 / gr[n]

    def same(x, ref):
        return x is not None and x * ref > 0 \
            and abs(x - ref) <= tol * abs(ref)

    def rows_agree(a, b):
        for o in strong:
            if o in (a, b):
                continue
            ra, rb = _rho_of(rho, a, o), _rho_of(rho, b, o)
            if ra is None or rb is None:
                return False
            if abs(ra) < edge and abs(rb) < edge:
                continue
            if not same(ra, rb):
                return False
        return True

    pairs = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            r = _rho_of(rho, a, b)
            if r is None or abs(r) < edge or homo.get(a) != homo.get(b) \
                    or not homo.get(a) or not (a in strong or b in strong):
                continue
            if same(shift(a), r) and same(shift(b), r) \
                    and rows_agree(a, b):
                pairs.append((a, b))
    pairs.sort(key=lambda ab: -abs(gr[ab[0]] * gr[ab[1]]))
    for a, b in pairs:
        if a not in free or b not in free:
            continue
        grp = [a, b]
        reads = [_rho_of(rho, a, b), shift(a), shift(b)]
        for c in sorted(free - {a, b}, key=lambda n: (-abs(gr[n]), n)):
            if homo.get(c) != homo.get(a):
                continue
            ref = sum(reads) / len(reads)
            got = [shift(c)] + [_rho_of(rho, c, m) for m in grp]
            if all(same(r, ref) for r in got) and rows_agree(c, a):
                grp.append(c)
                reads.extend(got)
        free -= set(grp)
        out.append(sorted(grp, key=lambda n: -abs(gr[n])))
    return out


def _lump_jet(g, H, vals, stamps, groups):
    """The jet with each group folded into one symbol, exactly: the
    chain rule through a sum is g_L = sum g_i, H_LL = sum_ij H_ij,
    H_Lj = sum_i H_ij, and the lump's value is the sum of the values.
    The lump is named by the sum it stands for, which is how it then
    prints, and stamped with its class so the coverage rules see it."""
    g, H, vals, stamps = dict(g), dict(H), dict(vals), dict(stamps)
    lumps = {}
    for mem in groups:
        name = "+".join(mem)
        kind = next(st[0] for st in stamps[mem[0]]
                    if st[0] in _CLASS)
        kind = {"cx": "c", "vccs": "g"}.get(kind, kind)
        others = [n for n in g if n not in mem]
        g[name] = sum(g[m] for m in mem)
        vals[name] = sum(vals[m] for m in mem)
        stamps[name] = [(kind, None, None, name)]
        for o in others:
            v = sum(H[m, o] for m in mem)
            H[name, o] = H[o, name] = v
        H[name, name] = sum(H[a, b] for a in mem for b in mem)
        for m in mem:
            del g[m], vals[m], stamps[m]
        for k in [k for k in H if k[0] in mem or k[1] in mem]:
            del H[k]
        lumps[name] = tuple(mem)
    return g, H, vals, stamps, lumps


def _unfold(block, lumps, g, H, vals, cls):
    """A plain block with a lump among its members is the plain sum
    the grammar always spelled -- a sum of a sum is the flat sum -- so
    it goes back to being one: the lump's members spliced in where it
    stood, shares and spread read from the raw jet, as before the
    fold. A lump inside a MONOMIAL stays: there it is a factor, and
    the sum it stands for is exactly what the addend multiplies."""
    if block.terms or not any(n in lumps for n in block.members):
        return block
    mem = tuple(m for n in block.members for m in lumps.get(n, (n,)))
    gr = {n: complex(g[n]).real for n in mem}
    a = block.exponent
    shares = {n: gr[n] / a for n in mem}
    tot = sum(abs(vals[n]) for n in mem)
    rs = [complex(H[x, y]).real / (gr[x] * gr[y])
          for i, x in enumerate(mem) for y in mem[i + 1:]]
    return Block(members=tuple(mem), exponent=a,
                 exponent_raw=block.exponent_raw,
                 exponent_from_rho=-len(rs) / sum(rs) if rs else None,
                 shares=shares,
                 plain=all(abs(shares[n] - abs(vals[n]) / tot) <= 0.10
                           for n in mem),
                 spread=(max(rs) - min(rs)) if rs else 0.0,
                 kind=cls.get(mem[0]))


def _strict_snap(e: float, tol: float = 0.12):
    """The nearest integer or half, within `tol` absolute, nonzero;
    None otherwise. The anchored reader's exponents are exact reads,
    not fits, so a miss by more than this is a fraction to explain."""
    h = round(e * 2.0) / 2.0
    return h if h != 0.0 and abs(e - h) <= tol else None


def _anchored_sum(names, rho, diag, gr, H, cls, sgn, homo, vals, *,
                  floor: float, trust: float, rho_tol: float,
                  agree_tol: float, plain_tol: float, member_tol: float):
    """A SUM OF PRODUCTS OF SUMS, read through its anchors.

    A pole of a loop is often a sum of time constants, each capacitor
    under its own product of conductances, resistances and loop-gain
    factors, over a common factor: 1/(R_G C_G + R' C1 (1 + K1) + R' C2
    (1 + K2)). A symbol that sits in one addend only -- each capacitor
    here -- ANCHORS that addend: its share is its own sensitivity over
    the sum's exponent, sigma_i = g_i/a, and its row of second
    derivatives gives every other symbol's exponent INSIDE that addend
    in one step, eps_ik = (H_ik + sigma_i g_k)/g_i, since the only way
    the anchor's share moves with x_k is through x_k's weight in its
    own addend against the sum. The anchors announce themselves: one
    class, one sign, shares that sum to one, every cross coupling at
    -1/a. A symbol whose exponent is the same in every addend is a
    factor of the whole, left to the plain grammar; one whose exponent
    differs is private to the addends it appears in, at the exponent
    each row says. Integer and half exponents name a monomial; a group
    of equal fractions modulo one on a dimensionless product is the x
    of a (1 + x)^b factor, each member with an integer of its own on
    top (gm R/(1 + gm R), the current divider at a degenerated mirror
    source, is gm and R at one under b = -1), asserted when x/(1 + x)
    at the operating point gives the fraction the sign of b implies;
    an addend may carry several. What remains fractional is printed
    as read and said in the problems. Returns (block, consumed, problems)
    or None when fewer than two anchors, or no private symbol, is
    found: then the plain grammar is the right reader."""
    best = None
    for kind in ("reactive", "conductive"):
        for sign in (+1, -1):
            cands = [n for n in names if cls.get(n) == kind
                     and sgn.get(n) == sign and n in homo
                     and abs(gr[n]) >= floor]
            if len(cands) < 2:
                continue
            for a in (-1.0, 1.0):
                shares = {n: gr[n] / a for n in cands}
                # a share under the trust floor anchors nothing: its
                # row is the anchor's small sensitivity divided out,
                # noise read as structure (a 2 % capacitor beside a
                # 98 % one made the plain (C2 + C1) a partial)
                anchors = [n for n in cands if shares[n] >= trust]
                if len(anchors) < 2:
                    continue
                # every pair at -1/a: the signature of addends of one sum
                ok = []
                for n in anchors:
                    good = True
                    for m in anchors:
                        if m == n:
                            continue
                        r = _rho_of(rho, n, m)
                        if r is None or abs(r - (-1.0 / a)) > rho_tol:
                            good = False
                            break
                    if good:
                        ok.append(n)
                if len(ok) < 2:
                    continue
                tot = sum(shares[n] for n in ok)
                if abs(tot - 1.0) > agree_tol:
                    continue
                score = abs(tot - 1.0)
                if best is None or score < best[0]:
                    best = (score, a, ok, {n: shares[n] for n in ok})
    if best is None:
        return None
    _s, a, anchors, shares = best
    others = [n for n in names if n not in anchors]
    eps = {}
    for i in anchors:
        for k in others:
            h = complex(H.get((i, k), H.get((k, i), 0.0))).real
            eps[i, k] = (h + shares[i] * gr[k]) / gr[i]
    # common factor: the same exponent in every addend (zero included)
    private = []
    for k in others:
        es = [eps[i, k] for i in anchors]
        if max(es) - min(es) > max(0.12, member_tol * max(abs(e) for e in es)):
            private.append(k)
    if not private:
        return None
    problems = []
    terms = []
    for i in anchors:
        # a private symbol belongs to the addends where it carries an
        # exponent worth trusting; 0.05 of a resistor in another
        # capacitor's time constant is the tail of a shared factor,
        # not a member
        mem = [i] + [k for k in private if abs(eps[i, k]) >= trust]
        raw = {i: 1.0}
        raw.update({k: eps[i, k] for k in mem[1:]})
        es = {i: 1.0}
        loose = []
        for k in mem[1:]:
            # a strict snap: 1.67 is 1 + a fraction, not 1.5, and the
            # fraction is what the (1 + x) reading below is for
            e = _strict_snap(raw[k])
            if e is None:
                loose.append(k)
            else:
                es[k] = e
        units = []
        if loose:
            # shared fractional parts on dimensionless products: the
            # members of one (1 + x)^b factor read at n_k + b tau, tau
            # = x/(1 + x) the factor's share and n_k an integer of the
            # member's own (R in R (1 + gm R) reads at 1 + tau; gm and
            # R in the current divider gm R/(1 + gm R) read at 1 - tau,
            # b = -1). What the members share is the fraction modulo
            # one, whichever integer each carries, so the loose symbols
            # group by it; a group's fraction phi is tau under b = +1
            # and 1 - tau under b = -1, and the values decide between
            # the two: x/(1 + x) at the operating point must give tau,
            # and every member's remainder must be an integer. Each
            # group that reads is one factor of the addend, and an
            # addend may carry several (C2 under both the PMOS loop's
            # (1 + gm R) and its own divider); a group that does not
            # read stays loose and is reported
            groups = []
            for k in sorted(loose, key=lambda n: -abs(gr[n])):
                phi = raw[k] - (raw[k] // 1.0)
                for grp in groups:
                    phi0 = grp[0][1]
                    d = abs(phi - phi0)
                    d = min(d, 1.0 - d)
                    if d <= 0.12 * max(min(phi0, 1.0 - phi0), 0.25):
                        grp.append((k, phi))
                        break
                else:
                    groups.append([(k, phi)])
            for grp in groups:
                ks = [k for k, _ in grp]
                if not all(cls.get(k) for k in ks):
                    continue
                net = {"reactive": 0.0, "conductive": 0.0}
                x = 1.0
                for k in ks:
                    net[cls[k]] += sgn[k]
                    x *= abs(vals[k])
                if net != {"reactive": 0.0, "conductive": 0.0}:
                    continue
                sv = x / (1.0 + x)
                phi = sum(f for _, f in grp) / len(grp)
                best = None
                for b in (1.0, -1.0):
                    tau = phi if b > 0 else 1.0 - phi
                    if abs(sv - tau) > plain_tol:
                        continue
                    ns = {}
                    for k in ks:
                        n = round(raw[k] - b * tau)
                        if abs(raw[k] - b * tau - n) > 0.12:
                            break
                        ns[k] = float(n)
                    else:
                        if best is None or abs(sv - tau) < best[0]:
                            best = (abs(sv - tau), b, ns)
                if best is None:
                    continue
                _d, b, ns = best
                units.append((tuple(sorted(ks, key=lambda n: -abs(gr[n]))), b))
                for k in ks:
                    loose.remove(k)
                    if ns[k]:
                        es[k] = ns[k]
                    else:
                        del raw[k]
                        mem.remove(k)
        for k in loose:
            es[k] = round(raw[k], 2)
            problems.append(f"inside {i}'s addend {k} reads at exponent "
                            f"{raw[k]:+.2f}, which no monomial carries")
        terms.append(Term(members=tuple(mem), exponents=es, exponents_raw=raw,
                          share=shares[i], share_from_rho=shares[i],
                          units=tuple(units)))
    # THE SELF-CHECK. The anchors' rows gave the exponents; the private
    # symbols' own couplings and diagonals were not consulted, and they
    # must agree with the structure read: for a sum of monomials
    # H_kl = a [sum_t s_t e_tk e_tl - (sum_t s_t e_tk)(sum_t s_t e_tl)]
    # plus, inside a (1 + x)^b factor, b tau (1 - tau) for two members
    # of its x. A pair that deviates is reported as a spread, so a jet
    # whose couplings do not match prints the structure as a partial
    # rather than a formula
    eff = {}
    for t in terms:
        i = t.members[0]
        e = {k: t.exponents[k] for k in t.members[1:]}
        facs = []
        for um, b in t.units:
            x = 1.0
            for k in um:
                x *= abs(vals[k])
            tau = x / (1.0 + x)
            for k in um:
                # on top of any integer the member carries alone
                e[k] = e.get(k, 0.0) + b * tau
            facs.append((set(um), b, tau))
        eff[i] = (e, facs)
    priv = sorted({k for t in terms for k in t.members[1:]}
                  | {k for t in terms for um, _b in t.units for k in um},
                  key=lambda n: -abs(gr[n]))
    worst = 0.0
    for x_i, k in enumerate(priv):
        for l in priv[x_i:]:
            pred = 0.0
            for i in anchors:
                e, facs = eff[i]
                pred += shares[i] * e.get(k, 0.0) * e.get(l, 0.0)
                for um, b, tau in facs:
                    if k in um and l in um:
                        pred += shares[i] * b * tau * (1.0 - tau)
            gk = sum(shares[i] * eff[i][0].get(k, 0.0) for i in anchors)
            gl = sum(shares[i] * eff[i][0].get(l, 0.0) for i in anchors)
            pred = a * (pred - gk * gl)
            act = complex(H.get((k, l), H.get((l, k), 0.0))).real
            dev = abs(act - pred) / max(abs(pred), 0.03)
            worst = max(worst, dev)
            if dev > rho_tol:
                problems.append(
                    f"inside the anchored sum the coupling of {k} and {l} "
                    f"reads {act:+.2f} where the structure predicts "
                    f"{pred:+.2f}: a spread this reading does not explain")
                break
        else:
            continue
        break
    # the addends must share a dimension, to the tolerance a reported
    # fraction leaves (0.78 of a resistor beside 0.77 of a gm nets to
    # -0.01, not to a different dimension)
    nets = {(round(t.net(sgn, cls)[0], 1), round(t.net(sgn, cls)[1], 1))
            for t in terms}
    if len(nets) > 1:
        problems.append("the addends do not share a dimension: "
                        + ", ".join(f"{t.members[0]} {t.net(sgn, cls)}"
                                    for t in terms))
    tot = sum(t.value(vals) for t in terms)
    for t in terms:
        t.share_value = t.value(vals) / tot if tot else None
    plain = all(t.share_value is not None
                and abs(t.share - t.share_value) <= plain_tol for t in terms)
    syms = tuple(sorted({n for t in terms for n in t.members}
                        | {n for t in terms for um, _b in t.units for n in um},
                        key=lambda n: -abs(gr[n])))
    net0 = next(iter(nets)) if len(nets) == 1 else None
    kind = (None if net0 is None else
            "reactive" if net0[0] and not net0[1] else
            "conductive" if net0[1] and not net0[0] else None)
    block = Block(members=syms, exponent=a, exponent_raw=a,
                  exponent_from_rho=a,
                  shares={t.text(): t.share for t in terms},
                  plain=plain, spread=worst, terms=tuple(terms), kind=kind)
    return block, set(syms), problems


def reconstruct(g, H, vals, stamps, target, *, two_pi: bool = True,
                floor: float = 0.02,
                trust: float = 0.08,
                edge: float = 0.30, rho_tol: float = 0.25,
                agree_tol: float = 0.20, plain_tol: float = 0.10,
                prefactor_band: float = 0.25,
                coverage_tol: float = 0.12,
                term_tol: float = 0.60,
                member_tol: float = 0.30,
                constant_tol: float = 0.10,
                lump: bool = False) -> Reconstruction:
    """Read the expression for a root from its jet.

    `g`, `H` come from secondorder.loop_pole_jet or loop_pair_jets;
    `vals` and `stamps` from marginledger._collect_stamps; `target` is
    the value of the reconstructed quantity, used only to report the
    prefactor. `two_pi` says whether that quantity is a FREQUENCY, so
    the 2 pi belongs in the rendering -- a damping ratio is
    dimensionless and carries none. Symbols moving the root by less
    than `floor` are left out of the formula and counted against its
    coverage instead. `term_tol` is how far the readings inside one
    addend of a sum of monomials may spread before a symbol is denied
    membership; `member_tol` how far a member's exponent may sit from
    the integer or half it is named at. Both are looser than the
    block tolerances, and anything inside them but outside the block
    tolerances is a reported reason, not a certified formula.
    `constant_tol` is how close a prefactor must sit to 2, 1/2, 4 or
    1/4 to be read as that number: tighter than `prefactor_band`,
    because a hand formula's 2 is exact and a ratio that merely drifts
    toward one half is a missing factor, not a constant.

    `lump` runs the pre-pass first: every group of symbols the root
    depends on only through their sum -- the signature is in
    _lump_cliques -- is folded into one symbol, exactly, and the
    reading is taken in that lumped space. That is what lets a
    cascode divide read as gds (G_node)/(gm + gmbs) without anyone
    naming the lumps by hand. A folded plain sum is unfolded again in
    the result, so nothing the plain grammar spelled changes; the
    lumps that stay appear under `lumps`, and print as the sums they
    stand for."""
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
    diag = {n: complex(H[n, n]).real / gr[n] ** 2 for n in names}
    if lump:
        groups = _lump_cliques(names, rho, diag, gr, homo, strong, edge,
                               rho_tol)
        if groups:
            g2, H2, v2, s2, lumps = _lump_jet(g, H, vals, stamps, groups)
            rec = reconstruct(
                g2, H2, v2, s2, target, two_pi=two_pi, floor=floor,
                trust=trust, edge=edge, rho_tol=rho_tol,
                agree_tol=agree_tol, plain_tol=plain_tol,
                prefactor_band=prefactor_band, coverage_tol=coverage_tol,
                term_tol=term_tol, member_tol=member_tol,
                constant_tol=constant_tol, lump=False)
            rec.blocks = tuple(_unfold(b, lumps, g, H, vals, cls)
                               for b in rec.blocks)
            if rec.aggregate:
                rec.aggregate["blocks"] = tuple(
                    _unfold(b, lumps, g, H, vals, cls)
                    for b in rec.aggregate["blocks"])
            named = {n for b in rec.blocks for n in b.members}
            named |= {n for n, _v, _w in rec.residue}
            rec.lumps = {k: v for k, v in lumps.items() if k in named}
            return rec
    # A SUM OF PRODUCTS OF SUMS first, through its anchors: the
    # capacitors of a sum of time constants, each in one addend, whose
    # rows give every other symbol's exponent inside that addend
    blocks, problems, residue, consumed = [], [], [], set()
    anchored = _anchored_sum(names, rho, diag, gr, H, cls, sgn, homo, vals,
                             floor=floor, trust=trust, rho_tol=rho_tol,
                             agree_tol=agree_tol, plain_tol=plain_tol,
                             member_tol=member_tol)
    if anchored is not None:
        blk, used, probs = anchored
        blocks.append(blk)
        consumed |= used
        problems += probs
    # SUMS OF MONOMIALS next: product partners announce themselves
    # through their diagonals, and an addend found that way takes its
    # symbols out of the plain partition that follows
    left = [n for n in names if n not in consumed]
    terms = _product_terms(left, rho, diag, gr, strong, edge, term_tol)
    if terms:
        b2, p2, r2, c2 = _monomial_blocks(
            terms, left, rho, diag, gr, cls, sgn, vals, edge, rho_tol,
            agree_tol, plain_tol, member_tol)
        blocks += b2
        problems += p2
        residue += r2
        consumed |= c2
    rest = [n for n in names if n not in consumed]
    raw, singles, entangled = _cliques(rest, rho, gr, homo, strong,
                                       edge, rho_tol, agree_tol)
    entangled += [n for n, _v, _w in residue]
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
            # A WEIGHTED ADDEND under the trust floor is not an addend
            # of this sum. A plain sum's shares ARE its values' shares,
            # so a member that weighs more among the values than it
            # moves the root carries a factor the grammar did not
            # read: the fast LDO's gate conductance at 20 uA seated
            # gds_M8, 1.6 % of the root, beside gds_M4yPa at 22 % of
            # their sum, where it is the NMOS cascode's divide
            # gds8 gds10/gm8' with every member under the trust floor.
            # Seated it cost the reading a quarter of its value; set
            # beside the sum as residue it costs the 1.6 % it moves
            # the root by, and the residue says what it is
            weak = [n for n in comp if abs(shares[n]) < trust
                    and abs(vals[n]) / tot - abs(shares[n]) > plain_tol]
            if weak and len(weak) < len(comp):
                for n in weak:
                    residue.append((
                        n, gr[n],
                        f"weighs {abs(vals[n]) / tot:.2f} of its sum with "
                        f"{'+'.join(m for m in comp if m not in weak)} "
                        f"but moves the root as {abs(shares[n]):.2f} of "
                        f"it: an addend with a factor this grammar did "
                        f"not read"))
                    entangled.append(n)
                comp = [n for n in comp if n not in weak]
                rs = [_rho_of(rho, x, y) for i, x in enumerate(comp)
                      for y in comp[i + 1:]]
                read = _read_exponent(comp, rs, gr, agree_tol)
                if read is None:
                    entangled += comp
                    continue
                a, a_rho, a_sum = read
                shares = {n: gr[n] / a for n in comp}
                tot = sum(abs(vals[n]) for n in comp)
            plain = all(abs(shares[n] - abs(vals[n]) / tot) <= plain_tol
                        for n in comp)
        blocks.append(Block(members=tuple(
            sorted(comp, key=lambda n: -abs(gr[n]))),
            exponent=a, exponent_raw=a_sum, exponent_from_rho=a_rho,
            shares=shares, plain=plain,
            spread=(max(rs) - min(rs)) if rs else 0.0,
            kind=cls.get(comp[0])))

    # DIVIDES last: what the plain partition refused because its two
    # exponent readings disagreed is exactly where a symbol stands in
    # the numerator and in a denominator sum at once. The sum becomes
    # a block and each overlapping member a lone block of its own, so
    # the renderer prints gm/(gm + gmK) as it would any other factor
    pool = [n for n in entangled if n not in consumed]
    for mem, a, shares, es, spread in _divide_groups(
            pool, rho, gr, H, homo, vals, edge, rho_tol, agree_tol):
        tot = sum(abs(vals[n]) for n in mem)
        blocks.append(Block(
            members=mem, exponent=a, exponent_raw=a,
            exponent_from_rho=None, shares=dict(shares),
            plain=all(abs(shares[n] - abs(vals[n]) / tot) <= plain_tol
                      for n in mem),
            spread=spread, kind=cls.get(mem[0]),
            divide=tuple(n for n in mem if es[n])))
        for n in mem:
            if es[n]:
                blocks.append(Block(members=(n,), exponent=es[n],
                                    exponent_raw=es[n],
                                    exponent_from_rho=None,
                                    shares={n: 1.0}, kind=cls.get(n)))
        entangled = [n for n in entangled if n not in mem]

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
    # a set: a divide names its overlapping member twice, once in the
    # sum and once alone, and it moves the root only once
    named = {n for b in blocks for n in b.members}
    coverage = class_sums(named)
    worst_cov = max(abs(coverage["reactive"] - t_re),
                    abs(coverage["conductive"] - t_co))

    import math

    constant = 1.0

    def render(bs, whole: bool):
        num = [b.text() for b in bs if b.exponent > 0]
        den = [b.text() for b in bs if b.exponent < 0]
        if constant > 1.0:
            num.insert(0, f"{constant:g}")
        elif constant < 1.0:
            den.insert(0, f"{1.0 / constant:g}")
        top = "*".join(num) or "1"
        if not whole:                       # an unread factor remains
            top = "?*" + top if num else "?"
        parts = (["2pi"] if two_pi else []) + den
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
            pref *= b.value(vals)
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
        # A BARE NUMBER is a factor too: zeta = 1/(2Q) is a formula
        # with a 2 in it, and a prefactor that lands on 2, 1/2, 4 or
        # 1/4 within the band has read it. The constant joins the
        # expression; the raw ratio stays reported as the prefactor
        near = [c for c in (2.0, 0.5, 4.0, 0.25)
                if abs(prefactor - c) <= constant_tol * c]
        # -- but only when nothing load-bearing is left unread. Two
        # movers of one class that cancel, gm at +0.5 and gmK at -0.5,
        # leave the coverage balanced and the ratio gm/(gm + gmK) in
        # the prefactor, and at equal shares that ratio IS one half: a
        # dropped quantity must not pass for a number
        in_blocks = {m for b in blocks for m in b.members}
        unread = sorted((n for n in entangled
                         if abs(gr[n]) >= trust and n not in in_blocks),
                        key=lambda n: -abs(gr[n]))
        # -- and only when the prefactor is the SOLE thing in the way:
        # a 2 written next to a spread complaint decorates a structure
        # that has not read, and on the raw folded-cascode jet it did
        if near and not unread and not problems:
            constant = near[0]
            named = named * constant if named is not None else None
        else:
            problems.append(
                f"the named structure reproduces {prefactor:.2f} of the "
                f"value, not 1: a factor is missing"
                + (" (" + ", ".join(f"{n} ({gr[n]:+.2f})"
                                    for n in unread[:3]) + " unread)"
                   if unread else ""))
    if worst_cov > coverage_tol:
        miss = sorted(entangled, key=lambda n: -abs(gr[n]))[:3]
        problems.insert(0, (
            f"the named blocks carry {coverage['reactive']:+.2f} / "
            f"{coverage['conductive']:+.2f} of the {t_re:+.0f} / "
            f"{t_co:+.0f} this quantity must show")
            + (", missing " + ", ".join(f"{n} ({gr[n]:+.2f})"
                                        for n in miss) if miss else ""))
    # a mover an addend could not seat is said beside the expression
    # whatever the verdict: left silent it would look absorbed
    res_why = [f"{n} ({v:+.2f}) {why}" for n, v, why in residue]
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
            kept = [b for b in blocks if b.kind == keep_c]
            if not kept or abs(t_other) < 1e-9:
                continue
            # once a sum of monomials is in play, a side whose blocks
            # already carry its class sum IS read: what keeps the
            # whole from certifying is said in the reason, and a G_eff
            # in place of a read side would hide the very structure
            # the designer asked for, or stand in for a lone CL
            if any(b.terms for b in blocks):
                # by the SYMBOLS' class, not the block's: an anchored
                # sum of time constants is a reactive block whose
                # addends carry conductances and resistances too
                # over the SET of names: a divide names its overlapping
                # member in the sum and alone, and it moves the root
                # only once
                cov_o = sum(sgn[n] * gr[n]
                            for n in {m for b in blocks for m in b.members}
                            if cls.get(n) == other_c)
                if abs(cov_o - t_other) <= coverage_tol:
                    continue
            cov_k = sum(sgn[n] * gr[n] for b in kept for n in b.members)
            if abs(cov_k - t_keep) > coverage_tol:
                continue
            prod = 1.0
            for b in kept:
                prod *= b.value(vals)
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
            if constant > 1.0:
                num.insert(0, f"{constant:g}")
            elif constant < 1.0:
                den.insert(0, f"{1.0 / constant:g}")
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
        # A partial names SOME of the structure, and that is worth
        # showing -- but only while it is still readable as the
        # quantity it claims to be. A frequency needs a factor on both
        # sides; where one side is entirely unread the "?" carries the
        # whole dimensional character, and printing it over 2pi asserts
        # a shape the visible content does not have. Field case: a
        # 248 MHz pole rendering as ? sqrt(gm_PM9 + gm_PM11) / 2pi,
        # which is not a frequency for any reading of "?". An aggregate
        # is exempt: G_eff/C_eff is a NAMED stand-in fixed by the
        # identity, not an unread remainder.
        sides = set()
        for b in blocks:
            # a mixed sum of monomials carries both dimensions
            sides |= {b.kind} if b.kind else {"reactive", "conductive"}
        if agg is None and two_pi and len(sides) < 2:
            return Reconstruction(verdict="shared", movers=movers,
                                  coverage=coverage, two_pi=two_pi,
                                  reason=(
                                      "only the "
                                      f"{(sides.pop() if sides else 'no')}"
                                      " side read, so no expression "
                                      "can carry the units: "
                                      + "; ".join(problems[:2])))
        return Reconstruction(verdict="partial", movers=movers,
                              blocks=tuple(blocks), coverage=coverage,
                              prefactor=prefactor, value=named,
                              two_pi=two_pi, aggregate=agg,
                              text=text, residue=tuple(residue),
                              constant=constant,
                              reason="; ".join(problems[:2] + res_why))
    return Reconstruction(verdict="formula", blocks=tuple(blocks),
                          prefactor=prefactor, value=named,
                          two_pi=two_pi,
                          coverage=coverage, movers=movers,
                          residue=tuple(residue),
                          reason="; ".join(res_why),
                          constant=constant,
                          text=render(blocks, whole=True))
