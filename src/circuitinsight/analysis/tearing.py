"""S-E: circuit tearing — cut discovery and exact split solves.

Splitting the circuit along the signal path turns one large hybrid solve
into two small ones: keep-set grids multiply in a monolithic solve but
ADD across the halves, and the composed answer arrives in the factored,
design-oriented cascade form. Two exact composition results are built
here (docs/interp-speedup-plan.md S-E(b)/S-E(c)):

* `split_tf` — open signal path, ONE-node cut v. The naive product
  H1*H2 ignores loading; the exact composition is the classical
  interstage form

      H  =  H1_open * Zin2 / (Zout1 + Zin2) * H2

  with H1_open/Zout1 solved on the input half ALONE and H2/Zin2 on the
  output half ALONE (exact by the substitution theorem for a one-node
  interface, bidirectional halves included).

* `split_loop_gain` — a feedback loop opened at its return-ratio probe
  is a CHAIN (the loop break IS a tear); when the probe's sense terminal
  is currentless (only controlled-source sense pins touch it — checked),
  the Tian loop gain is EXACTLY the chain transfer driven at the
  sense terminal, and the chain splits by `split_tf` composition.
  PROVEN symbolically on the nmc3 fixture against analysis.loopgain
  before this module existed (docs S-E(c)); the module test re-proves it
  through this code path. Probes whose sense node draws current
  (transistor gates with cgs, 2-node interfaces) need the Tian double
  injection / 2-node tear composition — future work, refused loudly.

Graph semantics for cut discovery (established in the S-E(b) scout):
zeroed V-sources are SHORTS (node classes contract, so supplies/bias
rails driven by ideal sources collapse into ground), zeroed I-sources
are OPENS, and every other primitive conservatively couples all its
terminals. Sub-circuit solves run through the ordinary build_mna +
solve_tf stack, so kept symbols, the mod-p backends, and the exact
probe self-checks all apply per half unchanged.
"""
from __future__ import annotations

import dataclasses
import itertools
from collections import defaultdict

import sympy as sp

from ..engine.mna import (MnaError, S, TransferFunction, build_mna,
                          sanitize, solve_tf, solve_tf_batch)
from ..engine.primitives import Primitive

__all__ = ["ComposedTF", "TearingError", "SplitAdvice", "ac_ground",
           "ac_ground_error", "advise_split", "find_cuts", "rank_cuts",
           "partition", "split_tf", "split_loop_gain"]

_DRV = "__tear_drv"
_ZPR = "__tear_zprobe"


class TearingError(MnaError):
    pass


# ------------------------------------------------------ small-signal graph
def ss_graph(primitives, ground, open_insts=(), keep_src=()):
    """Connectivity graph over contracted node classes. Returns (adj, find):
    adj maps class representatives to neighbour reps; find maps any node
    name to its class representative (ground -> '0')."""
    parent: dict = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            if rb == "0":
                ra, rb = rb, ra
            parent[rb] = ra

    for g in ground:
        union("0", g)
    for p in primitives:
        if (p.kind == "vsrc" and p.inst not in keep_src
                and p.inst not in open_insts):
            union(*p.nodes)                      # zeroed V source = short
    adj: dict = defaultdict(set)
    for p in primitives:
        if p.inst in open_insts or p.kind == "isrc":
            continue
        if p.kind == "vsrc" and p.inst not in keep_src:
            continue
        ns = {find(n) for n in p.nodes} - {"0"}
        ns = sorted(ns)
        for i in range(len(ns)):
            adj[ns[i]]
            for j in range(i + 1, len(ns)):
                adj[ns[i]].add(ns[j])
                adj[ns[j]].add(ns[i])
    return adj, find


def _separates(adj, cut: set, a: str, b: str) -> bool:
    if a in cut or b in cut:
        return False
    seen, todo = {a}, [a]
    while todo:
        v = todo.pop()
        for w in adj[v]:
            if w in cut or w in seen:
                continue
            if w == b:
                return False
            seen.add(w)
            todo.append(w)
    return True


def find_cuts(primitives, ground, a: str, b: str, open_insts=(),
              keep_src=(), max_size: int = 2) -> list[tuple[str, ...]]:
    """Vertex cuts (as class representatives) separating a from b in the
    small-signal graph, smallest first. a/b may be any node names."""
    adj, find = ss_graph(primitives, ground, open_insts, keep_src)
    ra, rb = find(a), find(b)
    # guard the seeds against the one failure mode that measured as
    # silently wrong: an INSTANCE name (e.g. the source 'Vin') seeds the
    # wrong component and every verdict degrades without an error. Nets
    # merely absent from the graph stay legal -- opened connections and
    # alias folding both produce that legitimately (the first, broader
    # version of this guard broke advise_split on a folded 'vout').
    insts = {p.inst for p in primitives}
    nets = {n for p in primitives for n in p.nodes}
    for name, r in ((a, ra), (b, rb)):
        if r not in adj and name in insts and name not in nets:
            raise TearingError(
                f"{name!r} is an instance name, not a net -- seeds must "
                f"be NET names (a source instance seeds the wrong "
                f"component and the cut ranking silently degrades; "
                f"advise_split resolves sources to their nets for you)")
    if ra == rb:
        raise TearingError(
            f"{a!r} and {b!r} are shorted together in the small-signal "
            f"graph; open the connecting source first (open_insts)")
    nodes = [n for n in adj if n not in (ra, rb)]
    out: list[tuple[str, ...]] = []
    for r in range(1, max_size + 1):
        for c in itertools.combinations(nodes, r):
            if any(set(prev) <= set(c) for prev in out):
                continue                          # supersets are redundant
            if _separates(adj, set(c), ra, rb):
                out.append(c)
    return out


def ac_ground(primitives, ground, nodes) -> list:
    """Return the primitives with `nodes` tied to ground -- the classic
    designer modelling step ("this bias rail is an AC ground"), and the
    lever that makes tearing work on transistor circuits.

    Bias distribution is what bridges every small vertex cut: the uA741's
    only exact 2-node cut leaves 15 primitives against 155, but declaring
    ONE bias node an AC ground turns it into 81 against 88 (balance 0.92).
    Structurally, an m-node cut with one node AC-grounded becomes an
    (m-1)-node cut -- and the interface size is the cost driver.

    This is an APPROXIMATION, so it is a separate, explicit step: it
    rewrites the circuit, and everything downstream is then exact FOR THE
    REWRITTEN CIRCUIT. Quantify what it costs with `ac_ground_error`
    before trusting the result (the error-controlled discipline of
    Daems/Gielen/Sansen, with the user-designated AC-ground terminals of
    Shi's stage-form reduction)."""
    gnd0 = ground[0] if ground else "0"
    tied = set(nodes)
    out = []
    for p in primitives:
        if tied & set(p.nodes):
            out.append(dataclasses.replace(
                p, nodes=tuple(gnd0 if n in tied else n for n in p.nodes)))
        else:
            out.append(p)
    return out


def _numeric_response(primitives, ground, inp: str, out: str, freqs,
                      alias=None):
    """v(out)/excitation(inp) at each frequency, by numeric MNA solves."""
    import numpy as np

    system = build_mna(primitives, ground, inp, alias)
    subs = {system.symbols[n]: sp.Float(v)
            for n, v in system.values.items() if n in system.symbols}
    A = system.A.xreplace(subs)
    free = set(A.free_symbols) - {S}
    if free:
        raise TearingError(
            f"ac_ground_error needs numeric values; missing {sorted(map(str, free))}")
    fn = sp.lambdify(S, A, "numpy")
    z = np.array(system.z.tolist(), dtype=complex).ravel()
    k = system.node_index[out]
    vals = []
    for f in np.atleast_1d(np.asarray(freqs, dtype=float)):
        M = np.asarray(fn(2j * np.pi * f), dtype=complex)
        try:
            vals.append(np.linalg.solve(M, z)[k])
        except np.linalg.LinAlgError:
            # tying this node to ground made the system degenerate (e.g. a
            # node already held by a source): report it as unusable rather
            # than crashing the advisory that called us
            vals.append(np.nan)
    return np.array(vals)


def ac_ground_error(primitives, ground, inp: str, out: str, nodes, freqs,
                    alias=None) -> dict:
    """What does declaring `nodes` AC grounds actually cost? Compares the
    exact and rewritten circuits numerically over `freqs` and returns
    {'worst_db', 'worst_deg', 'f_worst_db'} -- the budget check that must
    precede any use of `ac_ground` in a result the designer will trust."""
    import numpy as np

    ref = _numeric_response(primitives, ground, inp, out, freqs, alias)
    apx = _numeric_response(ac_ground(primitives, ground, nodes), ground,
                            inp, out, freqs, alias)
    fr = np.atleast_1d(np.asarray(freqs, dtype=float))
    with np.errstate(divide="ignore", invalid="ignore"):
        d_db = 20 * np.log10(np.abs(apx) / np.abs(ref))
    d_deg = np.degrees(np.angle(apx / ref))
    d_db = np.nan_to_num(d_db, nan=0.0, posinf=1e9, neginf=1e9)
    i = int(np.argmax(np.abs(d_db)))
    return {"worst_db": float(np.max(np.abs(d_db))),
            "worst_deg": float(np.max(np.abs(d_deg))),
            "f_worst_db": float(fr[i])}


def rank_cuts(primitives, ground, a: str, b: str, open_insts=(),
              keep_src=(), max_size: int = 2, keep=()) -> list[dict]:
    """Cuts ranked by whether tearing there actually PAYS.

    Measured lesson (uA741, 2026-07-28): a valid cut is not automatically
    a useful one. Tearing solves each half (1+m)*m times (clamp cofactors)
    instead of the whole circuit once, so the win must come from the
    halves being SMALLER and from keep-sets FACTORING (a monolithic grid
    is the product over all kept symbols; a split pays the sum of the
    per-side products). A lopsided cut -- the 741's only 2-node cut
    leaves 15 primitives on one side and 155 on the other -- multiplies
    the big side's cost by 6 and wins nothing: measured 87 s split vs
    24 s monolithic with numeric keeps.

    Each entry: cut, n_a/n_b (primitive counts), balance (0..1, 1 = even),
    keep_a/keep_b (kept symbols per side), grid_mono/grid_split (the
    interpolation grid sizes being traded), and `pays` -- the heuristic
    verdict: the keep-set grid must shrink by more than the (1+m)*m
    solve-count penalty."""
    cuts = find_cuts(primitives, ground, a, b, open_insts, keep_src,
                     max_size)
    out = []
    for cut in cuts:
        try:
            A, B = partition(primitives, ground, cut, a,
                             open_insts=open_insts, keep_src=keep_src)
        except TearingError:
            continue
        m = len(cut)
        na, nb = len(A), len(B)
        bal = min(na, nb) / max(na, nb) if max(na, nb) else 0.0
        ka = kb = None
        g_mono = g_split = None
        if keep:
            sysA = build_mna(A, ground, next(iter(
                p.inst for p in A if p.kind == "vsrc")) if any(
                p.kind == "vsrc" for p in A) else None)
            ka = _sub_keep(sysA, list(keep))
            kb = [k for k in keep if k not in ka]
            g_mono = 2 ** len(keep)          # order-of-magnitude proxy
            g_split = 2 ** len(ka) + 2 ** len(kb)
        pays = (bal >= 0.25 and (g_split is None
                                 or g_mono > (1 + m) * m * g_split))
        out.append({"cut": cut, "n_a": na, "n_b": nb, "balance": bal,
                    "keep_a": ka, "keep_b": kb, "grid_mono": g_mono,
                    "grid_split": g_split, "pays": pays})
    out.sort(key=lambda d: (not d["pays"], len(d["cut"]), -d["balance"]))
    return out


@dataclasses.dataclass
class SplitAdvice:
    """Whether tearing this circuit pays, and what would make it pay."""
    cuts: list                    # rank_cuts entries, best first
    ac_grounds: list              # {node, worst_db, worst_deg, cut, balance}
    keep: list

    def verdict(self) -> str:
        best = self.cuts[0] if self.cuts else None
        if best is None:
            head = "no cut of the requested size separates input from output"
        elif best["pays"]:
            head = (f"tear at {best['cut']}: sides {best['n_a']}/{best['n_b']}"
                    f" (balance {best['balance']:.2f})")
            if best["keep_a"] is not None:
                head += (f", keep splits {len(best['keep_a'])}/"
                         f"{len(best['keep_b'])}")
        else:
            head = (f"do not tear: best cut {best['cut']} is "
                    f"{'unbalanced' if best['balance'] < 0.25 else 'not worth'}"
                    f" ({best['n_a']}/{best['n_b']}"
                    + (f", keep-set too small: grid {best['grid_mono']} vs "
                       f"{best['grid_split']} split"
                       if best["grid_mono"] is not None else "") + ")")
        if not self.ac_grounds:
            return head
        g = self.ac_grounds[0]
        return (f"{head}. Declaring {g['node']!r} an AC ground "
                f"(costs {g['worst_db']:.2f} dB) would give cut {g['cut']} "
                f"at balance {g['balance']:.2f}")


def advise_split(primitives, ground, inp: str, out: str, *, keep=(),
                 open_insts=(), max_size: int = 2, freqs=None, alias=None,
                 err_budget_db: float = 0.5, max_candidates: int = 4
                 ) -> SplitAdvice:
    """Rank the cuts, and when none of them pays, look for an AC-ground
    designation that would fix it -- measuring what each costs.

    The advisory the tearing measurements earned: a valid cut is not a
    useful one (balance and keep-set size decide), and bias distribution
    is usually what ruins balance. Candidate AC grounds are proposed only
    when they IMPROVE the balance, and each is reported with the worst
    dB / degree deviation it introduces (nodes above `err_budget_db` are
    dropped -- on the uA741 that is exactly how a signal node costing
    252 dB is separated from a bias rail costing 0.02 dB)."""
    import numpy as np

    src = next((p for p in primitives if p.inst == inp), None)
    if src is None:
        raise TearingError(f"input source {inp!r} not found")
    gnd = set(ground) | {"0"}
    a_seed = next((n for n in src.nodes if n not in gnd), None)
    if a_seed is None:
        raise TearingError(f"input source {inp!r} has no non-ground node")

    keep = list(keep)
    ranked = rank_cuts(primitives, ground, a_seed, out, open_insts,
                       {inp}, max_size, keep)
    best_bal = ranked[0]["balance"] if ranked else 0.0
    if ranked and ranked[0]["pays"]:
        return SplitAdvice(cuts=ranked, ac_grounds=[], keep=keep)

    if freqs is None:
        freqs = np.geomspace(1.0, 1e10, 41)
    # candidates come from the AC-ground scan: it scores EVERY node exactly
    # (one inverse per frequency, Sherman-Morrison) and labels the mirror
    # references, instead of guessing from the graph and paying a full
    # re-solve per guess
    from .acground import scan_ac_grounds

    try:
        rep = scan_ac_grounds(primitives, ground, inp, out, freqs=freqs,
                              budget_db=err_budget_db, alias=alias)
    except Exception:
        return SplitAdvice(cuts=ranked, ac_grounds=[], keep=keep)

    out_g = []
    for c in rep.candidates:
        if not c.within_budget:
            continue
        g2 = tuple(ground) + (c.node,)
        try:
            r2 = rank_cuts(primitives, g2, a_seed, out, open_insts,
                           {inp}, max_size, keep)
        except TearingError:
            continue
        if not r2 or r2[0]["balance"] <= best_bal:
            continue                      # grounding it buys no balance
        e = r2[0]
        out_g.append({"node": c.node, "worst_db": c.worst_db,
                      "worst_deg": c.worst_deg, "cut": e["cut"],
                      "balance": e["balance"], "n_a": e["n_a"],
                      "n_b": e["n_b"], "pays": e["pays"], "kind": c.kind,
                      "controls": c.controls})
        if len(out_g) >= max_candidates:
            break
    out_g.sort(key=lambda d: (-d["balance"], d["worst_db"]))
    return SplitAdvice(cuts=ranked, ac_grounds=out_g, keep=keep)


# --------------------------------------------------------------- partition
def partition(primitives, ground, cut, a_seed: str, open_insts=(),
              keep_src=()):
    """Split the primitives at a cut (one node name, or a sequence of
    node names for multi-node cuts) into (side_A, side_B), side A being
    the component of `a_seed`. Interface-class shorts (0 V sources inside
    a cut class) go to BOTH sides so either half can reference any alias
    of a cut node; other interface-only elements (loading to ground, or
    elements BETWEEN two cut nodes) join side A."""
    cuts = (cut,) if isinstance(cut, str) else tuple(cut)
    adj, find = ss_graph(primitives, ground, open_insts, keep_src)
    rcs = {find(c) for c in cuts}
    ra = find(a_seed)
    if ra in rcs:
        raise TearingError(f"a_seed {a_seed!r} is inside the cut class")
    side_a = {ra}
    todo = [ra]
    while todo:
        v = todo.pop()
        for w in adj[v]:
            if w not in rcs and w not in side_a:
                side_a.add(w)
                todo.append(w)
    A: list = []
    B: list = []
    gnd = set(ground) | {"0"}
    deferred: list = []
    inst_side: dict = {}
    for p in primitives:
        if p.inst in open_insts:
            continue
        reps = {find(n) for n in p.nodes if n not in gnd} - {"0"}
        core = reps - rcs
        if not core:
            # interface-only element: a short inside a cut class serves
            # BOTH halves (pure topology); other loading elements are
            # assigned in a second pass, following their owner instance
            if p.kind == "vsrc":
                A.append(p)
                B.append(p)
            else:
                deferred.append(p)
            continue
        if core <= side_a:
            A.append(p)
            inst_side[p.inst] = "A"
        elif core & side_a:
            raise TearingError(
                f"{p.inst} spans both sides of the cut at {cuts!r}: "
                f"the cut is not a valid separator")
        else:
            B.append(p)
            inst_side[p.inst] = "B"
    for p in deferred:
        # an interface-only element (e.g. a kept gate capacitance of a
        # stage-2 device sitting AT the cut node) loads the interface
        # identically from either half; follow the owner instance so its
        # kept symbols stay in that instance's sub-solve, defaulting to A
        (B if inst_side.get(p.inst) == "B" else A).append(p)
    return A, B


# ------------------------------------------------------------ split solves
def _fingerprint(prims, ground) -> tuple:
    """Content identity of a sub-circuit: the latency-cache key component
    (Iordache's latency principle — re-solve only the half that changed)."""
    return (tuple(sorted((p.inst, p.param, p.kind, tuple(p.nodes), p.value)
                         for p in prims)), tuple(ground))


def _keep_key(keep) -> tuple | str:
    from ..keep import is_all

    if keep is None:
        return ()
    if is_all(keep):
        return "ALL"
    return tuple(keep)


def _cached(cache, key, compute):
    if cache is None:
        return compute()
    hit = cache.get(key)
    if hit is None:
        hit = compute()
        cache[key] = hit
    return hit


def _sub_keep(system, keep):
    """The caller's keep entries that match this half's symbols, using the
    SAME rule as hybrid_split (exact name or owning-instance suffix, raw
    or sanitized); a plain [] stays [] (fully numeric)."""
    from ..keep import is_all

    if keep is None or is_all(keep):
        return keep
    out = []
    for k in keep:
        fs = {k, sanitize(k)}
        if any(name in fs or any(name.endswith("_" + f) for f in fs)
               for name in system.symbols):
            out.append(k)
    return out


@dataclasses.dataclass
class ComposedTF(TransferFunction):
    """A composed (torn) transfer function, with the error its
    error-budgeted blocks actually cost -- measured, not assumed."""
    exact: TransferFunction | None = None
    achieved_mag_err_db: float = 0.0
    achieved_phase_err_deg: float = 0.0
    blocks_pruned: int = 0
    blocks_total: int = 0

    def report(self) -> str:
        if self.exact is None:
            return "exact composition (no budget applied)"
        return (f"budgeted composition: {self.blocks_pruned}/"
                f"{self.blocks_total} blocks simplified, worst "
                f"{self.achieved_mag_err_db:.3g} dB / "
                f"{self.achieved_phase_err_deg:.3g} deg vs the exact "
                f"composition")


def _budget_blocks(blocks, budget_db, budget_deg, fmin, fmax):
    """Simplify each interface block under a share of the error budget
    (Guerra's per-level error-controlled hierarchical analysis: the error
    is controllable on block-sized expressions, not on the long chain
    that composing them produces). Blocks that cannot be simplified
    inside their share pass through untouched."""
    from .simplify import simplify_tf

    share_db = budget_db / max(1, len(blocks))
    share_deg = budget_deg / max(1, len(blocks))
    out, pruned = [], 0
    for tf in blocks:
        try:
            s = simplify_tf(tf, mag_tol_db=share_db,
                            phase_tol_deg=share_deg, fmin=fmin, fmax=fmax)
        except Exception:
            out.append(tf)
            continue
        if sp.count_ops(s.expr) < sp.count_ops(tf.expr):
            pruned += 1
            out.append(TransferFunction(expr=s.expr, values=tf.values,
                                        symbols=tf.symbols))
        else:
            out.append(tf)
    return out, pruned


def _clear_rows(M, rhs):
    """Scale each row of (M | rhs) to POLYNOMIAL entries. Rational-function
    elimination explodes in expression size (measured on the Tian route);
    a per-row scale multiplies every Cramer determinant by the same factor,
    so it cancels in every solution component."""
    n = len(M)
    Mp = sp.zeros(n, n)
    r = sp.zeros(n, 1)
    for i in range(n):
        frs = [sp.fraction(sp.cancel(sp.together(M[i][j]))) for j in range(n)]
        lcm = sp.Integer(1)
        for _, d in frs:
            lcm = sp.lcm(lcm, d)
        for j, (nu, de) in enumerate(frs):
            Mp[i, j] = sp.expand(nu * sp.cancel(lcm / de))
        r[i] = sp.expand(sp.together(rhs[i]) * lcm)
    return Mp, r


def _interface_solve(M, rhs):
    """Solve the m x m interface system M v = rhs exactly. m = 1 and 2 use
    the closed forms; m >= 3 clears rows to polynomials and applies Cramer
    with Berkowitz determinants (division-free, so no rational blow-up)."""
    m = len(M)
    if m == 1:
        return [sp.together(rhs[0] / M[0][0])]
    if m == 2:
        det = sp.together(M[0][0] * M[1][1] - M[0][1] * M[1][0])
        return [sp.together((M[1][1] * rhs[0] - M[0][1] * rhs[1]) / det),
                sp.together((-M[1][0] * rhs[0] + M[0][0] * rhs[1]) / det)]
    Mp, r = _clear_rows(M, rhs)
    det = Mp.det(method="berkowitz")
    if det == 0:
        raise TearingError("singular interface system: the cut does not "
                           "separate the halves")
    out = []
    for j in range(m):
        Mk = Mp.copy()
        Mk[:, j] = r
        out.append(sp.together(Mk.det(method="berkowitz") / det))
    return out


def split_tf(primitives, ground, inp: str, out: str, cut,
             keep=(), method: str = "auto", open_insts=(),
             progress=None, cache: dict | None = None,
             budget_db: float | None = None, budget_deg: float = 5.0,
             fmin: float = 10.0, fmax: float = 1e10) -> TransferFunction:
    """Exact transfer function through a cut (one node name, or a pair of
    names): solve the two halves independently and compose.

    Both cut sizes run the same Norton machinery: each half is clamped at
    the cut nodes by 0 V sources and characterized by clamp BRANCH
    currents — side A by its short-circuit current vector L_A (input
    driving) and admittance block Y_A, side B by Y_B and the output
    couplings H_B. Reconnection is KCL at the interface,

        (Y_A + Y_B) v = -L_A ,      H = H_B . v .

    Per side, all m+m² (or m²+m) quantities are cofactors over ONE
    matrix (Tan-Shi suppression), solved by a single solve_tf_batch call
    that walks the interpolation grid once and shares the denominator.
    Any uniform branch-current sign convention cancels between Y and L."""
    cuts = (cut,) if isinstance(cut, str) else tuple(cut)
    if not cuts:
        raise TearingError("empty cut")

    src = next((p for p in primitives if p.inst == inp), None)
    if src is None:
        raise TearingError(f"input source {inp!r} not found")
    gnd = set(ground) | {"0"}
    a_seed = next((n for n in src.nodes if n not in gnd), None)
    if a_seed is None:
        raise TearingError(f"input source {inp!r} has no non-ground node")

    A, B = partition(primitives, ground, cuts, a_seed,
                     open_insts=open_insts, keep_src={inp})
    _, find = ss_graph(primitives, ground, open_insts, keep_src={inp})
    if find(out) in {find(n) for p in A for n in p.nodes}:
        raise TearingError(
            f"output {out!r} is on the input side of the cut {cuts!r}")

    gnd0 = ground[0] if ground else "0"
    m = len(cuts)
    S = [Primitive(inst=f"__tear_s{i+1}", param="", kind="vsrc",
                   nodes=(c, gnd0)) for i, c in enumerate(cuts)]
    s_names = [s.inst for s in S]

    def side_systems(prims, drives):
        return {d: build_mna(list(prims) + S, ground, d) for d in drives}

    def branch_tasks(systems, drives, outputs):
        tasks = []
        for d in drives:
            system = systems[d]
            for o in outputs:
                k = (system.branch_index[o] if o in system.branch_index
                     else o)
                tasks.append((system, k))
        return tasks

    # side A: short-circuit currents under the input + admittance block
    sysA = side_systems(A, [inp] + s_names)
    keep_a = _sub_keep(sysA[inp], keep)
    resA = _cached(
        cache, ("A", _fingerprint(A, ground), cuts, inp,
                _keep_key(keep_a), method),
        lambda: solve_tf_batch(
            branch_tasks(sysA, [inp] + s_names, s_names),
            keep=keep_a, method=method, progress=progress))
    L_A = resA[:m]
    Y_A = [resA[m + j * m: m + (j + 1) * m] for j in range(m)]

    # side B: admittance block + output couplings
    sysB = side_systems(B, s_names)
    keep_b = _sub_keep(sysB[s_names[0]], keep)
    resB = _cached(
        cache, ("B", _fingerprint(B, ground), cuts, out,
                _keep_key(keep_b), method),
        lambda: solve_tf_batch(
            branch_tasks(sysB, s_names, s_names)
            + [(sysB[d], out) for d in s_names],
            keep=keep_b, method=method, progress=progress))
    Y_B = [resB[j * m:(j + 1) * m] for j in range(m)]
    H_B = resB[m * m:]

    # KCL at the interface; Y[j][i] = current at clamp i when clamp j
    # drives, so the KCL matrix is the transpose-indexed sum
    M = [[sp.together(Y_A[j][i].expr + Y_B[j][i].expr) for j in range(m)]
         for i in range(m)]
    def compose(LA, YA, YB, HB):
        Mx = [[sp.together(YA[j][i].expr + YB[j][i].expr) for j in range(m)]
              for i in range(m)]
        vx = _interface_solve(Mx, [-LA[i].expr for i in range(m)])
        return sum(HB[j].expr * vx[j] for j in range(m))

    values: dict = {}
    symbols: dict = {}
    flat = L_A + [x for row in Y_A for x in row]         + [x for row in Y_B for x in row] + list(H_B)
    for tf in flat:
        values.update(tf.values)
        symbols.update(tf.symbols)

    expr = compose(L_A, Y_A, Y_B, H_B)
    if budget_db is None:
        return TransferFunction(expr=expr, values=values, symbols=symbols)

    # error-budgeted composition (S-E, Guerra + Kolka): shrink the blocks
    # first, compose them, then MEASURE what the assumption cost
    import numpy as np

    small, pruned = _budget_blocks(flat, budget_db, budget_deg, fmin, fmax)
    n_l, n_y = m, m * m
    LA2 = small[:n_l]
    YA2 = [small[n_l + j * m: n_l + (j + 1) * m] for j in range(m)]
    off = n_l + n_y
    YB2 = [small[off + j * m: off + (j + 1) * m] for j in range(m)]
    HB2 = small[off + n_y:]
    expr_b = compose(LA2, YA2, YB2, HB2)

    exact = TransferFunction(expr=expr, values=values, symbols=symbols)
    approx = TransferFunction(expr=expr_b, values=values, symbols=symbols)
    freqs = np.geomspace(fmin, fmax, 60)
    h0 = exact.numeric(freqs)
    h1 = approx.numeric(freqs)
    good = np.abs(h0) > np.max(np.abs(h0)) * 1e-9
    with np.errstate(divide="ignore", invalid="ignore"):
        d_db = 20 * np.log10(np.abs(h1[good]) / np.abs(h0[good]))
    d_deg = np.degrees(np.angle(h1[good] / h0[good]))
    return ComposedTF(expr=expr_b, values=values, symbols=symbols,
                      exact=exact,
                      achieved_mag_err_db=float(np.max(np.abs(d_db))),
                      achieved_phase_err_deg=float(np.max(np.abs(d_deg))),
                      blocks_pruned=pruned, blocks_total=len(flat))


def _sense_is_currentless(primitives, node: str, probe: str) -> bool:
    """True when, apart from the probe branch itself, only controlled-source
    SENSE terminals (vccs/cx control pins) touch `node`."""
    for p in primitives:
        if p.inst == probe or node not in p.nodes:
            continue
        if p.kind in ("vccs", "cx"):
            drive, sense = p.nodes[:2], p.nodes[2:]
            if node in drive:
                return False
            if node in sense:
                continue
        else:
            return False
    return True


def split_loop_gain(primitives, ground, probe: str, cut: str,
                    keep=(), method: str = "auto",
                    progress=None,
                    cache: dict | None = None) -> TransferFunction:
    """Tian loop gain composed through a cut of the OPENED loop.

    Two exact routes, chosen by probe topology:

    * sense terminal currentless (only controlled-source sense pins touch
      it — the behavioral-fixture case): T is the chain transfer from a
      unit drive at the sense node to the far probe terminal, split by
      the split_tf composition (factored, low-entropy output);
    * sense terminal draws current (transistor benches — gates carry
      capacitance): the full Tian double-injection composition. The ring
      is torn at the cut AND at the probe branch; each side is reduced to
      its pure admittance block (the loop input is nulled, so no source
      vectors); both interfaces plus the probe element are reconnected in
      one small bordered system whose two excitations (the probe's series
      unit voltage E1 and the unit current injection E2) yield the four
      Eq. 30 readouts (i_b, v_p / v_n under each), combined exactly as in
      analysis.loopgain."""
    prb = next((p for p in primitives if p.inst == probe), None)
    if prb is None or prb.kind != "vsrc":
        raise TearingError(f"probe {probe!r} is not a vsrc branch")
    p_node, n_node = prb.nodes
    if _sense_is_currentless(primitives, n_node, probe):
        gnd0 = ground[0] if ground else "0"
        opened = [p for p in primitives if p.inst != probe]
        drv = Primitive(inst=_DRV, param="", kind="vsrc",
                        nodes=(n_node, gnd0))
        T = split_tf(opened + [drv], ground, _DRV, p_node, cut,
                     keep=keep, method=method, progress=progress,
                     cache=cache)
        # Spectre stb sign convention (validated against loop_gain on
        # nmc3): with the probe branch (p, n) opened, a unit drive at the
        # n-side sense terminal returns +T at the p-side terminal.
        return TransferFunction(expr=T.expr, values=T.values,
                                symbols=T.symbols)
    return _split_loop_gain_tian(primitives, ground, probe, cut,
                                 keep=keep, method=method,
                                 progress=progress, cache=cache)


def _y_block(prims, ground, terminals, keep, method, progress,
             cache: dict | None = None):
    """Pure admittance block of a sub-circuit at `terminals` (all internal
    sources zeroed): clamp every terminal, drive each clamp in turn, read
    every clamp's branch current — one solve_tf_batch over one matrix.
    Returns (Y, keep_used) with Y[j][i] = current at clamp i when clamp j
    drives (the raw measurement orientation; callers index accordingly)."""
    gnd0 = ground[0] if ground else "0"
    S = [Primitive(inst=f"__tear_y{i+1}", param="", kind="vsrc",
                   nodes=(c, gnd0)) for i, c in enumerate(terminals)]
    systems = {s.inst: build_mna(list(prims) + S, ground, s.inst)
               for s in S}
    keep_used = _sub_keep(systems[S[0].inst], keep)
    tasks = []
    for sj in S:
        system = systems[sj.inst]
        for si in S:
            tasks.append((system, system.branch_index[si.inst]))
    res = _cached(
        cache, ("Y", _fingerprint(prims, ground), tuple(terminals),
                _keep_key(keep_used), method),
        lambda: solve_tf_batch(tasks, keep=keep_used, method=method,
                               progress=progress))
    m = len(S)
    return [res[j * m:(j + 1) * m] for j in range(m)], keep_used


def _split_loop_gain_tian(primitives, ground, probe, cut, keep=(),
                          method="auto", progress=None,
                          cache: dict | None = None) -> TransferFunction:
    """The double-injection composition (see split_loop_gain)."""
    prb = next(p for p in primitives if p.inst == probe)
    p_node, n_node = prb.nodes
    cuts = (cut,) if isinstance(cut, str) else tuple(cut)
    m = len(cuts)
    if p_node in cuts or n_node in cuts:
        raise TearingError("probe terminals may not be cut nodes")

    opened = [p for p in primitives if p.inst != probe]
    A, B = partition(opened, ground, cuts, n_node)
    _, find = ss_graph(opened, ground)
    b_nodes = {find(n) for p in B for n in p.nodes}
    if find(p_node) not in b_nodes:
        raise TearingError(
            f"probe terminal {p_node!r} is not on the far side of the "
            f"cut {cuts!r}")

    # side A terminals: (n, cuts...); side B: (cuts..., p)
    Y_A, _ = _y_block(A, ground, (n_node,) + cuts, keep, method, progress,
                      cache=cache)
    Y_B, _ = _y_block(B, ground, cuts + (p_node,), keep, method, progress,
                      cache=cache)

    # unknowns x = [v_n, v_c1..v_cm, v_p, i_b]; J^X_c = sum_d Y^X[d][c] v_d
    # (Y[j][i] = readout at clamp i under drive j). KCL rows:
    #   cuts:  J^A_c + J^B_c = 0
    #   n:     J^A_n - i_b   = 0        (branch current enters n)
    #   p:     J^B_p + i_b   = inj_p    (branch current leaves p)
    #   probe: v_p - v_n     = e
    # Any uniform clamp sign convention flips i_b's sign coherently in the
    # n/p rows and cancels in the Eq. 30 combination checked by the gates.
    n_unk = m + 3
    M = sp.zeros(n_unk, n_unk)
    iv_n, iv_p, iv_b = 0, m + 1, m + 2

    def add_side(Y, term_ix):
        # term_ix: unknown indices of this side's terminals, in clamp order
        for i, row_unk in enumerate(term_ix):
            for j, col_unk in enumerate(term_ix):
                M[row_unk, col_unk] += sp.together(Y[j][i].expr)

    add_side(Y_A, [iv_n] + list(range(1, m + 1)))
    add_side(Y_B, list(range(1, m + 1)) + [iv_p])
    M[iv_n, iv_b] = -1
    M[iv_p, iv_b] = 1
    # probe constraint row (reuses the last row index = iv_b's equation slot)
    C = sp.zeros(1, n_unk)
    C[0, iv_p], C[0, iv_n] = 1, -1
    M2 = M.copy()
    M2[n_unk - 1, :] = C                # rows 0..m+1 are the KCL equations
    rhs_e1 = sp.zeros(n_unk, 1)
    rhs_e1[n_unk - 1, 0] = 1            # E1: series unit voltage
    rhs_e2 = sp.zeros(n_unk, 1)
    rhs_e2[iv_p, 0] = 1                 # E2: unit current into p
    sol1 = M2.LUsolve(rhs_e1)
    sol2 = M2.LUsolve(rhs_e2)

    Bv, Dv = -sol1[iv_b], sol1[iv_p]
    Ai, Ci = -sol2[iv_b], sol2[iv_n]
    expr = -(2 * (Ai * Dv - Bv * Ci) - Ai + Dv) / \
        (2 * (Bv * Ci - Ai * Dv) + Ai - Dv + 1)
    if sp.count_ops(expr) <= 20000:
        expr = sp.cancel(sp.together(expr))

    values: dict = {}
    symbols: dict = {}
    for row in Y_A + Y_B:
        for tf in row:
            values.update(tf.values)
            symbols.update(tf.symbols)
    return TransferFunction(expr=expr, values=values, symbols=symbols)
