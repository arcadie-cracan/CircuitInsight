"""L-A: the margin ledger -- exact per-singularity attribution of PM/GM.

For a rational loop gain T(s), log-magnitude and phase are ADDITIVE over
its poles and zeros: at the worst unity crossing f_u every singularity
owns a closed-form share of the phase, and at the worst critical-phase
crossing f_180 a share of the gain in dB. The ledger evaluates those
shares and certifies them: the terms SUM to the directly evaluated
arg T(j 2 pi f_u) and |T(j 2 pi f_180)| -- the decomposition is exact,
only the presentation truncates (fold(), which gathers the small terms
into one remainder with a declared budget, in the approximation-contract
manner).

Anchoring: each share is the CHANGE from DC, so the DC term (a "gain"
entry, arg T(0) = +-180 in the stb convention and 20 log |T(0)|) plus
the shares reconstructs the response. Phase shares are the CONTINUOUS
rotation of (j 2 pi f - x) from f = 0 -- for an LHP root the naive
end-minus-start angle already is continuous; an RHP root's vertical
sweep path passes LEFT of the origin, and when f crosses Im x the atan2
branch banks a spurious full turn, restored in closed form. Conjugate
pairs merge into one entry labeled (f0, zeta); the exactness lives in
the actual members, the label is derived.

Crossings follow sensitivity.loop_margins -- the ONE margin extraction
(worst crossing, folded phase) -- on a wide grid of the ledger's own
rational reconstruction, then each crossing is refined by bisection so
the reported PM/GM is the reconstruction's own machine-precision value.
All magnitude arithmetic is carried in log space: a 36th-order product
of |j f - x| factors can leave float64 while its logarithm cannot.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _fold(x):
    """Angular distance form: fold to (-180, 180]."""
    return -(((-np.asarray(x) + 180.0) % 360.0) - 180.0)


def _start_angle(x):
    """angle(-x), continuous with the upward sweep. A REAL RHP root
    puts -x on the branch cut itself with a negative-zero imaginary
    part, and np.angle then answers -pi for a path that climbs off the
    cut at +pi -- a spurious full turn in that root's share (caught by
    the per-entry pins, invisible to the mod-360 certificate)."""
    if x.imag == 0.0 and x.real > 0:
        return np.pi
    return float(np.angle(-x))


def _dtheta_deg(x, f):
    """Continuous rotation of (j t - x) as t sweeps 0 -> f, in degrees.
    The moving point climbs a vertical line at real part -Re x; only an
    RHP root (line left of the origin) whose Im x lies inside the sweep
    crosses the atan2 branch cut, costing one restored turn."""
    f = np.asarray(f, dtype=float)
    d = np.angle(1j * f - x) - _start_angle(x)
    if x.real > 0:
        d = d - 2.0 * np.pi * ((x.imag > 0.0) & (x.imag < f))
    return np.degrees(d)


def _abs_db(x, f):
    """20 log10 |j f - x|, overflow-safe (no products are formed)."""
    f = np.asarray(f, dtype=float)
    return 10.0 * np.log10(x.real ** 2 + (f - x.imag) ** 2)


def _climb_db(x, f):
    """20 log10 |j f - x| - 20 log10 |x|: the root's climb from DC."""
    return _abs_db(x, f) - 10.0 * np.log10(x.real ** 2 + x.imag ** 2)


def _group(roots):
    """Real roots and conjugate pairs, each with its ACTUAL members.
    An unpaired complex root (a rooting artifact, or a genuinely
    lone root of a complex-coefficient system) stays a single member
    -- the certificate keeps everyone honest either way."""
    roots = sorted((complex(r) for r in roots),
                   key=lambda x: (abs(x), x.imag))
    out, used = [], [False] * len(roots)
    for i, x in enumerate(roots):
        if used[i]:
            continue
        used[i] = True
        if abs(x.imag) <= 1e-9 * max(abs(x), 1e-300):
            out.append([complex(x.real, 0.0)])
            continue
        best, bd = None, np.inf
        for j in range(i + 1, len(roots)):
            if not used[j]:
                d = abs(roots[j] - x.conjugate())
                if d < bd:
                    best, bd = j, d
        if best is not None and bd <= 1e-6 * max(abs(x), 1e-300):
            used[best] = True
            out.append([x, roots[best]])
        else:
            out.append([x])
    return out


@dataclass
class LedgerEntry:
    """One term of the decomposition. phase_deg is the share of
    arg T at f_u (negative = lag spent), mag_db the share of
    20 log|T| at f_180; either is None when that crossing is absent.
    node/share stay empty until the circuit mapping (L-B) fills them."""
    kind: str                    # "gain" | "pole" | "zero" | "rest"
    f0_hz: float
    zeta: float | None
    rhp: bool
    pair: bool
    phase_deg: float | None
    mag_db: float | None
    count: int = 1
    node: str = ""
    share: float = 0.0
    roots: tuple = ()            # the actual member value(s), complex Hz
    spacing: float = 0.0         # doublet: relative pole/zero distance
    f0b_hz: float = 0.0          # doublet: the zero side's frequency
    roots_b: tuple = ()          # doublet: the zero side's members
    root_sens: tuple = ()        # L-G: per-element root
                                 # sensitivities (semi-relative,
                                 # x ds/dx -- Truxal/Kuo)
    taus: tuple = ()             # L-H: top capacitor time constants
    members: tuple = ()          # rest: the folded entries themselves,
                                 # so the remainder reconciles against
                                 # the Expression's factored form
    shared: tuple = ()           # (symbol, df0) movers when NO formula
                                 # certifies: the root is distributed,
                                 # which is the insight itself
    tau_residual: float | None = None   # |sum tau + 1/s|/|1/s| (Euler)
    reconstruction: object = None       # L-H: the jet reading itself
    #                                     (analysis.reconstruct), of
    #                                     which `estimate` is the
    #                                     display projection
    estimate: dict | None = None        # L-H: named formula + its error
    zeta_estimate: dict | None = None   # pairs: zeta = 2 pi f0 C/G named

    def describe(self) -> str:
        # the RHP side rides on EVERY kind that can carry it. A folded
        # row that drops the marker reads as ordinary dynamics, and a
        # RHP pole paired with a near RHP zero is exactly the case the
        # stability literature warns is easy to miss
        side = "RHP " if self.rhp else ""
        if self.kind == "gain":
            what = "T0"
        elif self.kind == "rest":
            what = f"{side}{self.count} smaller terms"
        elif self.kind == "band":
            what = (f"{side}band {self.f0_hz / 1e6:.3g}-"
                    f"{self.f0b_hz / 1e6:.3g} MHz ({self.count} terms)")
        elif self.kind == "doublet":
            z = f" zeta={self.zeta:.2f}" if self.zeta is not None else ""
            if self.spacing > 1e-4:
                what = (f"{side}doublet {self.f0_hz / 1e6:.3g}/"
                        f"{self.f0b_hz / 1e6:.3g} MHz{z} "
                        f"({self.spacing:.1%} apart)")
            else:
                what = (f"{side}cancelling doublet "
                        f"{self.f0_hz / 1e6:.3g} MHz{z}")
        else:
            z = f" zeta={self.zeta:.2f}" if self.zeta is not None else ""
            side = "RHP " if self.rhp else ""
            what = (f"{side}{self.kind}"
                    f"{' pair' if self.pair else ''} "
                    f"{self.f0_hz / 1e6:.3g} MHz{z}")
        at = f" ({self.node}, {self.share:.0%})" if self.node else ""
        ph = (f"{self.phase_deg:+.1f} deg"
              if self.phase_deg is not None else "--")
        mg = f"{self.mag_db:+.1f} dB" if self.mag_db is not None else "--"
        return f"{what}{at}: {ph} at f_u, {mg} at f_180"


@dataclass
class MarginLedger:
    """The exact decomposition at the worst crossings, certified."""
    pm_deg: float | None
    f_unity_hz: float | None
    gm_db: float | None
    f_180_hz: float | None
    entries: list                     # gain anchor first, then by |share|
    phase_residual_deg: float | None  # |sum - direct|, the certificate
    mag_residual_db: float | None
    stability: dict | None = None     # L-J: the closed-loop verdict and
    #                                   why the margins may mislead; see
    #                                   stability_certificate

    def fold(self, budget_deg: float = 2.0, budget_db: float = 0.5,
             pair_tol: float = 0.3):
        """The entries worth reading. Two presentation moves over the
        EXACT entries (which stay untouched): PHYSICAL near-doublets --
        a pole and zero of the same structure within `pair_tol`
        relative root distance -- display as ONE line carrying the
        couple's net share and its spacing (the fc field case: dozens
        of +-130..170 deg shares netting a few degrees each; the
        spacing is the doublet-robustness indicator); then the tail
        whose TOTAL absolute share stays inside both budgets gathers
        into one "rest" entry. The budgets and the pairing are the
        declared approximation; the sums are preserved exactly."""
        anchor = self.entries[:1]
        keep = (_pair_near_doublets(self.entries[1:], pair_tol)
                if pair_tol else list(self.entries[1:]))
        keep.sort(key=lambda e: -(abs(e.phase_deg)
                                  if e.phase_deg is not None
                                  else abs(e.mag_db)
                                  if e.mag_db is not None else 0.0))
        tail: list = []
        while keep:
            e = keep[-1]
            t = tail + [e]
            ph = sum(abs(x.phase_deg) for x in t
                     if x.phase_deg is not None)
            mg = sum(abs(x.mag_db) for x in t if x.mag_db is not None)
            if ph > budget_deg or mg > budget_db:
                break
            tail.append(keep.pop())
        if not tail:
            return anchor + keep
        ph = (sum(x.phase_deg for x in tail if x.phase_deg is not None)
              if any(x.phase_deg is not None for x in tail) else None)
        mg = (sum(x.mag_db for x in tail if x.mag_db is not None)
              if any(x.mag_db is not None for x in tail) else None)
        return anchor + keep + [LedgerEntry(
            # an RHP member must not lose its side by being gathered:
            # the remainder is a PRESENTATION fold, and a row that
            # silently drops the marker contradicts the advisor
            kind="rest", f0_hz=0.0, zeta=None,
            rhp=any(x.rhp for x in tail), pair=False,
            phase_deg=ph, mag_db=mg, count=sum(x.count for x in tail),
            # WHO was folded: without the enumeration the remainder
            # row could not be reconciled against the Expression's
            # factored form, and the two views read as mismatched
            # root lists (field report)
            members=tuple(tail))]

    def describe(self) -> list[str]:
        head = []
        if self.pm_deg is not None:
            head.append(f"PM {self.pm_deg:.1f} deg at "
                        f"{self.f_unity_hz / 1e6:.3g} MHz")
        if self.gm_db is not None:
            head.append(f"GM {self.gm_db:.1f} dB at "
                        f"{self.f_180_hz / 1e6:.3g} MHz")
        lines = [" / ".join(head) if head else "no crossing in band"]
        if self.stability:
            lines.append(f"{self.stability['verdict']}: "
                         f"{self.stability['reason']}")
        return lines + [e.describe() for e in self.fold()]


def merge_doublets(entries, *, tol: float = 1e-5):
    """Merge COINCIDENT pole/zero couples into net-zero "doublet"
    entries. The Tian combine lifts T by det(A)^2: by Desnanot-Jacobi
    the denominator is det^2 + (N_A - N_D) det + 2 det M_2, so det(A)
    divides numerator and denominator alike and every closed-loop mode
    rides along as an ALGEBRAIC ghost -- a pole and zero at the same
    spot whose shares cancel identically at any operating point. The
    merge is root-distance-gated at `tol` RELATIVE, tight enough that a
    PHYSICAL near-doublet (a mirror pole shadowed by a transmission
    zero, percent-level spacing -- the mfm fragility case) stays two
    visible lines."""
    poles = [e for e in entries if e.kind == "pole" and e.roots]
    zeros = [e for e in entries if e.kind == "zero" and e.roots]
    pz = {id(e) for e in poles} | {id(e) for e in zeros}
    out = [e for e in entries if id(e) not in pz]
    used = set()
    for p in poles:
        mate = None
        for z in zeros:
            if id(z) in used or len(z.roots) != len(p.roots):
                continue
            d = max(abs(a - b) / max(abs(a), 1e-300)
                    for a, b in zip(p.roots, z.roots))
            if d <= tol:
                mate = z
                break
        if mate is None:
            out.append(p)
            continue
        used.add(id(mate))
        out.append(LedgerEntry(
            kind="doublet", f0_hz=p.f0_hz, zeta=p.zeta, rhp=p.rhp,
            pair=p.pair,
            phase_deg=(None if p.phase_deg is None
                       else p.phase_deg + mate.phase_deg),
            mag_db=(None if p.mag_db is None
                    else p.mag_db + mate.mag_db),
            count=p.count + mate.count, roots=p.roots))
    out.extend(z for z in zeros if id(z) not in used)
    return out


def _root_distance(pr, zr) -> float:
    """Relative distance between two same-structure root groups,
    conjugate members matched by imaginary sign."""
    ps = sorted(pr, key=lambda x: x.imag)
    zs = sorted(zr, key=lambda x: x.imag)
    return max(abs(a - b) / max(abs(a), 1e-300)
               for a, b in zip(ps, zs))


def _pair_near_doublets(entries, tol: float):
    """Pair each pole entry with a geometrically near zero entry of
    the same structure (same member count, root distance <= tol,
    nearest couples first) into a display doublet carrying the NET
    share, the spacing, and the pole side's modal owner. Presentation
    only -- fold() calls this on a COPY of the entries."""
    poles = [e for e in entries if e.kind == "pole" and e.roots]
    zeros = [e for e in entries if e.kind == "zero" and e.roots]
    cands = []
    for p in poles:
        for z in zeros:
            if len(z.roots) != len(p.roots):
                continue
            d = _root_distance(p.roots, z.roots)
            if d <= tol:
                cands.append((d, p, z))
    cands.sort(key=lambda t: t[0])
    used: set = set()
    pairs = []
    for d, p, z in cands:
        if id(p) in used or id(z) in used:
            continue
        used.update((id(p), id(z)))
        pairs.append((d, p, z))
    out = [e for e in entries if id(e) not in used]
    for d, p, z in pairs:
        out.append(LedgerEntry(
            kind="doublet", f0_hz=p.f0_hz, zeta=p.zeta, rhp=p.rhp,
            pair=p.pair,
            phase_deg=(None if p.phase_deg is None
                       else p.phase_deg + z.phase_deg),
            mag_db=(None if p.mag_db is None
                    else p.mag_db + z.mag_db),
            count=p.count + z.count, node=p.node, share=p.share,
            roots=p.roots, spacing=float(d), f0b_hz=z.f0_hz,
            roots_b=z.roots, root_sens=p.root_sens))
    return out


def attribute_ledger(led: MarginLedger, system, *,
                     tol: float = 1e-3) -> MarginLedger:
    """L-B: name the node that OWNS each mode among the entries. Only
    an entry whose root IS a natural frequency of the system -- det(A)
    root -- has a modal identity that participation
    (compensate.pole_participation) can localize; on a Tian loop gain
    those are exactly the ghost doublets (see merge_doublets), so a
    cancelling couple at least says WHICH closed-loop mode it is. The
    remaining true loop-gain poles are roots of the minor combination
    2 M_2 + (N_A - N_D) + det -- provably not the modes of any single
    network -- and stay unmapped BY DESIGN; naming them would invent
    physics. Zeros belong to the injection and read ports and stay
    unmapped for the same reason."""
    from .compensate import pole_participation

    cands = [e for e in led.entries
             if e.kind in ("pole", "doublet") and e.roots]
    if not cands:
        return led
    parts = pole_participation(system, n_poles=10 ** 6)
    for e in cands:
        r = e.roots[0]
        best, bd = None, np.inf
        for p_hz, ranked in parts:
            d = abs(p_hz - r) / max(abs(r), 1e-300)
            if d < bd:
                bd, best = d, ranked
        if best is not None and bd <= tol:
            e.node, e.share = best[0][0], float(best[0][1])
    return led


def compare_ledgers(ledgers, labels, *, ref_index: int = 0,
                    pair_tol: float = 0.3, match_span: float = 10.0,
                    cluster_gap: float = 2.0) -> dict:
    """The per-corner comparison: the REFERENCE corner's folded rows
    become the row set, and every corner's exact shares aggregate onto
    them by LOG-FREQUENCY BAND -- each row owns the band nearest its
    term (a Voronoi partition over the row centers), kind-agnostic.
    Individual singularities are not identifiable corner to corner on
    a dense bench (a pair splits to two real poles at zeta 1, couples
    re-pair, huge near-cancelling shares would straddle rows and read
    as +-280 deg of fake drift); frequency bands are stable, and a
    cancelling couple travels together, so the nets stay physical.
    An entry farther than `match_span` (a frequency ratio) from every
    row center lands in "other", computed as the exact remainder, so
    every column still sums to its own corner's total. The anchor row
    is fold-normalized to the reference so the +-180 boundary never
    reads as drift. `delta` (max - min over corners) ranks the rows:
    the band whose drift explains the margin's.

    Returns {"labels", "ref_index", "rows": [{"term": LedgerEntry,
    "cells", "f0s", "delta"}] (anchor first, then by drift), "other",
    "pm", "fu", "gm"}."""
    n = len(ledgers)
    ref_rows = ledgers[ref_index].fold(pair_tol=pair_tol)
    rows = [{"term": e, "cells": [None] * n, "f0s": [None] * n,
             "delta": None}
            for e in ref_rows if e.kind != "rest"]
    anchor_row = rows[0]

    def _center(e):
        if e.kind == "doublet" and e.f0b_hz:
            return float(np.sqrt(e.f0_hz * e.f0b_hz))
        return max(e.f0_hz, 1e-300)

    # crowded reference rows chain-merge into BANDS: where centers sit
    # closer than cluster_gap no per-root partition can keep a couple
    # together, so the cluster's stable NET is the only honest cell --
    # a well-separated bench keeps its full granularity
    ordered = sorted(rows[1:], key=lambda r: _center(r["term"]))
    groups: list[list] = []
    for row in ordered:
        c = _center(row["term"])
        if groups and c < cluster_gap * _center(groups[-1][-1]["term"]):
            groups[-1].append(row)
        else:
            groups.append([row])
    term_rows = []
    for g in groups:
        if len(g) == 1:
            term_rows.append(g[0])
            continue
        lo = _center(g[0]["term"])
        hi = _center(g[-1]["term"])
        term_rows.append({"term": LedgerEntry(
            kind="band", f0_hz=lo, zeta=None,
            rhp=any(x["term"].rhp for x in g), pair=False,
            phase_deg=None, mag_db=None, count=len(g), f0b_hz=hi),
            "cells": [None] * n, "f0s": [None] * n, "delta": None})
    spans = []
    for row in term_rows:
        e = row["term"]
        lo = np.log(_center(e) if e.kind != "band" else e.f0_hz)
        hi = np.log(_center(e) if e.kind != "band" else e.f0b_hz)
        spans.append((lo, hi))
    reach = np.log(match_span)
    other = [0.0] * n
    ref_anchor = ledgers[ref_index].entries[0].phase_deg
    for k, led in enumerate(ledgers):
        a = led.entries[0].phase_deg
        if a is not None and ref_anchor is not None:
            anchor_row["cells"][k] = float(
                ref_anchor + _fold(a - ref_anchor))
        exact = [e for e in led.entries[1:]
                 if e.phase_deg is not None]
        total = sum(e.phase_deg for e in exact)
        matched = 0.0
        for e in exact:
            lf = np.log(max(e.f0_hz, 1e-300))
            ds = [max(0.0, lo - lf, lf - hi) for lo, hi in spans]
            i = int(np.argmin(ds))
            if ds[i] > reach:
                continue
            row = term_rows[i]
            c = row["cells"][k]
            row["cells"][k] = (0.0 if c is None else c) + e.phase_deg
            if row["f0s"][k] is None:
                row["f0s"][k] = e.f0_hz or None
            matched += e.phase_deg
        other[k] = total - matched
    rows = [anchor_row] + term_rows
    for row in rows:
        vals = [c for c in row["cells"] if c is not None]
        row["delta"] = (max(vals) - min(vals)) if len(vals) > 1 else 0.0
    term_rows.sort(key=lambda r: -(r["delta"] or 0.0))
    return {"labels": list(labels), "ref_index": ref_index,
            "rows": [anchor_row] + term_rows, "other": other,
            "pm": [x.pm_deg for x in ledgers],
            "fu": [x.f_unity_hz for x in ledgers],
            "gm": [x.gm_db for x in ledgers]}


def _collect_stamps(system, primitives, alias):
    """Per-SYMBOL rank-one stamp patterns (kind, a, b), numeric values
    and metadata -- every parameter kind enters A as k(theta, s) a b^T,
    and matched devices share a symbol whose stamps sum. Shared by the
    element ledger (L-C) and the singularity root sensitivities (L-G). Returns
    (stamps, vals, meta); meta[name] = {"insts", "param", "explicit"},
    where explicit marks a param-less netlist component (R, C, L) --
    the OP-INVARIANT knobs, as opposed to device model parameters
    that track bias and geometry."""
    from ..engine.mna import symbol_name

    dim = system.A.rows
    nidx = system.node_index

    def _inc(na, nb):
        v = np.zeros(dim)
        i = nidx.get(na)
        j = nidx.get(nb)
        if i is not None:
            v[i] += 1.0
        if j is not None:
            v[j] -= 1.0
        return v

    stamps: dict[str, list] = {}
    vals: dict[str, float] = {}
    meta: dict[str, dict] = {}
    for p in primitives:
        if p.kind not in ("r", "g", "c", "vccs", "cx", "l", "vcvs"):
            continue
        name = symbol_name(p, alias)
        val = system.values.get(name)
        if val is None or val == 0:
            continue
        m = meta.setdefault(name, {"insts": set(), "param": p.param})
        m["insts"].add(p.inst)
        if p.kind in ("r", "g", "c"):
            a = _inc(p.nodes[0], p.nodes[1])
            b = a
        elif p.kind in ("vccs", "cx"):
            a = _inc(p.nodes[0], p.nodes[1])       # output pair, rows
            b = _inc(p.nodes[2], p.nodes[3])       # control pair, cols
        else:                                      # l, vcvs: branch row
            a = np.zeros(dim)
            a[system.branch_index[p.inst]] = 1.0
            b = a if p.kind == "l" else _inc(p.nodes[2], p.nodes[3])
        stamps.setdefault(name, []).append((p.kind, a, b, p.inst))
        vals[name] = val
    for m in meta.values():
        m["insts"] = tuple(sorted(m["insts"]))
        m["explicit"] = m["param"] == ""
    return stamps, vals, meta


def _kfac(kind, s, val):
    """dY/dtheta of one stamp at complex s."""
    if kind in ("c", "cx"):
        return s
    if kind in ("g", "vccs"):
        return 1.0
    if kind == "r":
        return -1.0 / (val * val)
    if kind == "l":
        return -s
    return -1.0                                    # vcvs


def _combine_dT(B, D, A_, C, dB, dD, dA, dC):
    """dT through the Tian combine: Dn = N + 1, so dT = dN/(N+1)^2."""
    Dn = 2 * (B * C - A_ * D) + A_ - D + 1
    dN = (1 - 2 * D) * dA + 2 * C * dB + 2 * B * dC - (2 * A_ + 1) * dD
    return dN / (Dn * Dn)


def _dT_by_symbol(stamps, vals, s, Minv, xv, xi, ib, ip) -> dict:
    """dT/dtheta per symbol from one solve set at complex s."""
    B = -xv[ib]
    D = xv[ip]
    A_ = -xi[ib]
    C = xi[ip]
    out = {}
    for name, ss in stamps.items():
        dxv_ib = dxv_ip = dxi_ib = dxi_ip = 0j
        for kind, a, b, _inst in ss:
            k = _kfac(kind, s, vals[name])
            u = Minv @ a
            cv = k * (b @ xv)
            ci = k * (b @ xi)
            dxv_ib -= cv * u[ib]
            dxv_ip -= cv * u[ip]
            dxi_ib -= ci * u[ib]
            dxi_ip -= ci * u[ip]
        out[name] = _combine_dT(B, D, A_, C, -dxv_ib, dxv_ip,
                                -dxi_ib, dxi_ip)
    return out


def _tian_context(system, probe: str):
    """A(s) evaluator, its s-derivative Q, and the probe rows -- the
    arbitrary-COMPLEX-s twin of the LoopGainUpdater constructor, for
    evaluating the Tian reads AT a root rather than on the axis."""
    import sympy as sp

    from ..engine.mna import S
    from .compensate import _numeric_A

    A_v = _numeric_A(system)
    fn = sp.lambdify(S, A_v, "numpy")
    Q = np.array(A_v.diff(S).tolist(), dtype=complex)
    ib = system.branch_index[probe]
    ip = None
    for i in range(len(system.node_index)):
        if system.A[i, ib] == 1:
            ip = i
            break
    return fn, Q, ib, ip


def _respell_sym(sym: str, hot: dict) -> str:
    """Spell a formula symbol on the device the ROOT acts through.
    A matched group's shared symbol is named after ONE member
    (keep-name); this analysis investigates physical origin, so the
    hot member's name replaces the representative's (field decision:
    the interchangeable-symbol algebra keeps the shared name
    elsewhere). Passives are their own instance and pass through."""
    inst = hot.get(sym)
    if not inst or "_" not in sym:
        return sym
    tgt = inst.replace(".", "_")
    if sym == tgt:
        return sym                    # a passive IS its instance
    head, _, own = sym.partition("_")
    if own == tgt:
        return sym
    return f"{head}_{tgt}"


def _respell_to(sym: str, inst: str) -> str:
    """_respell_sym's transformation onto an EXPLICIT member."""
    if not inst or "_" not in sym:
        return sym
    tgt = inst.replace(".", "_")
    if sym == tgt:
        return sym
    head, _, own = sym.partition("_")
    return sym if own == tgt else f"{head}_{tgt}"


def attach_root_sensitivities(led: MarginLedger, system, primitives, probe: str, *,
                  alias=None, top: int | None = None,
                  complexity: int = 1,
                  min_phase_deg: float = 2.0,
                  min_mag_db: float = 0.5) -> MarginLedger:
    """L-G: the per-element ROOT SENSITIVITIES of each margin-relevant singularity (semi-relative, x ds/dx) --
    attribution by sensitivity, which works where modal identity
    cannot (the minor-combination poles). For any simple root x of T,
    the implicit-function theorem gives

        dx/dtheta = -(dT/dtheta)/(dT/ds)  at s = x:

    dT/dtheta is the L-C rank-one form at complex s, dT/ds is exact
    via dA/ds = Q (dx = -A^-1 Q x on both solves), and at a pole both
    derivatives diverge at the same rate while the ratio converges --
    the computed root's own rooting error provides the offset that
    keeps floats finite. Per element: df0 (percent |x| shift per percent
    of the element, Re of the relative sensitivity) and dzeta (damping
    shift per percent). Entries below the share floors keep no sensitivities
    -- the readout stays about the margin."""
    from .reconstruct import reconstruct
    from .secondorder import loop_pair_jets

    stamps, vals, _ = _collect_stamps(system, primitives, alias)
    cands = [e for e in led.entries
             if e.kind in ("pole", "zero") and e.roots
             and ((e.phase_deg is not None
                   and abs(e.phase_deg) >= min_phase_deg)
                  or (e.mag_db is not None
                      and abs(e.mag_db) >= min_mag_db))]
    if not stamps or not cands:
        return led
    fn, Q, ib, ip = _tian_context(system, probe)
    dim = Q.shape[0]
    zv = np.zeros(dim, complex)
    zv[ib] = 1.0
    zi = np.zeros(dim, complex)
    zi[ip] = 1.0
    for e in cands:
        x = max(e.roots, key=lambda r: r.imag)
        # a DETERMINISTIC small offset off the root: at a pole the two
        # divergences cancel in the ratio, but evaluating exactly at
        # the computed root leaves the conditioning to the rooting
        # error's mercy (the fc dominant pole came back NaN); 1e-6
        # relative keeps every read finite and costs O(1e-6) accuracy
        s0 = 2 * np.pi * complex(x) * (1.0 + 1e-6)
        M = np.linalg.inv(np.asarray(fn(s0), complex))
        xv = M @ zv
        xi = M @ zi
        B = -xv[ib]
        D = xv[ip]
        A_ = -xi[ib]
        C = xi[ip]
        dxv = -(M @ (Q @ xv))
        dxi = -(M @ (Q @ xi))
        T_s = _combine_dT(B, D, A_, C, -dxv[ib], dxv[ip],
                          -dxi[ib], dxi[ip])
        if T_s == 0:
            continue
        rsens = []
        taus = []
        conds: dict = {}
        for name, ss in stamps.items():
            # per-INSTANCE partial derivatives: a MATCHED symbol sums
            # stamps of several devices, but usually only one member's
            # physical position moves this root -- lighting the whole
            # group on the schematic was misleading
            per: dict = {}
            for kind, a, b, inst in ss:
                k = _kfac(kind, s0, vals[name])
                u = M @ a
                cv = k * (b @ xv)
                ci = k * (b @ xi)
                d = per.setdefault(inst, [0j, 0j, 0j, 0j])
                d[0] -= cv * u[ib]
                d[1] -= cv * u[ip]
                d[2] -= ci * u[ib]
                d[3] -= ci * u[ip]
            tot = [sum(d[k2] for d in per.values()) for k2 in range(4)]
            T_th = _combine_dT(B, D, A_, C, -tot[0], tot[1],
                               -tot[2], tot[3])
            S_rel = vals[name] * (-T_th / T_s) / s0   # %/% (complex)
            if not np.isfinite(S_rel) or S_rel == 0:
                continue          # zero sensitivity is not a contributor
            parts = {i2: abs(_combine_dT(B, D, A_, C, -d[0], d[1],
                                         -d[2], d[3]))
                     for i2, d in per.items()}
            pmax = max(parts.values()) or 1.0
            hot = tuple(sorted(i2 for i2, v in parts.items()
                               if v >= 0.3 * pmax)) \
                or tuple(sorted(per))
            # how the symbol's sensitivity SPLITS over its hot members,
            # by their partials. The list and the map used to disagree
            # here: the list hung a shared symbol under its first hot
            # instance in full, the map lit every hot one in full --
            # so a matched pair was counted once in one view and twice
            # in the other. One split, read by both.
            htot = sum(parts[i2] for i2 in hot) or 1.0
            shares = {i2: parts[i2] / htot for i2 in hot}
            dzeta = float((-np.real(x * S_rel)
                           + x.real * np.real(S_rel)) / abs(x)) * 0.01
            rsens.append({"symbol": name,
                           "df0": float(np.real(S_rel)),
                           "dzeta": dzeta,
                           "mag": float(abs(S_rel)),
                           "insts": hot,
                           "shares": shares})
            kinds = {k for k, _, _, _ in ss}
            if kinds & {"c", "cx"}:
                # L-H: scaling every capacitor by k scales every
                # singularity by 1/k, so 1/s is HOMOGENEOUS of degree
                # one in the caps -- by Euler's theorem the per-cap
                # time constants tau_i = C_i d(-1/s)/dC_i sum EXACTLY
                # to -1/s. tau_i = S_rel_i / s (S_rel is the relative
                # sensitivity), R_eff,i = tau_i / C_i.
                taus.append({"symbol": name, "tau": S_rel / s0,
                             "c": vals[name], "insts": hot})
            if kinds & {"g", "vccs"}:
                conds[name] = vals[name]
            elif kinds & {"r"}:
                conds[name] = 1.0 / vals[name]
        rsens.sort(key=lambda m: -m["mag"])
        # every contributor is already computed; keeping them all is
        # a few small dicts per entry, and the display surfaces bound
        # themselves (column 2, tooltip 8, cross-light 3% floor)
        e.root_sens = tuple(rsens if top is None else rsens[:top])
        if taus:
            tsum = sum(t["tau"] for t in taus)
            e.tau_residual = float(abs(tsum + 1.0 / s0) * abs(s0))
            taus.sort(key=lambda t: -abs(t["tau"]))
            e.taus = tuple(
                {"symbol": t["symbol"],
                 "share": float(np.real(t["tau"] * (-s0))),
                 "r_eff": float(np.real(t["tau"]) / t["c"]),
                 "insts": t.get("insts", ())}
                for t in taus[:4])
        # THE FORMULA, read from the jet rather than searched for.
        # The level-2 couplings determine a monomial-of-sums
        # outright -- each block's exponent stated once by the
        # couplings and again by the shares -- so there is no
        # vocabulary to enumerate and no value to fit, and a root
        # that will not partition says so instead of printing a
        # form its own sensitivities contradict.
        rec = None
        try:
            # a PAIR carries two invariants and both come from the one
            # set of root derivatives, so the damping costs nothing
            # extra -- and a designer who cannot express zeta cannot
            # trade the peaking and settling it sets
            (gj, Hj), (gz, Hz) = loop_pair_jets(
                fn, Q, ib, ip, stamps, vals, 2 * np.pi * complex(x))
            if gz and e.zeta:
                rz = reconstruct(gz, Hz, vals, stamps, abs(e.zeta),
                                 two_pi=False,
                                 floor={1: 0.02, 2: 0.012}.get(
                                     complexity, 0.006),
                                 lump=True)
                if rz.verdict == "formula":
                    # the formula may carry a bare number -- the 2 of
                    # zeta = 1/(2Q) -- which the raw prefactor does
                    # not include and the estimate must
                    e.zeta_estimate = {
                        "text": rz.text, "tex": rz.tex,
                        "zeta_est": abs(e.zeta) * rz.constant / rz.prefactor
                        if rz.prefactor else abs(e.zeta),
                        "err": (rz.constant / rz.prefactor - 1.0)
                        if rz.prefactor else 0.0}
            # the COMPLEXITY knob now sets how small a mover may still
            # enter the expression: there is no search to widen, only
            # a threshold below which an element is left to the
            # coverage residual instead of being named. The reader
            # LUMPS for itself: symbols the root sees only through
            # their sum are folded before the reading, so a cascode
            # divide prints as the two-branch sum it is instead of a
            # bare G_eff waiting for the port to be asked. A plain sum
            # comes back unfolded, so nothing that read before reads
            # differently now.
            rec = reconstruct(gj, Hj, vals, stamps, abs(x),
                              floor={1: 0.02, 2: 0.012}.get(complexity,
                                                            0.006),
                              lump=True)
        except Exception:                          # noqa: BLE001
            rec = None                             # never fatal
        e.reconstruction = rec
        if rec is not None and rec.verdict == "formula":
            e.estimate = {"text": rec.text, "tex": rec.tex,
                          "f_est_hz": rec.value,
                          "prefactor": rec.prefactor,
                          "err": (rec.constant / rec.prefactor - 1.0)
                          if rec.prefactor else 0.0,
                          "blocks": rec.blocks}
        elif rec is not None:
            e.shared = rec.movers[:4]
        # the named formulas speak PHYSICAL origin: every symbol
        # re-spelled onto this root's hot member (the map and the
        # contributors list already do)
        hot = {}
        hot_all = {}
        for m in e.root_sens:
            ins = m.get("insts")
            if ins:
                hot[m["symbol"]] = ins[0]
                hot_all.setdefault(m["symbol"], tuple(ins))
        for t_ in e.taus:
            ins = t_.get("insts")
            if ins:
                hot.setdefault(t_["symbol"], ins[0])
                hot_all.setdefault(t_["symbol"], tuple(ins))
        if e.shared:
            e.shared = tuple((_respell_sym(s2, hot), v)
                             for s2, v in e.shared)
    return led


def rebias_exponent(symbol: str, region=None, *,
                    explicit: bool = False) -> float:
    """d ln(parameter) / d ln(bias): how much of a 1% re-bias each
    small-signal parameter actually sees.

    "Every parameter up 1% together" is not a knob anyone has, and it
    manufactures cancellations: a root going as gm/gds reads a net of
    exactly zero under uniform scaling, while re-biasing the device --
    gm as sqrt(I_D), gds as I_D in strong inversion -- moves it by
    0.5*S_gm + 1.0*S_gds. In subthreshold gm is proportional to I_D
    and the exponent is 1. Capacitances barely track bias. An explicit
    component (a netlist C or R) IS its own knob: exponent 1.
    Region codes follow Spectre: 3 is subthreshold."""
    if explicit:
        return 1.0
    head = symbol.split("_", 1)[0].lower()
    sub = region is not None and int(region) == 3
    if head in ("gm", "gmbs"):
        return 1.0 if sub else 0.5
    if head in ("gds", "gii"):
        return 1.0
    if head.startswith("c"):
        return 0.0
    return 1.0


def margin_equation(led: MarginLedger, *, budget_deg: float = 2.0,
                    pair_tol: float = 0.3,
                    fu_gate: float = 0.3) -> dict:
    """L-I: THE MARGIN EQUATION of this amplifier. The folded ledger's
    pole and zero terms become arctan terms -- theta = arctan(f_u/f)
    for a real root, the second-order arctan(2 zeta u/(1-u^2)) for a
    pair -- each frequency replaced by its certified L-H estimate
    where one exists, everything else (doublets, the gathered tail)
    a small exact remainder constant. Two sums come back:

    - pm_check_deg: the equation with EXACT frequencies. It must equal
      the ledger's PM to numerical precision -- the certificate that
      the equation's STRUCTURE is right (asserted in tests, printed
      nowhere).
    - pm_pred_deg: the equation with the estimated frequencies -- what
      the designer's formulas actually predict, with pm_err_deg the
      honest gap to the exact margin.

    When the anchor's DC gain and a dominant pole are both on the
    ledger, the classic crossover relation f_u ~ T0 f_dom is evaluated
    and included ONLY when it holds within `fu_gate` -- on a loop that
    is not a GBW loop the relation is absent, not wrong."""
    if led.f_unity_hz is None:
        raise ValueError("no unity crossing: no phase margin to explain")
    fu = led.f_unity_hz
    folded = led.fold(budget_deg=budget_deg, pair_tol=pair_tol)
    anchor = folded[0]

    def _theta(f0, zeta, pair):
        u = fu / f0
        if pair:
            return float(np.degrees(np.arctan2(2 * abs(zeta) * u,
                                               1 - u * u)))
        return float(np.degrees(np.arctan(u)))

    terms = []
    rest = 0.0
    for e in folded[1:]:
        if e.phase_deg is None:
            continue
        if e.kind not in ("pole", "zero"):
            rest += e.phase_deg
            continue
        sign = 1.0 if e.phase_deg >= 0 else -1.0
        th_ex = _theta(e.f0_hz, e.zeta, e.pair)
        f_pr = (e.estimate["f_est_hz"] if e.estimate is not None
                else e.f0_hz)
        th_pr = _theta(f_pr, e.zeta, e.pair)
        # a PARTIAL reading does not substitute -- its frequency would
        # be wrong, sometimes fourfold -- but the equation went silent
        # about it, so a bench whose terms all read partially printed
        # nothing but "(exact)" and looked as though nothing had been
        # found (field report on the folded cascode). It is carried
        # through and rendered as structure, never as a value.
        rec = getattr(e, "reconstruction", None)
        partial = None
        if e.estimate is None and rec is not None \
                and rec.verdict == "partial" and rec.blocks:
            # tex is late-bound and picks up a port expansion by
            # itself; text is a string, so it is taken through the
            # accessor that substitutes one
            partial = {"tex": rec.tex, "text": rec.display_text(),
                       "aggregate": rec.aggregate,
                       # an aggregate form is exact by its own
                       # identity, so the discarded reading's
                       # prefactor must not be quoted against it
                       "prefactor": (None if rec.aggregate
                                     else rec.prefactor)}
        terms.append({"kind": e.kind, "rhp": e.rhp, "pair": e.pair,
                      "sign": sign, "f0_hz": e.f0_hz,
                      "zeta": (abs(e.zeta) if e.pair else None),
                      "estimate": e.estimate, "partial": partial,
                      "zeta_estimate": e.zeta_estimate,
                      "theta_exact_deg": th_ex,
                      "theta_pred_deg": th_pr})
    # the HAND-equation convention: terms in ascending frequency, so
    # the subscripts the renderers assign (p1, p2, ..., z1, ...) index
    # frequency order -- f_p1 < f_p2 -- not contribution rank (the
    # ledger table keeps that ranking; the sum commutes)
    terms.sort(key=lambda t: t["f0_hz"])
    a_deg = float(anchor.phase_deg)
    if a_deg < 0:
        # the anchor is an angle mod 360: +180 and -180 are the same
        # anchor, and the +180 form is the one the hand equation uses
        a_deg += 360.0
    check = float(_fold(a_deg + sum(t["sign"] * t["theta_exact_deg"]
                                    for t in terms) + rest))
    pred = float(_fold(a_deg + sum(t["sign"] * t["theta_pred_deg"]
                                   for t in terms) + rest))
    fu_est = None
    poles = [t for t in terms if t["kind"] == "pole"]
    if anchor.mag_db is not None and poles:
        dom = min(poles, key=lambda t: t["f0_hz"])
        f_dom = (dom["estimate"]["f_est_hz"] if dom["estimate"]
                 else dom["f0_hz"])
        f_gbw = 10.0 ** (anchor.mag_db / 20.0) * f_dom
        err = abs(f_gbw - fu) / fu
        if err <= fu_gate:
            fu_est = {"f_est_hz": float(f_gbw), "err": float(err),
                      "t0_db": float(anchor.mag_db)}
    return {"pm_exact_deg": float(led.pm_deg),
            "f_unity_hz": float(fu),
            "anchor_deg": a_deg,
            "t0_db": (float(anchor.mag_db)
                      if anchor.mag_db is not None else None),
            "rest_deg": float(rest),
            "terms": terms,
            "pm_pred_deg": pred,
            "pm_err_deg": float(abs(pred - led.pm_deg)),
            "pm_check_deg": check,
            "fu_est": fu_est}


def _unit_tex(op, syms, tx, inline: bool = False) -> str:
    g = [tx(s) for s in syms]
    if op == "single":
        return g[0]
    if inline:
        return rf"{g[0]}\,{g[1]}/{g[2]}"
    return rf"\frac{{{g[0]}\,{g[1]}}}{{{g[2]}}}"


def _units_tex(units, tx, inline: bool = False) -> str:
    return "+".join(_unit_tex(op, syms, tx, inline)
                    for op, syms in units)


def _estimate_tex(est, tx) -> str:
    if "tex" in est:                      # the jet reading renders itself
        return est["tex"](tx)
    """LaTeX for a named L-H estimate: the conductance form over
    2 pi times the capacitance sum, one fraction. The level-2 pair
    form renders as omega0/2pi = sqrt(Ga Gb / (Ca Cb))/2pi, with
    each C a LUMPED node capacitance."""
    if est.get("form") == "pair":
        def _side(units_f):
            parts = [tx(s) for _op, syms in units_f for s in syms]
            inner = "+".join(parts)
            return inner if len(parts) == 1 else f"({inner})"

        def _cap(csyms_f):
            inner = "+".join(tx(s) for s in csyms_f)
            return inner if len(csyms_f) == 1 else f"({inner})"

        return (rf"\frac{{1}}{{2\pi}}\sqrt{{\frac"
                rf"{{{_side(est['g_units_a'])}\,"
                rf"{_side(est['g_units_b'])}}}"
                rf"{{{_cap(est['c_syms_a'])}\,"
                rf"{_cap(est['c_syms_b'])}}}}}")
    c = "+".join(tx(s) for s in est["c_syms"])
    units = est["g_units"]
    if len(units) == 1 and units[0][0] == "ratio":
        # the lone cascode boost folds its divider into the
        # denominator: g1 g2 / (2 pi g3 (C))
        g = [tx(s) for s in units[0][1]]
        return (rf"\frac{{{g[0]}\,{g[1]}}}"
                rf"{{2\pi\,{g[2]}\,({c})}}")
    return (rf"\frac{{{_units_tex(units, tx)}}}"
            rf"{{2\pi({c})}}")


def _fsplit(f) -> tuple[str, str]:
    """(number, unit) for a frequency -- 179 Hz must not print as
    0.000179 MHz, nor 4.34 GHz as 4.34e+03 MHz."""
    for scale, unit in ((1e12, "THz"), (1e9, "GHz"), (1e6, "MHz"),
                        (1e3, "kHz")):
        if f >= scale:
            return f"{f / scale:.3g}", unit
    return f"{f:.3g}", "Hz"


def _ftex(f) -> str:
    n, u = _fsplit(f)
    return rf"{n}\,\mathrm{{{u}}}"


def _mhz(f) -> str:
    n, u = _fsplit(f)
    return f"{n} {u}"


def margin_equation_names(eq) -> list[str]:
    """The per-term subscripts (p1, p2, ..., z1, ...) in the terms'
    own ascending-frequency order -- ONE naming, shared by the LaTeX
    and text renderers and the GUI contributors table, so a theta in
    the equation and its row in the table wear the same name."""
    out = []
    np_ = nz = 0
    for t in eq["terms"]:
        if t["kind"] == "pole":
            np_ += 1
            out.append(f"p{np_}")
        else:
            nz += 1
            out.append(f"z{nz}")
    return out


def margin_equation_latex(eq, texer=None) -> list[str]:
    """The equation as KaTeX lines. `texer` maps a symbol name to its
    LaTeX (the GUI passes view.symbol_tex with the session aliases);
    the default wraps names verbatim."""
    tx = texer or (lambda n: rf"\mathrm{{{n}}}")
    parts = [rf"{'-' if eq['anchor_deg'] < 0 else ''}180^\circ"]
    names = margin_equation_names(eq)
    for nm, t in zip(names, eq["terms"]):
        # every theta in the SUM is a LINK: \htmlData tags it for the
        # web shell, which scrolls to the definition line below and
        # selects the root's table row (the full cross-light)
        parts.append(rf"{'+' if t['sign'] > 0 else '-'}"
                     rf" \htmlData{{term={nm}}}{{\theta_{{{nm}}}}}")
    if abs(eq["rest_deg"]) >= 0.05:
        parts.append(rf"{'+' if eq['rest_deg'] >= 0 else '-'} "
                     rf"{abs(eq['rest_deg']):.1f}^\circ")
    lines = [r"\mathrm{PM} \approx " + " ".join(parts)
             + rf" = {eq['pm_pred_deg']:.1f}^\circ"
             + rf"\quad(\text{{exact }}{eq['pm_exact_deg']:.1f}^\circ)"]
    for nm, t in zip(names, eq["terms"]):
        if t["pair"]:
            # the second-order form written directly in frequencies
            # (u = f_u/f folded in): 2 zeta f_u f / (f^2 - f_u^2)
            th = (rf"\htmlData{{termdef={nm}}}{{\theta_{{{nm}}}}} = \arctan"
                  rf"\frac{{2\zeta_{{{nm}}} f_u f_{{{nm}}}}}"
                  rf"{{f_{{{nm}}}^2 - f_u^2}}"
                  rf"\Big|_{{\zeta_{{{nm}}}={t['zeta']:.2f}}}"
                  rf" = {t['theta_exact_deg']:.1f}^\circ")
        else:
            th = (rf"\htmlData{{termdef={nm}}}{{\theta_{{{nm}}}}}"
                  rf" = \arctan(f_u/f_{{{nm}}})"
                  rf" = {t['theta_exact_deg']:.1f}^\circ")
        if t["estimate"] is not None:
            # the parens anchor the term to its EXACT root -- the
            # value the subscripts sort by and the Sensitivities
            # table lists -- with the SIGNED deviation of the named
            # formula (field request: estimate values alone can read
            # out of order inside their error bars)
            est = t["estimate"]
            dev = (est["f_est_hz"] - t["f0_hz"]) / t["f0_hz"]
            th += (rf",\quad \htmlData{{froot={nm}}}{{f_{{{nm}}}}}"
                   rf" \approx {_estimate_tex(est, tx)}"
                   rf" = {_ftex(est['f_est_hz'])}"
                   rf"\;({_ftex(t['f0_hz'])}\ {dev:+.0%})"
                   .replace("%", r"\%"))
        else:
            th += (rf",\quad \htmlData{{froot={nm}}}{{f_{{{nm}}}}}"
                   rf" = {_ftex(t['f0_hz'])}"
                   rf"\;(\text{{exact}})")
            pa = t.get("partial")
            if pa is not None:
                # structure, not a value: what the jet could read of
                # this root, and how much of the root it accounts for
                short = (rf"\;\times{pa['prefactor']:.2f}"
                         if pa["prefactor"] else "")
                # the aggregate is tagged with the TERM name, so a
                # click on G_eff lands on the same row an f_p click
                # would -- one identity for the root, three ways in
                th += (rf",\quad \text{{reads }} {pa['tex'](tx, nm)}"
                       rf"{short}")
        ze = t.get("zeta_estimate")
        if ze is not None:
            zdev = ((ze["zeta_est"] - t["zeta"]) / t["zeta"]
                    if t.get("zeta") else ze["err"])
            th += (rf",\quad \zeta_{{{nm}}} \approx {ze['tex'](tx)}"
                   rf" = {ze['zeta_est']:.2f}"
                   rf"\;({t['zeta']:.2f}\ {zdev:+.0%})"
                   .replace("%", r"\%"))
        lines.append(th)
    # every theta references f_u, so its line comes FIRST, right
    # after the sum
    if eq["fu_est"] is not None:
        fe = eq["fu_est"]
        fdev = ((fe["f_est_hz"] - eq["f_unity_hz"])
                / eq["f_unity_hz"])
        fu_line = (
            rf"f_u \approx T_0 f_{{p1}}"
            rf" = {_ftex(fe['f_est_hz'])}"
            + rf"\;({_ftex(eq['f_unity_hz'])}\ {fdev:+.0%})"
            .replace("%", r"\%"))
    else:
        fu_line = (rf"f_u = {_ftex(eq['f_unity_hz'])}"
                   rf"\;(\text{{exact}})")
    lines.insert(1, fu_line)
    # the anchor as its own definition line: T0 is the DC loop gain
    # the sums start from -- +180 deg into the phase, its dB into the
    # gain-margin budget
    t0 = rf"\arg T_0 = {eq['anchor_deg']:.0f}^\circ"
    if eq["t0_db"] is not None:
        t0 = (rf"T_0 = {eq['t0_db']:+.1f}\,\mathrm{{dB}},\quad "
              + t0 + r"\;(\text{exact})")
    else:
        t0 += r"\;(\text{exact})"
    lines.insert(2, t0)
    return lines


def margin_equation_text(eq) -> list[str]:
    """The equation as plain text -- the no-WebEngine fallback, the
    log, and the field scripts."""
    parts = [f"{'-' if eq['anchor_deg'] < 0 else ''}180"]
    names = margin_equation_names(eq)
    for nm, t in zip(names, eq["terms"]):
        parts.append(f"{'+' if t['sign'] > 0 else '-'} th_{nm}")
    if abs(eq["rest_deg"]) >= 0.05:
        parts.append(f"{'+' if eq['rest_deg'] >= 0 else '-'} "
                     f"{abs(eq['rest_deg']):.1f}")
    lines = ["PM ~ " + " ".join(parts)
             + f" = {eq['pm_pred_deg']:.1f} deg "
             f"(exact {eq['pm_exact_deg']:.1f})"]
    for nm, t in zip(names, eq["terms"]):
        row = f"th_{nm} = {t['theta_exact_deg']:.1f} deg, f_{nm}"
        if t["estimate"] is not None:
            dev = ((t["estimate"]["f_est_hz"] - t["f0_hz"])
                   / t["f0_hz"])
            row += (f" ~ {t['estimate']['text']}"
                    f" = {_mhz(t['estimate']['f_est_hz'])}"
                    f" ({_mhz(t['f0_hz'])} {dev:+.0%})")
        else:
            row += f" = {_mhz(t['f0_hz'])} (exact)"
            pa = t.get("partial")
            if pa is not None:
                ag = pa.get("aggregate")
                short = (f" x{pa['prefactor']:.2f}"
                         if pa["prefactor"] else "")
                if ag:
                    # the value belongs to the SYMBOL. Once a port
                    # expansion has written the algebra out in its
                    # place there is nothing for it to attach to, and
                    # trailing it anyway would read as the frequency
                    # equalling a conductance. The expansion is its
                    # own statement; the value keeps to the log.
                    short = (f" with {ag['symbol']} = "
                             f"{ag['value']:.4g} {ag['unit']}"
                             if ag["symbol"] in pa["text"] else "")
                row += f", reads {pa['text']}{short}"
        ze = t.get("zeta_estimate")
        if ze is not None:
            zdev = ((ze["zeta_est"] - t["zeta"]) / t["zeta"]
                    if t.get("zeta") else ze["err"])
            row += (f", zeta_{nm} ~ {ze['text']} = {ze['zeta_est']:.2f}"
                    f" ({t['zeta']:.2f} {zdev:+.0%})")
        lines.append(row)
    if eq["fu_est"] is not None:
        fdev = ((eq["fu_est"]["f_est_hz"] - eq["f_unity_hz"])
                / eq["f_unity_hz"])
        fu_line = (f"f_u ~ T0*f_p1 = {_mhz(eq['fu_est']['f_est_hz'])}"
                   f" ({_mhz(eq['f_unity_hz'])} {fdev:+.0%})")
    else:
        fu_line = f"f_u = {_mhz(eq['f_unity_hz'])} (exact)"
    lines.insert(1, fu_line)
    t0 = f"arg T0 = {eq['anchor_deg']:.0f} deg"
    if eq["t0_db"] is not None:
        t0 = f"T0 = {eq['t0_db']:+.1f} dB, " + t0
    lines.insert(2, t0 + " (exact)")
    return lines


def margin_sensitivities(system, primitives, probe: str, led, *,
                         alias=None, rel_step: float = 1e-4) -> list[dict]:
    """L-C: which ELEMENT moves the margin. Complements the singularity
    ledger: L-A/L-B say what dynamics set PM and GM, this says which
    design variables move them, exactly, with no re-solve.

    Every parameter kind stamps rank-one: theta enters A as
    k(theta, s) a b^T (a = b for a passive pair; output/control
    incidences for vccs and cx; branch rows for l and vcvs), so
    dx/dtheta = -k (b.x) A^-1 a from the SAME cached per-frequency
    solves the compensation updater holds. The Tian combine obliges:
    its denominator is its numerator plus one, so dT = dN / (N + 1)^2.

    Both margin derivatives carry the IMPLICIT term -- the crossing
    itself moves: dPM = dphi/dtheta + (dphi/dlnf) dlnf_u/dtheta with
    dlnf_u/dtheta = -(dln|T|/dtheta)/(dln|T|/dlnf), and dGM the dual
    with phase holding the crossing. The frequency slopes come from
    two extra solves at f (1 +- rel_step). A grazing crossing makes
    the implicit term large -- that is physics, not error.

    Matched devices share a symbol; their stamps sum. Returns one row
    per NUMERIC symbol, {"symbol", "value", "dpm_deg_per_pct",
    "dgm_db_per_pct"} (None where that crossing is absent), ranked by
    |dpm|. Symbols without a numeric value are skipped -- this ledger
    is numeric at the operating point."""
    from .compensate import LoopGainUpdater

    fu, fg = led.f_unity_hz, led.f_180_hz
    pts: list[float] = []
    for f0 in (fu, fg):
        if f0 is not None:
            pts += [f0 * np.exp(-rel_step), f0, f0 * np.exp(rel_step)]
    if not pts:
        return []
    upd = LoopGainUpdater(system, probe, pts)
    T = upd.baseline()
    stamps, vals, meta = _collect_stamps(system, primitives, alias)
    if not stamps:
        return []

    def _dT(m):
        return _dT_by_symbol(stamps, vals, 2j * np.pi * upd.freqs[m],
                             upd.Minv[m], upd.xv[m], upd.xi[m],
                             upd.ib, upd.ip)

    rows = {name: {"symbol": name, "value": vals[name],
                   "insts": meta[name]["insts"],
                   "param": meta[name]["param"],
                   # L-J actionability: a netlist component (R, C, L)
                   # is an OP-INVARIANT knob -- the advice is exact;
                   # a device model parameter tracks bias and
                   # geometry -- directional advice, re-bias to apply
                   "knob": ("component" if meta[name]["explicit"]
                            else "device"),
                   "dpm_deg_per_pct": None, "dgm_db_per_pct": None}
            for name in stamps}
    off = 0
    if fu is not None:
        Tm, T0, Tp = T[off], T[off + 1], T[off + 2]
        dphi = float(_fold(np.degrees(np.angle(Tp) - np.angle(Tm)))) \
            / (2 * rel_step)                       # deg per ln f
        dlnm = float(np.log(np.abs(Tp) / np.abs(Tm))) / (2 * rel_step)
        for name, d in _dT(off + 1).items():
            g = d / T0
            dlnf = -g.real / dlnm                  # the crossing moves
            dpm = np.degrees(g.imag) + dphi * dlnf
            rows[name]["dpm_deg_per_pct"] = float(dpm * vals[name] * 0.01)
        off += 3
    if fg is not None:
        Tm, T0, Tp = T[off], T[off + 1], T[off + 2]
        dphi = float(_fold(np.degrees(np.angle(Tp) - np.angle(Tm)))) \
            / (2 * rel_step)
        dlnm = float(np.log(np.abs(Tp) / np.abs(Tm))) / (2 * rel_step)
        for name, d in _dT(off + 1).items():
            g = d / T0
            dlnf = -np.degrees(g.imag) / dphi      # phase holds f_180
            dgm = -(20.0 / np.log(10.0)) * (g.real + dlnm * dlnf)
            rows[name]["dgm_db_per_pct"] = float(dgm * vals[name] * 0.01)
    out = list(rows.values())
    out.sort(key=lambda r: -(abs(r["dpm_deg_per_pct"])
                             if r["dpm_deg_per_pct"] is not None
                             else abs(r["dgm_db_per_pct"] or 0.0)))
    return out


def _closed_loop_roots(tf, raw_n, raw_d) -> np.ndarray:
    """Roots of D - N (rad/s), the closed-loop poles in the ledger's
    convention, from the MINIMAL form of T.

    The assembled Tian loop gain is not minimal: its numerator and
    denominator share the closed-loop determinant (both are closed-loop
    responses of the probed system), so D - N of the raw rational is that
    determinant SQUARED up to artifact factors -- every closed-loop pole
    counted twice (field case: a 26-node LDO read 4 right-half-plane
    poles where its supply-to-output transfer, rooted exactly, has one
    pair). Converting the raw coefficients to float64 first was worse
    still: the ladder's far ends underflow and the truncated difference
    grows spurious right-half-plane roots on a stable loop.

    So: take the minimal zeros and poles (exact rooting, artifact pairs
    cancelled), work in sigma = s/w0 with w0 the geometric mean of the
    poles so no product overflows, fix the minimal numerator's gain by
    matching T at sigma = j (an exact-coefficient read, independent of
    the rooting), and root prod(sigma - p) - k prod(sigma - z). Falls
    back to the raw D - N, formed in sympy and rooted with scaled_roots,
    when the exact minimal route is not available."""
    import sympy as sp

    from ..engine.mna import scaled_roots

    zeros, poles = tf._minimal_pz
    if getattr(tf, "_minimal_exact", False) and len(poles):
        poles = np.asarray(poles, complex)
        zeros = np.asarray(zeros, complex)
        w0 = float(np.exp(np.mean(np.log(np.abs(poles[np.abs(poles) > 0])))))
        ps, zs = poles / w0, zeros / w0
        t_j = complex(np.asarray(tf.numeric_hp([w0 / (2 * np.pi)]))[0])
        k = t_j * np.prod(1j - ps) / np.prod(1j - zs)
        if abs(k.imag) <= 1e-6 * abs(k):
            k = k.real
        pd = np.atleast_1d(np.poly(ps))
        pn = k * np.atleast_1d(np.poly(zs) if len(zs) else np.ones(1))
        pn = np.concatenate([np.zeros(len(pd) - len(pn)), pn])
        return np.roots(pd - pn) * w0
    cn = [sp.sympify(c) for c in raw_n]
    cd = [sp.sympify(c) for c in raw_d]
    width = max(len(cn), len(cd))
    cn = [sp.Integer(0)] * (width - len(cn)) + cn
    cd = [sp.Integer(0)] * (width - len(cd)) + cd
    return np.asarray(scaled_roots([a - b for a, b in zip(cd, cn)]))


def stability_certificate(tf, *, f=None, m=None, acc=None, fu=None,
                          rel_tol: float = 1e-7) -> dict:
    """L-J: is the loop stable, and do its margins say so?

    The Bode margins answer a question only a loop with one unity
    crossing, one critical-phase crossing and no right-half-plane pole
    is asking. Beleca, Plesa, Onet and Neag (ROMJIST 2022) show the
    rest: a conditionally stable loop reads a negative gain margin and
    a healthy phase margin and is stable; a loop with a pole at the
    origin or in the right half-plane reads a positive phase margin
    and is unstable. Their criteria -- the phase crosses the critical
    line an EVEN number of times below f_u, the gain margin is
    undefined when the Nyquist curve cuts the critical axis more than
    once, and otherwise the encirclement count against the open-loop
    RHP poles -- are what this certificate reports. The verdict itself
    does not need the Nyquist plot: the loop gain is a known rational,
    so the closed-loop poles are the roots of D(s) - N(s) -- the
    ledger's convention has arg T(0) = 180 and the critical point at
    +1 -- and the loop is stable exactly when none sits in the right
    half-plane. The Nyquist count then follows as Z - P, and the Bode
    diagnostics say why the margins did or did not mislead.

    `f`, `m`, `acc`, `fu` are the ledger's own grid, magnitude in dB,
    continuous phase in degrees and refined unity crossing; without
    them only the exact verdict and the encirclement count are given.
    Returns {verdict, closed_loop_rhp, closed_loop_rhp_hz (the offending
    closed-loop poles, Hz), rhp_poles, origin_poles, encirclements_cw,
    unity_crossings, axis_cuts, cuts_before_fu, parity, gm_defined,
    bode_applies, reason}."""
    # N and D on ONE scale: the tf's own float path normalizes each
    # polynomial by its largest coefficient, which is fine for roots
    # of either and loses exactly the gain that D - N depends on
    import sympy as sp

    ex = tf._exact_coeffs
    if ex is not None:
        raw_n, raw_d = ex
    else:
        n, d = tf.num_den
        subs = tf._subs_map()
        raw_n = [c.xreplace(subs) for c in n.all_coeffs()]
        raw_d = [c.xreplace(subs) for c in d.all_coeffs()]
    closed = _closed_loop_roots(tf, raw_n, raw_d)
    big = float(np.max(np.abs(closed))) if len(closed) else 1.0
    z = int(np.sum(closed.real > rel_tol * np.maximum(np.abs(closed),
                                                       1e-12 * big)))
    poles = np.asarray(tf.poles(), complex)
    big_p = float(np.max(np.abs(poles))) if len(poles) else 1.0
    origin = int(np.sum(np.abs(poles) <= 1e-12 * big_p))
    p = int(np.sum(poles.real > rel_tol * np.maximum(np.abs(poles),
                                                     1e-12 * big_p)))
    rhp_hz = [complex(r) / (2 * np.pi) for r in closed
              if r.real > rel_tol * max(abs(r), 1e-12 * big)]
    out = {"verdict": "stable" if z == 0 else "unstable",
           "closed_loop_rhp": z, "closed_loop_rhp_hz": rhp_hz,
           "rhp_poles": p, "origin_poles": origin,
           "encirclements_cw": z - p,
           "unity_crossings": None, "axis_cuts": None,
           "cuts_before_fu": None, "parity": None,
           "gm_defined": None, "bode_applies": None}
    why = []
    if z:
        why.append(f"the closed loop has {z} right-half-plane "
                   f"pole{'s' if z > 1 else ''}")
    else:
        why.append("the closed loop has no right-half-plane pole")
    if p or origin:
        turns = p - z
        why.append(f"T has {p} right-half-plane and {origin} origin "
                   f"pole{'s' if p + origin > 1 else ''}, so the Bode "
                   f"margins do not decide; by Nyquist the critical point "
                   f"is encircled {abs(turns)} time"
                   f"{'s' if abs(turns) != 1 else ''} "
                   f"{'counter-clockwise' if turns > 0 else 'clockwise'} "
                   f"against {p} required")
    if f is not None and m is not None and acc is not None:
        m = np.asarray(m, float)
        acc = np.asarray(acc, float)
        n_u = int(np.sum(np.diff(np.sign(m)) != 0))
        phf = ((acc + 180.0) % 360.0) - 180.0
        cut = np.where((np.diff(np.sign(phf)) != 0)
                       & (np.abs(np.diff(phf)) < 180.0))[0]
        cuts = int(len(cut))
        before = (int(np.sum(np.asarray(f)[cut] < fu))
                  if fu is not None else None)
        parity = (None if before is None
                  else "even" if before % 2 == 0 else "odd")
        applies = (p == 0 and origin == 0 and n_u == 1)
        out.update({"unity_crossings": n_u, "axis_cuts": cuts,
                    "cuts_before_fu": before, "parity": parity,
                    "gm_defined": cuts <= 1, "bode_applies": applies})
        if applies and before is not None:
            why.append(f"the phase crosses the critical line {before} "
                       f"time{'s' if before != 1 else ''} below f_u "
                       f"({parity}), so the phase margin "
                       f"{'holds' if before % 2 == 0 else 'misleads'}")
        elif n_u != 1 and not (p or origin):
            why.append(f"|T| crosses unity {n_u} times, so the phase "
                       f"margin is the worst of {n_u} readings")
        if cuts > 1:
            why.append(f"the Nyquist curve cuts the critical axis {cuts} "
                       f"times, so the gain margin is undefined and the "
                       f"ledger's GM is the worst of them")
    out["reason"] = "; ".join(why)
    return out


def margin_ledger(tf, *, system=None, f_lo: float | None = None,
                  f_hi: float | None = None,
                  n_grid: int = 2001) -> MarginLedger:
    """The margin ledger of a solved loop gain (a TransferFunction, or
    anything with poles()/zeros() in complex Hz and numeric(freqs)).
    Pass the MnaSystem the loop gain was solved on as `system` to also
    name each pole entry's owning node (attribute_ledger). See the
    module docstring for the doctrine."""
    from .sensitivity import loop_margins

    poles = [complex(p) for p in np.asarray(tf.poles(), complex)]
    zeros = [complex(z) for z in np.asarray(tf.zeros(), complex)]
    if not poles:
        raise ValueError("the margin ledger needs at least one pole")
    mags = np.abs(np.asarray(poles + zeros))
    if np.any(mags < 1e-12 * max(float(mags.max()), 1.0)):
        raise ValueError(
            "origin singularity (integrator): the ledger anchors its "
            "shares at DC, which an origin root makes undefined")
    if f_lo is None:
        f_lo = max(float(mags.min()) / 1e3, 1e-3)
    if f_hi is None:
        f_hi = min(float(mags.max()) * 1e3, 1e15)

    # the gain constant, in log magnitude + angle, from ONE direct read
    # at a reference frequency clear of every singularity. The read
    # must be HIGH PRECISION: float64 evaluation of the assembled
    # rational cancels catastrophically where |T| is small, and an
    # anchor read there poisons PM/GM themselves (field case: the
    # folded-cascode anchor at 276 MHz was 1.76 dB / 3.61 deg wrong,
    # and the certificate residuals said exactly that)
    _read = getattr(tf, "numeric_hp", tf.numeric)
    f_ref = float(np.exp(np.mean(np.log(mags))))
    for bump in (1.37, 0.53, 3.1, 7.7):
        if np.min(np.abs(mags - f_ref * bump)) > 1e-3 * f_ref * bump:
            f_ref *= bump
            break
    T_ref = complex(np.asarray(_read([f_ref]))[0])

    def _sum(fn, f):
        z = sum(fn(x, f) for x in zeros) if zeros else 0.0
        p = sum(fn(x, f) for x in poles)
        return z - p

    k_db = 20.0 * np.log10(abs(T_ref)) - float(_sum(_abs_db, f_ref))
    t0_db = k_db + float(_sum(lambda x, f: 10.0 * np.log10(
        x.real ** 2 + x.imag ** 2), 0.0))
    anchor_raw = float(_fold(
        np.degrees(np.angle(T_ref))
        - float(_sum(lambda x, f: np.degrees(np.angle(1j * f - x)),
                     f_ref)))) + float(_sum(
                         lambda x, f: np.degrees(_start_angle(x)), 0.0))

    def m_at(f):
        return t0_db + float(_sum(_climb_db, f))

    def acc_at(f):
        return anchor_raw + float(_sum(_dtheta_deg, f))

    # worst crossings through THE margin extractor, on the ledger's own
    # reconstruction, then bisection-refined to its machine precision
    f = np.geomspace(f_lo, f_hi, n_grid)
    m = t0_db + _sum(_climb_db, f)
    acc = anchor_raw + _sum(_dtheta_deg, f)
    h = 10.0 ** (m / 20.0) * np.exp(1j * np.radians(acc))
    pm, fu, gm, fgm = loop_margins(f, h)

    def _refine(f0, g):
        k = int(np.searchsorted(f, f0))
        lo, hi = f[max(k - 2, 0)], f[min(k + 1, len(f) - 1)]
        if g(lo) * g(hi) > 0:
            return f0
        for _ in range(80):
            mid = np.sqrt(lo * hi)
            if g(lo) * g(mid) <= 0:
                hi = mid
            else:
                lo = mid
        return float(np.sqrt(lo * hi))

    if fu is not None:
        fu = _refine(fu, lambda x: m_at(x))
        pm = float(_fold(acc_at(fu)))
    if fgm is not None:
        fgm = _refine(fgm, lambda x: float(_fold(acc_at(x))))
        gm = -m_at(fgm)

    # the certificate: the shares SUM to the direct evaluation
    ph_res = mg_res = None
    if fu is not None:
        d = complex(np.asarray(_read([fu]))[0])
        ph_res = abs(float(_fold(acc_at(fu)
                                 - np.degrees(np.angle(d)))))
    if fgm is not None:
        d = complex(np.asarray(_read([fgm]))[0])
        mg_res = abs(m_at(fgm) - 20.0 * np.log10(abs(d)))

    entries = [LedgerEntry(kind="gain", f0_hz=0.0, zeta=None, rhp=False,
                           pair=False,
                           phase_deg=(float(_fold(anchor_raw))
                                      if fu is not None else None),
                           mag_db=(t0_db if fgm is not None else None))]
    for kind, groups in (("pole", _group(poles)), ("zero", _group(zeros))):
        sign = -1.0 if kind == "pole" else 1.0
        for members in groups:
            x0 = members[0]
            entries.append(LedgerEntry(
                kind=kind,
                f0_hz=float(np.mean([abs(x) for x in members])),
                zeta=(float(-x0.real / abs(x0))
                      if len(members) == 2 else None),
                rhp=bool(x0.real > 0), pair=len(members) == 2,
                phase_deg=(sign * float(sum(_dtheta_deg(x, fu)
                                            for x in members))
                           if fu is not None else None),
                mag_db=(sign * float(sum(_climb_db(x, fgm)
                                         for x in members))
                        if fgm is not None else None),
                count=len(members), roots=tuple(members)))
    entries[1:] = merge_doublets(entries[1:])
    entries[1:] = sorted(
        entries[1:],
        key=lambda e: -(abs(e.phase_deg) if e.phase_deg is not None
                        else abs(e.mag_db) if e.mag_db is not None
                        else 0.0))
    led = MarginLedger(pm_deg=pm, f_unity_hz=fu, gm_db=gm, f_180_hz=fgm,
                       entries=entries, phase_residual_deg=ph_res,
                       mag_residual_db=mg_res)
    try:
        led.stability = stability_certificate(tf, f=f, m=m, acc=acc, fu=fu)
    except Exception:                                # noqa: BLE001
        led.stability = None                         # never fatal
    return attribute_ledger(led, system) if system is not None else led
