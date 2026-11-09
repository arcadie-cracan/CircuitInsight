"""The conductance a capacitor sees at its own terminals, symbolically.

When a pole's CAPACITANCE side reads but its conductance side does not,
the reconstruction names the remainder as a single quantity G_eff and
stops. The missing algebra is not really missing: it is the
driving-point conductance at that capacitor's port, and the circuit
can be asked for it directly instead of having it inferred from
second-order couplings, which on a densely connected cascode is badly
posed.

At s = 0 every capacitor is an open, so the MNA matrix collapses to the
DC conductance matrix G with no s at all. That is what makes this
cheap beside a loop-gain solve: the s-degree drives the interpolation
grid, and here it is zero. The driving-point impedance at a port u is

    Z = u^T G^-1 u = -det([[G, u], [u^T, 0]]) / det(G)

so the answer is two determinants of one system, symbolic only in the
handful of conductances the root's sensitivity list names.

This is the open-circuit time constant of that capacitor, so it is the
right conductance exactly when the pole IS that node's time constant.
That is not assumed: the expression is evaluated at the operating
point and checked against the value the identity already fixed.
"""
from __future__ import annotations

import dataclasses
import time

import sympy as sp

from ..engine.mna import S, _det, hybrid_split


def port_conductance(system, u, keep, *, open_branch=None):
    """(num, den, value) of the DC driving-point conductance at `u`,
    the conductance being num/den.

    `u` is the port's incidence vector over the system's unknowns --
    exactly the `a` vector of the capacitor's own stamp. `keep` names
    the conductances left symbolic; everything else takes its
    operating-point value. Returns (None, None, None) when the port is
    degenerate (an open, or a short to a fixed node).

    The ratio is deliberately NOT cancelled. Both parts are already
    determinants, and forming the canonical rational is what actually
    costs: measured on the folded cascode, two kept symbols spent 4.5 s
    and 3.4 s on the determinants and 0.06 s on the cancel, while four
    kept symbols never finished the cancel at all. Every consumer here
    wants either a value -- which substitutes into both parts -- or a
    printed form, and neither needs a canonical one."""
    subs, _ = hybrid_split(system, list(keep))
    G = system.A.xreplace(subs).xreplace({S: sp.Integer(0)})
    # EXACT entries: an incidence vector is 0 and +-1, and a single
    # Float here drags the whole determinant off sympy's rational
    # domain into general expressions -- measured 42.7 s against 0.03 s
    # on a seventeen-row system with one symbol kept
    uv = sp.Matrix([[sp.nsimplify(x, rational=True)] for x in u])
    if open_branch is not None:
        # OPEN THE LOOP at the probe, then HOLD the far side. The
        # branch row is the constraint v_p = v_q and its column
        # carries i_b into the node equations, so dropping both
        # removes the branch and the signal can no longer come round.
        # That alone leaves the amplifier input floating with no DC
        # path -- the matrix goes singular -- so the probe node that
        # is NOT the port under test is held, which is the ordinary
        # break-the-loop-and-terminate move. Measured on the folded
        # cascode: closed 402 uS (522x too big), cut alone singular,
        # cut and held 0.663 uS against the 0.770 uS the identity
        # fixes.
        from .loopgain import _probe_nodes

        k = system.branch_index.get(open_branch)
        if k is None:
            raise KeyError(f"no branch {open_branch!r} to open")
        p, q = _probe_nodes(system, open_branch)
        support = {i for i in range(len(uv)) if uv[i, 0] != 0}
        far = [n for n in (p, q) if n not in support]
        drop = {k} | set(far[:1] or [p])
        keep_i = [i for i in range(G.rows) if i not in drop]
        G = G[keep_i, keep_i]
        uv = uv[keep_i, :]
    if all(x == 0 for x in uv):
        return None, None, None
    bordered = G.row_join(uv).col_join(
        uv.T.row_join(sp.Matrix([[0]])))
    dG = _det(G)
    dB = _det(bordered)
    if dB == 0 or dG == 0:
        return None, None, None
    # Z = -dB/dG, so the conductance is -dG/dB
    num, den = -dG, dB
    val = None
    try:
        sub = {system.symbols[k]: sp.Float(system.values[k])
               for k in keep if k in system.symbols}
        dv = complex(sp.N(den.xreplace(sub)))
        if dv != 0:
            val = float((complex(sp.N(num.xreplace(sub))) / dv).real)
    except Exception:                              # noqa: BLE001
        val = None
    return num, den, val


def branch_instances(system, u, stamps, *, open_branch=None) -> dict:
    """{symbol: the instance of it that acts on THIS port}.

    A matched group shares one symbol, named after one member. Printing
    that name against a port the member is not in points a reader at
    the wrong transistor: on the folded cascode the output branch runs
    through PM14 and PM15, while the shared symbols are spelled PM12
    and PM13 — right values, wrong devices, and a designer tracing the
    path finds nothing there.

    Which instance acts here is decidable. Each stamp is rank-one, so
    the port quantity u^T G^-1 u moves with (u^T M a)(b^T M u): one
    inverse, then an inner product per stamp. The LOOP MUST BE OPEN
    first — with feedback closed every element influences every other
    and matched twins in symmetric positions are indistinguishable
    (measured: 9.9e3 against 8.6e3, and the wrong one wins). Opened,
    the same pair separates by three orders of magnitude.
    """
    import numpy as _np

    from ..engine.mna import S, hybrid_split
    from .loopgain import _probe_nodes

    subs, _ = hybrid_split(system, [])
    A = _np.array(sp.matrix2numpy(
        system.A.xreplace(subs).xreplace({S: 0}).evalf(40)), dtype=complex)
    uv = _np.asarray(u, dtype=complex).ravel()
    keep_i = list(range(A.shape[0]))
    if open_branch is not None:
        k = system.branch_index.get(open_branch)
        if k is None:
            raise KeyError(f"no branch {open_branch!r} to open")
        p, q = _probe_nodes(system, open_branch)
        support = {i for i in range(len(uv)) if uv[i] != 0}
        far = [n for n in (p, q) if n not in support]
        drop = {k} | set(far[:1] or [p])
        keep_i = [i for i in keep_i if i not in drop]
        A = A[_np.ix_(keep_i, keep_i)]
        uv = uv[keep_i]
    try:
        M = _np.linalg.inv(A)
    except _np.linalg.LinAlgError:                   # pragma: no cover
        return {}
    Mu, uM = M @ uv, uv @ M
    out = {}
    for name, sts in stamps.items():
        best = None
        for _kind, a, b, inst in sts:
            if a is None or b is None:
                continue
            av = _np.asarray(a, dtype=complex).ravel()[keep_i]
            bv = _np.asarray(b, dtype=complex).ravel()[keep_i]
            w = abs(complex(uM @ av) * complex(bv @ Mu))
            if best is None or w > best[0]:
                best = (w, inst)
        if best is not None and best[0] > 0:
            out[name] = best[1]
    return out


def respell(name: str, inst: str | None) -> str:
    """`gds_I0_PM12` acting through I0.PM14 is spelled gds_I0_PM14.

    The same move the sensitivity views make (_ms_local_name): the
    shared symbol keeps its identity everywhere it is a JOIN KEY, and
    wears the hot device's name where it is being read as physics.
    """
    if not inst or "_" not in name:
        return name
    dev = inst.replace(".", "_")
    head, _, tail = name.partition("_")
    if tail == dev:
        return name
    return f"{head}_{dev}"


def eet_symbols(g, stamps, *, floor: float = 1e-3, cap: int = 16):
    """The symbol list for the EET -- deliberately NOT the one the
    determinants get.

    `conductance_for_aggregate` caps its keep set hard because the
    determinant cost explodes: on the folded cascode, 22 s at three
    symbols and 562 s at five. The EET costs milliseconds, so it wants
    every conductance that carries any weight. Starved of symbols it
    does not fail loudly -- branch_form simply cannot NAME the branch
    and reports it unnamed, which reads like a limit of the method
    rather than a malformed question. That cascode branch is
    gds_NM10 + gds_PM11 + gds_NM13 over gm_NM14 + gmbs_NM14, and none
    of those survives a three-symbol cap.
    """
    from .reconstruct import _CLASS

    gr = {n: abs(complex(v).real) for n, v in g.items()}
    return tuple(n for n in sorted(gr, key=lambda n: -gr[n])
                 if gr[n] >= floor
                 and any(_CLASS.get(st[0], ("",))[0] == "conductive"
                         for st in stamps.get(n, ())))[:cap]


def conductance_for_aggregate(system, rec, stamps, g, *, probe=None,
                              max_keep: int = 5, floor: float = 0.02,
                              rel_tol: float = 0.25):
    """Fill a reconstruction's G_eff with the actual port algebra.

    Picks the port from the capacitor block that certified, keeps the
    root's own conductance movers symbolic, and CHECKS the result
    against the value the identity fixed: agreement means the pole
    really is that node's time constant and the expression stands;
    disagreement means it is not, and the aggregate is left as it
    was rather than dressed in algebra that does not apply."""
    from .reconstruct import _CLASS

    agg = rec.aggregate
    if not agg or agg["unit"] != "S":
        return None
    caps = [b for b in agg["blocks"] if b.exponent < 0]
    if not caps:
        return None
    lead = max((n for b in caps for n in b.members),
               key=lambda n: abs(complex(g.get(n, 0)).real))
    sts = stamps.get(lead) or ()
    if not sts:
        return None
    gr = {n: complex(v).real for n, v in g.items()}
    keep = [n for n in sorted(gr, key=lambda n: -abs(gr[n]))
            if abs(gr[n]) >= floor
            and any(_CLASS.get(st[0], ("",))[0] == "conductive"
                    for st in stamps.get(n, ()))][:max_keep]
    if not keep:
        return None
    t0 = time.time()
    num, den, val = port_conductance(system, sts[0][1], keep,
                                     open_branch=probe)
    if num is None or not val:
        return None
    ok = abs(val / agg["value"] - 1.0) <= rel_tol
    return {"port": lead, "keep": tuple(keep), "num": num, "den": den,
            "value": float(val), "matches": bool(ok),
            "ratio": float(val / agg["value"]),
            "seconds": time.time() - t0}


def eet_at_port(system, u, symbol, *, open_branch=None, degree=None):
    """The EET decomposition of a port conductance in ONE element.

    Every element enters through a rank-one stamp, so det(G) and the
    bordered determinant are each POLYNOMIAL in it -- degree one for a
    lone element, degree k for a symbol k matched devices share. So
    rather than building a symbolic determinant and cancelling it,
    both are evaluated NUMERICALLY at a few scaled values and
    interpolated exactly. That is the whole cost: a couple of numeric
    determinants per degree, against a symbolic determinant whose
    cancellation would not finish.

    What comes back is the Middlebrook reading of the port:

        Y(g) = Y(0) (1 + g/g_n) / (1 + g/g_d)

    with Y(0) the conductance with the element removed, g_n where the
    numerator nulls and g_d where the denominator does. A degree-one
    element gives exactly that; a matched symbol gives the same form
    with several roots a side, and they are all returned."""
    import numpy as _np

    k = degree or max(1, system.stamp_counts.get(symbol, 1))
    base = float(system.values[symbol])
    lam = [sp.Rational(1, 2) + sp.Rational(i, 2) for i in range(k + 1)]
    xs, ns, ds = [], [], []
    for L in lam:
        sysl = dataclasses.replace(
            system, values={**system.values,
                            symbol: float(base * float(L))})
        num, den, _v = port_conductance(sysl, u, [],
                                        open_branch=open_branch)
        if num is None:
            return None
        xs.append(float(base * float(L)))
        ns.append(complex(num).real)
        ds.append(complex(den).real)
    # exact-degree interpolation in the element's own value. A
    # degenerate element -- one this port cannot see at all -- gives a
    # flat or vanishing determinant and the fit will not converge;
    # that is a "no reading", not an error to propagate.
    if not all(_np.isfinite(v) for v in ns + ds):
        return None
    try:
        pn = _np.polyfit(xs, ns, k)
        pd = _np.polyfit(xs, ds, k)
    except _np.linalg.LinAlgError:
        return None
    y0 = (_np.polyval(pn, 0.0) / _np.polyval(pd, 0.0)
          if _np.polyval(pd, 0.0) else None)
    return {"symbol": symbol, "degree": k, "value": base,
            "num_poly": pn, "den_poly": pd,
            "y_removed": y0,
            "y_shorted": (pn[0] / pd[0]) if pd[0] else None,
            "g_null": sorted(float(r.real) for r in _np.roots(pn)
                             if abs(r.imag) < 1e-9 * max(abs(r), 1)),
            "g_pole": sorted(float(r.real) for r in _np.roots(pd)
                             if abs(r.imag) < 1e-9 * max(abs(r), 1))}


def _y_with(system, u, scales, open_branch):
    """The port conductance with elements rescaled (0 removes one)."""
    sysl = dataclasses.replace(
        system, values={**system.values,
                        **{k: float(system.values[k] * v)
                           for k, v in scales.items()}})
    _n, _d, val = port_conductance(sysl, u, [], open_branch=open_branch)
    return val


def nested_eet(system, u, symbols, *, open_branch=None, tol=0.05):
    """Peel a port conductance into PARALLEL BRANCHES, then read each
    branch on its own.

    The first level asks what the port keeps when one element is
    removed. Where two removals leave conductances that add up to the
    whole, the port is two parallel branches and each element gates
    one of them -- which is what a cascoded output is. The second
    level then re-reads each branch with the other one removed, so the
    element that DIVIDES it (the cascode's gm) shows up as a branch
    conductance falling with that element rather than buried in a sum.

    Every step is numeric determinants on a rescaled system, so a
    whole decomposition costs what one symbolic determinant would not
    even begin to."""
    y0 = _y_with(system, u, {}, open_branch)
    if not y0:
        return None
    rem = {}
    for s in symbols:
        v = _y_with(system, u, {s: 0.0}, open_branch)
        if v is not None:
            rem[s] = v
    split = None
    for i, a in enumerate(symbols):
        for b in symbols[i + 1:]:
            if a not in rem or b not in rem:
                continue
            if abs((rem[a] + rem[b]) / y0 - 1.0) <= tol:
                err = abs((rem[a] + rem[b]) / y0 - 1.0)
                if split is None or err < split["error"]:
                    split = {"gates": (a, b), "error": err,
                             "branch": {a: y0 - rem[a],
                                        b: y0 - rem[b]}}
    out = {"total": y0, "removed": rem, "split": split, "branches": {}}
    if split:
        a, b = split["gates"]
        for gate, other in ((a, b), (b, a)):
            sysl = dataclasses.replace(
                system, values={**system.values, other: 0.0})
            reads = {}
            for s in symbols:
                if s in (gate, other):
                    continue
                r = eet_at_port(sysl, u, s, open_branch=open_branch)
                if r and r["y_removed"]:
                    reads[s] = r
            out["branches"][gate] = {
                "conductance": split["branch"][gate], "reads": reads}
    return out


def branch_form(info, gate, values, *, tol=1e-3, sum_tol=0.12,
                residual_tol=0.05, insts=None):
    """The printed form of one branch, read from its EET signature.

    Within a branch each element declares its role by what removing or
    shorting it does. Driving the branch to zero when SHORTED means
    the branch divides by it; when REMOVED, that it multiplies it;
    leaving it untouched, that it belongs to another branch.

    Divisors that sit in a SUM are recovered from the EET pole rather
    than written as a product. A branch going as 1/x has its pole at
    x = 0; one going as 1/(x + c) has it at x = -c, so each divisor
    implies a total, value + |pole|, and divisors of ONE sum all imply
    the SAME total. On the folded cascode gm_PM14 implies 139.6 + 46.2
    and gmbs_PM14 implies 30.5 + 155.4, both 185.8 uS -- that
    agreement is what proves them a sum, and what a product form would
    have written as gm*gmbs instead. Whatever the named members do not
    account for is reported as a residual rather than absorbed.

    The dimension is then checked strictly: a conductance branch must
    net one conductance, counting each sum as the single factor it is.
    """
    def _residual(rest):
        # the unnamed remainder of a sum is a CONDUCTANCE -- this is a
        # DC branch -- and it is read by a designer, so it wears
        # engineering notation and its unit: "+15.6 uS", not "+1.56e-05"
        from ..units import eng
        return ("+" if rest > 0 else "-") + eng(abs(rest), "S")

    def _spell(m):
        # a matched group's shared symbol wears the name of the device
        # it acts through HERE; without this the folded cascode's
        # output branch reads PM12/PM13 for transistors that are
        # PM14/PM15, which sends a reader to the wrong devices
        return respell(m, (insts or {}).get(m))

    y = info["conductance"]

    def _group(items, key):
        """Members implying the SAME total sit in one sum."""
        out = []
        for s, tot in items:
            for g in out:
                if abs(tot / g["total"] - 1.0) <= sum_tol:
                    g["members"].append(s)
                    n = len(g["members"])
                    g["total"] = (g["total"] * (n - 1) + tot) / n
                    break
            else:
                out.append({"total": tot, "members": [s]})
        return out

    ups, downs = [(gate, abs(values[gate]))], []
    for s, r in info["reads"].items():
        removed, shorted = r["y_removed"], r["y_shorted"]
        if removed is None or shorted is None:
            continue
        if abs(removed - y) <= tol * abs(y):
            continue                       # not in this branch
        if abs(shorted) <= tol * abs(y):
            poles = [-p for p in r["g_pole"] if p < 0]
            downs.append((s, abs(values[s]) + (min(poles) if poles
                                               else 0.0)))
        else:
            # a NUMERATOR sum reads the same way through the null:
            # a branch going as (x + c) nulls at x = -c, so each
            # member implies value + |null| and members of one sum
            # imply the same total. On the folded cascode gds_PM11,
            # gds_NM10 and gds_NM13 each imply 47.8 uS and their
            # values sum to exactly that -- the folded node's own
            # conductance, which removing any ONE of them cannot zero.
            nulls = [-p for p in r["g_null"] if p < 0]
            ups.append((s, abs(values[s]) + (min(nulls) if nulls
                                             else 0.0)))
    num_g, groups = _group(ups, "null"), _group(downs, "pole")
    num = [m for g in num_g for m in g["members"]]
    if not num or len(num_g) - len(groups) != 1:
        return None
    prod = 1.0
    for g in num_g:
        prod *= g["total"] or 1.0
    parts, tops = [], []
    for g in num_g:
        named = sum(abs(values[m]) for m in g["members"])
        rest = g["total"] - named
        body = "+".join(_spell(m) for m in g["members"])
        if abs(rest) > residual_tol * abs(g["total"]):
            body += _residual(rest)
        tops.append(f"({body})" if "+" in body else body)
    for grp in groups:
        named = sum(abs(values[m]) for m in grp["members"])
        rest = grp["total"] - named
        body = "+".join(_spell(m) for m in grp["members"])
        if abs(rest) > residual_tol * abs(grp["total"]):
            body += _residual(rest)
        parts.append(f"({body})" if len(grp["members"]) > 1 or
                     "+" in body else body)
        prod /= grp["total"] or 1.0
    text = "*".join(tops) + ("/" + "*".join(parts) if parts else "")
    return {"num": tuple(num), "den": tuple(parts), "text": text,
            "sums": tuple(g["total"] for g in groups),
            "constant": (y / prod) if prod else None}
