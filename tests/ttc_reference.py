"""Reference extraction for Hajimiri's time- and transfer-constant analysis.

A. Hajimiri, "Generalized time- and transfer-constant circuit analysis",
IEEE TCAS-I 57(6), pp. 1105-1121, 2010.

The denominator and numerator coefficients are multilinear in the reactive
elements, so every quantity of the method is a coefficient of the exact
transfer function, normalised by D(0):

    tau_i^0      coefficient of x_i in a1
    H^i          coefficient of x_i in the s^1 numerator term, over tau_i^0
    tau_j^i      coefficient of x_i x_j in a2, over tau_i^0

`TTC` extracts them symbolically. tests/test_bandwidth_ttc.py checks the
bandwidth report against it, and the method check in the development
repository's survey builds on it.
"""
import numpy as np
import sympy as sp

from circuitinsight import Analyzer

s = sp.Symbol("s")
PI2 = 2 * np.pi
F = np.logspace(4, 12, 8001)


# ------------------------------------------------------------ netlist helpers

def V(n, p, q="0"):
    return {"name": n, "device_type": "vsource", "terminals": {"p": p, "n": q}}


def R(n, p, q, x):
    return {"name": n, "device_type": "resistor", "terminals": {"p": p, "n": q}, "params": {"r": x}}


def C(n, p, q, x):
    return {"name": n, "device_type": "capacitor", "terminals": {"p": p, "n": q}, "params": {"c": x}}


def L(n, p, q, x):
    return {"name": n, "device_type": "inductor", "terminals": {"p": p, "n": q}, "params": {"l": x}}


def G(n, p, q, cp, cn, x):
    return {"name": n, "device_type": "vccs", "terminals": {"p": p, "n": q, "cp": cp, "cn": cn}, "params": {"gm": x}}


def analyzer(name, inst):
    return Analyzer.from_cin({"cin_version": "0.1", "design": {"name": name, "source": {"kind": "hand"}}, "top": "main",
                              "ground": ["0"], "definitions": {"main": {"ports": [], "instances": inst}}})


# ------------------------------------------------------------ the extraction

class TTC:
    def __init__(self, H, reactive):
        self.H = H
        self.x = {k: H.symbols[k] for k in reactive}
        N, D = sp.fraction(sp.together(H.expr))
        Np, Dp = sp.Poly(sp.expand(N), s), sp.Poly(sp.expand(D), s)
        self.d0 = Dp.coeff_monomial(1)
        self.d = [sp.expand(sp.cancel(Dp.coeff_monomial(s ** k) / self.d0)) for k in range(Dp.degree() + 1)]
        self.n = [sp.expand(sp.cancel(Np.coeff_monomial(s ** k) / self.d0)) for k in range(Np.degree() + 1)]

    def coef(self, poly, *names):
        e = poly
        for k in names:
            e = sp.expand(e).coeff(self.x[k], 1)
        for k in self.x:
            if k not in names:
                e = e.subs(self.x[k], 0)
        return sp.simplify(e)

    def h0(self):
        return sp.simplify(self.n[0]) if self.n else sp.Integer(0)

    def tau0(self, i):
        return sp.simplify(self.x[i] * self.coef(self.d[1], i)) if len(self.d) > 1 else sp.Integer(0)

    def hi(self, i):
        t = self.tau0(i)
        return sp.simplify(self.x[i] * self.coef(self.n[1], i) / t) if len(self.n) > 1 and t != 0 else None

    def tauij(self, j, i):
        """tau_j^i: element j with element i at its infinite value."""
        t = self.tau0(i)
        return sp.simplify(self.x[i] * self.x[j] * self.coef(self.d[2], i, j) / t) if len(self.d) > 2 and t != 0 else None

    def b(self, k):
        return self.d[k] if k < len(self.d) else sp.Integer(0)

    def a(self, k):
        return self.n[k] if k < len(self.n) else sp.Integer(0)


def values(H):
    return {H.symbols[k]: float(v) for k, v in H.values.items()}


def response(H, over=None):
    v = values(H)
    v.update({H.symbols[k]: x for k, x in (over or {}).items()})
    f = sp.lambdify(s, H.expr.xreplace(v), "numpy")
    return f(1j * PI2 * F)


def f3db(h):
    m = np.abs(h)
    i = np.where(m < m[0] / np.sqrt(2))[0]
    return F[i[0]] if len(i) else float("nan")


def par(a, b):
    return a * b / (a + b)


def ce_inst(extra=()):
    """The common-emitter stage of the paper's Examples VII-1 and VII-2."""
    return [V("V1", "vin"), R("R1", "vin", "b", "1k"), R("rp", "b", "0", "2.5k"), C("Cp", "b", "0", "100f"),
            C("Cm", "b", "c", "20f"), G("Gm", "c", "0", "b", "0", "40m"), R("R2", "c", "0", "2k"), C("CL", "c", "0", "200f")] + list(extra)
