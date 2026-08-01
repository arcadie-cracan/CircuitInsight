"""S-B Ben-Or/Tiwari sparse backend: kernel units + backend equivalence.

Same contract as the zpbatch tests: the sparse path must reproduce the
exact-QQ path's expressions bit-for-bit, and every mod-p building block is
checked against a directly computed reference."""
import random
import warnings
from pathlib import Path

import numpy as np
import pytest
import sympy as sp

from circuitinsight.engine import botsparse, interp, zpbatch

FIX = Path(__file__).resolve().parent / "fixtures" / "spectre"
P = zpbatch._primes(1)[0]


# ---------------------------------------------------------------- units
def _lfsr_seq(roots, coeffs, n, p):
    return np.array([sum(c * pow(r, j, p) for c, r in zip(coeffs, roots)) % p
                     for j in range(n)], dtype=np.int64)


def test_bm_recovers_recurrence_length_and_roots():
    rng = random.Random(1)
    roots = [rng.randrange(2, P) for _ in range(7)]
    coeffs = [rng.randrange(1, P) for _ in range(7)]
    seq = _lfsr_seq(roots, coeffs, 30, P)
    C, L = botsparse._bm(seq, P)
    assert L == 7
    # every true root must annihilate Lambda(x) = sum C[i] x^(L-i)
    for r in roots:
        acc = int(C[0])
        for i in range(1, L + 1):
            acc = (acc * r + int(C[i])) % P
        assert acc == 0


def test_bm_zero_sequence():
    C, L = botsparse._bm(np.zeros(16, dtype=np.int64), P)
    assert L == 0


def test_master_poly_and_tvand_roundtrip():
    rng = random.Random(2)
    T = 9
    R = np.array(sorted(rng.sample(range(2, 10_000), T)), dtype=np.int64)
    a = botsparse._master_poly(R, P)
    assert int(a[T]) == 1
    for r in R:                                   # every root annihilates
        acc = 0
        for i in range(T, -1, -1):
            acc = (acc * int(r) + int(a[i])) % P
        assert acc == 0
    # transposed-Vandermonde solve recovers known coefficients
    C_true = np.array([rng.randrange(1, P) for _ in range(T)],
                      dtype=np.int64)
    V = np.empty((T, 2), dtype=np.int64)
    for j in range(T):
        V[j, 0] = sum(int(c) * pow(int(r), j, P)
                      for c, r in zip(C_true, R)) % P
        V[j, 1] = sum(2 * int(c) * pow(int(r), j, P)
                      for c, r in zip(C_true, R)) % P
    got = botsparse._tvand_solve(R, V, P)
    assert got[:, 0].tolist() == C_true.tolist()
    assert got[:, 1].tolist() == (C_true * 2 % P).tolist()


def test_kronecker_support_recovery_synthetic():
    """Full support pipeline on a synthetic sparse polynomial: probe a known
    f at omega^j, BM, candidate scan -- the exponent tuples must come back
    exactly."""
    lens = [3, 4, 2]
    radK = [8, 2, 1]                              # C-order strides
    terms = {(2, 3, 1): 5, (0, 0, 0): 7, (1, 2, 0): 11, (2, 0, 1): 13}
    g = botsparse._proot(P)
    seq = np.array(
        [sum(c * pow(g, (sum(e * K for e, K in zip(exps, radK)) * j)
                     % (P - 1), P) for exps, c in terms.items()) % P
         for j in range(24)], dtype=np.int64)
    C, L = botsparse._bm(seq, P)
    assert L == len(terms)
    cand = botsparse._candidate_values(lens, radK, g, P)
    Es = botsparse._roots_to_exponents(C, L, cand, P)
    got = {tuple(int(i) for i in np.unravel_index(int(E), lens))
           for E in Es}
    assert got == set(terms)


def test_pow_seq():
    got = botsparse._pow_seq(3, 2, 7, 101)
    assert got.tolist() == [pow(3, j, 101) for j in range(2, 7)]


# ------------------------------------------------------ backend equivalence
@pytest.fixture(scope="module")
def _system():
    from circuitinsight.adapters.spectre import SpectreRun
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run = SpectreRun(FIX / "ota5t" / "tb_ota5t.cin.json",
                         FIX / "ota5t" / "psf")
        an = run.analyzer(cap_model="matrix")
        an.match("I0.MN0", "I0.MN1")
        an.match("I0.MP0", "I0.MP1")
    return an


def _solve(an, backend, keep):
    old = interp.PROBE_BACKEND
    interp.PROBE_BACKEND = backend
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            warnings.filterwarnings("error",
                                    message="zp-batch.*")  # no silent fallback
            return an.tf("VIND", "vout", keep=keep, method="interp")
    finally:
        interp.PROBE_BACKEND = old


def test_bot_backend_reproduces_the_qq_path_exactly(_system):
    keep = ["gm_I0_MN0", "gds_I0_MN0", "gm_I0_MP0"]
    h_bot = _solve(_system, "bot", keep)
    h_qq = _solve(_system, "qq", keep)
    assert sp.simplify(h_bot.expr - h_qq.expr) == 0


def test_bot_backend_with_reciprocal_and_cap_axes(_system):
    # a resistorless bench: exercise gds + a matched-pair degree-2 axis
    keep = ["gm_I0_MN0", "gds_I0_MP0", "gm_I0_MP0", "gds_I0_MN0"]
    h_bot = _solve(_system, "bot", keep)
    h_qq = _solve(_system, "qq", keep)
    assert sp.simplify(h_bot.expr - h_qq.expr) == 0
