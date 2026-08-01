"""S-C reduced rational-function backend: kernel units + equivalence.

The backend interpolates the REDUCED transfer function from det-ratio
values; equivalence with the exact-QQ path holds for the RATIO (num and
den individually differ by the cancelled content -- that is the point)."""
import random
import warnings
from pathlib import Path

import numpy as np
import pytest
import sympy as sp

from circuitinsight.engine import interp, ratfun, zpbatch

FIX = Path(__file__).resolve().parent / "fixtures" / "spectre"
P = zpbatch._primes(1)[0]


# ---------------------------------------------------------------- units
def test_lagrange_interpolates():
    rng = random.Random(3)
    coeffs = [rng.randrange(P) for _ in range(6)]
    ts = [rng.randrange(1, P) for _ in range(6)]
    gs = [ratfun._poly_eval(coeffs, t, P) for t in ts]
    got = ratfun._lagrange(ts, gs, P)
    assert [c % P for c in got] == coeffs


def test_cauchy_minimal_finds_reduced_degrees():
    """Values of a degree-(2,3) reduced rational function: the prescan must
    report exactly (2, 3) even with generous point count."""
    rng = random.Random(4)
    A = [rng.randrange(1, P) for _ in range(3)]      # deg 2
    B = [1] + [rng.randrange(1, P) for _ in range(3)]  # deg 3, B(0)=1
    ts, gs = [], []
    t = 1
    while len(ts) < 12:
        t += 1
        bv = ratfun._poly_eval(B, t, P)
        if bv == 0:
            continue
        ts.append(t)
        gs.append(ratfun._poly_eval(A, t, P) * pow(bv, -1, P) % P)
    dN, dB, b0_ok = ratfun._cauchy_minimal(ts, gs, P)
    assert (dN, dB) == (2, 3)
    assert b0_ok


def test_batch_solve_random_systems():
    rng = np.random.default_rng(5)
    B, u = 40, 7
    X = rng.integers(0, P, size=(B, u)).astype(np.int64)
    A = rng.integers(0, P, size=(B, u, u)).astype(np.int64)
    rhs = np.empty((B, u), dtype=np.int64)
    for i in range(B):                      # exact rhs = A @ x mod P
        rhs[i] = [(int(sum(int(a) * int(x) % P for a, x in
                           zip(A[i, r], X[i]))) % P) for r in range(u)]
    aug = np.concatenate([A, rhs[:, :, None]], axis=2)
    sol, bad = ratfun._batch_solve(aug, P)
    assert not bad.any()
    assert (sol == X).all()


def test_batch_solve_flags_singular():
    A = np.zeros((2, 3, 4), dtype=np.int64)
    A[0] = np.eye(3, 4, dtype=np.int64)
    A[1, 0, :] = [1, 2, 3, 4]
    A[1, 1, :] = [2, 4, 6, 8]               # dependent rows -> singular
    A[1, 2, :] = [0, 0, 1, 1]
    _, bad = ratfun._batch_solve(A, P)
    assert not bad[0] and bad[1]


def test_degree_buckets():
    lens = [3, 2, 2]
    radK = [4, 2, 1]
    tot = ratfun._degree_buckets(lens, radK)
    assert tot.shape[0] == 12
    assert tot[0] == 0
    assert tot[11] == 2 + 1 + 1             # (2,1,1)


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


def test_ratfun_matches_qq_ratio_exactly(_system):
    keep = ["gm_I0_MN0", "gds_I0_MN0", "gm_I0_MP0"]
    h_r = _solve(_system, "ratfun", keep)
    h_q = _solve(_system, "qq", keep)
    assert sp.simplify(h_r.expr - h_q.expr) == 0


def test_ratfun_denominator_is_reduced(_system):
    """The returned den must not be larger than the raw Cramer den."""
    keep = ["gm_I0_MN0", "gds_I0_MN0", "gm_I0_MP0", "gds_I0_MP0"]
    h_r = _solve(_system, "ratfun", keep)
    h_q = _solve(_system, "qq", keep)
    assert sp.simplify(h_r.expr - h_q.expr) == 0
    _, dr = sp.fraction(sp.together(h_r.expr))
    _, dq = sp.fraction(sp.together(h_q.expr))
    assert len(sp.Add.make_args(sp.expand(dr))) <= \
        len(sp.Add.make_args(sp.expand(dq)))
