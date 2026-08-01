"""S-A zp-batch backend: kernel unit tests + backend equivalence.

The kernel is exact-or-nothing: batched division-free determinants must
match exact determinants reduced mod p on every matrix INCLUDING zero-pivot
and singular ones, and the full pipeline must reproduce the exact-QQ path's
expressions bit-for-bit (the integration seam accepts zp results only
through the same off-grid exact probe self-check, so equivalence here is
the whole contract)."""
import random
import warnings
from pathlib import Path

import numpy as np
import pytest
import sympy as sp

from circuitinsight.engine import interp, zpbatch

FIX = Path(__file__).resolve().parent / "fixtures" / "spectre"
P26 = zpbatch._primes(1)[0]                     # largest 26-bit prime


# ------------------------------------------------------------ batched det
def _exact_dets_modp(mats, p):
    return [int(sp.Matrix(m.tolist()).det()) % p for m in mats]


@pytest.mark.parametrize("n", [1, 2, 3, 5, 8])
def test_batch_det_matches_exact(n):
    rng = np.random.default_rng(n)
    A = rng.integers(-9, 10, size=(60, n, n)).astype(np.int64)
    got = zpbatch._batch_det_modp(A.copy(), P26)
    assert got.astype(np.int64).tolist() == _exact_dets_modp(A, P26)


def test_batch_det_zero_pivots_and_singulars():
    """Zero leading pivots force the vectorized row swap; duplicated rows
    force det = 0 through the dead-matrix path -- a correct value, not an
    error."""
    rng = np.random.default_rng(7)
    A = rng.integers(-5, 6, size=(40, 4, 4)).astype(np.int64)
    A[::3, 0, 0] = 0                            # swap path
    A[::5, 2, :] = A[::5, 1, :]                 # singular
    A[7] = 0                                    # fully zero
    got = zpbatch._batch_det_modp(A.copy(), P26)
    assert got.astype(np.int64).tolist() == _exact_dets_modp(A, P26)


def test_batch_det_values_already_reduced():
    """Residue-range inputs (the real case): values in [0, p)."""
    rng = np.random.default_rng(3)
    A = rng.integers(0, P26, size=(30, 6, 6)).astype(np.int64)
    got = zpbatch._batch_det_modp(A.copy(), P26)
    assert got.astype(np.int64).tolist() == _exact_dets_modp(A, P26)


# ------------------------------------------------------- CRT lift helpers
def test_ratrec_roundtrip():
    import gmpy2
    m = 1
    for q in zpbatch._primes(4):
        m *= q
    for a, b in [(3, 7), (-123456, 991), (0, 1), (10**9, 1), (-1, 2)]:
        u = a * gmpy2.invert(b, m) % m
        assert zpbatch._ratrec(int(u), m) == (a, b)


def test_ratrec_refuses_when_modulus_too_small():
    # 1e9/1 cannot be told apart from garbage under a tiny modulus
    assert zpbatch._ratrec(123456789 % 101, 101) != (123456789, 1)


def test_powmod_arr():
    base = np.array([0, 1, 2, 12345, P26 - 1], dtype=np.int64)
    for e in (0, 1, 2, 31, 1000, P26 - 2):
        want = [pow(int(b), e, P26) for b in base]
        assert zpbatch._powmod_arr(base, e, P26).astype(
            np.int64).tolist() == want


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


def _solve(an, backend):
    old = interp.PROBE_BACKEND
    interp.PROBE_BACKEND = backend
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            warnings.filterwarnings("error",
                                    message="zp-batch.*")  # no silent fallback
            return an.tf("VIND", "vout",
                         keep=["gm_I0_MN0", "gds_I0_MN0", "gm_I0_MP0"],
                         method="interp")
    finally:
        interp.PROBE_BACKEND = old


def test_zp_backend_reproduces_the_qq_path_exactly(_system):
    """The whole contract: identical rational expressions, not merely close.
    Warnings are promoted to errors above, so this also asserts the zp path
    genuinely ran (a fallback to the QQ grid would warn)."""
    h_zp = _solve(_system, "zp")
    h_qq = _solve(_system, "qq")
    assert sp.simplify(h_zp.expr - h_qq.expr) == 0


def test_qq_backend_still_forceable(_system):
    h = _solve(_system, "qq")
    assert h.expr.free_symbols                    # sanity: symbolic result


def test_bot_progress_moves_and_cancel_is_uncatchable(_system):
    """Two halves of one field report. The bot discovery phase reported
    progress(0, J) -- a bar frozen at zero while J doubled -- so a
    minute-long solve showed 'evaluating 0/8192' throughout; done must
    move. And a cancel raised from the progress callback must reach the
    caller as a BaseException: the backend fallback machinery catches
    Exception to fall back to slower paths, and an Exception-cancel was
    swallowed there and turned into a silent serial RE-RUN of the
    abandoned solve. (That the GUI's _Cancelled IS a BaseException is
    asserted in the GUI suite, which skips without Qt.)"""
    calls = []
    old = interp.PROBE_BACKEND
    interp.PROBE_BACKEND = "bot"
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _system.tf("VIND", "vout",
                       keep=["gm_I0_MN0", "gds_I0_MN0", "gm_I0_MP0"],
                       method="interp",
                       progress=lambda d, t: calls.append((d, t)))
    finally:
        interp.PROBE_BACKEND = old
    assert calls and max(d for d, _t in calls) > 0

    class Stop(BaseException):
        pass

    def cancelling(d, t):
        raise Stop

    interp.PROBE_BACKEND = "bot"
    try:
        with pytest.raises(Stop):     # not swallowed into a fallback
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _system.tf("VIND", "vout",
                           keep=["gm_I0_MN0", "gds_I0_MN0", "gds_I0_MP0"],
                           method="interp", progress=cancelling)
    finally:
        interp.PROBE_BACKEND = old


def test_spread_is_not_the_fallback_predictor():
    """A disproven hypothesis, kept as a guard so it is not re-tried
    blind. The fc keep set fell back with "no stable rational
    reconstruction after 64 primes"; the spread at that call was 11.0 --
    BELOW ota5t's 16.1, which reconstructs fine. So the selector must
    not gate on spread: reconstruction size tracks the determinant's
    coefficients (products of ~n entries), not the entries' range."""
    from circuitinsight.engine.interp import resolve_backend

    big = 200_000
    assert resolve_backend(big, 11.0) == "bot"     # the FAILING case
    assert resolve_backend(big, 16.1) == "bot"     # the PASSING bench
    assert resolve_backend(big, 30.0) == "bot"     # spread is not a gate
    assert resolve_backend(1_000, 30.0) == "qq"    # size still decides


def test_prime_ceiling_is_high_enough_for_real_coefficients():
    """The measured fix for the field fallback. Rational reconstruction
    needs modulus proportional to the exact coefficients' SIZE (sums of
    products of ~n matrix entries), which a 30x30 hybrid grid drives far
    past a few dozen 62-bit primes: an fc keep set failed at 64, did all
    its probing for nothing, and re-ran the whole grid densely. At 256
    it converges -- and faster than the fallback route (184 s vs 232 s).
    The ceiling is free for easy cases: the loop stops as soon as two
    lifts agree."""
    from circuitinsight.engine import botsparse, ratfun, zpbatch

    for mod in (botsparse, ratfun, zpbatch):
        assert mod._MAX_PRIMES >= 256
