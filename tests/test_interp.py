"""Multilinear-interpolation solver: exact equivalence with the direct path.

The interp path must produce SYMBOLICALLY IDENTICAL transfer functions —
it is reconstruction, not approximation (docs/multilinear-solver-plan.md).
"""
import warnings
from pathlib import Path

import pytest
import sympy as sp

from circuitinsight import Analyzer
from circuitinsight.adapters.spectre import SpectreRun
from circuitinsight.analysis import compare_tf

GOLDEN = Path(__file__).resolve().parent / "golden" / "circuits"
OTA = Path(__file__).resolve().parent / "fixtures" / "spectre" / "ota5t"


def assert_equivalent(an, inp, out, keep):
    H_d = an.tf(inp, out, keep=keep, method="direct")
    H_i = an.tf(inp, out, keep=keep, method="interp")
    diff = sp.cancel(sp.together(H_d.expr - H_i.expr))
    assert sp.simplify(diff) == 0, f"paths disagree for keep={keep}"
    return H_i


def test_cs_amp_with_kept_resistor():
    # exercises the reciprocal (u = 1/R) path: RL kept symbolic
    an = Analyzer.from_cin(GOLDEN / "cs_amp.cin.json")
    H = assert_equivalent(an, "V1", "vout", keep=["M1", "RL", "CL"])
    # sanity: textbook dc gain still exact
    y = H.symbols
    dc = sp.simplify(H.expr.subs(sp.Symbol("s"), 0))
    expected = -y["RL"] * y["gm_M1"] / (y["RL"] * y["gds_M1"] + 1)
    assert sp.simplify(dc - expected) == 0


def test_ota5t_golden_matched_pairs():
    # matched symbols -> degree-2 axes in the tensor grid
    an = Analyzer.from_cin(GOLDEN / "ota5t.cin.json")
    an.match("M1", "M2")
    an.match("M3", "M4")
    assert_equivalent(an, "V1", "vout", keep=["M1", "M3", "CL"])


def test_transimpedance_kept_resistor():
    an = Analyzer.from_cin({
        "cin_version": "0.1", "top": "main", "ground": ["0"],
        "definitions": {"main": {"ports": [], "instances": [
            {"name": "I1", "device_type": "isource",
             "terminals": {"p": "0", "n": "vout"}},
            {"name": "R1", "device_type": "resistor",
             "terminals": {"p": "vout", "n": "0"}, "params": {"r": "1k"}},
            {"name": "C1", "device_type": "capacitor",
             "terminals": {"p": "vout", "n": "0"}, "params": {"c": "1n"}},
        ]}}})
    H = assert_equivalent(an, "I1", "vout", keep=["R1", "C1"])
    y = H.symbols
    s = sp.Symbol("s")
    assert sp.simplify(
        sp.cancel(H.expr) - y["R1"] / (1 + s * y["R1"] * y["C1"])) == 0


def test_real_bench_with_balun_and_switches():
    # equivalence proxy on the full testbench: the interp result must match
    # the simulator AC exactly as well as the exact model does
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        run = SpectreRun(OTA / "tb_ota5t.cin.json", OTA / "psf")
        an = run.analyzer(cap_model="matrix")      # SKY130 lumped errs ~28 dB
        an.match("I0.MN0", "I0.MN1")
        an.match("I0.MP0", "I0.MP1")
        H = an.tf("VIND", "vout",
                  keep=["gm_I0_MN0", "gds_I0_MN0", "gm_I0_MP0", "gds_I0_MP0"],
                  method="interp")
    ac = run.ac()
    # interp must reproduce the sim as well as the exact model does (charge matrix)
    r = compare_tf(H, ac.freq, ac.wave("vout"), ac.wave("vin_dm"))
    assert r.worst_mag_db < 0.1 and r.worst_phase_deg < 3.5


def test_solve_tf_batch_matches_individual_solves():
    """The multi-output (Tan-Shi cofactor) batch: same matrix, different
    excitations and outputs -- including a branch-current output -- must
    reproduce the per-output solves exactly, direct and hybrid."""
    from circuitinsight.engine.mna import build_mna, solve_tf, solve_tf_batch

    an = Analyzer.from_cin(GOLDEN / "cs_amp.cin.json")
    prims, gnd = an.primitives, an.flat.ground
    sys_v = build_mna(prims, gnd, "V1")
    tasks = [(sys_v, "vout"),
             (sys_v, sys_v.branch_index["V1"]),
             (sys_v, "vout")]
    for keep in (["M1", "RL"], []):
        batch = solve_tf_batch(tasks, keep=keep)
        for (system, out), tf in zip(tasks, batch):
            ref = solve_tf(system, out, keep=keep)
            assert sp.simplify(tf.expr - ref.expr) == 0


def test_solve_tf_batch_rejects_mismatched_matrices():
    from circuitinsight.engine.mna import MnaError, build_mna, solve_tf_batch

    an1 = Analyzer.from_cin(GOLDEN / "cs_amp.cin.json")
    s1 = build_mna(an1.primitives, an1.flat.ground, "V1")
    an2 = Analyzer.from_cin(GOLDEN / "ota5t.cin.json")
    s2 = build_mna(an2.primitives, an2.flat.ground, "V1")
    with pytest.raises(MnaError):
        solve_tf_batch([(s1, "vout"), (s2, "vout")], keep=[])


def test_zero_numerator_survives_the_fast_path():
    # two RC sections sharing only ground: H(in -> other section) == 0.
    # A zero numerator empties the coefficient tensor mid-transform, which
    # used to crash _transform_axis (StopIteration) instead of returning 0.
    an = Analyzer.from_cin({
        "cin_version": "0.1", "top": "main", "ground": ["0"],
        "definitions": {"main": {"ports": [], "instances": [
            {"name": "V1", "device_type": "vsource",
             "terminals": {"p": "a", "n": "0"}},
            {"name": "R1", "device_type": "resistor",
             "terminals": {"p": "a", "n": "0"}, "params": {"r": "1k"}},
            {"name": "R2", "device_type": "resistor",
             "terminals": {"p": "b", "n": "0"}, "params": {"r": "1k"}},
            {"name": "C2", "device_type": "capacitor",
             "terminals": {"p": "b", "n": "0"}, "params": {"c": "1n"}},
        ]}}})
    H = an.tf("V1", "b", keep=["R2", "C2"], method="interp")
    assert sp.simplify(H.expr) == 0


def test_empty_keep_falls_back_to_direct():
    an = Analyzer.from_cin(GOLDEN / "cs_amp.cin.json")
    H = an.tf("V1", "vout", keep=[], method="interp")
    assert not (H.expr.free_symbols - {sp.Symbol("s")})


def test_unknown_method_rejected():
    from circuitinsight.engine.mna import MnaError
    an = Analyzer.from_cin(GOLDEN / "cs_amp.cin.json")
    with pytest.raises(MnaError, match="method"):
        an.tf("V1", "vout", keep=["M1"], method="magic")


def test_all_numeric_sweeps_s_instead_of_one_symbolic_determinant():
    """keep=[] used to take a single QQ[s] determinant on the claim that it
    "is already optimal". It is not: a symbolic determinant carries
    polynomial arithmetic through every elimination step. Sampling s and
    reconstructing gives the SAME polynomials (checked by cross-multiplied
    identity against the direct path), reports progress, and is an order of
    magnitude faster on real circuits."""
    from circuitinsight.engine import interp as I

    an = Analyzer.from_cin(GOLDEN / "miller_ota.cin.json")
    seen = []
    h = an.tf("V1", "vout", keep=[], progress=lambda d, t: seen.append((d, t)))
    ref = an.tf("V1", "vout", keep=[], method="direct")

    n, d = h.num_den
    nr, dr = ref.num_den
    assert sp.expand(n.as_expr() * dr.as_expr()
                     - d.as_expr() * nr.as_expr()) == 0
    assert seen and seen[-1][0] == seen[-1][1]     # ran to completion
    assert I.LAST_SOLVE["backend"] == "qq-s"
    assert I.LAST_SOLVE["n_dense_dets"] == 2 * I.LAST_SOLVE["L"]


def test_s_degree_bound_covers_the_true_degree():
    """The bound is what removes the probe determinant, so it must never be
    short: every MNA entry is affine in s, so deg det <= the number of rows
    carrying an s term, and L = bound + 1 samples reconstruct exactly."""
    from circuitinsight.engine.interp import _s_degree_bound
    from circuitinsight.engine.mna import _det, hybrid_split

    for name in ("cs_amp.cin.json", "miller_ota.cin.json", "ota5t.cin.json"):
        an = Analyzer.from_cin(GOLDEN / name)
        system = an.system("V1")
        subs, kept = hybrid_split(system, [])
        assert not kept
        A = system.A.xreplace(subs)
        bound = _s_degree_bound(A)
        true = sp.Poly(sp.expand(_det(A)), sp.Symbol("s")).degree()
        assert bound >= true, f"{name}: bound {bound} < true degree {true}"


def test_all_numeric_solve_is_cancellable():
    """The other half of the point: a progress callback that raises must
    abandon the solve promptly, which sympy's det() can never allow."""
    class _Stop(Exception):
        pass

    an = Analyzer.from_cin(GOLDEN / "miller_ota.cin.json")

    def cb(done, total):
        raise _Stop

    with pytest.raises(_Stop):
        an.tf("V1", "vout", keep=[], progress=cb)


def test_parallel_s_sweep_matches_serial_exactly():
    """Above _S_SWEEP_PARALLEL_MIN dets the all-numeric sweep fans out to
    the worker pool; the reconstruction must be bit-identical to the
    serial path. (Under a full suite run Qt is already imported and the
    pool guard forces serial — the comparison still holds, it just
    compares serial to serial; standalone runs exercise the pool.)"""
    from circuitinsight.engine import interp as I

    n = 110                                      # L = n+1, dets = 2(n+1) >= 200
    inst = [{"name": "Vin", "device_type": "vsource",
             "terminals": {"p": "in", "n": "0"}}]
    prev = "in"
    for i in range(1, n + 1):
        node = f"n{i}"
        inst += [{"name": f"R{i}", "device_type": "resistor",
                  "terminals": {"p": prev, "n": node}, "params": {"r": "1k"}},
                 {"name": f"C{i}", "device_type": "capacitor",
                  "terminals": {"p": node, "n": "0"}, "params": {"c": "1p"}}]
        prev = node
    cin = {"cin_version": "0.1", "top": "m", "ground": ["0"],
           "definitions": {"m": {"ports": [], "instances": inst}}}
    an = Analyzer.from_cin(cin)

    I._S_CACHE.clear()
    h_par = an.tf("Vin", f"n{n}", keep=[])
    I._S_CACHE.clear()
    old = I._S_SWEEP_PARALLEL_MIN
    I._S_SWEEP_PARALLEL_MIN = 1 << 30            # force the serial path
    try:
        h_ser = an.tf("Vin", f"n{n}", keep=[])
    finally:
        I._S_SWEEP_PARALLEL_MIN = old
    n1, d1 = h_par.num_den
    n2, d2 = h_ser.num_den
    assert sp.expand(n1.as_expr() * d2.as_expr()
                     - d1.as_expr() * n2.as_expr()) == 0
