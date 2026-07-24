"""Trans-capacitance matrix consistency on real BSIM4 fixture data.

Guards the conventions the whole cap-modeling story rests on
(docs/transcap-analysis.md): Spectre stores signed derivatives dQ_i/dV_j,
rows/columns sum to zero, and the gate-drain pair is near-reciprocal while
gate-source is not.
"""
from pathlib import Path

import numpy as np
import pytest

from circuitinsight.adapters.spectre.opdata import load_dcopinfo

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "spectre"
T = ["g", "d", "s", "b"]


def kmat(raw):
    return np.array([[raw[f"c{a}{b}"] for b in T] for a in T])


def _rec(recs, dev):
    if dev in recs:
        return recs[dev]
    return next(r for n, r in recs.items() if n.startswith(dev + "."))


@pytest.mark.parametrize("psf,dev", [
    ("miller/psf", "I0.MN0"), ("miller/psf", "I0.MP2"),
])
def test_charge_consistency_and_reciprocity_structure(psf, dev):
    rec = _rec(load_dcopinfo(FIXTURES / psf), dev)
    K = kmat(rec.raw)

    # signed-derivative convention: positive diagonal, negative gs/gd entries
    assert K[0, 0] > 0 and K[0, 1] < 0 and K[0, 2] < 0

    # reference invariance + charge conservation: rows and columns sum ~ 0
    scale = np.abs(K).max()
    assert np.abs(K.sum(axis=1)).max() < 1e-3 * scale
    assert np.abs(K.sum(axis=0)).max() < 1e-3 * scale

    # on SKY130 BSIM4 BOTH gate pairs are strongly non-reciprocal (the
    # charge-matrix model captures what the lumped 5-cap model cannot)
    assert abs(K[0, 1] - K[1, 0]) > 0.05 * abs(K[0, 1])   # gate-drain
    assert abs(K[0, 2] - K[2, 0]) > 0.05 * abs(K[0, 2])   # gate-source


# ---------------------------------------------- exact-matrix model A/B

def _worst(run, cap_model, inp, innet, out):
    from circuitinsight.analysis import compare_tf

    H = run.analyzer(cap_model=cap_model).tf(inp, out, keep=[])
    ac = run.ac()
    r = compare_tf(H, ac.freq, ac.wave(out), ac.wave(innet))
    return r.worst_mag_db, r.worst_phase_deg


@pytest.mark.parametrize("cin,psf,rename,inp,innet,out,lumped_db", [
    ("ota5t/tb_ota5t.cin.json", "ota5t/psf", None, "VIND", "vin_dm", "vout", 30),
    ("miller/tb_ota2s.cin.json", "miller/psf", None, "VIND", "vin_dm", "vout", 22),
])
def test_matrix_model_beats_lumped(cin, psf, rename, inp, innet, out,
                                   lumped_db):
    import warnings

    from circuitinsight.adapters.spectre import SpectreRun
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        run = SpectreRun(FIXTURES / cin, FIXTURES / psf, rename=rename)
        lm, lp = _worst(run, "lumped", inp, innet, out)
        mm, mp = _worst(run, "matrix", inp, innet, out)
    assert lm < lumped_db                      # lumped stays in its envelope
    assert mm <= lm * 1.05 and mp <= lp * 1.05  # matrix never worse
    assert mm < 0.1                            # and tight everywhere


def test_5t_matrix_model_is_essentially_exact():
    import warnings

    from circuitinsight.adapters.spectre import SpectreRun
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        run = SpectreRun(FIXTURES / "ota5t/tb_ota5t.cin.json",
                         FIXTURES / "ota5t/psf")
        mm, mp = _worst(run, "matrix", "VIND", "vin_dm", "vout")
    # the 2 um redesign's large caps make lumped visibly deviate (0.561 dB),
    # while the matrix model stays exact (0.0047 dB) — a ~120x improvement
    assert mm < 0.1 and mp < 3.5


def test_follower_chain_exposes_lumped_model():
    """The shown contrast (docs/transcap-analysis.md §5): an N-P-N follower
    chain of long-channel devices activates the source-row asymmetry that CS
    topologies mask. Lumped fails visibly; the matrix model stays exact."""
    import warnings

    from circuitinsight.adapters.spectre import SpectreRun
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        run = SpectreRun(FIXTURES / "follower3/tb_follower3.cin.json",
                         FIXTURES / "follower3/psf")
        lm, lp = _worst(run, "lumped", "VIN", "in", "out")
        mm, mp = _worst(run, "matrix", "VIN", "in", "out")
    assert lm > 1.0 and lp > 3.0          # lumped visibly wrong (1.5 dB/5.4)
    assert mm < 0.1 and mp < 6.5          # matrix indistinguishable from sim


def test_matrix_model_requires_k_entries():
    from circuitinsight import Analyzer
    from circuitinsight.models.small_signal import ModelError
    with pytest.raises(ModelError, match="matrix"):
        Analyzer.from_cin(
            Path(__file__).resolve().parent / "golden" / "circuits"
            / "cs_amp.cin.json", cap_model="matrix")
