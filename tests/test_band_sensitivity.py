"""Band-sampled sensitivity ranking for keep-set selection.

Fast, simulator-free checks on circuits with known poles; the µA741
exercise (feature detection, complex-vs-magnitude metric) is in
test_ua741.
"""
import numpy as np
import pytest

from circuitinsight import Analyzer

# RC low-pass: H = 1/(1+sRC), single pole at 1/(2*pi*R*C) = ~1 kHz
RC = {
    "cin_version": "0.1",
    "design": {"name": "rc", "source": {"kind": "hand"}},
    "top": "m", "ground": ["0"],
    "definitions": {"m": {"ports": [], "instances": [
        {"name": "Vin", "device_type": "vsource",
         "terminals": {"p": "in", "n": "0"}},
        {"name": "R", "device_type": "resistor",
         "terminals": {"p": "in", "n": "out"}, "params": {"r": "1k"}},
        {"name": "C", "device_type": "capacitor",
         "terminals": {"p": "out", "n": "0"}, "params": {"c": "159n"}}],
    }},
}


def test_pole_detected_and_ranking_symmetric():
    bs = Analyzer.from_cin(RC).band_sensitivities("Vin", "out",
                                                  fmin=10, fmax=1e5)
    # the 1 kHz pole is found by the numeric pencil eigen-solve
    assert np.any(np.abs(bs.poles - 1000) < 100)
    # R and C are the only parameters and, for H=1/(1+sRC), have identical
    # relative sensitivity -- they must rank together and comparably
    scores = dict(bs.ranking)
    assert set(scores) == {"R", "C"}
    assert scores["R"] == pytest.approx(scores["C"], rel=0.05)


def test_metric_option_validated_and_selectable():
    an = Analyzer.from_cin(RC)
    with pytest.raises(ValueError, match="metric"):
        an.band_sensitivities("Vin", "out", metric="bogus")
    for m in ("complex", "magnitude"):
        bs = an.band_sensitivities("Vin", "out", metric=m, fmin=10, fmax=1e5)
        assert bs.metric == m
        assert bs.suggest_keep(1) == ["R"] or bs.suggest_keep(1) == ["C"]


def test_purely_resistive_falls_back_to_grid():
    # no reactive elements -> no features -> log-grid fallback, still ranks
    bare = {**RC, "definitions": {"m": {"ports": [], "instances": [
        i for i in RC["definitions"]["m"]["instances"] if i["name"] != "C"]}}}
    bs = Analyzer.from_cin(bare).band_sensitivities("Vin", "out",
                                                    fmin=10, fmax=1e5)
    assert bs.poles.size == 0 and len(bs.freqs) >= 2
    assert bs.ranking[0][0] == "R"


def test_frequency_resolved_matrix_and_targeted_ranking():
    an = Analyzer.from_cin(_miller())
    an.match("M1", "M2"); an.match("M3", "M4")
    bs = an.band_sensitivities("V1", "vout")
    # the full [symbol, frequency] matrix is exposed
    assert bs.matrix.shape == (len(bs.symbols), len(bs.freqs))
    # the compensation cap's importance is frequency-dependent: its rank
    # improves in the high-frequency sub-range (the gain elements fade there,
    # while CC still shapes the rolloff), even though its sensitivity peaks
    # at the dominant pole it creates
    fmid = np.sqrt(bs.freqs.min() * bs.freqs.max())
    lo_rank = [n for n, _ in bs.rank(fmax=fmid)].index("CC")
    hi_rank = [n for n, _ in bs.rank(fmin=fmid)].index("CC")
    assert hi_rank < lo_rank                     # more prominent up high
    # CC peaks near the dominant (low) pole, not the high one
    assert bs.peak_frequency("CC") < np.sqrt(bs.poles.min() * bs.poles.max())


def test_dominant_reactances_finds_the_compensation_cap():
    an = Analyzer.from_cin(_miller())
    an.match("M1", "M2"); an.match("M3", "M4")
    rr = an.dominant_reactances("V1", "vout", tol_db=1.0)
    # CC alone reproduces the whole response to well within tolerance
    assert rr.selected == ["CC"]
    assert rr.errors_db[-1] < 1.0
    assert rr.baseline_db > 20                  # far off with no reactances


def test_dominant_reactances_exclude_keeps_element_active():
    an = Analyzer.from_cin(_miller())
    an.match("M1", "M2"); an.match("M3", "M4")
    rr = an.dominant_reactances("V1", "vout", tol_db=1.0, exclude=("CC",))
    # CC is forced always-on, so it never appears as a *selected* addition
    assert "CC" not in rr.selected


def _miller():
    import json
    from pathlib import Path
    p = (Path(__file__).resolve().parent / "golden" / "circuits"
         / "miller_ota.cin.json")
    return json.loads(p.read_text())
