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


# the diode_bias field case, in passives: block A (the port's own
# RC at `smpl`) and block B (an RLC tank at `b`) share only the
# supply rail vdd! -- held by the ideal VSUP, so AC ground -- and
# actual ground. Block B's resonance is a FEATURE of the whole-matrix
# pencil, but its parameters cannot move Z(smpl) at all.
TWOBLOCK = {
    "cin_version": "0.1",
    "design": {"name": "twoblock", "source": {"kind": "hand"}},
    "top": "m", "ground": ["0"], "globals": ["vdd!"],
    "definitions": {"m": {"ports": [], "instances": [
        {"name": "VSUP", "device_type": "vsource",
         "terminals": {"p": "vdd!", "n": "0"}},
        {"name": "I0", "device_type": "isource",
         "terminals": {"p": "vdd!", "n": "smpl"}},
        {"name": "Ra", "device_type": "resistor",
         "terminals": {"p": "smpl", "n": "0"}, "params": {"r": "50k"}},
        {"name": "Ca", "device_type": "capacitor",
         "terminals": {"p": "smpl", "n": "0"}, "params": {"c": "5f"}},
        {"name": "Rb", "device_type": "resistor",
         "terminals": {"p": "vdd!", "n": "b"}, "params": {"r": "10k"}},
        {"name": "Lb", "device_type": "inductor",
         "terminals": {"p": "b", "n": "bl"}, "params": {"l": "10u"}},
        {"name": "Cb", "device_type": "capacitor",
         "terminals": {"p": "bl", "n": "0"}, "params": {"c": "8p"}}],
    }},
}


def test_decoupled_block_scores_zero_and_the_band_is_sampled():
    """The diode_bias field case: with the port's branch and an RLC
    block sharing only AC-ground rails, the block's parameters score
    EXACTLY zero on Z at the port -- and with the band given, the
    low-frequency plateau is sampled, so the port's own R tops the
    ranking with its |Z|*S ~ Z0 score (unbanded, the sampler spanned
    only the clustered features and the plateau was invisible)."""
    an = Analyzer.from_cin(TWOBLOCK)
    # the rail-hung parallel port resolves ground-referenced now
    assert an._port_nets("I0") == ("smpl", None, False)
    bs = an.band_sensitivities("I0", "smpl", fmin=1e3, fmax=1e9)
    scores = dict(bs.ranking)
    assert scores["Rb"] == 0 and scores["Lb"] == 0 and scores["Cb"] == 0
    assert bs.ranking[0][0] == "Ra"
    # the plateau is in the sample set: Ra's score is |Z0|-scale
    assert scores["Ra"] == pytest.approx(50e3, rel=0.1)
    assert bs.freqs.min() <= 1.1e3


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


def test_loop_margins_fold_the_unwrap_turn():
    """PM is an angular DISTANCE, not the unwrapped running phase: a
    sharply peaked loop pushes the phase up through +180 before the
    unity crossing, unwrap banks a full turn, and the crossing read
    360 deg high -- 'PM 386' ranked a compensation 26 deg from
    oscillation as an achiever."""
    import numpy as np

    from circuitinsight.analysis.sensitivity import loop_margins

    f = np.geomspace(1e4, 1e9, 400)
    x = np.linspace(0.0, 1.0, f.size)
    mag_db = 40.0 - 80.0 * x                 # unity crossing mid-band
    # continuous phase rising through +180 (a resonant peak) before
    # settling toward +386 at the crossing region
    ph = 170.0 + 260.0 * np.clip(x / 0.4, 0.0, 1.0) - 44.0 * x
    h = 10.0 ** (mag_db / 20.0) * np.exp(1j * np.radians(ph))
    pm, fpm, _gm, _fgm = loop_margins(f, h)
    assert pm is not None and fpm is not None
    assert -180.0 < pm <= 180.0              # folded, never 360-high
    truth = float(np.interp(np.log10(fpm), np.log10(f), ph))
    assert abs(pm - (((truth + 180.0) % 360.0) - 180.0)) < 2.0

    # an ordinary loop is untouched: phase falls 175 -> 40 at crossing
    ph2 = 175.0 - 135.0 * x
    h2 = 10.0 ** (mag_db / 20.0) * np.exp(1j * np.radians(ph2))
    pm2, _f2, _g2, _fg2 = loop_margins(f, h2)
    assert 100.0 < pm2 < 110.0               # mid-band crossing value


def test_loop_margins_take_the_worst_crossing():
    """A dip-peak-rolloff magnitude crosses unity three times: the
    first crossing carried 125 deg while the last had 28 -- robustness
    is the closest approach to the critical point, so PM reports the
    minimum over every crossing (fc field case: the net13-net3 Miller
    row wore PM 125 while grazing instability past the doublet)."""
    import numpy as np

    from circuitinsight.analysis.sensitivity import loop_margins

    f = np.geomspace(1e5, 1e9, 800)
    x = np.linspace(0.0, 1.0, f.size)
    # magnitude: falls through 0 dB, resonant bump back above, falls
    mag_db = 40.0 - 90.0 * x + 18.0 * np.exp(-((x - 0.62) / 0.05) ** 2)
    # phase: high early, cliff under the bump's far side
    ph = 130.0 - 20.0 * x - 100.0 / (1.0 + np.exp(-(x - 0.60) / 0.02))
    h = 10.0 ** (mag_db / 20.0) * np.exp(1j * np.radians(ph))
    m = mag_db
    crossings = np.where(np.diff(np.sign(m)))[0]
    assert len(crossings) == 3               # the scenario is real
    pm, fpm, _gm, _fgm = loop_margins(f, h)
    # the first crossing carries ~110 deg; the WORST (last) ~10-30
    first_ph = ph[crossings[0]]
    last_ph = ph[crossings[-1] + 1]
    assert first_ph > 100.0 and last_ph < 40.0
    assert pm < 40.0                          # worst, not first
    assert fpm > float(f[crossings[1]])      # at the late crossing
