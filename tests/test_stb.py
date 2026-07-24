"""Spectre stb (loop-gain) adapter: parse, margins, and bench consistency.

Fixture: tb_ota2s_stb.scs -- the SKY130 two-stage in closed unity feedback,
the DC/AC switch rig replaced by an iprobe in the feedback path (a DC short
exactly where S1 closed), plain-Miller Cc=16 pF. Spectre's stb implements
Tian's two-injection method at the probe; these results are the validation
reference for the reconstructed loop gain (docs/loopgain-plan.md, M-B+).
"""
from pathlib import Path

import numpy as np
import pytest

from circuitinsight.adapters.spectre import SpectreRun
from circuitinsight.adapters.spectre.stbdata import load_stb

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "spectre" / "miller"


@pytest.fixture(scope="module")
def stb():
    return load_stb(FIXTURES / "psf_stb")


def test_sweep_parses(stb):
    assert stb.freq[0] == pytest.approx(1e-3)
    assert stb.freq[-1] == pytest.approx(1e10)
    assert np.all(np.diff(stb.freq) > 0)
    T = stb.loop_gain
    assert T.dtype == complex and len(T) == len(stb.freq)
    # probe impedances/admittances looking both ways ship alongside
    assert {"ZL", "ZG", "YL", "YG"} <= set(stb.waves)


def test_low_frequency_loop_gain_is_a0(stb):
    """Unity feedback: T(0) = A0 of the amplifier -- the 85.48 dB the ac
    fixtures pin, now measured as *loop* gain through the Tian probe."""
    t0 = 20 * np.log10(abs(stb.loop_gain[0]))
    assert t0 == pytest.approx(85.48, abs=0.05)


def test_margins_pinned(stb):
    assert stb.phase_margin_deg == pytest.approx(59.898, abs=0.01)
    assert stb.phase_margin_freq_hz == pytest.approx(3.4892e6, rel=1e-3)
    assert stb.gain_margin_db == pytest.approx(9.015, abs=0.01)
    assert stb.gain_margin_freq_hz == pytest.approx(1.2550e7, rel=1e-3)
    assert "stable" in stb.margins["stb_state"]


def test_margins_consistent_with_wave(stb):
    """The scalar margins must re-derive from the loopGain sweep itself.
    Spectre's loopGain convention embeds the 180-degree reference: the phase
    starts at +180 at DC, PM is arg(T) at |T|=1 directly, and GM is -|T|dB
    where arg(T) crosses 0 -- a sign convention the reconstruction (M-B)
    must reproduce, so pin it here."""
    f = stb.freq
    T = stb.loop_gain
    m = 20 * np.log10(np.abs(T))
    ph = np.degrees(np.unwrap(np.angle(T)))
    assert ph[0] == pytest.approx(180, abs=0.01)         # the convention itself

    k = np.where(np.diff(np.sign(m)))[0][0]              # |T| = 1 crossing
    x = np.log10(f)
    xu = np.interp(0, [m[k + 1], m[k]], [x[k + 1], x[k]])
    pm = np.interp(xu, [x[k], x[k + 1]], [ph[k], ph[k + 1]])
    assert pm == pytest.approx(stb.phase_margin_deg, abs=0.1)
    assert 10 ** xu == pytest.approx(stb.phase_margin_freq_hz, rel=1e-3)

    j = np.where(np.diff(np.sign(ph)))[0][0]             # arg(T) = 0 crossing
    xj = np.interp(0, [ph[j + 1], ph[j]], [x[j + 1], x[j]])
    gm = -np.interp(xj, [x[j], x[j + 1]], [m[j], m[j + 1]])
    assert gm == pytest.approx(stb.gain_margin_db, abs=0.1)


def test_probe_bench_preserves_the_operating_point():
    """The iprobe is a DC short exactly where the switch rig closed the DC
    loop, so the stb bench's OP must equal the ac fixtures' OP -- the same
    invariance that lets one DC solve span all compensation states."""
    ref = SpectreRun(FIXTURES / "tb_ota2s.cin.json", FIXTURES / "psf").op_data
    from circuitinsight.adapters.spectre.opdata import load_dcopinfo

    probe = load_dcopinfo(FIXTURES / "psf_stb" / "dcOpInfo.info")
    for name, rec in probe.items():
        if rec.device_type != "mosfet":
            continue
        # stb record keys carry the wrapper suffix; map to the CIN name
        cin = name.split(".msky130")[0]
        assert cin in ref
        for p in ("gm", "gds"):
            assert rec.params[p] == pytest.approx(ref[cin][p], rel=1e-9), \
                f"{cin}.{p} differs between stb and ac benches"


def test_spectre_run_stb_accessor():
    run = SpectreRun(FIXTURES / "tb_ota2s.cin.json", FIXTURES / "psf")
    r = run.stb(name="../psf_stb/stb.stb")
    assert r.phase_margin_deg == pytest.approx(59.898, abs=0.01)


def test_stb_probe_discovery_and_any_vsource_candidates():
    """The stb probe comes from the psfascii header (or the run netlist,
    below); any vsource is a probe CANDIDATE, iprobe-tagged first --
    Spectre accepts any voltage source as the stb probe."""
    import warnings
    from circuitinsight.session import SessionController

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = SessionController.open(FIXTURES / "tb_ota2s_stb.cin.json",
                                   FIXTURES / "psf_stb")
    assert s.probes[0] == "IPRB0"                # declared intent first
    assert set(s.probes) >= {"IPRB0", "VIND", "VSUP"}
    assert s.stb_probe() == "IPRB0"              # psfascii header


def test_stb_probe_from_run_netlist(tmp_path):
    """Binary/absent stb results: the probe is read from the run's own
    netlist (the record of the analyses as DEFINED) -- the ADE launch
    path, where PSF data does not surface the setting."""
    import shutil
    import warnings
    from circuitinsight.adapters.spectre import SpectreRun

    shutil.copytree(FIXTURES / "psf", tmp_path / "psf")
    (tmp_path / "netlist").mkdir()
    (tmp_path / "netlist" / "input.scs").write_text(
        "stb stb start=1m stop=10G dec=20 probe=IPRB0 annotate=status\n")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = SpectreRun(FIXTURES / "tb_ota2s_stb.cin.json", tmp_path / "psf")
    assert r.stb_probe() == "IPRB0"
