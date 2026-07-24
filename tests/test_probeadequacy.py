"""M-G: probe-adequacy advisor (analysis/probeadequacy.py).

The killer capability: CircuitInsight holds the exact closed-loop det A,
so it can check whether a probe's margins tell the truth (PM-implied
damping vs actual pole damping) and name the loop dynamics the probe is
blind to (pole-moving devices with zero loop-gain elasticity). The fd
fixture is the discriminator: under matched symmetry the CMFB loop is
EXACTLY invisible to the DM probe.
"""
import warnings
from pathlib import Path

import numpy as np
import pytest

from circuitinsight.adapters.spectre import SpectreRun
from circuitinsight.analysis.probeadequacy import zeta_from_pm

FIX = Path(__file__).resolve().parent / "fixtures" / "spectre"


def _report(sub, cin, psf, probe):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run = SpectreRun(FIX / sub / cin, FIX / sub / psf)
        an = run.analyzer(cap_model="matrix")
        return an.assess_probe(probe)


@pytest.fixture(scope="module")
def miller():
    return _report("miller", "tb_ota2s_stb.cin.json", "psf_stb", "IPRB0")


@pytest.fixture(scope="module")
def fd_dm():
    return _report("fd", "tb_fdota_stb.cin.json", "psf_dm",
                   "FDPRB.IPRB_DM")


@pytest.fixture(scope="module")
def fd_cm():
    return _report("fd", "tb_fdota_stb.cin.json", "psf_cm",
                   "FDPRB.IPRB_CM")


def test_zeta_from_pm_second_order():
    """Exact second-order PM(zeta) relation, spot values: 45 deg ->
    ~0.42, 60 deg -> ~0.61, 76.3 deg -> ~1.0."""
    assert zeta_from_pm(45.0) == pytest.approx(0.4205, abs=2e-3)
    assert zeta_from_pm(60.0) == pytest.approx(0.6117, abs=2e-3)
    assert zeta_from_pm(76.3) == pytest.approx(1.0, abs=0.01)
    assert zeta_from_pm(0.0) == 0.0


def test_miller_probe_is_adequate(miller):
    """The flagship probe cuts the only global loop: margins consistent,
    and every strongly pole-moving device in the signal path is visible
    to T."""
    assert miller.margins_consistent
    assert miller.rhp_poles_hz.size == 0
    assert miller.pm_deg == pytest.approx(59.9, abs=0.2)
    assert miller.min_zeta > zeta_from_pm(miller.pm_deg) * 0.5
    vis = {v.name: v for v in miller.visibility}
    for dev in ("I0_MP2", "I0_MN1", "I0_MP0"):
        assert not vis[dev].unobserved
    assert "consistent" in miller.verdict()


def test_fd_dm_probe_flags_the_cmfb_loop(fd_dm):
    """Matched symmetry makes the CM/CMFB dynamics EXACTLY invisible to
    the DM probe: the advisor must flag the CMFB devices while keeping
    the DM path visible -- 'probe these separately' is the correct
    advice for a fully-differential circuit."""
    assert fd_dm.margins_consistent          # the DM verdict itself is fine
    unobs = set(fd_dm.unobserved)
    for dev in ("I0_MPA", "I0_MNS", "I0_MPLP", "I0_MPLN"):
        assert dev in unobs
    vis = {v.name: v for v in fd_dm.visibility}
    for dev in ("I0_MN1P", "I0_MN1N", "I0_MP2P", "I0_MP2N"):
        assert not vis[dev].unobserved
    # invisibility at numerical zero, not merely small: symmetric
    # decoupling (solver roundoff only)
    assert vis["I0_MPA"].t_elasticity < 1e-9
    assert vis["I0_MPA"].pole_elasticity > 0.5
    assert "unobserved" in fd_dm.verdict()


def test_fd_cm_probe_sees_the_cmfb_loop(fd_cm):
    """The CM probe observes the CMFB devices the DM probe missed; only
    genuinely local bias dynamics (the dummy diode) stay unobserved."""
    assert fd_cm.margins_consistent
    vis = {v.name: v for v in fd_cm.visibility}
    for dev in ("I0_MPA", "I0_MNS", "I0_MPLP", "I0_MNR"):
        assert not vis[dev].unobserved
    # the two probes together cover every strong pole-mover except
    # pure bias-chain poles
    strong = {v.name for v in fd_cm.visibility if v.pole_elasticity > 0.1}
    covered = ({v.name for v in fd_cm.visibility if not v.unobserved}
               | {"I0_MN1P", "I0_MN1N", "I0_MP2P", "I0_MP2N"})
    residue = strong - covered - {"I0_MPB", "MN2"}
    assert not residue


def test_session_surface():
    """session.assess_probe returns the cached report with a verdict."""
    from circuitinsight.session import SessionController

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = SessionController.open(
            FIX / "miller" / "tb_ota2s_stb.cin.json",
            FIX / "miller" / "psf_stb", cap_model="matrix")
        r1 = s.assess_probe("IPRB0")
        r2 = s.assess_probe("IPRB0")
    assert r1 is r2
    assert "consistent" in r1.verdict()


def test_gft_check_miller_clean(miller):
    """M-G(b): with the M-F designation the miller probe's GFT check is
    spotless -- exact identity, flat Hinf, no in-band feedthrough --
    and the verdict names the common break."""
    from circuitinsight.adapters.spectre import SpectreRun

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run = SpectreRun(FIX / "miller" / "tb_ota2s_stb.cin.json",
                         FIX / "miller" / "psf_stb")
        r = run.analyzer(cap_model="matrix").assess_probe(
            "IPRB0", gft={"inp": "VIND", "out": "vout",
                          "error": ("vin_p", -1)})
    assert r.gft_check.identity_residual == 0.0
    assert r.gft_check.hinf_flat_dev < 1e-3
    assert r.gft_check.feedthrough_crossover_hz is None
    assert "common break" in r.verdict()


def test_gft_check_fd_flags_hinf_peaking():
    """The fd DM designation is exact (residual 0.0) but its Hinf peaks
    near the loop crossover -- the detector must say so on the default
    band and stay quiet on the flat sub-band."""
    from circuitinsight.adapters.spectre import SpectreRun

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run = SpectreRun(FIX / "fd" / "tb_fdota_stb.cin.json",
                         FIX / "fd" / "psf_dm")
        an = run.analyzer(cap_model="matrix")
        wide = an.assess_probe(
            "FDPRB.IPRB_DM",
            gft={"inp": "VIND", "out": "FDPRB.dmo",
                 "error": ("vin_dm", +1)})
        narrow = an.assess_probe(
            "FDPRB.IPRB_DM",
            gft={"inp": "VIND", "out": "FDPRB.dmo",
                 "error": ("vin_dm", +1), "band": (1e2, 1e4)})
    assert wide.gft_check.identity_residual == 0.0
    assert wide.gft_check.hinf_flat_dev > 0.5
    assert "deviates" in wide.verdict()
    assert narrow.gft_check.hinf_flat_dev < 0.02   # below the 5% bar
    assert "deviates" not in narrow.verdict()
