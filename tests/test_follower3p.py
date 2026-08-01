"""End-to-end port-marker chain on the real follower bench (follower3p).

Fixture: follower3 with IPORT (0 A parallel port at `out`), VPORT1 (series
port in the CL=1 pF load branch), VPORT2 (series port in MN1's drain/supply
branch). psf/ is the gain run (VIN excited, VPORT currents saved) and
psf_z/ the impedance run (IPORT excited); both carry an xf result, which
isolates every source's transfer regardless of AC magnitudes.

This is the simulator validation of the whole impedance/equivalent stack:
Zout against xf and against a direct 1 A AC injection, Norton branch
currents against saved vsource currents, and the two-model contrast.
"""
import warnings
from pathlib import Path

import numpy as np
import pytest

from circuitinsight.adapters.spectre import SpectreRun

FIX = Path(__file__).resolve().parent / "fixtures" / "spectre" / "follower3p"


@pytest.fixture(scope="module")
def run():
    return SpectreRun(FIX / "tb_follower3p.cin.json", FIX / "psf")


@pytest.fixture(scope="module")
def refs(run):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        ac = run.ac()
        acz = SpectreRun(FIX / "tb_follower3p.cin.json", FIX / "psf_z").ac()
    return ac, run.xf(), acz


def _db(a, b):
    return np.abs(20 * np.log10(np.abs(a) / np.abs(b)))


def test_excited_sources_policy(run):
    # the CIN records both schematic AC magnitudes -> ac() must warn about
    # superposition and point at the xf result
    assert run.excited_sources() == ["VIN", "IPORT"]
    assert run.has_xf()
    with pytest.warns(UserWarning, match="superpose.*xf"):
        run.ac()


def test_xf_and_ac_readers_agree(run, refs):
    ac, xf, _ = refs
    assert len(xf.freq) == len(ac.freq)
    # VIN is the (only) ac-run excitation, so xf's VIN column and the ac
    # wave are the same physical quantity through two parsers
    g = np.abs(xf.tf("VIN")) / np.abs(ac.wave("out"))
    assert np.abs(g - 1).max() < 1e-4    # xf VIN col == ac wave (SKY130)
    with pytest.raises(KeyError, match="available"):
        xf.tf("NOPE")


def test_gain_and_zout_simulator_validated(run, refs):
    _, xf, acz = refs
    f = xf.freq
    an = run.analyzer(cap_model="matrix")
    H = an.tf("VIN", "out", keep=[])
    assert _db(H.numeric(f), xf.tf("VIN")).max() < 0.1

    Z = an.impedance(port="IPORT", keep=[]).numeric(f)
    # against xf's IPORT column (V/A = ohms)
    assert _db(Z, xf.tf("IPORT")).max() < 0.05

    # the lumped model is visibly off in impedance, as in the gain story
    Zl = run.analyzer(cap_model="lumped").impedance(
        port="IPORT", keep=[]).numeric(f)
    assert _db(Zl, xf.tf("IPORT")).max() > 0.1


def test_norton_currents_match_saved_branch_currents(run, refs):
    ac, _, _ = refs
    an = run.analyzer(cap_model="matrix")
    f = ac.freq
    for port, tol in (("VPORT1", 1e-2), ("VPORT2", 1e-2)):
        isc = an.equivalent("VIN", port=port, keep=[]).isc.numeric(f)
        ref = np.abs(ac.wave(f"{port}:p"))
        m = ref > 1e-2 * ref.max()          # skip nulls (ratio ill-defined there)
        ratio = np.abs(isc[m]) / ref[m]
        assert np.abs(ratio - 1).max() < tol, port

    # supply-branch current is another exposing quantity for the lumped model
    isc_l = run.analyzer(cap_model="lumped").equivalent(
        "VIN", port="VPORT2", keep=[]).isc.numeric(f)
    ratio_l = np.abs(isc_l) / np.abs(ac.wave("VPORT2:p"))
    assert np.abs(ratio_l - 1).max() > 0.1


def test_unloaded_zout_recovered_by_disabling_load_port(run):
    # opening VPORT1 removes the CL branch: the bare-follower numbers used
    # in the paper (Zdc = 1/(gm+gmbs+gds) of MN1 = 1014.8 ohm) come back
    an = run.analyzer(cap_model="matrix")
    Z = an.impedance(port="IPORT", disable=("VPORT1",), keep=[])
    assert complex(Z.numeric([1e3])[0]).real == pytest.approx(986.8, abs=0.2)
