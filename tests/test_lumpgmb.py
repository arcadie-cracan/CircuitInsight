"""mos_model='lumped-gmb': the exact per-device gm+gmb bundle.

The contract: lumping happens ONLY where it changes nothing — a device
whose gate and bulk sit at the same AC potential gets one ĝm symbol
whose value is the sum; a bulk-tied-to-source gmbs is dropped as inert;
everything else keeps its separate gmb. Exactness is asserted by
comparing transfer functions between the two models, not by trust."""
import warnings
from pathlib import Path

import numpy as np
import pytest

from circuitinsight.analyzer import Analyzer
from circuitinsight.engine.mna import MnaError

FIX = Path(__file__).resolve().parent / "fixtures" / "spectre"


def _cascode_cin():
    """Common-gate stage with an ideally-biased gate: the textbook
    (gm+gmb) case. M1: g at a DC bias net, b at ground, s driven."""
    return {
        "cin_version": "0.1",
        "design": {"name": "cg_tb", "source": {"kind": "hand"}},
        "top": "tb",
        "ground": ["0"],
        "definitions": {"tb": {"ports": [], "instances": [
            {"name": "VDD", "device_type": "vsource",
             "terminals": {"p": "vdd", "n": "0"}, "params": {"dc": "1.5"}},
            {"name": "VB", "device_type": "vsource",
             "terminals": {"p": "vb", "n": "0"}, "params": {"dc": "900m"}},
            {"name": "VIN", "device_type": "vsource",
             "terminals": {"p": "vs", "n": "0"},
             "params": {"dc": "200m", "mag": "1"}},
            {"name": "M1", "device_type": "mosfet",
             "terminals": {"d": "vout", "g": "vb", "s": "vs", "b": "0"},
             "params": {"polarity": "n", "gm": "1m", "gmbs": "0.2m",
                        "gds": "50u", "cgs": "10f", "cdb": "5f"}},
            # bulk tied to source: vbs == 0, the gmbs stamp is inert
            {"name": "M2", "device_type": "mosfet",
             "terminals": {"d": "vout", "g": "vs", "s": "0", "b": "0"},
             "params": {"polarity": "n", "gm": "0.5m", "gmbs": "0.1m",
                        "gds": "20u"}},
            {"name": "RL", "device_type": "resistor",
             "terminals": {"p": "vdd", "n": "vout"},
             "params": {"r": "20k"}},
        ]}},
    }


def test_lump_is_selective_and_sums_the_values():
    an = Analyzer.from_cin(_cascode_cin(), mos_model="lumped-gmb")
    by = {(p.inst, p.param): p for p in an.primitives}
    # M1 (gate at DC net, bulk at ground): ONE hatted symbol, summed
    assert ("M1", "gmhat") in by
    assert by[("M1", "gmhat")].value == pytest.approx(1.2e-3)
    assert ("M1", "gm") not in by and ("M1", "gmbs") not in by
    # M2 (bulk tied to source): the inert gmbs is dropped, gm stays gm
    assert ("M2", "gm") in by and ("M2", "gmbs") not in by
    assert ("M2", "gmhat") not in by
    assert an.lumped_gmb == {"M1": "lumped",
                             "M2": "dropped (bulk tied to source)"}


def test_lump_is_exact():
    """The whole point: both models produce the SAME transfer function."""
    sep = Analyzer.from_cin(_cascode_cin(), mos_model="separate")
    lmp = Analyzer.from_cin(_cascode_cin(), mos_model="lumped-gmb")
    f = np.logspace(2, 9, 40)
    h_sep = sep.tf("VIN", "vout", []).numeric(f)
    h_lmp = lmp.tf("VIN", "vout", []).numeric(f)
    np.testing.assert_allclose(h_lmp, h_sep, rtol=1e-9)


def test_input_on_a_justifying_source_is_refused():
    """Driving the bias source un-grounds the net that justified the
    bundle — the solve must refuse, not answer wrongly."""
    an = Analyzer.from_cin(_cascode_cin(), mos_model="lumped-gmb")
    with pytest.raises(MnaError, match="justified"):
        an.tf("VB", "vout", [])
    # the separate model answers the same question without complaint
    Analyzer.from_cin(_cascode_cin(),
                      mos_model="separate").tf("VB", "vout", [])


def test_separate_default_emits_no_hat():
    an = Analyzer.from_cin(_cascode_cin())
    params = {p.param for p in an.primitives}
    assert "gmhat" not in params and "gmbs" in params
    assert an.lumped_gmb == {}


def test_fc_lumps_honestly():
    """On the folded cascode the cascode gates sit on INTERNAL bias
    nets — real nodes with real impedance, not AC grounds — so the
    exact criterion must NOT hat them. The input pair (bulk tied to
    source) gets its inert gmbs dropped, and the response is untouched
    to numerical noise."""
    from circuitinsight.session import SessionController

    fc = FIX / "fc"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        sep = SessionController.open(fc / "tb_fc.cin.json", fc / "psf",
                                     cap_model="matrix")
        lmp = SessionController.open(fc / "tb_fc.cin.json", fc / "psf",
                                     cap_model="matrix",
                                     mos_model="lumped-gmb")
        r_sep = sep.solve("VIND", "vout", [])
        r_lmp = lmp.solve("VIND", "vout", [])
    info = lmp.lumped_gmb()
    assert all(how.startswith("dropped") for how in info.values()), info
    assert info, "the input pair's inert gmbs should be dropped"
    np.testing.assert_allclose(r_lmp.h, r_sep.h, rtol=1e-9)


def test_symbol_tex_renders_the_hat():
    pytest.importorskip("matplotlib")
    from circuitinsight.gui.view import symbol_tex

    assert symbol_tex("gmhat_I0_MN3") == r"\hat{g}_{m,\mathrm{MN3}}"
    assert symbol_tex("gmhat_I0_MN3", base=False) \
        == r"\hat{g}_{m,\mathrm{I0.MN3}}"


def test_fingerprint_changes_with_the_model_and_not_without():
    pytest.importorskip("PySide6")
    from circuitinsight.gui.state import fingerprint

    fc = FIX / "ota5t"
    base = fingerprint(fc / "tb_ota5t.cin.json", fc / "psf", "matrix",
                       [], "as imported")
    same = fingerprint(fc / "tb_ota5t.cin.json", fc / "psf", "matrix",
                       [], "as imported", mos_model="separate")
    lump = fingerprint(fc / "tb_ota5t.cin.json", fc / "psf", "matrix",
                       [], "as imported", mos_model="lumped-gmb")
    assert base == same          # old saved states keep their solutions
    assert lump != base


def test_lump_primitive_pass_is_shared():
    """The primitive-level pass reads the criterion off the stamps
    themselves — the reduction path feeds it rewired primitives."""
    from circuitinsight.engine.primitives import Primitive
    from circuitinsight.models.small_signal import lump_gmb_primitives

    prims = [
        Primitive("M1", "gm", "vccs", ("d1", "s1", "gnd", "s1"), 1e-3),
        Primitive("M1", "gmbs", "vccs", ("d1", "s1", "vdd", "s1"), 2e-4),
        Primitive("M1", "gds", "g", ("d1", "s1"), 5e-5),
        # signal-driven gate: stays separate
        Primitive("M2", "gm", "vccs", ("d2", "s2", "in", "s2"), 1e-3),
        Primitive("M2", "gmbs", "vccs", ("d2", "s2", "gnd", "s2"), 1e-4),
    ]
    info = {}
    out = lump_gmb_primitives(prims, frozenset({"gnd", "vdd"}), info,
                              tag="lumped (reduced)")
    by = {(p.inst, p.param) for p in out}
    assert ("M1", "gmhat") in by and ("M1", "gm") not in by
    assert ("M2", "gm") in by and ("M2", "gmbs") in by
    assert info == {"M1": "lumped (reduced)"}


def test_reduction_composes_the_hat_on_fc():
    """The composition: fc's cascode gates sit on internal bias nets, so
    the exact toggle alone hats nothing — but once the Reduce flow
    grounds a bias net (its cost measured end to end), the criterion
    holds on the reduced circuit and the cascodes earn their hats. The
    input guard must not object: structural grounding is
    input-independent."""
    from circuitinsight.session import SessionController

    fc = FIX / "fc"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        c = SessionController.open(fc / "tb_fc.cin.json", fc / "psf",
                                   cap_model="matrix",
                                   mos_model="lumped-gmb")
        assert not any(how == "lumped" for how in c.lumped_gmb().values())
        c.apply_reduction(["I0.net1"], inp="VIND", out="vout")
        info = c.lumped_gmb()
        composed = [n for n, how in info.items()
                    if how == "lumped (reduced)"]
        # NM1/NM3/NM5: gate on I0.net1 (now ground), bulk at gnd!,
        # source NOT ground — the textbook cascode hats
        assert composed, info
        names = {n for n, _, _ in c.rank_symbols("VIND", "vout")}
        assert any(s.startswith("gmhat_") for s in names)
        # the guard ignores structural lumps: the ordinary input solves
        r = c.solve("VIND", "vout", [])
        assert r.h.shape == r.freqs.shape
        c.revert_reduction()
        assert not any(how == "lumped (reduced)"
                       for how in c.lumped_gmb().values())
