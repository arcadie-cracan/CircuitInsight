"""Backend dispatch and SRR-record handling (without a Cadence install)."""
from pathlib import Path

import pytest

from circuitinsight.adapters.spectre.backends import (
    BackendError,
    RawRecord,
    _infer_srr_type,
    read_dcopinfo_raw,
    srr_available,
)
from circuitinsight.adapters.spectre.opdata import canonicalize, model_class

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "spectre" / "m0"

# SKY130 devices are sky130_fd_pr subckt wrappers, so the raw dcOpInfo instance
# key carries the expanded primitive name (the join resolves it back to "M1").
NFET = "M1.msky130_fd_pr__nfet_01v8"


def test_auto_selects_psfascii_for_fixture():
    records = read_dcopinfo_raw(FIXTURES / "psf")
    assert records[NFET].type_name == "bsim4"
    assert records[NFET].values["gm"] > 0


def test_explicit_psfascii_backend():
    records = read_dcopinfo_raw(FIXTURES / "psf" / "dcOpInfo.info", backend="psfascii")
    assert len(records) == 10


def test_binary_without_srr_gives_actionable_error(tmp_path):
    fake = tmp_path / "dcOpInfo.info"
    fake.write_bytes(b"\x00\x00\x05\x00binarypsf")
    if srr_available():  # pragma: no cover - dev boxes have no cadence
        pytest.skip("cdspythonsrr installed here")
    with pytest.raises(BackendError, match="psfascii"):
        read_dcopinfo_raw(fake)


def test_unknown_backend_rejected():
    with pytest.raises(BackendError, match="unknown backend"):
        read_dcopinfo_raw(FIXTURES / "psf", backend="hdf5")


def test_srr_type_inference():
    assert _infer_srr_type({"gm": 1, "gds": 1, "vdsat": 1, "ids": 1}) == "srr:mosfet"
    assert _infer_srr_type({"res": 1e4, "v": 1, "i": 1}) == "srr:resistor"
    assert _infer_srr_type({"cap": 1e-13, "v": 1}) == "srr:capacitor"
    assert _infer_srr_type({"v": 1, "i": 1, "pwr": 1}) == "srr:source"
    assert _infer_srr_type({"weird": 1}) == "srr:unknown"


def test_srr_pseudo_types_canonicalize():
    rec = canonicalize(RawRecord("M9", "srr:mosfet", {
        "gm": 4.83e-4, "gds": 7.5e-6, "gmbs": 8.9e-5, "vdsat": 0.13,
        "cgs": -2.77e-15, "cgd": -7.1e-16, "cgb": -1.4e-16,
        "cjd": 1.6e-15, "cdb": -1.3e-19, "cjs": 2.27e-15, "csb": -2.9e-16,
        "region": 2,
    }))
    assert rec.device_type == "mosfet"
    assert rec.params["cgs"] == pytest.approx(2.77e-15)
    assert rec.params["cdb"] == pytest.approx(1.6e-15 + 1.3e-19)
    # sources: class None -> join skips the type check
    assert model_class("srr:source") is None
    assert model_class("srr:unknown") is None
