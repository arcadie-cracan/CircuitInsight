"""Results-reading backends for Spectre PSF data.

Two backends, auto-selected by file format:

- **psfascii** (built-in, pure Python): reads `-format psfascii` output.
  Works anywhere, no Cadence installation needed. Preferred for CI/fixtures.
- **cdspythonsrr** (Cadence SRR Python bindings): reads native binary
  PSF/PSFXL, i.e. untouched ADE/Explorer results. Only available where the
  Cadence environment is (module env for licensing + the `cdspythonsrr`
  wheel from `$CDSHOME/tools/python/64bit/virtuoso/`). On <sim-host> a ready
  venv exists at `CircuitInsight/.venv-srr`.

Both produce the same RawRecord dicts; canonical mapping (opdata.py) is
backend-independent.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path


class BackendError(RuntimeError):
    pass


@dataclass
class RawRecord:
    name: str
    type_name: str                # psf struct name, or pseudo "srr:<class>"
    values: dict[str, object]
    props: dict[str, object] = field(default_factory=dict)


# ------------------------------------------------------------------ psfascii

def _read_psfascii(path: Path) -> dict[str, RawRecord]:
    from .psfascii import parse_psfascii

    psf = parse_psfascii(path)
    return {
        name: RawRecord(
            name=name,
            type_name=e.type_name,
            values=e.value if isinstance(e.value, dict) else {"value": e.value},
            props=e.props,
        )
        for name, e in psf.entries.items()
    }


# ------------------------------------------------------------- cdspythonsrr

def srr_import_error() -> Exception | None:
    """None if cdspythonsrr imports in THIS interpreter, else why it didn't.

    Detection is per-interpreter: having the wheel installed in some other venv
    is irrelevant. Distinguishing "not installed" from "installed but failed to
    import" (missing Cadence libs, no license) matters -- they need different
    fixes, and reporting both as "not available" misleads.
    """
    try:
        import cdspythonsrr  # noqa: F401
        return None
    except Exception as exc:
        return exc


def srr_available() -> bool:
    return srr_import_error() is None


def _infer_srr_type(params: dict) -> str:
    """SRR flattens outputs to inst:param with no struct name; classify by
    parameter signature."""
    keys = set(params)
    if {"gm", "gds", "vdsat"} <= keys:
        return "srr:mosfet"
    if "res" in keys:
        return "srr:resistor"
    if "cap" in keys:
        return "srr:capacitor"
    if "ind" in keys:
        return "srr:inductor"
    if {"gm", "cpi"} <= keys or {"gm", "vbe"} <= keys:
        return "srr:bjt"
    if keys <= {"v", "i", "pwr", "trise"}:
        return "srr:source"
    return "srr:unknown"


def _read_srr(psf_dir: Path, analysis: str | None = None) -> dict[str, RawRecord]:
    try:
        from cdspythonsrr.core.ocean import (getData, openResults, outputs,
                                             results, selectResult)
    except Exception as exc:
        raise BackendError(
            f"cdspythonsrr backend unavailable: {exc}. It needs the Cadence "
            f"module environment (licensing) and the wheel from "
            f"$CDSHOME/tools/python/64bit/virtuoso/. Alternatively rerun "
            f"spectre with '-format psfascii'."
        ) from exc

    openResults(str(psf_dir))
    names = results()
    if analysis is None:
        cands = [n for n in names if "dcOpInfo" in n]
        if not cands:
            raise BackendError(f"no dcOpInfo analysis in {psf_dir}; found {names}")
        analysis = cands[0]
    selectResult(analysis)

    grouped: dict[str, dict[str, object]] = {}
    for out in outputs():
        inst, sep, param = out.rpartition(":")
        if not sep:
            continue
        grouped.setdefault(inst, {})[param] = getData(out)

    return {
        inst: RawRecord(inst, _infer_srr_type(params), params)
        for inst, params in grouped.items()
    }


# ------------------------------------------------------------------ dispatch

def _is_psfascii(path: Path) -> bool:
    with open(path, "rb") as fh:
        return fh.read(6) == b"HEADER"


def read_dcopinfo_raw(
    path: str | Path, backend: str = "auto"
) -> dict[str, RawRecord]:
    """Read dcOpInfo records. `path` is a psf results directory or the
    dcOpInfo file itself. backend: 'auto' | 'psfascii' | 'cdspythonsrr'."""
    path = Path(path)
    dcop = path / "dcOpInfo.info" if path.is_dir() else path

    if backend == "psfascii":
        return _read_psfascii(dcop)
    if backend == "cdspythonsrr":
        return _read_srr(dcop.parent if dcop.is_file() else path)
    if backend != "auto":
        raise BackendError(f"unknown backend {backend!r}")

    if dcop.exists() and _is_psfascii(dcop):
        return _read_psfascii(dcop)
    err = srr_import_error()
    if err is None:
        return _read_srr(dcop.parent if dcop.exists() else path)
    kind = ("not installed" if isinstance(err, ModuleNotFoundError)
            else f"import failed -- {type(err).__name__}: {err}")
    raise BackendError(
        f"{dcop}: binary PSF results, and cdspythonsrr is unusable in the "
        f"interpreter running CircuitInsight.\n"
        f"  interpreter : {sys.executable}\n"
        f"  cdspythonsrr: {kind}\n"
        f"Detection is a plain `import cdspythonsrr` in THIS interpreter -- "
        f"having it in another venv does not count. Fixes: install the Cadence "
        f"wheel from $CDSHOME/tools/python/64bit/virtuoso/ into this "
        f"interpreter (its Python version must match the wheel's), run "
        f"CircuitInsight under an interpreter that has it, or rerun spectre "
        f"with '-format psfascii'."
    )
