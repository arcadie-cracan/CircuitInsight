"""Session states: save/load the user's accumulated work.

The principle (Arcadie's): the tool always STARTS at ground truth — the
numeric first light against the simulator reference — and everything
the user built on top (matches, keep ticks, band, strategy, budgets)
is a first-class, recoverable object instead of fragile in-session
state. A state has two layers with different contracts:

  selections — a small JSON manifest, loadable across versions forever.
  solution   — the expensive computed Result, restored ONLY when the
               fingerprint of the run that produced it matches the run
               now open: content hash of the CIN and the psf data files,
               the cap model, the matches, the circuit state and the
               package version. A stale solution is never shown as if
               current — the selections restore and Solve recomputes.

File format: a zip (suffix .cistate) holding manifest.json and,
optionally, result.pkl. The pickle is an internal format gated by the
fingerprint, so cross-version unpickling problems are handled by
falling back to selections-only.
"""
from __future__ import annotations

import hashlib
import io
import json
import pickle
import zipfile
from pathlib import Path

__all__ = ["fingerprint", "save_state", "load_state", "state_path"]

_FMT = 1


def fingerprint(cin: str | Path, psf: str | Path, cap_model: str,
                matches, circuit_state: str,
                mos_model: str = "separate") -> str:
    """Identity of the run a solution belongs to. Content-based for the
    data files (copies and touched mtimes must not invalidate), plus
    everything that changes the reconstruction."""
    from circuitinsight import __version__ as ver

    h = hashlib.sha256()
    h.update(Path(cin).read_bytes())
    psf = Path(psf)
    for p in sorted(psf.glob("*")):
        if p.is_file():
            h.update(p.name.encode())
            h.update(p.read_bytes())
    h.update(str(cap_model).encode())
    h.update(repr(sorted(tuple(g) for g in matches)).encode())
    h.update(str(circuit_state).encode())
    h.update(str(ver).encode())
    if mos_model != "separate":
        # hashed only when non-default, so every state saved before this
        # field existed keeps its fingerprint (and its stored solution)
        h.update(str(mos_model).encode())
    return h.hexdigest()


def state_path(cin: str | Path, name: str = "last") -> Path:
    """States live beside the CIN: <stem>.<name>.cistate."""
    cin = Path(cin)
    return cin.with_name(f"{cin.stem}.{name}.cistate")


def save_state(path: str | Path, manifest: dict, result=None) -> None:
    """manifest: JSON-able selections + the fingerprint. result: the
    shown Result object (pickled), or None for selections-only."""
    manifest = dict(manifest, format=_FMT)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("manifest.json",
                   json.dumps(manifest, indent=1, default=str))
        if result is not None:
            z.writestr("result.pkl",
                       pickle.dumps(result, protocol=4))
    Path(path).write_bytes(buf.getvalue())


def load_state(path: str | Path, expect_fingerprint: str | None = None):
    """(manifest, result_or_None, stale: bool). The manifest always
    loads; the pickled result loads only when expect_fingerprint
    matches the stored one AND unpickling succeeds — anything else is
    a stale solution, reported as such, never shown as current."""
    with zipfile.ZipFile(Path(path)) as z:
        manifest = json.loads(z.read("manifest.json").decode())
        stale = (expect_fingerprint is None
                 or manifest.get("fingerprint") != expect_fingerprint)
        result = None
        if not stale and "result.pkl" in z.namelist():
            try:
                result = pickle.loads(z.read("result.pkl"))
            except Exception:
                result, stale = None, True
    return manifest, result, stale
