"""SRR-backend integration smoke test — run ON <sim-host>:

    module load project/ICDESIGN pdk=SKY130
    cd ~/projects/ICDESIGN/SKY130/CircuitInsight
    PYTHONPATH=src .venv-srr/bin/python3 scripts/srr_smoke.py <binary psf dir>

Reads a native binary PSF results directory through cdspythonsrr and checks
the canonical records look sane. This cannot run in CI (needs Cadence
licensing), hence a script instead of a pytest module.

To rebuild .venv-srr: `module load tool/ANX.TO/ANACONDA3` for a numpy-equipped
python3, then `python3 -m venv --system-site-packages .venv-srr` and
`pip install --no-index --no-deps
$CDSHOME/tools/python/64bit/virtuoso/cdspythonsrr-*.whl`.
"""
import sys

from circuitinsight.adapters.spectre.backends import srr_available
from circuitinsight.adapters.spectre.opdata import load_dcopinfo

psf_dir = sys.argv[1] if len(sys.argv) > 1 else (
    "$HOME/projects/ICDESIGN/SKY130/simulation/worklib/"
    "tb_m0/maestro/results/maestro/ExplorerRun.0/1/worklib_tb_m0_1/psf"
)

assert srr_available(), "cdspythonsrr not importable — use .venv-srr under module env"

records = load_dcopinfo(psf_dir, backend="cdspythonsrr")
print(f"{len(records)} records: {sorted(records)}")

# design-agnostic: check the first MOSFET and any capacitor look canonical
mos = [r for r in records.values() if r.device_type == "mosfet"]
assert mos, "no MOSFET records found"
m = mos[0]
assert m.params["gm"] > 1e-6, m.params
assert m.params["cgs"] > 0, "cap magnitudes must be positive"
assert m.params.get("region") in (1, 2, 3), f"region={m.params.get('region')}"
print(f"{m.name} canonical:", {k: f"{v:.4g}" for k, v in m.params.items()})
caps = [r for r in records.values() if r.device_type == "capacitor"]
if caps:
    assert caps[0].params["c"] > 0, "capacitor value must be positive"
    print(f"{caps[0].name}: c = {caps[0].params['c']:.4g} F (from OP data)")

from circuitinsight.adapters.spectre.acdata import load_ac

ac = load_ac(psf_dir, backend="cdspythonsrr")
node = next(n for n, w in ac.waves.items() if w.dtype.kind == "c")
assert len(ac.freq) > 10
print(f"AC: {len(ac.freq)} points, {node}[0]={ac.wave(node)[0]:.4g}")

# xf: per-source transfer functions via the native reader. Only present in
# runs that included an `xf` analysis (e.g. the follower3p port bench).
import os

from circuitinsight.adapters.spectre.acdata import load_xf

if os.path.exists(os.path.join(psf_dir, "xf.xf")):
    xf = load_xf(psf_dir, backend="cdspythonsrr")
    assert len(xf.freq) > 10, "xf sweep too short"
    src = sorted(xf.transfers)
    print(f"xf: {len(xf.freq)} points, {len(src)} sources: {src}")
    # a 0 A parallel port marker's column is the port impedance in ohms;
    # IPORT on follower3p reads ~1015 ohm at dc (V/A output units)
    if "IPORT" in xf.transfers:
        z0 = xf.tf("IPORT")[0]
        assert 500 < abs(z0) < 5000, f"IPORT xf[0]={z0:.4g} not impedance-like"
        print(f"xf IPORT[0] = {z0:.6g} ohm  (parallel-port impedance)")
    else:
        print("  (no IPORT column — not a port bench; structure check only)")
else:
    print("xf: no xf.xf in results dir — skipped")

print("SRR backend smoke test: OK")
