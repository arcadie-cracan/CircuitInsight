"""Trans-capacitance matrix analysis on the checked-in fixtures.

Prints the full BSIM4 K-matrix (K_ij = dQ_i/dV_j) per device, verifies the
row/column zero-sum identities, and quantifies the non-reciprocity that the
lumped five-capacitor model drops. See docs/transcap-analysis.md.
"""
import warnings
from pathlib import Path

import numpy as np

from circuitinsight.adapters.spectre.opdata import load_dcopinfo

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
T = ["g", "d", "s", "b"]


def kmat(raw):
    # Spectre stores the signed derivatives dQ_i/dV_j directly
    return np.array([[raw[f"c{a}{b}"] for b in T] for a in T])


def main():
    cases = [
        ("tb_m0 CS", ROOT / "tests/fixtures/spectre/m0/psf", ["M1", "M2"]),
        ("tb_ota2s", ROOT / "tests/fixtures/spectre/miller/psf",
         ["I0.MN0", "I0.MP2"]),
    ]
    for label, path, devs in cases:
        recs = load_dcopinfo(path)
        for d in devs:
            raw, p = recs[d].raw, recs[d].params
            K = kmat(raw)
            print(f"--- {label} / {d} (gm={p['gm']*1e6:.0f}u, "
                  f"gm/id={p.get('gmoverid', 0):.1f}) ---")
            for i, a in enumerate(T):
                cells = " ".join(f"{K[i, j]*1e15:+8.3f}" for j in range(4))
                print(f"   {a} {cells}  row_sum={K[i].sum()*1e15:+.4f} fF")
            print("   col sums:",
                  " ".join(f"{K[:, j].sum()*1e15:+.4f}" for j in range(4)))
            for (i, j), nm in [((0, 1), "gd/dg"), ((0, 2), "gs/sg"),
                               ((0, 3), "gb/bg"), ((1, 2), "ds/sd")]:
                dc = K[i, j] - K[j, i]
                rel = abs(dc) / max(abs(K[i, j]), abs(K[j, i]), 1e-30)
                bound = 2 * np.pi * 1e10 * abs(dc) / p["gm"]
                print(f"   dC({nm}) = {dc*1e15:+8.3f} fF ({100*rel:5.1f}%)"
                      f"   w*dC/gm @10GHz = {100*bound:5.2f}%")


if __name__ == "__main__":
    main()
