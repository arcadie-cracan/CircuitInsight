"""L-J: the stability certificate on conditionally stable loops.

Beleca, Plesa, Onet, Neag, "Methods for assessing the stability of
conditionally stable circuits by using small-signal simulations",
ROMJIST 25(2) 2022, pp. 205-223, give four loop gains on which the Bode
margins mislead. They are rational, so they enter here as the
TransferFunction they are, in the ledger's convention (arg T(0) = 180,
the critical point at +1), which is the paper's T with its sign
flipped. The paper's own numbers are reproduced: T1 has GM -52 dB and
PM 80.3 deg and is stable; T3 has a pole at the origin and one in the
right half-plane, reads a positive phase margin, and is stable by the
Nyquist count; T2 and T4 are unstable.
"""
import pytest
import sympy as sp

from circuitinsight.analysis.marginledger import (margin_ledger,
                                                  stability_certificate)
from circuitinsight.engine.mna import S, TransferFunction


def _loop(expr):
    """The paper's T(s) in the ledger's convention."""
    return TransferFunction(sp.together(-expr))


T1 = 1e4 * (S + 1) ** 2 / (S / 0.1 + 1) ** 3
T2 = 1e4 * (S + 1) ** 2 / ((S / 0.1 + 1) ** 3 * (S / 6 + 1) ** 2)
T3 = 16 * (S + 1) / (S * (S / 0.1 - 1) * (S / 5 + 1) ** 2)
T4 = 500 * (S + 1) / (S * (S / 0.1 - 1) * (S / 5 + 1) ** 2)


def test_t1_is_stable_with_a_negative_gain_margin():
    """Two zeros lift the phase back above the critical line after three
    poles pushed it through: the phase crosses the line twice below
    f_u, an even count, the gain margin at either crossing is deeply
    negative, and the loop is stable all the same."""
    led = margin_ledger(_loop(T1))
    assert led.pm_deg == pytest.approx(80.3, abs=0.5)
    assert led.gm_db == pytest.approx(-52.0, abs=1.0)
    st = led.stability
    assert st["verdict"] == "stable"
    assert st["closed_loop_rhp"] == 0 and st["rhp_poles"] == 0
    assert st["unity_crossings"] == 1
    assert st["cuts_before_fu"] == 2 and st["parity"] == "even"
    assert st["axis_cuts"] == 2 and st["gm_defined"] is False
    assert st["bode_applies"] is True
    assert "even" in st["reason"] and "undefined" in st["reason"]
    assert any(l.startswith("stable:") for l in led.describe())


def test_t2_is_unstable_with_three_crossings():
    """Two more poles take the phase through the line a third time
    before f_u: odd, and the closed loop has poles in the right
    half-plane."""
    led = margin_ledger(_loop(T2))
    st = led.stability
    assert st["verdict"] == "unstable"
    assert st["closed_loop_rhp"] >= 1
    assert st["cuts_before_fu"] == 3 and st["parity"] == "odd"
    assert led.pm_deg < 0


def test_t3_is_stable_by_the_nyquist_count_only():
    """A pole at the origin and one in the right half-plane: the ledger
    refuses (its shares anchor at DC), the Bode margins would read a
    positive phase margin and call it unstable, and the closed loop
    has no right-half-plane pole. The one open-loop RHP pole is matched
    by one counter-clockwise encirclement, Z - P = -1."""
    with pytest.raises(ValueError):
        margin_ledger(_loop(T3))
    st = stability_certificate(_loop(T3))
    assert st["verdict"] == "stable"
    assert st["rhp_poles"] == 1 and st["origin_poles"] == 1
    assert st["closed_loop_rhp"] == 0
    assert st["encirclements_cw"] == -1
    assert "counter-clockwise" in st["reason"]
    assert st["bode_applies"] is None          # no grid: not judged


def test_t4_is_unstable():
    st = stability_certificate(_loop(T4))
    assert st["verdict"] == "unstable"
    assert st["closed_loop_rhp"] >= 1
    assert st["encirclements_cw"] == st["closed_loop_rhp"] - 1


def test_a_plain_loop_reads_as_the_margins_say():
    """One unity crossing, no critical cut below it, nothing in the
    right half-plane: the margins decide, and the certificate says so
    in the same words a designer would."""
    T = 1e3 / ((S / 10 + 1) * (S / 1e4 + 1) * (S / 1e5 + 1))
    led = margin_ledger(_loop(T))
    st = led.stability
    assert st["verdict"] == "stable"
    assert st["bode_applies"] is True
    assert st["cuts_before_fu"] == 0 and st["parity"] == "even"
    assert st["gm_defined"] is True
    assert "holds" in st["reason"]

