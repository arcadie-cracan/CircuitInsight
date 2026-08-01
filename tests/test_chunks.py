"""Passive networks as one equivalent admittance.

Two things to establish: the elimination is EXACT for an arbitrary
interior (a T-network with an internal node), and the chunker refuses to
swallow device-model elements -- a transistor's gds and cpi are passive
by stamp but they are the OP parameters this tool exists to expose."""
import warnings
from pathlib import Path

import numpy as np
import pytest
import sympy as sp

from circuitinsight import Analyzer
from circuitinsight.analysis.chunks import chunk_admittance, passive_chunks

FIX = Path(__file__).resolve().parent / "fixtures" / "spectre"


def _tee_cin():
    """A two-terminal ladder bridging two active nodes: a -R1- m -(R2||CM)- b.
    Node m is INTERNAL, so the chunk must eliminate it. (A shunt to ground
    inside the network would make it a two-PORT, which chunk_admittance
    correctly refuses -- ground counts as a terminal.)"""
    inst = [
        {"name": "V1", "device_type": "vsource",
         "terminals": {"p": "in", "n": "0"}},
        {"name": "G1", "device_type": "vccs",
         "terminals": {"p": "0", "n": "a", "cp": "in", "cn": "0"},
         "params": {"gm": "1m"}},
        {"name": "RA", "device_type": "resistor",
         "terminals": {"p": "a", "n": "0"}, "params": {"r": "10k"}},
        # the ladder: a -R1- m -(R2 || CM)- b, m internal
        {"name": "R1", "device_type": "resistor",
         "terminals": {"p": "a", "n": "m"}, "params": {"r": "1k"}},
        {"name": "R2", "device_type": "resistor",
         "terminals": {"p": "m", "n": "b"}, "params": {"r": "2k"}},
        {"name": "CM", "device_type": "capacitor",
         "terminals": {"p": "m", "n": "b"}, "params": {"c": "1p"}},
        {"name": "G2", "device_type": "vccs",
         "terminals": {"p": "0", "n": "out", "cp": "b", "cn": "0"},
         "params": {"gm": "2m"}},
        {"name": "RB", "device_type": "resistor",
         "terminals": {"p": "b", "n": "0"}, "params": {"r": "5k"}},
        {"name": "RL", "device_type": "resistor",
         "terminals": {"p": "out", "n": "0"}, "params": {"r": "5k"}},
    ]
    return {"cin_version": "0.1", "top": "main", "ground": ["0"],
            "definitions": {"main": {"ports": [], "instances": inst}}}


def test_tee_network_is_one_chunk_with_an_internal_node():
    an = Analyzer.from_cin(_tee_cin())
    chunks = passive_chunks(an.primitives, an.flat.ground,
                            keep_terminals=("out",))
    tee = [c for c in chunks if "R1" in c.names]
    assert len(tee) == 1
    c = tee[0]
    assert set(c.names) == {"R1", "R2", "CM"}
    assert set(c.terminals) == {"a", "b"}
    assert c.internal == ("m",)          # eliminated by the equivalent
    assert c.symbols_saved == 2          # three elements -> one symbol


def test_tee_equivalent_admittance_is_exact():
    """Y_eq of the T, symbolic in its own elements, checked against the
    textbook expression AND numerically."""
    an = Analyzer.from_cin(_tee_cin())
    c = [x for x in passive_chunks(an.primitives, an.flat.ground,
                                   keep_terminals=("out",))
         if "R1" in x.names][0]
    y, z = chunk_admittance(c, an.flat.ground)
    R1, R2, CM = (z.symbols["R1"], z.symbols["R2"], z.symbols["CM"])
    s = sp.Symbol("s")
    # Z(a->b) = R1 + R2 || (1/sCM)
    expect = sp.cancel(1 / (R1 + R2 / (1 + s * R2 * CM)))
    assert sp.simplify(sp.cancel(y - expect)) == 0

    vals = {R1: 1e3, R2: 2e3, CM: 1e-12}
    for f in (1e3, 1e7, 1e9):
        w = 2j * np.pi * f
        got = complex(y.xreplace({s: w}).xreplace(vals))
        ref = 1.0 / (1e3 + 2e3 / (1 + w * 2e3 * 1e-12))
        assert got == pytest.approx(ref, rel=1e-9)


def test_chunker_leaves_device_model_elements_alone():
    """gds and cpi are passive by stamp but they are OP parameters: the
    chunker must not fold six devices' substrate caps into a 'network'."""
    from circuitinsight.adapters.spectre import SpectreRun

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run = SpectreRun(FIX / "fd" / "tb_fdota_stb.cin.json",
                         FIX / "fd" / "psf_dm")
        an = run.analyzer(cap_model="matrix")
    chunks = passive_chunks(an.primitives, an.flat.ground,
                            keep_terminals=("voutp",))
    named = {n for c in chunks for n in c.names}
    assert not any("MP" in n or "MN" in n for n in named)
    # what it SHOULD find: the CMFB sense pairs, R || C
    assert len(chunks) == 2
    for c in chunks:
        assert set(c.kinds) == {"c", "r"}
        y, z = chunk_admittance(c, an.flat.ground)
        r = [n for n in c.names if n.startswith("RCM")][0]
        cc = [n for n in c.names if n.startswith("CCM")][0]
        assert sp.simplify(y - (z.symbols[cc] * sp.Symbol("s")
                                + 1 / z.symbols[r])) == 0
