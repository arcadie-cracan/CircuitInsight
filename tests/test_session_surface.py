"""The frozen GUI-session surface (engine-process plan E-A).

The engine will move to its own process; everything the GUI needs must be
a NAMED SessionController member, and the chatty read-only state must
survive a pickle. These tests pin the freeze so a new private poke or an
unpicklable snapshot field fails here, not in phase E-C.
"""
import pickle
import re
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src" / "circuitinsight"
FC = Path(__file__).resolve().parent / "fixtures" / "spectre" / "fc"


def _open_fc():
    from circuitinsight.session import SessionController

    return SessionController.open(str(FC / "tb_fc.cin.json"),
                                  str(FC / "psf"))


def test_gui_never_touches_session_privates():
    """No `controller._name` anywhere under gui/: the process boundary
    can only carry the public surface."""
    # three disguises, all found the hard way in the proc-mode
    # shakeout: controller._x, the c = self.controller alias, and the
    # string form getattr(c, "_x", default) whose default swallows the
    # miss silently
    poke = re.compile(r"\bcontroller\._[a-zA-Z]"
                      r"|\bc\._[a-zA-Z]"
                      r"|getattr\((?:self\.controller|c|ses), ?[\"']_")
    hits = []
    for p in (SRC / "gui").rglob("*.py"):
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(),
                                 start=1):
            if poke.search(line):
                hits.append(f"{p.name}:{i}: {line.strip()}")
    assert not hits, "private session pokes in the GUI:\n" + "\n".join(hits)


def test_snapshot_is_plain_and_picklable():
    """The read-only mirror crosses the pipe as-is: plain containers,
    round-trips through pickle, and carries the fields the GUI's chatty
    paths read."""
    s = _open_fc()
    snap = s.session_snapshot()
    again = pickle.loads(pickle.dumps(snap))
    assert again == snap or again.keys() == snap.keys()
    for key in ("devices", "nets", "ground", "probes", "matches",
                "sym_aliases", "op_point_labels", "stb_probe", "has_stb",
                "circuit_state", "input_ports", "output_nets",
                "match_value_policy", "analyses", "lumped_gmb"):
        assert key in snap, key
    assert snap["devices"] and all(
        set(d) == {"name", "device_type", "terminals"}
        for d in snap["devices"])
    # no engine objects leak through: every leaf is a builtin
    def flat(v):
        if isinstance(v, dict):
            for x in v.values():
                yield from flat(x)
        elif isinstance(v, (list, tuple, set)):
            for x in v:
                yield from flat(x)
        else:
            yield v
    bad = [v for v in flat(snap)
           if v is not None and not isinstance(v, (str, int, float, bool))]
    assert not bad, f"non-builtin leaves in the snapshot: {bad[:5]}"


def test_alias_api_replaces_the_dict_pokes():
    s = _open_fc()
    s.set_alias("gm_I0_NM1", r"g_{m1}")
    assert s.sym_aliases["gm_I0_NM1"] == r"g_{m1}"
    s.set_alias("gm_I0_NM1", "")
    assert "gm_I0_NM1" not in s.sym_aliases
    s.set_aliases({"a": "A"})
    s.set_aliases({"b": "B"}, merge=True)
    assert s.sym_aliases == {"a": "A", "b": "B"}
    s.set_aliases({})
    assert s.sym_aliases == {}


def test_promoted_methods_cover_the_old_pokes():
    """The four bench pokes, by name, doing what the raw analyzer did."""
    s = _open_fc()
    probe = s.probes[0]
    idx = s.probe_node_index(probe)
    assert isinstance(idx, dict) and idx
    upd = s.loop_updater(probe)
    assert hasattr(upd, "baseline") and hasattr(upd, "with_branch")
    assert len(upd.freqs) == 300
    fr, packed = s.ac_reference(s.input_ports()[0], s.output_nets()[0])
    assert (fr is None) == (packed is None)
    with pytest.raises(Exception):
        s.mode_loop("no_such_probe", "nor_this")