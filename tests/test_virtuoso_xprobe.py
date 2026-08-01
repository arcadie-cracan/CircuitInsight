"""M10 cross-probe: the Python half, against a fake SKILL workspace.

The Cadence half (skill/cin_xprobe.il) can only be exercised against a live
Virtuoso and is checklisted there. What IS testable here is everything that
decides what gets sent: which device a clicked symbol names, that a missing
bridge degrades to a reason instead of an exception, and that a session which
goes away mid-click cannot take the GUI down.
"""
import pytest

from circuitinsight.virtuoso import xprobe


# ------------------------------------------------------------ fake bridge
class FakeWs:
    """Stands in for skillbridge's Workspace: ws["Name"](args) -> result."""

    def __init__(self, defined=("CInHighlight", "CInXprobeHere"),
                 fail=False):
        self.defined = set(defined)
        self.fail = fail
        self.calls = []
        self.closed = False

    def __getitem__(self, name):
        def call(*args):
            if self.fail:
                raise RuntimeError("socket closed")
            self.calls.append((name, args))
            if name in ("isCallable", "boundp"):
                # SKILL is handed a SYMBOL here, not a string -- skillbridge's
                # Symbol carries the bare name
                return getattr(args[0], "name", args[0]) in self.defined
            if name == "CInHighlight":
                return bool(args[0])
            if name == "CInSelection":
                return ["MN0", "MN1"]
            if name == "CInHighlightClear":
                return True
            return None
        return call

    def close(self):
        self.closed = True


INSTANCES = ["I0.MN0", "I0.MN1", "I0.MP0", "I0.Cc", "MN0", "I0.sub_amp.M1"]


# --------------------------------------------------- symbol -> instance
def test_symbol_resolves_to_the_longest_matching_instance():
    """gm_I0_MN0 is I0.MN0, not the top-level MN0 that also exists: a shorter
    tail match would cross-probe a different device with a similar name."""
    assert xprobe.instance_for_symbol("gm_I0_MN0", INSTANCES) == "I0.MN0"
    assert xprobe.instance_for_symbol("gds_I0_MP0", INSTANCES) == "I0.MP0"
    assert xprobe.instance_for_symbol("gm_MN0", INSTANCES) == "MN0"


def test_passive_symbol_is_its_own_instance():
    """A passive's join key IS the instance path (no quantity prefix)."""
    assert xprobe.instance_for_symbol("I0_Cc", INSTANCES) == "I0.Cc"


def test_instance_names_containing_underscores_still_match():
    """Resolution matches against REAL instance names, so an underscore inside
    one is not mistaken for a hierarchy separator."""
    assert xprobe.instance_for_symbol("gm_I0_sub_amp_M1",
                                      INSTANCES) == "I0.sub_amp.M1"


def test_unknown_or_empty_symbol_resolves_to_nothing():
    assert xprobe.instance_for_symbol("gm_NOPE", INSTANCES) is None
    assert xprobe.instance_for_symbol("", INSTANCES) is None
    assert xprobe.instance_for_symbol("gm_I0_MN0", []) is None


# ------------------------------------------------------------- connection
def test_connect_reports_a_reason_when_the_bridge_is_absent(monkeypatch):
    monkeypatch.setattr(xprobe, "AVAILABLE", False)
    probe, why = xprobe.CrossProbe.connect()
    assert probe is None and "skillbridge" in why


def test_connect_reports_a_reason_when_no_server_listens(monkeypatch):
    class Boom:
        @staticmethod
        def open(_id=None):
            raise RuntimeError("no server found")
    monkeypatch.setattr(xprobe, "AVAILABLE", True)
    monkeypatch.setattr(xprobe, "Workspace", Boom)
    probe, why = xprobe.CrossProbe.connect()
    assert probe is None and "no Virtuoso skill server" in why


def test_connect_refuses_a_session_without_the_skill_helpers(monkeypatch):
    """Connected but cin_xprobe.il not loaded: that must be reported, not
    discovered later as every highlight silently failing."""
    monkeypatch.setattr(xprobe, "AVAILABLE", True)
    monkeypatch.setattr(xprobe, "Workspace",
                        type("W", (), {"open": staticmethod(
                            lambda _id=None: FakeWs(defined=()))}))
    probe, why = xprobe.CrossProbe.connect()
    assert probe is None and "cin_xprobe.il" in why


def test_connect_succeeds_when_helpers_are_present(monkeypatch):
    monkeypatch.setattr(xprobe, "AVAILABLE", True)
    monkeypatch.setattr(xprobe, "Workspace",
                        type("W", (), {"open": staticmethod(
                            lambda _id=None: FakeWs())}))
    probe, why = xprobe.CrossProbe.connect()
    assert probe is not None and why is None


# ---------------------------------------------------------------- probing
def test_highlight_sends_deduplicated_paths():
    ws = FakeWs()
    probe = xprobe.CrossProbe(ws)
    assert probe.highlight(["I0.MN0", "I0.MN0", "I0.MN1"])
    assert ws.calls[-1] == ("CInHighlight", (["I0.MN0", "I0.MN1"],))


def test_highlight_of_nothing_is_a_no_op():
    ws = FakeWs()
    assert not xprobe.CrossProbe(ws).highlight([])
    assert not xprobe.CrossProbe(ws).highlight([None])
    assert ws.calls == []


def test_a_dead_session_never_raises_into_a_click_handler():
    """Virtuoso exiting must degrade the feature, not crash the GUI."""
    probe = xprobe.CrossProbe(FakeWs(fail=True))
    assert probe.highlight(["I0.MN0"]) is False
    assert probe.clear() is False
    assert probe.selection() == []
    assert probe.ready() is False
    probe.close()                        # must not raise either


def test_selection_comes_back_as_a_list_of_names():
    assert xprobe.CrossProbe(FakeWs()).selection() == ["MN0", "MN1"]


@pytest.mark.skipif(not xprobe.AVAILABLE,
                    reason="Symbol comes from skillbridge; without the "
                           "[virtuoso] extra ready() falls back to a str, and "
                           "connect() never reaches it anyway")
def test_readiness_asks_skill_with_a_symbol_not_a_string():
    """isCallable takes a SYMBOL. Sending the Python str made SKILL see the
    string "CInHighlight", answer nil, and the GUI report "not defined" while
    the helpers sat there working -- the first live GUI failure."""
    ws = FakeWs()
    assert xprobe.CrossProbe(ws).ready()
    name, args = ws.calls[0]
    assert name == "isCallable"
    assert getattr(args[0], "name", None) == "CInHighlight"   # Symbol, not str


def test_connect_pins_the_cross_probe_root():
    """The root must be pinned to the top cellview, or the designer switching
    to a sub-schematic to view a highlight breaks the next probe."""
    ws = FakeWs()
    xprobe.CrossProbe(ws).pin_root()
    assert ws.calls[-1][0] == "CInXprobeHere"
