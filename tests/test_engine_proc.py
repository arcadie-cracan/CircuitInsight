"""The engine child + proxy transport (engine-process plan E-B).

Real spawn, real fc fixture: handshake, snapshot, property reads,
a solve with results crossing as pickles, remote errors with their
tracebacks, cancel plumbing, and a clean shutdown. The Qt guard pins
the child's import graph -- spawn re-imports the host module, and a
child that pulls Chromium in would be a crash surface.
"""
import subprocess
import sys
from pathlib import Path

import pytest

FC = Path(__file__).resolve().parent / "fixtures" / "spectre" / "fc"


@pytest.fixture(scope="module")
def engine():
    from circuitinsight.engine.proxy import EngineProxy

    px = EngineProxy.spawn(FC / "tb_fc.cin.json", FC / "psf")
    yield px
    px.close()


def test_the_child_never_imports_qt():
    import os

    code = ("import sys; import circuitinsight.engine.host; "
            "import circuitinsight.session; "
            "bad = [m for m in sys.modules if 'PySide6' in m]; "
            "assert not bad, bad")
    env = dict(os.environ)
    src = str(Path(__file__).resolve().parent.parent / "src")
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run([sys.executable, "-c", code], check=True, env=env)


def test_handshake_open_and_snapshot(engine):
    snap = engine.snapshot()
    assert snap["probes"] and snap["devices"]
    assert engine.alive()


def test_property_reads_cross_the_glance_lane(engine):
    assert engine.get("probes") == engine.snapshot()["probes"]
    assert engine.get("has_stb") == engine.snapshot()["has_stb"]


def test_a_solve_crosses_as_a_result(engine):
    inp = engine.call("suggested_input")
    out = engine.call("suggested_output")
    seen = []
    r = engine.call("solve", inp, out, (), lane="solve",
                    progress=lambda f, t: seen.append(f))
    assert type(r).__name__ == "Result"
    assert r.keep == []
    st = engine.stats[-1]
    assert st["method"] == "solve" and st["lane"] == "solve"
    assert st["reply_bytes"] > 0 and st["seconds"] > 0


def test_remote_errors_carry_their_traceback(engine):
    from circuitinsight.engine.proxy import EngineError

    with pytest.raises(EngineError) as ei:
        engine.call("no_such_method")
    assert ei.value.remote_type == "AttributeError"
    assert "no_such_method" in ei.value.remote_traceback


def test_cancel_is_acknowledged_while_idle(engine):
    engine.cancel("solve")            # sets the flag; next progress
    assert engine.alive()             # checkpoint would raise -- but the
                                      # lane is idle, so nothing dies


def test_close_ends_the_child():
    from circuitinsight.engine.proxy import EngineProxy

    px = EngineProxy.spawn(FC / "tb_fc.cin.json", FC / "psf")
    assert px.alive()
    px.close()
    assert not px.alive()


@pytest.fixture(scope="module")
def remote():
    from circuitinsight.engine.remote import open_engine

    ses = open_engine(FC / "tb_fc.cin.json", FC / "psf", mode="proc")
    yield ses
    ses.close_engine()


def test_remote_session_serves_properties_locally(remote):
    from circuitinsight.session import DeviceInfo

    assert remote.probes and isinstance(remote.probes, list)
    assert all(isinstance(d, DeviceInfo) for d in remote.devices)
    assert remote.circuit_state == "as imported"
    assert isinstance(remote.sym_aliases, dict)
    n_before = len(remote.engine_stats())
    _ = (remote.probes, remote.devices, remote.matches, remote.ground)
    assert len(remote.engine_stats()) == n_before   # no round trips


def test_remote_session_forwards_and_refreshes(remote):
    inp = remote.suggested_input()
    out = remote.suggested_output()
    r = remote.attach_template(remote.solve(inp, out, ()))
    assert type(r).__name__ == "Result"
    assert r.template_text is not None
    st = remote.engine_stats()[-1]
    assert st["lane"] == "solve"
    groups = remote.suggest_matches()
    if groups:
        remote.set_matches(*groups)
        assert remote.matches == [tuple(g) for g in groups]


def test_remote_alias_mirror_stays_true(remote):
    remote.set_alias("gm_I0_NM1", "g_{m1}")
    assert remote.sym_aliases.get("gm_I0_NM1") == "g_{m1}"
    engine_truth = remote._px.call("session_snapshot")["sym_aliases"]
    assert engine_truth.get("gm_I0_NM1") == "g_{m1}"


def test_results_arrive_with_warm_display_caches(remote):
    """Plan E-D: the engine warms the expression lines BEFORE the
    pickle, so the GUI's own _expr_lines call is a dictionary lookup
    -- the round_expr bill never lands in the GUI process."""
    remote.set_display_prefs(base=True, wrap=False)
    inp = remote.suggested_input()
    out = remote.suggested_output()
    r = remote.solve(inp, out, ())
    cache = getattr(r, "_expr_lines_cache", None)
    assert cache, "the Result must carry its warmed cache across"
    key = (True, False, tuple(sorted(remote.sym_aliases.items())))
    assert key in cache and cache[key]


def test_updater_crosses_by_recipe(remote):
    upd = remote.loop_updater(remote.probes[0])
    base = upd.baseline()
    assert len(base) == len(upd.freqs) == 300


def test_soft_cancel_closes_the_call_with_an_error_frame():
    """Plan E-F: cancel is ENGINE-driven so the lane stays protocol-
    clean -- the running call ends in an EngineCancelled error frame,
    never in orphaned progress frames. Driven in-thread against the
    host's serve loop with a fake session: fully deterministic."""
    import pickle
    import threading
    from multiprocessing import Pipe

    from circuitinsight.engine.host import _PICKLE, _serve

    class _Slow:
        ping = "pong"

        def grind(self, progress=None):
            for k in range(10_000):
                progress(k, 10_000)          # the cancel checkpoint
            return "never"

    parent, child = Pipe()
    evt = threading.Event()
    t = threading.Thread(target=_serve, args=(_Slow(), child, evt),
                         daemon=True)
    t.start()
    parent.send_bytes(pickle.dumps(
        ("call", "grind", (), {}, True, False), _PICKLE))
    # a few progress frames prove the call is running, then cancel
    for _ in range(3):
        msg = pickle.loads(parent.recv_bytes())
        assert msg[0] == "progress"
    evt.set()
    while True:
        msg = pickle.loads(parent.recv_bytes())
        if msg[0] != "progress":
            break
    assert msg[0] == "err" and msg[1] == "EngineCancelled"
    # the lane is clean: the next call answers normally
    parent.send_bytes(pickle.dumps(("get", "ping"), _PICKLE))
    msg = pickle.loads(parent.recv_bytes())
    assert msg == ("ok", "pong")
    parent.close()


def test_hard_restart_replays_the_session_state(remote):
    """Plan E-F: kill-and-respawn brings back a session with every
    state-bearing call replayed -- aliases here, and the engine is a
    NEW process that still answers."""
    old_pid = remote._px._proc.pid
    remote.set_alias("gm_I0_NM1", "g_{m,restart}")
    remote.engine_restart()
    assert remote._px._proc.pid != old_pid
    assert remote.engine_alive()
    truth = remote._px.call("session_snapshot")["sym_aliases"]
    assert truth.get("gm_I0_NM1") == "g_{m,restart}"
    r = remote.solve(remote.suggested_input(),
                     remote.suggested_output(), ())
    assert type(r).__name__ == "Result"


def test_open_failure_reports_not_hangs():
    from circuitinsight.engine.proxy import EngineError, EngineProxy

    with pytest.raises(EngineError) as ei:
        EngineProxy.spawn(FC / "no_such.cin.json", FC / "psf",
                          timeout=120.0)
    assert ei.value.remote_traceback