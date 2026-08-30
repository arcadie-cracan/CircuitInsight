"""The GUI-side handle on the engine child (engine-process plan E-B).

EngineProxy spawns the child, performs the handshake, and forwards
calls over the three lanes. A call blocks its CALLING thread on the
lane's Connection -- which releases the GIL -- and invokes the caller's
progress/note callbacks inline between messages, so the existing worker
threads keep their exact shape: `fn(progress_cb)` becomes
`proxy.call("solve", ..., progress=progress_cb, lane="solve")`.

One outstanding request per lane, enforced with a per-lane lock; the
glance lane stays free while a solve runs, so short reads and cancel
never queue behind it. Telemetry (`stats`) records every call's wall
time and reply size -- the numbers phase E-H's payload decisions need.

Qt-free on purpose: workers already re-emit results as signals; the
proxy stays testable headless.
"""
from __future__ import annotations

import multiprocessing as mp
import pickle
import threading
import time

from .host import PROTO, _PICKLE, main as _host_main


class EngineError(RuntimeError):
    """A call failed engine-side; carries the remote type and traceback."""

    def __init__(self, remote_type: str, message: str,
                 remote_traceback: str = ""):
        super().__init__(f"{remote_type}: {message}")
        self.remote_type = remote_type
        self.remote_traceback = remote_traceback


class EngineDied(EngineError):
    """The child vanished mid-call (crash or kill)."""

    def __init__(self, detail: str = ""):
        super().__init__("EngineDied", detail or "the engine process died")


class EngineProxy:
    LANES = ("glance", "solve", "advisor")

    def __init__(self, proc, conns: dict):
        self._proc = proc
        self._conns = conns
        self._locks = {name: threading.Lock() for name in conns}
        self._snap: dict | None = None
        self.stats: list[dict] = []
        #: unsolicited engine notes (log lines) land here when the
        #: active call gave no note callback of its own
        self.on_note = None

    # ------------------------------------------------------------ spawn
    @classmethod
    def spawn(cls, cin, op, *, open_kw: dict | None = None,
              timeout: float = 300.0, pump=None) -> "EngineProxy":
        ctx = mp.get_context("spawn")             # never fork: Qt threads
        parents, children = {}, []
        for name in cls.LANES:
            p, c = ctx.Pipe()
            parents[name] = p
            children.append(c)
        # NOT daemonic: a daemonic process may not have children, and
        # the engine's whole point includes the cpu-2 determinant POOL
        # (field report: "daemonic processes are not allowed to have
        # children; running sequentially" -- an 81 s solve went serial).
        # Orphan protection comes from the pipes instead: the child
        # exits when the parent's end closes (host.main EOF path).
        proc = ctx.Process(target=_host_main, args=tuple(children),
                           name="circuitinsight-engine", daemon=False)
        proc.start()
        for c in children:                        # parent keeps its ends only
            c.close()
        self = cls(proc, parents)
        #: everything a respawn needs (hard cancel, crash recovery)
        self.recipe = (cin, op, dict(open_kw or {}), timeout)
        try:
            hello, _ = self._recv("glance", timeout, pump)
            if hello[0] != "hello" or hello[1] != PROTO:
                raise EngineError("ProtocolError",
                                  f"bad handshake: {hello!r}")
            from .. import __version__
            if hello[2] != __version__:
                raise EngineError(
                    "VersionMismatch",
                    f"engine {hello[2]} vs GUI {__version__} -- "
                    f"a stale install is answering")
            self._send("glance", ("open", str(cin), str(op),
                                  dict(open_kw or {})))
            self._snap = self._reply("glance", timeout=timeout, pump=pump)
        except BaseException:
            self.close(grace=0.5)
            raise
        return self

    # ------------------------------------------------------------ wire
    def _send(self, lane: str, msg) -> None:
        try:
            self._conns[lane].send_bytes(pickle.dumps(msg, _PICKLE))
        except (BrokenPipeError, OSError) as e:
            raise EngineDied(str(e)) from e

    def _recv(self, lane: str, timeout: float | None = None, pump=None):
        """Blocking receive; with `pump`, poll in short slices and call
        it between them -- the GUI thread keeps its event loop alive
        while it waits (the splash during open, first light)."""
        conn = self._conns[lane]
        try:
            if pump is not None:
                waited = 0.0
                while not conn.poll(0.05):
                    pump()
                    waited += 0.05
                    if timeout is not None and waited > timeout:
                        raise EngineError(
                            "Timeout",
                            f"no reply on {lane} within {timeout}s")
            elif timeout is not None and not conn.poll(timeout):
                raise EngineError("Timeout",
                                  f"no reply on {lane} within {timeout}s")
            data = conn.recv_bytes()
        except (EOFError, OSError) as e:
            raise EngineDied(str(e)) from e
        return pickle.loads(data), len(data)

    def _reply(self, lane: str, progress=None, note=None,
               timeout: float | None = None, pump=None):
        """Drain one call's messages to the terminal reply; returns the
        value and remembers the terminal frame's size in ._last_bytes."""
        while True:
            msg, size = self._recv(lane, timeout, pump)
            kind = msg[0]
            if kind == "progress":
                if progress is not None:
                    progress(*msg[1])
            elif kind == "note":
                if note is not None:
                    note(msg[1])
                elif self.on_note is not None:
                    self.on_note(msg[1])
            elif kind == "ok":
                self._last_bytes = size
                return msg[1]
            elif kind == "err":
                raise EngineError(msg[1], msg[2], msg[3])
            else:
                raise EngineError("ProtocolError",
                                  f"unexpected {kind!r} on {lane}")

    # ------------------------------------------------------------ calls
    def call(self, method: str, *args, lane: str = "glance",
             progress=None, note=None, timeout: float | None = None,
             pump=None, **kw):
        t0 = time.perf_counter()
        with self._locks[lane]:
            self._send(lane, ("call", method, args, kw,
                              progress is not None, note is not None))
            out = self._reply(lane, progress, note, timeout, pump)
        self.stats.append(dict(method=method, lane=lane,
                               seconds=time.perf_counter() - t0,
                               reply_bytes=self._last_bytes))
        return out

    def get(self, name: str, timeout: float | None = 60.0):
        """A property read, on the glance lane."""
        with self._locks["glance"]:
            self._send("glance", ("get", name))
            return self._reply("glance", timeout=timeout)

    def snapshot(self) -> dict:
        """The read-only mirror from open time; refresh() re-fetches."""
        return self._snap or {}

    def refresh(self) -> dict:
        self._snap = self.call("session_snapshot")
        return self._snap

    def cancel(self, lane: str = "solve") -> None:
        """Ask the engine to abandon the lane's running call at its next
        progress checkpoint (the soft half of plan E-F)."""
        with self._locks["glance"]:
            self._send("glance", ("cancel", lane))
            self._reply("glance", timeout=10.0)

    # ------------------------------------------------------------ life
    def alive(self) -> bool:
        return self._proc.is_alive()

    def kill(self) -> None:
        """Terminate without protocol: the hard-cancel path. Any thread
        blocked on a lane wakes with EngineDied."""
        try:
            self._proc.terminate()
            self._proc.join(2.0)
        except Exception:                         # noqa: BLE001
            pass
        for c in self._conns.values():
            try:
                c.close()
            except OSError:
                pass

    def close(self, grace: float = 2.0) -> None:
        try:
            with self._locks["glance"]:
                self._send("glance", ("close",))
                self._reply("glance", timeout=grace)
        except EngineError:
            pass                                  # dying is the goal
        self._proc.join(grace)
        if self._proc.is_alive():
            self._proc.terminate()
            self._proc.join(grace)
        for c in self._conns.values():
            try:
                c.close()
            except OSError:
                pass
