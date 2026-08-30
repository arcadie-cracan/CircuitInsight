"""The engine child process (engine-process plan E-B).

Runs the real SessionController in its own interpreter so the solve's
GIL never touches the GUI's. The parent talks over three
multiprocessing Connections -- the lanes -- with pickled tuples:

    ("hello", PROTO, version)           child -> parent, once, glance lane
    ("open", cin, op, kwargs)           parent -> child, once, glance lane
    ("call", method, args, kw,          parent -> child, any lane
             want_progress, want_note)
    ("get", name)                       parent -> child, property read
    ("cancel", lane)                    parent -> child, glance lane
    ("close",)                          parent -> child, glance lane
    ("ok", value)                       child -> parent, terminal reply
    ("err", type_name, str, traceback)  child -> parent, terminal reply
    ("progress", args_tuple)            child -> parent, mid-call,
                                        the callback's args VERBATIM
    ("note", text)                      child -> parent, mid-call

One outstanding request per lane (the proxy enforces it), so replies
need no request ids. The solve and advisor lanes run on their own
threads HERE, in the child -- their GIL contention is the child's
problem, which is the whole point.

MUST NEVER IMPORT QT, directly or transitively (test-pinned): spawn
re-imports this module, and the session layer is GUI-neutral by
charter.
"""
from __future__ import annotations

import os
import pickle
import threading
import traceback

PROTO = 1

_PICKLE = pickle.HIGHEST_PROTOCOL


class EngineCancelled(Exception):
    """Raised inside a session call when the parent asked to cancel."""


def _send(conn, msg) -> int:
    data = pickle.dumps(msg, _PICKLE)
    conn.send_bytes(data)
    return len(data)


def _serve(ses, conn, cancel_evt) -> None:
    """One lane's request loop: block, dispatch, reply. EOF means the
    parent is gone; the glance loop (in main) handles shutdown."""
    while True:
        try:
            msg = pickle.loads(conn.recv_bytes())
        except (EOFError, OSError):
            return
        kind = msg[0]
        if kind == "get":
            try:
                _send(conn, ("ok", getattr(ses, msg[1])))
            except BaseException as e:            # noqa: BLE001
                _send(conn, ("err", type(e).__name__, str(e),
                             traceback.format_exc()))
            continue
        if kind != "call":
            _send(conn, ("err", "ProtocolError",
                         f"unexpected message {kind!r} on a call lane", ""))
            continue
        _, method, args, kw, want_progress, want_note = msg
        cancel_evt.clear()

        def _progress(*args):
            # VERBATIM forwarding: the session's callbacks carry
            # whatever shape the caller expects -- interp sends
            # (done, total) evaluation units; coercing them to
            # (frac, text) broke the GUI's phase attribution
            if cancel_evt.is_set():
                raise EngineCancelled("cancelled by the parent")
            if want_progress:
                _send(conn, ("progress", args))

        if want_progress:
            kw["progress"] = _progress
        if want_note:
            kw["note"] = lambda text: _send(conn, ("note", str(text)))
        try:
            out = getattr(ses, method)(*args, **kw)
            _send(conn, ("ok", out))
        except BaseException as e:                # noqa: BLE001
            _send(conn, ("err", type(e).__name__, str(e),
                         traceback.format_exc()))


def _shutdown() -> None:
    """Exit the child, reaping the determinant pool first: atexit does
    not run under os._exit, and the pool's workers are OUR children now
    that the engine process is non-daemonic."""
    try:
        from . import interp
        pool = getattr(interp, "_POOL", None)
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)
    except Exception:                             # noqa: BLE001
        pass
    os._exit(0)


def main(glance_conn, solve_conn, advisor_conn) -> None:
    """Child entry point (the spawn target). Handshake, open, serve."""
    from .. import __version__

    _send(glance_conn, ("hello", PROTO, __version__))
    try:
        msg = pickle.loads(glance_conn.recv_bytes())
    except (EOFError, OSError):
        return
    if msg[0] != "open":
        _send(glance_conn, ("err", "ProtocolError",
                            f"expected open, got {msg[0]!r}", ""))
        return
    _, cin, op, kwargs = msg
    kwargs = dict(kwargs or {})
    stderr_path = kwargs.pop("_stderr_path", None)
    if stderr_path:
        # the child inherits the GUI's stderr; engine-side warnings
        # (region notes, lambdified overflow spew) belong in a file
        # the GUI tails into its Log, not on the launching console
        try:
            f = open(stderr_path, "a", buffering=1)
            os.dup2(f.fileno(), 2)
            import sys
            sys.stderr = f
        except OSError:
            pass                                  # keep the console then
        # one line per warning: the default formatter echoes the SOURCE
        # line of the warn call as a second line -- for a multi-line
        # call that echo is a bare "warnings.warn(", which read as
        # noise in the Log tail
        import warnings

        warnings.formatwarning = (
            lambda msg, cat, fn, lineno, line=None:
            f"{fn}:{lineno}: {cat.__name__}: {msg}\n")
    try:
        from ..session import SessionController
        ses = SessionController.open(cin, op, **kwargs)
        _send(glance_conn, ("ok", ses.session_snapshot()))
    except BaseException as e:                    # noqa: BLE001
        _send(glance_conn, ("err", type(e).__name__, str(e),
                            traceback.format_exc()))
        return

    events = {"solve": threading.Event(), "advisor": threading.Event()}
    for name, conn in (("solve", solve_conn), ("advisor", advisor_conn)):
        threading.Thread(target=_serve, args=(ses, conn, events[name]),
                         name=f"lane-{name}", daemon=True).start()

    glance_evt = threading.Event()                # glance calls are short;
    while True:                                   # cancel targets the others
        try:
            msg = pickle.loads(glance_conn.recv_bytes())
        except (EOFError, OSError):
            _shutdown()                           # parent died: no orphan
        kind = msg[0]
        if kind == "close":
            _send(glance_conn, ("ok", None))
            _shutdown()
        if kind == "cancel":
            events.get(msg[1], glance_evt).set()
            _send(glance_conn, ("ok", None))
            continue
        if kind == "get":
            try:
                _send(glance_conn, ("ok", getattr(ses, msg[1])))
            except BaseException as e:            # noqa: BLE001
                _send(glance_conn, ("err", type(e).__name__, str(e),
                                    traceback.format_exc()))
            continue
        if kind == "call":
            _, method, args, kw, want_progress, want_note = msg
            try:
                _send(glance_conn, ("ok", getattr(ses, method)(*args, **kw)))
            except BaseException as e:            # noqa: BLE001
                _send(glance_conn, ("err", type(e).__name__, str(e),
                                    traceback.format_exc()))
            continue
        _send(glance_conn, ("err", "ProtocolError",
                            f"unexpected message {kind!r}", ""))
