"""RemoteSession: the SessionController the GUI holds when the engine
lives in its own process (engine-process plan E-C).

Same names, same shapes: properties are served locally from the
versioned snapshot (the chatty paths never touch the pipe), methods
forward over the lane their workload belongs on, and mutating calls
refresh the snapshot before returning so the next property read is
already true. Everything the GUI can hold after E-A's surface freeze
crosses as a pickle; LoopGainUpdater re-derives itself on arrival (its
__getstate__ ships the recipe).

`open_engine` is the one entry point: CIN_ENGINE=inproc (or
mode="inproc") returns the plain in-process SessionController --
the default when no environment says otherwise, so the test suite
and headless scripts keep today's single-process path; launch.py
opts the real app into "proc".
"""
from __future__ import annotations

import os

from .proxy import EngineProxy

#: methods whose workload is the solve lane: the long, GIL-heavy jobs
#: the GUI runs through _launch
_SOLVE = {
    "solve", "simplify", "reduce_solve", "loop_gain", "impedance_result",
    "template_form", "attach_template", "expand_equivalents",
    "margin_ledger", "margin_ledger_set", "apply_reduction",
    "reduction_joint", "acground_joint", "approximation_report",
    "explain_numerals", "explain_per_numeral", "gft_quartet", "mode_loop",
    "suggest_compensation", "suggest_robust_compensation",
    "suggest_multi_compensation", "suggestion_robustness",
    "suggest_keep", "suggest_story_keep", "reduction_responses",
}

#: the advisor lane: the _run_bg background hints
_ADVISOR = {
    "estimate", "estimate_loop", "estimate_z", "rank_symbols",
    "rank_symbols_z", "order_certificate", "scan_ac_grounds",
    "scan_removals", "pole_attribution", "advise_split", "assess_probe",
    "match_conflicts", "suggest_matches", "preview_reduction",
}

#: calls that change session state: the snapshot refreshes before the
#: caller sees the return value
_MUTATORS = {
    "set_matches", "set_match_value_policy", "set_match_representative",
    "set_op_point", "apply_reduction", "revert_reduction",
    "declare_ac_loop_gain",
}

#: state-bearing calls a fresh engine must replay after a hard restart
#: (hard cancel, crash recovery): the mutators plus the settings that
#: live outside the snapshot's mutator set
_REPLAY = _MUTATORS | {"set_probe_backend"}

#: snapshot-served read-only properties (session properties + data
#: attributes), in the exact spelling the GUI uses
_SNAP_PROPS = (
    "cin_path", "ground", "nets", "probes", "tagged_probes", "ports",
    "reductions", "matches", "match_value_policy", "sym_aliases",
    "op_point_labels", "circuit_state", "cap_model", "mos_model",
    "ac_loop_gain",
)

_known_methods: set | None = None
_known_props: set | None = None


def _surface():
    """SessionController's public callables and properties, resolved
    once. The facade refuses names outside them: an unknown attribute
    must FAIL, never become a truthy bound forwarder -- bool(callable)
    silently unlocked a bench gate before this guard existed."""
    global _known_methods, _known_props
    if _known_methods is None:
        from ..session import SessionController
        _known_methods = {
            n for n, v in vars(SessionController).items()
            if callable(v) and not n.startswith("_")}
        _known_props = {
            n for n, v in vars(SessionController).items()
            if isinstance(v, property) and not n.startswith("_")}
    return _known_methods, _known_props


class RemoteSession:
    def __init__(self, proxy: EngineProxy):
        self._px = proxy
        self._devices = None
        #: how the engine warms display caches before pickling a
        #: Result (plan E-D); the GUI sets these at every launch
        self._display_prefs = {"base": True, "wrap": False}
        #: the ordered state-bearing calls since open -- what a fresh
        #: engine replays after a hard restart (plan E-F)
        self._replay: list[tuple] = []

    def set_display_prefs(self, *, base: bool, wrap: bool) -> None:
        self._display_prefs = {"base": bool(base), "wrap": bool(wrap)}

    # --------------------------------------------------------- plumbing
    # NO event pumping during calls: processEvents inside a blocking
    # engine call lets a timer handler make a NESTED call on the same
    # lane -- the per-lane lock is non-reentrant and the same thread,
    # a permanent freeze (measured: the proc-mode suite hung for an
    # hour). A GUI-thread call blocks synchronously, exactly as the
    # in-process session always has; true async first light is E-D's.
    def _call(self, name, args, kw):
        lane = ("solve" if name in _SOLVE
                else "advisor" if name in _ADVISOR else "glance")
        if name in _SOLVE:
            # Result-producing work warms its display caches ENGINE-
            # side, before the pickle; callbacks ride as proxy params
            progress = kw.pop("progress", None)
            note = kw.pop("note", None)
            out = self._px.call("call_prepared", name, tuple(args), kw,
                                dict(self._display_prefs), lane=lane,
                                progress=progress, note=note)
        else:
            out = self._px.call(name, *args, lane=lane, **kw)
        if name in _REPLAY:
            self._replay.append((name, args, kw))
        if name in _MUTATORS:
            self._px.refresh()
            self._devices = None
        return out

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        if name in _SNAP_PROPS:
            return self._px.snapshot()[name]
        if name == "has_stb":
            return self._px.snapshot()["has_stb"]
        if name == "op_points":
            # the GUI needs truthiness and len() only; the real point
            # records stay engine-side
            return list(range(self._px.snapshot().get("op_points_n", 0)))
        if name == "devices":
            if self._devices is None:
                from ..session import DeviceInfo
                self._devices = [DeviceInfo(**d)
                                 for d in self._px.snapshot()["devices"]]
            return self._devices
        methods, props = _surface()
        if name in props:
            return self._px.get(name)     # a property left un-mirrored:
                                          # served live off the engine
        if name not in methods:
            raise AttributeError(name)    # unknown names FAIL loudly

        def bound(*args, **kw):
            return self._call(name, args, kw)
        bound.__name__ = name
        return bound

    # ------------------------------------------------- explicit surface
    # (methods whose semantics need more than generic forwarding)
    def sources(self):
        return list(self._px.snapshot()["sources"])

    def input_ports(self):
        return list(self._px.snapshot()["input_ports"])

    def output_nets(self):
        return list(self._px.snapshot()["output_nets"])

    def analyses(self):
        return list(self._px.snapshot()["analyses"])

    def stb_probe(self):
        return self._px.snapshot()["stb_probe"]

    def set_alias(self, name: str, tex: str) -> None:
        self._px.call("set_alias", name, tex)
        self._replay.append(("set_alias", (name, tex), {}))
        snap = self._px.snapshot()
        if tex:
            snap["sym_aliases"][name] = tex
        else:
            snap["sym_aliases"].pop(name, None)

    def set_aliases(self, mapping: dict, *, merge: bool = False) -> None:
        self._px.call("set_aliases", dict(mapping), merge=merge)
        self._replay.append(("set_aliases", (dict(mapping),),
                             {"merge": merge}))
        snap = self._px.snapshot()
        if merge:
            snap["sym_aliases"].update(mapping)
        else:
            snap["sym_aliases"] = dict(mapping)

    def session_snapshot(self) -> dict:
        return self._px.snapshot()

    # ------------------------------------------------------------- life
    def engine_cancel(self, lane: str = "solve") -> None:
        """Soft cancel: the engine abandons the lane's running call at
        its next progress checkpoint and closes the call with an
        EngineCancelled error frame -- the lane stays protocol-clean."""
        self._px.cancel(lane)

    def engine_restart(self) -> None:
        """Hard cancel / crash recovery: kill the child, spawn a fresh
        one from the same recipe, replay every state-bearing call since
        open. Anything blocked on the old child wakes with EngineDied;
        shown results live GUI-side and survive untouched (plan E-F)."""
        cin, op, open_kw, timeout = self._px.recipe
        self._px.kill()
        self._px = EngineProxy.spawn(cin, op, open_kw=open_kw,
                                     timeout=timeout)
        for name, args, kw in list(self._replay):
            try:
                self._px.call(name, *args, **kw)
            except Exception:                     # noqa: BLE001
                pass          # a replay line that no longer applies
                              # must not block the restart
        self._px.refresh()
        self._devices = None

    def engine_alive(self) -> bool:
        return self._px.alive()

    def engine_stats(self) -> list[dict]:
        return self._px.stats

    def close_engine(self) -> None:
        self._px.close()


def open_engine(cin, op, *, mode: str | None = None, pump=None,
                stderr_path=None, on_note=None, **open_kw):
    """The one place that decides where the session lives. Default is
    IN-PROCESS: tests and headless scripts keep the single-interpreter
    path unless CIN_ENGINE (or mode=) says 'proc' -- launch.py sets it
    for the real app."""
    mode = (mode or os.environ.get("CIN_ENGINE") or "inproc").lower()
    if mode != "proc":
        from ..session import SessionController
        return SessionController.open(cin, op, **open_kw)
    kw = dict(open_kw)
    if stderr_path:
        kw["_stderr_path"] = str(stderr_path)
    px = EngineProxy.spawn(cin, op, open_kw=kw, pump=pump)
    px.on_note = on_note
    return RemoteSession(px)
