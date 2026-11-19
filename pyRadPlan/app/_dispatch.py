"""Transport-agnostic request dispatch of the application protocol (``web/PROTOCOL.md``)."""

from __future__ import annotations

import inspect
import itertools
import platform
import traceback
from typing import Any, Callable, Optional

from . import _analysis, _basedata, _io, _settings
from ._codec import decode, encode
from ._session import EventListener, PreconditionError, Session


class UnknownSessionError(KeyError):
    """No session with the requested id exists."""

    def __str__(self) -> str:
        return str(self.args[0]) if self.args else "Unknown session"


class UnknownMethodError(LookupError):
    """The request names a method the protocol does not define."""


class SessionManager:
    """Creates, looks up and closes sessions by id.

    One manager serves one process: under Pyodide one per web worker, in a server one
    per process with one session per user or connection.

    Parameters
    ----------
    factory
        Called with the new session's id (and any options) to create a session.
    max_sessions
        Refuse to create more sessions than this (``None``: unlimited).
    allow_pickle
        Whether sessions may load pickle files, which run code (never under Pyodide).
    allow_global_settings
        Whether requests may change what is shared by the whole process: ``app.settings_set``
        and ``base_data.add`` / ``base_data.remove``. A host serving several users turns
        this off.

    Notes
    -----
    The dispatch surface assumes one trusted user per process: the log capture is a handler
    on the global ``pyRadPlan`` logger and the settings are process-wide, so the flags
    only guard a single-user host against surprising requests; they are no isolation
    between users.
    """

    def __init__(
        self,
        factory: Callable[..., Session] = Session,
        max_sessions: Optional[int] = None,
        *,
        allow_pickle: bool = True,
        allow_global_settings: bool = True,
    ) -> None:
        self._factory = factory
        self._max_sessions = max_sessions
        self.allow_pickle = allow_pickle
        self.allow_global_settings = allow_global_settings
        self._sessions: dict[str, Session] = {}
        self._counter = itertools.count(1)

    def create(self, session_id: Optional[str] = None, **options: Any) -> Session:
        """Create a session (with a fresh ``"s<n>"`` id unless one is given)."""
        if self._max_sessions is not None and len(self._sessions) >= self._max_sessions:
            raise RuntimeError(f"At most {self._max_sessions} sessions are allowed")
        if session_id is None:
            session_id = f"s{next(self._counter)}"
            while session_id in self._sessions:
                session_id = f"s{next(self._counter)}"
        elif session_id in self._sessions:
            raise ValueError(f"Session {session_id!r} exists already")
        session = self._factory(session_id, **options)
        session.allow_pickle = session.allow_pickle and self.allow_pickle
        self._sessions[session_id] = session
        return session

    def get(self, session_id: Optional[str]) -> Session:
        """Return the session with *session_id*."""
        try:
            return self._sessions[session_id]
        except KeyError:
            raise UnknownSessionError(f"Unknown session {session_id!r}") from None

    def close(self, session_id: str) -> None:
        """Close and forget a session."""
        self.get(session_id).close()
        del self._sessions[session_id]

    def close_all(self) -> None:
        """Close every session."""
        for session_id in list(self._sessions):
            self.close(session_id)

    def ids(self) -> list[str]:
        """Return the ids of the open sessions."""
        return list(self._sessions)

    def __len__(self) -> int:
        return len(self._sessions)


#: Method name -> (handler, needs a session). Handlers take the manager, the session (or
#: ``None``) and the decoded params as keyword arguments.
_METHODS: dict[str, tuple[Callable[..., Any], bool]] = {}


def method(name: str, session: bool = True) -> Callable[[Callable], Callable]:
    """Register a protocol method; *session* says whether it acts on a session."""

    def register(handler: Callable) -> Callable:
        _METHODS[name] = (handler, session)
        return handler

    return register


def available_methods() -> list[str]:
    """Names of the registered protocol methods."""
    return sorted(_METHODS)


def app_info() -> dict[str, Any]:
    """Version, Python, array backend and the registered phantoms, machines and solvers."""
    import pyRadPlan  # noqa: PLC0415
    from pyRadPlan import settings  # noqa: PLC0415
    from pyRadPlan.dose.engines import get_available_engines  # noqa: PLC0415
    from pyRadPlan.io import available_phantoms  # noqa: PLC0415
    from pyRadPlan.machines import get_available_machines  # noqa: PLC0415
    from pyRadPlan.optimization.solvers import get_available_solvers  # noqa: PLC0415

    machines = get_available_machines()
    modes = list(dict.fromkeys(["photons", *machines]))
    backend = settings.xp.preferred_cpu_array_backend
    if settings.xp.preferred_gpu_array_backend:
        backend = f"{backend} (gpu: {settings.xp.preferred_gpu_array_backend})"
    return {
        "version": getattr(pyRadPlan, "__version__", None),
        "python": platform.python_version(),
        "platform": platform.system() or platform.platform(),
        "backend": backend,
        "phantoms": available_phantoms(),
        "machines": machines,
        "solvers": list(get_available_solvers()),
        "engines": {mode: list(get_available_engines(mode)) for mode in modes},
        "methods": available_methods(),
    }


@method("app.info", session=False)
def _app_info(manager: SessionManager) -> dict:
    return {**app_info(), **_settings.runtime_info(manager.allow_pickle)}


@method("session.create", session=False)
def _session_create(
    manager: SessionManager, profile: Optional[str] = None, options: Optional[dict] = None
) -> dict:
    session = manager.create(profile=profile, options=options)
    return {"session": session.id, "options": session.state()["options"]}


@method("session.close")
def _session_close(manager: SessionManager, session: Session) -> dict:
    manager.close(session.id)
    return {}


@method("session.state")
def _session_state(session: Session) -> dict:
    return session.state()


@method("data.load_phantom")
def _load_phantom(session: Session, name: str) -> dict:
    session.load_phantom(name)
    return session.state()


@method("data.load_file")
def _load_file(session: Session, filename: str, data: Any, kind: str = "auto") -> dict:
    _io.load_file(session, filename, data, kind)
    return session.state()


@method("plan.get")
def _plan_get(session: Session, detail: bool = False) -> Optional[dict]:
    return session.plan_detail() if detail else session.get_plan()


@method("plan.set")
def _plan_set(session: Session, **changes: Any) -> dict:
    session.set_plan(changes)
    return session.state()


@method("workflow.generate_stf")
def _generate_stf(session: Session) -> dict:
    session.generate_stf()
    return session.state()


@method("workflow.calc_dose_influence")
def _calc_dose_influence(session: Session, physical_only: bool = False) -> dict:
    session.calc_dose_influence(physical_only=physical_only)
    return session.state()


@method("workflow.optimize")
def _optimize(
    session: Session, solver: Optional[str] = None, max_iter: Optional[int] = None
) -> dict:
    summary = session.optimize(solver=solver, max_iter=max_iter)
    return {**session.state(), "optimization": summary}


@method("workflow.recalc")
def _recalc(session: Session, physical_only: bool = False) -> dict:
    session.recalc(physical_only=physical_only)
    return session.state()


@method("workflow.run_all")
def _run_all(
    session: Session,
    solver: Optional[str] = None,
    max_iter: Optional[int] = None,
    physical_only: bool = False,
) -> dict:
    session.run_all(solver=solver, max_iter=max_iter, physical_only=physical_only)
    return session.state()


@method("view.volume")
def _volume(session: Session, name: str, phase: int = 0, encoding: str = "float32") -> dict:
    return session.views.volume(name, phase, encoding)


@method("view.structure_masks")
def _structure_masks(
    session: Session, names: Optional[list] = None, encoding: str = "uint8"
) -> dict:
    return session.views.structure_masks(names, encoding)


@method("analysis.dvh")
def _dvh(
    session: Session,
    quantity: Optional[str] = None,
    names: Optional[list] = None,
    num_points: int = 500,
    keys: Optional[list] = None,
    range: str = "shared",  # noqa: A002 - protocol parameter name
) -> dict:
    if keys is not None:
        return _analysis.dvh_sets(session, keys, names, num_points, range)
    return session.views.dvh(quantity, names, num_points, range)


# App, files, base data, views and analysis.


def _require_global_settings(manager: SessionManager) -> None:
    if not manager.allow_global_settings:
        raise PreconditionError(
            "This host does not allow changing settings or base data shared by all sessions",
            "not_available",
        )


def _refuse_while_busy(manager: SessionManager) -> None:
    busy = [s for s in manager._sessions.values() if s.busy]
    if busy:
        raise PreconditionError(f"{busy[0]._busy.label} is still running", "busy")


@method("app.licenses", session=False)
def _app_licenses(name: Optional[str] = None) -> Any:
    return _settings.licenses(name)


@method("app.settings", session=False)
def _app_settings() -> dict:
    return _settings.settings_sections()


@method("app.settings_set", session=False)
def _app_settings_set(manager: SessionManager, section: str, values: dict) -> dict:
    _require_global_settings(manager)
    _refuse_while_busy(manager)
    return _settings.settings_set(section, values)


@method("session.log_level")
def _session_log_level(session: Session, level: str) -> dict:
    import logging  # noqa: PLC0415

    levels = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
    }
    if level not in levels:
        raise ValueError(f"Unknown log level {level!r}; use one of {list(levels)}")
    if session.busy:
        raise PreconditionError(f"{session._busy.label} is still running", "busy")
    session.log_level = levels[level]
    return {"level": level}


@method("data.load_dij")
def _load_dij(session: Session, filename: str, data: Any) -> dict:
    _io.load_dij(session, filename, data)
    return session.state()


@method("export.save")
def _export_save(
    session: Session,
    object: str,  # noqa: A002 - protocol parameter name
    format: str,  # noqa: A002 - protocol parameter name
    quantities: Optional[list] = None,
    structure_format: Optional[str] = None,
    basename: Optional[str] = None,
) -> dict:
    return _io.save(session, object, format, quantities, structure_format, basename)


@method("base_data.list", session=False)
def _base_data_list(kind: str) -> dict:
    return _basedata.list_kind(kind)


@method("base_data.add", session=False)
def _base_data_add(
    manager: SessionManager,
    kind: str,
    filename: str,
    data: Any,
    name: Optional[str] = None,
    radiation_mode: Optional[str] = None,
    overwrite: bool = False,
    dry_run: bool = False,
) -> dict:
    if not dry_run:
        _require_global_settings(manager)
        _refuse_while_busy(manager)
    return _basedata.add(
        kind, filename, data, name, radiation_mode, overwrite, dry_run, manager.allow_pickle
    )


@method("base_data.remove", session=False)
def _base_data_remove(manager: SessionManager, kind: str, file: str) -> dict:
    _require_global_settings(manager)
    _refuse_while_busy(manager)
    return _basedata.remove(kind, file)


@method("view.beams")
def _view_beams(session: Session, draft: Optional[dict] = None) -> dict:
    from ._views import beam_geometry  # noqa: PLC0415

    ws = session.workspace
    return beam_geometry(ws.pln, ws.stf, ws.cst, draft)


@method("analysis.qi")
def _analysis_qi(
    session: Session,
    quantity: str,
    names: Optional[list] = None,
    ref_vols: Optional[list] = None,
    ref_doses: Optional[list] = None,
) -> dict:
    return _analysis.quality_indicators(session, quantity, names, ref_vols, ref_doses)


@method("analysis.compare_units")
def _analysis_compare_units(session: Session, a: str, b: str) -> dict:
    return _analysis.compare_quantity_units(session, a, b)


_default_manager = SessionManager()


def default_manager() -> SessionManager:
    """Return the process-wide manager :func:`dispatch` uses when none is passed."""
    return _default_manager


def _call(handler: Callable, manager: SessionManager, session: Optional[Session], params: dict):
    context = {"manager": manager, "session": session}
    reserved = set(params) & set(context)
    if reserved:
        raise TypeError(f"Reserved parameter name(s): {sorted(reserved)}")
    parameters = inspect.signature(handler).parameters
    if not any(p.kind is p.VAR_KEYWORD for p in parameters.values()):
        unknown = set(params) - set(parameters)
        if unknown:
            raise TypeError(f"Unexpected parameter(s): {sorted(unknown)}")
    missing = [
        name
        for name, p in parameters.items()
        if name not in context
        and p.kind is p.POSITIONAL_OR_KEYWORD
        and p.default is p.empty
        and name not in params
    ]
    if missing:
        raise TypeError(f"Missing parameter(s): {missing}")
    kwargs = {k: v for k, v in context.items() if k in parameters}
    return handler(**kwargs, **params)


def _error_response(request_id: Any, exc: BaseException, with_traceback: bool) -> dict:
    message = str(exc)
    if isinstance(exc, KeyError) and not isinstance(exc, UnknownSessionError) and exc.args:
        message = str(exc.args[0])
    error = {"type": getattr(exc, "protocol_type", None) or type(exc).__name__, "message": message}
    code = getattr(exc, "code", None)
    if isinstance(code, str):
        error["code"] = code
    details = _error_details(exc)
    if details is not None:
        error["details"] = details
        error.setdefault("code", "invalid")
    if with_traceback:
        error["traceback"] = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    return {"id": request_id, "ok": False, "error": error}


def dispatch(
    request: dict,
    emit: Optional[EventListener] = None,
    *,
    manager: Optional[SessionManager] = None,
    with_traceback: bool = True,
) -> dict:
    """Handle one protocol request and return its response.

    Parameters
    ----------
    request
        ``{"id", "session"?, "method", "params"?, "buffers"?}``; array descriptors in the
        params index ``buffers``.
    emit
        Receives the session's events while the request runs.
    manager
        Session manager to use; the process-wide :func:`default_manager` when omitted.
    with_traceback
        Include the formatted traceback in error responses.

    Returns
    -------
    dict
        ``{"id", "ok": True, "result", "buffers"}`` or ``{"id", "ok": False, "error"}``.
        Errors (including :class:`KeyboardInterrupt`, which Pyodide's interrupt buffer
        raises to cancel a run) never propagate.
    """
    manager = manager if manager is not None else _default_manager
    request_id = request.get("id") if isinstance(request, dict) else None
    try:
        if not isinstance(request, dict):
            raise TypeError("A request must be a dict")
        name = request.get("method")
        if name not in _METHODS:
            raise UnknownMethodError(f"Unknown method {name!r}")
        handler, needs_session = _METHODS[name]
        params = decode(request.get("params") or {}, request.get("buffers") or ())
        if not isinstance(params, dict):
            raise TypeError("params must be an object")
        session = manager.get(request.get("session")) if needs_session else None
        if session is not None:
            with session.listening(emit):
                result = _call(handler, manager, session, params)
        else:
            result = _call(handler, manager, None, params)
            if name == "session.create" and emit is not None:
                created = manager.get(result["session"])
                emit({"event": "state", "session": created.id, "state": created.state()})
        payload, buffers = encode(result)
        return {"id": request_id, "ok": True, "result": payload, "buffers": buffers}
    except (Exception, KeyboardInterrupt) as exc:  # noqa: BLE001 - errors become responses
        return _error_response(request_id, exc, with_traceback)


def _error_details(exc: BaseException) -> Optional[list]:
    details = getattr(exc, "details", None)
    if isinstance(details, list):
        return details
    from pydantic import ValidationError  # noqa: PLC0415

    if isinstance(exc, ValidationError):
        from ._forms import validation_details  # noqa: PLC0415

        return validation_details(exc)
    return None


from . import _methods_editors  # noqa: E402, F401
