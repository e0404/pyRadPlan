"""A planning session: workspace, workflow operations, stage tracking, views and events."""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Mapping, Optional, Sequence

import numpy as np

from pyRadPlan.core import (
    ComputeCancelledError,
    ComputeControl,
    ProgressReport,
    StatusReport,
    WarningReport,
    observe_control,
    observe_reports,
)

from ._plan import DEFAULT_PLAN, build_plan, plan_summary
from ._profiles import running_in_pyodide
from ._results import carry_forward, is_tagged
from ._session_v1 import SessionV1Mixin
from ._stages import (
    SCENARIO_LABELS_KEY,
    SCENARIO_SUFFIX,
    StageTracker,
    available_actions,
    merge_loaded_data,
    nested_fraction,
    progress_detail,
    recalculation_angles_match,
    stage_status,
    status_text,
)
from ._views import ViewCache, grid_info, result_quantities
from ._workspace import WORKSPACE_KEYS, Workspace

logger = logging.getLogger(__name__)

#: An event sink, receiving the event dicts of ``web/PROTOCOL.md``.
EventListener = Callable[[dict], None]

#: Sessions emitting an event in the current context; an event emitted by a listener of
#: the same session is dropped instead of recursing.
_EMITTING: ContextVar[tuple[int, ...]] = ContextVar("pyRadPlan_app_emitting", default=())


class PreconditionError(RuntimeError):
    """A workflow step cannot run in the session's current state.

    Parameters
    ----------
    message
        What is missing or wrong, addressed to the user.
    code
        Machine-readable reason a frontend can branch on, e.g. to offer
        ``physical_only`` for ``"bio_model_unsupported"``.
    details
        JSON-able entries the protocol sends as ``error.details`` (e.g. tissue issues).
    data
        Python objects for an in-process frontend that builds its own message, e.g. the
        ``TissueIssue`` objects of ``"bio_model_unsupported"``; never serialized.
    """

    def __init__(
        self,
        message: str,
        code: str = "not_available",
        details: Optional[list] = None,
        data: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.details = details
        self.data = dict(data or {})


@dataclass
class Operation:
    """A prepared workflow step, split so a frontend can choose where it runs.

    :meth:`Session.prepare` validates the preconditions and captures the inputs; only
    :attr:`compute` does expensive work and it never touches the workspace, so it can run
    on a worker thread (Qt) or synchronously (Pyodide); :attr:`commit` writes the result
    to the workspace and belongs on the thread owning the workspace. Hosts schedule the
    parts through :meth:`Session.begin`, :meth:`Session.run`, :meth:`Session.finish` and
    :meth:`Session.abort`.
    """

    name: str
    label: str
    compute: Callable[[], Any]
    commit: Callable[[Any], Any]
    control: ComputeControl = field(default_factory=ComputeControl)
    #: Iteration cap of an optimization, used to turn its status into a fraction.
    max_iter: Optional[int] = None
    #: Non-blocking issues found while preparing (e.g. ``TissueIssue`` warnings), already
    #: reported as ``log`` events.
    warnings: list = field(default_factory=list)
    #: A stop request ends the computation with its result so far (an optimization keeps
    #: its last iterate) instead of discarding it.
    keeps_partial_result: bool = False


class _EventLogHandler(logging.Handler):
    """Forwards log records emitted during an operation as ``log`` events."""

    def __init__(self, session: Session, level: int) -> None:
        super().__init__(level)
        self._session = session
        self.setFormatter(logging.Formatter("%(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = self.format(record)
        except Exception:  # noqa: BLE001 - a malformed record must not break compute
            message = str(record.msg)
        self._session._log(record.levelname.lower(), message, record.name)


@contextmanager
def _interrupt_requests_stop(operation: Operation) -> Iterator[None]:
    """Under Pyodide, let the first interrupt stop *operation* the way the Qt Stop does.

    The browser's Stop writes SIGINT to Pyodide's interrupt buffer, which raises
    ``KeyboardInterrupt`` at the next bytecode check and so discards the run. For an
    operation keeping its partial result, the first interrupt requests a cooperative stop
    through its control instead (an optimization ends after the current iteration); a
    second one raises ``KeyboardInterrupt`` as before.

    Library code can turn the ``KeyboardInterrupt`` into another error: numpy reports any
    exception raised during the dtype promotion of a structured comparison (which runs
    Python code, e.g. inside ``np.unique(..., axis=...)``) as a ``TypeError``. An error
    escaping after the interrupt was raised is therefore re-raised as the interrupt.
    """
    if not running_in_pyodide():
        yield
        return
    import signal  # noqa: PLC0415

    control = operation.control
    interrupted = False

    def handler(signum: int, frame: Any) -> None:
        nonlocal interrupted
        if operation.keeps_partial_result and not control.stop_requested:
            control.request_stop()
            return
        interrupted = True
        raise KeyboardInterrupt

    try:
        previous = signal.signal(signal.SIGINT, handler)
    except ValueError:  # not the main thread
        yield
        return
    try:
        yield
    except Exception as exc:
        if interrupted and not isinstance(exc, ComputeCancelledError):
            raise KeyboardInterrupt from exc
        raise
    finally:
        signal.signal(signal.SIGINT, previous)


def dij_payload(loaded: Mapping[str, Any], name: str) -> dict[str, Any]:
    """Return the steering information and dose influence of a loaded file (Qt "Load Dij").

    Raises
    ------
    PreconditionError
        ``code="no_dij"`` if the file holds no dose influence matrix.
    """
    found = {key: loaded[key] for key in ("stf", "dij") if loaded.get(key) is not None}
    if "dij" not in found:
        raise PreconditionError(f"No dose-influence matrix found in {name}", "no_dij")
    return found


def _derived(build: Callable[[], Any], default: Any) -> Any:
    try:
        return build()
    except Exception:  # noqa: BLE001 - the state reports what it cannot derive as missing
        logger.debug("Could not derive a part of the session state", exc_info=True)
        return default


def _json_number(value: Any) -> Any:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        value = float(value)
        return value if np.isfinite(value) else None
    if isinstance(value, str):
        return value
    if np.ndim(value) == 0:
        try:
            return _json_number(value.item())
        except (AttributeError, ValueError, TypeError):
            pass
    return None


class Session(SessionV1Mixin):
    """One planning session, independent of any GUI toolkit.

    Owns (or wraps) a :class:`Workspace` and offers the workflow operations of the Qt
    workflow widget as plain method calls. The convenience methods run synchronously in
    the calling thread; a frontend that wants the work elsewhere uses :meth:`prepare` and
    schedules the :class:`Operation` with :meth:`begin`, :meth:`run` and :meth:`finish`
    (see ``docs/development/app_layer.md``).

    Parameters
    ----------
    session_id
        Identifier put into every event.
    workspace
        Workspace to work on; a new one when omitted.
    auto_plan
        Apply :data:`default_plan` as soon as CT and structures are loaded without a plan,
        as the Qt plan form does.
    default_plan
        Partial ``PlanSummary`` of the plan applied by *auto_plan* and :meth:`run_all`.
    solver_preference
        Solvers tried in order when neither the request nor the plan names one; the first
        registered one is used. Under Pyodide only ``"scipy"`` is registered. Empty: the
        optimization problem's default solver applies.
    max_iter
        Default iteration cap of the optimization; ``None`` keeps the solver's.
    log_level
        If given, the ``pyRadPlan`` logger is lowered to this level while an operation
        runs, so its records reach the ``log`` events. Records always propagate as usual,
        so a console handler would print them too.
    profile
        ``"native"`` or ``"web"`` (default: ``"web"`` under Pyodide); gives the default
        plan unless *default_plan* is passed, and the locked plan fields.
    options
        Overrides of the profile (``default_plan`` entries are merged, ``locked_fields``
        replaced).
    capture_logs
        Forward the ``pyRadPlan`` log records emitted while an operation runs as ``log``
        events. A host with its own log handler (the Qt log panel) turns this off.
    """

    def __init__(  # noqa: PLR0913 - keyword-only host options
        self,
        session_id: str = "s1",
        *,
        workspace: Optional[Workspace] = None,
        auto_plan: bool = True,
        default_plan: Optional[Mapping[str, Any]] = None,
        solver_preference: Sequence[str] = ("ipopt", "scipy"),
        max_iter: Optional[int] = None,
        log_level: Optional[int] = None,
        profile: Optional[str] = None,
        options: Optional[Mapping[str, Any]] = None,
        capture_logs: bool = True,
    ) -> None:
        self.id = session_id
        self._owns_workspace = workspace is None
        self.workspace = workspace if workspace is not None else Workspace()
        self.auto_plan = auto_plan
        self.default_plan = dict(DEFAULT_PLAN if default_plan is None else default_plan)
        self.solver_preference = tuple(solver_preference)
        self.max_iter = max_iter
        self.log_level = log_level
        self.capture_logs = capture_logs

        self._listeners: list[EventListener] = []
        self._tracker = StageTracker().update([], self.workspace.present(), self.workspace.cst)
        self.views = ViewCache(self.workspace.get)
        self._busy: Optional[Operation] = None
        self._idle_callbacks: list[Callable[[], Any]] = []
        self._logged: set[tuple[str, str]] = set()
        self._init_v1(profile, options, explicit_default_plan=default_plan is not None)
        self._unsubscribe = self.workspace.subscribe(self._on_workspace_changed)

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    def add_listener(self, callback: EventListener) -> Callable[[], None]:
        """Receive this session's events; returns a function removing the listener."""
        self._listeners.append(callback)

        def remove() -> None:
            try:
                self._listeners.remove(callback)
            except ValueError:
                pass

        return remove

    @contextmanager
    def listening(self, callback: Optional[EventListener]) -> Iterator[None]:
        """Add *callback* as a listener for the duration of the ``with`` block."""
        if callback is None:
            yield
            return
        remove = self.add_listener(callback)
        try:
            yield
        finally:
            remove()

    def _emit(self, event: dict) -> None:
        active = _EMITTING.get()
        if not self._listeners or id(self) in active:
            return
        token = _EMITTING.set((*active, id(self)))
        try:
            event = {"event": event["event"], "session": self.id, **event}
            for callback in list(self._listeners):
                try:
                    callback(event)
                except Exception:  # noqa: BLE001 - a listener must never break compute
                    logging.getLogger("pyRadPlan.app.events").debug(
                        "Event listener raised", exc_info=True
                    )
        finally:
            _EMITTING.reset(token)

    def _emit_state(self) -> None:
        if self._listeners and id(self) not in _EMITTING.get():
            self._emit({"event": "state", "state": self.state()})

    def _log(self, level: str, message: str, logger_name: str, report: bool = False) -> None:
        # Algorithms both log a warning and report it; the log shows it once.
        repeat = (level, message) in self._logged
        if repeat and not report:
            return
        if self._busy is not None:
            self._logged.add((level, message))
        event = {"event": "log", "level": level, "message": message, "logger": logger_name}
        if report:
            # A WarningReport: what the Qt workflow collects and shows after the run.
            event["report"] = True
            if repeat:
                event["repeat"] = True
        self._emit(self.history.number(event))

    def _warn(self, message: str) -> None:
        self._log("warning", message, "pyRadPlan.app", report=True)

    def _on_workspace_changed(self, changed_keys: list[str]) -> None:
        ws = self.workspace
        self._tracker = self._tracker.update(changed_keys, ws.present(), ws.cst)
        self.views.invalidate()
        self._v1_workspace_changed(changed_keys)
        # A running operation emits the state once it ends, after all of its writes.
        if self._busy is None:
            self._emit_state()

    def _on_report(self, operation: Operation) -> Callable[[Any], None]:
        def observe(report: Any) -> None:
            if isinstance(report, WarningReport):
                self._warn(report.message)
            elif isinstance(report, ProgressReport):
                if not report.levels:
                    return
                self._emit(
                    {
                        "event": "progress",
                        "label": operation.label,
                        "fraction": nested_fraction(report.levels),
                        "detail": progress_detail(report.levels),
                    }
                )
            elif isinstance(report, StatusReport):
                data = {key: _json_number(value) for key, value in report.data.items()}
                data = {key: value for key, value in data.items() if value is not None}
                iteration = data.get("iteration")
                fraction = None
                if operation.max_iter and isinstance(iteration, (int, float)):
                    fraction = min(max(float(iteration) / operation.max_iter, 0.0), 1.0)
                self._v1_status(operation, data)
                detail = " · ".join(
                    f"{key} {value:.6g}" if isinstance(value, float) else f"{key} {value}"
                    for key, value in data.items()
                )
                self._emit(
                    {
                        "event": "progress",
                        "label": operation.label,
                        "fraction": fraction,
                        "detail": detail or report.message,
                        "data": data,
                    }
                )

        return observe

    @contextmanager
    def _capture_logs(self) -> Iterator[None]:
        if not self.capture_logs:
            yield
            return
        package_logger = logging.getLogger("pyRadPlan")
        handler = _EventLogHandler(self, self.log_level or logging.NOTSET)
        previous = package_logger.level
        if self.log_level is not None and package_logger.getEffectiveLevel() > self.log_level:
            package_logger.setLevel(self.log_level)
        package_logger.addHandler(handler)
        try:
            yield
        finally:
            package_logger.removeHandler(handler)
            package_logger.setLevel(previous)

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    @property
    def tracker(self) -> StageTracker:
        """The staleness flags of the computed products."""
        return self._tracker

    @property
    def busy(self) -> bool:
        """Whether an operation is running."""
        return self._busy is not None

    def stages(self) -> dict[str, str]:
        """Status (missing / done / stale) of every workflow stage."""
        return stage_status(self.workspace.present(), self._tracker)

    def actions(self) -> dict[str, bool]:
        """Which workflow actions can run now."""
        if self._busy is not None:
            return dict.fromkeys(available_actions(()), False)
        return available_actions(self.workspace.present())

    def state(self) -> dict[str, Any]:
        """Return the JSON-able ``SessionState`` (see ``web/PROTOCOL.md``).

        Parts that cannot be derived from the stored objects (e.g. placeholders a host
        put into the workspace) are reported as missing instead of raising.
        """
        ws = self.workspace
        present = ws.present()

        def ct_info() -> Optional[dict[str, Any]]:
            if ws.ct is None:
                return None
            scenarios = int(getattr(ws.ct, "num_of_ct_scen", 1))
            return {**grid_info(ws.ct), "num_scenarios": scenarios}

        def structures() -> list[dict[str, Any]]:
            if ws.cst is None:
                return []
            return [
                {
                    "name": voi.name,
                    "type": voi.voi_type,
                    "color": [int(c) for c in voi.color],
                    "visible": bool(voi.visible),
                    "index": i,
                }
                for i, voi in enumerate(ws.cst.vois)
            ]

        state = {
            "session": self.id,
            "has": {key: key in present for key in WORKSPACE_KEYS},
            "stages": self.stages(),
            "actions": self.actions(),
            "ct": _derived(ct_info, None),
            "plan": _derived(lambda: plan_summary(ws.pln), None),
            "quantities": result_quantities(ws.result),
            "structures": _derived(structures, []),
            "status": self._busy.label if self._busy else status_text(present, self._tracker),
            "busy": self._busy.label if self._busy else None,
        }
        return self._v1_state(state)

    # ------------------------------------------------------------------
    # Host-scheduled operations
    # ------------------------------------------------------------------

    def begin(self, operation: Operation) -> None:
        """Mark *operation* as running (on the thread owning the workspace).

        Starts its history and emits the busy state. Every :meth:`begin` must be followed
        by :meth:`finish` or :meth:`abort`.

        Raises
        ------
        PreconditionError
            ``code="busy"`` while another operation runs.
        """
        if self._busy is not None:
            raise PreconditionError(f"{self._busy.label} is still running", "busy")
        self._busy = operation
        self._logged.clear()
        self._v1_begin(operation)
        self._emit_state()

    def run(self, operation: Operation) -> Any:
        """Compute *operation* and return its value; may run on a worker thread.

        Installs the report, control and log observation in the calling thread, so the
        progress, status and warning reports of the algorithms become events. Never writes
        the workspace.
        """
        if self._busy is not operation:
            raise RuntimeError(f"{operation.label}: call begin() before run()")
        with (
            observe_reports(self._on_report(operation)),
            observe_control(operation.control),
            self._capture_logs(),
            _interrupt_requests_stop(operation),
        ):
            return operation.compute()

    def finish(self, operation: Operation, value: Any) -> Any:
        """Commit the computed *value* to the workspace and end *operation*.

        Returns what the commit returns. A failing commit aborts the operation and
        re-raises.
        """
        if self._busy is not operation:
            raise RuntimeError(f"{operation.label} is not running")
        try:
            committed = operation.commit(value)
        except BaseException as exc:
            self.abort(operation, exc)
            raise
        self._v1_end(operation, "stopped" if operation.control.stop_requested else "done")
        self._emit({"event": "progress", "label": operation.label, "fraction": 1.0, "detail": ""})
        self._busy = None
        self._emit_state()
        self._run_idle_callbacks()
        return committed

    def abort(self, operation: Operation, exc: Optional[BaseException] = None) -> None:
        """End *operation* without a result: ``"stopped"`` when cancelled, else ``"failed"``."""
        if self._busy is not operation:
            return
        cancelled = isinstance(exc, (ComputeCancelledError, KeyboardInterrupt))
        self._v1_end(operation, "stopped" if cancelled else "failed")
        self._busy = None
        self._emit_state()
        self._run_idle_callbacks()

    def call_when_idle(self, callback: Callable[[], Any]) -> None:
        """Call *callback* now, or once the running operation has ended (or was aborted).

        Lets an in-process frontend keep an edit the session refuses while busy, e.g. a
        structure edited while a calculation runs, and apply it afterwards on the thread
        ending the operation. A failing callback is logged.
        """
        if self._busy is None:
            callback()
        else:
            self._idle_callbacks.append(callback)

    def _run_idle_callbacks(self) -> None:
        callbacks, self._idle_callbacks = self._idle_callbacks, []
        for callback in callbacks:
            try:
                callback()
            except Exception:  # noqa: BLE001 - one deferred edit must not break the others
                logger.exception("A deferred edit failed")

    # ------------------------------------------------------------------
    # Running operations
    # ------------------------------------------------------------------

    def execute(self, operation: Operation) -> Any:
        """Run a prepared operation synchronously and commit its result.

        Progress, status and warning reports of the pyRadPlan algorithms it calls become
        ``progress`` / ``log`` events; the state is emitted when it starts and ends. The
        same as :meth:`begin`, :meth:`run` and :meth:`finish` in one thread.
        """
        self.begin(operation)
        try:
            value = self.run(operation)
        except BaseException as exc:
            self.abort(operation, exc)
            raise
        return self.finish(operation, value)

    def cancel(self) -> None:
        """Request a cooperative stop of the running operation (no-op when idle).

        Only reachable from another thread, or under Pyodide through its interrupt
        buffer (which raises :class:`KeyboardInterrupt` instead).
        """
        if self._busy is not None:
            self._busy.control.request_stop()

    def prepare(self, action: str, **params: Any) -> Operation:
        """Validate and capture the inputs of *action*, returning an :class:`Operation`."""
        builder = getattr(self, f"_prepare_{action}", None)
        if builder is None:
            raise ValueError(f"Unknown operation {action!r}")
        return builder(**params)

    def _require(self, action: str) -> None:
        if self._busy is not None:
            raise PreconditionError(f"{self._busy.label} is still running", "busy")
        if not available_actions(self.workspace.present())[action]:
            needs = {
                "generate_stf": "a CT, structures and a plan",
                "calc_dose_influence": "a CT, structures and a plan",
                "optimize": "steering information and a dose influence matrix",
                "recalc": "a dose influence matrix and an optimized result",
            }[action]
            raise PreconditionError(f"{action} needs {needs}")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def _commit_loaded(self, data: Mapping[str, Any]) -> list[str]:
        ws = self.workspace
        payload, warnings = merge_loaded_data(data, ws.has("ct"), WORKSPACE_KEYS)
        for message in warnings:
            self._warn(message)
        after = {**ws.snapshot(), **payload}
        if (
            self.auto_plan
            and after.get("pln") is None
            and after.get("ct") is not None
            and after.get("cst") is not None
        ):
            try:
                payload["pln"] = build_plan(None, self.default_plan)
            except Exception as exc:  # noqa: BLE001 - the data is loaded nonetheless
                message = f"Default plan not applied: {type(exc).__name__}: {exc}"
                self._warn(message)
                warnings.append(message)
        self._v1_loaded(data, payload)
        if payload:
            ws.set_many(**payload)
        return warnings

    def merge_loaded(self, data: Mapping[str, Any]) -> list[str]:
        """Merge objects a host loaded itself into the workspace, as a load does.

        A CT starts a new patient (see :func:`merge_loaded_data`); snapshot tags travel with
        a CT or result (``saved_result_tags``). Returns the user-facing warnings, which are
        also reported as ``log`` events.
        """
        self._require_idle()
        return self._commit_loaded(data)

    def _prepare_load_dij(self, path: Any, label: Optional[str] = None) -> Operation:
        name = os.path.basename(os.fspath(path))

        def compute() -> dict:
            from pyRadPlan.io import load_data  # noqa: PLC0415

            return dij_payload(load_data(path), name)

        return Operation(
            "load_dij",
            label or f"Loading dose influence from {name}",
            compute,
            lambda found: self.workspace.set_many(**found),
        )

    def _prepare_load_path(self, path: Any, label: Optional[str] = None) -> Operation:
        from ._io import load_path_data  # noqa: PLC0415

        return Operation(
            "load_path",
            label or f"Loading {os.path.basename(os.fspath(path))}",
            lambda: load_path_data(path),
            self._commit_loaded,
        )

    # Native file operations (paths, not uploads), defined in ``_io``.

    def _prepare_save_path(self, **params: Any) -> Operation:
        from ._io import prepare_save_path  # noqa: PLC0415

        return prepare_save_path(self, **params)

    def _prepare_import_image(self, **params: Any) -> Operation:
        from ._io import prepare_import_image  # noqa: PLC0415

        return prepare_import_image(self, **params)

    def _prepare_import_binary(self, **params: Any) -> Operation:
        from ._io import prepare_import_binary  # noqa: PLC0415

        return prepare_import_binary(self, **params)

    def _prepare_scan_dicom(self, **params: Any) -> Operation:
        from ._io import prepare_scan_dicom  # noqa: PLC0415

        return prepare_scan_dicom(self, **params)

    def _prepare_import_dicom(self, **params: Any) -> Operation:
        from ._io import prepare_import_dicom  # noqa: PLC0415

        return prepare_import_dicom(self, **params)

    def _prepare_load_phantom(self, name: str) -> Operation:
        from pyRadPlan.io import phantom_path  # noqa: PLC0415

        return self._prepare_load_path(phantom_path(name), label=f"Loading {name}")

    def _prepare_load_file(self, filename: str, data: Any) -> Operation:
        from ._io import _load_upload, _payload_bytes, _upload_name  # noqa: PLC0415

        basename = _upload_name(filename)
        payload = _payload_bytes(data)
        return Operation(
            "load_file",
            f"Loading {basename}",
            lambda: _load_upload(basename, payload),
            self._commit_loaded,
        )

    def load_phantom(self, name: str) -> None:
        """Load a bundled phantom (see :func:`pyRadPlan.io.available_phantoms`)."""
        self.execute(self.prepare("load_phantom", name=name))

    def load_path(self, path: Any) -> None:
        """Load everything a file or DICOM folder holds, as the Qt "Load File" does."""
        self.execute(self.prepare("load_path", path=path))

    def load_file(self, filename: str, data: Any) -> None:
        """Load a file from its raw bytes; *filename* selects the importer by its ending."""
        self.execute(self.prepare("load_file", filename=filename, data=data))

    # ------------------------------------------------------------------
    # Plan
    # ------------------------------------------------------------------

    def get_plan(self) -> Optional[dict[str, Any]]:
        """Return the ``PlanSummary`` of the current plan (``None`` without one)."""
        return plan_summary(self.workspace.pln)

    def set_plan(self, changes: Mapping[str, Any]) -> None:
        """Apply a partial ``PlanSummary`` to the current (or the default) plan."""
        if self._busy is not None:
            raise PreconditionError(f"{self._busy.label} is still running", "busy")
        self.workspace.pln = self._build_plan_v1(self.workspace.pln, changes)

    def _ensure_plan(self) -> None:
        if self.workspace.pln is None:
            self.workspace.pln = build_plan(None, self.default_plan)

    # ------------------------------------------------------------------
    # Checks before a calculation
    # ------------------------------------------------------------------

    def _tissue_issues(self, pln: Any, cst: Any) -> list:
        """Return the issues of the tissue check the plan's dose engine runs."""
        from pyRadPlan.biology._editing import (  # noqa: PLC0415
            load_machine_quietly,
            plan_dose_engine,
            tissue_issues_quietly,
        )
        from pyRadPlan.biology.models import BiologicalModel  # noqa: PLC0415

        model = getattr(pln, "bio_model", None)
        if not isinstance(model, BiologicalModel) or model.model == "none" or cst is None:
            return []
        machine = getattr(pln, "machine", None)
        if isinstance(machine, str) or machine is None:
            machine = load_machine_quietly(
                getattr(pln, "radiation_mode", ""), machine or "Generic"
            )
        return tissue_issues_quietly(machine, model, cst, plan_dose_engine(pln))

    def _check_bio_model(self, step: str, physical_only: bool) -> list:
        """Check the biological model as the Qt widget does, with the answer given up front.

        Invalid reference alpha_x / beta_x always block. Other errors block unless
        *physical_only* is set, which switches the plan's model to ``none`` (written to the
        workspace, as the Qt dialog does). Warnings are reported as ``log`` events and
        returned, unless the model was switched off. The errors travel as
        ``PreconditionError.data``: ``issues`` (the ``TissueIssue`` errors) and ``model``
        (the plan's ``BiologicalModel``).
        """
        ws = self.workspace
        issues = self._tissue_issues(ws.pln, ws.cst)
        errors = [issue for issue in issues if issue.is_error]
        warnings = [issue for issue in issues if not issue.is_error]
        for issue in warnings:
            self._warn(f"{step}: {issue}")
        if not errors:
            return warnings
        text = "\n".join(f"- {issue}" for issue in errors)
        model = ws.pln.bio_model
        data = {"issues": errors, "model": model}
        if any(issue.kind == "invalid_reference" for issue in errors):
            raise PreconditionError(
                f"{step}: invalid reference photon parameters alpha_x / beta_x, which no "
                f"biological model can be evaluated with:\n{text}",
                "invalid_reference",
                data=data,
            )
        if not physical_only:
            raise PreconditionError(
                f"{step}: the biological model '{model.model}' cannot be evaluated for the "
                f"structures, machine and dose engine of the plan:\n{text}\n"
                "Fix the tissue parameters, choose another model, or pass "
                "physical_only=true to calculate the physical dose only.",
                "bio_model_unsupported",
                data=data,
            )
        from pyRadPlan.plan import validate_pln  # noqa: PLC0415

        pln = ws.pln
        fields = {name: deepcopy(getattr(pln, name)) for name in type(pln).model_fields}
        fields["bio_model"] = "none"
        ws.pln = validate_pln(type(pln)(**fields))
        return []

    def _check_objective_quantities(self) -> None:
        """Biological objectives need a dose influence computed with a biological model."""
        ws = self.workspace
        dij = ws.dij
        if dij is None or getattr(dij, "bio_model", None) is not None:
            return
        if getattr(dij, "alpha_dose", None) is not None:
            return
        from pyRadPlan.quantities import get_available_quantities  # noqa: PLC0415

        quantities = get_available_quantities()
        found: list[tuple[str, str]] = []
        for voi in getattr(ws.cst, "vois", None) or []:
            for objective in getattr(voi, "objectives", None) or []:
                identifier = getattr(objective, "quantity", None)
                quantity = quantities.get(identifier) if isinstance(identifier, str) else None
                if quantity is not None and quantity.biological:
                    entry = (voi.name, identifier)
                    if entry not in found:
                        found.append(entry)
        if found:
            raise PreconditionError(
                "These objectives are evaluated on biological quantities: "
                + ", ".join(f"{name}: {identifier}" for name, identifier in found)
                + ", but the dose influence holds the physical dose only. Select a "
                "biological model and calculate the dose influence again, or evaluate the "
                "objectives on the physical dose.",
                "biological_objectives",
                data={"objectives": found},
            )

    # ------------------------------------------------------------------
    # Workflow
    # ------------------------------------------------------------------

    def _prepare_generate_stf(self) -> Operation:
        self._require("generate_stf")
        ws = self.workspace
        ct, cst, pln = ws.ct, ws.cst, ws.pln

        def compute() -> Any:
            from pyRadPlan import generate_stf  # noqa: PLC0415

            return generate_stf(ct, cst, pln)

        return Operation(
            "generate_stf", "Generating beam geometry", compute, lambda stf: ws.set("stf", stf)
        )

    def _prepare_calc_dose_influence(self, physical_only: bool = False) -> Operation:
        self._require("calc_dose_influence")
        warnings = self._check_bio_model("Calculate dose influence", physical_only)
        ws = self.workspace
        ct, cst, pln = ws.ct, ws.cst, ws.pln

        def compute() -> tuple:
            from pyRadPlan import calc_dose_influence, generate_stf  # noqa: PLC0415

            stf = generate_stf(ct, cst, pln)
            return stf, calc_dose_influence(ct, cst, stf, pln)

        def commit(value: tuple) -> None:
            stf, dij = value
            ws.set_many(stf=stf, dij=dij)

        return Operation(
            "calc_dose_influence",
            "Calculating dose influence",
            compute,
            commit,
            warnings=warnings,
        )

    def _preferred_solver(self) -> Optional[str]:
        from pyRadPlan.optimization.solvers import get_available_solvers  # noqa: PLC0415

        solvers = get_available_solvers()
        return next((name for name in self.solver_preference if name in solvers), None)

    def _prop_opt(self, pln: Any, solver: Optional[str], max_iter: Optional[int]) -> dict:
        prop_opt = dict(pln.prop_opt or {})
        if solver is not None:
            prop_opt["solver"] = solver
        elif "solver" not in prop_opt:
            preferred = self._preferred_solver()
            if preferred is not None:
                prop_opt["solver"] = preferred
        max_iter = max_iter if max_iter is not None else self.max_iter
        if max_iter is not None:
            prop_opt["max_iter"] = int(max_iter)
        return prop_opt

    def _prepare_optimize(
        self, solver: Optional[str] = None, max_iter: Optional[int] = None
    ) -> Operation:
        self._require("optimize")
        self._check_objective_quantities()
        ws = self.workspace
        ct, cst, stf, dij, pln = ws.ct, ws.cst, ws.stf, ws.dij, ws.pln
        prop_opt = self._prop_opt(pln, solver, max_iter)
        unchanged = prop_opt == dict(pln.prop_opt or {})
        opt_pln = pln if unchanged else pln.model_copy(update={"prop_opt": prop_opt})
        previous = dict(ws.result or {})

        def compute() -> tuple:
            from pyRadPlan import fluence_optimization  # noqa: PLC0415

            info: dict = {}
            weights = fluence_optimization(ct, cst, stf, dij, opt_pln, opt_info=info)
            scenarios = dij.scenarios is not None and len(dij.scenarios) > 1
            result = dij.compute_result_ct_grid(
                weights, num_of_fractions=pln.result_dose_factor, scenarios=scenarios
            )
            if scenarios:
                result[SCENARIO_LABELS_KEY] = tuple(dij.scenario_labels())
            return weights, result, info

        def commit(value: tuple) -> dict:
            weights, result, info = value
            result["w"] = weights
            carry_forward(previous, result, self.saved_tags)
            ws.result = result
            history = info.get("obj_history") or []
            solver_config = prop_opt.get("solver")
            return {
                # A configured solver is stored as {"name": ..., <options>}; report its name.
                "solver": solver_config.get("name")
                if isinstance(solver_config, Mapping)
                else solver_config,
                "num_iter": info.get("num_iter"),
                "objective": float(history[-1]) if history else None,
            }

        max_iter_value = prop_opt.get("max_iter")
        return Operation(
            "optimize",
            "Optimizing fluence",
            compute,
            commit,
            max_iter=int(max_iter_value) if isinstance(max_iter_value, (int, float)) else None,
            keeps_partial_result=True,
        )

    def _prepare_recalc(self, physical_only: bool = False) -> Operation:
        self._require("recalc")
        ws = self.workspace
        previous = dict(ws.result or {})
        weights = previous.get("w")
        if weights is None:
            raise PreconditionError("No beamlet weights found in the current result", "no_weights")
        if not recalculation_angles_match(ws.pln, ws.stf):
            raise PreconditionError(
                "The plan's beam angles or beam count differ from the stored steering "
                "information. Calculate the dose influence and optimize again before "
                "recalculating dose.",
                "beam_geometry_changed",
            )
        warnings = self._check_bio_model("Recalculate dose", physical_only)
        ct, cst, stf, pln = ws.ct, ws.cst, ws.stf, ws.pln

        def compute() -> tuple:
            from pyRadPlan import calc_dose_forward  # noqa: PLC0415
            from pyRadPlan.stf import resolve_iso_centers  # noqa: PLC0415

            iso_centers = resolve_iso_centers(
                (pln.prop_stf or {}).get("iso_center"), cst, len(stf.beams)
            )
            recalc_stf = stf
            if not np.allclose([beam.iso_center for beam in stf.beams], iso_centers):
                recalc_stf = deepcopy(stf)
                for beam, iso_center in zip(recalc_stf.beams, iso_centers):
                    beam.iso_center = iso_center
            new_result = calc_dose_forward(ct, cst, recalc_stf, pln, weights)
            if any(key.endswith(SCENARIO_SUFFIX) for key in new_result):
                new_result[SCENARIO_LABELS_KEY] = tuple(pln.scenarios.scenarios().labels())
            return recalc_stf, new_result

        def commit(value: tuple) -> None:
            recalc_stf, new_result = value
            result = {
                key: val
                for key, val in previous.items()
                if is_tagged(key, self.saved_tags)
                or not (key.endswith(SCENARIO_SUFFIX) or key == SCENARIO_LABELS_KEY)
            }
            result.update(new_result)
            if recalc_stf is stf:
                ws.result = result
            else:
                ws.set_many(stf=recalc_stf, result=result)

        return Operation("recalc", "Recalculating dose", compute, commit, warnings=warnings)

    def generate_stf(self) -> None:
        """Generate the steering information (beam geometry) for the plan."""
        self.execute(self.prepare("generate_stf"))

    def calc_dose_influence(self, physical_only: bool = False) -> None:
        """Generate the beams and calculate their dose influence (Qt "Calc. Dose Influence")."""
        self.execute(self.prepare("calc_dose_influence", physical_only=physical_only))

    def optimize(self, solver: Optional[str] = None, max_iter: Optional[int] = None) -> dict:
        """Optimize the fluence and store the result on the CT grid.

        Returns the solver used, the iteration count and the final objective value.
        """
        summary = self.execute(self.prepare("optimize", solver=solver, max_iter=max_iter))
        logger.info("Optimization finished: %s", summary)
        return summary

    def recalc(self, physical_only: bool = False) -> None:
        """Recalculate the dose of the current weights with the current plan."""
        self.execute(self.prepare("recalc", physical_only=physical_only))

    def run_all(
        self,
        solver: Optional[str] = None,
        max_iter: Optional[int] = None,
        physical_only: bool = False,
    ) -> None:
        """Run every step that is missing or stale: plan, dose influence, optimization."""
        if not self.workspace.has("ct", "cst"):
            raise PreconditionError(
                "run_all needs a CT and structures; load data first", "no_data"
            )
        self._ensure_plan()
        stages = self.stages()
        if stages["dij"] != "done" or stages["stf"] == "stale":
            self.calc_dose_influence(physical_only=physical_only)
        if self.stages()["optimization"] != "done":
            self.optimize(solver=solver, max_iter=max_iter)

    # ------------------------------------------------------------------
    # Views
    # ------------------------------------------------------------------

    def volume(self, name: str, phase: int = 0) -> dict[str, Any]:
        """``"ct"`` or a result quantity as a ``Volume`` on the CT grid, ``(z, y, x)``."""
        return self.views.volume(name, phase)

    def structure_masks(self, names: Optional[Sequence[str]] = None) -> dict[str, Any]:
        """uint8 masks of the structures on the CT grid, ``(z, y, x)``."""
        return self.views.structure_masks(names)

    def dvh(
        self,
        quantity: Optional[str] = None,
        names: Optional[Sequence[str]] = None,
        num_points: int = 500,
    ) -> dict[str, Any]:
        """Cumulative DVHs of the structures for a result quantity."""
        return self.views.dvh(quantity, names, num_points)

    # ------------------------------------------------------------------

    def close(self) -> None:
        """Release the views and listeners; clears the workspace if the session owns it."""
        self._unsubscribe()
        self._listeners.clear()
        self.views.clear()
        if self._owns_workspace:
            self.workspace.clear()
