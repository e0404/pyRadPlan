"""The *Agent* tab of the agentic main window: run the planning agent and follow it."""

from __future__ import annotations

import contextvars
import html
import logging
import threading
import time
from typing import Any, Callable, Optional

from pydantic import BaseModel
from PySide6.QtCore import QThread, Signal, Slot, Qt
from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from pyRadPlan.app._workspace import DEFAULT_RESULT_NAME, unique_result_name
from pyRadPlan.core import ComputeCancelledError, ComputeControl, ProgressReport, StatusReport
from pyRadPlan.gui.workspace import WorkspaceManager
from pyRadPlan.optimization.objectives import Objective

from .._base import Worker, WorkspaceWidget
from ..ai import ai_disabled_reason
from ..workflow import WorkflowWidget
from ._agentic_plan_widget import AgenticPlanWidget
from ._phase_indicator import PhaseIndicator
from ._progress_plot import GoalProgressPlot
from ._result_widget import AgentResultWidget

logger = logging.getLogger(__name__)

#: Longest tool argument / error text shown in the agent log.
_MAX_LOG_TEXT = 160
#: Lines kept in the agent log.
_MAX_LOG_LINES = 5000

_PHASES = (("thinking", "Thinking"), ("dij", "Calculating D_ij"), ("optimize", "Optimizing"))

#: Why a run paused for the planner (see PlanningState.review_reason).
_REVIEW_REASONS = {
    "pause": "paused on request",
    "review": "review after {n} optimization(s)",
    "budget": "optimization budget used up: raise it to let the agent optimize further",
    "hard_goals": "all hard goals met for the first time",
}


def _short(text: str, limit: int = _MAX_LOG_TEXT) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _brief(value: Any) -> str:
    """Compact form of a tool argument: models show only their non-default fields."""
    if isinstance(value, Objective):
        params = "".join(f", {k}={v}" for k, v in zip(value.parameter_names, value.parameters))
        return f"{value.name}(penalty={value.priority:g}{params})"
    if isinstance(value, BaseModel):
        fields = value.model_dump(exclude_defaults=True)
        return f"{type(value).__name__}({', '.join(f'{k}={v!r}' for k, v in fields.items())})"
    if isinstance(value, list):
        return f"[{', '.join(map(_brief, value))}]"
    return repr(value)


def _reason_text(reason: str, num_optimizations: int) -> str:
    return _REVIEW_REASONS.get(reason, reason).format(n=num_optimizations)


def _plan_line(plan: Any) -> str:
    """One line on a :class:`~pyRadPlan.ai.agents.PlanRecord` for the log and tooltips."""
    from pyRadPlan.ai.agents._planning._views import format_mix  # noqa: PLC0415

    ev = plan.evaluation
    failed = ev["num_failed"]
    opt = ev.get("optimizer") or {}
    iterations = f" · {opt['iterations']} iterations" if opt.get("iterations") else ""
    if opt.get("converged"):
        iterations += " (converged)"
    if opt.get("start_from") is not None:
        iterations += f" from plan {opt['start_from']}"
    passed = "all hard goals met" if ev["all_hard_passed"] else "hard goals failed"
    if plan.blend:
        kind = f"Blend {plan.index} ({format_mix(plan.blend['plans'], plan.blend['weights'])})"
    else:
        kind = f"Optimization {plan.index}"
    return f"{kind}: score {ev['score']:g} · {passed} · {failed} goal(s) failed{iterations}"


def _pareto_line(result: dict) -> str:
    from pyRadPlan.ai.agents._planning._views import format_mix, short_goal  # noqa: PLC0415

    head = f"Pareto front of plans {', '.join(map(str, result['plans']))}"
    if "compromise" not in result:
        return f"{head}: no blend of {result['num_blends']} meets all hard goals"
    c = result["compromise"]
    weak = ", ".join(short_goal(w["goal"]) for w in c["weakest_axes"])
    return (
        f"{head}: {result['front_size']} of {result['num_blends']} blends on the front · "
        f"compromise {format_mix(result['plans'], c['weights'])} (score {c['score']:g})"
        + (f" · weakest {weak}" if weak else "")
    )


def _usage_line(usage: Any) -> Optional[str]:
    """One line on the requests and tokens of a run that failed (a pydantic-ai usage)."""
    requests = getattr(usage, "requests", None)
    if requests is None:
        return None
    parts = [f"{requests} request(s)"]
    for attr, label in (
        ("tool_calls", "tool call(s)"),
        ("input_tokens", "input tokens"),
        ("output_tokens", "output tokens"),
    ):
        value = getattr(usage, attr, None)
        if value:
            parts.append(f"{value} {label}")
    return "Usage so far: " + ", ".join(parts)


def _is_resumable(exc: Any) -> bool:
    """Whether *exc* carries the conversation of the failed run (``PlanningRunError``)."""
    return callable(getattr(exc, "all_messages", None)) and hasattr(exc, "usage")


class _DetachableCall:
    """Makes a blocking call in a daemon thread and waits for it until detached.

    A model request cannot be interrupted, and a ``QThread`` must not be destroyed while
    it runs. The run's ``QThread`` therefore only waits for a daemon thread making the
    call: :meth:`detach` ends that wait, so the ``QThread`` finishes at once, while the
    daemon thread ends on its own or with the interpreter. Events and reports of the
    call reach Qt only through :meth:`send_event` / :meth:`send_report`, which forward
    nothing (and hold no Qt object) once detached.
    """

    def __init__(self, fn: Callable, on_event: Callable, on_report: Callable) -> None:
        self._fn = fn
        self._sinks: dict[str, Callable] = {"event": on_event, "report": on_report}
        self._lock = threading.Lock()
        self._done = threading.Event()
        self._outcome: Optional[tuple] = None

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        context = contextvars.copy_context()  # the report observers and the control

        def job() -> None:
            try:
                outcome = (context.run(self._fn, *args, **kwargs), None)
            except BaseException as exc:  # noqa: BLE001 - raised again in the waiting thread
                outcome = (None, exc)
            self._outcome = outcome
            self._done.set()

        threading.Thread(target=job, name="pyRadPlan-agent", daemon=True).start()
        self._done.wait()
        if self._outcome is None:
            raise ComputeCancelledError("The agent run was detached.")
        result, exc = self._outcome
        if exc is not None:
            raise exc
        return result

    def _send(self, sink: str, *args: Any) -> None:
        with self._lock:  # detach() waits for a signal that is being emitted
            emit = self._sinks.get(sink)
            if emit is not None:
                emit(*args)

    def send_event(self, event: str, **data: Any) -> None:
        """Forward an agent event (the ``PlanningState`` listener)."""
        self._send("event", event, data)

    def send_report(self, report: object) -> None:
        """Forward a progress or status report of a computation."""
        self._send("report", report)

    def detach(self) -> None:
        """Stop waiting for the call and forward nothing more."""
        with self._lock:
            self._sinks.clear()
        self._done.set()


class AgentPanel(WorkspaceWidget):
    """Runs the planning agent and shows what it does.

    Tabs: *Plan* (the plan editor, placed by the main window), *Agentic Plan* (run
    settings), *Progress* (goals per plan) and *Agentic Result* (report and every plan).
    When a run ends or pauses, the best distinct plans are emitted with
    :attr:`plans_selected`; a paused run is continued, redirected or finished from the
    box below the log.

    Parameters
    ----------
    workspace:
        Shared workspace with the patient and the plan to start from.
    workflow_widget:
        Provides *Load File* / *Load Folder*.
    before_run:
        Called before a run; returns False to cancel it (e.g. unapplied plan edits).
    """

    #: Emitted (True / False) when a run starts / ends.
    busy_changed = Signal(bool)
    #: Emitted when a run starts: plans of an earlier run are outdated.
    run_started = Signal()
    #: Emitted after a run with one dict per selected plan: ``key`` (the plan index),
    #: ``label``, ``tooltip`` and ``state`` (the workspace objects of its tab).
    plans_selected = Signal(list)
    #: Emitted with one such dict to open a plan tab.
    plan_requested = Signal(dict)
    #: Emitted when the run is cleared: its plan tabs are outdated.
    run_cleared = Signal()

    _event_received = Signal(str, object)
    _report_received = Signal(object)

    def __init__(
        self,
        workspace: Optional[WorkspaceManager] = None,
        workflow_widget: Optional[WorkflowWidget] = None,
        before_run: Optional[Callable[[], bool]] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(workspace, parent)
        self._workflow = workflow_widget
        self._before_run = before_run
        self._thread: Optional[QThread] = None
        self._worker: Optional[Worker] = None
        self._control: Optional[ComputeControl] = None
        self._call: Optional[_DetachableCall] = None
        self._state: Any = None
        self._ct: Any = None  # the CT the last run belongs to
        self._result_before: Any = None  # the workspace's active result before the run
        self._shown_result: Any = None  # the plan of the run that replaced the active result
        self._last_state: Any = None  # PlanningState of the last run, for Load plan
        self._picks: list = []  # PlanRecords of the last run's selected plans
        self._agent_doses: dict = {}  # the picks as workspace results, by result name
        self._summary = ""
        # (PlanningState, last result or the error of a failed run) to continue
        self._session: Optional[tuple] = None
        self._disabled_reason = ai_disabled_reason()

        tools: dict[str, str] = {}
        models: list[str] = []
        if self._disabled_reason is None:
            from pyRadPlan.ai.agents import available_models, planning_tools  # noqa: PLC0415

            tools, models = planning_tools(), available_models()
        self.agentic_plan_widget = AgenticPlanWidget(self._ws, tools, models)
        self.progress_plot = GoalProgressPlot()
        self.result_widget = AgentResultWidget()
        self.result_widget.load_requested.connect(self._on_load_requested)

        self._setup_ui()
        self._event_received.connect(self._on_agent_event)
        self._report_received.connect(self._on_compute_report)
        if self._workflow is not None:
            self._workflow.busy_changed.connect(lambda _busy: self._update_buttons())
        self.initialize()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        split = QSplitter(Qt.Orientation.Vertical)
        split.addWidget(self._build_workflow_box())

        self.settings_tabs = QTabWidget()
        #: Page the main window puts the plan editor on.
        self.plan_page = QWidget()
        plan_lay = QVBoxLayout(self.plan_page)
        plan_lay.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.agentic_plan_widget)
        self.settings_tabs.addTab(self.plan_page, "Plan")
        self.settings_tabs.addTab(scroll, "Agentic Plan")
        self.settings_tabs.addTab(self.progress_plot, "Progress")
        self.settings_tabs.addTab(self.result_widget, "Agentic Result")
        self.settings_tabs.setCurrentIndex(1)
        split.addWidget(self.settings_tabs)

        log_box = QGroupBox("Agent log")
        log_lay = QVBoxLayout(log_box)
        log_lay.setContentsMargins(4, 4, 4, 4)
        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumBlockCount(_MAX_LOG_LINES)
        self._log.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
        log_lay.addWidget(self._log)
        log_lay.addWidget(self._build_feedback_box())
        split.addWidget(log_box)

        for index, stretch in enumerate((0, 3, 1)):
            split.setStretchFactor(index, stretch)
        root.addWidget(split)

    def _build_feedback_box(self) -> QWidget:
        self._feedback_box = QWidget()
        lay = QVBoxLayout(self._feedback_box)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        self._lbl_feedback = QLabel()
        self._lbl_feedback.setWordWrap(True)
        lay.addWidget(self._lbl_feedback)
        self._txt_feedback = QPlainTextEdit()
        self._txt_feedback.setMaximumHeight(60)
        lay.addWidget(self._txt_feedback)
        row = QHBoxLayout()
        self._btn_continue = QPushButton("Continue")
        self._btn_continue.clicked.connect(lambda _checked=False: self.continue_agent("continue"))
        self._btn_redirect = QPushButton("Redirect")
        self._btn_redirect.setToolTip("Skip the pending step and give the agent your text")
        self._btn_redirect.clicked.connect(lambda _checked=False: self.continue_agent("redirect"))
        self._btn_finish = QPushButton("Finish")
        self._btn_finish.setToolTip("End the run: the agent writes its final report")
        self._btn_finish.clicked.connect(lambda _checked=False: self.continue_agent("finish"))
        for btn in (self._btn_continue, self._btn_redirect, self._btn_finish):
            row.addWidget(btn)
        row.addStretch()
        lay.addLayout(row)
        self._feedback_box.hide()
        return self._feedback_box

    def _update_feedback_box(self) -> None:
        if self._session is None or self.is_busy:
            self._feedback_box.hide()
            return
        from pydantic_ai import DeferredToolRequests  # noqa: PLC0415

        previous = self._session[1]
        paused = isinstance(getattr(previous, "output", None), DeferredToolRequests)
        failed = isinstance(previous, BaseException)
        if paused:
            text = (
                "The agent waits for you. Continue (your text, if any, goes to the agent with "
                "the next result), Redirect (skip the next step, the agent reads your text) or "
                "Finish. Goals, budgets and tools of Agentic Plan can be changed now."
            )
        elif failed:
            text = (
                "The run stopped early. Continue resumes it where it stopped, with a new "
                "request budget (your text, if any, goes to the agent); Finish asks for the "
                "final report."
            )
            if getattr(previous, "limit_exceeded", False):
                text += (
                    " A limit was reached: raise Requests or Optimizations in Agentic Plan "
                    "if the agent needs more."
                )
        else:
            text = "Continue the conversation with a new instruction."
        self._lbl_feedback.setText(text)
        self._txt_feedback.setPlaceholderText(
            "Feedback for the agent (optional)" if paused or failed else "Next instruction"
        )
        self._btn_redirect.setVisible(paused)
        self._btn_finish.setVisible(paused or failed)
        self._feedback_box.show()

    def _build_workflow_box(self) -> QGroupBox:
        box = QGroupBox("Workflow")
        lay = QVBoxLayout(box)
        lay.setContentsMargins(6, 4, 6, 4)
        lay.setSpacing(4)

        row = QHBoxLayout()
        row.addWidget(QLabel("Status:"))
        self._lbl_status = QLabel()
        self._lbl_status.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        row.addWidget(self._lbl_status)
        lay.addLayout(row)

        row = QHBoxLayout()
        self._phases: dict[str, PhaseIndicator] = {}
        for key, title in _PHASES:
            dot = PhaseIndicator()
            self._phases[key] = dot
            row.addWidget(dot)
            label = QLabel(title)
            label.setStyleSheet("font-weight: bold;")
            row.addWidget(label)
            row.addSpacing(10)
        row.addStretch()
        lay.addLayout(row)

        self._progress = QProgressBar()
        self._progress.setRange(0, 0)
        self._progress.setTextVisible(False)
        self._progress.setMaximumHeight(8)
        self._progress.hide()
        lay.addWidget(self._progress)

        row = QHBoxLayout()
        self._btn_load_file = QPushButton("Load File")
        self._btn_load_folder = QPushButton("Load Folder")
        if self._workflow is not None:
            self._btn_load_file.clicked.connect(self._workflow.load_file)
            self._btn_load_folder.clicked.connect(self._workflow.load_folder)
        self._btn_run = QPushButton("Run agent")
        self._btn_run.clicked.connect(self._on_run_clicked)
        self._btn_pause = QPushButton("Pause")
        self._btn_pause.setToolTip("Pause before the agent's next optimization or beam change")
        self._btn_pause.clicked.connect(self.pause_agent)
        self._btn_clear = QPushButton("Clear run")
        self._btn_clear.setToolTip("Reset the run (log, progress, results), or everything")
        self._btn_clear.clicked.connect(self._on_clear_clicked)
        for btn in (
            self._btn_load_file,
            self._btn_load_folder,
            self._btn_run,
            self._btn_pause,
            self._btn_clear,
        ):
            row.addWidget(btn)
        lay.addLayout(row)
        return box

    # ------------------------------------------------------------------
    # Workspace -> UI
    # ------------------------------------------------------------------

    def _do_update(self, changed_keys: list) -> None:
        ct = self._ws.ct
        # A refresh reports the same CT again; another one is another patient, whose
        # workspace the plans and the conversation of the last run must not reach
        if ct is not self._ct and not self.is_busy:
            self._ct = ct
            self.clear_run()
        self._update_buttons()
        if not self.is_busy:
            self._lbl_status.setText(self._idle_status())

    def _idle_status(self) -> str:
        if not self._ws.has("ct", "cst"):
            return "No data loaded"
        if not self._ws.has("pln"):
            return "CT and structures loaded – configure a plan to continue"
        if self._disabled_reason:
            return "AI not available (see the Run button's tooltip)"
        return self._summary or "Ready to run the agent"

    def _update_buttons(self) -> None:
        running = self.is_busy
        loading = self._workflow is not None and self._workflow.is_busy
        for btn in (self._btn_load_file, self._btn_load_folder):
            btn.setEnabled(not running and not loading)
        can_run = self._disabled_reason is None and self._ws.has("ct", "cst", "pln")
        stopping = self._control is not None and self._control.stop_requested
        self._btn_run.setEnabled((running and not stopping) or (can_run and not loading))
        self._btn_run.setText("Abort agent" if running else "Run agent")
        self._btn_clear.setEnabled(not running and not loading)
        pausing = self._state is not None and self._state.pause_requested
        self._btn_pause.setEnabled(running and not stopping and not pausing)
        self._btn_run.setToolTip(
            self._disabled_reason
            or ("Stop after the current step" if running else "Let the agent create plans")
        )

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    @property
    def is_busy(self) -> bool:
        """Whether a run is active: from its start until its result was handled."""
        return self._thread is not None

    def _on_run_clicked(self) -> None:
        if self.is_busy:
            self._control.request_stop()
            self._lbl_status.setText("Aborting after the current step…")
            self._update_buttons()
            return
        self.run_agent()

    def run_agent(self) -> None:
        """Start the agent on the workspace's patient and plan."""
        panel = self.agentic_plan_widget
        if not panel.model:
            QMessageBox.warning(self, "Run agent", "Select or enter a model first.")
            return
        try:
            protocol = panel.protocol()
        except ValueError as exc:
            QMessageBox.warning(self, "Run agent", str(exc))
            return
        if self._before_run is not None and not self._before_run():
            return

        from pyRadPlan.ai.agents import (  # noqa: PLC0415
            PlanningState,
            pop_last_run_usage,
            run_planning_agent,
        )
        from pyRadPlan.ai.agents._planning._evaluation import DISTINCT_MIN_GAIN  # noqa: PLC0415

        self.run_started.emit()
        pop_last_run_usage()  # drop the usage of an earlier run
        ws = self._ws
        self._session = None
        self._result_before, self._shown_result = ws.result, None
        state = PlanningState(
            ct=ws.ct, cst=ws.cst, pln=ws.pln.model_copy(deep=True), protocol=protocol
        )
        self._apply_limits(state)
        task = panel.task
        if panel.num_plans > 1:
            # Without this the agent does not know that several plans are wanted
            task += (
                f"\n\nDeliver {panel.num_plans} plans that are each good in a different way. "
                "They are picked from all your optimizations by the goals: each must beat "
                f"every other pick by at least {DISTINCT_MIN_GAIN:.0%} of a goal's limit on "
                "some goal. Save each one with plan_versions."
            )
        self._last_state, self._picks = None, []
        self.progress_plot.set_plans([])
        self._log.clear()
        self.result_widget.start_run(
            [
                f"Run with {panel.model}: {_short(panel.task)}",
                f"{len(protocol.goals)} goal(s), {len(panel.tools())} tools, at most "
                f"{panel.max_optimizations} optimizations and {panel.request_limit} requests",
                f"{panel.max_dij_calcs} dose calculation(s), {panel.beam_range[0]}–"
                f"{panel.beam_range[1]} beams"
                + ("" if panel.allow_couch else " (coplanar)")
                + f", at most {panel.max_iter} iterations per optimization",
            ]
        )
        self._start(
            state,
            run_planning_agent,
            task,
            max_optimizations=panel.max_optimizations,
        )

    def _apply_limits(self, state: Any) -> None:
        """Copy the budgets and limits of *Agentic Plan* to *state*."""
        panel = self.agentic_plan_widget
        state.max_optimizations = panel.max_optimizations
        state.max_dij_calcs = panel.max_dij_calcs
        state.min_beams, state.max_beams = panel.beam_range
        state.allow_couch = panel.allow_couch
        state.max_iter = panel.max_iter
        state.review_every = panel.review_every

    def _start(self, state: Any, fn: Callable, *args: Any, **kwargs: Any) -> None:
        """Run *fn* (*state*, *args*, ...) with the panel's model, tools and limits."""
        panel = self.agentic_plan_widget
        self._state = state
        self._control = state.control = ComputeControl()
        for dot in self._phases.values():
            dot.set_state("idle")
        self._set_phase("thinking")
        self.settings_tabs.setCurrentWidget(self.progress_plot)
        self._call = _DetachableCall(fn, self._event_received.emit, self._report_received.emit)
        state.listener = self._call.send_event
        self._worker = Worker(
            self._call,
            state,
            *args,
            model=panel.model,
            request_limit=panel.request_limit,
            compact_after=panel.compact_after,
            tools=panel.tools(),
            report_cb=self._call.send_report,
            control=self._control,
            **kwargs,
        )
        self._thread = QThread(self)
        self._worker.moveToThread(self._thread)
        # Bound slots, so Qt runs them in the GUI thread
        self._worker.finished.connect(self._on_run_finished)
        self._worker.error.connect(self._on_run_error)
        self._thread.started.connect(self._worker.run)
        self._thread.start()
        self._set_busy(True)

    def pause_agent(self) -> None:
        """Pause the run before the agent's next optimization or beam change."""
        if not self.is_busy:
            return
        self._state.pause_requested = True
        self._lbl_status.setText("Pausing before the next optimization or beam change…")
        self._update_buttons()

    def continue_agent(self, action: str = "continue") -> None:
        """Continue the paused or finished run: ``continue``, ``redirect`` or ``finish``."""
        from pydantic_ai import DeferredToolRequests  # noqa: PLC0415

        from pyRadPlan.ai.agents import continue_planning_agent  # noqa: PLC0415

        if self._session is None or self.is_busy:
            return
        state, previous = self._session
        text = self._txt_feedback.toPlainText().strip()
        paused = isinstance(getattr(previous, "output", None), DeferredToolRequests)
        failed = isinstance(previous, BaseException)  # resumes without an instruction
        if not paused and not failed and not text:
            QMessageBox.warning(self, "Continue", "Enter an instruction for the agent first.")
            return
        try:
            protocol = self.agentic_plan_widget.protocol()
        except ValueError as exc:
            QMessageBox.warning(self, "Continue", str(exc))
            return
        if not paused:  # the report's plan tabs and doses are replaced at the next report
            self.run_started.emit()
            self._result_before, self._shown_result = self._ws.result, None
        self._log_line(f"Planner: {action}" + (f": {_short(text)}" if text else ""), "#2980b9")
        self._txt_feedback.clear()
        self._apply_limits(state)
        self._start(
            state,
            continue_planning_agent,
            previous,
            text,
            action=action,
            protocol=protocol,
        )

    def _set_busy(self, busy: bool) -> None:
        self._progress.setVisible(busy)
        self.result_widget.set_loading_enabled(not busy)
        self.agentic_plan_widget.set_running(busy)
        self._update_buttons()
        self._update_feedback_box()
        self.busy_changed.emit(busy)

    def _set_phase(self, phase: Optional[str]) -> None:
        """Spin the dot of *phase*; a phase that spun before turns green."""
        for key, dot in self._phases.items():
            if key == phase:
                dot.set_state("active")
            elif dot.state == "active":
                dot.set_state("done")
        text = {
            "thinking": "Thinking…",
            "dij": "Calculating dose influence…",
            "optimize": "Optimizing…",
        }.get(phase)
        if text:
            self._lbl_status.setText(text)

    @Slot(str, object)
    def _on_agent_event(self, event: str, data: dict) -> None:
        if event == "tool":
            args = ", ".join(f"{k}={_brief(v)}" for k, v in data["args"].items() if v is not None)
            self._log_line(f"→ {data['name']}({_short(args)})")
            self._set_phase(None)
            self._lbl_status.setText(f"Running {data['name']}…")
        elif event == "tool_done":
            if data["error"]:
                self._log_line(f"  ✗ {_short(data['error'])}", "#e67e22")
            self._set_phase("thinking")
        elif event == "phase":
            self._set_phase(data["phase"])
        elif event == "plan":
            plan = data["plan"]
            self._log_line(f"  {_plan_line(plan)}", "#27ae60")
            self.progress_plot.set_plans(self._state.plans)
            self.result_widget.add_plan(plan)
            self._ws.result = plan.result
            self._shown_result = self._ws.result
        elif event == "pareto":
            self._log_line(f"  {_pareto_line(data['result'])}", "#8e44ad")
        elif event == "review":
            reason = _reason_text(data["reason"], self._state.num_optimizations)
            self._log_line(f"⏸ Before {data['name']}: {reason}", "#2980b9")
        elif event == "compact":
            self._log_line(
                f"  History compacted after {data['num_optimizations']} optimizations", "#7f8c8d"
            )

    @Slot(object)
    def _on_compute_report(self, report: object) -> None:
        if isinstance(report, ProgressReport) and report.levels:
            levels = " · ".join(
                f"{lvl.name} {lvl.current}/{lvl.total}" if lvl.total else lvl.name
                for lvl in report.levels
            )
            self._lbl_status.setText(f"Calculating dose influence · {levels}")
        elif isinstance(report, StatusReport) and "iteration" in report.data:
            data = report.data
            objective = f" · f={data['objective']:.4g}" if "objective" in data else ""
            n = (len(self._state.plans) + 1) if self._state is not None else ""
            self._lbl_status.setText(
                f"Optimization {n} · iteration {int(data['iteration'])}{objective}"
            )

    @Slot(object)
    def _on_run_finished(self, result: Any) -> None:
        self._on_run_done(result, None)

    @Slot(object)
    def _on_run_error(self, exc: object) -> None:
        self._on_run_done(None, exc)

    def _on_run_done(self, result: Any, exc: Optional[BaseException]) -> None:
        self._cleanup_thread()
        from pyRadPlan.ai.agents import pop_last_run_usage, select_distinct_plans  # noqa: PLC0415

        from pydantic_ai import DeferredToolRequests  # noqa: PLC0415

        state, self._state = self._state, None
        report, status = None, ""
        paused = result is not None and isinstance(result.output, DeferredToolRequests)
        usage = pop_last_run_usage()
        if result is not None:
            self._session = (state, result)
        elif _is_resumable(exc):  # it holds the conversation so far: the run can go on
            self._session = (state, exc)
            usage = usage or _usage_line(exc.usage)
        else:
            self._session = None
        if paused:
            reasons = {m.get("reason") for m in result.output.metadata.values()} - {None}
            status = "Waiting for the planner: " + ", ".join(
                _reason_text(r, state.num_optimizations) for r in sorted(reasons)
            )
        elif result is not None:
            report = result.output
        elif isinstance(exc, ComputeCancelledError):
            status = "Aborted on request."
            self._log_line(status, "#e67e22")
        else:
            # The error of a run that can go on names its cause itself
            name = "" if self._session is not None else f"{type(exc).__name__}: "
            status = f"Agent stopped: {name}{exc}"
            self._log_line(status, "#e74c3c")

        picks = select_distinct_plans(state.plans, n=self.agentic_plan_widget.num_plans)
        self._last_state, self._picks = state, picks
        self.progress_plot.set_plans(state.plans, picks)
        self.result_widget.mark_picks(picks)
        for dot in self._phases.values():
            dot.set_state("idle")
        if picks:
            blends = len(state.plans) - state.num_optimizations
            self._summary = (
                f"{state.num_optimizations} optimization(s)"
                + (f", {blends} blend(s)" if blends else "")
                + f", {len(picks)} plan(s) selected: "
                + ", ".join(f"Plan {i}" for i in range(1, len(picks) + 1))
            )
        else:
            self._summary = "The agent produced no plan"
        unresolved = state.plans[-1].evaluation.get("unresolved_goals", []) if state.plans else []
        self.result_widget.set_report(
            report, status, unresolved, usage=usage, summary=self._summary
        )
        self._set_busy(False)
        if paused:
            self._lbl_status.setText(status)
            return
        self._lbl_status.setText(self._summary)
        self.settings_tabs.setCurrentWidget(self.result_widget)
        self._publish_plans(state, picks)

    def _publish_plans(self, state: Any, picks: list[dict]) -> None:
        """Add the picked doses to the workspace results and emit one workspace state per plan.

        The active result is set back to the one before the run, and the picks of an
        earlier run are replaced.
        """
        ws = self._ws
        active = ws.active_result
        results = {n: r for n, r in ws.results.items() if n not in self._agent_doses}
        if self._result_before is None:
            results.pop(active, None)
        else:
            results[active] = self._result_before
        doses: dict = {}
        for i, p in enumerate(picks, start=1):
            name = unique_result_name(f"Plan_{i}", [*results, *doses])
            doses[name] = p.result
        self._agent_doses = doses
        ws.set_many(results={**results, **doses}, active_result=active)
        self._result_before = self._shown_result = None
        self.plans_selected.emit(
            [self._plan_tab(state, p, f"Plan {i}") for i, p in enumerate(picks, start=1)]
        )

    def _plan_tab(self, state: Any, plan: Any, label: str) -> dict:
        """Return the plan tab (label, tooltip, workspace state) of one plan."""
        from pyRadPlan.plan import validate_pln  # noqa: PLC0415

        pln = state.pln.model_copy(deep=True)
        pln.prop_stf = {**(pln.prop_stf or {}), **plan.beams}
        return {
            "key": plan.index,
            "label": label,
            "tooltip": _plan_line(plan),
            "state": {
                "cst": plan.cst,
                "pln": validate_pln(pln),
                "stf": None,
                "dij": None,
                "results": {DEFAULT_RESULT_NAME: plan.result, **self._agent_doses},
                "active_result": DEFAULT_RESULT_NAME,
            },
        }

    def _on_load_requested(self, index: int) -> None:
        """Open optimization *index* of the last run as a plan tab (or switch to it)."""
        state = self._last_state
        plan = next((p for p in state.plans if p.index == index), None) if state else None
        if plan is None:
            return
        number = next((n for n, p in enumerate(self._picks, 1) if p.index == index), None)
        kind = "Blend" if plan.blend else "Opt"
        label = f"Plan {number}" if number else f"{kind} {index}"
        self.plan_requested.emit(self._plan_tab(state, plan, label))

    # ------------------------------------------------------------------
    # Clear
    # ------------------------------------------------------------------

    def _on_clear_clicked(self) -> None:
        box = QMessageBox(self)
        box.setWindowTitle("Clear run")
        box.setIcon(QMessageBox.Icon.Question)
        box.setText("Do you really want to reset the current state?")
        box.setInformativeText(
            "Clear run: the log, progress, results and plan tabs of the run; the goals, "
            "agent settings and the plan stay.\n"
            "Clear everything: also the goals, agent settings, the plan and its results. "
            "Only the patient (CT and structures) stays."
        )
        btn_all = box.addButton("Clear everything", QMessageBox.ButtonRole.DestructiveRole)
        btn_run = box.addButton("Clear run", QMessageBox.ButtonRole.AcceptRole)
        box.addButton(QMessageBox.StandardButton.Cancel)
        box.setDefaultButton(QMessageBox.StandardButton.Cancel)
        box.exec()
        if box.clickedButton() is btn_run:
            self.clear_run()
        elif box.clickedButton() is btn_all:
            self.clear_everything()

    def clear_run(self) -> None:
        """Forget the last run: log, progress, results, its doses and plan tabs.

        Of the workspace results, only those the run put there go: its selected plans,
        and the plan a paused run shows in place of the active result, which is the one
        before the run again.
        """
        if self.is_busy:
            return
        from pyRadPlan.ai.agents import pop_last_run_usage  # noqa: PLC0415

        pop_last_run_usage()
        self._log.clear()
        self.progress_plot.set_plans([])
        self.result_widget.clear()
        self._last_state, self._picks, self._summary = None, [], ""
        self._session = None
        self._feedback_box.hide()
        self._txt_feedback.clear()
        for dot in self._phases.values():
            dot.set_state("idle")
        self.run_cleared.emit()
        doses, self._agent_doses = self._agent_doses, {}
        before, self._result_before = self._result_before, None
        shown, self._shown_result = self._shown_result, None
        ws = self._ws
        active = ws.active_result
        current = ws.results
        results = {n: r for n, r in current.items() if doses.get(n) is not r}
        if shown is not None and results.get(active) is shown:
            if before is None:
                del results[active]
            else:
                results[active] = before
            ws.set_many(results=results, active_result=active)
        elif len(results) != len(current):
            ws.set_many(results=results, active_result=active)
        self._lbl_status.setText(self._idle_status())

    def clear_everything(self) -> None:
        """Clear the run, the agent settings and goals, and reset the plan.

        Only the patient (CT and structures) is kept.
        """
        if self.is_busy:
            return
        from pyRadPlan.plan import PhotonPlan  # noqa: PLC0415

        self.clear_run()
        self.agentic_plan_widget.reset()
        if self._ws.has("ct", "cst"):
            self._ws.set_many(pln=PhotonPlan(), stf=None, dij=None, result=None)
        else:
            self._ws.clear(["pln", "stf", "dij", "result"])

    def _cleanup_thread(self) -> None:
        if self._call is not None:
            self._call.detach()  # drops its Qt signals here, not in the daemon thread
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait()
            self._thread.deleteLater()
        if self._worker is not None:
            self._worker.deleteLater()
        self._thread = self._worker = self._control = self._call = None

    def shutdown(self, timeout_ms: int = 5000) -> None:
        """Stop a running agent before the window closes, within about *timeout_ms*.

        The agent is asked to stop, which it does after its current step. A model
        request cannot be interrupted: if the run has not ended after *timeout_ms*, it
        is detached. Its thread ends, and the request goes on in a daemon thread that
        reaches neither this panel nor the workspace any more and ends with the request
        or the interpreter. Acceptable only because the application is exiting.
        """
        if not self.is_busy:
            return
        self._control.request_stop()
        self._worker.finished.disconnect()
        self._worker.error.disconnect()
        self._thread.quit()
        if not self._thread.wait(timeout_ms):
            self._call.detach()
        self._state = None
        self._cleanup_thread()

    # ------------------------------------------------------------------
    # Log
    # ------------------------------------------------------------------

    def _log_line(self, text: str, color: Optional[str] = None) -> None:
        line = html.escape(f"{time.strftime('%H:%M:%S')} {text}").replace("  ", "&nbsp;&nbsp;")
        self._log.appendHtml(f'<span style="color:{color};">{line}</span>' if color else line)
