"""The agent panel's runs in their worker thread, with fake agent functions."""

import sys
import threading
import time
from types import SimpleNamespace

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from pyRadPlan.core import ComputeCancelledError, ProgressReporter
from pyRadPlan.gui.workspace import WorkspaceManager


def _wait_until(condition, timeout=10.0):
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        QApplication.processEvents()
        time.sleep(0.005)


def _agent_threads():
    return [t for t in threading.enumerate() if t.name == "pyRadPlan-agent"]


def _agent_threads_ended():
    # The thread reports its result just before it ends
    _wait_until(lambda: not _agent_threads())
    return True


class _FakeState(SimpleNamespace):
    """Stands in for ``PlanningState``: what the panel reads of it."""

    def __init__(self, **fields):
        super().__init__(plans=[], num_optimizations=0, pause_requested=False, **fields)

    def add_plan(self, cst, dose):
        from pyRadPlan.ai.agents import PlanRecord
        from pyRadPlan.result import create_result

        index = len(self.plans) + 1
        goal = {"goal": "Core max <= 1.6", "hard": True, "value": 1.5, "passed": True}
        goal.update(margin=0.1, rel_margin=0.06)
        plan = PlanRecord(
            index=index,
            beams={"gantry_angles": [0, 180], "couch_angles": [0, 0]},
            objectives={},
            evaluation={
                "all_hard_passed": True,
                "num_failed": 0,
                "score": float(index),
                "goals": [goal],
            },
            result=create_result({"physical_dose": dose}),
            cst=cst,
        )
        self.plans.append(plan)
        self.num_optimizations = index
        self.listener("plan", plan=plan)
        return plan


class _RunError(Exception):
    """The shape of ``pyRadPlan.ai.agents.PlanningRunError``."""

    output = None
    limit_exceeded = True

    def __init__(self, message):
        super().__init__(message)
        self.messages = ["request", "response"]
        self.usage = SimpleNamespace(requests=40, tool_calls=7, input_tokens=1200, output_tokens=0)

    def all_messages(self):
        return self.messages


def _report(summary="Done."):
    from pyRadPlan.ai.agents import PlanningReport

    return SimpleNamespace(output=PlanningReport(all_hard_goals_met=True, summary=summary))


@pytest.fixture
def agent(qapp, monkeypatch, test_data_photons):
    """A panel on the photon test patient whose runs call ``agent.run`` / ``agent.resume``."""
    pytest.importorskip("pydantic_ai")
    import pyRadPlan.ai.agents as agents
    from pyRadPlan.gui.widgets.agent import AgentPanel
    from pyRadPlan.plan import PhotonPlan

    ct, cst, result = test_data_photons
    ws = WorkspaceManager()
    ws.set_many(ct=ct, cst=cst, pln=PhotonPlan())
    panel = AgentPanel(ws)
    panel.agentic_plan_widget._cmb_model.setEditText("test")
    handle = SimpleNamespace(
        panel=panel, ws=ws, cst=cst, dose=result["physicalDose"], run=None, resume=None, calls=[]
    )

    def run(state, task, **kwargs):
        handle.calls.append(("run", task, kwargs))
        return handle.run(state)

    def resume(state, previous, text, **kwargs):
        handle.calls.append(("continue", previous, text, kwargs))
        return handle.resume(state)

    monkeypatch.setattr(agents, "PlanningState", _FakeState)
    monkeypatch.setattr(agents, "run_planning_agent", run)
    monkeypatch.setattr(agents, "continue_planning_agent", resume)
    yield handle
    panel.shutdown(timeout_ms=2000)
    assert not panel.is_busy and _agent_threads_ended()
    panel.close()
    panel.deleteLater()


def _run(agent):
    agent.panel.run_agent()
    assert agent.panel.is_busy
    _wait_until(lambda: not agent.panel.is_busy)


def test_run_completes_and_publishes_the_plan(agent):
    panel, ws = agent.panel, agent.ws
    busy = []
    panel.busy_changed.connect(busy.append)
    statuses = []
    panel._report_received.connect(lambda _report: statuses.append(panel._lbl_status.text()))

    def run(state):
        ProgressReporter().report_status(iteration=3, objective=1.5)  # as an optimizer does
        state.add_plan(agent.cst, agent.dose)
        return _report()

    agent.run = run
    _run(agent)

    assert busy == [True, False] and panel._thread is None and _agent_threads_ended()
    assert len(statuses) == 1 and statuses[0].endswith("iteration 3 · f=1.5")
    assert agent.calls[0][2]["model"] == "test"
    assert panel.result_widget.table.rowCount() == 1
    assert "Optimization 1: score 1" in panel._log.toPlainText()
    assert "Done." in panel.result_widget.report.toPlainText()
    assert list(ws.results) == ["Plan_1"] and ws.result is None
    assert panel._session is not None and not panel._feedback_box.isHidden()
    assert panel._log.maximumBlockCount() > 0


def test_run_error_ends_the_run_without_a_session(agent):
    panel = agent.panel

    def fail(state):
        raise RuntimeError("boom")

    agent.run = fail
    _run(agent)

    assert "Agent stopped: RuntimeError: boom" in panel._log.toPlainText()
    assert "RuntimeError: boom" in panel.result_widget.report.toPlainText()
    assert panel._session is None and panel._feedback_box.isHidden()
    assert panel._btn_run.text() == "Run agent" and _agent_threads_ended()


def test_run_error_with_the_conversation_can_be_continued(agent):
    panel, ws = agent.panel, agent.ws
    error = _RunError("The planning run stopped early (UsageLimitExceeded: 40 requests).")

    def fail(state):
        state.add_plan(agent.cst, agent.dose)
        raise error

    agent.run = fail
    _run(agent)

    report = panel.result_widget.report.toPlainText()
    assert "Agent stopped: The planning run stopped early" in report
    assert "Usage so far: 40 request(s), 7 tool call(s), 1200 input tokens" in report
    assert list(ws.results) == ["Plan_1"]  # what the run reached is kept
    assert panel._session[1] is error and not panel._feedback_box.isHidden()
    assert panel._btn_redirect.isHidden() and not panel._btn_finish.isHidden()
    assert "Requests" in panel._lbl_feedback.text()

    panel.agentic_plan_widget._spn_requests.setValue(60)
    agent.resume = lambda state: _report("Resumed.")
    panel.continue_agent()  # no instruction needed
    _wait_until(lambda: not panel.is_busy)

    kind, previous, text, kwargs = agent.calls[-1]
    assert (kind, previous, text) == ("continue", error, "")
    assert kwargs["request_limit"] == 60 and kwargs["action"] == "continue"
    assert "Resumed." in panel.result_widget.report.toPlainText()
    assert panel._session[1].output.summary == "Resumed."


def test_abort_stops_the_run_at_the_next_step(agent):
    panel = agent.panel

    def wait_for_stop(state):
        while not state.control.stop_requested:
            time.sleep(0.005)
        raise ComputeCancelledError("stopped")

    agent.run = wait_for_stop
    panel.run_agent()
    assert panel._btn_run.text() == "Abort agent"
    panel._on_run_clicked()
    assert not panel._btn_run.isEnabled()
    _wait_until(lambda: not panel.is_busy)

    assert "Aborted on request." in panel._log.toPlainText()
    assert panel._session is None and _agent_threads_ended()


def test_clear_run_of_a_paused_run_restores_the_result_before(agent):
    from pydantic_ai import DeferredToolRequests
    from pydantic_ai.messages import ToolCallPart

    from pyRadPlan.result import create_result

    panel, ws = agent.panel, agent.ws
    before = create_result({"physical_dose": agent.dose})
    ws.result = before
    before = ws.result
    call = ToolCallPart("optimize", {}, tool_call_id="c1")
    paused = SimpleNamespace(
        output=DeferredToolRequests(approvals=[call], metadata={"c1": {"reason": "pause"}})
    )
    agent.run = lambda state: (state.add_plan(agent.cst, agent.dose), paused)[1]
    _run(agent)
    assert ws.result is not before and "paused on request" in panel._lbl_status.text()

    panel.clear_run()
    assert ws.result is before and list(ws.results) == [ws.active_result]
    assert panel._session is None and panel.result_widget.table.rowCount() == 0


def test_clear_run_of_a_paused_run_without_a_result_before(agent):
    from pydantic_ai import DeferredToolRequests
    from pydantic_ai.messages import ToolCallPart

    panel, ws = agent.panel, agent.ws
    call = ToolCallPart("optimize", {}, tool_call_id="c1")
    paused = SimpleNamespace(
        output=DeferredToolRequests(approvals=[call], metadata={"c1": {"reason": "pause"}})
    )
    agent.run = lambda state: (state.add_plan(agent.cst, agent.dose), paused)[1]
    _run(agent)
    assert ws.result is not None

    panel.clear_run()
    assert ws.result is None and not ws.results


def test_another_patient_ends_the_last_run(agent):
    from pyRadPlan.result import create_result

    panel, ws = agent.panel, agent.ws
    agent.run = lambda state: (state.add_plan(agent.cst, agent.dose), _report())[1]
    _run(agent)
    assert panel._session is not None and list(ws.results) == ["Plan_1"]

    ws.refresh()  # the same patient
    assert panel._session is not None and panel.result_widget.table.rowCount() == 1

    cleared = []
    panel.run_cleared.connect(lambda: cleared.append(True))
    other = create_result({"physical_dose": agent.dose})
    ws.set_many(ct=ws.ct.model_copy(), cst=agent.cst, results={"Plan_1": other})

    assert cleared and panel._session is None and panel._last_state is None
    assert not panel._picks and not panel._agent_doses and not panel._log.toPlainText()
    assert panel.result_widget.table.rowCount() == 0 and panel._feedback_box.isHidden()
    assert ws.results["Plan_1"] is not None  # the new patient's result of the same name stays
    assert list(ws.results) == ["Plan_1"]
    panel.continue_agent()
    panel._on_load_requested(1)
    assert not panel.is_busy and len(agent.calls) == 1


def test_shutdown_does_not_wait_for_a_blocked_request(agent):
    panel = agent.panel
    release = threading.Event()
    entered = threading.Event()

    def blocked(state):
        entered.set()
        release.wait(30)
        state.add_plan(agent.cst, agent.dose)  # must not reach the panel any more
        return _report()

    agent.run = blocked
    panel.run_agent()
    assert entered.wait(10)
    try:
        start = time.monotonic()
        panel.shutdown(timeout_ms=200)
        elapsed = time.monotonic() - start
        assert elapsed < 5.0 and not panel.is_busy and panel._thread is None
        assert len(_agent_threads()) == 1  # the request goes on by itself
    finally:
        release.set()
        for thread in _agent_threads():
            thread.join(10)
    QApplication.processEvents()
    assert not _agent_threads()
    assert panel.result_widget.table.rowCount() == 0 and panel._session is None


def test_agentic_plan_widget_without_pydantic_ai(qapp, monkeypatch):
    from pyRadPlan.gui.widgets.agent import AgenticPlanWidget
    from pyRadPlan.gui.widgets.ai import AI_MISSING_TIP
    from pyRadPlan.plan import PhotonPlan

    monkeypatch.setitem(sys.modules, "pydantic_ai", None)
    ws = WorkspaceManager()
    widget = AgenticPlanWidget(ws)
    failures = []
    widget.update_failed.connect(failures.append)
    ws.set_many(pln=PhotonPlan(num_of_fractions=20, dose_convention="total"))
    ws.set_many(pln=PhotonPlan(num_of_fractions=20, dose_convention="per_fraction"))

    assert not failures and widget._btn_load.menu().isEmpty()
    for button in (widget._btn_load, widget._btn_import, widget._btn_export):
        assert not button.isEnabled() and button.toolTip() == AI_MISSING_TIP
    widget.close()
    widget.deleteLater()


def test_goal_operator_uses_the_compact_combo_box(qapp):
    from pyRadPlan.gui.widgets._base import CompactComboBox
    from pyRadPlan.gui.widgets.agent import AgenticPlanWidget
    from pyRadPlan.gui.widgets.agent._agentic_plan_widget import _OP

    widget = AgenticPlanWidget(WorkspaceManager())
    widget._add_goal_row("PTV", "D95", ">=", 57.0)
    combo = widget._tbl_goals.cellWidget(0, _OP)
    assert isinstance(combo, CompactComboBox) and combo.currentText() == ">="
    widget.close()
    widget.deleteLater()
