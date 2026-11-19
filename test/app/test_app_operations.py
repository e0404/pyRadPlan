"""Host-scheduled operations (begin / run / finish / abort) and the session's host options."""

import logging
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from pyRadPlan.app import Operation, PreconditionError, Session, Workspace
from pyRadPlan.core import ProgressReporter


def _reporting_operation(session, commit=None):
    def compute():
        reporter = ProgressReporter()
        reporter.console_progress = False
        for _ in reporter.track(range(3), name="Step"):
            pass
        reporter.report_warning("careful")
        logging.getLogger("pyRadPlan.test").warning("logged")
        return threading.current_thread().name

    return Operation(
        "custom", "Custom", compute, commit or (lambda value: session.workspace.set("pln", value))
    )


def test_an_operation_runs_on_a_worker_thread_and_commits_on_the_host_thread():
    session = Session()
    events = []
    session.add_listener(events.append)
    seen = []
    session.workspace.subscribe(seen.append)
    operation = _reporting_operation(session)

    session.begin(operation)
    assert session.busy and events[-1]["state"]["busy"] == "Custom"
    with pytest.raises(PreconditionError) as busy:
        session.begin(_reporting_operation(session))
    assert busy.value.code == "busy"

    values = []
    worker = threading.Thread(target=lambda: values.append(session.run(operation)), name="w")
    worker.start()
    worker.join(10)
    assert values == ["w"] and not seen  # run never writes the workspace
    kinds = [event["event"] for event in events]
    assert "progress" in kinds
    assert any(e["event"] == "log" and e["message"] == "careful" for e in events)
    assert any(e["event"] == "log" and e["message"] == "logged" for e in events)

    session.finish(operation, values[0])
    assert session.workspace.pln == "w" and seen == [["pln"]]
    assert not session.busy and events[-1]["state"]["busy"] is None
    session.close()


def test_run_needs_begin_and_abort_ends_the_operation():
    session = Session()
    operation = _reporting_operation(session)
    with pytest.raises(RuntimeError, match="begin"):
        session.run(operation)
    session.begin(operation)
    session.abort(operation, RuntimeError("engine failed"))
    assert not session.busy
    session.abort(operation)  # a second abort is a no-op

    def failing_commit(value):
        raise ValueError("bad result")

    operation = _reporting_operation(session, commit=failing_commit)
    session.begin(operation)
    value = session.run(operation)
    with pytest.raises(ValueError, match="bad result"):
        session.finish(operation, value)
    assert not session.busy


def test_a_host_with_its_own_log_handler_skips_log_capture():
    session = Session(capture_logs=False)
    events = []
    session.add_listener(events.append)
    session.execute(_reporting_operation(session))
    messages = [e["message"] for e in events if e["event"] == "log"]
    assert messages == ["careful"]  # the warning report, not the log record


def test_events_from_another_thread_are_not_dropped_while_one_is_emitted():
    session = Session()
    seen = []
    inside, release = threading.Event(), threading.Event()

    def listener(event):
        seen.append(event.get("message"))
        if event.get("message") == "main":
            session._log("info", "nested", "test")  # re-entrant in this context: dropped
            inside.set()
            release.wait(5)

    session.add_listener(listener)

    def other():
        inside.wait(5)
        session._log("info", "worker", "test")
        release.set()

    thread = threading.Thread(target=other)
    thread.start()
    session._log("info", "main", "test")
    thread.join(5)
    assert seen == ["main", "worker"]


def test_state_is_built_only_for_listeners(monkeypatch):
    session = Session()
    calls = []
    original = Session.state
    monkeypatch.setattr(Session, "state", lambda self: calls.append(1) or original(self))
    session.workspace.pln = object()
    assert calls == []
    session.add_listener(lambda event: None)
    session.workspace.pln = object()
    assert calls == [1]


def test_state_tolerates_placeholder_objects():
    workspace = Workspace()
    session = Session(workspace=workspace, auto_plan=False)
    voi = SimpleNamespace(name="PTV", objectives=[])
    workspace.set_many(
        ct=object(),
        cst=SimpleNamespace(vois=[voi]),
        pln=object(),
        stf=object(),
        dij=object(),
        result={"w": np.ones(2)},
    )
    state = session.state()
    assert state["ct"] is None and state["plan"] is None and state["structures"] == []
    assert set(state["stages"].values()) == {"done"}
    assert state["status"] == "Optimization result available"
    workspace.ct = object()
    assert session.tracker.dij_stale and session.state()["stages"]["dij"] == "stale"


def test_merge_loaded_reports_its_warnings():
    session = Session(auto_plan=False)
    warnings = session.merge_loaded({"cst": object()})
    assert len(warnings) == 1 and "Load a CT" in warnings[0]
    assert session.workspace.cst is None


def _placeholder_patient(session):
    session.workspace.set_many(
        ct=object(), cst=object(), pln=SimpleNamespace(prop_stf={}), stf="current"
    )


def test_dose_influence_always_regenerates_the_beams(monkeypatch):
    # A current stf is replaced too: a structure edit moves the automatic isocenter
    # without outdating the beams.
    generated = iter(["first", "second"])
    monkeypatch.setattr("pyRadPlan.generate_stf", lambda ct, cst, pln: next(generated))
    monkeypatch.setattr("pyRadPlan.calc_dose_influence", lambda ct, cst, stf, pln: f"dij({stf})")
    session = Session(auto_plan=False)
    _placeholder_patient(session)
    assert session.stages()["stf"] == "done"
    session.calc_dose_influence()
    assert session.workspace.stf == "first" and session.workspace.dij == "dij(first)"
    session.workspace.cst = object()
    assert session.stages()["stf"] == "done"
    session.calc_dose_influence()
    assert session.workspace.stf == "second" and session.workspace.dij == "dij(second)"


@pytest.mark.parametrize(("preference", "solver"), [((), None), (("scipy",), "scipy")])
def test_an_empty_solver_preference_keeps_the_plan(monkeypatch, preference, solver):
    from pyRadPlan.plan import PhotonPlan

    received = []

    def optimize(ct, cst, stf, dij, pln, **kwargs):
        received.append(pln)
        return np.ones(2)

    monkeypatch.setattr("pyRadPlan.fluence_optimization", optimize)
    dij = SimpleNamespace(
        bio_model=None,
        alpha_dose=None,
        scenarios=None,
        compute_result_ct_grid=lambda weights, **kw: {"physical_dose": weights},
    )
    pln = PhotonPlan()
    session = Session(auto_plan=False, solver_preference=preference)
    session.workspace.set_many(ct=object(), cst=object(), pln=pln, stf=object(), dij=dij)
    summary = session.optimize()
    assert (received[0] is pln) is (solver is None)
    assert (received[0].prop_opt or {}).get("solver") == solver
    assert summary["solver"] == solver


@pytest.mark.parametrize("keeps", [True, False])
def test_an_interrupt_under_pyodide_stops_an_optimization_with_its_result(monkeypatch, keeps):
    """The browser's Stop (SIGINT) asks an optimization to stop; a second one interrupts."""
    import signal

    monkeypatch.setattr("pyRadPlan.app._session.running_in_pyodide", lambda: True)
    session = Session()
    events = []
    session.add_listener(events.append)
    seen = []

    def compute():
        signal.raise_signal(signal.SIGINT)
        seen.append(operation.control.stop_requested)
        signal.raise_signal(signal.SIGINT)
        return "last iterate"

    operation = Operation("optimize", "Optimizing", compute, lambda v: v)
    operation.keeps_partial_result = keeps
    previous = signal.getsignal(signal.SIGINT)
    session.history.start_optimization("Optimizing")
    with pytest.raises(KeyboardInterrupt):
        session.execute(operation)
    assert seen == ([True] if keeps else [])
    assert signal.getsignal(signal.SIGINT) is previous

    def stop_once():
        signal.raise_signal(signal.SIGINT)
        return "last iterate"

    stopping = Operation("optimize", "Optimizing", stop_once, lambda v: v)
    stopping.keeps_partial_result = True
    assert session.execute(stopping) == "last iterate"
    assert session.history.optimization["outcome"] == "stopped"


def test_an_interrupt_library_code_turns_into_another_error_still_stops(monkeypatch):
    """An interrupt reported as another error still stops the operation.

    numpy does this: an exception raised while it promotes the dtypes of a structured
    comparison (Python code, run e.g. by ``np.unique(..., axis=...)``) becomes a TypeError.
    """
    import signal

    monkeypatch.setattr("pyRadPlan.app._session.running_in_pyodide", lambda: True)

    def compute():
        try:
            signal.raise_signal(signal.SIGINT)
        except KeyboardInterrupt:
            raise TypeError("Cannot compare structured arrays") from None

    operation = Operation("dose_influence", "Dose influence", compute, lambda v: v)
    with pytest.raises(KeyboardInterrupt) as raised:
        Session().execute(operation)
    assert isinstance(raised.value.__cause__, TypeError)

    def fails():
        raise TypeError("a genuine error")

    failing = Operation("dose_influence", "Dose influence", fails, lambda v: v)
    with pytest.raises(TypeError, match="genuine"):
        Session().execute(failing)
