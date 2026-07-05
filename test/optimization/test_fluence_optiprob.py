from pyRadPlan.core import ComputeControl, ProgressReporter, StatusReport, observe_control
from pyRadPlan.optimization.problems import (
    NonLinearFluencePlanningProblem,
    NonLinearPlanningProblem,
    PlanningProblem,
)
from pyRadPlan.plan import create_pln


def test_construct():
    pln = create_pln({"radiation_mode": "protons", "machine": "Generic"})
    prob = NonLinearFluencePlanningProblem(pln)

    assert isinstance(prob, NonLinearFluencePlanningProblem)
    assert isinstance(prob, NonLinearPlanningProblem)
    assert isinstance(prob, PlanningProblem)


def test_construct_noplan():
    prob = NonLinearFluencePlanningProblem()

    assert isinstance(prob, NonLinearFluencePlanningProblem)
    assert isinstance(prob, NonLinearPlanningProblem)
    assert isinstance(prob, PlanningProblem)


def test_problem_is_progress_reporter():
    prob = NonLinearFluencePlanningProblem()
    assert isinstance(prob, ProgressReporter)


def test_emit_solver_status_reports_and_controls():
    prob = NonLinearFluencePlanningProblem()

    reports = []
    prob.add_report_observer(reports.append)

    # Forwards arbitrary data as a StatusReport and returns "continue" by default.
    assert prob._emit_solver_status(message="iter 1", iteration=1, objective=12.3) is True
    statuses = [r for r in reports if isinstance(r, StatusReport)]
    assert statuses and statuses[-1].data["objective"] == 12.3
    assert statuses[-1].data["iteration"] == 1

    # Honours a cooperative stop request via the active control.
    control = ComputeControl()
    with observe_control(control):
        assert prob._emit_solver_status(iteration=2) is True
        control.request_stop()
        assert prob._emit_solver_status(iteration=3) is False
