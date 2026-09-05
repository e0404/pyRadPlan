import logging

import pytest

from pyRadPlan.core import ComputeControl, ProgressReporter, StatusReport, observe_control
from pyRadPlan.optimization.problems import (
    NonLinearFluencePlanningProblem,
    NonLinearPlanningProblem,
    PlanningProblem,
)
from pyRadPlan.optimization.solvers import OptimizerSciPy, SolverBase
from pyRadPlan.plan import create_pln


@pytest.fixture
def only_scipy(monkeypatch):
    """Pretend ipopt is not installed, i.e. only the scipy solver is registered."""
    monkeypatch.setattr(
        "pyRadPlan.optimization.problems._optiprob.get_available_solvers",
        lambda: {"scipy": OptimizerSciPy},
    )


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


def test_unavailable_default_solver_falls_back(only_scipy, caplog):
    with caplog.at_level(logging.WARNING, logger="pyRadPlan.optimization.problems._optiprob"):
        prob = NonLinearFluencePlanningProblem()

    assert prob.solver == "scipy"
    assert any("ipopt" in m and "falling back" in m for m in caplog.messages)


@pytest.mark.parametrize("solver", ["ipopt", {"name": "ipopt", "options": {"tol": 1e-12}}])
def test_unavailable_explicit_solver_raises(only_scipy, solver):
    pln = create_pln(
        {"radiation_mode": "protons", "machine": "Generic", "prop_opt": {"solver": solver}}
    )
    with pytest.raises(ValueError, match="explicitly requested"):
        NonLinearFluencePlanningProblem(pln)


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


def _solved_solver(small_proton_case, prop_opt):
    """Run a problem through initialization and hand back the configured solver."""
    pln, ct, cst, stf, dij = small_proton_case
    pln.prop_opt = prop_opt

    prob = NonLinearFluencePlanningProblem(pln)
    prob.solve(ct, cst, stf, dij)
    return prob.solver


def test_solver_dict_configuration_survives(small_proton_case):
    """max_iter/display from the solver dict are kept when not set at the top level."""
    solver = _solved_solver(
        small_proton_case, {"solver": {"name": "scipy", "max_iter": 42, "display": False}}
    )

    assert solver.max_iter == 42
    assert solver.display is False


def test_top_level_overrides_solver_dict(small_proton_case):
    """An explicitly set top-level value takes precedence over the same key in the dict."""
    solver = _solved_solver(
        small_proton_case,
        {"solver": {"name": "scipy", "max_iter": 42, "display": False}, "max_iter": 7},
    )

    assert solver.max_iter == 7
    # display was not set at the top level, so the dict value still stands.
    assert solver.display is False


def test_unset_properties_leave_solver_defaults(small_proton_case):
    """Without any configuration the solver keeps its own defaults."""
    reference = OptimizerSciPy()
    solver = _solved_solver(small_proton_case, {"solver": "scipy"})

    assert solver.max_iter == reference.max_iter
    assert solver.display == reference.display


class _DirectSolver(SolverBase):
    """A non-iterative solver: no iterations to cap and no output to toggle."""

    name = "Direct"
    short_name = "direct"
    gpu_compatible = False

    def _solve_problem(self, x0):
        return x0, {"status": 0}

    def _callback(self, *args, **kwargs):
        return True


@pytest.mark.parametrize("prop", ["max_iter", "display"])
def test_unsupported_solver_property_is_reported(prop, caplog):
    """A knob the solver does not define is warned about, not silently attached to it."""
    prob = NonLinearFluencePlanningProblem()
    prob.solver = _DirectSolver()
    setattr(prob, prop, 200 if prop == "max_iter" else False)

    with caplog.at_level(logging.WARNING, logger="pyRadPlan.optimization.problems._optiprob"):
        prob._propagate_solver_properties()

    assert not hasattr(prob.solver, prop)
    assert any(prop in message and "direct" in message for message in caplog.messages)


def test_supported_properties_still_reach_the_solver():
    """The guard does not get in the way of a solver that does define the knobs."""
    prob = NonLinearFluencePlanningProblem()
    prob.solver = OptimizerSciPy()
    prob.max_iter = 200
    prob.display = False

    prob._propagate_solver_properties()

    assert prob.solver.max_iter == 200
    assert prob.solver.display is False
