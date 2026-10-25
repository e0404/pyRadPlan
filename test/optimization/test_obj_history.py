"""Tests for obj_history tracking on the planning problem."""

import pytest

from pyRadPlan.optimization.problems import NonLinearFluencePlanningProblem
from pyRadPlan.optimization.solvers import get_available_solvers

SOLVERS = sorted(get_available_solvers())


def test_obj_history_none_by_default():
    """obj_history is None when not explicitly set."""
    prob = NonLinearFluencePlanningProblem()
    assert prob.obj_history is None


@pytest.mark.parametrize("solver_name", SOLVERS)
def test_problem_objective_fills_history(solver_name, small_proton_case):
    """Setting obj_history makes the problem's own objective record each evaluation."""
    pln, ct, cst, stf, dij = small_proton_case
    pln.prop_opt = {"solver": solver_name, "display": False, "max_iter": 25}

    prob = NonLinearFluencePlanningProblem(pln)
    prob.obj_history = []
    prob.solve(ct, cst, stf, dij)

    assert len(prob.obj_history) > 0
    assert all(isinstance(value, float) for value in prob.obj_history)


@pytest.mark.parametrize("solver_name", SOLVERS)
def test_history_stays_empty_when_disabled(solver_name, small_proton_case):
    """A problem left with obj_history=None accumulates nothing while solving."""
    pln, ct, cst, stf, dij = small_proton_case
    pln.prop_opt = {"solver": solver_name, "display": False, "max_iter": 25}

    prob = NonLinearFluencePlanningProblem(pln)
    prob.solve(ct, cst, stf, dij)

    assert prob.obj_history is None


def test_history_is_reset_between_solves(small_proton_case):
    """Regression: a second solve appended to the first one's history instead of replacing it."""
    pln, ct, cst, stf, dij = small_proton_case
    pln.prop_opt = {"solver": SOLVERS[0], "display": False, "max_iter": 25}

    prob = NonLinearFluencePlanningProblem(pln)
    prob.obj_history = []

    prob.solve(ct, cst, stf, dij)
    first = list(prob.obj_history)
    held = prob.obj_history  # a caller keeping the list must see the second run, not both

    prob.solve(ct, cst, stf, dij)

    assert prob.obj_history == first
    assert held is prob.obj_history


def test_nonlin_fluence_declares_that_it_records():
    """The declaration is what fluence_optimization keys off, so it must match reality."""
    assert NonLinearFluencePlanningProblem.records_obj_history is True
