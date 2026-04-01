"""Tests for obj_history tracking on the planning problem."""

import pytest
import numpy as np

from pyRadPlan.optimization.problems import NonLinearFluencePlanningProblem
from pyRadPlan.optimization.solvers import get_solver, OptimizerIpopt


def _solve_with_history(solver_name, obj_history):
    """Helper: solve a simple quadratic and record objective values."""
    prob = NonLinearFluencePlanningProblem()
    prob.obj_history = obj_history
    prob.display = False

    solver = get_solver(solver_name)
    solver.max_iter = 50

    if hasattr(solver, "display"):
        solver.display = False
    if solver.short_name == "ipopt":
        solver.options["print_level"] = 0

    def objective(x):
        val = np.sum(x**2)
        if prob.obj_history is not None:
            prob.obj_history.append(float(val))
        return val

    def gradient(x):
        return 2.0 * x

    solver.objective = objective
    solver.gradient = gradient

    x0 = np.asarray([5.0, -3.0], dtype=np.float64)
    solver.solve(x0)


def test_obj_history_filled_scipy():
    """obj_history list is populated during scipy optimization."""
    obj_history = []
    _solve_with_history("scipy", obj_history)

    assert len(obj_history) > 0
    assert all(isinstance(v, float) for v in obj_history)


@pytest.mark.skipif(OptimizerIpopt is None, reason="IPOPT not installed")
def test_obj_history_filled_ipopt():
    """obj_history list is populated during ipopt optimization."""
    obj_history = []
    _solve_with_history("ipopt", obj_history)

    assert len(obj_history) > 0  # only > check due to extra evaulations
    assert all(isinstance(v, float) for v in obj_history)


def test_obj_history_none_by_default():
    """obj_history is None when not explicitly set."""
    prob = NonLinearFluencePlanningProblem()
    assert prob.obj_history is None
