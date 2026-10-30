import logging

import numpy as np
import pytest

from pyRadPlan.core import (
    ComputeControl,
    ProgressReporter,
    StatusReport,
    observe_control,
    xp_utils,
)
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


def test_constant_rbe_evaluates_only_the_quantities_it_uses(small_proton_case, monkeypatch):
    """The photon LQ effect a dij with alphax / betax allows is not evaluated for a constant RBE."""
    pln, ct, cst, stf, dij = small_proton_case
    num_voxels = dij.dose_grid.num_voxels
    dij.alphax = np.full((num_voxels, 1), 0.1)
    dij.betax = np.full((num_voxels, 1), 0.05)
    dij.bio_model = {"model": "constant_rbe", "rbe": 1.1}
    pln.prop_opt = {"solver": {"name": "scipy", "max_iter": 2, "display": False}}

    prob = NonLinearFluencePlanningProblem(pln)
    prob.solve(ct, cst, stf, dij)

    assert sorted(q.identifier for q in prob._quantities) == ["physical_dose", "rbe_x_dose"]

    # The physical dose has no objective of its own, so it is differentiated only through the
    # chain rule of rbe_x_dose: one backprojection per scenario and gradient evaluation.
    dose = next(q for q in prob._quantities if q.identifier == "physical_dose")
    calls = []
    backprojection = dose._compute_chain_derivative_single_scenario
    monkeypatch.setattr(
        dose,
        "_compute_chain_derivative_single_scenario",
        lambda *args, **kwargs: calls.append(1) or backprojection(*args, **kwargs),
    )
    xp = prob._array_backend
    prob._objective_gradient(xp.full((dij.total_num_of_bixels,), 0.5, dtype=xp.float64))
    assert len(calls) == len(dose.scenarios)


@pytest.mark.parametrize("rbe_path", ["constant", "effect"])
def test_objective_jacobian_has_one_row_per_objective(small_proton_case, rbe_path):
    """Without the bypass, each objective gets its own gradient row, also across quantities."""
    pln, ct, cst, stf, dij = small_proton_case
    if rbe_path == "constant":
        dij.bio_model = {"model": "constant_rbe", "rbe": 1.1}
    else:
        num_voxels = dij.dose_grid.num_voxels
        dij.alphax = np.full((num_voxels, 1), 0.1)
        dij.betax = np.full((num_voxels, 1), 0.05)
        alpha_dose = np.empty_like(dij.physical_dose)
        sqrt_beta_dose = np.empty_like(dij.physical_dose)
        for i, mat in enumerate(dij.physical_dose.flat):
            alpha_dose.flat[i] = 0.12 * mat
            sqrt_beta_dose.flat[i] = np.sqrt(0.05) * mat
        dij.alpha_dose = alpha_dose
        dij.sqrt_beta_dose = sqrt_beta_dose
    pln.prop_opt = {"solver": {"name": "scipy", "max_iter": 2, "display": False}}

    prob = NonLinearFluencePlanningProblem(pln)
    prob.solve(ct, cst, stf, dij)
    rbe_x_dose = next(q for q in prob._quantities if q.identifier == "rbe_x_dose")
    assert rbe_x_dose.path == rbe_path
    xp = prob._array_backend
    x = xp.full((dij.total_num_of_bixels,), 0.5, dtype=xp.float64)
    gradient = xp_utils.to_numpy(prob._objective_gradient(x))

    prob.bypass_objective_jacobian = False
    prob._grad_cache = prob._grad_cache_intermediate = None
    for q in prob._quantities:
        q._w_grad_cache = None
    num_objectives = sum(len(obj_info[1]) for obj_info in prob._objective_list)
    jacobian = xp_utils.to_numpy(prob._objective_jacobian(x))

    assert num_objectives > 1
    assert prob._num_objectives == num_objectives
    assert jacobian.shape == (num_objectives, dij.total_num_of_bixels)
    assert np.allclose(jacobian.sum(axis=0), gradient, rtol=1e-5)
