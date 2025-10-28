import pytest
import array_api_strict as xp
import array_api_compat
import array_api_extra as xpx


from pyRadPlan.optimization.solvers import get_solver, OptimizerIpopt, SolverBase

if OptimizerIpopt is None:
    pytest.skip("IPOPT not installed", allow_module_level=True)


def test_get_solver_ipopt():
    solver = get_solver("ipopt")
    assert isinstance(solver, OptimizerIpopt)
    assert isinstance(solver, SolverBase)
    assert solver.short_name == "ipopt"


def test_simple_problem_ipopt():
    solver = get_solver("ipopt")

    # Define the problem
    def objective(x):
        return xp.sum(x**2)

    def gradient(x):
        return 2 * x

    solver.objective = objective
    solver.gradient = gradient

    # Initial guess
    x0 = xp.asarray([1.0, 1.0], dtype=xp.float64)

    # Solve
    result, status = solver.solve(x0)

    assert array_api_compat.array_namespace(result) is xp
    assert isinstance(status, int)

    assert xp.all(xpx.isclose(result, 0.0, atol=1e-4))
