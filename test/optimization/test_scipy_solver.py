import array_api_strict as xp
import array_api_compat
import array_api_extra as xpx

from pyRadPlan.optimization.solvers import get_solver, OptimizerSciPy, SolverBase


def test_get_solver_scipy():
    solver = get_solver("scipy")
    assert isinstance(solver, OptimizerSciPy)
    assert isinstance(solver, SolverBase)
    assert solver.short_name == "scipy"
    assert solver.method == "L-BFGS-B"


def test_simple_problem_scipy():
    solver = get_solver("scipy")

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
    result = solver.solve(x0)

    assert xp.all(xpx.isclose(result[0], 0.0))
