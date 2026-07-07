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


def _quadratic_solver():
    solver = get_solver("scipy")
    solver.objective = lambda x: xp.sum(x**2)
    solver.gradient = lambda x: 2 * x
    return solver


def test_scipy_emits_status_through_callback():
    solver = _quadratic_solver()
    reports = []
    solver.status_callback = lambda message="", **data: reports.append(data) or True

    solver.solve(xp.asarray([1.0, 1.0], dtype=xp.float64))

    assert reports, "expected at least one per-iteration status report"
    # iteration is always provided; objective is provided when scipy supplies it.
    assert all("iteration" in d for d in reports)
    assert [d["iteration"] for d in reports] == sorted(d["iteration"] for d in reports)


def test_scipy_status_callback_can_stop():
    solver = _quadratic_solver()
    calls = []

    def _cb(message="", **data):
        calls.append(data)
        return False  # request stop on the very first iteration

    solver.status_callback = _cb
    x, _info = solver.solve(xp.asarray([5.0, 5.0], dtype=xp.float64))

    assert len(calls) == 1  # stopped after the first callback
    # Stopping returns a valid (partial) iterate, not a crash.
    assert x is not None
