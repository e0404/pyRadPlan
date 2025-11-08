import pytest
import array_api_strict as xp
import array_api_compat
import array_api_extra as xpx


from pyRadPlan.optimization.solvers import get_solver, OptimizerIpopt, SolverBase

if OptimizerIpopt is None:
    pytest.skip("IPOPT not installed", allow_module_level=True)


def build_quadratic_problem(solver):
    def objective(x):
        return float(x[0] ** 2 + x[1] ** 2)

    def gradient(x):
        return np.asarray([2 * x[0], 2 * x[1]], dtype=np.float64)

    solver.objective = objective
    solver.gradient = gradient
    return np.asarray([1.0, -1.0], dtype=np.float64)


def test_validate_problem_basic():
    solver = get_solver("ipopt")
    x0 = build_quadratic_problem(solver)
    # ensure problem builds through _validate_ipopt_problem without executing solve
    cfg = {
        "n": x0.size,
        "x_l": np.full_like(x0, -10.0),
        "x_u": np.full_like(x0, 10.0),
        "m": 0,
        "g_l": np.empty((0,)),
        "g_u": np.empty((0,)),
        "eval_f": solver.objective,
        "eval_grad_f": solver.gradient,
        "eval_g": lambda _x, _out: None,
        "eval_jac_g": lambda _x, _out: None,
        "eval_h": None,
        "sparsity_indices_jac_g": (np.array([], dtype=int), np.array([], dtype=int)),
        "sparsity_indices_h": (np.array([], dtype=int), np.array([], dtype=int)),
        "intermediate_callback": lambda *a: True,
        "ipopt_options": solver.options,
    }
    problem = solver._validate_ipopt_problem(cfg)
    assert problem is not None


def test_ipopt_quadratic_solution():
    solver = get_solver("ipopt")
    x0 = build_quadratic_problem(solver)
    # API: allow_keyboard_cancel is configured on the solver, not per-call
    solver.allow_keyboard_cancel = False
    res, status = solver.solve(x0)
    assert np.allclose(res, 0.0, atol=1e-4)
    assert isinstance(status, int)


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
