import pytest
import array_api_strict as xp
import array_api_compat
import array_api_extra as xpx
import numpy as np

from pyRadPlan import settings
from pyRadPlan.optimization.solvers import get_solver, OptimizerIpopt, SolverBase
from pyRadPlan.util.openmp import blocked_by_openmp

skipif_no_ipopt = pytest.mark.skipif(
    OptimizerIpopt is None,
    reason="IPOPT is not installed on this system.",
)


def build_quadratic_problem(solver):
    def objective(x):
        return x[0] ** 2 + x[1] ** 2

    def gradient(x):
        return np.asarray([2 * x[0], 2 * x[1]], dtype=np.float64)

    solver.objective = objective
    solver.gradient = gradient
    return np.asarray([1.0, -1.0], dtype=np.float64)


@skipif_no_ipopt
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


@skipif_no_ipopt
def test_ipopt_quadratic_solution():
    solver = get_solver("ipopt")
    x0 = build_quadratic_problem(solver)
    # API: allow_keyboard_cancel is configured on the solver, not per-call
    solver.allow_keyboard_cancel = False
    res, result_info = solver.solve(x0)
    assert np.allclose(res, 0.0, atol=1e-4)
    assert isinstance(result_info, dict)
    assert isinstance(result_info["status"], int)
    assert result_info["num_iter"] > 0


@skipif_no_ipopt
def test_get_solver_ipopt():
    solver = get_solver("ipopt")
    assert isinstance(solver, OptimizerIpopt)
    assert isinstance(solver, SolverBase)
    assert solver.short_name == "ipopt"


@skipif_no_ipopt
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
    result, result_info = solver.solve(x0)

    assert array_api_compat.array_namespace(result) is xp
    assert isinstance(result_info, dict)
    assert isinstance(result_info["status"], int)

    assert xp.all(xpx.isclose(result, 0.0, atol=1e-4))


def _shifted_quadratic(solver):
    target = np.array([2.0, -1.0])
    solver.display = False
    solver.objective = lambda x: np.sum((x - target) ** 2)
    solver.gradient = lambda x: 2 * (x - target)


@skipif_no_ipopt
def test_ipopt_respects_bounds():
    """Regression: `bounds` was ignored and IPOPT always solved on [0, inf)."""
    solver = get_solver("ipopt")
    _shifted_quadratic(solver)
    solver.bounds = (np.array([0.0, -5.0]), 1.0)

    x, _ = solver.solve(np.zeros(2))

    assert np.allclose(x, [1.0, -1.0], atol=1e-6)


@skipif_no_ipopt
def test_ipopt_accepts_float32():
    """Regression: a float32 x0 was read as float64 memory and gave garbage."""
    solver = get_solver("ipopt")
    _shifted_quadratic(solver)

    x, _ = solver.solve(np.full(2, 0.5, dtype=np.float32))

    assert np.allclose(x, [2.0, 0.0], atol=1e-6)


@skipif_no_ipopt
def test_ipopt_leaves_x0_untouched():
    """Regression: IPOPT wrote the solution into the caller's x0."""
    solver = get_solver("ipopt")
    _shifted_quadratic(solver)
    x0 = np.zeros(2)

    x, _ = solver.solve(x0)

    assert np.array_equal(x0, np.zeros(2))
    assert x is not x0


@skipif_no_ipopt
def test_ipopt_evaluates_on_the_device_of_x0(monkeypatch):
    """Regression: with a GPU preferred, a CPU x0 was evaluated (and returned) on the GPU."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    if blocked_by_openmp("ipyopt"):
        pytest.skip("IPOPT blocked by an OpenMP clash with torch")
    monkeypatch.setattr(settings.xp, "prefer_gpu", True)
    x0 = torch.ones(2, dtype=torch.float64)

    def objective(x):
        assert x.device == x0.device
        return torch.sum(x**2)

    solver = get_solver("ipopt")
    solver.display = False
    solver.objective = objective
    solver.gradient = lambda x: 2 * x

    x, _ = solver.solve(x0)

    assert x.device == x0.device
