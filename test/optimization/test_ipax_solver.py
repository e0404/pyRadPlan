import pytest
import array_api_strict as xp
import array_api_compat
import array_api_extra as xpx
import numpy as np


from pyRadPlan.optimization.solvers import get_solver, OptimizerIpax, SolverBase

if OptimizerIpax is None:
    pytest.skip("ipax not installed", allow_module_level=True)

import ipax  # noqa: E402  (only importable once the solver registered)


def build_quadratic_problem(solver):
    def objective(x):
        return float(x[0] ** 2 + x[1] ** 2)

    def gradient(x):
        return np.asarray([2 * x[0], 2 * x[1]], dtype=np.float64)

    solver.objective = objective
    solver.gradient = gradient
    return np.asarray([1.0, -1.0], dtype=np.float64)


def test_get_solver_ipax():
    solver = get_solver("ipax")
    assert isinstance(solver, OptimizerIpax)
    assert isinstance(solver, SolverBase)
    assert solver.short_name == "ipax"
    assert solver.gpu_compatible is True


def test_ipax_quadratic_solution():
    solver = get_solver("ipax")
    x0 = build_quadratic_problem(solver)
    # API: allow_keyboard_cancel is configured on the solver, not per-call
    solver.allow_keyboard_cancel = False
    res, status = solver.solve(x0)
    # Interior-point methods approach the (bound) optimum from the feasible
    # interior, so they settle a little short of exactly zero.
    assert np.allclose(res, 0.0, atol=5e-3)


def test_simple_problem_ipax():
    solver = get_solver("ipax")

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
    assert xp.all(xpx.isclose(result, 0.0, atol=5e-3))


def test_bound_array_handling():
    solver = get_solver("ipax")
    x0 = np.asarray([1.0, -1.0], dtype=np.float64)

    # Finite bounds broadcast to full-length vectors.
    lower = solver._bound_array(np, -1.0, x0, is_lower=True)
    upper = solver._bound_array(np, 5.0, x0, is_lower=False)
    assert lower.shape == x0.shape
    assert upper.shape == x0.shape
    assert np.all(lower == -1.0)
    assert np.all(upper == 5.0)

    # Infinite scalar bounds collapse to ipax's "unbounded" sentinel (None).
    assert solver._bound_array(np, -np.inf, x0, is_lower=True) is None
    assert solver._bound_array(np, np.inf, x0, is_lower=False) is None
    assert solver._bound_array(np, None, x0, is_lower=True) is None


def test_ipax_respects_bounds():
    solver = get_solver("ipax")

    def objective(x):
        return float(x[0] ** 2 + x[1] ** 2)

    def gradient(x):
        return np.asarray([2 * x[0], 2 * x[1]], dtype=np.float64)

    solver.objective = objective
    solver.gradient = gradient
    # Force the unconstrained minimum (origin) out of the feasible region.
    solver.bounds = [1.0, np.inf]

    x0 = np.asarray([2.0, 2.0], dtype=np.float64)
    res, status = solver.solve(x0)

    assert np.all(res >= 1.0 - 1e-4)
    assert np.allclose(res, 1.0, atol=1e-3)


def test_ipax_reports_status_per_iteration():
    """The iteration callback must reach ``status_callback`` (regression: it raised)."""
    solver = get_solver("ipax")
    x0 = build_quadratic_problem(solver)

    seen = []

    def status_callback(message="", **data):
        seen.append(data)
        return True

    solver.status_callback = status_callback
    solver.solve(x0)

    assert seen, "solver never reported iteration status"
    assert seen[0]["iteration"] == 0
    assert {"objective", "kkt_error", "mu", "step"} <= set(seen[0])


def test_ipax_status_callback_can_stop():
    """Returning False from ``status_callback`` terminates the solve cooperatively."""
    solver = get_solver("ipax")
    x0 = build_quadratic_problem(solver)

    calls = []

    def status_callback(message="", **data):
        calls.append(data)
        return len(calls) < 2

    solver.status_callback = status_callback
    solver.solve(x0)

    assert len(calls) == 2


def test_ipax_forwards_budgets_to_options(monkeypatch):
    """``max_iter`` / ``max_time`` are plumbed into the ipax options object."""
    solver = get_solver("ipax")
    x0 = build_quadratic_problem(solver)
    solver.max_iter = 7
    solver.max_time = 42.0

    captured = {}
    real_solve = ipax.solve

    def spy(problem, x0, *, options=None, **kwargs):
        captured["options"] = options
        return real_solve(problem, x0, options=options, **kwargs)

    monkeypatch.setattr(ipax, "solve", spy)
    solver.solve(x0)

    assert captured["options"].max_iter == 7
    assert captured["options"].max_time == 42.0


@pytest.mark.parametrize("backend", ["strict", "torch"])
@pytest.mark.parametrize("force_cpu", [False, True])
def test_backend_arrays_and_vector_bounds(backend, force_cpu):
    namespace = xp if backend == "strict" else pytest.importorskip("array_api_compat.torch")
    solver = get_solver("ipax")
    solver.allow_keyboard_cancel = False
    solver.force_cpu_inner_solve = force_cpu

    def objective(x):
        assert array_api_compat.array_namespace(x) is namespace
        return namespace.sum(x**2)

    solver.objective = objective
    solver.gradient = lambda x: 2 * x
    solver.bounds = (
        namespace.asarray([1.0, 2.0], dtype=namespace.float64),
        namespace.asarray([3.0, 4.0], dtype=namespace.float64),
    )
    x0 = namespace.asarray([2.0, 3.0], dtype=namespace.float64)
    result, info = solver.solve(x0)
    assert array_api_compat.array_namespace(result) is namespace
    assert array_api_compat.device(result) == array_api_compat.device(x0)
    assert np.allclose(np.asarray(result), [1.0, 2.0], atol=1e-3)
    assert info["status"] in ("optimal", "acceptable")


def test_cpu_inner_solve_keeps_the_input_cuda_device():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    solver = get_solver("ipax")
    solver.allow_keyboard_cancel = False
    solver.force_cpu_inner_solve = True
    x0 = torch.tensor([2.0, 3.0], dtype=torch.float64, device="cuda")

    def objective(x):
        assert x.device == x0.device
        return torch.sum(x**2)

    solver.objective = objective
    solver.gradient = lambda x: 2 * x
    solver.bounds = (torch.ones_like(x0), torch.full_like(x0, 4.0))
    result, _ = solver.solve(x0)
    assert result.device == x0.device
    assert torch.allclose(result, torch.ones_like(result), atol=1e-3)
