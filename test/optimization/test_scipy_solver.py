import logging

import array_api_strict as xp
import array_api_compat
import array_api_extra as xpx
import numpy as np

from scipy.optimize import Bounds, minimize

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


# A problem whose stopping point actually responds to the tolerance, so that a tolerance which
# never reaches SciPy is visible as an unchanged iteration count.
_PROBE_TARGET = np.arange(1, 21, dtype=float)
_PROBE_X0 = np.full(20, 50.0)


def _probe_objective(x):
    return float(np.sum((x - _PROBE_TARGET) ** 2) + 1e-3 * np.sum(x**4))


def _probe_gradient(x):
    return 2.0 * (x - _PROBE_TARGET) + 4e-3 * x**3


def _probe_solver(**config):
    solver = get_solver({"name": "scipy", **config})
    solver.display = False
    solver.allow_keyboard_cancel = False
    solver.objective = _probe_objective
    solver.gradient = _probe_gradient
    solver.bounds = [np.full(20, -1e3), np.full(20, 1e3)]
    return solver


def _solve_probe(**config):
    return _probe_solver(**config).solve(_PROBE_X0)


def test_abs_obj_tol_reaches_scipy():
    """Regression: `abs_obj_tol` was shadowed by ftol/gtol in `options` and did nothing.

    SciPy maps ``minimize(tol=...)`` onto the method's tolerance options with ``setdefault``,
    so seeding ``ftol``/``gtol`` in ``options`` silently discarded it and every tolerance
    produced a bit-identical run.
    """
    _, loose = _solve_probe(abs_obj_tol=1e-1)
    _, tight = _solve_probe(abs_obj_tol=1e-14)

    assert loose["num_iter"] < tight["num_iter"], (
        "a looser objective tolerance must stop the solver earlier; identical counts mean "
        "abs_obj_tol never reached SciPy"
    )
    assert loose["fun"] > tight["fun"]


def test_default_tolerance_matches_the_historical_options():
    """Making `abs_obj_tol` effective must not change what a default solve computes.

    The solver used to hard-code ``ftol``/``gtol`` at 1e-5 in ``options``; the default
    ``abs_obj_tol`` is set to the same value so existing plans are unaffected.
    """
    x, info = _solve_probe()

    legacy = minimize(
        x0=_PROBE_X0,
        fun=_probe_objective,
        jac=_probe_gradient,
        method="L-BFGS-B",
        bounds=Bounds(lb=np.full(20, -1e3), ub=np.full(20, 1e3)),
        options={"ftol": 1e-5, "gtol": 1e-5, "maxiter": OptimizerSciPy().max_iter},
    )

    assert info["num_iter"] == legacy.nit
    assert np.array_equal(np.asarray(x), legacy.x)


def test_stored_options_are_not_mutated_by_a_solve():
    """Regression: `maxiter` was written into the stored dict, overwriting what a caller set."""
    solver = _probe_solver(options={"maxiter": 3})

    _, info = solver.solve(_PROBE_X0)

    assert solver.options == {"maxiter": 3}, "the stored dict is left as the caller wrote it"
    assert info["num_iter"] > 3, "the attribute (500), not options (3), capped the run"


def test_iteration_cap_from_options_is_reported(caplog):
    """The iteration cap really is an attribute, so a value in `options` is warned about."""
    solver = _probe_solver(options={"maxiter": 3})

    with caplog.at_level(logging.WARNING, logger="pyRadPlan.optimization.solvers._scipy_solver"):
        solver.solve(_PROBE_X0)

    assert any("'maxiter'" in m and "max_iter" in m for m in caplog.messages)


def test_explicit_tolerances_stay_authoritative():
    """SciPy exposes several tolerances per method; an explicit one is kept, not replaced.

    A single `abs_obj_tol` cannot express independently tuned `ftol`/`gtol`/`xtol`, and callers
    on earlier releases configured them through the public `options` dictionary, so what they
    set stays in force - the precedence SciPy itself applies.
    """
    solver = _probe_solver(options={"ftol": 1e-14, "gtol": 1e-14})
    solver.abs_obj_tol = 1e-1

    options = solver._effective_options()
    assert options["ftol"] == 1e-14
    assert options["gtol"] == 1e-14

    _, explicit = solver.solve(_PROBE_X0)
    _, from_attribute = _solve_probe(abs_obj_tol=1e-1)
    assert explicit["num_iter"] > from_attribute["num_iter"], (
        "the tighter explicit tolerance must outrank the looser abs_obj_tol"
    )


def test_a_shadowed_abs_obj_tol_is_reported(caplog):
    """Configuring both is ambiguous, so the one that loses is named rather than left silent."""
    solver = _probe_solver(options={"ftol": 1e-14})
    solver.abs_obj_tol = 1e-1

    with caplog.at_level(logging.WARNING, logger="pyRadPlan.optimization.solvers._scipy_solver"):
        solver.solve(_PROBE_X0)

    assert any("'ftol'" in m and "abs_obj_tol" in m for m in caplog.messages)


def test_no_warning_when_only_a_tolerance_option_is_set(caplog):
    """Setting a tolerance the ordinary way, with abs_obj_tol left alone, is not noteworthy."""
    solver = _probe_solver(options={"ftol": 1e-14})

    with caplog.at_level(logging.WARNING, logger="pyRadPlan.optimization.solvers._scipy_solver"):
        solver.solve(_PROBE_X0)

    assert not any("abs_obj_tol" in m for m in caplog.messages)


def test_max_iter_caps_tnc_which_does_not_take_maxiter():
    """Regression: TNC's cap is `maxfun`; sending `maxiter` left the run uncapped.

    SciPy reports `maxiter` as an unknown option for TNC and carries on, so `max_iter` had no
    effect at all for that method.
    """
    solver = _probe_solver()
    solver.method = "TNC"
    solver.max_iter = 1

    assert solver._effective_options() == {"maxfun": 1}

    _, info = solver.solve(_PROBE_X0)
    assert info["nfev"] <= 1
