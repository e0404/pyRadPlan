import pytest
import logging

from pyRadPlan.optimization.solvers import get_available_solvers, get_solver, SolverBase


def test_get_available_solvers():
    solvers = get_available_solvers()
    assert isinstance(solvers, dict)
    assert len(solvers) > 0
    assert all([isinstance(k, str) for k in solvers.keys()])
    assert all([issubclass(v, SolverBase) for v in solvers.values()])


def test_get_solver():
    solvers = get_available_solvers()
    for solver_name, solver_class in solvers.items():
        solver = get_solver(solver_name)
        assert isinstance(solver, solver_class)
        assert isinstance(solver, SolverBase)
        assert solver.short_name == solver_name
        assert solver.name == solver_class.name
        assert solver.short_name == solver_class.short_name


def test_get_solver_invalid():
    with pytest.raises(ValueError):
        get_solver("invalid_solver")
    with pytest.raises(ValueError):
        get_solver(123)
    with pytest.raises(ValueError):
        get_solver({"name": "invalid_solver"})
    with pytest.raises(ValueError):
        get_solver({"options": {"tol": 1e-12}})  # missing 'name'


def test_get_solver_from_dict():
    solvers = get_available_solvers()
    for solver_name, solver_class in solvers.items():
        solver = get_solver({"name": solver_name})
        assert isinstance(solver, solver_class)
        assert solver.short_name == solver_name


def test_get_solver_from_dict_assigns_attributes():
    solvers = get_available_solvers()
    if "scipy" not in solvers:
        pytest.skip("scipy solver not registered")
    solver = get_solver({"name": "scipy", "max_iter": 42, "display": False})
    assert solver.max_iter == 42
    assert solver.display is False


def test_get_solver_from_dict_merges_options():
    # Needs a solver that actually ships default options to merge into; OptimizerSciPy has
    # none, because its tolerances come from `abs_obj_tol` rather than from `options`.
    with_defaults = [
        name for name, cls in get_available_solvers().items() if getattr(cls(), "options", None)
    ]
    if not with_defaults:
        pytest.skip("no registered solver ships default options")

    name = with_defaults[0]
    untouched = next(iter(get_solver(name).options))

    solver = get_solver({"name": name, "options": {"print_level": 1}})

    # The user-provided key should be set, and unrelated defaults should be preserved.
    assert solver.options["print_level"] == 1
    assert untouched in solver.options


def test_get_solver_from_dict_warns_on_unknown_attribute(caplog):
    solvers = get_available_solvers()
    if "scipy" not in solvers:
        pytest.skip("scipy solver not registered")
    with caplog.at_level(logging.WARNING, logger="pyRadPlan.optimization.solvers._factory"):
        get_solver({"name": "scipy", "not_a_real_attribute": 123})
    assert any("not_a_real_attribute" in m for m in caplog.messages)
