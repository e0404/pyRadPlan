import pytest
import logging

from pyRadPlan.core import field_constraints
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


def test_get_solver_from_dict_merges_options(monkeypatch):
    from pyRadPlan.optimization.solvers import OptimizerSciPy
    from pyRadPlan.optimization.solvers._factory import SOLVERS

    # Exercise dictionary merging even when optional IPOPT is absent. ipax's
    # dataclass options have a separate test below.
    class SolverWithOptions(OptimizerSciPy):
        short_name = "test-options"

        def __init__(self):
            super().__init__()
            self.options = {"ftol": 1e-8, "maxls": 20}

    monkeypatch.setitem(SOLVERS, SolverWithOptions.short_name, SolverWithOptions)
    solver = get_solver({"name": SolverWithOptions.short_name, "options": {"ftol": 1e-6}})
    assert solver.options == {"ftol": 1e-6, "maxls": 20}
    assert get_solver(SolverWithOptions.short_name).options == {"ftol": 1e-8, "maxls": 20}


def test_get_solver_from_dict_warns_on_unknown_attribute(caplog):
    solvers = get_available_solvers()
    if "scipy" not in solvers:
        pytest.skip("scipy solver not registered")
    with caplog.at_level(logging.WARNING, logger="pyRadPlan.optimization.solvers._factory"):
        get_solver({"name": "scipy", "not_a_real_attribute": 123})
    assert any("not_a_real_attribute" in m for m in caplog.messages)


def test_solver_dict_merges_into_dataclass_options():
    if "ipax" not in get_available_solvers():
        pytest.skip("ipax not available")
    default = get_solver("ipax").options
    solver = get_solver(
        {"name": "ipax", "options": {"hessian": "exact", "krylov": {"method": "minres"}}}
    )
    assert solver.options.hessian == "exact"
    assert solver.options.krylov.method == "minres"
    assert solver.options.krylov.preconditioner == default.krylov.preconditioner
    assert solver.options.acceptable == default.acceptable


def test_solver_config_models_hide_the_wiring():
    for solver_cls in get_available_solvers().values():
        fields = solver_cls.config_model().model_fields
        hidden = {
            name
            for name, info in fields.items()
            if (meta := field_constraints(info).get("param_meta")) and not meta.configurable
        }
        assert {"objective", "gradient", "bounds", "status_callback"} <= hidden
        assert "max_iter" in fields and "max_iter" not in hidden
