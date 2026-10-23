import pytest

pytest.importorskip("PySide6")

from pyRadPlan.gui.widgets.plan._optimization_dialog import OptimizationConfigDialog
from pyRadPlan.optimization.problems import NonLinearFluencePlanningProblem
from pyRadPlan.optimization.solvers import get_available_solvers, get_solver


def _dialog(initial=None):
    return OptimizationConfigDialog(NonLinearFluencePlanningProblem, initial=initial)


def test_untouched_dialog_keeps_prop_opt_empty(qapp):
    dialog = _dialog()
    assert dialog.values() == {}
    assert "solver" not in dialog.problem_form._editors
    assert "convert_dose_objectives" not in dialog.problem_form._editors
    assert "apply_overlap" in dialog.problem_form._editors


def test_solver_form_shows_the_solver_defaults(qapp):
    dialog = _dialog({"solver": "scipy"})
    assert dialog.solver_name() == "scipy"
    assert dialog.solver_form._editors["max_iter"].value() == get_solver("scipy").max_iter
    assert "objective" not in dialog.solver_form._editors
    assert dialog.values() == {"solver": "scipy"}


def test_selecting_a_solver_writes_it(qapp):
    dialog = _dialog()
    other = next(name for name in get_available_solvers() if name != dialog.solver_name())
    dialog.select_solver(other)
    assert dialog.values() == {"solver": other}


def test_solver_edits_and_unshown_entries_are_kept(qapp):
    dialog = _dialog(
        {"solver": {"name": "scipy", "max_iter": 17}, "convert_dose_objectives": False}
    )
    dialog.solver_form._set_value("method", "TNC")
    assert dialog.values() == {
        "convert_dose_objectives": False,
        "solver": {"name": "scipy", "max_iter": 17, "method": "TNC"},
    }


def test_problem_level_solver_knobs_move_into_the_solver(qapp):
    dialog = _dialog({"solver": {"name": "scipy", "max_iter": 17}, "max_iter": 40})
    assert dialog.solver_form._editors["max_iter"].value() == 40
    assert dialog.values() == {"solver": {"name": "scipy", "max_iter": 40}}


def test_switching_solvers_keeps_each_solvers_edits(qapp):
    if "ipax" not in get_available_solvers():
        pytest.skip("ipax not available")
    dialog = _dialog({"solver": "scipy"})
    dialog.solver_form._set_value("method", "TNC")
    dialog.select_solver("ipax")
    dialog.solver_form._set_nested("options", ("hessian",), "exact")
    assert dialog.values() == {"solver": {"name": "ipax", "options": {"hessian": "exact"}}}
    dialog.select_solver("scipy")
    assert dialog.values() == {"solver": {"name": "scipy", "method": "TNC"}}

    dialog.select_solver("ipax")
    solver = get_solver(dialog.values()["solver"])
    assert solver.options.hessian == "exact"


def test_ipax_options_hide_the_entries_the_solver_sets(qapp):
    if "ipax" not in get_available_solvers():
        pytest.skip("ipax not available")
    dialog = _dialog({"solver": "ipax"})
    editors = dialog.solver_form._editors
    assert "options.hessian" in editors
    assert "options.max_iter" not in editors
    assert "abs_obj_tol" not in editors


def test_unavailable_solver_is_kept(qapp):
    spec = {"name": "not_a_solver", "max_iter": 3}
    dialog = _dialog({"solver": spec})
    assert dialog.solver_form is None
    assert dialog.values() == {"solver": spec}


@pytest.mark.parametrize("name", ["scipy", "ipax"])
def test_solver_instance_keeps_hidden_configuration(qapp, name):
    if name not in get_available_solvers():
        pytest.skip(f"{name} not available")
    solver = get_solver(name)
    solver.allow_keyboard_cancel = False
    solver.cancel_key = "x"
    dialog = _dialog({"solver": solver})
    dialog.solver_form._set_value("max_iter", 17)
    rebuilt = get_solver(dialog.values()["solver"])
    assert rebuilt.allow_keyboard_cancel is False
    assert rebuilt.cancel_key == "x"
    assert rebuilt.max_iter == 17
    assert solver.max_iter != 17
