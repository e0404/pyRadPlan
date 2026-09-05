"""Tests for the ``opt_info`` reporting channel of ``fluence_optimization``."""

import logging

import numpy as np
import pytest

from pyRadPlan import fluence_optimization
from pyRadPlan.optimization.solvers import get_available_solvers

SOLVERS = sorted(get_available_solvers())


@pytest.fixture
def optimized(request, small_proton_case):
    """Run a full optimization with ``opt_info`` for the solver named by the test parameter."""
    pln, ct, cst, stf, dij = small_proton_case
    pln.prop_opt = {"solver": request.param, "display": False, "max_iter": 25}

    opt_info = {}
    w = fluence_optimization(ct, cst, stf, dij, pln, opt_info=opt_info)
    return w, opt_info


@pytest.mark.parametrize("optimized", SOLVERS, indirect=True)
def test_opt_info_is_populated(optimized):
    """Every registered solver fills all documented keys of the OptInfo contract."""
    _, opt_info = optimized

    assert set(opt_info) == {"obj_history", "num_iter", "result_info"}


@pytest.mark.parametrize("optimized", SOLVERS, indirect=True)
def test_result_info_is_a_dict(optimized):
    """Solvers report result information as a dictionary, not a bare status code."""
    _, opt_info = optimized

    assert isinstance(opt_info["result_info"], dict)


@pytest.mark.parametrize("optimized", SOLVERS, indirect=True)
def test_num_iter_comes_from_the_solver(optimized):
    """The iteration count is taken from the solver's own normalized "num_iter" key."""
    _, opt_info = optimized

    assert isinstance(opt_info["num_iter"], int)
    assert opt_info["num_iter"] >= 0
    assert opt_info["num_iter"] == opt_info["result_info"]["num_iter"]


@pytest.mark.parametrize("optimized", SOLVERS, indirect=True)
def test_obj_history_records_every_evaluation(optimized):
    """The history is filled by the problem's objective, so it holds one value per evaluation."""
    _, opt_info = optimized

    history = opt_info["obj_history"]
    assert len(history) > 0
    assert all(isinstance(value, float) for value in history)
    # Line searches evaluate the objective more often than they take iterations.
    assert len(history) >= opt_info["num_iter"]


def test_no_opt_info_leaves_history_disabled(small_proton_case):
    """Without opt_info the objective history stays off, so no values are accumulated."""
    pln, ct, cst, stf, dij = small_proton_case
    pln.prop_opt = {"solver": SOLVERS[0], "display": False, "max_iter": 2}

    w = fluence_optimization(ct, cst, stf, dij, pln)

    assert w.shape == (dij.total_num_of_bixels,)


def test_opt_info_is_keyword_only(small_proton_case):
    """opt_info is an explicit keyword argument, so a misspelling cannot pass silently."""
    pln, ct, cst, stf, dij = small_proton_case
    pln.prop_opt = {"solver": SOLVERS[0], "display": False, "max_iter": 2}

    with pytest.raises(TypeError):
        fluence_optimization(ct, cst, stf, dij, pln, optinfo={})


class _NonRecordingProblem:
    """Stand-in for a planning problem that does not fill an objective history."""

    short_name = "non_recording"
    records_obj_history = False
    obj_history = None

    def solve(self, _ct, _cst, _stf, dij):
        return np.zeros(dij.total_num_of_bixels), {"num_iter": 3}


def test_history_is_absent_rather_than_empty_when_unsupported(
    small_proton_case, monkeypatch, caplog
):
    """Regression: an unsupported history was reported as [], i.e. "zero evaluations".

    The empty list is indistinguishable from a solve that never evaluated the objective, so a
    problem that does not record leaves the key unset - the way num_iter already did.
    """
    pln, ct, cst, stf, dij = small_proton_case
    monkeypatch.setattr(
        "pyRadPlan.optimization._fluence_optimization.get_problem_from_pln",
        lambda _pln: _NonRecordingProblem(),
    )

    opt_info = {}
    with caplog.at_level(logging.WARNING, logger="pyRadPlan.optimization._fluence_optimization"):
        fluence_optimization(ct, cst, stf, dij, pln, opt_info=opt_info)

    assert "obj_history" not in opt_info
    assert opt_info["num_iter"] == 3
    assert any("non_recording" in m and "obj_history" in m for m in caplog.messages)


def test_reused_opt_info_does_not_keep_values_from_an_earlier_solve(
    small_proton_case, monkeypatch
):
    """Regression: keys this solve does not fill were left holding the previous solve's values.

    The dictionary belongs to the caller, so reusing it must report this run or nothing.
    """
    pln, ct, cst, stf, dij = small_proton_case
    pln.prop_opt = {"solver": SOLVERS[0], "display": False, "max_iter": 2}

    opt_info = {}
    fluence_optimization(ct, cst, stf, dij, pln, opt_info=opt_info)
    assert opt_info["obj_history"], "the first solve must actually populate the history"

    monkeypatch.setattr(
        "pyRadPlan.optimization._fluence_optimization.get_problem_from_pln",
        lambda _pln: _NonRecordingProblem(),
    )
    fluence_optimization(ct, cst, stf, dij, pln, opt_info=opt_info)

    assert "obj_history" not in opt_info, "the first solve's history was reported as this one's"


def test_reused_opt_info_is_cleared_when_a_solve_raises(small_proton_case, monkeypatch):
    """A failed solve must not leave the previous run's numbers behind either."""
    pln, ct, cst, stf, dij = small_proton_case
    pln.prop_opt = {"solver": SOLVERS[0], "display": False, "max_iter": 2}

    opt_info = {}
    fluence_optimization(ct, cst, stf, dij, pln, opt_info=opt_info)
    assert set(opt_info) == {"obj_history", "num_iter", "result_info"}

    class _FailingProblem(_NonRecordingProblem):
        def solve(self, *_args):
            raise RuntimeError("solver blew up")

    monkeypatch.setattr(
        "pyRadPlan.optimization._fluence_optimization.get_problem_from_pln",
        lambda _pln: _FailingProblem(),
    )
    with pytest.raises(RuntimeError, match="solver blew up"):
        fluence_optimization(ct, cst, stf, dij, pln, opt_info=opt_info)

    assert opt_info == {}
