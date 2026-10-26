"""Tests for the routing of IPOPT's native (C-level) output."""

import logging
import os

import numpy as np
import pytest

from pyRadPlan.optimization.solvers import OptimizerIpopt, get_solver

pytestmark = pytest.mark.skipif(OptimizerIpopt is None, reason="IPOPT is not installed.")

IPOPT_TABLE_HEADER = "iter    objective"


def _solve_capturing_fd(**config):
    """Solve a trivial problem, returning whatever IPOPT wrote to the real stdout descriptor."""
    solver = get_solver({"name": "ipopt", "max_iter": 3, **config})
    solver.options["print_timing_statistics"] = "no"
    solver.objective = lambda x: np.sum(x**2)
    solver.gradient = lambda x: 2.0 * x

    saved_fd = os.dup(1)
    read_fd, write_fd = os.pipe()
    os.dup2(write_fd, 1)
    os.close(write_fd)
    try:
        solver.solve(np.asarray([5.0, -3.0]))
    finally:
        os.dup2(saved_fd, 1)
        os.close(saved_fd)

    native = os.read(read_fd, 1 << 20).decode(errors="replace")
    os.close(read_fd)
    return native


def test_native_mode_writes_to_the_descriptor():
    """ "native" leaves IPOPT printing to standard output, as the library does by default."""
    assert IPOPT_TABLE_HEADER in _solve_capturing_fd(output_mode="native")


def test_logging_mode_captures_the_full_table(caplog):
    """ "logging" moves the output off the descriptor and into the logger, in full."""
    with caplog.at_level(logging.INFO, logger="pyRadPlan.optimization.solvers._ipopt"):
        native = _solve_capturing_fd(output_mode="logging")

    assert IPOPT_TABLE_HEADER not in native

    logged = "\n".join(caplog.messages)
    assert IPOPT_TABLE_HEADER in logged
    # The convergence summary is only available from the native output, not the callback.
    assert "Number of Iterations" in logged
    assert "EXIT:" in logged


def test_display_false_silences_both_channels(caplog):
    """A disabled display produces no output at all, whichever route is configured."""
    with caplog.at_level(logging.INFO, logger="pyRadPlan.optimization.solvers._ipopt"):
        native = _solve_capturing_fd(output_mode="logging", display=False)

    assert native.strip() == ""
    assert IPOPT_TABLE_HEADER not in "\n".join(caplog.messages)


def test_unknown_output_mode_raises():
    """A misspelled mode fails loudly instead of silently picking a route."""
    with pytest.raises(ValueError, match="Unknown output_mode"):
        _solve_capturing_fd(output_mode="verbose")


def test_iteration_count_survives_capture():
    """Capturing the output does not disturb the callback-derived iteration count."""
    solver = get_solver({"name": "ipopt", "max_iter": 3, "output_mode": "logging"})
    solver.objective = lambda x: np.sum(x**2)
    solver.gradient = lambda x: 2.0 * x

    _, result_info = solver.solve(np.asarray([5.0, -3.0]))

    assert result_info["num_iter"] == 3


def test_display_toggle_does_not_stick():
    """Silencing one solve must not leave print_level at 0 for the next."""
    solver = get_solver({"name": "ipopt", "max_iter": 2, "output_mode": "native"})
    solver.objective = lambda x: np.sum(x**2)
    solver.gradient = lambda x: 2.0 * x

    solver.display = False
    solver.solve(np.asarray([5.0, -3.0]))
    solver.display = True

    assert solver.options["print_level"] == 5


def test_user_print_level_survives_a_silenced_solve():
    """A print_level chosen via the solver dict is not overwritten by display=False."""
    solver = get_solver(
        {"name": "ipopt", "max_iter": 2, "output_mode": "native", "options": {"print_level": 3}}
    )
    solver.objective = lambda x: np.sum(x**2)
    solver.gradient = lambda x: 2.0 * x

    solver.display = False
    solver.solve(np.asarray([5.0, -3.0]))

    assert solver.options["print_level"] == 3


def test_attribute_backed_options_are_reported_not_applied(caplog):
    """max_iter etc. in `options` are ignored in favour of the attribute, and say so."""
    solver = get_solver(
        {"name": "ipopt", "max_iter": 2, "output_mode": "native", "options": {"max_iter": 100}}
    )
    solver.objective = lambda x: np.sum(x**2)
    solver.gradient = lambda x: 2.0 * x

    with caplog.at_level(logging.WARNING, logger="pyRadPlan.optimization.solvers._ipopt"):
        _, result_info = solver.solve(np.asarray([5.0, -3.0]))

    assert result_info["num_iter"] == 2, "the attribute (2), not options (100), capped the run"
    assert solver.options["max_iter"] == 100, "the stored dict is left as the user wrote it"
    assert any("'max_iter'" in m and "ignored" in m for m in caplog.messages)
