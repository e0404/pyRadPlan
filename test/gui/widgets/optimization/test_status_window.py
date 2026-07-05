"""Tests for the live optimization status window."""

import pytest

pytest.importorskip("PySide6")

from pyRadPlan.core import ComputeControl
from pyRadPlan.gui.widgets.optimization import OptimizationStatusWidget


def test_update_from_report_grows_curve(qapp):
    win = OptimizationStatusWidget()
    win.update_from_report({"iteration": 1, "objective": 10.0})
    summary = win.update_from_report({"iteration": 2, "objective": 8.0})

    series = win._metrics["objective"]
    assert series["xs"] == [1, 2]
    assert series["ys"] == [10.0, 8.0]
    # Relative change is derived from the local series (no rel_change in the report).
    assert "iter 2" in summary and "Δf" in summary


def test_update_ignores_unconfigured_keys(qapp):
    win = OptimizationStatusWidget()  # default: only "objective"
    win.update_from_report({"iteration": 1, "objective": 1.0, "constraint_violation": 0.5})
    assert "constraint_violation" not in win._metrics
    assert win._metrics["objective"]["ys"] == [1.0]


def test_configure_extra_metrics(qapp):
    win = OptimizationStatusWidget(
        metrics=[("objective", "Objective"), ("constraint_violation", "Constraint")]
    )
    win.update_from_report({"iteration": 1, "objective": 2.0, "constraint_violation": 0.3})
    assert win._metrics["constraint_violation"]["ys"] == [0.3]
    assert win._metrics["objective"]["ys"] == [2.0]


def test_buttons_drive_bound_control(qapp):
    win = OptimizationStatusWidget()
    control = ComputeControl()
    win.bind_control(control)

    win._btn_pause.setChecked(True)
    assert control.is_paused
    win._btn_pause.setChecked(False)
    assert not control.is_paused

    win._on_stop_clicked()
    assert control.stop_requested
    assert not win._btn_stop.isEnabled()


def test_window_placed_fully_on_screen(qapp):
    win = OptimizationStatusWidget()
    win.show()
    qapp.processEvents()
    try:
        available = win.screen().availableGeometry()
        # The whole window (incl. the Pause/Stop row at the bottom) is visible.
        assert available.contains(win.frameGeometry())
    finally:
        win.close()


def test_finalize_disables_controls(qapp):
    win = OptimizationStatusWidget()
    win.bind_control(ComputeControl())
    win.finalize()
    assert not win._btn_pause.isEnabled()
    assert not win._btn_stop.isEnabled()
