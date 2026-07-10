import pytest

from pyRadPlan.core import ProgressLevel, ProgressReport, StatusReport
from pyRadPlan.gui.widgets.workflow import WorkflowWidget
from pyRadPlan.gui.workspace import WorkspaceManager


def test_workflow_widget_init(qapp):
    w = WorkflowWidget(WorkspaceManager())
    assert w is not None
    assert not w._progress.isVisibleTo(w)  # busy bar hidden until a run starts


def test_busy_toggles_progress_and_buttons(qapp):
    w = WorkflowWidget(WorkspaceManager())
    events = []
    w.busy_changed.connect(events.append)

    w._set_busy(True)
    assert events == [True]
    assert w._progress.isVisibleTo(w)
    assert all(not btn.isEnabled() for btn in w._action_buttons)

    w._set_busy(False)
    assert events == [True, False]
    assert not w._progress.isVisibleTo(w)


def test_busy_text_shown_in_status(qapp):
    ws = WorkspaceManager()
    w = WorkflowWidget(ws)

    # A no-op long task: capture that busy_text lands in the status label.
    def _noop():
        return None

    w._run_in_thread(_noop, busy_text="Working…")
    assert w._lbl_status.text() == "Working…"
    # Let the worker thread finish and clean up.
    if w._thread is not None:
        w._thread.wait(2000)
    qapp.processEvents()


def test_progress_report_drives_combined_nested_bar(qapp):
    w = WorkflowWidget(WorkspaceManager())

    report = ProgressReport(
        levels=(
            ProgressLevel("Beam", 0, 2),
            ProgressLevel("Ray", 50, 100),
        )
    )
    w._on_compute_report(report)

    # Bar shows combined nested progress: 0/2 + (1/2)*(50/100) = 0.25.
    assert w._progress.maximum() == w._PROGRESS_STEPS
    assert w._progress.value() == round(0.25 * w._PROGRESS_STEPS)
    assert w._lbl_status.text() == "Beam 0/2 · Ray 50/100"


def test_nested_fraction_advances_with_outer_level(qapp):
    w = WorkflowWidget(WorkspaceManager())
    levels = (ProgressLevel("Beam", 1, 2), ProgressLevel("Ray", 50, 100))
    assert w._nested_fraction(levels) == pytest.approx(0.75)
    # Outermost indeterminate -> None (bar pulses).
    assert w._nested_fraction((ProgressLevel("Setup", 0, None),)) is None


def test_status_report_drives_busy_bar_and_status(qapp):
    w = WorkflowWidget(WorkspaceManager())
    w._progress.setRange(0, 10)
    w._progress.setValue(3)

    w._on_compute_report(StatusReport(message="iter 1", data={"iteration": 1, "objective": 1.2}))

    # An optimization status report drives the indeterminate (pulsing) busy bar...
    assert w._progress.minimum() == 0 and w._progress.maximum() == 0
    # ...and surfaces the per-iteration summary in the status line.
    assert "Optimizing" in w._lbl_status.text()
    assert "iter 1" in w._lbl_status.text()


def test_indeterminate_progress_sets_busy_range(qapp):
    w = WorkflowWidget(WorkspaceManager())
    w._on_compute_report(ProgressReport(levels=(ProgressLevel("Setup", 0, None),)))
    assert w._progress.minimum() == 0 and w._progress.maximum() == 0  # pulsing


def test_plan_change_marks_downstream_stale(qapp):
    ws = WorkspaceManager()
    w = WorkflowWidget(ws)

    # Loading everything at once leaves the products current, not stale.
    ws.set_many(ct=object(), cst=object(), pln=object(), stf=object(), dij=object(), result={})
    assert not w._dij_stale and not w._result_stale

    # Changing the plan invalidates the dose influence and the result.
    ws.pln = object()
    assert w._dij_stale and w._result_stale
    assert w._indicators["dij"].toolTip().startswith("Outdated")

    # Recomputing the dose influence clears its flag (and the result stays stale).
    ws.set_many(stf=object(), dij=object())
    assert not w._dij_stale and w._result_stale

    # Re-optimizing clears the result flag.
    ws.result = {"w": 1}
    assert not w._result_stale
