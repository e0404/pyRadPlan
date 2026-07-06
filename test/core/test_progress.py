"""Tests for the ProgressReporter mixin and report value objects."""

import threading
import time

from pyRadPlan.core import (
    ComputeControl,
    ProgressReport,
    ProgressReporter,
    StatusReport,
    observe_control,
    observe_reports,
)


class _Algo(ProgressReporter):
    """Minimal algorithm using the mixin; no console bar, no throttling."""

    console_progress = False

    def __init__(self):
        super().__init__()
        self.min_report_interval = 0.0


def _collect(algo):
    reports = []
    algo.add_report_observer(reports.append)
    return reports


def test_single_level_progress():
    algo = _Algo()
    reports = _collect(algo)

    out = list(algo.track(range(3), name="Beam"))
    assert out == [0, 1, 2]

    progress = [r for r in reports if isinstance(r, ProgressReport)]
    # push (0/3), three advances (1,2,3), pop (empty)
    assert progress[0].levels[0].name == "Beam"
    assert progress[0].levels[0].total == 3
    # the last non-empty snapshot reached current == total
    non_empty = [r for r in progress if r.levels]
    assert non_empty[-1].levels[0].current == 3


def test_nested_multi_level_progress():
    algo = _Algo()
    reports = _collect(algo)

    seen_depths = set()
    for _i in algo.track(range(2), name="Beam"):
        for _j in algo.track(range(2), name="Ray"):
            pass
        seen_depths.add(max(len(r.levels) for r in reports if isinstance(r, ProgressReport)))

    # At some point two levels (Beam + Ray) were active simultaneously.
    two_level = [r for r in reports if isinstance(r, ProgressReport) and len(r.levels) == 2]
    assert two_level, "expected nested Beam/Ray progress reports"
    names = {(r.levels[0].name, r.levels[1].name) for r in two_level}
    assert ("Beam", "Ray") in names


def test_status_is_separate_from_progress():
    algo = _Algo()
    reports = _collect(algo)

    algo.report_status(iteration=5, objective=12.3, message="iter 5")

    statuses = [r for r in reports if isinstance(r, StatusReport)]
    assert len(statuses) == 1
    assert statuses[0].data["objective"] == 12.3
    assert statuses[0].data["iteration"] == 5
    # A status report carries no progress levels.
    assert not any(isinstance(r, ProgressReport) for r in reports)


def test_context_scoped_observer():
    algo = _Algo()  # no observer registered on the instance
    reports = []

    with observe_reports(reports.append):
        list(algo.track(range(2), name="Beam"))

    assert any(isinstance(r, ProgressReport) for r in reports)

    # Outside the context, new reports are not collected.
    reports.clear()
    list(algo.track(range(2), name="Beam"))
    assert reports == []


def test_observer_exception_does_not_break_iteration():
    algo = _Algo()

    def _bad(_report):
        raise ValueError("boom")

    algo.add_report_observer(_bad)
    # Must complete despite the failing observer.
    assert list(algo.track(range(3), name="Beam")) == [0, 1, 2]


def test_cancel_check():
    algo = _Algo()
    assert not algo.should_cancel()
    algo.set_cancel_check(lambda: True)
    assert algo.should_cancel()


def test_compute_control_stop_unblocks_and_flags():
    control = ComputeControl()
    assert not control.stop_requested
    assert not control.is_paused

    control.pause()
    assert control.is_paused

    # request_stop must clear stop flag state and release a paused wait.
    control.request_stop()
    assert control.stop_requested
    assert not control.is_paused  # stop takes precedence over pause
    # wait_if_paused returns immediately once stopped.
    control.wait_if_paused(timeout=1.0)


def test_compute_control_pause_blocks_then_resumes():
    control = ComputeControl()
    control.pause()

    released = threading.Event()

    def _worker():
        control.wait_if_paused(timeout=2.0)
        released.set()

    t = threading.Thread(target=_worker)
    t.start()
    # Still paused: the worker should be blocked.
    assert not released.wait(timeout=0.2)
    control.resume()
    assert released.wait(timeout=2.0)
    t.join()


def test_checkpoint_honours_active_control():
    algo = _Algo()
    control = ComputeControl()

    with observe_control(control):
        assert algo.checkpoint() is True  # running
        control.request_stop()
        assert algo.checkpoint() is False  # stop requested
        assert algo.should_cancel() is True

    # Outside the context the control no longer affects the reporter.
    assert algo.should_cancel() is False
    assert algo.checkpoint() is True


def test_checkpoint_blocks_while_paused():
    algo = _Algo()
    control = ComputeControl()
    control.pause()

    returned = []

    def _worker():
        with observe_control(control):
            returned.append(algo.checkpoint())

    t = threading.Thread(target=_worker)
    t.start()
    time.sleep(0.1)
    assert returned == []  # blocked in checkpoint while paused
    control.resume()
    t.join(timeout=2.0)
    assert returned == [True]


def test_observe_control_none_is_noop():
    algo = _Algo()
    with observe_control(None):
        assert algo.checkpoint() is True


def test_progress_level_fraction():
    algo = _Algo()
    reports = _collect(algo)
    with algo.progress("Phase", total=4) as handle:
        handle.update(2)
    prog = [r for r in reports if isinstance(r, ProgressReport) and r.levels]
    mid = [r for r in prog if r.levels[0].current == 2][0]
    assert mid.levels[0].fraction == 0.5
