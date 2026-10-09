"""Tests for the logging helpers in pyRadPlan.util."""

import logging
import os
import subprocess
import sys
import threading
import warnings

import pytest

from pyRadPlan.util import native_output_to_logger, warnings_to_logger
from pyRadPlan.util import logging_utils
from pyRadPlan.util.logging_utils import _logging_writes_to_fd

# A child process inherits the descriptor, which is how an external tool's output reaches us.
_CHILD_WRITER = "import sys; print('from the child'); sys.stdout.flush()"


def test_native_output_is_logged_and_kept_off_the_descriptor(caplog):
    """Output written to the descriptor is logged instead of reaching standard output."""
    saved_fd = os.dup(1)
    read_fd, write_fd = os.pipe()
    os.dup2(write_fd, 1)
    os.close(write_fd)
    try:
        with caplog.at_level(logging.INFO, logger="pyRadPlan.util.logging_utils"):
            with native_output_to_logger("tag"):
                os.write(1, b"hello from C\n")
    finally:
        os.dup2(saved_fd, 1)
        os.close(saved_fd)

    leaked = os.read(read_fd, 4096).decode(errors="replace")
    os.close(read_fd)

    assert "hello from C" not in leaked
    assert any("tag: hello from C" in message for message in caplog.messages)


def test_descriptor_is_restored_after_the_block():
    """The descriptor is handed back, including when the block raises."""
    saved_fd = os.dup(1)
    read_fd, write_fd = os.pipe()
    os.dup2(write_fd, 1)
    os.close(write_fd)
    try:
        try:
            with native_output_to_logger("tag"):
                raise RuntimeError("boom")
        except RuntimeError:
            pass
        os.write(1, b"after\n")
    finally:
        os.dup2(saved_fd, 1)
        os.close(saved_fd)

    restored = os.read(read_fd, 4096).decode(errors="replace")
    os.close(read_fd)

    assert "after" in restored


def test_captures_output_of_a_subprocess_inheriting_the_descriptor(caplog):
    """The capture works at the OS level, so an inherited descriptor is captured too."""
    saved_fd = os.dup(1)
    read_fd, write_fd = os.pipe()
    os.dup2(write_fd, 1)
    os.close(write_fd)
    try:
        with caplog.at_level(logging.INFO, logger="pyRadPlan.util.logging_utils"):
            with native_output_to_logger("child"):
                subprocess.run([sys.executable, "-c", _CHILD_WRITER], check=True)
    finally:
        os.dup2(saved_fd, 1)
        os.close(saved_fd)
    os.close(read_fd)

    assert any("child: from the child" in message for message in caplog.messages)


def test_repeated_warnings_are_not_swallowed(caplog):
    """The "always" filter keeps repeats of the same warning reaching the logger."""
    with caplog.at_level(logging.WARNING, logger="pyRadPlan.util.logging_utils"):
        with warnings_to_logger("solver"):
            for _ in range(3):
                warnings.warn("repeated", UserWarning)

    assert sum("solver: UserWarning: repeated" in message for message in caplog.messages) == 3


class _StreamOnFd:
    """A stream that claims a given descriptor, as a StreamHandler on stdout would."""

    def __init__(self, fd):
        self._fd = fd

    def fileno(self):
        return self._fd

    def write(self, text):
        pass

    def flush(self):
        pass


def _logger_writing_to(fd, name):
    sink = logging.getLogger(name)
    sink.setLevel(logging.INFO)
    sink.propagate = False
    sink.handlers = [logging.StreamHandler(_StreamOnFd(fd))]
    return sink


def test_capture_is_refused_when_the_logger_writes_to_the_same_descriptor():
    """Capturing fd 1 into a logger that writes to fd 1 would loop, so it is refused.

    The redirect is skipped entirely, so output written inside the block still reaches the
    original descriptor rather than a pipe nobody can drain.
    """
    sink = _logger_writing_to(1, "test_loop_guard")

    saved_fd = os.dup(1)
    read_fd, write_fd = os.pipe()
    os.dup2(write_fd, 1)
    os.close(write_fd)
    try:
        with native_output_to_logger("looping", target=sink):
            # If the guard had not fired, this would go into an inner pipe instead.
            os.write(1, b"payload\n")
    finally:
        os.dup2(saved_fd, 1)
        os.close(saved_fd)

    passed_through = os.read(read_fd, 1 << 16)
    os.close(read_fd)

    assert b"payload" in passed_through


def test_capture_proceeds_when_the_logger_writes_elsewhere(caplog):
    """A logger writing to a different descriptor is safe, so capture still happens."""
    sink = _logger_writing_to(2, "test_loop_guard_stderr")
    records = []
    sink.handlers.append(logging.Handler())
    sink.handlers[-1].emit = lambda record: records.append(record.getMessage())

    saved_fd = os.dup(1)
    read_fd, write_fd = os.pipe()
    os.dup2(write_fd, 1)
    os.close(write_fd)
    try:
        with native_output_to_logger("safe", target=sink):
            os.write(1, b"captured\n")
    finally:
        os.dup2(saved_fd, 1)
        os.close(saved_fd)

    leaked = os.read(read_fd, 1 << 16)
    os.close(read_fd)

    assert b"captured" not in leaked
    assert any("safe: captured" in message for message in records)


def test_user_ignore_filters_keep_applying(caplog):
    """The catch-all filter is appended, so a deliberate ignore set by the user still wins."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="deliberately silenced")
        with caplog.at_level(logging.WARNING, logger="pyRadPlan.util.logging_utils"):
            with warnings_to_logger("solver"):
                warnings.warn("deliberately silenced", UserWarning)

    assert not any("deliberately silenced" in message for message in caplog.messages)


def test_leaked_hook_passes_warnings_through():
    """A leaked hook defers to the function it replaced instead of swallowing warnings.

    Another thread's catch_warnings, entered during the block and left after it, restores
    this block's hook as if it were the original. With no capture active it must pass through.
    """
    received = []
    stand_in_original = lambda message, *args, **kwargs: received.append(str(message))  # noqa: E731

    solver_in, main_in, solver_out = threading.Event(), threading.Event(), threading.Event()

    def solver_thread():
        with warnings_to_logger("solver"):
            solver_in.set()
            main_in.wait()
        solver_out.set()

    with warnings.catch_warnings():
        warnings.showwarning = stand_in_original
        thread = threading.Thread(target=solver_thread)
        thread.start()
        solver_in.wait()
        with warnings.catch_warnings():  # entered during the solve, left after it: a straddle
            main_in.set()
            solver_out.wait()
        thread.join()

        leaked = "warnings_to_logger" in getattr(warnings.showwarning, "__qualname__", "")
        warnings.simplefilter("always")
        warnings.warn("after the straddle", UserWarning)

    assert leaked, "the straddle should have restored the solver's hook (the hazard under test)"
    assert any("after the straddle" in message for message in received)


def test_second_capture_of_the_same_descriptor_is_refused(caplog):
    """Overlapping captures would restore a dead pipe; the inner one declines instead."""
    saved_fd = os.dup(1)
    read_fd, write_fd = os.pipe()
    os.dup2(write_fd, 1)
    os.close(write_fd)
    collected = []
    reader = threading.Thread(target=lambda: collected.append(os.read(read_fd, 1 << 16)))
    reader.start()

    a_in, b_in, a_out = threading.Event(), threading.Event(), threading.Event()

    def outer():
        with native_output_to_logger("outer"):
            a_in.set()
            b_in.wait()
        a_out.set()

    def inner():
        a_in.wait()
        with native_output_to_logger("inner"):
            b_in.set()
            a_out.wait()

    try:
        with caplog.at_level(logging.WARNING, logger="pyRadPlan.util.logging_utils"):
            threads = [threading.Thread(target=outer), threading.Thread(target=inner)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        os.write(1, b"still-alive\n")
    finally:
        os.dup2(saved_fd, 1)
        os.close(saved_fd)
        reader.join(timeout=5.0)

    assert any("already being captured" in message for message in caplog.messages)
    assert b"still-alive" in b"".join(collected)


def test_last_resort_handler_counts_when_nothing_is_configured(monkeypatch):
    """With no handler anywhere, logging falls back to lastResort, which must be consulted.

    pytest swaps stderr for a capture object, so lastResort is replaced by a stand-in that
    claims descriptor 2 the way the real one does in a plain process.
    """
    monkeypatch.setattr(logging, "lastResort", logging.StreamHandler(_StreamOnFd(2)))
    bare = logging.getLogger("test.logging_utils.unconfigured")
    root = logging.getLogger()
    saved_handlers = root.handlers[:]
    root.handlers.clear()
    try:
        assert _logging_writes_to_fd(bare, 2)
        assert not _logging_writes_to_fd(bare, 1)
    finally:
        root.handlers.extend(saved_handlers)


class _ThreadingWithUnstartableThread:
    """Stands in for the threading module inside logging_utils, with a Thread that won't start.

    Scoped to that module rather than patching ``threading`` itself, which would hand every
    other thread in the process the same broken Thread for the duration of the test.
    """

    class Thread:
        def __init__(self, *_args, **_kwargs):
            pass

        def start(self):
            raise RuntimeError("cannot start a new thread")

    def __getattr__(self, name):
        return getattr(threading, name)


def _scratch_descriptor(tmp_path, name):
    """A real writable descriptor standing in for stdout.

    The failure under test leaves a descriptor pointing at a pipe nobody drains. Probing that
    on the process' own stdout wedges the test run instead of failing it, so the checks below
    run against a descriptor of their own.
    """
    path = tmp_path / name
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_BINARY", 0)
    return os.open(path, flags), path


def test_descriptor_is_restored_when_the_pump_cannot_start(monkeypatch, tmp_path):
    """Regression: a failed setup left the descriptor pointing at a pipe nobody would drain.

    Writes to it then went into that pipe (and blocked once it filled) instead of reaching the
    original target, so the descriptor is handed back before the failure propagates.
    """
    monkeypatch.setattr(logging_utils, "threading", _ThreadingWithUnstartableThread())
    fd, path = _scratch_descriptor(tmp_path, "captured.txt")

    try:
        with pytest.raises(RuntimeError, match="cannot start a new thread"):
            with native_output_to_logger("doomed", fd=fd):
                pass

        os.write(fd, b"still-usable\n")
    finally:
        os.close(fd)

    assert path.read_bytes() == b"still-usable\n", (
        "the write did not reach the original target, so the descriptor was left redirected"
    )
    assert fd not in logging_utils._captured_fds, "the failed capture must release its claim"


def test_a_failed_setup_leaks_no_descriptors(monkeypatch, tmp_path):
    """Regression: a failed setup leaked the saved descriptor and the pipe's read end.

    Descriptor numbers are handed out lowest-first, so a leak shows up as the number a probing
    dup() returns creeping upwards across repeated attempts.
    """
    monkeypatch.setattr(logging_utils, "threading", _ThreadingWithUnstartableThread())
    fd, _path = _scratch_descriptor(tmp_path, "leak.txt")

    def _probe():
        probe_fd = os.dup(fd)
        os.close(probe_fd)
        return probe_fd

    def _attempt():
        with pytest.raises(RuntimeError):
            with native_output_to_logger("doomed", fd=fd):
                pass

    try:
        for _ in range(3):  # settle any one-off allocations before measuring
            _attempt()

        before = _probe()
        for _ in range(20):
            _attempt()

        assert _probe() == before, "descriptor numbers crept up, so the failed setup leaked"
    finally:
        os.close(fd)
