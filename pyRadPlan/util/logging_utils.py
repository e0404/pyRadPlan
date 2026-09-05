"""Logging helpers for pyRadPlan."""

import logging
import os
import sys
import threading
import warnings
from contextlib import contextmanager
from typing import Optional

logger = logging.getLogger(__name__)


class _ActiveCount:
    """Thread-safe count of active captures, entered once per capture."""

    def __init__(self):
        self._count = 0
        self._lock = threading.Lock()

    @property
    def active(self) -> bool:
        return self._count > 0

    def __enter__(self):
        with self._lock:
            self._count += 1

    def __exit__(self, *_exc):
        with self._lock:
            self._count -= 1


_warning_captures = _ActiveCount()

# Descriptors currently redirected by native_output_to_logger. A second capture of the same
# descriptor would save the first one's pipe as "the original" and restore that dead pipe on
# exit, so it is refused instead.
_captured_fds: set[int] = set()
_captured_fds_lock = threading.Lock()


@contextmanager
def warnings_to_logger(name: str, level: int = logging.WARNING):
    """Route ``warnings.warn`` calls through a logger, tagged with ``name``.

    Parameters
    ----------
    name : str
        Tag prepended to each warning message in the log record.
    level : int, optional
        Logging level for emitted records, by default ``logging.WARNING``.

    Notes
    -----
    The warnings module is process-global, so this redirects warnings raised by *any* thread
    for the duration of the block, not only the calling one.

    Before Python 3.14 ``warnings.catch_warnings`` is not thread-safe either: another thread
    entering its own ``catch_warnings`` during this block and leaving it afterwards restores
    this block's hook as if it were the original, permanently. The hook therefore checks
    whether a capture is still active and otherwise passes the warning on to the function it
    replaced, so a leaked hook degrades to the behaviour that was there before.
    """
    original = warnings.showwarning

    def _hook(message, category, *args, **kwargs):
        if _warning_captures.active:
            logger.log(level, "%s: %s: %s", name, category.__name__, message)
        else:
            original(message, category, *args, **kwargs)

    with warnings.catch_warnings():
        # Appended, so it only catches warnings that would otherwise fall through to the
        # default once-per-location filter; ignore filters set by the user or by the stdlib
        # defaults keep applying.
        warnings.simplefilter("always", append=True)
        warnings.showwarning = _hook
        with _warning_captures:
            yield


@contextmanager
def native_output_to_logger(
    name: str,
    level: int = logging.INFO,
    target: Optional[logging.Logger] = None,
    fd: int = 1,
):
    """Route output written to an OS file descriptor through a logger, tagged with ``name``.

    Extension libraries write to file descriptor 1 directly instead of going through
    ``sys.stdout``, so their output cannot be captured by replacing ``sys.stdout`` and is lost
    wherever the process' descriptor is not displayed - most notably in Jupyter. This
    redirects the descriptor into a pipe and logs whatever arrives on it, one record per line.

    Parameters
    ----------
    name : str
        Tag prepended to each captured line in the log record.
    level : int, optional
        Logging level for emitted records, by default ``logging.INFO``.
    target : logging.Logger, optional
        Logger receiving the records. Defaults to this module's logger.
    fd : int, optional
        File descriptor to capture, by default ``1`` (standard output).

    Notes
    -----
    File descriptors are process-global, so this captures everything written to *fd* while the
    block is active, including output produced by other threads.

    The captured text is drained by a background thread. This is not an optimization: a pipe
    holds a limited amount of data, and a writer producing more than that blocks forever if
    nobody reads the other end.
    """
    sink = target if target is not None else logger

    if _logging_writes_to_fd(sink, fd):
        # Capturing here would feed the sink's own records straight back into the pipe the
        # pump is reading, which never terminates. Leaving the output alone is the safe answer.
        sink.warning(
            "Not capturing descriptor %d for '%s': a handler of logger '%s' writes to that "
            "same descriptor, which would loop. Log to stderr instead to capture it.",
            fd,
            name,
            sink.name,
        )
        yield
        return

    with _captured_fds_lock:
        already_captured = fd in _captured_fds
        if not already_captured:
            _captured_fds.add(fd)

    if already_captured:
        sink.warning(
            "Not capturing descriptor %d for '%s': it is already being captured. A second "
            "capture of the same descriptor would leave it pointing at a dead pipe afterwards.",
            fd,
            name,
        )
        yield
        return

    def _make_pump(read_fd: int) -> threading.Thread:
        def _pump():
            with os.fdopen(read_fd, "r", errors="replace") as pipe:
                for line in pipe:
                    text = line.rstrip("\r\n")
                    if text.strip():
                        sink.log(level, "%s: %s", name, text)

        return threading.Thread(target=_pump, name=f"native-output-{name}", daemon=True)

    try:
        try:
            saved_fd, pump = _redirect_fd_to_pump(fd, _make_pump)
        except OSError:
            # No descriptor to capture, or no pipe to be had (e.g. a fully detached process
            # or an exhausted descriptor table); leave the output where it is.
            yield
            return

        try:
            yield
        finally:
            _flush_python_buffer(fd)
            # Restoring the descriptor closes the pipe's write end, which ends the pump's loop.
            os.dup2(saved_fd, fd)
            os.close(saved_fd)
            pump.join(timeout=5.0)
    finally:
        with _captured_fds_lock:
            _captured_fds.discard(fd)


def _redirect_fd_to_pump(fd: int, make_pump) -> tuple[int, threading.Thread]:
    """Point *fd* at a fresh pipe and start a thread draining it into the log.

    Returns the saved original descriptor, which the caller restores to end the capture, and
    the running pump.

    A setup that fails part-way must not leave *fd* pointing at a pipe with no reader, where
    the next write to it would block once the pipe fills, nor keep the descriptors it opened.
    Everything is therefore handed back here before the failure propagates, leaving the caller
    only a successful setup to undo.
    """
    saved_fd = os.dup(fd)
    read_fd = write_fd = None
    redirected = False

    try:
        read_fd, write_fd = os.pipe()
        pump = make_pump(read_fd)

        _flush_python_buffer(fd)
        os.dup2(write_fd, fd)
        redirected = True
        os.close(write_fd)
        write_fd = None

        pump.start()
    except BaseException:
        if redirected:
            os.dup2(saved_fd, fd)
        if write_fd is not None:
            os.close(write_fd)
        if read_fd is not None:
            # The pump never ran, so nothing else will close its end of the pipe.
            os.close(read_fd)
        os.close(saved_fd)
        raise

    return saved_fd, pump


def _flush_python_buffer(fd: int) -> None:
    """Flush the Python-level stream wrapping *fd*, so buffered text is not misrouted."""
    stream = sys.stdout if fd == 1 else sys.stderr if fd == 2 else None
    if stream is not None:
        try:
            stream.flush()
        except (ValueError, OSError):
            pass


def _logging_writes_to_fd(sink: logging.Logger, fd: int) -> bool:
    """Report whether any handler reachable from *sink* writes to file descriptor *fd*.

    With no handler configured anywhere, logging falls back to ``logging.lastResort`` (stderr),
    which is checked as well so that capturing descriptor 2 is refused in that setup.
    """
    handlers = []
    current = sink
    while current is not None:
        handlers.extend(current.handlers)
        current = current.parent if current.propagate else None

    if not handlers and logging.lastResort is not None:
        handlers.append(logging.lastResort)

    for handler in handlers:
        stream = getattr(handler, "stream", None)
        if stream is None:
            continue
        try:
            if stream.fileno() == fd:
                return True
        except (OSError, ValueError, AttributeError):
            # A stream without a usable descriptor cannot be the one we are capturing.
            continue
    return False
