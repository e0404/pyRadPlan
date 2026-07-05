"""Progress and status reporting for pyRadPlan compute algorithms.

This module provides a light-weight, GUI-agnostic observability layer that any
"state-changing" algorithm (dose engines, optimizers, steering generators, …)
can mix in to report what it is doing, without taking a dependency on a console
(``tqdm``) or a GUI (``Qt``).

Two kinds of reports are emitted, kept deliberately separate so that consumers
can react to one without being confused by the other:

* :class:`ProgressReport` -- a *stack* of :class:`ProgressLevel` entries
  describing nested, (optionally) determinate progress (e.g. ``Beam 1/2`` ->
  ``Ray 53/100``).  Suitable for driving progress bars.
* :class:`StatusReport` -- arbitrary key/value data and an optional message
  (e.g. ``{"iteration": 5, "objective": 12.3}``).  Suitable for live metric
  plots.  Algorithms without iterative progress (optimization) can push these
  *without* implying any bar progress.

Algorithms mix in :class:`ProgressReporter` and call :meth:`ProgressReporter.track`
(a drop-in replacement for ``tqdm``) and/or :meth:`ProgressReporter.report_status`.
Consumers either call :meth:`ProgressReporter.add_report_observer` on a concrete
instance, or -- when the instance is created internally (e.g. by
``calc_dose_influence``) -- register a context-scoped observer via
:func:`observe_reports`.
"""

from __future__ import annotations

import logging
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Iterator, Mapping, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

#: A consumer of compute reports.
ReportObserver = Callable[["ComputeReport"], None]

#: Context-scoped observers picked up automatically by every reporter.  Set via
#: :func:`observe_reports`.  Because it is a :class:`~contextvars.ContextVar`,
#: observers registered in a worker thread are visible to algorithms run in that
#: same thread (the GUI pattern), but do not leak across unrelated threads.
_active_observers: ContextVar[tuple[ReportObserver, ...]] = ContextVar(
    "_active_observers", default=()
)

#: Context-scoped cooperative control (pause/stop) picked up by
#: :meth:`ProgressReporter.checkpoint`.  Set via :func:`observe_control`.  Like
#: :data:`_active_observers` this is per-thread, so a control installed in a worker
#: thread reaches the algorithm running in that same thread without leaking elsewhere.
_active_control: ContextVar[Optional["ComputeControl"]] = ContextVar(
    "_active_control", default=None
)


class ComputeCancelledError(RuntimeError):
    """Raised when a cooperative cancellation has been requested."""


# ---------------------------------------------------------------------------
# Report value objects
# ---------------------------------------------------------------------------


class ComputeReport:
    """Marker base class for all reports (use ``isinstance`` to discriminate)."""


@dataclass(frozen=True)
class ProgressLevel:
    """One level of (optionally determinate) progress."""

    name: str
    current: int
    total: Optional[int]

    @property
    def fraction(self) -> Optional[float]:
        """Completed fraction in ``[0, 1]``, or ``None`` when indeterminate."""
        if self.total:
            return min(max(self.current / self.total, 0.0), 1.0)
        return None


@dataclass(frozen=True)
class ProgressReport(ComputeReport):
    """A snapshot of the (possibly nested) progress stack."""

    levels: tuple[ProgressLevel, ...] = ()
    message: str = ""

    @property
    def top(self) -> Optional[ProgressLevel]:
        """The outermost level (best for an overall bar), or ``None``."""
        return self.levels[0] if self.levels else None

    @property
    def leaf(self) -> Optional[ProgressLevel]:
        """The innermost (deepest) level, or ``None``."""
        return self.levels[-1] if self.levels else None


@dataclass(frozen=True)
class StatusReport(ComputeReport):
    """Arbitrary status/metric data emitted by an algorithm."""

    message: str = ""
    data: Mapping[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Context-scoped observer registration
# ---------------------------------------------------------------------------


@contextmanager
def observe_reports(callback: ReportObserver) -> Iterator[None]:
    """Register *callback* as an observer for the duration of the ``with`` block.

    Any :class:`ProgressReporter` whose reports are emitted within this context
    (in the same thread) will notify *callback*.  This lets a caller observe an
    algorithm it did not construct itself (e.g. the engine created inside
    ``calc_dose_influence``).
    """
    token = _active_observers.set(_active_observers.get() + (callback,))
    try:
        yield
    finally:
        _active_observers.reset(token)


# ---------------------------------------------------------------------------
# Cooperative pause / stop control
# ---------------------------------------------------------------------------


class ComputeControl:
    """Thread-safe pause/stop signal for a running compute algorithm.

    A consumer (e.g. a GUI thread) creates a control, hands it to the algorithm's
    thread via :func:`observe_control`, and then calls :meth:`pause`/:meth:`resume`/
    :meth:`request_stop` from its own thread.  The algorithm cooperatively polls the
    control via :meth:`ProgressReporter.checkpoint` at safe points (e.g. between
    iterations): :meth:`wait_if_paused` blocks there until resumed or stopped, and
    :attr:`stop_requested` signals it should return its best result so far.
    """

    def __init__(self) -> None:
        # _resume_event is *set* while running and *cleared* while paused, so that
        # wait_if_paused() blocks exactly when paused.
        self._resume_event = threading.Event()
        self._resume_event.set()
        self._stop_event = threading.Event()

    def pause(self) -> None:
        """Request a pause; the next :meth:`wait_if_paused` will block."""
        self._resume_event.clear()

    def resume(self) -> None:
        """Resume after a pause, unblocking :meth:`wait_if_paused`."""
        self._resume_event.set()

    def request_stop(self) -> None:
        """Request cooperative cancellation (also unblocks a paused wait)."""
        self._stop_event.set()
        self._resume_event.set()

    @property
    def is_paused(self) -> bool:
        """Whether a pause is currently in effect (and no stop was requested)."""
        return not self._resume_event.is_set() and not self._stop_event.is_set()

    @property
    def stop_requested(self) -> bool:
        """Whether cooperative cancellation has been requested."""
        return self._stop_event.is_set()

    def wait_if_paused(self, timeout: Optional[float] = None) -> None:
        """Block while paused, returning immediately once resumed or stopped."""
        self._resume_event.wait(timeout)


@contextmanager
def observe_control(control: Optional[ComputeControl]) -> Iterator[None]:
    """Install *control* as the active cooperative control for the ``with`` block.

    Any :class:`ProgressReporter` whose :meth:`~ProgressReporter.checkpoint` runs in
    this context (same thread) will pause/stop according to *control*.  Passing
    ``None`` is a no-op, which keeps call sites simple.
    """
    if control is None:
        yield
        return
    token = _active_control.set(control)
    try:
        yield
    finally:
        _active_control.reset(token)


# ---------------------------------------------------------------------------
# The mixin
# ---------------------------------------------------------------------------


class _ActiveLevel:
    """Mutable bookkeeping for a level currently on the stack."""

    __slots__ = ("name", "total", "current")

    def __init__(self, name: str, total: Optional[int]) -> None:
        self.name = name
        self.total = total
        self.current = 0


class ProgressHandle:
    """Handle yielded by :meth:`ProgressReporter.progress` to advance a level."""

    __slots__ = ("_reporter", "_level")

    def __init__(self, reporter: "ProgressReporter", level: _ActiveLevel) -> None:
        self._reporter = reporter
        self._level = level

    def advance(self, n: int = 1) -> None:
        """Advance the current count by *n* and emit a report.

        Throttled, except the completing update (``current >= total``) is always
        emitted so consumers see the level reach 100%.
        """
        self._level.current += n
        self._reporter._emit_progress(force=self._is_complete())

    def update(self, current: int) -> None:
        """Set the absolute current count and emit a report (see :meth:`advance`)."""
        self._level.current = current
        self._reporter._emit_progress(force=self._is_complete())

    def _is_complete(self) -> bool:
        return self._level.total is not None and self._level.current >= self._level.total


class ProgressReporter:
    """Mixin giving a compute algorithm a progress + status reporting channel.

    Subclasses report progress with :meth:`track` (a ``tqdm`` drop-in) or the
    :meth:`progress` context manager, and arbitrary status with
    :meth:`report_status`.  By default progress is also rendered to the console
    via ``tqdm``; set :attr:`console_progress` to ``False`` to suppress that.
    """

    #: Whether :meth:`track` also renders a console ``tqdm`` bar.
    console_progress: bool = True

    #: Minimum seconds between *throttled* observer notifications (level
    #: push/pop and status are always emitted immediately).
    min_report_interval: float = 0.05

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._report_observers: list[ReportObserver] = []
        self._progress_stack: list[_ActiveLevel] = []
        self._cancel_check: Optional[Callable[[], bool]] = None
        self._last_emit_monotonic: float = 0.0
        super().__init__(*args, **kwargs)

    # -- observer / cancellation registration --------------------------------

    def add_report_observer(self, callback: ReportObserver) -> None:
        """Register *callback* to receive this algorithm's reports."""
        self._report_observers.append(callback)

    def remove_report_observer(self, callback: ReportObserver) -> None:
        """Remove a previously registered observer (no-op if absent)."""
        try:
            self._report_observers.remove(callback)
        except ValueError:
            pass

    def set_cancel_check(self, predicate: Optional[Callable[[], bool]]) -> None:
        """Install a predicate polled by :meth:`should_cancel`."""
        self._cancel_check = predicate

    def should_cancel(self) -> bool:
        """Return whether cooperative cancellation has been requested."""
        if self._cancel_check is not None and self._cancel_check():
            return True
        control = _active_control.get()
        return control is not None and control.stop_requested

    def checkpoint(self) -> bool:
        """Cooperative pause/stop point; return whether to continue.

        Blocks while the active :class:`ComputeControl` is paused, then returns
        ``False`` if cancellation has been requested (via the control or an installed
        :meth:`set_cancel_check`) and ``True`` otherwise.  Algorithms call this at safe
        points (e.g. between iterations) and stop when it returns ``False``.
        """
        control = _active_control.get()
        if control is not None:
            control.wait_if_paused()
        return not self.should_cancel()

    # -- progress reporting --------------------------------------------------

    @contextmanager
    def progress(self, name: str, total: Optional[int] = None) -> Iterator[ProgressHandle]:
        """Push a progress level for the duration of the ``with`` block."""
        level = _ActiveLevel(name, total)
        self._progress_stack.append(level)
        self._emit_progress(force=True)
        try:
            yield ProgressHandle(self, level)
        finally:
            self._progress_stack.pop()
            self._emit_progress(force=True)

    def track(
        self,
        iterable: Iterable[T],
        name: str,
        total: Optional[int] = None,
        **tqdm_kwargs: Any,
    ) -> Iterator[T]:
        """Iterate *iterable* as a progress level (a ``tqdm`` drop-in).

        Pushes a level named *name*, yields each item, and advances the level
        after each.  When :attr:`console_progress` is ``True`` a nested ``tqdm``
        bar is also shown (extra ``tqdm_kwargs`` such as ``unit`` are forwarded).
        """
        if total is None:
            try:
                total = len(iterable)  # type: ignore[arg-type]
            except TypeError:
                total = None

        with self.progress(name, total) as handle:
            iterator: Iterable[T] = iterable
            if self.console_progress:
                try:
                    from tqdm import tqdm  # noqa: PLC0415

                    iterator = tqdm(iterable, desc=name, total=total, leave=False, **tqdm_kwargs)
                except ImportError:
                    iterator = iterable
            for item in iterator:
                yield item
                handle.advance()

    def report_status(self, message: str = "", **data: Any) -> None:
        """Emit a :class:`StatusReport` with arbitrary metric *data*."""
        self._dispatch(StatusReport(message=message, data=dict(data)))

    # -- internals -----------------------------------------------------------

    def _emit_progress(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and (now - self._last_emit_monotonic) < self.min_report_interval:
            return
        self._last_emit_monotonic = now
        levels = tuple(
            ProgressLevel(lvl.name, lvl.current, lvl.total) for lvl in self._progress_stack
        )
        self._dispatch(ProgressReport(levels=levels))

    def _dispatch(self, report: ComputeReport) -> None:
        for callback in (*self._report_observers, *_active_observers.get()):
            try:
                callback(report)
            except Exception:  # noqa: BLE001 - an observer must never break compute
                logger.exception("Progress/status observer raised; ignoring.")
