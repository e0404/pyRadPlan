"""Base class and shared building blocks for workspace-aware pyRadPlan GUI widgets.

This is the PySide6 analogue of matRad's ``matRad_Widget``: a widget that binds
to a :class:`~pyRadPlan.gui.workspace.WorkspaceManager` and is notified through
:meth:`_do_update` whenever the shared pipeline objects change.  Subclasses build
their UI in ``__init__`` and override :meth:`_do_update`.

The module also hosts small widgets/helpers shared across the GUI package:
:class:`Worker` (run a callable in a ``QThread``), :class:`AdaptiveDoubleSpinBox`
and the number-list parse/format helpers used by free-text angle/level fields.
"""

from __future__ import annotations

import logging
import math
from contextlib import contextmanager
from typing import Any, Callable, Iterator, Optional

from PySide6.QtCore import QObject, Qt, Signal, Slot
from PySide6.QtWidgets import QAbstractSpinBox, QDoubleSpinBox, QWidget

from pyRadPlan.core import ComputeControl, observe_control, observe_reports
from pyRadPlan.gui.workspace import WorkspaceManager

logger = logging.getLogger(__name__)


def parse_number_list(text: str, item_type: type = float) -> list:
    """Parse a comma- or blank-separated list of numbers from free text.

    Raises
    ------
    ValueError
        If any token cannot be converted to *item_type*.
    """
    tokens = [t for t in text.replace(",", " ").split() if t]
    return [item_type(t) for t in tokens]


def format_number_list(values) -> str:
    """Format numbers as a blank-separated string (inverse of :func:`parse_number_list`)."""
    return " ".join(f"{float(v):g}" for v in values)


class Worker(QObject):
    """Runs a callable in a QThread and reports the outcome via signals.

    If *report_cb* is given, the callable runs inside an
    :func:`~pyRadPlan.core.observe_reports` context so any compute algorithm it
    triggers reports progress/status to *report_cb* (called in this thread).  If
    *control* is given, it is installed via :func:`~pyRadPlan.core.observe_control`
    so the algorithm can be cooperatively paused/stopped from another thread.
    """

    finished = Signal(object)
    error = Signal(object)

    def __init__(
        self,
        fn: Callable,
        *args: Any,
        report_cb: Optional[Callable] = None,
        control: Optional[ComputeControl] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self._fn = fn
        self._args = args
        self._kwargs = kwargs
        self._report_cb = report_cb
        self._control = control

    @Slot()
    def run(self) -> None:
        try:
            if self._report_cb is not None:
                with observe_reports(self._report_cb), observe_control(self._control):
                    result = self._fn(*self._args, **self._kwargs)
            else:
                with observe_control(self._control):
                    result = self._fn(*self._args, **self._kwargs)
            self.finished.emit(result)
        except Exception as exc:  # noqa: BLE001
            self.error.emit(exc)
        finally:
            # Release the callable (and anything captured in its closure, e.g.
            # large pipeline objects) as soon as the run is over.
            self._fn = self._args = self._kwargs = None


class AdaptiveDoubleSpinBox(QDoubleSpinBox):
    """A double spin box whose displayed precision adapts to the magnitude.

    Larger values are shown with fewer decimals (``>=10`` -> 1, ``1..10`` -> 2,
    ``<1`` -> 3, and one more for each further power of ten below that), while a
    high internal precision keeps small values from being rounded away.
    """

    _INTERNAL_DECIMALS = 6

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        super().setDecimals(self._INTERNAL_DECIMALS)
        self.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.PlusMinus)
        self.setAlignment(Qt.AlignRight)
        self.setMinimumWidth(90)
        # Emit valueChanged only on commit (focus-out/Enter/steps), not per
        # keystroke: listeners often push whole workspace updates per change.
        self.setKeyboardTracking(False)

    @classmethod
    def _display_decimals(cls, value: float) -> int:
        magnitude = abs(value)
        if magnitude >= 10.0:
            return 1
        if magnitude >= 1.0 or magnitude == 0.0:
            return 2
        # 3 decimals for [0.1, 1), one more for each further power of ten below.
        return min(2 - math.floor(math.log10(magnitude)), cls._INTERNAL_DECIMALS)

    def textFromValue(self, value: float) -> str:  # noqa: N802 - Qt override name
        return f"{value:.{self._display_decimals(value)}f}"


class WorkspaceWidget(QWidget):
    """A QWidget bound to a :class:`WorkspaceManager`.

    Subclasses should:

    1. Build their UI in ``__init__`` (after calling ``super().__init__``).
    2. Override :meth:`_do_update` to refresh the UI from the workspace.
    3. Call :meth:`initialize` at the end of ``__init__`` for the first render.
    4. Optionally set :attr:`_watched_keys` so the widget only reacts to relevant
       pipeline objects (``ct``, ``cst``, ``pln``, ``stf``, ``dij``, ``result``).

    When a widget writes back to the workspace it should do so inside the
    :meth:`hold_updates` context manager so it does not react to its own change
    (mirrors matRad's ``updateLock``).  Other widgets still update normally.

    Parameters
    ----------
    workspace:
        The shared workspace.  Falls back to ``WorkspaceManager.instance()``.
    parent:
        Optional Qt parent widget.
    """

    #: Pipeline keys this widget reacts to.  Empty means "react to everything".
    _watched_keys: tuple[str, ...] = ()

    #: Emitted (with a human-readable message) when :meth:`_do_update` raises.
    update_failed = Signal(str)

    def __init__(
        self,
        workspace: Optional[WorkspaceManager] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._ws = workspace or WorkspaceManager.instance()
        self._update_locked = False
        self._ws.workspace_changed.connect(self._on_workspace_changed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def workspace(self) -> WorkspaceManager:
        """The :class:`WorkspaceManager` this widget is bound to."""
        return self._ws

    def initialize(self) -> None:
        """Perform the first update.  Call once at the end of subclass init."""
        self._dispatch_update([])

    @contextmanager
    def hold_updates(self) -> Iterator[None]:
        """Suspend this widget's own reaction to ``workspace_changed``.

        Use around writes to the workspace so the widget does not refresh itself
        from a change it just made (which would clobber in-progress edits).
        """
        previous = self._update_locked
        self._update_locked = True
        try:
            yield
        finally:
            self._update_locked = previous

    # ------------------------------------------------------------------
    # Internal dispatch
    # ------------------------------------------------------------------

    @Slot(list)
    def _on_workspace_changed(self, changed_keys: list) -> None:
        if self._update_locked:
            return
        # An empty changed list means "refresh everything" (initialize/refresh).
        if self._watched_keys and changed_keys:
            if not any(k in self._watched_keys for k in changed_keys):
                return
        self._dispatch_update(list(changed_keys))

    def _dispatch_update(self, changed_keys: list) -> None:
        try:
            self._do_update(changed_keys)
        except Exception as exc:  # noqa: BLE001 - never let an update crash the GUI
            logger.exception("%s update failed", type(self).__name__)
            self.update_failed.emit(f"{type(exc).__name__}: {exc}")

    # ------------------------------------------------------------------
    # To be overridden
    # ------------------------------------------------------------------

    def _do_update(self, changed_keys: list) -> None:
        """Refresh the widget from the workspace.

        Parameters
        ----------
        changed_keys:
            The pipeline keys that changed.  Empty list signals a full refresh.
        """
