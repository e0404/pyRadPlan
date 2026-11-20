"""Centralised data store for a pyRadPlan GUI session.

Replaces MATLAB's base workspace: the pipeline objects (ct, cst, pln, stf, dij)
and the results live here and interested widgets subscribe to *workspace_changed*.
The objects are held by a Qt-free :class:`pyRadPlan.app.Workspace`, which the
manager's :class:`pyRadPlan.app.Session` (stage tracking, workflow operations)
shares with the web frontend's code.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from PySide6.QtCore import QObject, Signal

from pyRadPlan.app import WORKSPACE_KEYS, Session, Workspace

if TYPE_CHECKING:
    from pyRadPlan.result import PlanResult


class WorkspaceManager(QObject):
    """Central data store for the pyRadPlan GUI.

    Holds the treatment-planning pipeline objects and notifies listeners via
    *workspace_changed* (carrying a list of the changed key names) whenever any
    object is replaced or cleared.

    Results are kept as a collection of named
    :class:`~pyRadPlan.result.PlanResult` objects, one per computation: the
    *active* result is the one optimization and recalculation write to and that
    :attr:`result` returns; snapshots and imported doses are further entries.
    Any change of the collection is announced as ``"result"``.

    The objects live in :attr:`store`, observed first by :attr:`session`, so the
    session's stage flags are current when *workspace_changed* reaches the widgets.

    A process-wide singleton is available via :meth:`instance`.
    """

    workspace_changed = Signal(list)  # list[str] of changed key names

    _KEYS = WORKSPACE_KEYS
    _instance: Optional[WorkspaceManager] = None

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        #: The Qt-free store holding the pipeline objects.
        self.store = Workspace()
        #: The workflow session on :attr:`store`. The plan form applies the default plan
        #: and the optimization problem picks its solver, as before the session existed.
        self.session = Session(
            workspace=self.store,
            profile="native",
            auto_plan=False,
            solver_preference=(),
            capture_logs=False,
        )
        self.store.subscribe(self.workspace_changed.emit)

    # ------------------------------------------------------------------
    # Singleton
    # ------------------------------------------------------------------

    @classmethod
    def instance(cls) -> WorkspaceManager:
        """Return the process-wide singleton, creating it on first call."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------
    # Generic read / write
    # ------------------------------------------------------------------

    def set_many(self, **kwargs: Any) -> None:
        """Set multiple pipeline objects and emit a single *workspace_changed*.

        Besides the pipeline objects, ``results`` replaces the whole collection of
        results (``active_result`` names the active one among them).
        """
        self.store.set_many(**kwargs)

    def clear(self, keys: Optional[list[str]] = None) -> None:
        """Set *keys* to *None* (or all keys when *keys* is omitted).

        Clearing ``result`` drops every result, not only the active one.
        """
        self.store.clear(keys)

    def has(self, *keys: str) -> bool:
        """Return *True* only when every named key holds a non-*None* value."""
        return self.store.has(*keys)

    def present(self) -> frozenset[str]:
        """Return the keys currently holding a value."""
        return self.store.present()

    def refresh(self) -> None:
        """Re-emit *workspace_changed* for every key, forcing all widgets to update."""
        self.store.refresh()

    @property
    def keys(self) -> tuple[str, ...]:
        """The names of the pipeline objects managed by the workspace."""
        return self._KEYS

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    @property
    def results(self) -> dict[str, PlanResult]:
        """All results by name (a copy; use :meth:`add_result` to change them)."""
        return self.store.results

    @property
    def active_result(self) -> str:
        """Name of the result that :attr:`result` reads and writes."""
        return self.store.active_result

    @active_result.setter
    def active_result(self, name: str) -> None:
        self.store.active_result = name

    def add_result(self, name: str, result: Any) -> None:
        """Store *result* under *name*, replacing a result of that name."""
        self.store.add_result(name, result)

    def remove_result(self, name: str) -> None:
        """Drop the result *name*; unknown names are ignored."""
        self.store.remove_result(name)

    def result_entries(self) -> list[tuple[str, PlanResult]]:
        """``(tag, result)`` of every result, the active one first with the tag ``""``."""
        return self.store.result_entries()

    # ------------------------------------------------------------------
    # Typed properties — one per pyRadPlan pipeline object
    # ------------------------------------------------------------------

    @property
    def ct(self):
        """The CT image object, or ``None``."""
        return self.store.get("ct")

    @ct.setter
    def ct(self, value) -> None:
        self.store.set("ct", value)

    @property
    def cst(self):
        """The structure set (VOIs and objectives), or ``None``."""
        return self.store.get("cst")

    @cst.setter
    def cst(self, value) -> None:
        self.store.set("cst", value)

    @property
    def pln(self):
        """The treatment plan, or ``None``."""
        return self.store.get("pln")

    @pln.setter
    def pln(self, value) -> None:
        self.store.set("pln", value)

    @property
    def stf(self):
        """The steering information (beam geometry), or ``None``."""
        return self.store.get("stf")

    @stf.setter
    def stf(self, value) -> None:
        self.store.set("stf", value)

    @property
    def dij(self):
        """The dose-influence matrix, or ``None``."""
        return self.store.get("dij")

    @dij.setter
    def dij(self, value) -> None:
        self.store.set("dij", value)

    @property
    def result(self):
        """The active result as a ``PlanResult``, or ``None``.

        Assigning a dict of named quantities is allowed; it is validated into a
        ``PlanResult`` on the way in.  Assigning ``None`` drops the active result
        and keeps the others (see :meth:`clear` to drop all).
        """
        return self.store.get("result")

    @result.setter
    def result(self, value) -> None:
        self.store.set("result", value)
