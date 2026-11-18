"""Centralised data store for a pyRadPlan GUI session.

Replaces MATLAB's base workspace: all six pipeline objects (ct, cst, pln, stf,
dij, result) live here and interested widgets subscribe to *workspace_changed*.
The objects are held by a Qt-free :class:`pyRadPlan.app.Workspace`, which the
manager's :class:`pyRadPlan.app.Session` (stage tracking, workflow operations)
shares with the web frontend's code.
"""

from __future__ import annotations

from typing import Any, Optional

from PySide6.QtCore import QObject, Signal

from pyRadPlan.app import WORKSPACE_KEYS, Session, Workspace


class WorkspaceManager(QObject):
    """Central data store for the pyRadPlan GUI.

    Holds the six treatment-planning pipeline objects and notifies listeners
    via *workspace_changed* (carrying a list of the changed key names) whenever
    any object is replaced or cleared.

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
        """Set multiple pipeline objects and emit a single *workspace_changed*."""
        self.store.set_many(**kwargs)

    def clear(self, keys: Optional[list[str]] = None) -> None:
        """Set *keys* to *None* (or all keys when *keys* is omitted)."""
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
        """The optimization/forward-dose result dict, or ``None``."""
        return self.store.get("result")

    @result.setter
    def result(self, value) -> None:
        self.store.set("result", value)
