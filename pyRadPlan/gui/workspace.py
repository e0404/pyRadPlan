"""Centralised data store for a pyRadPlan GUI session.

Replaces MATLAB's base workspace: all six pipeline objects (ct, cst, pln, stf,
dij, result) live here and interested widgets subscribe to *workspace_changed*.
"""

from __future__ import annotations

from typing import Any, Optional

from PySide6.QtCore import QObject, Signal


class WorkspaceManager(QObject):
    """Central data store for the pyRadPlan GUI.

    Holds the six treatment-planning pipeline objects and notifies listeners
    via *workspace_changed* (carrying a list of the changed key names) whenever
    any object is replaced or cleared.

    A process-wide singleton is available via :meth:`instance`.
    """

    workspace_changed = Signal(list)  # list[str] of changed key names

    _KEYS = ("ct", "cst", "pln", "stf", "dij", "result")
    _instance: Optional[WorkspaceManager] = None

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._data: dict[str, Any] = dict.fromkeys(self._KEYS)

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

    @staticmethod
    def _coerce(key: str, value: Any) -> Any:
        """Validate *value* into the model the rest of the GUI expects for *key*.

        Only ``result`` is coerced: it is written from many places (optimization,
        dose import, matRad files) that would otherwise leave a bare dict behind,
        and every consumer treats it as a
        :class:`~pyRadPlan.result.PlanResult`.
        """
        if key != "result" or value is None:
            return value

        # Deferred: keeps the result model out of GUI module import time.
        from pyRadPlan.result import validate_result  # noqa: PLC0415

        return validate_result(value)

    def _set(self, key: str, value: Any) -> None:
        self._data[key] = self._coerce(key, value)
        self.workspace_changed.emit([key])

    def _get(self, key: str) -> Any:
        return self._data[key]

    def set_many(self, **kwargs: Any) -> None:
        """Set multiple pipeline objects and emit a single *workspace_changed*."""
        changed = [k for k in kwargs if k in self._KEYS]
        for k in changed:
            self._data[k] = self._coerce(k, kwargs[k])
        if changed:
            self.workspace_changed.emit(changed)

    def clear(self, keys: Optional[list[str]] = None) -> None:
        """Set *keys* to *None* (or all keys when *keys* is omitted)."""
        targets = list(keys) if keys is not None else list(self._KEYS)
        for k in targets:
            if k in self._KEYS:
                self._data[k] = None
        self.workspace_changed.emit(targets)

    def has(self, *keys: str) -> bool:
        """Return *True* only when every named key holds a non-*None* value."""
        return all(self._data.get(k) is not None for k in keys)

    def refresh(self) -> None:
        """Re-emit *workspace_changed* for every key, forcing all widgets to update."""
        self.workspace_changed.emit(list(self._KEYS))

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
        return self._get("ct")

    @ct.setter
    def ct(self, value) -> None:
        self._set("ct", value)

    @property
    def cst(self):
        """The structure set (VOIs and objectives), or ``None``."""
        return self._get("cst")

    @cst.setter
    def cst(self, value) -> None:
        self._set("cst", value)

    @property
    def pln(self):
        """The treatment plan, or ``None``."""
        return self._get("pln")

    @pln.setter
    def pln(self, value) -> None:
        self._set("pln", value)

    @property
    def stf(self):
        """The steering information (beam geometry), or ``None``."""
        return self._get("stf")

    @stf.setter
    def stf(self, value) -> None:
        self._set("stf", value)

    @property
    def dij(self):
        """The dose-influence matrix, or ``None``."""
        return self._get("dij")

    @dij.setter
    def dij(self, value) -> None:
        self._set("dij", value)

    @property
    def result(self):
        """The optimization/forward-dose result as a ``PlanResult``, or ``None``.

        Assigning a dict of named quantities is allowed; it is validated into a
        ``PlanResult`` on the way in.
        """
        return self._get("result")

    @result.setter
    def result(self, value) -> None:
        self._set("result", value)
