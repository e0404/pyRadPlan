"""Plain-Python observable store of the six pipeline objects."""

from __future__ import annotations

import logging
from typing import Any, Callable, Iterable, Optional

logger = logging.getLogger(__name__)

#: Names of the pipeline objects, in workflow order.
WORKSPACE_KEYS: tuple[str, ...] = ("ct", "cst", "pln", "stf", "dij", "result")

#: Callback receiving the list of changed keys.
WorkspaceListener = Callable[[list[str]], None]


class Workspace:
    """Observable store of ``ct, cst, pln, stf, dij, result``.

    The Qt-free counterpart of :class:`pyRadPlan.gui.workspace.WorkspaceManager`, with the
    same keys and change semantics: every write notifies the subscribers once with the
    list of changed keys, :meth:`set_many` publishes several objects in one notification,
    and an empty list means "refresh everything".

    Listeners are called synchronously in the writing thread, in subscription order. A
    listener that raises is logged and skipped, so one consumer cannot break the others.
    """

    keys: tuple[str, ...] = WORKSPACE_KEYS

    def __init__(self) -> None:
        self._data: dict[str, Any] = dict.fromkeys(WORKSPACE_KEYS)
        self._listeners: list[WorkspaceListener] = []

    def subscribe(self, callback: WorkspaceListener) -> Callable[[], None]:
        """Call *callback* with the changed keys after every change.

        Returns
        -------
        Callable[[], None]
            Removes the subscription; calling it again is a no-op.
        """
        self._listeners.append(callback)

        def unsubscribe() -> None:
            try:
                self._listeners.remove(callback)
            except ValueError:
                pass

        return unsubscribe

    def _notify(self, changed: list[str]) -> None:
        for callback in list(self._listeners):
            try:
                callback(list(changed))
            except Exception:  # noqa: BLE001 - a listener must never break the store
                logger.exception("Workspace listener raised; ignoring.")

    def get(self, key: str) -> Any:
        """Return the object stored under *key* (``None`` if unset)."""
        self._check_key(key)
        return self._data[key]

    def set(self, key: str, value: Any) -> None:
        """Store *value* under *key* and notify."""
        self._check_key(key)
        self._data[key] = value
        self._notify([key])

    def set_many(self, **kwargs: Any) -> None:
        """Store several objects and notify once; unknown keys are ignored."""
        changed = [k for k in kwargs if k in WORKSPACE_KEYS]
        for k in changed:
            self._data[k] = kwargs[k]
        if changed:
            self._notify(changed)

    def clear(self, keys: Optional[Iterable[str]] = None) -> None:
        """Set *keys* (all keys when omitted) to ``None`` and notify."""
        targets = list(keys) if keys is not None else list(WORKSPACE_KEYS)
        for k in targets:
            if k in WORKSPACE_KEYS:
                self._data[k] = None
        self._notify(targets)

    def has(self, *keys: str) -> bool:
        """Whether every named key holds a non-``None`` value."""
        return all(self._data.get(k) is not None for k in keys)

    def present(self) -> frozenset[str]:
        """Return the keys currently holding a value."""
        return frozenset(k for k, v in self._data.items() if v is not None)

    def refresh(self) -> None:
        """Notify every key as changed, forcing consumers to update."""
        self._notify(list(WORKSPACE_KEYS))

    def snapshot(self) -> dict[str, Any]:
        """Return a shallow copy of the stored objects."""
        return dict(self._data)

    @staticmethod
    def _check_key(key: str) -> None:
        if key not in WORKSPACE_KEYS:
            raise KeyError(f"Unknown workspace key {key!r}; expected one of {WORKSPACE_KEYS}")

    def __getattr__(self, name: str) -> Any:
        if name in WORKSPACE_KEYS:
            return self.__dict__["_data"][name]
        raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in WORKSPACE_KEYS:
            self.set(name, value)
        else:
            super().__setattr__(name, value)
