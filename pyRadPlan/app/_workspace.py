"""Plain-Python observable store of the pipeline objects and the results."""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Callable, Iterable, Optional

if TYPE_CHECKING:
    from pyRadPlan.result import PlanResult

logger = logging.getLogger(__name__)

#: Names of the pipeline objects, in workflow order.
WORKSPACE_KEYS: tuple[str, ...] = ("ct", "cst", "pln", "stf", "dij", "result")

#: Name the result of an optimization or recalculation is stored under by default.
DEFAULT_RESULT_NAME = "current"
#: Name given to a dose that a patient file brings along without a plan result.
IMPORTED_RESULT_NAME = "imported"

# Result names prefix the keys results are listed by, "<result>/<quantity>" (see
# ``pyRadPlan.quantities._display_names.result_key``, not imported here to keep this
# module free of the quantity registry).
_INVALID_NAME_PARTS = re.compile(r"[\s/]+")

#: Callback receiving the list of changed keys.
WorkspaceListener = Callable[[list[str]], None]


def unique_result_name(name: str, taken: Iterable[str]) -> str:
    """Return *name* as a result name, or ``name_2``/``name_3``/... if it is *taken*."""
    name = _INVALID_NAME_PARTS.sub("_", str(name).strip()) or "result"
    taken = set(taken)
    if name not in taken:
        return name
    i = 2
    while f"{name}_{i}" in taken:
        i += 1
    return f"{name}_{i}"


class Workspace:
    """Observable store of ``ct, cst, pln, stf, dij`` and the results.

    The Qt-free counterpart of :class:`pyRadPlan.gui.workspace.WorkspaceManager`, with the
    same keys and change semantics: every write notifies the subscribers once with the
    list of changed keys, :meth:`set_many` publishes several objects in one notification,
    and an empty list means "refresh everything".

    Results are kept as a collection of named :class:`~pyRadPlan.result.PlanResult`
    objects, one per computation. The *active* result is the one optimization and
    recalculation write to and that the key ``"result"`` reads and writes; snapshots and
    imported doses are further entries. Any change of the collection is announced as
    ``"result"``.

    Listeners are called synchronously in the writing thread, in subscription order. A
    listener that raises is logged and skipped, so one consumer cannot break the others.
    """

    keys: tuple[str, ...] = WORKSPACE_KEYS

    def __init__(self) -> None:
        self._data: dict[str, Any] = dict.fromkeys(WORKSPACE_KEYS[:-1])
        self._results: dict[str, PlanResult] = {}
        self._active_result = DEFAULT_RESULT_NAME
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

    def _store(self, key: str, value: Any) -> None:
        if key == "result":
            self._store_result(self._active_result, value)
        else:
            self._data[key] = value

    def get(self, key: str) -> Any:
        """Return the object stored under *key* (``None`` if unset).

        ``"result"`` is the active result; ``"results"`` returns :meth:`result_entries`.
        """
        if key == "results":
            return self.result_entries()
        self._check_key(key)
        if key == "result":
            return self._results.get(self._active_result)
        return self._data[key]

    def set(self, key: str, value: Any) -> None:
        """Store *value* under *key* and notify."""
        self._check_key(key)
        self._store(key, value)
        self._notify([key])

    def set_many(self, **kwargs: Any) -> None:
        """Store several objects and notify once; unknown keys are ignored.

        Besides the pipeline objects, ``results`` replaces the whole collection of
        results (``active_result`` names the active one among them).
        """
        changed = [k for k in kwargs if k in WORKSPACE_KEYS]
        if "results" in kwargs:
            self._replace_results(kwargs["results"] or {}, kwargs.get("active_result"))
        for k in changed:
            self._store(k, kwargs[k])
        if "results" in kwargs and "result" not in changed:
            changed.append("result")
        if changed:
            self._notify(changed)

    def clear(self, keys: Optional[Iterable[str]] = None) -> None:
        """Set *keys* (all keys when omitted) to ``None`` and notify.

        Clearing ``result`` drops every result, not only the active one.
        """
        targets = list(keys) if keys is not None else list(WORKSPACE_KEYS)
        for k in targets:
            if k == "result":
                self._replace_results({}, None)
            elif k in WORKSPACE_KEYS:
                self._data[k] = None
        self._notify(targets)

    def has(self, *keys: str) -> bool:
        """Whether every named key holds a non-``None`` value."""
        return all(k in WORKSPACE_KEYS and self.get(k) is not None for k in keys)

    def present(self) -> frozenset[str]:
        """Return the keys currently holding a value."""
        return frozenset(k for k in WORKSPACE_KEYS if self.get(k) is not None)

    def refresh(self) -> None:
        """Notify every key as changed, forcing consumers to update."""
        self._notify(list(WORKSPACE_KEYS))

    def snapshot(self) -> dict[str, Any]:
        """Return a shallow copy of the stored objects (``result`` is the active result)."""
        return {**self._data, "result": self._results.get(self._active_result)}

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    def _store_result(self, name: str, value: Any) -> None:
        # Results are written from many places (optimization, dose import, matRad
        # files) that may hand over a plain mapping of quantities.
        from pyRadPlan.result import validate_result  # noqa: PLC0415

        if value is None:
            self._results.pop(name, None)
            return
        if not isinstance(name, str) or not name or _INVALID_NAME_PARTS.search(name):
            raise ValueError(
                f"Invalid result name {name!r}: use a name without whitespace and '/'."
            )
        self._results[name] = validate_result(value)

    def _replace_results(self, results: Mapping[str, Any], active: Optional[str]) -> None:
        self._results = {}
        for name, value in results.items():
            self._store_result(name, value)
        self._active_result = active if active in self._results else DEFAULT_RESULT_NAME

    @property
    def results(self) -> dict[str, PlanResult]:
        """All results by name (a copy; use :meth:`add_result` to change them)."""
        return dict(self._results)

    @property
    def active_result(self) -> str:
        """Name of the result that the key ``"result"`` reads and writes."""
        return self._active_result

    @active_result.setter
    def active_result(self, name: str) -> None:
        if name not in self._results:
            raise KeyError(f"There is no result named {name!r}.")
        self._active_result = name
        self._notify(["result"])

    def add_result(self, name: str, result: Any) -> None:
        """Store *result* under *name*, replacing a result of that name."""
        if result is None:
            raise ValueError("Cannot add an empty result; use remove_result() to drop one.")
        self._store_result(name, result)
        self._notify(["result"])

    def remove_result(self, name: str) -> None:
        """Drop the result *name*; unknown names are ignored."""
        if self._results.pop(name, None) is not None:
            self._notify(["result"])

    def result_entries(self) -> list[tuple[str, PlanResult]]:
        """``(tag, result)`` of every result, the active one first with the tag ``""``.

        The tag prefixes the keys the quantities of a result are listed by (see
        :func:`pyRadPlan.quantities._display_names.result_key`).
        """
        active = self._results.get(self._active_result)
        entries = [] if active is None else [("", active)]
        entries += [(n, r) for n, r in self._results.items() if n != self._active_result]
        return entries

    @staticmethod
    def _check_key(key: str) -> None:
        if key not in WORKSPACE_KEYS:
            raise KeyError(f"Unknown workspace key {key!r}; expected one of {WORKSPACE_KEYS}")

    def __getattr__(self, name: str) -> Any:
        if name in WORKSPACE_KEYS:
            return self.get(name)
        raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in WORKSPACE_KEYS:
            self.set(name, value)
        else:
            super().__setattr__(name, value)
