"""Centralised data store for a pyRadPlan GUI session.

Replaces MATLAB's base workspace: the pipeline objects (ct, cst, pln, stf, dij)
and the results live here and interested widgets subscribe to *workspace_changed*.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any, Optional

from PySide6.QtCore import QObject, Signal

if TYPE_CHECKING:
    from pyRadPlan.result import PlanResult

#: Name the result of an optimization or recalculation is stored under by default.
DEFAULT_RESULT_NAME = "current"
#: Name given to a dose that a patient file brings along without a plan result.
IMPORTED_RESULT_NAME = "imported"
#: Separates the result name from the quantity in the keys widgets list results by.
RESULT_KEY_SEPARATOR = "/"

_RESULT_NAME = re.compile(rf"[^\s{RESULT_KEY_SEPARATOR}]+")
_LEGACY_IMPORT_PREFIX = "import_"
_LEGACY_TAGS_KEY = "saved_result_tags"
_CONTAINERS = ("quantities", "beam_quantities", "scenario_quantities")


def unique_result_name(name: str, taken: Iterable[str]) -> str:
    """Return *name* as a result name, or ``name_2``/``name_3``/... if it is *taken*."""
    name = re.sub(rf"[\s{RESULT_KEY_SEPARATOR}]+", "_", name.strip()) or "result"
    taken = set(taken)
    if name not in taken:
        return name
    i = 2
    while f"{name}_{i}" in taken:
        i += 1
    return f"{name}_{i}"


def _split_legacy_result(result: PlanResult, tags: Iterable[str]) -> dict[str, PlanResult]:
    """Split a result that holds snapshots and imported doses under mangled names.

    Sessions saved up to version 0.9 keep everything in one result: snapshots as
    ``"<quantity>_<tag>"`` and imported doses as ``"import_<name>"``.
    """
    from pyRadPlan.result import PlanResult  # noqa: PLC0415

    tags = sorted({tag for tag in tags if isinstance(tag, str) and tag}, key=len, reverse=True)
    kept: dict[str, dict] = {container: {} for container in _CONTAINERS}
    split: dict[str, dict[str, dict]] = {}

    for container in _CONTAINERS:
        for key, value in getattr(result, container).items():
            tag = next((t for t in tags if key.endswith(f"_{t}") and len(key) > len(t) + 1), None)
            if tag is not None:
                parts = split.setdefault(tag, {c: {} for c in _CONTAINERS})
                parts[container][key[: -len(tag) - 1]] = value
            elif container == "quantities" and key.startswith(_LEGACY_IMPORT_PREFIX):
                name = unique_result_name(key[len(_LEGACY_IMPORT_PREFIX) :], [*split, *tags])
                split[name] = {"quantities": {"physical_dose": value}}
            else:
                kept[container][key] = value

    results: dict[str, PlanResult] = {}
    if any(kept.values()) or result.fluence is not None:
        results[DEFAULT_RESULT_NAME] = result.model_copy(update=kept) if split else result
    for name, parts in split.items():
        labels = result.scenario_labels
        scenarios = parts.get("scenario_quantities") or {}
        if labels is not None and not any(len(v) == len(labels) for v in scenarios.values()):
            labels = None
        results[unique_result_name(name, results)] = PlanResult(scenario_labels=labels, **parts)
    return results


def loaded_results(data: Mapping) -> tuple[dict[str, Any], Optional[str]]:
    """Results of a loaded file as ``(results by name, name of the active one)``.

    A session file brings its named results.  Any other plan result becomes the
    active one; a dose without a plan result is kept as an imported result.
    """
    from pyRadPlan.result import validate_result  # noqa: PLC0415

    results = data.get("results")
    if isinstance(results, Mapping) and results:
        return dict(results), data.get("active_result")
    result = data.get("result")
    if result is not None:
        result = validate_result(result)
        legacy = any(key.startswith(_LEGACY_IMPORT_PREFIX) for key in result.quantities)
        if legacy or data.get(_LEGACY_TAGS_KEY):
            return _split_legacy_result(result, data.get(_LEGACY_TAGS_KEY) or ()), None
        return {DEFAULT_RESULT_NAME: result}, None
    if data.get("dose") is not None:
        return {IMPORTED_RESULT_NAME: {"physical_dose": data["dose"]}}, None
    return {}, None


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

    A process-wide singleton is available via :meth:`instance`.
    """

    workspace_changed = Signal(list)  # list[str] of changed key names

    _KEYS = ("ct", "cst", "pln", "stf", "dij", "result")
    _instance: Optional[WorkspaceManager] = None

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._data: dict[str, Any] = dict.fromkeys(self._KEYS[:-1])
        self._results: dict[str, PlanResult] = {}
        self._active_result = DEFAULT_RESULT_NAME

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

    def _store(self, key: str, value: Any) -> None:
        if key == "result":
            self._store_result(self._active_result, value)
        else:
            self._data[key] = value

    def _set(self, key: str, value: Any) -> None:
        self._store(key, value)
        self.workspace_changed.emit([key])

    def _get(self, key: str) -> Any:
        if key == "result":
            return self._results.get(self._active_result)
        return self._data[key]

    def set_many(self, **kwargs: Any) -> None:
        """Set multiple pipeline objects and emit a single *workspace_changed*.

        Besides the pipeline objects, ``results`` replaces the whole collection of
        results (``active_result`` names the active one among them).
        """
        changed = [k for k in kwargs if k in self._KEYS]
        if "results" in kwargs:
            self._replace_results(kwargs["results"] or {}, kwargs.get("active_result"))
        for k in changed:
            self._store(k, kwargs[k])
        if "results" in kwargs and "result" not in changed:
            changed.append("result")
        if changed:
            self.workspace_changed.emit(changed)

    def clear(self, keys: Optional[list[str]] = None) -> None:
        """Set *keys* to *None* (or all keys when *keys* is omitted).

        Clearing ``result`` drops every result, not only the active one.
        """
        targets = list(keys) if keys is not None else list(self._KEYS)
        for k in targets:
            if k == "result":
                self._replace_results({}, None)
            elif k in self._KEYS:
                self._data[k] = None
        self.workspace_changed.emit(targets)

    def has(self, *keys: str) -> bool:
        """Return *True* only when every named key holds a non-*None* value."""
        return all(k in self._KEYS and self._get(k) is not None for k in keys)

    def refresh(self) -> None:
        """Re-emit *workspace_changed* for every key, forcing all widgets to update."""
        self.workspace_changed.emit(list(self._KEYS))

    @property
    def keys(self) -> tuple[str, ...]:
        """The names of the pipeline objects managed by the workspace."""
        return self._KEYS

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
        if not isinstance(name, str) or not _RESULT_NAME.fullmatch(name):
            raise ValueError(
                f"Invalid result name {name!r}: use a name without whitespace "
                f"and '{RESULT_KEY_SEPARATOR}'."
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
        """Name of the result that :attr:`result` reads and writes."""
        return self._active_result

    @active_result.setter
    def active_result(self, name: str) -> None:
        if name not in self._results:
            raise KeyError(f"There is no result named {name!r}.")
        self._active_result = name
        self.workspace_changed.emit(["result"])

    def add_result(self, name: str, result: Any) -> None:
        """Store *result* under *name*, replacing a result of that name."""
        if result is None:
            raise ValueError("Cannot add an empty result; use remove_result() to drop one.")
        self._store_result(name, result)
        self.workspace_changed.emit(["result"])

    def remove_result(self, name: str) -> None:
        """Drop the result *name*; unknown names are ignored."""
        if self._results.pop(name, None) is not None:
            self.workspace_changed.emit(["result"])

    def result_entries(self) -> list[tuple[str, PlanResult]]:
        """``(tag, result)`` of every result, the active one first with the tag ``""``.

        The tag prefixes the keys widgets list the quantities of a result by (see
        :func:`pyRadPlan.gui.widgets._display_names.result_key`).
        """
        active = self._results.get(self._active_result)
        entries = [] if active is None else [("", active)]
        entries += [(n, r) for n, r in self._results.items() if n != self._active_result]
        return entries

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
        """The active result as a ``PlanResult``, or ``None``.

        Assigning a dict of named quantities is allowed; it is validated into a
        ``PlanResult`` on the way in.  Assigning ``None`` drops the active result
        and keeps the others (see :meth:`clear` to drop all).
        """
        return self._get("result")

    @result.setter
    def result(self, value) -> None:
        self._set("result", value)
