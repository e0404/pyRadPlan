"""Revision counters telling a frontend which products to refetch (SessionState v1)."""

from __future__ import annotations

import itertools
from typing import Any, Callable, Iterable

from ._results import result_components
from ._workspace import WORKSPACE_KEYS

_base_data_revision = itertools.count(1)
_BASE_DATA = {"revision": 0}


def bump_base_data_revision() -> int:
    """Count a change of the base data files (process-wide); returns the new revision."""
    _BASE_DATA["revision"] = next(_base_data_revision)
    return _BASE_DATA["revision"]


def base_data_revision() -> int:
    """Return the current process-wide base data revision."""
    return _BASE_DATA["revision"]


class Revisions:
    """Per-key and per-product counters, updated from the workspace notifications.

    ``workspace[key]`` counts the writes of each key; ``volumes[name]`` counts the
    replacements of the object a volume (``"ct"`` or a result component key) is built
    from; ``masks`` counts changes of the CT, the structure names or any mask; ``beams``
    counts changes of pln, stf or cst (the automatic isocenter).
    """

    def __init__(self, get: Callable[[str], Any]) -> None:
        self._get = get
        self.workspace: dict[str, int] = dict.fromkeys(WORKSPACE_KEYS, 0)
        self.volumes: dict[str, int] = {}
        self.masks = 0
        self.beams = 0
        self._volume_sources: dict[str, Any] = {}
        self._mask_sources: tuple = ()
        self._counter = itertools.count(1)
        self.update(WORKSPACE_KEYS, initial=True)

    def _sources(self) -> dict[str, Any]:
        sources = {}
        cube = getattr(self._get("ct"), "cube_hu", None)
        if cube is not None:
            sources["ct"] = cube
        sources.update(result_components(self._get("result")))
        return sources

    def update(self, changed_keys: Iterable[str], initial: bool = False) -> None:
        """Count a workspace change of *changed_keys* (all keys for an empty list)."""
        keys = set(changed_keys) or set(WORKSPACE_KEYS)
        if not initial:
            for key in keys & set(WORKSPACE_KEYS):
                self.workspace[key] += 1
        if keys & {"ct", "result"}:
            sources = self._sources()
            known = {id(src): self.volumes[n] for n, src in self._volume_sources.items()}
            volumes = {}
            for name, source in sources.items():
                # A snapshot copies the same object under a new key: same revision.
                volumes[name] = known.get(id(source)) or next(self._counter)
            self.volumes = volumes
            self._volume_sources = sources
        if keys & {"ct", "cst"}:
            cst = self._get("cst")
            vois = list(getattr(cst, "vois", None) or [])
            names = tuple(getattr(voi, "name", None) for voi in vois)
            objects = (self._get("ct"), *(getattr(voi, "mask", None) for voi in vois))
            previous_names, previous_objects = self._mask_sources or ((), ())
            if (
                names != previous_names
                or len(objects) != len(previous_objects)
                or any(a is not b for a, b in zip(objects, previous_objects))
            ):
                self.masks += 1
            self._mask_sources = (names, objects)
        if keys & {"pln", "stf", "cst"} and not initial:
            self.beams += 1

    def as_dict(self) -> dict[str, Any]:
        """Return the ``revisions`` part of SessionState v1."""
        return {
            "workspace": dict(self.workspace),
            "volumes": dict(self.volumes),
            "masks": self.masks,
            "beams": self.beams,
            "base_data": base_data_revision(),
        }
