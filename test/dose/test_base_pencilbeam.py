"""Tests for guarantee-flag discipline on PencilBeamEngineAbstract subclasses.

Every *concrete* (fully-implemented, non-abstract) subclass of
PencilBeamEngineAbstract that lives inside the ``pyRadPlan`` package must
define (or inherit from a concrete parent) both class-level boolean flags:

* ``_guarantee_canonical``  – CSC columns are already in sorted, unique-row order
* ``_guarantee_nonzero``    – no structural-zero entries are ever written

This is checked at test time by discovering the complete subclass graph and
filtering to production classes only (module path starts with ``pyRadPlan``).
"""

import importlib
import pkgutil
import sys
from typing import Iterator

import pytest

import pyRadPlan.dose.engines


def _import_all_engine_submodules() -> None:
    """Walk every module under pyRadPlan.dose.engines and import it.

    This ensures that all engine classes are defined and therefore show up
    in the ``__subclasses__()`` graph before we collect them.
    """
    pkg = importlib.import_module("pyRadPlan.dose.engines")
    for _finder, name, _ispkg in pkgutil.walk_packages(pkg.__path__, prefix=pkg.__name__ + "."):
        if name not in sys.modules:
            try:
                importlib.import_module(name)
            except Exception:  # noqa: BLE001  – skip engines with missing optional deps
                pass


def _iter_subclasses(cls) -> Iterator[type]:
    """Yield all transitive subclasses of *cls*."""
    for sub in cls.__subclasses__():
        yield sub
        yield from _iter_subclasses(sub)


def _pyradplan_pb_engines():
    """Return all PencilBeamEngineAbstract subclasses in pyRadPlan."""
    from pyRadPlan.dose.engines._base_pencilbeam import PencilBeamEngineAbstract

    _import_all_engine_submodules()

    result = []
    for cls in _iter_subclasses(PencilBeamEngineAbstract):
        # Skip abstract classes (they still have unresolved abstract methods)
        if getattr(cls, "__abstractmethods__", frozenset()):
            continue
        # Skip anything not defined in the pyRadPlan package (e.g. test helpers)
        if not getattr(cls, "__module__", "").startswith("pyRadPlan"):
            continue
        result.append(cls)
    return result


@pytest.mark.parametrize(
    "engine_cls",
    _pyradplan_pb_engines(),
    ids=lambda c: c.__qualname__,
)
class TestGuaranteeFlagsPresent:
    """Every discovered engine must carry both guarantee flags."""

    def test_guarantee_canonical_defined(self, engine_cls):
        assert hasattr(engine_cls, "_dij_guarantee_canonical"), (
            f"{engine_cls.__qualname__} (module: {engine_cls.__module__}) "
            "is missing the class variable '_dij_guarantee_canonical'. "
            "See PencilBeamEngineAbstract for the expected semantics."
        )

    def test_guarantee_canonical_is_bool(self, engine_cls):
        val = engine_cls._dij_guarantee_canonical
        assert isinstance(val, bool), (
            f"{engine_cls.__qualname__}._dij_guarantee_canonical must be a plain bool, "
            f"got {type(val)!r}"
        )

    def test_guarantee_nonzero_defined(self, engine_cls):
        assert hasattr(engine_cls, "_dij_guarantee_nonzero"), (
            f"{engine_cls.__qualname__} (module: {engine_cls.__module__}) "
            "is missing the class variable '_dij_guarantee_nonzero'. "
            "See PencilBeamEngineAbstract for the expected semantics."
        )

    def test_guarantee_nonzero_is_bool(self, engine_cls):
        val = engine_cls._dij_guarantee_nonzero
        assert isinstance(val, bool), (
            f"{engine_cls.__qualname__}._dij_guarantee_nonzero must be a plain bool, got {type(val)!r}"
        )
