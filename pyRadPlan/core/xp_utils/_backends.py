"""Deferred loading of the array backends that xp_utils treats specially.

pyRadPlan works with any Array API namespace. A few backends get extra treatment
(devices, streams, sparse matrices, GPU detection, jitting), but importing them takes
seconds, mostly for libraries that are never used in a session. xp_utils therefore
imports a backend only when it selects or probes it (:func:`import_backend`). Code
handling arrays looks up the backends that are imported already
(:func:`loaded_backend`): no array of a backend that was never imported can exist.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from types import ModuleType
from typing import Optional

logger = logging.getLogger(__name__)

#: Array backends with special treatment in xp_utils, by module name, with their display
#: names. Other Array API namespaces work through the generic code paths.
BACKENDS = {"cupy": "CuPy", "torch": "PyTorch", "jax": "JAX"}

_import_observers: list[Callable[[str], None]] = []


def backend_installed(name: str) -> bool:
    """Return whether the top-level module *name* can be found, without importing it."""
    if sys.modules.get(name) is not None:
        return True
    try:
        return importlib.util.find_spec(name) is not None
    except ValueError:  # in sys.modules without a spec
        return False


def import_backend(name: str) -> Optional[ModuleType]:
    """Import the top-level module *name*, or return ``None`` if it cannot be imported.

    Observers registered with :func:`observe_backend_imports` are told about the import
    before it starts. An import running in another thread is waited for.
    """
    if name not in sys.modules:
        if importlib.util.find_spec(name) is None:
            return None
        for observer in list(_import_observers):
            observer(name)
    try:
        return importlib.import_module(name)
    except ImportError:
        logger.debug("Array backend %s is installed but cannot be imported", name, exc_info=True)
        return None


def loaded_backend(name: str, backend: Optional[str] = None) -> Optional[ModuleType]:
    """Return the module *name* if its backend is imported already, else ``None``.

    Never imports a backend, only modules belonging to an imported one, e.g.
    ``"jax.experimental.sparse"`` or ``loaded_backend("cupyx.scipy.fft", "cupy")``.
    An import running in another thread is waited for.

    Parameters
    ----------
    name:
        Module to return.
    backend:
        Top-level module of the backend; defaults to the top-level package of *name*.
    """
    if (backend or name.partition(".")[0]) not in sys.modules:
        return None
    try:
        return importlib.import_module(name)
    except ImportError:
        return None


@contextmanager
def observe_backend_imports(observer: Callable[[str], None]) -> Iterator[None]:
    """Call *observer* with the module name before each :func:`import_backend` import."""
    _import_observers.append(observer)
    try:
        yield
    finally:
        _import_observers.remove(observer)
