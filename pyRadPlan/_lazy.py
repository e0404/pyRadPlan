"""Lazy attribute access for package namespaces (PEP 562).

Kept free of third-party imports: :mod:`pyRadPlan` uses it before anything heavy is
loaded, so ``import pyRadPlan`` (and with it ``import pyRadPlan.gui``) stays cheap.
"""

import importlib
import importlib.util
import sys
from collections.abc import Callable, Iterable, Mapping
from typing import Any


def attach_lazy(
    package: str, imports: Mapping[str, Iterable[str]], eager: Iterable[str] = ()
) -> tuple[Callable[[str], Any], Callable[[], list[str]], list[str]]:
    """Make the names of a package's ``from <module> import ...`` statements lazy.

    Parameters
    ----------
    package:
        ``__name__`` of the calling package.
    imports:
        Maps each module, relative to *package* like in a ``from`` import, to the names
        that would be imported from it. A module is imported when one of its names is
        first accessed, and the name is then cached in the package namespace.
    eager:
        Public names the package defines itself, listed first in ``__all__``.

    Returns
    -------
    tuple
        The package's ``__getattr__``, ``__dir__`` and ``__all__``. Attributes that are
        no lazy name resolve to submodules of *package*, so ``pyRadPlan.dose`` works
        without importing it first.
    """
    origins = {name: module for module, names in imports.items() for name in names}
    namespace = sys.modules[package].__dict__

    def module_getattr(name: str) -> Any:
        module = origins.get(name)
        if module is None:
            if name.startswith("__") or importlib.util.find_spec(f"{package}.{name}") is None:
                raise AttributeError(f"module {package!r} has no attribute {name!r}")
            return importlib.import_module(f"{package}.{name}")
        source = importlib.import_module(module, package)
        try:
            value = getattr(source, name)
        except AttributeError:
            # ``from .core import xp_utils`` also imports submodules
            value = importlib.import_module(f"{source.__name__}.{name}")
        namespace[name] = value
        return value

    def module_dir() -> list[str]:
        return sorted({*namespace, *origins})

    return module_getattr, module_dir, [*eager, *origins]
