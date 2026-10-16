"""
Radiobiology: biological models and the tissue properties they evaluate.

Subpackages
-----------
models
    Biological models (RBE and LQ formalisms), their evaluators and registry.
tissues
    Radiobiological tissue properties, provenances and cell line presets.

The subpackages are imported on first access: ``tissues`` is a leaf imported by the structure
set and the machines, while ``models`` builds on both, so importing it here would be circular.
"""

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import models, tissues

_SUBPACKAGES = ("models", "tissues")

__all__ = ["models", "tissues"]


def __getattr__(name: str):
    if name in _SUBPACKAGES:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_SUBPACKAGES))
