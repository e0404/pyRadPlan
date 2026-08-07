"""Registry and factory for import/export backends.

Backends self-register their importer and/or exporter from their package
``__init__`` (see :mod:`pyRadPlan.io.matlab` / :mod:`pyRadPlan.io.dicom`), mirroring
the dose-engine factory. This module therefore imports no backend and resolves
formats purely from the registered classes.
"""

import os
import warnings
import logging
from typing import Optional, Type

from .base import BaseImporter, BaseExporter

logger = logging.getLogger(__name__)

#: Registered importer/exporter classes, keyed by format (e.g. ``"mat"``, ``"dcm"``).
IMPORTERS: dict[str, Type[BaseImporter]] = {}
EXPORTERS: dict[str, Type[BaseExporter]] = {}

#: Map of file extensions (lower case, with dot) to format keys, built on registration.
_EXTENSION_MAP: dict[str, str] = {}

#: Default format used by :func:`save_data` when none can be inferred.
#: Change this single constant to switch the default export format.
DEFAULT_SAVE_FORMAT = "mat"


def _register(cls, registry: dict, base: type, role: str) -> None:
    """Validate and register an importer/exporter class into ``registry``."""
    if not issubclass(cls, base):
        raise ValueError(f"{role} must be a subclass of {base.__name__}.")
    if cls.format is None:
        raise ValueError(f"{role} must define a 'format' attribute.")
    if not cls.extensions:
        raise ValueError(f"{role} must define a non-empty 'extensions' attribute.")

    if cls.format in registry:
        warnings.warn(
            f"{role} for format '{cls.format}' is already registered as "
            f"{registry[cls.format].__name__}; keeping it and ignoring {cls.__name__}."
        )
        return

    registry[cls.format] = cls
    for ext in cls.extensions:
        _EXTENSION_MAP[ext.lower()] = cls.format


def register_importer(importer_cls: Type[BaseImporter]) -> None:
    """Register a new importer class (keyed by its ``format``)."""
    _register(importer_cls, IMPORTERS, BaseImporter, "Importer")


def register_exporter(exporter_cls: Type[BaseExporter]) -> None:
    """Register a new exporter class (keyed by its ``format``)."""
    _register(exporter_cls, EXPORTERS, BaseExporter, "Exporter")


def get_importer(fmt: str) -> Type[BaseImporter]:
    """Return the importer class for a format key."""
    try:
        return IMPORTERS[fmt]
    except KeyError:
        raise ValueError(
            f"No importer for format {fmt!r}. Available: {sorted(IMPORTERS)}."
        ) from None


def get_exporter(fmt: str) -> Type[BaseExporter]:
    """Return the exporter class for a format key."""
    try:
        return EXPORTERS[fmt]
    except KeyError:
        raise ValueError(
            f"No exporter for format {fmt!r}. Available: {sorted(EXPORTERS)}."
        ) from None


def get_available_formats() -> set[str]:
    """Return the set of formats with a registered importer or exporter."""
    return set(IMPORTERS) | set(EXPORTERS)


def is_container_format(fmt: str) -> bool:
    """Return True if the format stores multiple objects in a single file."""
    return get_exporter(fmt).container


def default_extension(fmt: str) -> str:
    """Return the default (first) file extension for a format."""
    return get_exporter(fmt).extensions[0]


def format_for_extension(ext: str) -> Optional[str]:
    """Return the format key registered for a file extension (with dot), or None."""
    return _EXTENSION_MAP.get(ext.lower())


def format_for_path(path: os.PathLike) -> Optional[str]:
    """Return the format whose registered extension is the longest matching suffix.

    Handles compound extensions such as ``.nii.gz`` (which ``os.path.splitext``
    would split as ``.gz``). Returns ``None`` if nothing matches.
    """
    name = os.fspath(path).lower()
    best_ext = None
    for ext in _EXTENSION_MAP:
        if name.endswith(ext) and (best_ext is None or len(ext) > len(best_ext)):
            best_ext = ext
    return _EXTENSION_MAP[best_ext] if best_ext is not None else None


def detect_format(path: os.PathLike) -> str:
    """
    Detect the format key for a given path.

    Parameters
    ----------
    path : os.PathLike
        A file or directory.

    Returns
    -------
    str
        The detected format key (e.g. ``"mat"`` or ``"dcm"``).
    """
    path = os.fspath(path)

    if os.path.isdir(path):
        for fmt, importer_cls in IMPORTERS.items():
            if importer_cls.handles_directory(path):
                return fmt
        raise ValueError(f"Could not detect a supported format in directory: {path}")

    fmt = format_for_path(path)
    if fmt is None:
        raise ValueError(f"Unsupported file extension: {path!r}")
    return fmt
