"""Data input/output and file handling.

This package provides a small, extensible import/export framework.

Top-level API
-------------
- :func:`load_patient` -- load a CT and StructureSet from a file or DICOM folder.
- :func:`load_data` -- load everything available (ct, cst, dose, ...) into a dict.
- :func:`save_data` -- save pyRadPlan objects, format chosen from extension/argument.
- :func:`load_tg119` -- load the bundled TG119 phantom.

Low-level handlers
------------------
Per-format handlers (:class:`MatlabHandler`, :class:`DicomHandler`, :class:`NpzHandler`,
:class:`PickleHandler`, :class:`NiftiHandler`, :class:`NrrdHandler`,
:class:`MetaImageHandler`) bundle an importer and exporter and can be used directly::

    handler = DicomHandler(path)
    ct = handler.load_ct()
    cst = handler.load_cst(ct)
    handler.save(ct=ct, cst=cst)

The individual ``*Importer`` / ``*Exporter`` classes are available from the backend
submodules (e.g. ``from pyRadPlan.io.dicom import DicomExporter``).
"""

from .base import BaseImporter, BaseExporter
from .matlab import MatlabHandler, validate_matrad_patient
from .dicom import DicomHandler
from .npz import NpzHandler
from .pickle import PickleHandler
from .sitk_based import (
    NiftiHandler,
    NrrdHandler,
    MetaImageHandler,
    load_binary_patient,
    list_image_files,
)
from ._factory import (
    register_importer,
    register_exporter,
    get_importer,
    get_exporter,
    get_available_formats,
)
from ._load_save import load_patient, load_data, save_data, load_tg119

__all__ = [
    "load_patient",
    "load_data",
    "save_data",
    "load_tg119",
    "load_binary_patient",
    "list_image_files",
    "BaseImporter",
    "BaseExporter",
    "MatlabHandler",
    "DicomHandler",
    "NpzHandler",
    "PickleHandler",
    "NiftiHandler",
    "NrrdHandler",
    "MetaImageHandler",
    "validate_matrad_patient",
    "register_importer",
    "register_exporter",
    "get_importer",
    "get_exporter",
    "get_available_formats",
]
