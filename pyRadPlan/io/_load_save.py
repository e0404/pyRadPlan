"""Top-level import/export API: ``load_patient``, ``load_data`` and ``save_data``."""

import os
import warnings
import logging
from typing import Any, Optional, Union

import SimpleITK as sitk

from pyRadPlan.core import BaseDataFile, get_base_data_kind, list_base_data
from pyRadPlan.ct import CT
from pyRadPlan.cst import StructureSet

from .matlab import validate_matrad_patient
from ._factory import (
    DEFAULT_SAVE_FORMAT,
    detect_format,
    get_importer,
    get_exporter,
    get_available_formats,
    is_container_format,
    default_extension,
    format_for_path,
)

logger = logging.getLogger(__name__)

_FORMAT_ALIASES = {"matlab": "mat", "dicom": "dcm"}


def _normalize_format(fmt: str) -> str:
    """Normalize a user-supplied format string to a registry key."""
    key = fmt.lower().lstrip(".")
    key = _FORMAT_ALIASES.get(key, key)
    if key not in get_available_formats():
        raise ValueError(f"Unsupported format: {fmt!r}")
    return key


def list_phantoms() -> list[BaseDataFile]:
    """Return the phantom files, shipped ones first, then the user's and the shared ones.

    Phantoms are the ``.mat`` (matRad) and ``.npz`` files in the ``phantoms`` folders of
    the base data (see :mod:`pyRadPlan.core._base_data`): ``pyRadPlan/data/phantoms``,
    ``<data_dir>/phantoms`` and ``<root>/phantoms`` for each root in
    ``pyRadPlan.settings.base_data_path``. A phantom is named by its file name without
    the ending; a file named like one listed before it is skipped, whatever its ending.
    """
    phantoms: list[BaseDataFile] = []
    seen: set[str] = set()
    for entry in list_base_data("phantoms"):
        stem = os.path.splitext(entry.name)[0].casefold()
        if stem not in seen:
            seen.add(stem)
            phantoms.append(entry)
    return phantoms


def available_phantoms() -> list[str]:
    """Names of the available phantoms (see :func:`list_phantoms`), e.g. ``["TG119"]``."""
    return [os.path.splitext(entry.name)[0] for entry in list_phantoms()]


def phantom_path(name: str) -> Any:
    """
    Resolve a phantom by name (see :func:`list_phantoms`).

    Parameters
    ----------
    name : str
        Phantom name, case-insensitive, with or without its file ending
        (``"TG119"``, ``"tg119.mat"``).

    Returns
    -------
    Traversable
        Path-like object of the phantom file, accepted by :func:`load_patient` and
        :func:`load_data`.

    Raises
    ------
    FileNotFoundError
        If no phantom has that name.
    """
    stem, ending = os.path.splitext(name)
    if ending.lower() not in get_base_data_kind("phantoms").endings:
        stem = name
    phantoms = list_phantoms()
    for entry in phantoms:
        if os.path.splitext(entry.name)[0].casefold() == stem.casefold():
            return entry.path
    names = ", ".join(os.path.splitext(entry.name)[0] for entry in phantoms)
    raise FileNotFoundError(f"No phantom named {name!r}; available: {names}")


def load_tg119() -> tuple[CT, StructureSet]:
    """
    Load the included TG119 phantom.

    Returns
    -------
    tuple[CT, StructureSet]
        The CT and StructureSet objects.
    """
    return load_patient(phantom_path("TG119"))


def load_patient(
    filename: os.PathLike,
    extra_plan_data: Optional[dict] = None,
    extra_data: Optional[dict] = None,
) -> tuple[CT, StructureSet]:
    """
    Load a patient (CT and StructureSet) from a file or DICOM folder.

    The format is chosen automatically from the path.

    Parameters
    ----------
    filename : os.PathLike
        Path to the file or DICOM directory.
    extra_plan_data : Optional[dict]
        If provided, updated with additional validated pyRadPlan structures found
        in the file (e.g. ``pln``, ``stf``, ``dij``, ``result``). Only populated
        for matRad ``.mat`` files.
    extra_data : Optional[dict]
        If provided, updated with the raw leftover data found in the file. Only
        populated for matRad ``.mat`` files.

    Returns
    -------
    tuple[CT, StructureSet]
        The CT and StructureSet objects.
    """
    path = os.path.normpath(os.fspath(filename))
    if not os.path.exists(path):
        raise FileNotFoundError(f"Patient file not found: {path}")

    fmt = detect_format(path)
    importer = get_importer(fmt)(path)

    with importer.progress("Importing patient", total=2) as step:
        ct = importer.load_ct()
        if ct is None:
            raise ValueError("ct is missing from the patient file.")
        step.advance()

        cst = importer.load_cst(ct)
        if cst is None:
            warnings.warn("cst/StructureSet is missing from the patient file.")
        step.advance()

    if (isinstance(extra_plan_data, dict) or isinstance(extra_data, dict)) and fmt == "mat":
        mdict = dict(importer.mdict)
        patient_dict = validate_matrad_patient(mdict, remove_matrad_structures=True)
        patient_dict.pop("ct", None)
        patient_dict.pop("cst", None)
        if isinstance(extra_plan_data, dict):
            extra_plan_data.update(patient_dict)
        if isinstance(extra_data, dict):
            extra_data.update(mdict)

    return ct, cst


def load_data(path: os.PathLike) -> dict:
    """
    Load everything available from a file or DICOM folder.

    Parameters
    ----------
    path : os.PathLike
        Path to the file or DICOM directory.

    Returns
    -------
    dict
        A dictionary containing the loaded objects. May contain ``"ct"``,
        ``"cst"``, ``"dose"`` and any further structures the backend recognizes.
    """
    p = os.path.normpath(os.fspath(path))
    if not os.path.exists(p):
        raise FileNotFoundError(f"File not found: {p}")

    fmt = detect_format(p)
    importer = get_importer(fmt)(p)
    return importer.load_data()


def _collect_objects(data, ct, cst, dose, extra) -> dict:
    """Gather the named objects to save into a single dict."""
    objects: dict = {}
    if data is not None:
        if not isinstance(data, dict):
            raise TypeError("'data' must be a dict of named objects.")
        objects.update(data)
    for key, value in (("ct", ct), ("cst", cst), ("dose", dose)):
        if value is not None:
            objects[key] = value
    objects.update({k: v for k, v in extra.items() if v is not None})
    if not objects:
        raise ValueError("Nothing to save: provide ct, cst, dose, extra objects or a data dict.")
    return objects


def _resolve_format(file_name, fmt) -> str:
    """Resolve the export format key from an explicit value or the file extension."""
    if fmt is not None:
        return _normalize_format(fmt)
    if file_name is not None:
        path = os.fspath(file_name)
        if os.path.splitext(path)[1]:
            resolved = format_for_path(path)
            if resolved is None:
                raise ValueError(f"Unsupported file extension: {path!r}")
            return resolved
    return DEFAULT_SAVE_FORMAT


def save_data(  # noqa: PLR0913 - convenience keyword arguments are intentional
    data: Optional[dict] = None,
    *,
    file_name: Optional[Union[str, os.PathLike]] = None,
    format: Optional[str] = None,  # noqa: A002 - intentional public API name
    ct: Optional[CT] = None,
    cst: Optional[StructureSet] = None,
    dose: Optional[sitk.Image] = None,
    **extra,
) -> Union[str, list[str]]:
    """
    Save pyRadPlan objects to disk.

    The export format is resolved in this order: explicit ``format`` argument,
    then the extension of ``file_name``, then :data:`DEFAULT_SAVE_FORMAT`.

    Parameters
    ----------
    data : dict, optional
        A dictionary of named objects to save (e.g. ``{"ct": ct, "cst": cst}``).
    file_name : str or os.PathLike, optional
        Target file name. If it has no extension, the format's default extension
        is appended. For DICOM, this is treated as the output directory. If
        omitted, each object is written to ``<name>.<ext>`` (e.g. ``ct.mat``).
    format : str, optional
        Explicit format (e.g. ``"mat"``, ``"dcm"``).
    ct, cst : optional
        Convenience keyword arguments for the most common objects.
    dose : sitk.Image, optional
        A dose distribution to save.
    **extra
        Further named objects passed through to the exporter.

    Returns
    -------
    str or list[str]
        The path(s) written.
    """
    objects = _collect_objects(data, ct, cst, dose, extra)
    fmt = _resolve_format(file_name, format)

    ext = default_extension(fmt)
    exporter_cls = get_exporter(fmt)

    # DICOM is directory-based and always writes the full multi-file set together.
    if not is_container_format(fmt):
        target = os.fspath(file_name) if file_name is not None else os.getcwd()
        exporter_cls(target).save(**objects)
        return target

    # Container formats (e.g. .mat): one combined file when a name is given,
    # otherwise one file per object named after its key.
    if file_name is not None:
        target = os.fspath(file_name)
        if not os.path.splitext(target)[1]:
            target += ext
        exporter_cls(target).save(**objects)
        return target

    written = []
    for key, value in objects.items():
        target = f"{key}{ext}"
        exporter_cls(target).save(**{key: value})
        written.append(target)
    return written
