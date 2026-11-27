"""Helper utilities for DICOM structure import.

The structure-naming heuristics are format-neutral and shared with the binary
importer; they live in :mod:`pyRadPlan.io._helpers` and are re-exported here so
existing ``from ._helpers import ...`` imports keep working.
"""

import os

from pydicom.misc import is_dicom

from .._helpers import (
    RT_ROI_INTERPRETED_TYPES,
    determine_structure_type,
    generate_colors,
    roi_interpreted_type,
    voi_type_from_interpreted_type,
)

__all__ = [
    "find_dicom_files",
    "RT_ROI_INTERPRETED_TYPES",
    "determine_structure_type",
    "generate_colors",
    "roi_interpreted_type",
    "voi_type_from_interpreted_type",
]


def find_dicom_files(directory) -> list[str]:
    """
    List the DICOM files at the top level of a directory.

    A file counts as DICOM if its extension is ``.dcm`` (case-insensitive) or if
    :func:`pydicom.misc.is_dicom` recognizes it. Subdirectories are not searched.

    Parameters
    ----------
    directory : str or os.PathLike
        The directory to inspect.

    Returns
    -------
    list[str]
        Sorted paths of the DICOM files; empty if the directory does not exist.
    """
    directory = os.fspath(directory)
    if not os.path.isdir(directory):
        return []
    found = []
    for entry in sorted(os.listdir(directory)):
        path = os.path.join(directory, entry)
        if not os.path.isfile(path):
            continue
        if entry.lower().endswith(".dcm"):
            found.append(path)
            continue
        try:
            if is_dicom(path):
                found.append(path)
        except OSError:
            continue
    return found
