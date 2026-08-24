"""Import a dose distribution from a DICOM RTDOSE file."""

import os
import re
import logging
import warnings
from typing import Union

import pydicom
import SimpleITK as sitk

logger = logging.getLogger(__name__)

#: ``DoseSummationType`` values that denote a per-beam (rather than plan-level)
#: distribution. matRad exports one such file per beam alongside the summed dose.
_BEAM_LEVEL_SUMMATION = {
    "BEAM",
    "BEAM_SESSION",
    "CONTROL_POINT",
    "CONTROL_POINT_SESSION",
}


def _dose_descriptor(ds: pydicom.Dataset) -> str:
    """Return a human-readable descriptor (comment/series description) for a dose."""
    return str(getattr(ds, "DoseComment", "") or getattr(ds, "SeriesDescription", "") or "")


def _is_let_descriptor(descriptor: str) -> bool:
    """Return True if the descriptor names an LET cube (token match, not substring)."""
    tokens = re.split(r"[^a-z]+", descriptor.lower())
    return "let" in tokens


def _select_dose_file(dose_files: list) -> Union[str, os.PathLike]:
    """Select the plan-level physical dose from several RTDOSE candidates.

    matRad exports the summed plan dose, one dose per beam and auxiliary cubes
    (e.g. LET) all with modality ``RTDOSE``. Prefer a plan-level distribution
    (``DoseSummationType`` not per-beam) and, among those, a physical dose over
    an auxiliary cube. Selection is deterministic (candidates sorted by name) and
    only warns when the choice is genuinely ambiguous.
    """
    infos = []
    for f in sorted(dose_files, key=lambda p: os.path.basename(os.fspath(p))):
        try:
            ds = pydicom.dcmread(f, stop_before_pixels=True)
        except Exception:  # noqa: BLE001 - unreadable files are simply skipped
            continue
        summation = str(getattr(ds, "DoseSummationType", "") or "").upper()
        infos.append((f, summation, _dose_descriptor(ds)))

    if not infos:
        # Nothing readable; fall back to the raw first entry.
        return dose_files[0]

    # Prefer plan-level distributions over per-beam ones.
    plan_level = [i for i in infos if i[1] not in _BEAM_LEVEL_SUMMATION]
    pool = plan_level or infos

    # Prefer physical dose over auxiliary cubes matRad also stores as RTDOSE (LET).
    physical = [i for i in pool if not _is_let_descriptor(i[2])]
    pool = physical or pool

    if len(pool) > 1:
        warnings.warn(
            f"Multiple candidate RTDOSE distributions found; using "
            f"'{os.path.basename(os.fspath(pool[0][0]))}'.",
            stacklevel=2,
        )
    return pool[0][0]


def import_dose(dose_files: Union[str, os.PathLike, list]) -> sitk.Image:
    """
    Read a DICOM RTDOSE file into a SimpleITK image (in Gy).

    The raw pixel values are multiplied by ``DoseGridScaling`` since the GDCM
    reader does not apply it automatically. If several RTDOSE files are passed,
    the plan-level physical dose is selected (see :func:`_select_dose_file`).

    Parameters
    ----------
    dose_files : str, os.PathLike or list
        Path to an RTDOSE file, or a list of such paths.

    Returns
    -------
    sitk.Image
        The dose distribution.
    """
    if isinstance(dose_files, (str, os.PathLike)):
        dose_files = [dose_files]
    dose_files = list(dose_files)

    if not dose_files:
        raise ValueError("No RTDOSE file provided.")

    dose_file = dose_files[0] if len(dose_files) == 1 else _select_dose_file(dose_files)

    image = sitk.ReadImage(dose_file)
    image = sitk.Cast(image, sitk.sitkFloat32)

    ds = pydicom.dcmread(dose_file, stop_before_pixels=True)
    scaling = float(getattr(ds, "DoseGridScaling", 1.0))
    if scaling != 1.0:
        image = image * scaling

    return image
