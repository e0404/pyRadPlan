"""Import a CT image from a DICOM series."""

import logging
import warnings

import pydicom
import SimpleITK as sitk

from pyRadPlan.ct import validate_ct, CT

logger = logging.getLogger(__name__)


def import_ct(directory: str, ct_files: list[str]) -> CT:
    """
    Read a CT DICOM series into a pyRadPlan :class:`CT`.

    Uses :class:`SimpleITK.ImageSeriesReader`, which sorts the slices and sets
    origin, spacing and direction (orientation) from the DICOM headers.

    Parameters
    ----------
    directory : str
        Directory containing the DICOM files.
    ct_files : list[str]
        The CT image files found in the directory.

    Returns
    -------
    CT
        The imported CT object.
    """
    if not ct_files:
        raise ValueError("No CT slices provided.")

    reader = sitk.ImageSeriesReader()

    # Resolve the (sorted) file list for the CT series.
    series_uid = pydicom.dcmread(ct_files[0], stop_before_pixels=True).SeriesInstanceUID
    file_names = reader.GetGDCMSeriesFileNames(directory, series_uid)
    if not file_names:
        file_names = sorted(ct_files)
    elif len(file_names) < len(ct_files):
        warnings.warn(
            f"Multiple CT series found in {directory}; using the first "
            f"({len(file_names)} of {len(ct_files)} slices).",
            stacklevel=2,
        )

    reader.SetFileNames(file_names)
    image = reader.Execute()
    image = sitk.Cast(image, sitk.sitkFloat32)

    return validate_ct(cube_hu=image)
