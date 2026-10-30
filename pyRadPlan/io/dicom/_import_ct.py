"""Import a CT image from a DICOM series."""

import logging
import warnings
from typing import Optional

import numpy as np
import pydicom
import SimpleITK as sitk

from pyRadPlan.core import ProgressReporter
from pyRadPlan.ct import validate_ct, CT

from ._helpers import warn_if_not_equally_spaced

logger = logging.getLogger(__name__)


def import_ct(
    directory: str, ct_files: list[str], reporter: Optional[ProgressReporter] = None
) -> CT:
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
    reporter : ProgressReporter, optional
        Reporter used to publish the slice-reading progress. Defaults to a
        private reporter, which still reaches context-scoped observers.

    Returns
    -------
    CT
        The imported CT object.
    """
    if not ct_files:
        raise ValueError("No CT slices provided.")

    reporter = reporter if reporter is not None else ProgressReporter()
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
    reader.MetaDataDictionaryArrayUpdateOn()

    num_slices = len(file_names)
    logger.info("Reading %d CT slice(s) from %s.", num_slices, directory)
    with reporter.progress("Reading CT slices", total=num_slices) as step:
        # ITK reports a 0..1 fraction while the series is being read.
        reader.AddCommand(
            sitk.sitkProgressEvent,
            lambda: step.update(int(reader.GetProgress() * num_slices)),
        )
        image = reader.Execute()
        step.update(num_slices)

    warn_if_not_equally_spaced(_slice_positions(reader, num_slices), "CT slices")

    image = sitk.Cast(image, sitk.sitkFloat32)

    return validate_ct(cube_hu=image)


def _slice_positions(reader: sitk.ImageSeriesReader, num_slices: int) -> list[float]:
    """Return the slice positions (mm) along the slice normal from the read headers."""
    try:
        orientation = [float(v) for v in reader.GetMetaData(0, "0020|0037").split("\\")]
        normal = np.cross(orientation[:3], orientation[3:])
        return [
            float(
                np.dot([float(v) for v in reader.GetMetaData(i, "0020|0032").split("\\")], normal)
            )
            for i in range(num_slices)
        ]
    except (RuntimeError, ValueError):
        # Missing or malformed ImagePositionPatient / ImageOrientationPatient
        return []
