"""Import a CT image from a DICOM series."""

import logging
import os
import warnings
from typing import Optional

import pydicom
import SimpleITK as sitk

from pyRadPlan.core import ProgressReporter
from pyRadPlan.ct import validate_ct, CT, DicomInfo

logger = logging.getLogger(__name__)


def import_ct(
    directory: str,
    ct_files: list[str],
    reporter: Optional[ProgressReporter] = None,
    headers: Optional[dict[str, pydicom.Dataset]] = None,
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
    headers : dict[str, pydicom.Dataset], optional
        Pixel-free headers already read, keyed by file path. Slices missing
        here are read from disk.

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

    image = sitk.Cast(image, sitk.sitkFloat32)

    cached = {os.path.normcase(os.path.normpath(p)): ds for p, ds in (headers or {}).items()}

    def _header(path: str) -> pydicom.Dataset:
        ds = cached.get(os.path.normcase(os.path.normpath(path)))
        return ds if ds is not None else pydicom.dcmread(path, stop_before_pixels=True)

    slice_headers = [_header(f) for f in file_names]
    dicom_info = DicomInfo.model_validate(slice_headers[0])
    dicom_info.sop_instance_uids = [str(ds.SOPInstanceUID) for ds in slice_headers]

    return validate_ct(cube_hu=image, dicom_info=dicom_info)
