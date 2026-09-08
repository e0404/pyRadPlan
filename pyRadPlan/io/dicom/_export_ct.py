"""Export a pyRadPlan CT as a DICOM CT image series."""

import os

import numpy as np
import SimpleITK as sitk
from pydicom.dataset import Dataset
from pydicom.uid import generate_uid

from pyRadPlan.ct import CT

from ._export_common import (
    CT_IMAGE_STORAGE,
    UIDContext,
    populate_common,
    direction_to_orientation,
)


def export_ct(ct: CT, directory: str, ctx: UIDContext) -> dict:
    """
    Write a CT as a DICOM series into ``directory``.

    Parameters
    ----------
    ct : CT
        The CT to export.
    directory : str
        Output directory.
    ctx : UIDContext
        Shared identifiers (study, frame of reference, patient).

    Returns
    -------
    dict
        Information for cross-referencing: ``series_uid`` and the per-slice
        ``sop_instance_uids``.
    """
    image = ct.cube_hu
    if image.GetDimension() == 4:
        image = image[:, :, :, 0]

    arr = sitk.GetArrayFromImage(image)  # (z, y, x)
    nz, ny, nx = arr.shape
    spacing = image.GetSpacing()
    orientation = direction_to_orientation(image.GetDirection())

    series_uid = generate_uid()
    sop_instance_uids = []

    arr_i16 = np.clip(np.round(arr), -32768, 32767).astype(np.int16)

    for z in range(nz):
        position = image.TransformIndexToPhysicalPoint((0, 0, int(z)))

        ds = Dataset()
        sop_uid = populate_common(ds, ctx, CT_IMAGE_STORAGE, series_uid, "CT", 1)
        sop_instance_uids.append(sop_uid)

        ds.InstanceNumber = z + 1
        ds.ImagePositionPatient = [float(p) for p in position]
        ds.ImageOrientationPatient = [float(o) for o in orientation]
        ds.PixelSpacing = [float(spacing[1]), float(spacing[0])]  # row (y), column (x)
        ds.SliceThickness = float(spacing[2])

        ds.Rows = ny
        ds.Columns = nx
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 1  # signed
        ds.RescaleIntercept = 0
        ds.RescaleSlope = 1
        ds.RescaleType = "HU"

        ds.PixelData = arr_i16[z].tobytes()

        ds.save_as(os.path.join(directory, f"CT_{z + 1:04d}.dcm"), enforce_file_format=True)

    return {"series_uid": series_uid, "sop_instance_uids": sop_instance_uids}
