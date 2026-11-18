"""Export a pyRadPlan CT as a DICOM CT image series."""

import os
from typing import Optional

import numpy as np
import SimpleITK as sitk
from pydicom.dataset import Dataset
from pydicom.uid import generate_uid

from pyRadPlan.ct import CT

from ._export_common import (
    CT_IMAGE_STORAGE,
    ExportContext,
    populate_common,
    direction_to_orientation,
    ds_list,
)


def export_ct(ct: CT, directory: str, ctx: ExportContext) -> dict:
    """
    Write a CT as a DICOM series into ``directory``.

    A 4D CT is written as one series per phase (``CT_p01_0001.dcm``, ...). The series
    share the study and the frame of reference, have their own Series Instance UIDs and
    a Series Description naming the phase, and carry the Temporal Position Identifier
    (1-based phase) and the Number of Temporal Positions.

    Parameters
    ----------
    ct : CT
        The CT to export.
    directory : str
        Output directory.
    ctx : ExportContext
        Resolved export context (study, frame of reference, patient).

    Returns
    -------
    dict
        Information for cross-referencing: ``series_uid`` and the per-slice
        ``sop_instance_uids`` of the (first) series, and ``phases``, the same
        information for every phase.
    """
    image = ct.cube_hu
    if image.GetDimension() == 4:
        num_phases = image.GetSize()[3]
        phases = [
            _export_series(
                image[:, :, :, p],
                directory,
                ctx,
                f"CT_p{p + 1:02d}_{{:04d}}.dcm",
                f"pyRadPlan CT phase {p + 1}/{num_phases} - research use only",
                p + 1,
                (p + 1, num_phases),
            )
            for p in range(num_phases)
        ]
    else:
        phases = [
            _export_series(
                image,
                directory,
                ctx,
                "CT_{:04d}.dcm",
                "pyRadPlan CT export - research use only",
                1,
            )
        ]
    return {**phases[0], "phases": phases}


def _export_series(  # noqa: PLR0913 - one series needs its naming and numbering
    image: sitk.Image,
    directory: str,
    ctx: ExportContext,
    file_pattern: str,
    description: str,
    series_number: int,
    temporal: Optional[tuple[int, int]] = None,
) -> dict:
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
        sop_uid = populate_common(
            ds, ctx, CT_IMAGE_STORAGE, series_uid, "CT", series_number, description
        )
        sop_instance_uids.append(sop_uid)

        ds.ContentDate = ctx.export_date
        ds.ContentTime = ctx.export_time
        ds.PatientPosition = ctx.patient_position
        ds.KVP = ctx.kvp
        ds.ImageType = ["DERIVED", "SECONDARY", "AXIAL"]
        ds.AcquisitionNumber = series_number
        ds.InstanceNumber = z + 1
        if temporal is not None:
            ds.TemporalPositionIdentifier = temporal[0]
            ds.NumberOfTemporalPositions = temporal[1]
        ds.SliceLocation = ds_list([position[2]])[0]
        ds.ImagePositionPatient = ds_list(position)
        ds.ImageOrientationPatient = ds_list(orientation)
        ds.PixelSpacing = ds_list([spacing[1], spacing[0]])  # row (y), column (x)
        ds.SliceThickness = ds_list([spacing[2]])[0]

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

        ds.save_as(os.path.join(directory, file_pattern.format(z + 1)), enforce_file_format=True)

    return {"series_uid": series_uid, "sop_instance_uids": sop_instance_uids}
