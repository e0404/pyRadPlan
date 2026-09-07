"""Export a dose distribution as a DICOM RTDOSE file."""

import os

import numpy as np
import SimpleITK as sitk
from pydicom.dataset import Dataset
from pydicom.tag import Tag
from pydicom.uid import generate_uid

from ._export_common import (
    RT_DOSE_STORAGE,
    UIDContext,
    populate_common,
    direction_to_orientation,
)

_UINT32_MAX = 2**32 - 1


def export_dose(dose: sitk.Image, directory: str, ctx: UIDContext) -> str:
    """
    Write a dose distribution as a multi-frame DICOM RTDOSE file.

    Parameters
    ----------
    dose : sitk.Image
        The dose distribution (in Gy).
    directory : str
        Output directory.
    ctx : UIDContext
        Shared identifiers.

    Returns
    -------
    str
        The path of the written RTDOSE file.
    """
    arr = sitk.GetArrayFromImage(dose).astype(np.float64)  # (z, y, x)
    nz, ny, nx = arr.shape
    spacing = dose.GetSpacing()
    orientation = direction_to_orientation(dose.GetDirection())
    position = dose.TransformIndexToPhysicalPoint((0, 0, 0))

    max_value = float(arr.max()) if arr.size else 0.0
    if max_value > 0:
        scaling = max_value / _UINT32_MAX
        # Divide in float64; clip guards against float rounding overflowing uint32.
        pixels = np.clip(np.round(arr / scaling), 0, _UINT32_MAX).astype("<u4")
    else:
        scaling = 1.0
        pixels = np.zeros_like(arr, dtype="<u4")

    ds = Dataset()
    populate_common(ds, ctx, RT_DOSE_STORAGE, generate_uid(), "RTDOSE", 2)

    ds.InstanceNumber = 1
    ds.ImagePositionPatient = [float(p) for p in position]
    ds.ImageOrientationPatient = [float(o) for o in orientation]
    ds.PixelSpacing = [float(spacing[1]), float(spacing[0])]  # row (y), column (x)
    ds.SliceThickness = float(spacing[2])

    ds.Rows = ny
    ds.Columns = nx
    ds.NumberOfFrames = nz
    ds.FrameIncrementPointer = Tag(0x3004, 0x000C)  # -> GridFrameOffsetVector
    # Assumes frames advance along +z by the z-spacing (axial, identity/positive
    # z-direction). ImagePositionPatient above is direction-aware; a flipped or
    # oblique z-direction would need the offsets projected onto the slice normal.
    ds.GridFrameOffsetVector = [float(k * spacing[2]) for k in range(nz)]

    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 32
    ds.BitsStored = 32
    ds.HighBit = 31
    ds.PixelRepresentation = 0

    ds.DoseUnits = "GY"
    ds.DoseType = "PHYSICAL"
    ds.DoseSummationType = "PLAN"
    ds.DoseGridScaling = scaling

    ds.PixelData = pixels.tobytes()

    out_path = os.path.join(directory, "RTDOSE.dcm")
    ds.save_as(out_path, enforce_file_format=True)
    return out_path
