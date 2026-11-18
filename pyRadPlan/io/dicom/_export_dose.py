"""Export a dose distribution as a DICOM RTDOSE file."""

import os
import warnings

import numpy as np
import SimpleITK as sitk
from typing import Optional

from pydicom.dataset import Dataset
from pydicom.tag import Tag
from pydicom.uid import generate_uid
from pydicom.valuerep import format_number_as_ds

from ._export_dvh import DvhExport, add_dvh_module
from ._export_common import (
    RT_DOSE_STORAGE,
    ExportContext,
    PlanReference,
    populate_common,
    direction_to_orientation,
)

_UINT32_MAX = 2**32 - 1


def _ds(values) -> list:
    """Quantize numbers to Decimal Strings (16 characters at most)."""
    return [float(format_number_as_ds(float(v))) for v in values]


def _frame_offsets(dose: sitk.Image, orientation: list) -> list:
    """Grid frame offsets: frame positions projected on the slice normal, relative to frame 0."""
    normal = np.cross(orientation[:3], orientation[3:])
    origin = np.array(dose.TransformIndexToPhysicalPoint((0, 0, 0)))
    return [
        float(np.dot(np.array(dose.TransformIndexToPhysicalPoint((0, 0, k))) - origin, normal))
        for k in range(dose.GetSize()[2])
    ]


def _referenced_plan_sequence(
    plan_ref: PlanReference, summation: str, beam_number: Optional[int]
) -> list:
    item = Dataset()
    item.ReferencedSOPClassUID = plan_ref.sop_class_uid
    item.ReferencedSOPInstanceUID = plan_ref.sop_instance_uid
    if summation != "PLAN":
        group = Dataset()
        group.ReferencedFractionGroupNumber = plan_ref.fraction_group_number
        if summation == "BEAM":
            beam = Dataset()
            beam.ReferencedBeamNumber = beam_number
            group.ReferencedBeamSequence = [beam]
        item.ReferencedFractionGroupSequence = [group]
    return [item]


def export_dose(  # noqa: PLR0913 - DICOM dose attributes
    dose: sitk.Image,
    directory: str,
    ctx: ExportContext,
    *,
    file_name: str = "RTDOSE.dcm",
    dose_type: str = "PHYSICAL",
    summation_type: str = "PLAN",
    comment: Optional[str] = None,
    series_number: int = 2,
    plan_ref: Optional[PlanReference] = None,
    beam_number: Optional[int] = None,
    dvh: Optional[DvhExport] = None,
) -> str:
    """
    Write a dose distribution as a multi-frame DICOM RTDOSE file.

    Parameters
    ----------
    dose : sitk.Image
        The dose distribution (in Gy). Flipped and oblique grids are supported.
    directory : str
        Output directory.
    ctx : ExportContext
        Shared identifiers.
    file_name : str
        Name of the written file.
    dose_type : str
        DICOM Dose Type, e.g. ``"PHYSICAL"`` or ``"EFFECTIVE"``.
    summation_type : str
        DICOM Dose Summation Type (``"PLAN"``, ``"FRACTION"`` or ``"BEAM"``).
    comment : str, optional
        Dose Comment; also used as the series description.
    series_number : int
        Series number.
    plan_ref : PlanReference, optional
        Plan to reference in the Referenced RT Plan Sequence.
    beam_number : int, optional
        Beam number of a ``"BEAM"`` dose (required with ``plan_ref``).
    dvh : DvhExport, optional
        Structures to write the RT DVH module for (one cumulative DVH per structure,
        computed from the dose as stored in the file).

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
        # DoseGridScaling is a DS (16 characters at most), so quantize it before use.
        scaling = float(format_number_as_ds(max_value / _UINT32_MAX))
        # Divide in float64; clip guards against float rounding overflowing uint32.
        pixels = np.clip(np.round(arr / scaling), 0, _UINT32_MAX).astype("<u4")
    else:
        scaling = 1.0
        pixels = np.zeros_like(arr, dtype="<u4")

    ds = Dataset()
    populate_common(
        ds,
        ctx,
        RT_DOSE_STORAGE,
        generate_uid(),
        "RTDOSE",
        series_number,
        comment or "pyRadPlan dose - research use only",
    )
    ds.OperatorsName = ""

    ds.InstanceNumber = 1
    ds.ImagePositionPatient = _ds(position)
    ds.ImageOrientationPatient = _ds(orientation)
    ds.PixelSpacing = _ds([spacing[1], spacing[0]])  # row (y), column (x)
    ds.SliceThickness = _ds([spacing[2]])[0]

    ds.Rows = ny
    ds.Columns = nx
    ds.NumberOfFrames = nz
    ds.FrameIncrementPointer = Tag(0x3004, 0x000C)  # -> GridFrameOffsetVector
    ds.GridFrameOffsetVector = _ds(_frame_offsets(dose, orientation))

    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 32
    ds.BitsStored = 32
    ds.HighBit = 31
    ds.PixelRepresentation = 0

    ds.DoseUnits = "GY"
    ds.DoseType = dose_type
    ds.DoseSummationType = summation_type
    if comment:
        ds.DoseComment = comment
    ds.DoseGridScaling = scaling
    if plan_ref is not None:
        ds.ReferencedRTPlanSequence = _referenced_plan_sequence(
            plan_ref, summation_type, beam_number
        )

    if dvh is not None:
        stored = sitk.GetImageFromArray(pixels.astype(np.float64) * scaling)
        stored.CopyInformation(dose)
        add_dvh_module(ds, stored, dose_type, dvh)

    ds.PixelData = pixels.tobytes()

    out_path = os.path.join(directory, file_name)
    ds.save_as(out_path, enforce_file_format=True)
    return out_path


#: Result key -> DICOM Dose Type of the dose quantities that are exported.
RESULT_DOSE_TYPES = {"physical_dose": "PHYSICAL", "rbe_x_dose": "EFFECTIVE"}


def _is_image_entry(value) -> bool:
    if isinstance(value, sitk.Image):
        return True
    return (
        isinstance(value, (list, tuple))
        and any(v is not None for v in value)
        and all(v is None or isinstance(v, sitk.Image) for v in value)
    )


def export_result(  # noqa: PLR0913 - export options
    result: dict,
    directory: str,
    ctx: ExportContext,
    *,
    summation_type: str = "PLAN",
    per_beam: bool = False,
    plan_ref: Optional[PlanReference] = None,
    dvh: Optional[DvhExport] = None,
) -> list[str]:
    """
    Write the dose quantities of a result dictionary as RTDOSE files.

    Physical dose is written with Dose Type ``PHYSICAL`` and RBE-weighted dose with
    ``EFFECTIVE`` as ``RTDOSE_<quantity>.dcm``. Other quantities (LET, effect, alpha, ...)
    are skipped with a warning, as are scenario results beyond the nominal scenario (the
    unsuffixed entries). With ``per_beam``, the ``<quantity>_beam`` lists are written as
    ``RTDOSE_<quantity>_beam<N>.dcm`` with Dose Summation Type ``BEAM``. With ``dvh``, every
    file also gets the RT DVH module.

    Returns
    -------
    list of str
        The paths of the written files.
    """
    names = {k: v for k, v in result.items() if _is_image_entry(v)}
    scenario_keys = [k for k in names if k.endswith("_scenario")]
    skipped = sorted(
        {
            k.removesuffix("_scenario").removesuffix("_beam")
            for k in names
            if k.removesuffix("_scenario").removesuffix("_beam") not in RESULT_DOSE_TYPES
        }
    )
    if skipped:
        warnings.warn(
            f"DICOM RTDOSE export skips non-dose quantities: {', '.join(skipped)}.",
            stacklevel=3,
        )
    if any(sum(v is not None for v in names[k]) > 1 for k in scenario_keys):
        warnings.warn(
            "DICOM RTDOSE export writes the nominal scenario only; "
            "the other scenarios of the result are skipped.",
            stacklevel=3,
        )

    written = []
    series_number = 2
    for quantity, dose_type in RESULT_DOSE_TYPES.items():
        image = names.get(quantity)
        if image is not None:
            written.append(
                export_dose(
                    image,
                    directory,
                    ctx,
                    file_name=f"RTDOSE_{quantity}.dcm",
                    dose_type=dose_type,
                    summation_type=summation_type,
                    comment=quantity,
                    series_number=series_number,
                    plan_ref=plan_ref,
                    dvh=dvh,
                )
            )
            series_number += 1
    if per_beam:
        found = False
        for quantity, dose_type in RESULT_DOSE_TYPES.items():
            for index, image in enumerate(names.get(f"{quantity}_beam", ())):
                if image is None:
                    continue
                found = True
                number = plan_ref.beam_number(index) if plan_ref else index + 1
                written.append(
                    export_dose(
                        image,
                        directory,
                        ctx,
                        file_name=f"RTDOSE_{quantity}_beam{index + 1}.dcm",
                        dose_type=dose_type,
                        summation_type="BEAM",
                        comment=f"{quantity} beam {index + 1}",
                        series_number=series_number,
                        plan_ref=plan_ref,
                        beam_number=number,
                        dvh=dvh,
                    )
                )
                series_number += 1
        if not found:
            warnings.warn("per_beam_doses requested, but the result has no per-beam doses.")
    return written
