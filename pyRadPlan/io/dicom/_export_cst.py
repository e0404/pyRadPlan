"""Export a pyRadPlan StructureSet as a DICOM RTSTRUCT file."""

import os

import numpy as np
import contourpy
import SimpleITK as sitk
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence
from pydicom.uid import generate_uid

from pyRadPlan.cst import StructureSet

from ._export_common import (
    CT_IMAGE_STORAGE,
    DETACHED_STUDY_MANAGEMENT,
    RT_STRUCT_STORAGE,
    UIDContext,
    populate_common,
)


def _slice_contours(mask_slice: np.ndarray) -> list:
    """Return closed contour polygons (index coordinates) for one mask slice."""
    if not mask_slice.any():
        return []
    gen = contourpy.contour_generator(
        z=mask_slice.astype(float), line_type=contourpy.LineType.Separate
    )
    return [line for line in gen.lines(0.5) if len(line) >= 3]


def _voi_contour_sequence(voi, ct_image) -> Sequence:
    """Build the ContourSequence (closed planar polygons) for a single VOI."""
    mask = sitk.GetArrayFromImage(voi.mask)  # (z, y, x)
    if mask.ndim == 4:
        mask = mask[0]

    contours = Sequence()
    for z in range(mask.shape[0]):
        for line in _slice_contours(mask[z]):
            contour_data = []
            for xi, yi in line:
                phys = ct_image.TransformContinuousIndexToPhysicalPoint(
                    (float(xi), float(yi), float(z))
                )
                contour_data.extend([float(phys[0]), float(phys[1]), float(phys[2])])
            item = Dataset()
            item.ContourGeometricType = "CLOSED_PLANAR"
            item.NumberOfContourPoints = len(line)
            item.ContourData = contour_data
            contours.append(item)
    return contours


def _referenced_frame_of_reference(ctx: UIDContext, ct_info: dict) -> Sequence:
    """Build the ReferencedFrameOfReferenceSequence linking the RTSTRUCT to the CT.

    When ``ct_info`` carries the CT series and per-slice SOP instance UIDs (from
    :func:`export_ct`), the full ``RTReferencedStudy → RTReferencedSeries →
    ContourImageSequence`` chain is emitted so third-party viewers can associate
    the structures with the CT. Otherwise only the frame of reference is set.
    """
    ref_for = Dataset()
    ref_for.FrameOfReferenceUID = ctx.frame_uid

    series_uid = ct_info.get("series_uid")
    sop_uids = ct_info.get("sop_instance_uids") or []
    if series_uid and sop_uids:
        contour_images = Sequence()
        for sop_uid in sop_uids:
            image_item = Dataset()
            image_item.ReferencedSOPClassUID = CT_IMAGE_STORAGE
            image_item.ReferencedSOPInstanceUID = sop_uid
            contour_images.append(image_item)

        ref_series = Dataset()
        ref_series.SeriesInstanceUID = series_uid
        ref_series.ContourImageSequence = contour_images

        ref_study = Dataset()
        ref_study.ReferencedSOPClassUID = DETACHED_STUDY_MANAGEMENT
        ref_study.ReferencedSOPInstanceUID = ctx.study_uid
        ref_study.RTReferencedSeriesSequence = Sequence([ref_series])

        ref_for.RTReferencedStudySequence = Sequence([ref_study])

    return Sequence([ref_for])


def export_cst(cst: StructureSet, directory: str, ctx: UIDContext, ct_info: dict) -> str:
    """
    Write a StructureSet as a DICOM RTSTRUCT file.

    Parameters
    ----------
    cst : StructureSet
        The StructureSet to export.
    directory : str
        Output directory.
    ctx : UIDContext
        Shared identifiers.
    ct_info : dict
        The return value of :func:`export_ct`, used for series referencing.

    Returns
    -------
    str
        The path of the written RTSTRUCT file.
    """
    ct_image = cst.ct_image.cube_hu

    ds = Dataset()
    populate_common(ds, ctx, RT_STRUCT_STORAGE, generate_uid(), "RTSTRUCT", 3)
    ds.StructureSetLabel = "pyRadPlan"
    ds.StructureSetName = "pyRadPlan"

    ds.ReferencedFrameOfReferenceSequence = _referenced_frame_of_reference(ctx, ct_info)

    roi_seq = Sequence()
    contour_seq = Sequence()
    obs_seq = Sequence()

    for i, voi in enumerate(cst.vois, start=1):
        roi_item = Dataset()
        roi_item.ROINumber = i
        roi_item.ReferencedFrameOfReferenceUID = ctx.frame_uid
        roi_item.ROIName = voi.name
        roi_item.ROIGenerationAlgorithm = "AUTOMATIC"
        roi_seq.append(roi_item)

        color = list(voi.visible_color) if voi.visible_color is not None else [255, 0, 0]
        roi_contour = Dataset()
        roi_contour.ROIDisplayColor = [int(c) for c in color]
        roi_contour.ReferencedROINumber = i
        roi_contour.ContourSequence = _voi_contour_sequence(voi, ct_image)
        contour_seq.append(roi_contour)

        obs = Dataset()
        obs.ObservationNumber = i
        obs.ReferencedROINumber = i
        obs.RTROIInterpretedType = "PTV" if voi.voi_type == "TARGET" else "ORGAN"
        obs.ROIInterpreter = ""
        obs_seq.append(obs)

    ds.StructureSetROISequence = roi_seq
    ds.ROIContourSequence = contour_seq
    ds.RTROIObservationsSequence = obs_seq

    out_path = os.path.join(directory, "RTSTRUCT.dcm")
    ds.save_as(out_path, enforce_file_format=True)
    return out_path
