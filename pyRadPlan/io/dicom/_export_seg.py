"""Export a pyRadPlan StructureSet as a DICOM SEG file."""

import os
from typing import Optional

import numpy as np
import SimpleITK as sitk
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence
from pydicom.tag import Tag
from pydicom.uid import generate_uid

from pyRadPlan.cst import StructureSet

from ._export_common import (
    CT_IMAGE_STORAGE,
    SEG_STORAGE,
    ExportContext,
    first_phase_image,
    populate_common,
    direction_to_orientation,
    warn_if_multiphase,
)


def _segment_item(segment_number: int, voi) -> Dataset:
    """Build a SegmentSequence item describing a single VOI."""
    item = Dataset()
    item.SegmentNumber = segment_number
    item.SegmentLabel = voi.name
    item.SegmentDescription = voi.name
    item.SegmentAlgorithmType = "MANUAL"

    category = Dataset()
    category.CodeValue = "T-D0050"
    category.CodingSchemeDesignator = "SRT"
    category.CodeMeaning = "Tissue"
    item.SegmentedPropertyCategoryCodeSequence = Sequence([category])
    item.SegmentedPropertyTypeCodeSequence = Sequence([category])
    return item


def _referenced_series(ct_info: dict) -> Sequence:
    """Build the ReferencedSeriesSequence pointing at the exported source CT slices."""
    instances = Sequence()
    for sop_uid in ct_info.get("sop_instance_uids") or []:
        item = Dataset()
        item.ReferencedSOPClassUID = CT_IMAGE_STORAGE
        item.ReferencedSOPInstanceUID = sop_uid
        instances.append(item)
    series = Dataset()
    series.SeriesInstanceUID = ct_info["series_uid"]
    series.ReferencedInstanceSequence = instances
    return Sequence([series])


def _dimension_index(org_uid: str) -> list:
    """Build the two DimensionIndexSequence items (segment number, slice position)."""
    segment = Dataset()
    segment.DimensionOrganizationUID = org_uid
    segment.DimensionIndexPointer = Tag(0x0062, 0x000B)
    segment.FunctionalGroupPointer = Tag(0x0062, 0x000A)
    segment.DimensionDescriptionLabel = "Segment Number"
    position = Dataset()
    position.DimensionOrganizationUID = org_uid
    position.DimensionIndexPointer = Tag(0x0020, 0x0032)
    position.FunctionalGroupPointer = Tag(0x0020, 0x9113)
    position.DimensionDescriptionLabel = "Image Position Patient"
    return [segment, position]


def _shared_functional_groups(orientation: list, spacing: tuple) -> Sequence:
    """Build the SharedFunctionalGroupsSequence (orientation and pixel measures)."""
    shared = Dataset()

    plane_orient = Dataset()
    plane_orient.ImageOrientationPatient = [float(o) for o in orientation]
    shared.PlaneOrientationSequence = Sequence([plane_orient])

    measures = Dataset()
    measures.PixelSpacing = [float(spacing[1]), float(spacing[0])]
    measures.SliceThickness = float(spacing[2])
    shared.PixelMeasuresSequence = Sequence([measures])

    return Sequence([shared])


def _new_seg_dataset(ctx: ExportContext, size: tuple) -> Dataset:
    """Create a SEG dataset with header and image-description tags populated."""
    ds = Dataset()
    populate_common(
        ds,
        ctx,
        SEG_STORAGE,
        generate_uid(),
        "SEG",
        4,
        "pyRadPlan segmentation - research use only",
    )

    ds.ContentQualification = "RESEARCH"
    ds.ContentDate = ctx.export_date
    ds.ContentTime = ctx.export_time
    ds.ContentDescription = "Structure set exported from pyRadPlan"
    ds.SegmentationType = "BINARY"
    ds.ContentLabel = "SEGMENTATION"
    ds.ContentCreatorName = "pyRadPlan"
    ds.InstanceNumber = 1

    ds.ImageType = ["DERIVED", "PRIMARY"]
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.Rows = size[1]
    ds.Columns = size[0]
    # SegmentationType "BINARY" mandates 1-bit-per-pixel packed frames.
    ds.BitsAllocated = 1
    ds.BitsStored = 1
    ds.HighBit = 0
    ds.PixelRepresentation = 0
    ds.LossyImageCompression = "00"
    return ds


def export_seg(
    cst: StructureSet, directory: str, ctx: ExportContext, ct_info: Optional[dict] = None
) -> str:
    """
    Write a StructureSet as a multi-frame DICOM SEG file.

    Each VOI becomes a segment; only slices containing the segment produce a
    frame. The result round-trips with :func:`pyRadPlan.io.dicom.import_seg`.

    Parameters
    ----------
    cst : StructureSet
        The StructureSet to export.
    directory : str
        Output directory.
    ctx : ExportContext
        Resolved export context.
    ct_info : dict, optional
        The return value of :func:`export_ct`, used to reference the source series.

    Returns
    -------
    str
        The path of the written SEG file.
    """
    reference = first_phase_image(cst)
    warn_if_multiphase(cst, "SEG")
    size = reference.GetSize()
    spacing = reference.GetSpacing()
    orientation = direction_to_orientation(reference.GetDirection())

    ds = _new_seg_dataset(ctx, size)

    segment_seq = Sequence()
    per_frame_seq = Sequence()
    frames = []

    org_uid = generate_uid()
    org = Dataset()
    org.DimensionOrganizationUID = org_uid

    for i, voi in enumerate(cst.vois, start=1):
        segment_seq.append(_segment_item(i, voi))

        mask = sitk.GetArrayFromImage(voi.mask)  # (z, y, x)
        if mask.ndim == 4:
            mask = mask[0]

        for z in range(mask.shape[0]):
            slice_mask = mask[z]
            if not slice_mask.any():
                continue

            position = reference.TransformIndexToPhysicalPoint((0, 0, int(z)))

            per_frame = Dataset()
            plane_pos = Dataset()
            plane_pos.ImagePositionPatient = [float(p) for p in position]
            per_frame.PlanePositionSequence = Sequence([plane_pos])
            seg_id = Dataset()
            seg_id.ReferencedSegmentNumber = i
            per_frame.SegmentIdentificationSequence = Sequence([seg_id])
            content = Dataset()
            content.DimensionIndexValues = [i, z + 1]
            per_frame.FrameContentSequence = Sequence([content])
            per_frame_seq.append(per_frame)

            frames.append((slice_mask > 0).astype(np.uint8))

    if ct_info and ct_info.get("series_uid"):
        ds.ReferencedSeriesSequence = _referenced_series(ct_info)
    ds.DimensionOrganizationType = "3D"
    ds.DimensionOrganizationSequence = Sequence([org])
    ds.DimensionIndexSequence = Sequence(_dimension_index(org_uid))
    ds.SegmentSequence = segment_seq
    ds.SharedFunctionalGroupsSequence = _shared_functional_groups(orientation, spacing)
    ds.PerFrameFunctionalGroupsSequence = per_frame_seq
    ds.NumberOfFrames = len(frames)

    # BINARY segmentation: pack all frames into a contiguous 1-bit stream
    # (LSB-first, as required by DICOM); pydicom's pixel_array unpacks it back.
    if frames:
        bits = np.stack(frames).astype(np.uint8).ravel()
    else:
        bits = np.zeros(size[1] * size[0], dtype=np.uint8)
        ds.NumberOfFrames = 1
    ds.PixelData = np.packbits(bits, bitorder="little").tobytes()

    out_path = os.path.join(directory, "SEG.dcm")
    ds.save_as(out_path, enforce_file_format=True)
    return out_path
