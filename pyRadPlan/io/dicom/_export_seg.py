"""Export a pyRadPlan StructureSet as a DICOM SEG file."""

import os
from datetime import datetime

import numpy as np
import SimpleITK as sitk
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence
from pydicom.uid import generate_uid

from pyRadPlan.cst import StructureSet

from ._export_common import (
    SEG_STORAGE,
    UIDContext,
    populate_common,
    direction_to_orientation,
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


def _new_seg_dataset(ctx: UIDContext, size: tuple) -> Dataset:
    """Create a SEG dataset with header and image-description tags populated."""
    ds = Dataset()
    populate_common(ds, ctx, SEG_STORAGE, generate_uid(), "SEG", 4)

    now = datetime.now()
    ds.ContentDate = now.strftime("%Y%m%d")
    ds.ContentTime = now.strftime("%H%M%S")
    ds.SegmentationType = "BINARY"
    ds.ContentLabel = "SEGMENTATION"
    ds.ContentCreatorName = ctx.patient_name
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


def export_seg(cst: StructureSet, directory: str, ctx: UIDContext) -> str:
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
    ctx : UIDContext
        Shared identifiers.

    Returns
    -------
    str
        The path of the written SEG file.
    """
    reference = cst.ct_image.cube_hu
    size = reference.GetSize()
    spacing = reference.GetSpacing()
    orientation = direction_to_orientation(reference.GetDirection())

    ds = _new_seg_dataset(ctx, size)

    segment_seq = Sequence()
    per_frame_seq = Sequence()
    frames = []

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
            per_frame_seq.append(per_frame)

            frames.append((slice_mask > 0).astype(np.uint8))

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
