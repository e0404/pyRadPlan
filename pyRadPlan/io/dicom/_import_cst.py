"""Structure import: DICOM RTSTRUCT/SEG to pyRadPlan VOIs.

Refactored from a previous (overengineered) implementation. The contour->mask
and SEG orientation handling are preserved; the output is now a list of
pyRadPlan :class:`VOI` objects backed by SimpleITK masks aligned to the CT.
"""

import logging

import numpy as np
import matplotlib.path as mpath
import SimpleITK as sitk
from scipy.interpolate import interp1d

from pyRadPlan.ct import CT
from pyRadPlan.cst import VOI, validate_voi

from ._helpers import determine_structure_type

logger = logging.getLogger(__name__)


def _mask_cube_to_voi(cube_xyz, name, ct, visible_color):
    """Build a VOI from an (x, y, z) numpy mask cube aligned to the CT."""
    # SimpleITK arrays are ordered (z, y, x).
    sitk_arr = np.ascontiguousarray(np.transpose(cube_xyz, (2, 1, 0)).astype(np.uint8))
    mask = sitk.GetImageFromArray(sitk_arr)
    mask.CopyInformation(ct.cube_hu)
    return validate_voi(
        name=str(name),
        voi_type=determine_structure_type(name),
        mask=mask,
        ct_image=ct,
        visible_color=visible_color,
    )


def import_rtstruct(structure_dataset, ct: CT) -> list[VOI]:
    """
    Convert an RTSTRUCT dataset to a list of pyRadPlan VOIs.

    Parameters
    ----------
    structure_dataset : pydicom.Dataset
        The RTSTRUCT DICOM dataset.
    ct : CT
        The reference pyRadPlan CT object.

    Returns
    -------
    list[VOI]
        The imported VOIs (empty structures are skipped).
    """
    if (
        not hasattr(structure_dataset, "ROIContourSequence")
        or not structure_dataset.ROIContourSequence
    ):
        return []

    structure_lookup = {seq.ROINumber: seq for seq in structure_dataset.StructureSetROISequence}

    vois = []
    for roi_contour in structure_dataset.ROIContourSequence:
        roi_structure = structure_lookup.get(roi_contour.ReferencedROINumber)
        if roi_structure is None:
            logger.warning("No structure found for ROI number %s", roi_contour.ReferencedROINumber)
            continue

        name = roi_structure.ROIName
        try:
            cube_xyz = _compute_segment_mask(ct, roi_contour)
            if cube_xyz is None or not cube_xyz.any():
                logger.info("Skipping empty structure '%s'.", name)
                continue

            visible_color = None
            if hasattr(roi_contour, "ROIDisplayColor"):
                visible_color = tuple(int(c) for c in roi_contour.ROIDisplayColor)

            vois.append(_mask_cube_to_voi(cube_xyz, name, ct, visible_color))
        except Exception as exc:  # noqa: BLE001 - one bad ROI must not abort the import
            logger.warning("Failed to process ROI '%s': %s", name, exc)
            continue

    return vois


def import_seg(seg_dataset, ct_datasets: list, ct: CT) -> list[VOI]:
    """
    Convert a SEG dataset to a list of pyRadPlan VOIs.

    Parameters
    ----------
    seg_dataset : pydicom.Dataset
        The SEG DICOM dataset (with pixel data).
    ct_datasets : list
        The CT pydicom datasets (geometry only is required) for orientation.
    ct : CT
        The reference pyRadPlan CT object.

    Returns
    -------
    list[VOI]
        The imported VOIs.
    """
    if not hasattr(seg_dataset, "SegmentSequence"):
        logger.warning("SEG file has no SegmentSequence.")
        return []

    pixel_array = seg_dataset.pixel_array
    segments = seg_dataset.SegmentSequence

    frame_positions, frame_segments = _map_seg_frames(seg_dataset)
    ct_z = ct.z

    vois = []
    for segment in segments:
        segment_number = getattr(segment, "SegmentNumber", None)
        segment_label = getattr(segment, "SegmentLabel", f"Segment_{segment_number}")

        cube_xyz = np.zeros(ct.cube_dim, dtype=np.uint8)

        segment_frame_indices = [
            idx for idx, seg_num in enumerate(frame_segments) if seg_num == segment_number
        ]

        for frame_idx in segment_frame_indices:
            seg_z_pos = frame_positions[frame_idx]
            ct_slice_idx = int(np.argmin(np.abs(ct_z - seg_z_pos)))

            if pixel_array.ndim == 3:
                frame_data = pixel_array[frame_idx, :, :]
                frame_data = _apply_seg_orientation_correction(frame_data, ct_datasets)
                cube_xyz[:, :, ct_slice_idx] = frame_data.T

        if not cube_xyz.any():
            logger.info("Skipping empty SEG segment '%s'.", segment_label)
            continue

        # CIELab color conversion is omitted; let StructureSet assign a color.
        vois.append(_mask_cube_to_voi(cube_xyz, segment_label, ct, None))

    return vois


# ---------------------------------------------------------------------------
# Private helpers for RTSTRUCT conversion
# ---------------------------------------------------------------------------


def _process_contour_points(sequence):
    """Extract and validate contour points from a contour sequence item."""
    if sequence.ContourGeometricType == "POINT":
        return None

    contour_data = np.array(sequence.ContourData)
    points = contour_data.reshape(-1, 3)

    # Close contour if not already closed.
    if not np.array_equal(points[0], points[-1]):
        points = np.vstack([points, points[0]])

    # Round z-coordinates for numerical precision.
    points[:, 2] = np.round(points[:, 2] * 1e10) / 1e10

    return points


def _create_axis_interpolator(ct, dim_idx):
    """Create an interpolator for a specific dimension (0=x, 1=y) respecting CT direction."""
    size = ct.cube_dim[dim_idx]

    idx_start = [0, 0, 0]
    val_start = ct.cube_hu.TransformIndexToPhysicalPoint(idx_start)[dim_idx]

    idx_end = [0, 0, 0]
    idx_end[dim_idx] = size - 1
    val_end = ct.cube_hu.TransformIndexToPhysicalPoint(idx_end)[dim_idx]

    coords = np.linspace(val_start, val_end, size)
    indices = np.arange(size)

    if coords[-1] < coords[0]:
        coords = np.flip(coords)
        indices = np.flip(indices)

    return interp1d(coords, indices, kind="linear", fill_value="extrapolate")


def _create_slice_mask(points, ct):
    """Create a binary mask for a single slice from contour points."""
    points_x, points_y, z_pos = points[:, 0], points[:, 1], points[0, 2]

    if not (min(ct.z) <= z_pos <= max(ct.z)):
        return None, None

    x_interp = _create_axis_interpolator(ct, 0)
    y_interp = _create_axis_interpolator(ct, 1)

    interpolated_x = x_interp(points_x)
    interpolated_y = y_interp(points_y)

    vertices = np.column_stack((interpolated_x, interpolated_y))
    path = mpath.Path(vertices)

    x_grid, y_grid = np.meshgrid(
        range(ct.cube_dim[0]),
        range(ct.cube_dim[1]),
        indexing="ij",
    )
    grid_points = np.column_stack((x_grid.ravel(), y_grid.ravel()))

    mask = path.contains_points(grid_points).reshape(ct.cube_dim[:2])
    return mask, z_pos


def _compute_segment_mask(ct, roi_contour):
    """Compute an (x, y, z) binary mask cube from RTSTRUCT contour data."""
    if not hasattr(roi_contour, "ContourSequence") or not roi_contour.ContourSequence:
        return None

    segment_cube = np.zeros(ct.cube_dim, dtype=np.uint8)
    ct_z_positions = ct.z

    for sequence in roi_contour.ContourSequence:
        points = _process_contour_points(sequence)
        if points is None:
            continue
        mask, z_pos = _create_slice_mask(points, ct)
        if mask is None:
            continue
        slice_idx = int(np.argmin(np.abs(ct_z_positions - z_pos)))
        segment_cube[:, :, slice_idx] |= mask.astype(np.uint8)

    return segment_cube


# ---------------------------------------------------------------------------
# Private helpers for SEG conversion
# ---------------------------------------------------------------------------


def _map_seg_frames(seg_dataset):
    """Map SEG frames to z-positions and referenced segment numbers."""
    frame_positions = []
    frame_segments = []

    if hasattr(seg_dataset, "PerFrameFunctionalGroupsSequence"):
        for frame_group in seg_dataset.PerFrameFunctionalGroupsSequence:
            z_pos = None
            seg_num = None

            if hasattr(frame_group, "PlanePositionSequence"):
                pos_seq = frame_group.PlanePositionSequence[0]
                if hasattr(pos_seq, "ImagePositionPatient"):
                    z_pos = float(pos_seq.ImagePositionPatient[2])

            if hasattr(frame_group, "SegmentIdentificationSequence"):
                seg_id = frame_group.SegmentIdentificationSequence[0]
                if hasattr(seg_id, "ReferencedSegmentNumber"):
                    seg_num = int(seg_id.ReferencedSegmentNumber)

            frame_positions.append(z_pos)
            frame_segments.append(seg_num)

    return frame_positions, frame_segments


def _apply_seg_orientation_correction(mask_data, ct_datasets):
    """Apply orientation and patient position corrections to SEG mask data."""
    corrected_mask = mask_data.copy()

    if not ct_datasets:
        return corrected_mask

    if hasattr(ct_datasets[0], "ImageOrientationPatient"):
        orientation = ct_datasets[0].ImageOrientationPatient
        is_radiological = orientation[0] < 0 and orientation[4] < 0
        if is_radiological:
            corrected_mask = np.flip(corrected_mask, axis=(0, 1))

    if hasattr(ct_datasets[0], "PatientPosition"):
        patient_position = getattr(ct_datasets[0], "PatientPosition", "HFS")

        if patient_position == "HFP":
            corrected_mask = np.flip(corrected_mask, axis=0)
        elif patient_position == "FFP":
            corrected_mask = np.flip(corrected_mask, axis=(0, 1))
        elif patient_position == "HFS":
            corrected_mask = np.flip(corrected_mask, axis=1)

    return corrected_mask
