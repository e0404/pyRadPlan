"""Structure import: DICOM RTSTRUCT/SEG to pyRadPlan VOIs.

Refactored from a previous (overengineered) implementation. The contour->mask
and SEG orientation handling are preserved; the output is now a list of
pyRadPlan :class:`VOI` objects backed by SimpleITK masks aligned to the CT.
"""

import logging
from typing import Optional

import numpy as np
import SimpleITK as sitk
from scipy.interpolate import interp1d

from pyRadPlan.core import ProgressReporter
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


def import_rtstruct(
    structure_dataset, ct: CT, reporter: Optional[ProgressReporter] = None
) -> list[VOI]:
    """
    Convert an RTSTRUCT dataset to a list of pyRadPlan VOIs.

    Parameters
    ----------
    structure_dataset : pydicom.Dataset
        The RTSTRUCT DICOM dataset.
    ct : CT
        The reference pyRadPlan CT object.
    reporter : ProgressReporter, optional
        Reporter used to publish the per-structure progress. Defaults to a
        private reporter, which still reaches context-scoped observers.

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

    reporter = reporter if reporter is not None else ProgressReporter()
    contours = structure_dataset.ROIContourSequence
    logger.info("Converting %d RTSTRUCT contour set(s) to masks.", len(contours))

    interpolators = _axis_interpolators(ct)

    vois = []
    for roi_contour in reporter.track(contours, name="Structure", unit="roi"):
        roi_structure = structure_lookup.get(roi_contour.ReferencedROINumber)
        if roi_structure is None:
            logger.warning("No structure found for ROI number %s", roi_contour.ReferencedROINumber)
            continue

        name = roi_structure.ROIName
        try:
            cube_xyz = _compute_segment_mask(ct, roi_contour, interpolators)
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


def import_seg(
    seg_dataset, ct_datasets: list, ct: CT, reporter: Optional[ProgressReporter] = None
) -> list[VOI]:
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
    reporter : ProgressReporter, optional
        Reporter used to publish the per-segment progress. Defaults to a
        private reporter, which still reaches context-scoped observers.

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

    reporter = reporter if reporter is not None else ProgressReporter()
    logger.info("Converting %d SEG segment(s) to masks.", len(segments))

    vois = []
    for segment in reporter.track(segments, name="Segment", unit="seg"):
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


def _axis_interpolators(ct):
    """Return the (x, y) world-to-voxel-index interpolators for *ct*.

    They depend only on the CT grid, so they are built once and reused for every
    contour rather than rebuilt per contour.
    """
    return _create_axis_interpolator(ct, 0), _create_axis_interpolator(ct, 1)


def _create_slice_mask(points, ct, x_interp, y_interp):
    """Rasterize one contour into the slice grid.

    Returns ``(window, x_start, y_start, z_pos)``, where *window* covers only the
    contour's bounding box in voxel indices -- a voxel outside that box cannot be
    inside the contour, so testing the whole slice against every contour (a
    512x512 slice is 262144 point-in-polygon tests) is wasted work. The caller
    ORs the window into the cube at the given offsets.

    ``(None, ...)`` is returned when the contour lies outside the CT grid.
    """
    points_x, points_y, z_pos = points[:, 0], points[:, 1], points[0, 2]

    if not (min(ct.z) <= z_pos <= max(ct.z)):
        return None, 0, 0, None

    vertices = np.column_stack((x_interp(points_x), y_interp(points_y)))
    if not np.isfinite(vertices).all():
        # Would make the bounding box below meaningless; the whole-slice test this
        # replaced simply found nothing for such a contour, so skip it the same way.
        return None, 0, 0, None

    # Bounding box in voxel indices, padded by one voxel and clipped to the grid.
    num_x, num_y = ct.cube_dim[0], ct.cube_dim[1]
    x_start = max(int(np.floor(vertices[:, 0].min())) - 1, 0)
    x_stop = min(int(np.ceil(vertices[:, 0].max())) + 2, num_x)
    y_start = max(int(np.floor(vertices[:, 1].min())) - 1, 0)
    y_stop = min(int(np.ceil(vertices[:, 1].max())) + 2, num_y)
    if x_start >= x_stop or y_start >= y_stop:
        return None, 0, 0, None

    window = _fill_polygon(vertices, x_start, x_stop, y_start, y_stop)
    return window, x_start, y_start, z_pos


def _fill_polygon(vertices, x_start, x_stop, y_start, y_stop):
    """Rasterize a closed polygon into a boolean window by scanline filling.

    A voxel is filled when its *centre* lies inside the polygon under the
    even-odd rule, a centre exactly on the boundary counting as inside so that a
    structure narrower than one voxel does not vanish. The scanline intersects
    each voxel row with the edges instead of testing every voxel against every
    edge, which turns the cost from ``voxels x edges`` into ``rows x edges``.

    This reproduces the per-voxel ``matplotlib.path.Path.contains_points`` test
    it replaced. The two can only part ways where vertices land exactly on voxel
    centres, or for self-intersecting contours (where the even-odd and nonzero
    winding rules genuinely differ); real RTSTRUCT contours are simple polygons
    on coordinates that do not line up with the grid.

    The convention agrees with MITK Workbench's contour-to-mask algorithm: on a
    512x512x297 CT the two differed by 3 voxels out of 17.6 million across four
    structures, two of which were voxel-identical. ``test_io_dicom.py`` pins the
    rule itself on a grid small enough to read, and requires the bundled RTSTRUCT
    to reproduce a matRad export voxel for voxel.

    Parameters
    ----------
    vertices : np.ndarray
        ``(N, 2)`` polygon vertices in fractional voxel indices, closed or not.
    x_start, x_stop, y_start, y_stop : int
        Half-open window in voxel indices to rasterize into.

    Returns
    -------
    np.ndarray
        Boolean array of shape ``(x_stop - x_start, y_stop - y_start)``.
    """
    num_x, num_y = x_stop - x_start, y_stop - y_start
    window = np.zeros((num_x, num_y), dtype=bool)
    if num_x <= 0 or num_y <= 0:
        return window

    if not np.array_equal(vertices[0], vertices[-1]):
        vertices = np.vstack([vertices, vertices[0]])
    edge_x0, edge_y0 = vertices[:-1, 0], vertices[:-1, 1]
    edge_x1, edge_y1 = vertices[1:, 0], vertices[1:, 1]

    # One scanline per voxel row: an edge crosses it when its endpoints straddle
    # the row. Horizontal edges never straddle, so they drop out on their own.
    # ">=" rather than ">" decides which side a vertex sitting exactly on a
    # scanline falls: it reproduces the edge the previous per-voxel test picked.
    # For any y not exactly on the lattice the two are the same comparison.
    rows = np.arange(y_start, y_stop, dtype=float)[:, None]
    crosses = (edge_y0[None, :] >= rows) != (edge_y1[None, :] >= rows)
    if not crosses.any():
        return window

    # x where each crossing edge meets the scanline (+inf keeps non-crossings last).
    slope_denominator = np.where(crosses, (edge_y1 - edge_y0)[None, :], 1.0)
    crossings = (
        edge_x0[None, :]
        + (rows - edge_y0[None, :]) * (edge_x1 - edge_x0)[None, :] / slope_denominator
    )
    crossings = np.sort(np.where(crosses, crossings, np.inf), axis=1)

    # Even-odd: between the 1st and 2nd crossing is inside, 3rd to 4th, and so on.
    num_spans = int(crosses.sum(axis=1).max()) // 2
    if num_spans == 0:
        return window
    span_from = crossings[:, 0 : 2 * num_spans : 2]
    span_to = crossings[:, 1 : 2 * num_spans : 2]

    # Voxel centres covered by a span, clipped to the window. A centre landing
    # exactly on the boundary counts as inside, which keeps a structure narrower
    # than one voxel from disappearing and matches what the previous per-voxel
    # test did at such ties. For any coordinate not exactly on the lattice this
    # is identical to taking the strict interior.
    first = np.maximum(np.ceil(span_from), x_start)
    last = np.minimum(np.floor(span_to), x_stop - 1)
    usable = np.isfinite(span_from) & np.isfinite(span_to) & (first <= last)
    if not usable.any():
        return window

    span_rows = np.broadcast_to(np.arange(num_y)[:, None], span_from.shape)[usable]
    # Paint the spans with a +1/-1 difference array and a cumulative sum, which
    # avoids a Python loop over the (ragged) spans.
    deltas = np.zeros((num_x + 1, num_y), dtype=np.int32)
    np.add.at(deltas, ((first[usable] - x_start).astype(np.intp), span_rows), 1)
    np.add.at(deltas, ((last[usable] - x_start).astype(np.intp) + 1, span_rows), -1)
    return np.cumsum(deltas, axis=0)[:-1] > 0


def _compute_segment_mask(ct, roi_contour, interpolators=None):
    """Compute an (x, y, z) binary mask cube from RTSTRUCT contour data.

    *interpolators* is the ``(x, y)`` pair from :func:`_axis_interpolators`; it
    depends only on *ct*, so the caller builds it once for the whole structure set.
    """
    if not hasattr(roi_contour, "ContourSequence") or not roi_contour.ContourSequence:
        return None

    x_interp, y_interp = interpolators if interpolators is not None else _axis_interpolators(ct)

    segment_cube = np.zeros(ct.cube_dim, dtype=np.uint8)
    ct_z_positions = ct.z

    for sequence in roi_contour.ContourSequence:
        points = _process_contour_points(sequence)
        if points is None:
            continue
        window, x_start, y_start, z_pos = _create_slice_mask(points, ct, x_interp, y_interp)
        if window is None:
            continue
        slice_idx = int(np.argmin(np.abs(ct_z_positions - z_pos)))
        num_x, num_y = window.shape
        segment_cube[x_start : x_start + num_x, y_start : y_start + num_y, slice_idx] |= (
            window.astype(np.uint8)
        )

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
