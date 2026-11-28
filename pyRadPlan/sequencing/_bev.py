"""Beam's eye view geometry of a sequenced beam, shared by the plots and the GUIs."""

from typing import Optional, Union

import numpy as np
from scipy import ndimage

from pyRadPlan.core import np2sitk
from pyRadPlan.ct import CT
from pyRadPlan.cst import StructureSet
from pyRadPlan.geometry import lps
from pyRadPlan.result._beam_sequence import BeamSequence, Segment
from pyRadPlan.stf import Beam, SteeringInformation


def beam_label(beam_index: int, stf: Optional[SteeringInformation]) -> str:
    """Label of a beam: its index and, with steering information, its gantry angle."""
    if stf is not None and beam_index < len(stf.beams):
        angle = getattr(stf.beams[beam_index], "gantry_angle", None)
        if angle is not None:
            return f"Beam {beam_index} ({float(angle):.0f}°)"
    return f"Beam {beam_index}"


def shape_segments(beam_sequence: BeamSequence) -> list[Segment]:
    """Return the segments of a beam that have leaf positions to draw, in delivery order.

    A viewer shows the state a segment starts in.
    """
    return [
        segment
        for segment in beam_sequence.segments
        if segment.start.mlc is not None and segment.start.mlc.leaf_positions is not None
    ]


def field_window(beam_sequence: BeamSequence) -> Optional[list[float]]:
    """Return the field of a sequenced beam on the axes of its leaves.

    The jaws of the first segment bound the field. An axis without a jaw takes the extent
    of the leaf positions (along the leaves) or of the leaf rows (across them) of all
    segments.

    Returns
    -------
    list of float or None
        ``[travel_min, travel_max, stack_min, stack_max]`` in mm, ``None`` when neither
        jaws nor leaves give the field.
    """
    if not beam_sequence.segments:
        return None
    mlcs = [segment.start.mlc for segment in shape_segments(beam_sequence)]
    orientation = mlcs[0].device_orientation if mlcs else "X"
    jaws = {jaw.device_orientation: jaw.positions for jaw in beam_sequence.segments[0].start.jaws}
    travel = jaws.get(orientation)
    stack = jaws.get("Y" if orientation == "X" else "X")

    if travel is None and mlcs:
        travel = np.concatenate(
            [np.asarray(mlc.leaf_positions, dtype=float).ravel() for mlc in mlcs]
        )
    rows = [mlc for mlc in mlcs if mlc.leaf_position_boundaries is not None]
    if stack is None and rows:
        stack = np.concatenate(
            [
                np.asarray(mlc.leaf_position_boundaries, dtype=float).ravel() + offset
                for mlc in rows
                for offset in (0.0, mlc.leaf_width)
            ]
        )
    if travel is None or stack is None or not len(travel) or not len(stack):
        return None
    return [
        float(np.min(travel)),
        float(np.max(travel)),
        float(np.min(stack)),
        float(np.max(stack)),
    ]


def stf_beam_index(beam_sequence: BeamSequence, position: int) -> int:
    """Return the index of the steering-information beam a sequenced beam is planned on.

    Parameters
    ----------
    beam_sequence : BeamSequence
        The sequenced beam.
    position : int
        Position of *beam_sequence* in its sequence, used when its segments do not name
        their beam.
    """
    for segment in beam_sequence.segments:
        if segment.stf_beam is not None:
            return segment.stf_beam
    return position


def beam_bixels(stf: Optional[SteeringInformation], stf_beam: int) -> Optional[slice]:
    """Return the slice of beam *stf_beam* among all beamlets of *stf*, ``None`` without it."""
    if stf is None or not 0 <= stf_beam < len(stf.beams):
        return None
    offset = sum(len(beam.rays) for beam in stf.beams[:stf_beam])
    return slice(offset, offset + len(stf.beams[stf_beam].rays))


def native_fluence_map(beam: Beam, values: np.ndarray, device_orientation: str) -> tuple:
    """Scatter one beam's per-ray fluence values onto a dense 2D BEV grid.

    Parameters
    ----------
    beam : Beam
        The steering-information beam (for ray positions and bixel width).
    values : np.ndarray
        Per-ray fluence values for this beam, in ray order.
    device_orientation : str
        "X" or "Y", from the corresponding ``MLC.device_orientation``.

    Returns
    -------
    tuple
        ``(grid, window)``: ``grid`` is a ``(dim_stack, dim_travel)`` array (``nan``
        where no ray exists) and ``window`` is
        ``[travel_min, travel_max, stack_min, stack_max]`` in mm, like :func:`field_window`.
    """
    travel_axis, stack_axis = (0, 2) if device_orientation == "X" else (2, 0)
    travel_pos = np.array([ray.ray_pos_bev[travel_axis] for ray in beam.rays])
    stack_pos = np.array([ray.ray_pos_bev[stack_axis] for ray in beam.rays])
    bw = beam.bixel_width

    min_travel, max_travel = travel_pos.min(), travel_pos.max()
    min_stack, max_stack = stack_pos.min(), stack_pos.max()
    dim_travel = int((max_travel - min_travel) / bw + 1)
    dim_stack = int((max_stack - min_stack) / bw + 1)

    grid = np.full((dim_stack, dim_travel), np.nan)
    travel_idx = ((travel_pos - min_travel) / bw).astype(int)
    stack_idx = ((stack_pos - min_stack) / bw).astype(int)
    grid[stack_idx, travel_idx] = values

    window = [
        min_travel - 0.5 * bw,
        max_travel + 0.5 * bw,
        min_stack - 0.5 * bw,
        max_stack + 0.5 * bw,
    ]
    return grid, window


def target_bev_points(
    ct: CT, cst: StructureSet, beam: Beam, device_orientation: str
) -> Optional[np.ndarray]:
    """Project the target volume into *beam*'s isocenter (beam's-eye-view) plane.

    Parameters
    ----------
    ct : CT
        Provides the grid the target voxel indices refer to.
    cst : StructureSet
        Supplies the target volume (union of all TARGET VOIs).
    beam : Beam
        Steering-information beam, for its geometry (angles, isocenter, SAD).
    device_orientation : str
        "X" or "Y", from the corresponding ``MLC.device_orientation``.

    Returns
    -------
    np.ndarray or None
        ``(N, 2)`` array of (horizontal, vertical) positions in mm, on the same
        axes as :func:`native_fluence_map`'s window, or ``None`` when the
        structure set holds no target.
    """
    mask = cst.target_union_mask()
    if mask is None:
        return None
    voxels = np2sitk.sitk_mask_to_linear_indices(mask, order="numpy")
    if voxels is None or np.size(voxels) == 0:
        return None

    coords = np2sitk.linear_indices_to_grid_coordinates(voxels, ct.grid, index_type="numpy").T
    iso_center = np.asarray(beam.iso_center, dtype=float).reshape((3, 1))
    rotation_t = lps.get_beam_rotation_matrix(beam.gantry_angle, beam.couch_angle).transpose()

    bev = rotation_t @ (coords - iso_center)
    # Divergent projection onto the isocenter plane; axis 1 is the beam's own axis.
    bev = beam.sad * bev / (beam.sad + bev[1, :])

    travel_axis, stack_axis = (0, 2) if device_orientation == "X" else (2, 0)
    return np.column_stack((bev[travel_axis, :], bev[stack_axis, :]))


def target_outline_mask(
    points: Optional[np.ndarray], resolution: Union[float, tuple[float, float]] = 3.0
) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Rasterize projected target points into the filled region their outline encloses.

    Parameters
    ----------
    points : np.ndarray or None
        ``(N, 2)`` positions, as returned by :func:`target_bev_points`.
    resolution : float or tuple of float
        Bin size, per axis when the two axes are not in the same unit (e.g. mm
        horizontally but leaf-pair index vertically). Must be at least the
        projected voxel spacing on each axis.

    Returns
    -------
    tuple or None
        ``(x, y, occupied)``: the bin centers along both axes and the boolean mask
        ``(len(y), len(x))`` of the bins the target covers, holes closed. ``None`` when
        there is nothing to outline.
    """
    if points is None or len(points) == 0:
        return None

    res_x, res_y = resolution if isinstance(resolution, tuple) else (resolution, resolution)
    res_x, res_y = max(float(res_x), 1e-3), max(float(res_y), 1e-3)
    x, y = points[:, 0], points[:, 1]
    x_edges = np.arange(x.min() - res_x, x.max() + 2 * res_x, res_x)
    y_edges = np.arange(y.min() - res_y, y.max() + 2 * res_y, res_y)
    if x_edges.size < 2 or y_edges.size < 2:  # noqa: PLR2004
        return None

    hist, _, _ = np.histogram2d(x, y, bins=(x_edges, y_edges))
    occupied = hist.T > 0
    occupied = ndimage.binary_closing(occupied, structure=np.ones((3, 3)))
    occupied = ndimage.binary_fill_holes(occupied)
    return (
        0.5 * (x_edges[:-1] + x_edges[1:]),
        0.5 * (y_edges[:-1] + y_edges[1:]),
        occupied,
    )
