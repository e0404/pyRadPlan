"""Visualization of MLC aperture shapes from a sequenced plan result."""

from typing import Optional, Union, Literal
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec

from scipy import ndimage

from pyRadPlan.core import np2sitk
from pyRadPlan.ct import CT
from pyRadPlan.cst import StructureSet
from pyRadPlan.geometry import lps
from pyRadPlan.machines import MLC
from pyRadPlan.result import PlanResult, Sequence
from pyRadPlan.result._beam_sequence import BeamSequence
from pyRadPlan.stf import Beam, SteeringInformation, validate_stf
from pyRadPlan.visualization._plot_dose_comparison import (
    _beam_label,
    _draw_dose_comparison_panels,
)


def _leaf_row_bounds(mlc: MLC, leaf_coordinate: str) -> tuple[np.ndarray, np.ndarray]:
    """Lower/upper vertical edge of every active leaf row of one MLC segment.

    For ``"physical"``, rows span ``[boundary, boundary + leaf_width]`` -- the
    boundaries are the rows' lower edges. For ``"leaf_num"``, rows are relabeled
    1..N from the lowest to the highest z.
    """
    n = len(mlc.leaf_positions)
    if leaf_coordinate == "physical":
        low = np.asarray(mlc.leaf_position_boundaries, dtype=float)
        return low, low + mlc.leaf_width
    centers = np.arange(1, n + 1, dtype=float)
    return centers - 0.5, centers + 0.5


def _weight_color(weight: float, w_max: float) -> tuple:
    """Open-aperture colour encoding a segment's weight: light grey (low) to blue (high)."""
    frac = min(max(weight / w_max, 0.0), 1.0) if w_max > 0 else 0.0
    light = np.array([0.92, 0.92, 0.92])
    blue = np.array([0.15, 0.35, 0.75])
    return tuple(light + frac * (blue - light))


def _draw_aperture(ax: plt.Axes, mlc: MLC, mlc_window: list[float], leaf_coordinate: str) -> None:
    """Draw the blocked (closed) leaf regions of one MLC segment into ax."""
    travel_min, travel_max = mlc_window[0], mlc_window[1]
    y_low, y_high = _leaf_row_bounds(mlc, leaf_coordinate)

    for (left, right), lo, hi in zip(mlc.leaf_positions, y_low, y_high):
        # filled blocked (closed) regions on either side of the aperture
        ax.add_patch(
            Rectangle(
                (travel_min, lo),
                left - travel_min,
                hi - lo,
                facecolor="dimgray",
                edgecolor="black",
                linewidth=0.5,
                zorder=3,
            )
        )
        ax.add_patch(
            Rectangle(
                (right, lo),
                travel_max - right,
                hi - lo,
                facecolor="dimgray",
                edgecolor="black",
                linewidth=0.5,
                zorder=3,
            )
        )

    ax.set_xlim(travel_min, travel_max)
    ax.set_ylim(y_low.min(), y_high.max())
    ax.set_xlabel("horiz. pos. [mm]")
    ax.set_ylabel("vert. pos. [mm]" if leaf_coordinate == "physical" else "leaf pair #")


def _target_bev_points(
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
        axes as :func:`_native_fluence_map`'s window, or ``None`` when the
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


def _leaf_num_points(points: np.ndarray, mlc: MLC) -> np.ndarray:
    """Map BEV points' vertical coordinate from mm onto the leaf-pair index axis."""
    boundaries = np.asarray(mlc.leaf_position_boundaries, dtype=float)
    if boundaries.size == 0 or not mlc.leaf_width:
        return points
    rows = 1.0 + (points[:, 1] - boundaries[0]) / mlc.leaf_width
    return np.column_stack((points[:, 0], rows))


def _draw_bixel_grid(
    ax: plt.Axes,
    window: list[float],
    bixel_width: float,
    vertical: Optional[np.ndarray] = None,
    color: str = "white",
) -> None:
    """Draw the ray/bixel grid over an axis.

    Parameters
    ----------
    window : list[float]
        ``[travel_min, travel_max, stack_min, stack_max]`` in mm, as returned by
        :func:`_native_fluence_map` (already the bixel *boundaries*).
    vertical : np.ndarray, optional
        Pre-transformed vertical line positions, for an axis that is not in mm
        (e.g. the leaf-pair index). Defaults to the window's own mm positions.
    """
    if not bixel_width:
        return

    horizontal = np.arange(window[0], window[1] + 0.5 * bixel_width, bixel_width)
    if vertical is None:
        vertical = np.arange(window[2], window[3] + 0.5 * bixel_width, bixel_width)

    for x in horizontal:
        ax.axvline(x, color=color, linewidth=0.4, alpha=0.6, zorder=3)
    for y in vertical:
        ax.axhline(y, color=color, linewidth=0.4, alpha=0.6, zorder=3)


def _draw_target_outline(
    ax: plt.Axes,
    points: Optional[np.ndarray],
    resolution: Union[float, tuple[float, float]] = 3.0,
    color: str = "red",
) -> None:
    """Outline the projected target on *ax*.

    Parameters
    ----------
    resolution : float or tuple of float
        Bin size, per axis when the two axes are not in the same unit (e.g. mm
        horizontally but leaf-pair index vertically). Must be at least the
        projected voxel spacing on each axis.
    """
    if points is None or len(points) == 0:
        return

    res_x, res_y = resolution if isinstance(resolution, tuple) else (resolution, resolution)
    res_x, res_y = max(float(res_x), 1e-3), max(float(res_y), 1e-3)
    x, y = points[:, 0], points[:, 1]
    x_edges = np.arange(x.min() - res_x, x.max() + 2 * res_x, res_x)
    y_edges = np.arange(y.min() - res_y, y.max() + 2 * res_y, res_y)
    if x_edges.size < 2 or y_edges.size < 2:
        return

    hist, _, _ = np.histogram2d(x, y, bins=(x_edges, y_edges))
    occupied = hist.T > 0
    occupied = ndimage.binary_closing(occupied, structure=np.ones((3, 3)))
    occupied = ndimage.binary_fill_holes(occupied)

    ax.contour(
        0.5 * (x_edges[:-1] + x_edges[1:]),
        0.5 * (y_edges[:-1] + y_edges[1:]),
        occupied.astype(float),
        levels=[0.5],
        colors=color,
        linewidths=1.2,
        zorder=5,
    )


def _draw_mlc_borders(
    ax: plt.Axes,
    mlc: MLC,
    mlc_window: list[float],
    leaf_coordinate: str = "physical",
    color: str = "black",
) -> None:
    """Outline one segment's leaves as lines, for overlaying on a fluence map."""
    travel_min, travel_max = mlc_window[0], mlc_window[1]
    y_low, y_high = _leaf_row_bounds(mlc, leaf_coordinate)

    for (left, right), lo, hi in zip(mlc.leaf_positions, y_low, y_high):
        for start, stop in ((travel_min, left), (right, travel_max)):
            if stop <= start:
                continue  # leaf fully retracted past the field edge
            ax.add_patch(
                Rectangle(
                    (start, lo),
                    stop - start,
                    hi - lo,
                    fill=False,
                    edgecolor=color,
                    linewidth=0.8,
                    zorder=4,
                )
            )


def _per_beam_layout(n_shapes: int) -> tuple[int, int]:
    """Two rows of shapes, as wide as it takes."""
    cols = max(int(np.ceil(n_shapes / 2)), 1)
    rows = max(int(np.ceil(n_shapes / cols)), 1)
    return rows, cols


_SHAPE_ROW_HEIGHT = 1.8
_SHAPE_COL_WIDTH = 1.9
_COMPARISON_ROW_HEIGHT = 4.5
_COMPARISON_COL_WIDTH = 5.0


def _shape_grid_layout(n_shapes: int) -> dict:
    """Rows/cols/figsize for the shape-grid-only figure."""
    rows, cols = _per_beam_layout(n_shapes)
    return {
        "rows": rows,
        "cols": cols,
        "figsize": (_SHAPE_COL_WIDTH * cols, _SHAPE_ROW_HEIGHT * rows),
    }


def _col_edges(n_cols: int, n_panels: int) -> np.ndarray:
    """Column boundaries splitting ``n_cols`` GridSpec columns into ``n_panels`` equal spans."""
    return np.linspace(0, n_cols, n_panels + 1).round().astype(int)


def _comparison_layout(show_fluence: bool, show_dose: bool) -> dict:
    """Rows/cols/column-edges/figsize for the fluence+dose comparison figure."""
    n_panels = 3
    total_rows = int(show_fluence) + int(show_dose)
    edges = _col_edges(n_panels, n_panels)
    return {
        "n_cols": n_panels,
        "total_rows": total_rows,
        "edges": edges,
        "figsize": (_COMPARISON_COL_WIDTH * n_panels, _COMPARISON_ROW_HEIGHT * total_rows),
    }


def _native_fluence_map(beam: Beam, values: np.ndarray, device_orientation: str) -> tuple:
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
        ``[travel_min, travel_max, stack_min, stack_max]`` in mm, matching ``mlc_window``.
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


def _draw_fluence_axis(
    ax: plt.Axes, fig: Figure, grid: np.ndarray, window: list[float], vmax: float, title: str
) -> None:
    """Draw one fluence heatmap (with its own colorbar) into an existing axis."""
    im = ax.imshow(
        grid, origin="lower", extent=window, aspect="equal", cmap="viridis", vmin=0, vmax=vmax
    )
    ax.set_xlabel("x (BEV) [mm]")
    ax.set_ylabel("z (BEV) [mm]")
    ax.set_title(title, fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.8)


def _draw_fluence_diff_axis(
    ax: plt.Axes, fig: Figure, diff: np.ndarray, window: list[float], vmax_abs: float, title: str
) -> None:
    """Draw one fluence-difference heatmap."""
    im = ax.imshow(
        diff,
        origin="lower",
        extent=window,
        aspect="equal",
        cmap="RdBu_r",
        vmin=-vmax_abs,
        vmax=vmax_abs,
    )
    ax.set_xlabel("x (BEV) [mm]")
    ax.set_ylabel("z (BEV) [mm]")
    ax.set_title(title, fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.8)


def _add_fluence_comparison_row(
    fig: Figure,
    gs: GridSpec,
    row: int,
    edges: np.ndarray,
    beam_ix: int,
    beam: BeamSequence,
    stf: SteeringInformation,
    sequence: Sequence,
    unsequenced_result: PlanResult,
) -> list[plt.Axes]:
    """Add the target/delivered/difference fluence comparison as one extra GridSpec row."""
    device_orientation = beam.mlc_sequence[0].device_orientation
    target_grid, window = _native_fluence_map(
        stf.beams[beam_ix], unsequenced_result.fluence[beam.bixel_ix], device_orientation
    )
    delivered_grid, _ = _native_fluence_map(
        stf.beams[beam_ix], sequence.fluence[beam.bixel_ix], device_orientation
    )
    vmax = max(np.nanmax(target_grid), np.nanmax(delivered_grid))

    diff_grid = delivered_grid - target_grid
    finite_abs_diff = np.abs(diff_grid[np.isfinite(diff_grid)])
    max_abs_diff = float(finite_abs_diff.max()) if finite_abs_diff.size else 0.0
    max_abs_diff = max_abs_diff or 1.0

    ax_target = fig.add_subplot(gs[row, edges[0] : edges[1]])
    ax_delivered = fig.add_subplot(gs[row, edges[1] : edges[2]])
    ax_diff = fig.add_subplot(gs[row, edges[2] : edges[3]])

    _draw_fluence_axis(ax_target, fig, target_grid, window, vmax, "Target fluence (optimization)")
    _draw_fluence_axis(
        ax_delivered, fig, delivered_grid, window, vmax, "Delivered fluence (sequencing)"
    )
    _draw_fluence_diff_axis(
        ax_diff, fig, diff_grid, window, max_abs_diff, "Difference (sequencing − optimization)"
    )
    return [ax_target, ax_delivered, ax_diff]


def _beam_dose(result: PlanResult, role: str, quantity: str, beam_ix: int):
    """One beam's dose for the named quantity, or a message saying what is there."""
    doses = (result.beam_quantities or {}).get(quantity)
    if not doses:
        available = ", ".join(sorted(result.beam_quantities or {})) or "none"
        raise ValueError(
            f"The '{role}' result holds no per-beam '{quantity}'. Available: {available}."
        )
    return doses[beam_ix].values


def _add_dose_comparison_row(  # noqa: PLR0913
    fig: Figure,
    gs: GridSpec,
    row: int,
    edges: np.ndarray,
    beam_ix: int,
    unsequenced_result: PlanResult,
    sequenced_result: PlanResult,
    *,
    image_volume: Optional[CT],
    cst: Optional[StructureSet],
    view_slice: Optional[int],
    plane: str,
    overlay_unit: str,
    overlay_cmap: str,
) -> list[plt.Axes]:
    """Add this beam's optimized/delivered/difference dose as one extra GridSpec row."""
    axes = [fig.add_subplot(gs[row, edges[i] : edges[i + 1]]) for i in range(3)]

    before = _beam_dose(unsequenced_result, "unsequenced_result", "physical_dose", beam_ix)
    after = _beam_dose(sequenced_result, "sequenced_result", "physical_dose_sequenced", beam_ix)
    _draw_dose_comparison_panels(
        axes,
        before,
        after,
        image_volume=image_volume,
        cst=cst,
        view_slice=view_slice,
        plane=plane,
        overlay_unit=overlay_unit,
        overlay_cmap=overlay_cmap,
        title_before="Target dose (optimization)",
        title_after="Delivered dose (sequencing)",
        title_diff="Difference (sequencing − optimization)",
    )
    return axes


def _draw_shape_grid(
    fig: Figure,
    beam_ix: int,
    beam: BeamSequence,
    stf: Optional[SteeringInformation],
    leaf_coordinate: str,
    rows: int,
    cols: int,
) -> list[plt.Axes]:
    """Draw one beam's MLC shape grid (one subplot per delivered segment) into ``fig``."""
    n_shapes = beam.num_of_shapes
    gs = GridSpec(rows, cols, figure=fig)
    fig.suptitle(_beam_label(beam_ix, stf))

    created_axes = []
    w_max = float(np.max(beam.shapes_weight)) if n_shapes > 0 else 0.0

    for j in range(n_shapes):
        r, c = divmod(j, cols)
        ax = fig.add_subplot(gs[r, c])
        created_axes.append(ax)
        weight = float(beam.shapes_weight[j])
        ax.set_facecolor(_weight_color(weight, w_max))
        _draw_aperture(ax, beam.mlc_sequence[j], beam.mlc_window, leaf_coordinate)
        ax.set_title(f"Shape {j + 1}, w={weight:.2g}", fontsize=8)

    for j in range(n_shapes, rows * cols):
        r, c = divmod(j, cols)
        ax = fig.add_subplot(gs[r, c])
        ax.axis("off")
        created_axes.append(ax)

    return created_axes


def _draw_comparison_content(  # noqa: PLR0913
    fig: Figure,
    beam_ix: int,
    beam: BeamSequence,
    stf: Optional[SteeringInformation],
    sequence: Sequence,
    unsequenced_result: PlanResult,
    show_fluence: bool,
    show_dose: bool,
    n_cols: int,
    edges: np.ndarray,
    *,
    sequenced_result: Optional[PlanResult] = None,
    image_volume: Optional[CT] = None,
    cst: Optional[StructureSet] = None,
    view_slice: Optional[int] = None,
    plane: str = "axial",
    overlay_unit: str = "Gy",
    overlay_cmap: str = "jet",
) -> list[plt.Axes]:
    """Draw one beam's fluence/dose comparison rows into ``fig``.

    The fluence and dose rows share the same column ``edges`` (see
    `_comparison_layout`), so their pre-/post-sequencing panels line up.
    """
    total_rows = int(show_fluence) + int(show_dose)
    gs = GridSpec(total_rows, n_cols, figure=fig)
    fig.suptitle(f"{_beam_label(beam_ix, stf)} — fluence / dose comparison")

    created_axes = []
    row = 0
    if show_fluence:
        created_axes.extend(
            _add_fluence_comparison_row(
                fig, gs, row, edges, beam_ix, beam, stf, sequence, unsequenced_result
            )
        )
        row += 1

    if show_dose:
        created_axes.extend(
            _add_dose_comparison_row(
                fig,
                gs,
                row,
                edges,
                beam_ix,
                unsequenced_result,
                sequenced_result,
                image_volume=image_volume,
                cst=cst,
                view_slice=view_slice,
                plane=plane,
                overlay_unit=overlay_unit,
                overlay_cmap=overlay_cmap,
            )
        )
        row += 1

    return created_axes


def plot_aperture_info(  # noqa: PLR0913
    result: Union[PlanResult, Sequence],
    *,
    stf: Optional[SteeringInformation] = None,
    unsequenced_result: Optional[PlanResult] = None,
    beam_indices: Optional[Union[int, list[int]]] = None,
    leaf_coordinate: Literal["leaf_num", "physical"] = "leaf_num",
    image_volume: Optional[CT] = None,
    cst: Optional[StructureSet] = None,
    view_slice: Optional[int] = None,
    plane: str = "axial",
    overlay_unit: str = "Gy",
    overlay_cmap: str = "jet",
    save_filename: Optional[str] = None,
    show_plot: bool = True,
) -> list[Figure]:
    """Plot MLC aperture shapes of a sequenced plan.

    When ``unsequenced_result`` (and, for the dose row, ``image_volume``/
    ``cst``) are given, a second, separate figure per beam holds the fluence
    and/or dose comparison rows.

    Parameters
    ----------
    result : PlanResult or Sequence
        A sequenced plan result (``run_sequencing()``'s return value) or its
        ``.sequence`` directly. Must be a ``PlanResult`` (not a bare
        ``Sequence``) for the dose comparison row.
    stf : SteeringInformation, optional
        Steering information used to label each beam with its gantry angle,
        if available, and required (together with ``unsequenced_result``) to
        plot the fluence comparison. Beams are labeled "Beam N" otherwise.
    unsequenced_result : PlanResult, optional
        The plan result *before* sequencing. Required (together with ``stf``)
        to plot the fluence comparison, and (with ``image_volume``/``cst``) to
        plot the dose comparison.
    beam_indices : int or list of int, optional
        Which beams to plot. Defaults to all beams.
    leaf_coordinate : {"leaf_num", "physical"}
        Vertical axis of the aperture: a simple 1..N row index (default) or
        the true physical z position in mm.
    image_volume : CT, optional
        CT image for the dose comparison row's grayscale background. Requires
        ``unsequenced_result``; at least one of ``image_volume``/``cst`` must
        be given to add the dose comparison row.
    cst : StructureSet, optional
        Structure set for the dose comparison row's contour overlays.
    view_slice : int, optional
        Slice index for the dose comparison row. Defaults to the middle slice.
    plane : str
        Slice orientation for the dose comparison row: "axial" (default),
        "coronal", or "sagittal".
    overlay_unit : str
        Physical unit for the dose comparison row's colorbars. Default "Gy".
    overlay_cmap : str
        Colormap for the dose comparison row's optimized/delivered panels.
        Default "jet".
    save_filename : str, optional
        If given, each beam's shape-grid figure is saved to
        ``f"{save_filename}_beam{i}.png"``, and (if present) its comparison
        figure to ``f"{save_filename}_beam{i}_comparison.png"``.
    show_plot : bool
        If True (default), show the figure(s).

    Returns
    -------
    list of matplotlib.figure.Figure
        The shape-grid figure for each plotted beam, followed by its
        fluence/dose comparison figure when ``unsequenced_result`` (and/or
        ``image_volume``/``cst``) is given.
    """
    sequenced_result = result if isinstance(result, PlanResult) else None
    sequence = result.sequence if isinstance(result, PlanResult) else result
    if not isinstance(sequence, Sequence) or not sequence.beam:
        raise ValueError("No sequencing data to visualize.")

    if leaf_coordinate not in ("leaf_num", "physical"):
        raise ValueError("leaf_coordinate must be 'leaf_num' or 'physical'")

    if unsequenced_result is not None and stf is None:
        raise ValueError("stf is required (for ray positions) when unsequenced_result is given.")

    show_dose = image_volume is not None or cst is not None
    if show_dose and unsequenced_result is None:
        raise ValueError(
            "image_volume/cst (dose comparison) require unsequenced_result to be given."
        )
    if show_dose and sequenced_result is None:
        raise ValueError(
            "image_volume/cst (dose comparison) require result to be a PlanResult (not a bare "
            "Sequence), so per-beam dose can be read from result.beam.physical_dose_sequenced."
        )

    if stf is not None:
        stf = validate_stf(stf)

    if beam_indices is None:
        beam_indices = list(range(len(sequence.beam)))
    elif isinstance(beam_indices, int):
        beam_indices = [beam_indices]

    show_fluence = unsequenced_result is not None

    figures = []
    for beam_ix in beam_indices:
        beam: BeamSequence = sequence.beam[beam_ix]

        shape_layout = _shape_grid_layout(beam.num_of_shapes)
        shape_fig = plt.figure(figsize=shape_layout["figsize"])
        _draw_shape_grid(
            shape_fig,
            beam_ix,
            beam,
            stf,
            leaf_coordinate,
            shape_layout["rows"],
            shape_layout["cols"],
        )
        shape_fig.tight_layout()
        if save_filename is not None:
            shape_fig.savefig(f"{save_filename}_beam{beam_ix}.png")
        figures.append(shape_fig)

        if show_fluence or show_dose:
            comp_layout = _comparison_layout(show_fluence, show_dose)
            comp_fig = plt.figure(figsize=comp_layout["figsize"])
            _draw_comparison_content(
                comp_fig,
                beam_ix,
                beam,
                stf,
                sequence,
                unsequenced_result,
                show_fluence,
                show_dose,
                comp_layout["n_cols"],
                comp_layout["edges"],
                sequenced_result=sequenced_result,
                image_volume=image_volume,
                cst=cst,
                view_slice=view_slice,
                plane=plane,
                overlay_unit=overlay_unit,
                overlay_cmap=overlay_cmap,
            )
            comp_fig.tight_layout()
            if save_filename is not None:
                comp_fig.savefig(f"{save_filename}_beam{beam_ix}_comparison.png")
            figures.append(comp_fig)

    if show_plot:
        plt.show()

    return figures
