"""Visualization of MLC aperture shapes from a sequenced plan result."""

from typing import Optional, Union, Literal
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec

from pyRadPlan.ct import CT
from pyRadPlan.cst import StructureSet
from pyRadPlan.machines import MLC
from pyRadPlan.result import PlanResult, Sequence
from pyRadPlan.result._beam_sequence import BeamSequence
from pyRadPlan.stf import Beam, SteeringInformation, validate_stf
from pyRadPlan.visualization._plot_dose_comparison import _draw_dose_comparison_panels


def _leaf_row_bounds(mlc: MLC, leaf_coordinate: str) -> tuple[np.ndarray, np.ndarray]:
    """Lower/upper vertical edge of every active leaf row of one MLC segment.

    For ``"physical"``, rows sit at their true z position with height
    ``mlc.leaf_width``. For ``"leaf_num"``, rows are relabeled 1..N from the
    lowest to the highest z.
    """
    n = len(mlc.leaf_positions)
    if leaf_coordinate == "physical":
        centers = np.asarray(mlc.leaf_position_boundaries, dtype=float)
        half = mlc.leaf_width / 2.0
    else:
        centers = np.arange(1, n + 1, dtype=float)
        half = 0.5
    return centers - half, centers + half


def _weight_color(weight: float, w_max: float) -> tuple:
    """Open-aperture colour encoding a segment's weight: light grey (low) to blue (high)."""
    frac = min(max(weight / w_max, 0.0), 1.0) if w_max > 0 else 0.0
    light = np.array([0.92, 0.92, 0.92])
    blue = np.array([0.15, 0.35, 0.75])
    return tuple(light + frac * (blue - light))


def _draw_aperture(ax: plt.Axes, mlc: MLC, mlc_window: list[float], leaf_coordinate: str) -> None:
    """Draw the blocked (closed) leaf regions of one MLC segment into ax."""
    x_min, x_max = mlc_window[0], mlc_window[1]
    y_low, y_high = _leaf_row_bounds(mlc, leaf_coordinate)

    for (left, right), lo, hi in zip(mlc.leaf_positions, y_low, y_high):
        # filled blocked (closed) regions on either side of the aperture
        ax.add_patch(
            Rectangle(
                (x_min, lo),
                left - x_min,
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
                x_max - right,
                hi - lo,
                facecolor="dimgray",
                edgecolor="black",
                linewidth=0.5,
                zorder=3,
            )
        )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_low.min(), y_high.max())
    ax.set_xlabel("horiz. pos. [mm]")
    ax.set_ylabel("vert. pos. [mm]" if leaf_coordinate == "physical" else "leaf pair #")


def _beam_label(beam_index: int, stf: Optional[SteeringInformation]) -> str:
    if stf is not None and beam_index < len(stf.beams):
        angle = getattr(stf.beams[beam_index], "gantry_angle", None)
        if angle is not None:
            return f"Beam {beam_index} ({float(angle):.0f}°)"
    return f"Beam {beam_index}"


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
        ``(grid, window)``: ``grid`` is a ``(dim_z, dim_x)`` array (``nan``
        where no ray exists) and ``window`` is
        ``[x_min, x_max, z_min, z_max]`` in mm, matching ``mlc_window``.
    """
    travel_axis, stack_axis = (0, 2) if device_orientation == "X" else (2, 0)
    x_pos = np.array([ray.ray_pos_bev[travel_axis] for ray in beam.rays])
    z_pos = np.array([ray.ray_pos_bev[stack_axis] for ray in beam.rays])
    bw = beam.bixel_width

    min_x, max_x = x_pos.min(), x_pos.max()
    min_z, max_z = z_pos.min(), z_pos.max()
    dim_x = int((max_x - min_x) / bw + 1)
    dim_z = int((max_z - min_z) / bw + 1)

    grid = np.full((dim_z, dim_x), np.nan)
    x_idx = ((x_pos - min_x) / bw).astype(int)
    z_idx = ((z_pos - min_z) / bw).astype(int)
    grid[z_idx, x_idx] = values

    window = [min_x - 0.5 * bw, max_x + 0.5 * bw, min_z - 0.5 * bw, max_z + 0.5 * bw]
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


def _add_dose_comparison_row(
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

    before = unsequenced_result.beam.physical_dose[beam_ix].values
    after = sequenced_result.beam.physical_dose[beam_ix].values
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


def _draw_comparison_content(
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


def plot_aperture_info(
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
            "Sequence), so per-beam dose can be read from result.beam.physical_dose."
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
