from typing import Optional, Union, Literal, NamedTuple, TypedDict
from typing_extensions import Unpack
import warnings
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import pint
from pyRadPlan import CT, validate_ct, StructureSet, validate_cst

# Initialize Units
ureg = pint.UnitRegistry()


class PlotSliceKwargs(TypedDict, total=False):
    plane: Union[Literal["axial", "coronal", "sagittal"], int]
    overlay_alpha: float
    overlay_unit: Union[str, pint.Unit, pint.Quantity]
    overlay_rel_threshold: float
    overlay_window: Optional[tuple[float, float]]
    overlay_cmap: str
    contour_line_width: float
    save_filename: Optional[str]
    show_plot: bool
    use_global_max: bool
    image_window: Optional[tuple[float, float]]
    image_cmap: str
    window_mode: Literal["minmax", "centerwidth"]
    ct_window: Optional[tuple[float, float]]
    ax: Optional[plt.Axes]


class _PlotCtx(NamedTuple):
    """Per-axes drawing context shared by all ``_draw_*`` helpers."""

    ax: plt.Axes
    x_plot: np.ndarray
    y_plot: np.ndarray
    transpose: bool


class _OverlayStyle(NamedTuple):
    """Static overlay-display settings (independent of the slice)."""

    unit: pint.Unit
    alpha: float
    rel_threshold: float
    cmap: str
    vmin: Optional[float]
    vmax: Optional[float]


class _SliceConfig(NamedTuple):
    """Bundle of per-figure settings consumed by ``_render_slice``."""

    plane: int
    image_cube: Optional[np.ndarray]
    image_vmin: Optional[float]
    image_vmax: Optional[float]
    image_cmap: str
    cst: Optional[StructureSet]
    contour_line_width: float
    overlay: Optional[np.ndarray]
    overlay_style: _OverlayStyle
    global_overlay_max: Optional[float]
    z_plot: np.ndarray


def _validate_inputs(
    image_volume: Optional[Union[CT, dict, sitk.Image]],
    cst: Optional[Union[StructureSet, dict, list]],
) -> tuple[Optional[CT], Optional[StructureSet], Optional[np.ndarray], Optional[tuple]]:
    """Validate inputs and return validated objects plus the image cube view and array shape."""
    if image_volume is None and cst is None:
        raise ValueError("Nothing to visualize!")

    image_cube = None
    array_shape = None

    if image_volume is not None:
        if not isinstance(image_volume, CT):
            raise NotImplementedError(
                "Other input for image_volume than CT is not implemented yet."
            )
        image_volume = validate_ct(image_volume)
        image_cube = sitk.GetArrayViewFromImage(image_volume.cube_hu)
        array_shape = image_cube.shape

    if cst is not None:
        cst = validate_cst(cst)
        array_shape = cst.ct_image.size[::-1]

    return image_volume, cst, image_cube, array_shape


def _resolve_axes(
    image_volume: Optional[CT], cst: Optional[StructureSet], plane: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (x_plot, y_plot, z_plot) physical-coordinate axes for the chosen plane."""
    axis_orders = {0: ("x", "y", "z"), 1: ("z", "x", "y"), 2: ("z", "y", "x")}
    x_axis, y_axis, z_axis = axis_orders[plane]
    source = image_volume if image_volume is not None else cst.ct_image
    return getattr(source, x_axis), getattr(source, y_axis), getattr(source, z_axis)


def _resolve_view_slices(
    view_slice: Optional[Union[list[int], np.ndarray, int]],
    cst: Optional[StructureSet],
    plane: int,
    array_shape: tuple,
) -> Union[list[int], np.ndarray]:
    """Resolve ``view_slice`` to a sequence (target iso-center or middle slice when None)."""
    if view_slice is None:
        has_target = cst is not None and cst.target_union_voxels().size > 0
        if has_target:  # center on the target's iso-center
            iso_center = cst.target_center_of_mass()
            world_axis = {0: cst.ct_image.z, 1: cst.ct_image.y, 2: cst.ct_image.x}[plane]
            world_coord = iso_center[2 - plane]
            return [int(np.argmin(np.abs(world_axis - world_coord)))]
        return [int(np.round(array_shape[plane] / 2))]
    if isinstance(view_slice, (int, np.integer)):
        return [view_slice]
    return view_slice


def _normalize_overlay_unit(overlay_unit: Union[str, pint.Unit, pint.Quantity]) -> pint.Unit:
    """Normalize ``overlay_unit`` to a ``pint.Unit`` so formatting and predicates are uniform."""
    if isinstance(overlay_unit, str):
        return ureg(overlay_unit).units
    if isinstance(overlay_unit, pint.Quantity):
        return overlay_unit.units
    return overlay_unit


def _compute_image_window(
    image_cube: np.ndarray,
    image_window: Optional[tuple[float, float]],
    window_mode: str,
) -> tuple[float, float]:
    """Resolve (vmin, vmax) for the grayscale image display."""
    if image_window is None:
        return float(np.min(image_cube)), float(np.max(image_cube))
    if window_mode == "minmax":
        if image_window[0] > image_window[1]:
            raise ValueError("Invalid image_window: min must be less than max")
        return image_window[0], image_window[1]
    if window_mode == "centerwidth":
        center, width = image_window
        return center - width / 2, center + width / 2
    raise ValueError(f"Invalid window_mode: {window_mode}")


def _draw_image_slice(
    ctx: _PlotCtx, image_2d: np.ndarray, vmin: float, vmax: float, cmap: str
) -> None:
    if ctx.transpose:
        image_2d = image_2d.T
    ctx.ax.pcolormesh(
        ctx.x_plot, ctx.y_plot, image_2d, cmap=cmap, vmin=vmin, vmax=vmax, shading="nearest"
    )


def _draw_contours(
    ctx: _PlotCtx, cst: StructureSet, slice_indexing: tuple, line_width: float
) -> None:
    for voi in cst.vois:
        mask = sitk.GetArrayViewFromImage(voi.mask)[slice_indexing]
        color = voi.visible_color
        color = [tuple(float(c) / 255.0 for c in voi.visible_color)]

        if ctx.transpose:
            mask = mask.T
        ctx.ax.contour(
            ctx.x_plot,
            ctx.y_plot,
            mask,
            levels=[0.5],
            colors=color,
            linewidths=line_width,
        )


def _draw_overlay(
    ctx: _PlotCtx, overlay_2d: np.ndarray, style: _OverlayStyle, current_max: float
) -> None:
    if ctx.transpose:
        overlay_2d = overlay_2d.T
    vmin = style.vmin if style.vmin is not None else 0
    vmax = style.vmax if style.vmax is not None else current_max
    im = ctx.ax.pcolormesh(
        ctx.x_plot,
        ctx.y_plot,
        overlay_2d,
        cmap=style.cmap,
        alpha=style.alpha * (overlay_2d > style.rel_threshold * current_max),
        vmin=vmin,
        vmax=vmax,
        shading="nearest",
    )
    plt.colorbar(im, ax=ctx.ax, label="" if style.unit.dimensionless else f"{style.unit:~P}")


def _draw_scale_bar(ctx: _PlotCtx, length_mm: float = 50) -> None:
    x_plot, y_plot = ctx.x_plot, ctx.y_plot
    x0 = x_plot[0] - 0.05 * (x_plot[0] - x_plot[-1])
    y0 = y_plot[-1] - 0.05 * (y_plot[-1] - y_plot[0])
    x_end = x0 + length_mm
    ctx.ax.plot([x0, x_end], [y0, y0], "w-", linewidth=3)
    ctx.ax.text(
        (x0 + x_end) / 2,
        y0 - (y_plot[-1] - y_plot[0]) * 0.03,
        f"{length_mm} mm",
        color="w",
        ha="center",
        fontsize=10,
    )


def _apply_deprecated_aliases(
    image_volume: Optional[Union[CT, dict, sitk.Image]],
    ct: Optional[Union[CT, dict, sitk.Image]],
    kwargs: dict,
) -> Optional[Union[CT, dict, sitk.Image]]:
    """Map deprecated ``ct`` / ``ct_window`` arguments onto their replacements (mutates kwargs)."""
    if ct is not None:
        if image_volume is not None:
            raise TypeError("Specify either 'image_volume' or 'ct' (alias), not both.")
        warnings.warn(
            "The 'ct' argument is deprecated; use 'image_volume' instead.",
            DeprecationWarning,
            stacklevel=3,
        )
        image_volume = ct
    if "ct_window" in kwargs:
        if "image_window" in kwargs:
            raise TypeError("Specify either 'image_window' or 'ct_window' (alias), not both.")
        warnings.warn(
            "The 'ct_window' argument is deprecated; use 'image_window' instead.",
            DeprecationWarning,
            stacklevel=3,
        )
        kwargs["image_window"] = kwargs.pop("ct_window")
    return image_volume


def _finalize_figure(save_filename: Optional[str], show_plot: bool) -> None:
    """Apply tight layout, optionally save, optionally show. Used only for internal figures."""
    plt.tight_layout()
    if save_filename is not None:
        try:
            plt.savefig(save_filename)
        except Exception as e:
            print(e)
    if show_plot:
        plt.show()


def _render_slice(ctx: _PlotCtx, slice_idx: int, cfg: _SliceConfig) -> None:
    """Render a single slice (image, contours, overlay, title, scale bar) into ``ctx.ax``."""
    ctx.ax.set_aspect("equal", adjustable="box")
    ctx.ax.invert_yaxis()
    ctx.ax.tick_params(
        axis="both",
        which="both",
        bottom=False,
        top=False,
        labelbottom=False,
        left=False,
        right=False,
        labelleft=False,
    )
    slice_indexing = tuple(slice(None) if j != cfg.plane else slice_idx for j in range(3))

    if cfg.image_cube is not None:
        _draw_image_slice(
            ctx, cfg.image_cube[slice_indexing], cfg.image_vmin, cfg.image_vmax, cfg.image_cmap
        )

    if cfg.cst is not None:
        _draw_contours(ctx, cfg.cst, slice_indexing, cfg.contour_line_width)

    if cfg.overlay is not None:
        overlay_2d = cfg.overlay[slice_indexing]
        current_max = (
            cfg.global_overlay_max if cfg.global_overlay_max is not None else np.max(overlay_2d)
        )
        _draw_overlay(ctx, overlay_2d, cfg.overlay_style, current_max)

    if cfg.image_cube is not None:
        ctx.ax.set_title(f"Slice ID = {slice_idx} at {cfg.z_plot[slice_idx]:.1f} mm")
        _draw_scale_bar(ctx)
    else:
        ctx.ax.set_title(f"Slice ID = {slice_idx}")


def plot_slice(
    image_volume: Optional[Union[CT, dict, sitk.Image]] = None,
    cst: Optional[Union[StructureSet, dict, list]] = None,
    overlay: Optional[Union[sitk.Image, np.ndarray]] = None,
    view_slice: Optional[Union[list[int], np.ndarray, int]] = None,
    *,
    ct: Optional[Union[CT, dict, sitk.Image]] = None,
    **kwargs: Unpack[PlotSliceKwargs],
):
    """Plot one or multiple slices of the image_volume with a given overlay.

    Parameters
    ----------
    image_volume : image_volume
        The image_volume object.
    cst : StructureSet
        The StructureSet object.
    overlay : sitk.Image or np.ndarray
        The overlay image to visualize.
    view_slice : List[int], array, int or None
        Slice indices to visualize.

    **kwargs
    ----------
    plane : str or int
        The plane to visualize. Can be "axial", "coronal", or "sagittal". Default is "axial".
    overlay_alpha : float
        The alpha value for the overlay. Default is 0.5.
    overlay_unit : str, pint.Unit, or pint.Quantity
        The unit of the overlay. Default is "".
    overlay_rel_threshold : float
        The relative threshold for the overlay. Default is 0.01.
    contour_line_width : float
        The line width for the contour lines. Default is 1.0.
    save_filename : str
        The filename to save the plot. Default is None.
    show_plot : bool
        If True, show the plot. Default is True.
    use_global_max : bool
        If True, use the overlay's global maximum for scaling. Default is False.
    image_window: Optional[tuple[float, float]]
        Minimum and maximum image value for image_volume visualization
    image_cmap : str
        Colormap for the image_volume. Default is "bone".
    window_mode: Literal["minmax", "centerwidth"]
        Windowing mode for image_volume visualization. Can be "minmax" or "centerwidth".
        Default is "minmax". In "minmax" mode, image_window is interpreted as (min, max).
        In "centerwidth" mode, image_window is interpreted as (center, width).
    overlay_window : Optional[tuple[float, float]]
        Value window (vmin, vmax) for the overlay colormap. Defaults to (0, max) when None.
    overlay_cmap : str
        Colormap for the overlay. Default is "jet".
    ax : plt.Axes, optional
        Existing axes to plot on. If provided, only single slice plotting is supported.
    """
    image_volume = _apply_deprecated_aliases(image_volume, ct, kwargs)

    plane = kwargs.get("plane", "axial")
    overlay_unit = _normalize_overlay_unit(kwargs.get("overlay_unit", pint.Unit("")))
    save_filename = kwargs.get("save_filename", None)
    show_plot = kwargs.get("show_plot", True)
    use_global_max = kwargs.get("use_global_max", False)
    ax = kwargs.get("ax", None)

    plane = {"axial": 0, "coronal": 1, "sagittal": 2}.get(plane, plane)
    if not isinstance(plane, int) or not 0 <= plane <= 2:
        raise ValueError("Invalid plane")

    image_volume, cst, image_cube, array_shape = _validate_inputs(image_volume, cst)
    x_plot, y_plot, z_plot = _resolve_axes(image_volume, cst, plane)
    view_slice = _resolve_view_slices(view_slice, cst, plane, array_shape)

    image_vmin = image_vmax = None
    if image_cube is not None:
        image_vmin, image_vmax = _compute_image_window(
            image_cube, kwargs.get("image_window", None), kwargs.get("window_mode", "minmax")
        )

    global_overlay_max = None
    if overlay is not None:
        if isinstance(overlay, sitk.Image):
            overlay = sitk.GetArrayViewFromImage(overlay)
        if use_global_max:
            global_overlay_max = float(np.max(overlay))

    overlay_window = kwargs.get("overlay_window", None)
    overlay_vmin = overlay_window[0] if overlay_window is not None else None
    overlay_vmax = overlay_window[1] if overlay_window is not None else None

    cfg = _SliceConfig(
        plane=plane,
        image_cube=image_cube,
        image_vmin=image_vmin,
        image_vmax=image_vmax,
        image_cmap=kwargs.get("image_cmap", "bone"),
        cst=cst,
        contour_line_width=kwargs.get("contour_line_width", 1.0),
        overlay=overlay,
        overlay_style=_OverlayStyle(
            unit=overlay_unit,
            alpha=kwargs.get("overlay_alpha", 0.5),
            rel_threshold=kwargs.get("overlay_rel_threshold", 0.01),
            cmap=kwargs.get("overlay_cmap", "jet"),
            vmin=overlay_vmin,
            vmax=overlay_vmax,
        ),
        global_overlay_max=global_overlay_max,
        z_plot=z_plot,
    )

    if ax is not None:
        if len(view_slice) > 1:
            raise ValueError("External axes only supports single slice plotting")
        axes = [ax]
    else:
        num_slices = len(view_slice)
        cols = int(np.ceil(np.sqrt(num_slices)))
        rows = int(np.ceil(num_slices / cols))
        _, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
        axes = np.array(axes).flatten()

    transpose = plane != 0
    for i, slice_idx in enumerate(view_slice):
        _render_slice(_PlotCtx(axes[i], x_plot, y_plot, transpose), slice_idx, cfg)

    if ax is None:  # Only manage layout and saving for internal figures
        _finalize_figure(save_filename, show_plot)
