from typing import Optional, Union, Literal, TypedDict
from typing_extensions import Unpack
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
    overlay_unit: Union[str, pint.Unit]
    overlay_rel_threshold: float
    contour_line_width: float
    save_filename: Optional[str]
    show_plot: bool
    use_global_max: bool
    ct_window: Optional[tuple[float, float]]
    ax: Optional[plt.Axes]


def plot_slice(  # noqa: PLR0913
    ct: Optional[Union[CT, dict]] = None,
    cst: Optional[Union[StructureSet, dict, list]] = None,
    overlay: Optional[Union[sitk.Image, np.ndarray]] = None,
    view_slice: Optional[Union[list[int], np.ndarray, int]] = None,
    **kwargs: Unpack[PlotSliceKwargs],
):
    """Plot one or multiple slices of the CT with a given overlay.

    Parameters
    ----------
    ct : CT
        The CT object.
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
    overlay_unit : str or pint.Unit
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
    ct_window : tuple[float, float]
        The window for the CT. Default is None.
    ax : plt.Axes, optional
        Existing axes to plot on. If provided, only single slice plotting is supported.
    """
    plane = kwargs.get("plane", "axial")
    overlay_alpha = kwargs.get("overlay_alpha", 0.5)
    overlay_unit = kwargs.get("overlay_unit", pint.Unit(""))
    overlay_rel_threshold = kwargs.get("overlay_rel_threshold", 0.01)
    contour_line_width = kwargs.get("contour_line_width", 1.0)
    save_filename = kwargs.get("save_filename", None)
    show_plot = kwargs.get("show_plot", True)
    use_global_max = kwargs.get("use_global_max", False)
    ct_window = kwargs.get("ct_window", None)
    ax = kwargs.get("ax", None)

    if ct is not None:
        ct = validate_ct(ct)
        cube_hu = sitk.GetArrayViewFromImage(ct.cube_hu)
        array_shape = cube_hu.shape

    if cst is not None:
        cst = validate_cst(cst)
        array_shape = cst.ct_image.size[::-1]

    if ct is None and cst is None:
        raise ValueError("Nothing to visualize!")

    vmin, vmax = ct_window if ct_window is not None else (None, None)

    plane = {"axial": 0, "coronal": 1, "sagittal": 2}.get(plane, plane)
    if not isinstance(plane, int) or not 0 <= plane <= 2:
        raise ValueError("Invalid plane")

    overlay_unit = ureg(overlay_unit) if isinstance(overlay_unit, str) else overlay_unit

    if view_slice is None:
        view_slice = [int(np.round(array_shape[plane] / 2))]
    elif isinstance(view_slice, int) or isinstance(view_slice, np.integer):
        view_slice = [view_slice]

    # Handle external axes (for side-by-side plotting)
    if ax is not None:
        if len(view_slice) > 1:
            raise ValueError("External axes only supports single slice plotting")
        axes = [ax]
        fig = ax.figure
    else:
        num_slices = len(view_slice)
        cols = int(np.ceil(np.sqrt(num_slices)))
        rows = int(np.ceil(num_slices / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
        axes = np.array(axes).flatten()

    # Prepare overlay if provided.
    if overlay is not None:
        if isinstance(overlay, sitk.Image):
            overlay = sitk.GetArrayViewFromImage(overlay)
        if use_global_max:
            global_overlay_max = np.max(overlay)

    for i, slice_idx in enumerate(view_slice):
        current_ax = axes[i]
        slice_indexing = tuple(slice(None) if j != plane else slice_idx for j in range(3))

        current_ax.tick_params(
            axis="both",
            which="both",
            bottom=False,
            top=False,
            labelbottom=False,
            left=False,
            right=False,
            labelleft=False,
        )

        # Visualize the CT slice.
        if ct is not None:
            current_ax.imshow(cube_hu[slice_indexing], cmap="gray", vmin=vmin, vmax=vmax)

        # Visualize the VOIs from the StructureSet.
        if cst is not None:
            for v, voi in enumerate(cst.vois):
                mask = sitk.GetArrayViewFromImage(voi.mask)
                cmap = plt.colormaps["cool"]
                color = cmap(v / len(cst.vois))
                current_ax.contour(
                    mask[slice_indexing],
                    levels=[0.5],
                    colors=[color],
                    linewidths=contour_line_width,
                )

        # Visualize the overlay.
        if overlay is not None:
            current_max = global_overlay_max if use_global_max else np.max(overlay[slice_indexing])
            im_overlay = current_ax.imshow(
                overlay[slice_indexing],
                cmap="jet",
                interpolation="nearest",
                alpha=overlay_alpha
                * (overlay[slice_indexing] > overlay_rel_threshold * current_max),
                vmin=0,
                vmax=current_max,
            )
            plt.colorbar(im_overlay, ax=current_ax, label=f"{overlay_unit:~P}")

        current_ax.set_title(f"Slice z={slice_idx}")

        # Add a scale bar if a CT is provided.
        if ct is not None:
            disp_im = cube_hu[slice_indexing]

            spacing = ct.cube_hu.GetSpacing()

            # axial or coronal: horizontal axis uses x spacing.
            # sagittal: horizontal axis uses y spacing.
            scale_spacing = spacing[0] if plane in (0, 1) else spacing[1]

            chosen_length_mm = 50  # define a 50 mm scale bar
            pixel_length = chosen_length_mm / scale_spacing

            h, w = disp_im.shape
            # placing at a defined space
            x0 = w * 0.05
            y0 = h * 0.95
            x_end = x0 + pixel_length
            current_ax.plot([x0, x_end], [y0, y0], "w-", linewidth=3)
            current_ax.text(
                (x0 + x_end) / 2,
                y0 - h * 0.03,
                f"{chosen_length_mm} mm",
                color="w",
                ha="center",
                fontsize=10,
            )

    # delete unused axes (only for internal figure creation)
    # if ax is None:
    #     for j in range(len(view_slice), len(axes)):
    #         axes[j].axis("off")

    if ax is None:  # Only manage layout and saving for internal figures
        plt.tight_layout()
        if save_filename is not None:
            try:
                plt.savefig(save_filename)
            except Exception as e:
                print(e)
        if show_plot:
            plt.show()
