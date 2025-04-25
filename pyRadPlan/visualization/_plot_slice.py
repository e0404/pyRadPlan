from typing import Optional, Union, Literal

import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import pint
from pyRadPlan import CT, validate_ct, StructureSet, validate_cst

# Initialize Units
ureg = pint.UnitRegistry()


def plot_slice(  # noqa: PLR0913
    ct: Optional[Union[CT, dict]] = None,
    cst: Optional[Union[StructureSet, dict, list]] = None,
    overlay: Optional[Union[sitk.Image, np.ndarray]] = None,
    view_slice: Optional[Union[list[int], np.ndarray, int]] = None,
    plane: Union[Literal["axial", "coronal", "sagittal"], int] = "axial",
    overlay_alpha: float = 0.5,
    overlay_unit: Union[str, pint.Unit] = pint.Unit(""),
    overlay_rel_threshold: float = 0.01,
    contour_line_width: float = 1.0,
    save_filename: Optional[str] = None,
    show_plot: bool = True,
    use_global_max: bool = False,
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
    plane : str or int
        The plane to visualize. Can be "axial", "coronal", or "sagittal".
    overlay_alpha : float
        The alpha value for the overlay.
    overlay_unit : str or pint.Unit
        The unit of the overlay.
    overlay_rel_threshold : float
        The relative threshold for the overlay.
    contour_line_width : float
        The line width for the contour lines.
    save_filename : str
        The filename to save the plot. Default is None
    show_plot : bool
        If True, show the plot. Default is True.
    use_global_max : bool
        If True, use the overlay's global maximum for sclaing
    """

    if ct is not None:
        ct = validate_ct(ct)
        cube_hu = sitk.GetArrayViewFromImage(ct.cube_hu)
        array_shape = cube_hu.shape

    if cst is not None:
        cst = validate_cst(cst)
        array_shape = cst.ct_image.size[::-1]

    if ct is None and cst is None:
        raise ValueError("Nothing to visualize!")

    plane = {"axial": 0, "coronal": 1, "sagittal": 2}.get(plane, plane)
    if not isinstance(plane, int) or not 0 <= plane <= 2:
        raise ValueError("Invalid plane")

    if isinstance(overlay_unit, str):
        overlay_unit = ureg(overlay_unit)

    if view_slice is None:
        view_slice = [int(np.round(array_shape[plane] / 2))]
    elif isinstance(view_slice, int):
        view_slice = [view_slice]

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
        ax = axes[i]
        slice_indexing = tuple(slice(None) if j != plane else slice_idx for j in range(3))

        ax.tick_params(
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
            ax.imshow(cube_hu[slice_indexing], cmap="gray")

        # Visualize the VOIs from the StructureSet.
        if cst is not None:
            for v, voi in enumerate(cst.vois):
                mask = sitk.GetArrayViewFromImage(voi.mask)
                cmap = plt.colormaps["cool"]
                color = cmap(v / len(cst.vois))
                ax.contour(
                    mask[slice_indexing],
                    levels=[0.5],
                    colors=[color],
                    linewidths=contour_line_width,
                )

        # Visualize the overlay.
        if overlay is not None:
            if use_global_max:
                current_max = global_overlay_max
            else:
                current_max = np.max(overlay[slice_indexing])
            im_overlay = ax.imshow(
                overlay[slice_indexing],
                cmap="jet",
                interpolation="nearest",
                alpha=overlay_alpha
                * (overlay[slice_indexing] > overlay_rel_threshold * current_max),
                vmin=0,
                vmax=current_max,
            )
            plt.colorbar(im_overlay, ax=ax, label=f"{overlay_unit:~P}")

        ax.set_title(f"Slice z={slice_idx}")

        # Add a scale bar if a CT is provided.
        if ct is not None:
            disp_im = cube_hu[slice_indexing]

            spacing = ct.cube_hu.GetSpacing()
            if plane in (0, 1):  # axial or coronal: horizontal axis uses x spacing.
                scale_spacing = spacing[0]
            else:  # sagittal: horizontal axis uses y spacing.
                scale_spacing = spacing[1]
            chosen_length_mm = 50  # define a 50 mm scale bar
            pixel_length = chosen_length_mm / scale_spacing

            h, w = disp_im.shape
            # placing at a defined space
            x0 = w * 0.05
            y0 = h * 0.95
            x_end = x0 + pixel_length
            ax.plot([x0, x_end], [y0, y0], "w-", linewidth=3)
            ax.text(
                (x0 + x_end) / 2,
                y0 - h * 0.03,
                f"{chosen_length_mm} mm",
                color="w",
                ha="center",
                fontsize=10,
            )

    # delete unsused axes
    for j in range(num_slices, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    if save_filename is not None:
        try:
            plt.savefig(save_filename)
        except Exception as e:
            print(e)
    if show_plot:
        plt.show()
