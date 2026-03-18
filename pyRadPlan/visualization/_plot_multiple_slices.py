from typing import Optional, Union, Literal, TypedDict
from typing_extensions import Unpack
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import pint
import re
from pyRadPlan import CT, StructureSet
from pyRadPlan.visualization._plot_slice import plot_slice

# Initialize Units
ureg = pint.UnitRegistry()


class PlotMultipleSlicesKwargs(TypedDict, total=False):
    plane: Union[Literal["axial", "coronal", "sagittal"], int]
    overlay_alpha: float
    overlay_unit: Union[str, pint.Unit, list[Union[str, pint.Unit]]]
    overlay_rel_threshold: float
    contour_line_width: float
    save_filename: Optional[str]
    show_plot: bool
    use_global_max: bool
    overlay_titles: Optional[list[str]]


def plot_multiple_slices(
    ct: Optional[Union[CT, dict]] = None,
    cst: Optional[Union[StructureSet, dict, list]] = None,
    overlays: Optional[Union[sitk.Image, np.ndarray]] = None,
    view_slice: Optional[Union[list[int], np.ndarray, int]] = None,
    **kwargs: Unpack[PlotMultipleSlicesKwargs],
):
    """Plot multiple distributions for given slices with dynamic figure sizing.

    Parameters
    ----------
    ct : CT
        The CT object.
    cst : StructureSet
        The StructureSet object.
    overlays : list of sitk.Image or np.ndarray
        List of overlay images to visualize.
    view_slice : List[int], array, int or None
        Slice indices to visualize.

    **kwargs
    ----------
    plane : str or int
        The plane to visualize. Can be "axial", "coronal", or "sagittal". Default is "axial".
    overlay_alpha : float
        The alpha value for the overlay. Default is 0.5.
    overlay_unit : list of str or pint.Unit
        List of units for each overlay. Default is "".
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
    overlay_titles : list of str, optional
        Custom titles for each overlay type. Default is None.
    """
    overlay_unit = kwargs.get("overlay_unit", pint.Unit(""))
    save_filename = kwargs.get("save_filename", None)
    show_plot = kwargs.get("show_plot", True)
    overlay_titles = kwargs.get("overlay_titles", None)

    if ct is None and overlays is None:
        raise ValueError("At least one of 'ct' or 'overlays' must be provided.")
    # Ensure inputs are lists/arrays
    view_slice = [view_slice] if not isinstance(view_slice, (list, np.ndarray)) else view_slice
    overlays = [overlays] if not isinstance(overlays, list) else overlays

    if not isinstance(overlay_unit, list):
        overlay_unit = [overlay_unit] * len(overlays)
    elif len(overlay_unit) != len(overlays):
        raise ValueError("Length of 'overlay_unit' must match the number of overlays.")

    n_slices = len(view_slice) if view_slice is not None else 1
    n_overlays = len(overlays) if overlays is not None else 1

    # TODO: Not sure if figsize per subplot should be an attribute. Add above if needed.
    figsize_per_subplot = (5, 5)
    # Calculate dynamic figure size
    width = figsize_per_subplot[0] * n_slices
    height = figsize_per_subplot[1] * n_overlays
    figsize = (width, height)

    fig, axes = plt.subplots(n_overlays, n_slices, figsize=figsize)

    # Normalize axes to shape (n_overlays, n_slices)
    axes = np.asarray(axes, dtype=object).reshape(n_overlays, n_slices)

    for i, overlay in enumerate(overlays):
        for j, slice_idx in enumerate(view_slice):
            # Using the already existing plot_slice function to plot each overlay
            # Prepare kwargs for plot_slice
            plot_slice_kwargs = kwargs.copy()
            plot_slice_kwargs.update(
                {
                    "overlay_unit": overlay_unit[i],
                    "show_plot": False,
                    "ax": axes[i, j],
                }
            )
            # Remove arguments that are not in plot_slice kwargs or handled explicitly
            plot_slice_kwargs.pop("overlay_titles", None)
            plot_slice_kwargs.pop("save_filename", None)  # Handled at the end

            plot_slice(ct=ct, cst=cst, overlay=overlay, view_slice=slice_idx, **plot_slice_kwargs)

            # Try to extract slice index from the title set by plot_slice
            # Title format: "Slice z={slice_idx}"
            idx = (
                int(re.search(r"Slice [xyz]=(\d+)", axes[i, j].get_title()).group(1))
                if slice_idx is None
                else slice_idx
            )

            # Set title
            title = (
                f"{overlay_titles[i]} - Slice {idx}"
                if overlay_titles and i < len(overlay_titles)
                else f"Overlay {i + 1} - Slice {idx}"
            )

            axes[i, j].set_title(title)

    plt.tight_layout()
    if save_filename:
        plt.savefig(save_filename)
    if show_plot:
        plt.show()
