"""Comparison of two dose distributions (before/after), with their difference."""

from typing import Optional
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from pyRadPlan.ct import CT
from pyRadPlan.cst import StructureSet
from pyRadPlan.result import PlanResult
from pyRadPlan.visualization._plot_slice import plot_slice, _resolve_view_slices, _validate_inputs


_DIFF_CMAP = "RdBu_r"


def _diff_slice_max_abs(
    diff_arr: np.ndarray,
    image_volume: Optional[CT],
    cst: Optional[StructureSet],
    view_slice: Optional[int],
    plane: str,
) -> float:
    """Max absolute difference on the slice that will actually be displayed."""
    plane_idx = {"axial": 0, "coronal": 1, "sagittal": 2}.get(plane, plane)
    try:
        _, validated_cst, _, _ = _validate_inputs(image_volume, cst)
        slice_idx = _resolve_view_slices(view_slice, validated_cst, plane_idx, diff_arr.shape)[0]
    except (ValueError, AttributeError):
        slice_idx = (
            int(view_slice)
            if isinstance(view_slice, (int, np.integer))
            else int(round(diff_arr.shape[plane_idx] / 2))
        )
    indexer = [slice(None)] * diff_arr.ndim
    indexer[plane_idx] = slice_idx
    return float(np.max(np.abs(diff_arr[tuple(indexer)]))) or 1.0


def _draw_dose_comparison_panels(
    axes: list[plt.Axes],
    before: sitk.Image,
    after: sitk.Image,
    *,
    image_volume: Optional[CT],
    cst: Optional[StructureSet],
    view_slice: Optional[int],
    plane: str,
    overlay_unit: str,
    overlay_cmap: str,
    title_before: str = "Before",
    title_after: str = "After",
    title_diff: Optional[str] = None,
) -> None:
    """Draw the before/after/difference dose panels into three existing axes."""
    diff_arr = sitk.GetArrayFromImage(after) - sitk.GetArrayFromImage(before)
    diff = sitk.GetImageFromArray(diff_arr)
    diff.CopyInformation(after)
    max_abs_diff = _diff_slice_max_abs(diff_arr, image_volume, cst, view_slice, plane)

    if title_diff is None:
        title_diff = f"Difference ({title_after} − {title_before})"

    plot_slice(
        image_volume=image_volume,
        cst=cst,
        overlay=before,
        view_slice=view_slice,
        plane=plane,
        overlay_unit=overlay_unit,
        overlay_cmap=overlay_cmap,
        ax=axes[0],
        show_plot=False,
    )
    axes[0].set_title(title_before, fontsize=9)

    plot_slice(
        image_volume=image_volume,
        cst=cst,
        overlay=after,
        view_slice=view_slice,
        plane=plane,
        overlay_unit=overlay_unit,
        overlay_cmap=overlay_cmap,
        ax=axes[1],
        show_plot=False,
    )
    axes[1].set_title(title_after, fontsize=9)

    plot_slice(
        image_volume=image_volume,
        cst=cst,
        overlay=diff,
        view_slice=view_slice,
        plane=plane,
        overlay_unit=overlay_unit,
        overlay_cmap=_DIFF_CMAP,
        overlay_window=(-max_abs_diff, max_abs_diff),
        ax=axes[2],
        show_plot=False,
    )
    axes[2].set_title(title_diff, fontsize=9)


def plot_dose_comparison(
    before: PlanResult,
    after: PlanResult,
    *,
    title_before: str = "Before",
    title_after: str = "After",
    title_diff: Optional[str] = None,
    image_volume: Optional[CT] = None,
    cst: Optional[StructureSet] = None,
    view_slice: Optional[int] = None,
    plane: str = "axial",
    overlay_unit: str = "Gy",
    overlay_cmap: str = "jet",
    save_filename: Optional[str] = None,
    show_plot: bool = True,
) -> Figure:
    """Compare two dose distributions and their difference on one slice.

    One figure, three panels on the same slice: ``before``'s dose, ``after``'s
    dose, and their difference. At least one of ``image_volume``/``cst`` must
    be given, same as for `~pyRadPlan.visualization.plot_slice`.

    Parameters
    ----------
    before : PlanResult
        The "before" dose.
    after : PlanResult
        The "after" dose.
    title_before : str
        Title of the first panel. Default "Before".
    title_after : str
        Title of the second panel. Default "After".
    title_diff : str, optional
        Title of the difference panel. Defaults to
        ``f"Difference ({title_after} − {title_before})"``.
    image_volume : CT, optional
        CT image for the grayscale background.
    cst : StructureSet, optional
        Structure set for contour overlays.
    view_slice : int, optional
        Slice index to display. Defaults to the middle slice.
    plane : str
        "axial" (default), "coronal", or "sagittal".
    overlay_unit : str
        Physical unit for the dose colorbars. Default "Gy".
    overlay_cmap : str
        Colormap for the two dose panels. Default "jet".
    save_filename : str, optional
        If given, the figure is saved to this path.
    show_plot : bool
        If True (default), show the figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    before_dose = before.physical_dose.values
    after_dose = after.physical_dose.values

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    _draw_dose_comparison_panels(
        list(axes),
        before_dose,
        after_dose,
        image_volume=image_volume,
        cst=cst,
        view_slice=view_slice,
        plane=plane,
        overlay_unit=overlay_unit,
        overlay_cmap=overlay_cmap,
        title_before=title_before,
        title_after=title_after,
        title_diff=title_diff,
    )

    fig.tight_layout()
    if save_filename is not None:
        fig.savefig(save_filename)
    if show_plot:
        plt.show()

    return fig
