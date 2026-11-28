"""Comparison of two dose distributions (before/after), with their difference."""

import os
from typing import Literal, Optional, Union
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from pyRadPlan.ct import CT
from pyRadPlan.cst import StructureSet
from pyRadPlan.result import PlanResult, QuantityDistribution
from pyRadPlan.sequencing._bev import beam_label
from pyRadPlan.stf import SteeringInformation
from pyRadPlan.visualization._plot_slice import plot_slice, _resolve_view_slices, _validate_inputs


_DIFF_CMAP = "RdBu_r"


def _beam_filename(save_filename: str, beam_index: int) -> str:
    """``dose.png`` -> ``dose_beam0.png``, keeping the extension usable by savefig."""
    root, ext = os.path.splitext(save_filename)
    return f"{root}_beam{beam_index}{ext}"


def _quantity(result: PlanResult, role: str, quantity: str) -> QuantityDistribution:
    """One named quantity of a plan result."""
    dist = result.quantities.get(quantity) if result.quantities else None
    if dist is None:
        available = ", ".join(sorted(result.quantities)) if result.quantities else "none"
        raise ValueError(
            f"The '{role}' result has no quantity '{quantity}'. Available: {available}."
        )
    return dist


def _beam_doses(result: PlanResult, role: str, quantity: str) -> list[QuantityDistribution]:
    """Per-beam dose of one plan result, for the named quantity."""
    doses = result.beam_quantities.get(quantity) if result.beam_quantities else None
    if not doses:
        raise ValueError(
            f"The '{role}' result holds no per-beam '{quantity}'. beam_indices requires "
            "a dose calculation that stored per-beam doses under that name."
        )
    return doses


def _resolve_beam_indices(
    beam_indices: Union[int, list[int], Literal["all"]], n_beams: int
) -> list[int]:
    """Normalize the ``beam_indices`` argument into a list of valid beam indices."""
    if isinstance(beam_indices, str):
        if beam_indices != "all":
            raise ValueError("beam_indices must be an int, a list of int, or 'all'.")
        return list(range(n_beams))

    indices = (
        [int(beam_indices)]
        if isinstance(beam_indices, (int, np.integer))
        else [int(i) for i in beam_indices]
    )
    out_of_range = [i for i in indices if not 0 <= i < n_beams]
    if out_of_range:
        raise IndexError(f"beam_indices {out_of_range} out of range for {n_beams} beam(s).")
    return indices


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


def _draw_dose_comparison_panels(  # noqa: PLR0913
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


def plot_dose_comparison(  # noqa: PLR0913
    before: PlanResult,
    after: PlanResult,
    *,
    title_before: str = "Before",
    title_after: str = "After",
    title_diff: Optional[str] = None,
    quantity_before: str = "physical_dose",
    quantity_after: str = "physical_dose",
    beam_indices: Optional[Union[int, list[int], Literal["all"]]] = None,
    stf: Optional[SteeringInformation] = None,
    image_volume: Optional[CT] = None,
    cst: Optional[StructureSet] = None,
    view_slice: Optional[int] = None,
    plane: str = "axial",
    overlay_unit: str = "Gy",
    overlay_cmap: str = "jet",
    save_filename: Optional[str] = None,
    show_plot: bool = True,
) -> list[Figure]:
    """Compare two dose distributions and their difference on one slice.

    The first figure holds three panels on the same slice: ``before``'s dose,
    ``after``'s dose, and their difference. At least one of
    ``image_volume``/``cst`` must be given, same as for
    `~pyRadPlan.visualization.plot_slice`.

    With ``beam_indices``, each selected beam gets its own, separate figure of
    the same three panels, drawn from the per-beam doses of the selected quantities.

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
    quantity_before, quantity_after : str
        Which quantity to take from each result, both defaulting to
        ``"physical_dose"``.
    beam_indices : int, list of int or "all", optional
        Beams to additionally show one comparison figure each for; "all" takes
        every beam. Requires per-beam doses in both results. Defaults to None,
        i.e. only the total dose is shown.
    stf : SteeringInformation, optional
        Steering information used to title the per-beam figures with their
        gantry angle, if available. They are titled "Beam N" otherwise.
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
        If given, the total-dose figure is saved to this path, and each
        per-beam figure next to it with a ``_beam{i}`` suffix on its stem.
    show_plot : bool
        If True (default), show the figure(s).

    Returns
    -------
    list of matplotlib.figure.Figure
        The total-dose figure, followed by one figure per beam in
        ``beam_indices``.
    """
    before_dose = _quantity(before, "before", quantity_before).values
    after_dose = _quantity(after, "after", quantity_after).values

    beam_doses: list[tuple[int, sitk.Image, sitk.Image]] = []
    if beam_indices is not None:
        before_beams = _beam_doses(before, "before", quantity_before)
        after_beams = _beam_doses(after, "after", quantity_after)
        if len(before_beams) != len(after_beams):
            raise ValueError(
                "'before' and 'after' hold per-beam doses for a different number of beams "
                f"({len(before_beams)} vs. {len(after_beams)})."
            )
        beam_doses = [
            (beam_ix, before_beams[beam_ix].values, after_beams[beam_ix].values)
            for beam_ix in _resolve_beam_indices(beam_indices, len(before_beams))
        ]

    panel_kwargs = {
        "image_volume": image_volume,
        "cst": cst,
        "view_slice": view_slice,
        "plane": plane,
        "overlay_unit": overlay_unit,
        "overlay_cmap": overlay_cmap,
        "title_before": title_before,
        "title_after": title_after,
        "title_diff": title_diff,
    }

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    _draw_dose_comparison_panels(list(axes), before_dose, after_dose, **panel_kwargs)
    fig.tight_layout()
    if save_filename is not None:
        fig.savefig(save_filename)
    figures = [fig]

    for beam_ix, beam_before, beam_after in beam_doses:
        beam_fig, beam_axes = plt.subplots(1, 3, figsize=(16, 5.5))
        beam_fig.suptitle(beam_label(beam_ix, stf))
        _draw_dose_comparison_panels(list(beam_axes), beam_before, beam_after, **panel_kwargs)
        beam_fig.tight_layout()
        if save_filename is not None:
            beam_fig.savefig(_beam_filename(save_filename, beam_ix))
        figures.append(beam_fig)

    if show_plot:
        plt.show()

    return figures
