"""Graphical user interface (GUI) applications."""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import SimpleITK as sitk

from pyRadPlan.cst._cst import StructureSet
from pyRadPlan.ct._ct import CT

from .windows._result_win import _launch_result_window


def gui() -> None:
    """Launch the main GUI application."""
    raise NotImplementedError("Main GUI application is not yet implemented.")


def launch_viewer(
    ct: CT,
    cst: StructureSet = None,
    result: Optional[Union[dict, Union[np.ndarray, sitk.Image]]] = None,
) -> None:
    """Launch the quantity viewer with optional CT background and VOI contours.

    Parameters
    ----------
    ct:
        CT object.
    cst:
        StructureSet object providing VOIs for contour display.
    result:
        Dict or single image/array.
    """

    # TODO: Maybe remove this fallback if imports are possible
    if ct is None or cst is None or result is None:
        raise NotImplementedError(
            "Launching viewer without CT, CST, and result is not supported yet."
        )

    # TODO: Overhaul needed once result is implemented properly
    if isinstance(result, (np.ndarray, sitk.Image)):
        # Wrap single array/image into dict
        result = {"quantity": result}

    _launch_result_window(ct=ct, cst=cst, result=result)


def analysis_viewer() -> None:
    """Launch DVH and QI analysis application."""
    raise NotImplementedError("DVH and QI analysis application is not yet implemented.")
