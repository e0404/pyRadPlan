"""Graphical user interface (GUI) applications."""

from __future__ import annotations

import sys
from typing import Optional, Union

import numpy as np
import SimpleITK as sitk

from pyRadPlan.cst._cst import StructureSet
from pyRadPlan.ct._ct import CT

from .windows._result_win import _launch_result_window


def main(argv: Optional[list[str]] = None) -> None:
    """Command-line entry point for the ``pyRadPlanGUI`` console script.

    Parses *argv* (defaults to ``sys.argv``) and launches :func:`gui`.  Kept
    separate from :func:`gui` so programmatic calls never touch the host
    process's command line (e.g. Jupyter's ``-f kernel.json``).
    """
    import argparse  # noqa: PLC0415

    parser = argparse.ArgumentParser(
        prog="pyRadPlanGUI",
        description="Launch the pyRadPlan main GUI.",
    )
    parser.add_argument(
        "patient",
        nargs="?",
        default=None,
        help=(
            "Patient dataset to load on startup: a path (any supported file or folder) "
            "or the name of a bundled phantom, e.g. TG119."
        ),
    )
    gui(parser.parse_args(argv).patient)


def gui(patient: Optional[str] = None) -> None:
    """Launch the main GUI application (matRad-style main window).

    Parameters
    ----------
    patient:
        Optional patient dataset to load on startup: a path in any format
        supported by :func:`pyRadPlan.io.load_data` (a matRad ``*.mat`` file, a
        DICOM folder, ``*.npz``/``*.nrrd``/NIfTI, ...) or the name of a bundled
        phantom such as ``"TG119"`` (see :func:`pyRadPlan.io.available_phantoms`).
        When ``None`` (the default), the GUI starts with an empty workspace.
    """
    workspace = None
    if patient is not None:
        # Imports deferred: only needed when a patient file is actually given.
        import os  # noqa: PLC0415

        from pyRadPlan.gui.workspace import WorkspaceManager  # noqa: PLC0415
        from pyRadPlan.io import load_data, phantom_path  # noqa: PLC0415

        if not os.path.exists(patient):
            try:
                patient = phantom_path(patient)
            except FileNotFoundError as exc:
                raise FileNotFoundError(f"Patient dataset not found: {patient}. {exc}") from None

        data = load_data(patient)
        workspace = WorkspaceManager.instance()
        payload = {k: data[k] for k in workspace.keys if data.get(k) is not None}
        if "result" not in payload and data.get("dose") is not None:
            payload["result"] = {"physical_dose": data["dose"]}
        workspace.set_many(**payload)

    # Deferred: avoids pulling in the full Qt main-window stack at package import time.
    from .windows._main_win import launch_main_window  # noqa: PLC0415

    launch_main_window(workspace)


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


def analysis_viewer(
    cst: Optional[StructureSet] = None,
    result: Optional[Union[dict, np.ndarray, sitk.Image]] = None,
) -> None:
    """Launch the standalone DVH and QI analysis application.

    Parameters
    ----------
    cst:
        StructureSet providing the VOI masks to analyze. When ``None``, the
        structure set held by the shared :class:`WorkspaceManager` is used.
    result:
        Quantity result mapping (e.g. ``{"physical_dose": ...}``) or a single
        image/array. When ``None``, the workspace ``result`` is used.

    Raises
    ------
    ValueError
        If no structure set or no result is available.
    """
    # Deferred: keeps the Qt stack out of package import time, as in gui().
    from PySide6.QtWidgets import QApplication  # noqa: PLC0415

    from pyRadPlan.gui.widgets._result_widget import QUANTITY_META  # noqa: PLC0415
    from pyRadPlan.gui.windows._analysis_win import show_analysis  # noqa: PLC0415
    from pyRadPlan.gui.workspace import WorkspaceManager  # noqa: PLC0415

    if cst is None or result is None:
        workspace = WorkspaceManager.instance()
        cst = cst if cst is not None else workspace.cst
        result = result if result is not None else workspace.result

    if cst is None:
        raise ValueError("No structure set available for analysis; pass cst=... .")
    if result is None:
        raise ValueError("No result available for analysis; pass result=... .")

    if isinstance(result, (np.ndarray, sitk.Image)):
        result = {"quantity": result}

    # Native array order for quantities and masks alike: QIs and DVHs are
    # order-invariant voxel reductions, but a quantity and its mask must agree.
    quantities: dict[str, np.ndarray] = {}
    overlay_labels: dict[str, str] = {}
    overlay_units: dict[str, str] = {}
    for key, value in result.items():
        if isinstance(value, sitk.Image):
            array = sitk.GetArrayFromImage(value)
        elif isinstance(value, np.ndarray):
            array = value
        else:
            continue
        if array.ndim != 3:
            continue
        label, unit = QUANTITY_META.get(key, (key, ""))
        quantities[key] = array
        overlay_labels[key] = label
        overlay_units[key] = unit

    if not quantities:
        raise ValueError("Result contains no 3D quantity to analyze.")

    app = QApplication.instance() or QApplication(sys.argv)

    # show_analysis accepts a StructureSet directly and extracts the VOI masks.
    window = show_analysis(
        quantities=quantities,
        masks=cst,
        overlay={"voi_colors": _voi_colors(cst)},
        overlay_units=overlay_units,
        overlay_labels=overlay_labels,
        initial_quantity=next(iter(quantities)),
        voi_types={voi.name: getattr(voi, "voi_type", "") for voi in cst.vois},
    )
    if window is None:
        raise ValueError("Structure set contains no usable VOI masks.")

    app.exec()


def _voi_colors(cst: StructureSet) -> dict[str, tuple[int, int, int]]:
    """Return RGB colors (0-255) for every VOI, falling back to its type default color."""
    return {
        voi.name: tuple(int(c) for c in (voi.visible_color or voi.default_color))
        for voi in cst.vois
    }
