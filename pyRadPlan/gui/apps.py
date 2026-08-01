"""Graphical user interface (GUI) applications."""

from __future__ import annotations

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
        help="Path to a patient dataset (any supported file or folder) to load on startup.",
    )
    gui(parser.parse_args(argv).patient)


def gui(patient: Optional[str] = None) -> None:
    """Launch the main GUI application (matRad-style main window).

    Parameters
    ----------
    patient:
        Optional path to a patient dataset to load on startup. Any format
        supported by :func:`pyRadPlan.io.load_data` is accepted (a matRad
        ``*.mat`` file, a DICOM folder, ``*.npz``/``*.nrrd``/NIfTI, ...). When
        ``None`` (the default), the GUI starts with an empty workspace.
    """
    workspace = None
    if patient is not None:
        # Imports deferred: only needed when a patient file is actually given.
        import os  # noqa: PLC0415

        from pyRadPlan.gui.workspace import WorkspaceManager  # noqa: PLC0415
        from pyRadPlan.io import load_data  # noqa: PLC0415

        if not os.path.exists(patient):
            raise FileNotFoundError(f"Patient dataset not found: {patient}")

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


def analysis_viewer() -> None:
    """Launch DVH and QI analysis application."""
    raise NotImplementedError("DVH and QI analysis application is not yet implemented.")
