"""Graphical user interface (GUI) applications."""

from __future__ import annotations

import importlib
import logging
import os
import sys
from importlib.util import find_spec
from typing import TYPE_CHECKING, Optional, Union

import numpy as np
import SimpleITK as sitk

# The Qt windows are imported lazily (in gui() behind the startup splash), so probe the
# GUI dependencies here: pyRadPlan.gui derives GUI_AVAILABLE from importing this module.
from PySide6 import QtWidgets  # noqa: F401

if find_spec("pyqtgraph") is None:
    raise ImportError("No module named 'pyqtgraph'")

if TYPE_CHECKING:
    from pyRadPlan.cst._cst import StructureSet
    from pyRadPlan.ct._ct import CT
    from pyRadPlan.gui.windows._splash_win import StartupSplash
    from pyRadPlan.gui.workspace import WorkspaceManager

logger = logging.getLogger(__name__)

#: Startup imports announced on the splash as (status, module, weight), one per top-level
#: subpackage the GUI uses, weights roughly in warm-start milliseconds.  ``import
#: pyRadPlan`` is lazy, so these run behind the splash; modules already loaded are skipped.
_STARTUP_IMPORTS = (
    ("Loading core…", "pyRadPlan.core.xp_utils", 1200.0),
    ("Loading CT…", "pyRadPlan.ct", 10.0),
    ("Loading structure sets…", "pyRadPlan.cst", 600.0),
    ("Loading plans…", "pyRadPlan.plan", 370.0),
    ("Loading beam geometry…", "pyRadPlan.stf", 100.0),
    ("Loading dose engines…", "pyRadPlan.dose", 360.0),
    ("Loading optimization…", "pyRadPlan.optimization", 1500.0),
    ("Loading analysis…", "pyRadPlan.analysis", 780.0),
    ("Loading data import/export…", "pyRadPlan.io", 470.0),
    ("Loading AI planning agents…", "pyRadPlan.ai.agents", 120.0),
    ("Loading GUI widgets…", "pyRadPlan.gui.widgets", 1000.0),
    ("Loading main window…", "pyRadPlan.gui.windows._main_win", 30.0),
)
#: Weight of loading the array backend the settings select (CuPy, warm).
_BACKEND_WEIGHT = 500.0
_PATIENT_WEIGHT = 300.0


def main(argv: Optional[list[str]] = None) -> None:
    """Command-line entry point for the ``pyRadPlanGUI`` console script.

    Parses *argv* (defaults to ``sys.argv``) and launches :func:`gui`.  Kept
    separate from :func:`gui` so programmatic calls never touch the host
    process's command line (e.g. Jupyter's ``-f kernel.json``).
    """
    import argparse  # noqa: PLC0415
    import signal  # noqa: PLC0415

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
    args = parser.parse_args(argv)
    # Python's SIGINT handler only runs once Qt's event loop hands control back, which
    # it never does, so Ctrl+C in the launching terminal would not stop the GUI.
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    gui(args.patient)


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
    from PySide6.QtWidgets import QApplication  # noqa: PLC0415

    from .windows._splash_win import (  # noqa: PLC0415
        BUILD_WINDOW_WEIGHT,
        SHOW_WINDOW_WEIGHT,
        StartupSplash,
    )

    _ensure_std_streams()
    _app = QApplication.instance() or QApplication(sys.argv)
    imports = [entry for entry in _STARTUP_IMPORTS if entry[1] not in sys.modules]
    total = (
        sum(weight for *_, weight in imports)
        + _BACKEND_WEIGHT
        + BUILD_WINDOW_WEIGHT
        + SHOW_WINDOW_WEIGHT
    )
    steps = len(imports) + 3
    if patient is not None:
        total += _PATIENT_WEIGHT
        steps += 1
    splash = StartupSplash(total, steps)
    splash.show()
    try:
        for message, module, weight in imports:
            splash.step(message, weight)
            importlib.import_module(module)

        splash.step("Detecting array backends…", _BACKEND_WEIGHT)
        _load_array_backend(splash)

        workspace = None
        if patient is not None:
            splash.step(f"Loading patient {patient}…", _PATIENT_WEIGHT)
            workspace = _load_patient(_resolve_patient(patient))

        # Deferred: avoids pulling in the full Qt main-window stack at package import time.
        from .windows._main_win import launch_main_window  # noqa: PLC0415

        launch_main_window(workspace, splash=splash)
    finally:
        splash.close()


def _ensure_std_streams() -> None:
    """Replace missing standard streams with :data:`os.devnull`.

    A process started without a console (``pythonw``, the desktop bundle) has
    ``sys.stdout`` and ``sys.stderr`` set to ``None``, so any ``print`` or progress bar
    in a computation would raise. The GUI shows its output in the log panel instead.
    """
    for name in ("stdout", "stderr"):
        if getattr(sys, name) is None:
            setattr(sys, name, open(os.devnull, "w", encoding="utf-8"))


def _load_array_backend(splash: StartupSplash) -> None:
    """Import the array backend the settings select, listing the probed backends.

    Only a preload: the GUI also serves viewing data, so a backend that cannot be loaded
    does not stop it. Computations choose the backend again and report the error.
    """
    from pyRadPlan.core import xp_utils  # noqa: PLC0415

    probed = []

    def announce(name: str) -> None:
        probed.append(xp_utils.BACKENDS.get(name, name))
        splash.set_message(f"Detecting array backends: {', '.join(probed)}…")

    try:
        with xp_utils.observe_backend_imports(announce):
            xp_utils.choose_array_api_namespace()
    except Exception as exc:  # noqa: BLE001 - any backend failure only skips the preload
        logger.warning("Cannot load the configured array backend: %s", exc)


def _resolve_patient(patient: str) -> str:
    """Return the path of *patient*, a file/folder path or a bundled phantom name."""
    import os  # noqa: PLC0415

    from pyRadPlan.io import phantom_path  # noqa: PLC0415

    if os.path.exists(patient):
        return patient
    try:
        return phantom_path(patient)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Patient dataset not found: {patient}. {exc}") from None


def _load_patient(path: str) -> WorkspaceManager:
    """Load the patient dataset at *path* into the shared workspace and return it."""
    from pyRadPlan.gui.workspace import WorkspaceManager  # noqa: PLC0415
    from pyRadPlan.io import load_data  # noqa: PLC0415

    data = load_data(path)
    workspace = WorkspaceManager.instance()
    payload = {k: data[k] for k in workspace.keys if data.get(k) is not None}
    if "result" not in payload and data.get("dose") is not None:
        payload["result"] = {"physical_dose": data["dose"]}
    workspace.set_many(**payload)
    return workspace


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

    from .windows._result_win import _launch_result_window  # noqa: PLC0415

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

    from pyRadPlan.gui.widgets._result_widget import (  # noqa: PLC0415
        QUANTITY_META,
        default_quantity_key,
    )
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
        initial_quantity=default_quantity_key(list(quantities)),
        voi_types={voi.name: getattr(voi, "voi_type", "") for voi in cst.vois},
    )
    if window is None:
        raise ValueError("Structure set contains no usable VOI masks.")

    app.exec()


def _voi_colors(cst: StructureSet) -> dict[str, tuple[int, int, int]]:
    """Return RGB colors (0-255) for every VOI."""
    return {voi.name: tuple(int(c) for c in voi.color) for voi in cst.vois}
