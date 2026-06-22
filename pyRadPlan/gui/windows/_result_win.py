"""A simple Qt window to visualize 3D quantity distributions."""

from __future__ import annotations

import os
import sys

import numpy as np
from PySide6.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox
from PySide6.QtGui import QAction

import SimpleITK as sitk

from pyRadPlan.ct import CT
from pyRadPlan.cst import StructureSet
from pyRadPlan.gui.widgets import ViewingWidget
from pyRadPlan import validate_ct, validate_cst


class QuantityWindow(QMainWindow):
    """A simple main window hosting the ViewerWidget as central widget."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("pyRadPlan Plan Result Viewer")
        self.viewer = ViewingWidget(self)
        self.setCentralWidget(self.viewer)

        self._create_menu()

    # --- Menu and actions ---------------------------------------------
    def _create_menu(self) -> None:
        menubar = self.menuBar()

        file_menu = menubar.addMenu("&File")
        act_open = QAction("Open NPY/NPZ…", self)
        act_open.triggered.connect(self._open_array)
        file_menu.addAction(act_open)

        file_menu.addSeparator()
        act_quit = QAction("Quit", self)
        act_quit.triggered.connect(self.close)
        file_menu.addAction(act_quit)

        menubar.addMenu("&View")

    # Might be nice to have once io utilities are in place :)
    def _open_array(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            caption="Open 3D numpy array",
            filter="NumPy arrays (*.npy *.npz)",
        )
        if not path:
            return
        try:
            vol = self._load_array(path)
        except Exception as exc:  # noqa: BLE001 - show to user
            QMessageBox.critical(self, "Error", f"Failed to load file:\n{exc}")
            return
        if vol.ndim != 3:
            QMessageBox.warning(self, "Wrong shape", f"Expected 3D array, got shape {vol.shape}")
            return

    # Load array function.
    # TODO: Use io utilities in the future once implemented!
    @staticmethod
    def _load_array(path: str) -> np.ndarray:
        _, ext = os.path.splitext(path.lower())
        if ext == ".npy":
            return np.load(path)
        if ext == ".npz":
            data = np.load(path)
            # Heuristic: first array in container
            for key in data.files:
                return data[key]
            raise ValueError(".npz file contains no arrays")
        raise ValueError(f"Unsupported file extension: {ext}")


def _compute_isocenter_vox(ct: CT, cst: StructureSet) -> np.ndarray | None:
    """Compute isocenter voxel coordinates (z, x, y) from the CST target center of mass."""
    iso = None
    if hasattr(cst, "target_center_of_mass"):
        try:
            iso = cst.target_center_of_mass()
        except Exception:
            pass

    if iso is None:
        return None

    origin = np.array(ct.cube_hu.GetOrigin())
    spacing = np.array(ct.cube_hu.GetSpacing())
    iso = np.array(iso).flatten()

    # Voxel index = (physical - origin) / spacing; iso/origin/spacing are (x, y, z)
    isocenter_vox_phys = (iso - origin) / spacing

    # Viewer array is (Z, X, Y) after transpose; isocenter_vox_phys is (x, y, z)
    return np.array([isocenter_vox_phys[2], isocenter_vox_phys[0], isocenter_vox_phys[1]])


def _process_result(result: dict | None) -> dict | np.ndarray | None:
    """Convert a result dict or raw array into viewer-ready form (transposed arrays)."""
    if result is None:
        return None
    if isinstance(result, dict):
        return _process_dict_result(result)
    return _process_raw_result(result)


def _process_dict_result(result_dict: dict) -> dict:
    """Convert each value in a result dict into a transposed array (or list thereof)."""
    quantities_dict: dict = {}
    for k, v in result_dict.items():
        if isinstance(v, sitk.Image):
            arr = sitk.GetArrayFromImage(v)
            if arr.ndim == 3:
                quantities_dict[k] = np.transpose(arr, (0, 2, 1))
        elif isinstance(v, np.ndarray) and v.ndim == 3:
            quantities_dict[k] = np.transpose(v, (0, 2, 1))
        elif isinstance(v, list):
            processed = _process_list_quantity(v)
            if processed:
                quantities_dict[k] = processed
    return quantities_dict


def _process_list_quantity(items: list) -> list:
    """Transpose each 3-D array in a list of images/arrays."""
    processed = []
    for item in items:
        if isinstance(item, sitk.Image):
            arr = sitk.GetArrayFromImage(item)
        elif isinstance(item, np.ndarray):
            arr = item
        else:
            continue
        if arr.ndim == 3:
            processed.append(np.transpose(arr, (0, 2, 1)))
    return processed


def _process_raw_result(result: object) -> np.ndarray:
    """Convert a single raw image or array to a transposed numpy array."""
    if isinstance(result, sitk.Image):
        result_arr = sitk.GetArrayFromImage(result)
    else:
        result_arr = np.asarray(result)
    return np.transpose(result_arr, (0, 2, 1))


def _build_overlay_meta(
    quantity_data: dict | np.ndarray | None,
) -> tuple[dict[str, str], dict[str, str]]:
    """Build overlay unit and label dicts from the processed quantity data."""
    quantity_meta: dict[str, tuple[str, str]] = {
        "physical_dose": ("Dose", "Gy"),
        "physical_dose_beam": ("Dose", "Gy"),
        "let": ("LET", "keV/\u00b5m"),
        "let_beam": ("LET", "keV/\u00b5m"),
        "effect": ("Effect", ""),
        "effect_beam": ("Effect", ""),
        "rbe_x_dose": ("RBE-weighted Dose", "Gy (RBE)"),
        "rbe_x_dose_beam": ("RBE-weighted Dose", "Gy (RBE)"),
        "alpha_dose": ("Alpha Dose", "Gy"),
        "alpha_dose_beam": ("Alpha Dose", "Gy"),
        "sqrt_beta_dose": ("Sqrt(Beta) Dose", "Gy\u00bd"),
        "sqrt_beta_dose_beam": ("Sqrt(Beta) Dose", "Gy\u00bd"),
        "let_dose": ("LET\u00b7Dose", "Gy\u00b7keV/\u00b5m"),
        "let_dose_beam": ("LET\u00b7Dose", "Gy\u00b7keV/\u00b5m"),
    }

    overlay_units: dict[str, str] = {}
    overlay_labels: dict[str, str] = {}
    if isinstance(quantity_data, dict):
        for k, v in quantity_data.items():
            base = k.split()[0] if " " in k else k
            qname, unit = quantity_meta.get(base, ("Quantity", ""))
            if isinstance(v, list):
                for i in range(len(v)):
                    expanded = f"{k} {i}"
                    overlay_units[expanded] = unit
                    overlay_labels[expanded] = qname
            else:
                overlay_units[k] = unit
                overlay_labels[k] = qname
    else:
        overlay_units = {"Physical quantity": "Gy"}
        overlay_labels = {"Physical quantity": "Dose"}

    return overlay_units, overlay_labels


def _launch_result_window(ct: CT, cst: StructureSet, result: dict | None = None) -> None:
    """Launch the Qt viewer with the given CT, CST and quantity result.

    Parameters
    ----------
    ct : CT
        pyRadPlan CT model.
    cst : CST
        pyRadPlan StructureSet with ``vois`` providing VOI names.
    result : Any | None
        Quantity result mapping (expects "physical_quantity") or a raw 3D numpy array.
    """
    app = QApplication.instance() or QApplication(sys.argv)

    win = QuantityWindow()

    ct = validate_ct(ct)
    cst = validate_cst(cst, ct)

    isocenter_vox = _compute_isocenter_vox(ct, cst)

    # TODO. Replace once result is merged!
    # result = validate_result(result, ct)
    quantity_data = _process_result(result)
    overlay_units, overlay_labels = _build_overlay_meta(quantity_data)

    ct_arr = sitk.GetArrayFromImage(ct.cube_hu)

    if ct_arr is not None:
        ct_arr = np.transpose(ct_arr, (0, 2, 1))
        win.viewer.set_data(
            ct_arr,
            quantity_data,
            overlay_unit=overlay_units,
            overlay_label=overlay_labels,
            isocenter=isocenter_vox,
        )

    win.viewer.set_vois(cst.vois)

    win.viewer.set_masks(
        {voi.name: np.transpose(sitk.GetArrayFromImage(voi.mask), (0, 2, 1)) for voi in cst.vois}
    )

    win.show()
    # Start the event loop (blocks until window close)
    app.exec()


if __name__ == "__main__":
    # Minimal demo: load TG119 to preview the UI standalone
    from pyRadPlan import load_tg119

    _ct, _cst = load_tg119()
    sigma = 0.40
    quantity = sitk.GetArrayFromImage(_ct.cube_hu) * 0.001
    _launch_result_window(_ct, _cst, result=quantity)
