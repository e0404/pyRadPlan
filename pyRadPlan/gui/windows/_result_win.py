"""A simple Qt window to visualize 3D quantity distributions."""

from __future__ import annotations

import os
import sys
from typing import Optional

import numpy as np
from PySide6.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox
from PySide6.QtGui import QAction

import SimpleITK as sitk

from pyRadPlan.ct import CT
from pyRadPlan.cst import StructureSet
from pyRadPlan.gui.widgets import ViewingWidget
from pyRadPlan.gui.workspace import WorkspaceManager
from pyRadPlan import validate_ct, validate_cst


class QuantityWindow(QMainWindow):
    """A simple main window hosting the ViewerWidget as central widget."""

    def __init__(self, workspace: Optional[WorkspaceManager] = None) -> None:
        super().__init__()
        self.setWindowTitle("pyRadPlan Plan Result Viewer")
        self.workspace = workspace or WorkspaceManager.instance()
        self.viewer = ViewingWidget(self.workspace, self)
        self.setCentralWidget(self.viewer)
        # Viewer refresh errors are caught by the widget base class; show them
        # to the user instead of only logging (launch_viewer's data may be bad).
        self.viewer.update_failed.connect(
            lambda msg: QMessageBox.critical(self, "Viewer error", msg)
        )

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


def _launch_result_window(ct: CT, cst: StructureSet, result: dict | None = None) -> None:
    """Launch the Qt viewer with the given CT, CST and quantity result.

    Populates a :class:`WorkspaceManager` with the data; the viewer derives its
    display arrays from the workspace and updates automatically.

    Parameters
    ----------
    ct : CT
        pyRadPlan CT model.
    cst : CST
        pyRadPlan StructureSet with ``vois`` providing VOI names.
    result : Any | None
        Quantity result mapping or a raw 3D numpy array / image.
    """
    app = QApplication.instance() or QApplication(sys.argv)

    ct = validate_ct(ct)
    cst = validate_cst(cst, ct)

    workspace = WorkspaceManager()
    win = QuantityWindow(workspace)
    workspace.set_many(ct=ct, cst=cst, result=result)

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
