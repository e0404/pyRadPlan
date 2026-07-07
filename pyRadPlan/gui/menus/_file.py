"""File menu for the pyRadPlan main window.

Collects all data I/O (loading, importing, exporting) under a single menu so the
workflow widget can focus on the compute steps.  A matRad ``*.mat`` file may hold
a ct, cst and dose at once, so funnelling these through one menu avoids the
confusion of several look-alike load/import buttons.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtWidgets import QMenu, QWidget

from pyRadPlan.gui.workspace import WorkspaceManager
from pyRadPlan.gui.widgets.workflow import WorkflowWidget


class FileMenu(QMenu):
    """The main window's *File* menu, wired to a :class:`WorkflowWidget`.

    The actual file handling lives on the workflow widget (so its background
    thread, progress bar and error dialogs are reused); this menu only exposes
    those entry points and keeps their enabled state in sync with the workspace.

    Parameters
    ----------
    workspace:
        Shared :class:`WorkspaceManager`.
    workflow_widget:
        The workflow widget providing the load/import/export implementations.
    parent:
        Optional Qt parent widget.
    """

    def __init__(
        self,
        workspace: WorkspaceManager,
        workflow_widget: WorkflowWidget,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__("&File", parent)
        self._ws = workspace
        self._wf = workflow_widget

        self._act_load_mat = self.addAction("Load Patient (.mat)…")
        self._act_load_mat.triggered.connect(self._wf.load_mat)

        self._act_load_dicom = self.addAction("Load DICOM…")
        self._act_load_dicom.triggered.connect(self._wf.load_dicom)
        self._act_load_dicom.setEnabled(False)

        self.addSeparator()
        self._act_import_dose = self.addAction("Import Dose…")
        self._act_import_dose.triggered.connect(self._wf.import_dose)

        self.addSeparator()
        self._act_save = self.addAction("Save Workspace…")
        self._act_save.setEnabled(False)  # not yet implemented
        self._act_export_bin = self.addAction("Export Binary…")
        self._act_export_bin.triggered.connect(self._wf.export_binary)
        self._act_export_bin.setEnabled(False)
        self._act_export_dicom = self.addAction("Export DICOM…")
        self._act_export_dicom.triggered.connect(self._wf.export_dicom)
        self._act_export_dicom.setEnabled(False)

        self.addSeparator()
        self._act_exit = self.addAction("Exit")
        self._act_exit.triggered.connect(self._on_exit)

        self._ws.workspace_changed.connect(self._refresh)
        self._refresh([])

    def _refresh(self, _changed_keys: list) -> None:
        # Dose cubes are imported into the current result and matched against the
        # CT grid, so a CT must be present.
        self._act_import_dose.setEnabled(self._ws.has("ct"))

    def _on_exit(self) -> None:
        window = self.window()
        if window is not None:
            window.close()
