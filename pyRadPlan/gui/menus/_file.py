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

        self._act_load_file = self.addAction("Load File…")
        self._act_load_file.triggered.connect(self._wf.load_file)

        self._act_load_folder = self.addAction("Load Folder…")
        self._act_load_folder.setToolTip(
            "Import a folder of DICOM data, or a folder of image files "
            "(CT + binary structure masks)"
        )
        self._act_load_folder.triggered.connect(self._wf.load_folder)

        self.addSeparator()
        self._act_save_plan = self.addAction("Save Plan…")
        self._act_save_plan.triggered.connect(self._wf.save_plan)
        self._act_save_dij = self.addAction("Save Dij…")
        self._act_save_dij.triggered.connect(self._wf.save_dij)
        self._act_save_cst = self.addAction("Save CST…")
        self._act_save_cst.setToolTip("Includes the structures' objectives")
        self._act_save_cst.triggered.connect(self._wf.save_cst)
        self._act_save_result = self.addAction("Save Result…")
        self._act_save_result.triggered.connect(self._wf.save_result)

        self.addSeparator()
        self._act_save = self.addAction("Save Workspace…")
        self._act_save.triggered.connect(self._wf.save_workspace)

        self.addSeparator()
        self._act_exit = self.addAction("Exit")
        self._act_exit.triggered.connect(self._on_exit)

        # Tooltips show even for disabled actions.
        self.setToolTipsVisible(True)

        self._ws.workspace_changed.connect(self._refresh)
        self._refresh([])

    def _refresh(self, _changed_keys: list) -> None:
        # Each save action needs its object present; Save Workspace anchors on the CT.
        self._act_save_plan.setEnabled(self._ws.has("pln"))
        self._act_save_dij.setEnabled(self._ws.has("dij"))
        self._act_save_cst.setEnabled(self._ws.has("cst"))
        self._act_save_result.setEnabled(self._ws.has("result"))
        self._act_save.setEnabled(self._ws.has("ct"))

    def _on_exit(self) -> None:
        window = self.window()
        if window is not None:
            window.close()
