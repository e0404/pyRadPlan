"""File menu for the pyRadPlan main window.

Collects all data I/O (loading, importing, exporting) under a single menu so the
workflow widget can focus on the compute steps.  A matRad ``*.mat`` file may hold
a ct, cst and dose at once, so funnelling these through one menu avoids the
confusion of several look-alike load/import buttons.
"""

from __future__ import annotations

import os
from typing import Optional, get_args

from PySide6.QtCore import Signal
from PySide6.QtGui import QAction, QActionGroup
from PySide6.QtWidgets import QMenu, QWidget

from pyRadPlan._settings import GuiSettings, GuiUiMode, get_settings
from pyRadPlan.core import user_base_data_dir
from pyRadPlan.gui.workspace import WorkspaceManager
from pyRadPlan.gui.widgets.workflow import WorkflowWidget
from pyRadPlan.io import list_phantoms

from ._base_data import open_folder_in_browser
from ._settings import apply_settings_values

#: Tooltip per GUI presentation mode (``settings.gui.ui_mode``).
_UI_MODE_TIPS = {
    "default": "Show every control",
    "educational": "Reduce the widgets to the basic operations (e.g. beam angles) for teaching",
}


class FileMenu(QMenu):
    """The main window's *File* menu, wired to a :class:`WorkflowWidget`.

    The actual file handling lives on the workflow widget (so its background
    thread, progress bar and error dialogs are reused); this menu only exposes
    those entry points and keeps their enabled state in sync with the workspace.
    It also offers the *UI Mode* switch, which writes the GUI presentation mode
    to ``pyRadPlan.settings.gui.ui_mode`` and emits :attr:`ui_mode_changed`.

    Parameters
    ----------
    workspace:
        Shared :class:`WorkspaceManager`.
    workflow_widget:
        The workflow widget providing the load/import/export implementations.
    parent:
        Optional Qt parent widget.
    """

    #: Emitted with the new mode name after a UI mode was picked in this menu.
    ui_mode_changed = Signal(str)

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

        self._phantom_menu = self.addMenu("Open Example Phantom")
        self._phantom_menu.setToolTipsVisible(True)
        # Rebuilt on every opening, so phantoms copied into a folder appear without a restart.
        self._phantom_menu.aboutToShow.connect(self._rebuild_phantom_menu)
        self._rebuild_phantom_menu()

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
        self._ui_mode_menu = self.addMenu("UI Mode")
        self._ui_mode_menu.setToolTipsVisible(True)
        self._ui_mode_group = QActionGroup(self)
        self._ui_mode_group.setExclusive(True)
        self._ui_mode_actions: dict[str, QAction] = {}
        for mode in get_args(GuiUiMode):
            action = self._ui_mode_menu.addAction(mode.capitalize())
            action.setCheckable(True)
            action.setToolTip(_UI_MODE_TIPS.get(mode, ""))
            action.triggered.connect(lambda checked=False, m=mode: self._on_ui_mode(m))
            self._ui_mode_group.addAction(action)
            self._ui_mode_actions[mode] = action
        self.refresh_ui_mode()

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

    def _rebuild_phantom_menu(self) -> None:
        """List the phantoms of all phantom folders, grouped by source (see list_phantoms)."""
        menu = self._phantom_menu
        menu.clear()
        previous_source = None
        for entry in list_phantoms():
            if previous_source is not None and entry.source != previous_source:
                menu.addSeparator()
            previous_source = entry.source
            name = os.path.splitext(entry.name)[0]
            action = menu.addAction(name)
            action.setData(name)
            if entry.source == "bundled":
                action.setToolTip("Shipped with pyRadPlan")
            else:
                action.setToolTip(str(entry.path))
            action.triggered.connect(lambda checked=False, n=name: self._wf.load_phantom(n))
        menu.addSeparator()
        open_folder = menu.addAction("Open Phantom Folder")
        folder = user_base_data_dir("phantoms")
        open_folder.setToolTip(f"Phantoms (.mat, .npz) copied to {folder} are listed here")
        open_folder.triggered.connect(lambda checked=False: open_folder_in_browser(folder))

    def refresh_ui_mode(self) -> None:
        """Check the *UI Mode* entry matching the configured ``settings.gui.ui_mode``."""
        action = self._ui_mode_actions.get(get_settings().gui.ui_mode)
        if action is not None:
            action.setChecked(True)

    def _on_ui_mode(self, mode: str) -> None:
        apply_settings_values(GuiSettings, get_settings().gui, {"ui_mode": mode})
        self.ui_mode_changed.emit(mode)

    def _on_exit(self) -> None:
        window = self.window()
        if window is not None:
            window.close()
