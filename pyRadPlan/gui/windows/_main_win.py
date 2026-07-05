"""Main application window reproducing matRad's ``matRad_MainGUI`` layout.

A fixed three-column layout around a central slice viewer, with every widget
bound to a shared :class:`~pyRadPlan.gui.workspace.WorkspaceManager`:

- Left column:  Workflow / Plan / Objectives & constraints / Log
- Center:       Logo banner (top), the Slice Viewer (large), Visualization
- Right column: Viewer Options / Structure Visibility / Info

Widgets that are not yet implemented are shown as stubs and toolbar actions
without an implementation are present but disabled.
"""

from __future__ import annotations

import sys
from typing import Optional

from PySide6.QtWidgets import (
    QApplication,
    QGroupBox,
    QHBoxLayout,
    QMainWindow,
    QMessageBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QIcon

from pyRadPlan.gui.assets import asset_path
from pyRadPlan.gui.menus import FileMenu, SettingsMenu, ViewMenu
from pyRadPlan.gui.workspace import WorkspaceManager
from pyRadPlan.gui.widgets import (
    WorkflowWidget,
    PlanWidget,
    OptimizationWidget,
    ViewingWidget,
    LogConsoleWidget,
)
from pyRadPlan.gui.widgets._logo_widget import LogoWidget
from pyRadPlan.gui.widgets._info_widget import InfoWidget


def _group(title: str, content: QWidget) -> QGroupBox:
    """Wrap *content* in a titled group box (matRad's uipanel equivalent)."""
    box = QGroupBox(title)
    lay = QVBoxLayout(box)
    lay.setContentsMargins(4, 4, 4, 4)
    lay.addWidget(content)
    return box


class MainWindow(QMainWindow):
    """matRad-style main window for pyRadPlan.

    Parameters
    ----------
    workspace:
        Shared :class:`WorkspaceManager`.  Falls back to the singleton.
    """

    #: Fraction of the default font size used throughout the window, keeping the
    #: dense control panels readable without crowding.
    _FONT_SCALE = 0.8

    def __init__(self, workspace: Optional[WorkspaceManager] = None) -> None:
        super().__init__()
        self.setWindowTitle("pyRadPlan (NOT FOR CLINICAL USE!)")
        self.setWindowIcon(QIcon(str(asset_path("logos", "pyradplan_logo_skull_square.svg"))))
        self.workspace = workspace or WorkspaceManager.instance()

        # Slightly smaller base font; child widgets inherit it, shrinking both
        # the text and the controls sized from font metrics by ~10%.
        self._apply_compact_font()

        # The viewer wires the slice renderer + its control widgets but does not
        # lay them out: we place its sub-widgets into the matRad-style panels.
        self._viewer = ViewingWidget(self.workspace, build_layout=False)

        self.workflow_widget = WorkflowWidget(self.workspace)
        self.plan_widget = PlanWidget(self.workspace)
        self.optimization_widget = OptimizationWidget(self.workspace)
        self.logo_widget = LogoWidget()
        self.info_widget = InfoWidget()

        # Persistent log view (bottom-left panel), fed by the root logger.
        self.log_console = LogConsoleWidget()

        # Lock the input widgets while a workflow computation is running.
        self.workflow_widget.busy_changed.connect(self._on_busy_changed)

        # Surface widget refresh failures (otherwise only logged) to the user.
        for widget in (
            self.workflow_widget,
            self.plan_widget,
            self.optimization_widget,
            self._viewer,
        ):
            name = type(widget).__name__
            widget.update_failed.connect(
                lambda msg, name=name: self.statusBar().showMessage(f"{name}: {msg}", 10000)
            )

        self._build_layout()
        self._build_menu_bar()

    def _apply_compact_font(self) -> None:
        font = self.font()
        point = font.pointSizeF()
        if point > 0:
            font.setPointSizeF(point * self._FONT_SCALE)
        else:
            pixel = font.pixelSize()
            if pixel > 0:
                font.setPixelSize(max(1, round(pixel * self._FONT_SCALE)))
        self.setFont(font)

    def _on_busy_changed(self, busy: bool) -> None:
        """Disable input editors while a background computation runs."""
        self.plan_widget.setEnabled(not busy)
        self.optimization_widget.setEnabled(not busy)
        self._file_menu.setEnabled(not busy)

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt override)
        """Confirm quitting during a computation and stop the worker thread.

        Without this, closing the window would destroy a still-running QThread,
        which aborts the whole process (Qt fatal).
        """
        if self.workflow_widget.is_busy:
            answer = QMessageBox.question(
                self,
                "Computation running",
                "A computation is still running. Quit anyway?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                event.ignore()
                return
            self.workflow_widget.shutdown()
        # The log handler outlives the widget otherwise, and logging into a
        # destroyed QObject bridge would crash on interpreter shutdown.
        self.log_console.detach()
        super().closeEvent(event)

    # ------------------------------------------------------------------
    # Menu bar
    # ------------------------------------------------------------------

    def _build_menu_bar(self) -> None:
        self._file_menu = FileMenu(self.workspace, self.workflow_widget, self)
        log_toggle = QAction("Log Panel", self)
        log_toggle.setCheckable(True)
        log_toggle.setChecked(True)
        log_toggle.toggled.connect(self._log_panel.setVisible)
        self._view_menu = ViewMenu(self, panel_actions=(log_toggle,))
        self._settings_menu = SettingsMenu(self)
        for menu in (self._file_menu, self._view_menu, self._settings_menu):
            self.menuBar().addMenu(menu)

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _build_layout(self) -> None:
        # A horizontal splitter of three columns, each itself a vertical splitter,
        # so the user can resize or fully collapse any panel (helpful on small or
        # oddly proportioned displays).
        self._main_splitter = QSplitter(Qt.Horizontal)
        self._main_splitter.addWidget(self._build_left_column())
        self._main_splitter.addWidget(self._build_center_column())
        self._main_splitter.addWidget(self._build_right_column())
        # The left control column gets the most room; the center viewer is kept
        # a bit smaller so the dense panels stay readable.
        for index, stretch in enumerate((50, 37, 13)):
            self._main_splitter.setStretchFactor(index, stretch)
        self._main_splitter.setSizes([560, 620, 220])

        central = QWidget()
        root = QHBoxLayout(central)
        root.setContentsMargins(4, 4, 4, 4)
        root.addWidget(self._main_splitter)
        self.setCentralWidget(central)

    @staticmethod
    def _vsplit(*items: tuple[QWidget, int]) -> QSplitter:
        split = QSplitter(Qt.Vertical)
        for index, (widget, stretch) in enumerate(items):
            split.addWidget(widget)
            split.setStretchFactor(index, stretch)
        return split

    def _build_left_column(self) -> QWidget:
        # Fixed control panels keep their natural height; the objectives table
        # takes most of the column's slack (mirrors matRad's tall objectives
        # panel), with the log console growing at half its rate below it.
        self._log_panel = _group("Log", self.log_console)
        col = self._vsplit(
            (_group("Workflow", self.workflow_widget), 0),
            (_group("Plan", self.plan_widget), 0),
            (_group("Objectives && constraints", self.optimization_widget), 2),
            (self._log_panel, 1),
        )
        col.setMinimumWidth(330)
        return col

    def _build_center_column(self) -> QWidget:
        # Logo banner stays compact; the visualization controls sit below the
        # slice viewer, which takes the remaining height.
        self.logo_widget.setMaximumHeight(80)
        return self._vsplit(
            (self.logo_widget, 0),
            (_group("Slice Viewer", self._viewer.quantity_widget), 1),
            (_group("Visualization", self._viewer.vis_widget), 0),
        )

    def _build_right_column(self) -> QWidget:
        # Viewer options and info stay compact; the structure list absorbs slack.
        col = self._vsplit(
            (_group("Viewer Options", self._viewer.opts_widget), 0),
            (_group("Structure Visibility", self._viewer.vois_widget), 1),
            (_group("Info", self.info_widget), 0),
        )
        col.setMinimumWidth(240)
        return col


def launch_main_window(workspace: Optional[WorkspaceManager] = None) -> None:
    """Create and show the main window, starting the Qt event loop."""
    app = QApplication.instance() or QApplication(sys.argv)
    win = MainWindow(workspace)
    # Sensible restored-window size, but open maximized by default.
    win.resize(1500, 850)
    win.showMaximized()
    app.exec()


if __name__ == "__main__":
    from pyRadPlan import load_tg119, PhotonPlan

    ws = WorkspaceManager.instance()
    _ct, _cst = load_tg119()
    ws.set_many(ct=_ct, cst=_cst, pln=PhotonPlan())
    launch_main_window(ws)
