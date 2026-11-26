"""Main application window reproducing matRad's ``matRad_MainGUI`` layout.

A fixed three-column layout around a central slice viewer, with every widget
bound to a shared :class:`~pyRadPlan.gui.workspace.WorkspaceManager`:

- Left column:  Workflow / Plan / Objectives & constraints / Log
  (agentic UI mode: an *Agent* tab, then one tab per plan the agent selected,
  each holding the column above)
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
    QStackedWidget,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QIcon

from pyRadPlan._settings import get_settings
from pyRadPlan.gui.assets import asset_path
from pyRadPlan.gui.menus import BaseDataMenu, FileMenu, HelpMenu, SettingsMenu, ViewMenu
from pyRadPlan.gui.workspace import WorkspaceManager
from pyRadPlan.gui.widgets import (
    WorkflowWidget,
    PlanWidget,
    OptimizationWidget,
    ViewingWidget,
    LogConsoleWidget,
)
from pyRadPlan.gui.widgets._logo_widget import LogoWidget
from pyRadPlan.gui.widgets._ui_mode import UIMode
from pyRadPlan.gui.widgets._info_widget import InfoWidget
from pyRadPlan.gui.windows._analysis_win import close_analysis_windows
from pyRadPlan.gui.windows._splash_win import (
    BUILD_WINDOW_WEIGHT,
    SHOW_WINDOW_WEIGHT,
    StartupSplash,
)


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

        # Beams dragged in the viewer are edits of the plan form, applied with it.
        self._viewer.set_beam_editor(self.plan_widget.set_gantry_angle)
        self.plan_widget.beam_angles_edited.connect(self._viewer.set_edited_beam_angles)
        self._viewer.set_edited_beam_angles(self.plan_widget.edited_beam_angles())
        self.workflow_widget.add_pending_changes_source(
            "plan", self.plan_widget.has_pending_changes, self.plan_widget.apply_changes
        )

        # Lock the input widgets while a workflow computation is running.
        self.workflow_widget.busy_changed.connect(self._on_busy_changed)

        # Agentic mode: built on first use, as it loads pydantic-ai. Each tab after the
        # Agent tab is a plan; the workspace holds the state of the current tab and
        # _tab_states the others (None for the current one).
        self.agent_panel = None
        self._tab_states: list[Optional[dict]] = [None]
        self._tab_keys: list = [None]  # per tab: the optimization it shows (see AgentPanel)
        self._current_tab = 0
        self._switching_tabs = False
        self.workspace.workspace_changed.connect(self._on_workspace_changed)

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
        self._apply_layout_mode()

    def _apply_compact_font(self) -> None:
        # Set on the application, not the window: widgets with a style sheet (or whose
        # style sheet is cleared again) fall back to the application font.
        app = QApplication.instance()
        if app.property("compact_font_applied"):
            return
        font = app.font()
        point = font.pointSizeF()
        if point > 0:
            font.setPointSizeF(point * self._FONT_SCALE)
        else:
            pixel = font.pixelSize()
            if pixel > 0:
                font.setPixelSize(max(1, round(pixel * self._FONT_SCALE)))
        app.setFont(font)
        app.setProperty("compact_font_applied", True)

    def apply_ui_mode_setting(self) -> None:
        """Push the configured ``settings.gui.ui_mode`` to the widgets and the File menu."""
        self.plan_widget.set_ui_mode(get_settings().gui.ui_mode)
        self._file_menu.refresh_ui_mode()
        self._apply_layout_mode()

    def _on_busy_changed(self, busy: bool) -> None:
        """Disable input editors while a background computation or the agent runs."""
        busy = busy or self._is_busy()  # the other source may still run
        self.plan_widget.setEnabled(not busy)
        self.optimization_widget.setEnabled(not busy)
        self._file_menu.setEnabled(not busy)
        self._agent_tabs.tabBar().setEnabled(not busy)

    def _is_busy(self) -> bool:
        agent_busy = self.agent_panel is not None and self.agent_panel.is_busy
        return self.workflow_widget.is_busy or agent_busy

    # ------------------------------------------------------------------
    # Agentic mode
    # ------------------------------------------------------------------

    def _apply_layout_mode(self) -> None:
        """Show the classic left column, or the Agent and plan tabs in agentic mode."""
        if get_settings().gui.ui_mode == UIMode.AGENTIC and self.agent_panel is None:
            self._build_agent_panel()
        self._place_widgets()

    def _build_agent_panel(self) -> None:
        from pyRadPlan.gui.widgets.agent import AgentPanel  # noqa: PLC0415
        from pyRadPlan.gui.widgets.ai import busy_cursor  # noqa: PLC0415

        with busy_cursor():  # loads pydantic-ai
            self.agent_panel = AgentPanel(
                self.workspace, self.workflow_widget, before_run=self._apply_plan_edits
            )
        self.agent_panel.busy_changed.connect(self._on_busy_changed)
        self.agent_panel.run_started.connect(self._clear_plan_tabs)
        self.agent_panel.plans_selected.connect(self._add_plan_tabs)
        self.agent_panel.run_cleared.connect(self._clear_plan_tabs)
        self.agent_panel.plan_requested.connect(self._show_plan_tab)
        self.agent_panel.update_failed.connect(
            lambda msg: self.statusBar().showMessage(f"AgentPanel: {msg}", 10000)
        )
        self._switching_tabs = True
        try:
            self._agent_tabs.insertTab(0, self.agent_panel, "Agent")
            self._agent_tabs.setCurrentIndex(0)
        finally:
            self._switching_tabs = False

    def _apply_plan_edits(self) -> bool:
        if not self.plan_widget.has_pending_changes():
            return True
        return self.plan_widget.apply_changes()

    def _place_widgets(self) -> None:
        """Put the classic column and the plan editor where the current mode shows them.

        In agentic mode, the Agent tab holds the plan editor and a plan tab the classic
        column; otherwise the classic column is the left column.
        """
        agentic = self.agent_panel is not None and get_settings().gui.ui_mode == UIMode.AGENTIC
        on_plan_tab = agentic and self._current_tab > 0
        self._left_stack.setCurrentIndex(1 if agentic else 0)
        if on_plan_tab:
            column_host = self._agent_tabs.widget(self._current_tab)
        else:
            column_host = self._classic_page
        column_host.layout().addWidget(self._classic_column)
        if agentic and not on_plan_tab:
            plan_host = self.agent_panel.plan_page
        else:
            plan_host = self._plan_group
        plan_host.layout().addWidget(self.plan_widget)
        self._classic_column.show()
        self.plan_widget.show()

    def _on_agent_tab_changed(self, index: int) -> None:
        """Swap the workspace to the state of the tab at *index*.

        The dose influence of a plan tab is dropped when leaving it, to save memory;
        it can be recalculated in the tab's workflow.
        """
        if self._switching_tabs or index < 0 or index == self._current_tab:
            return
        ws = self.workspace
        state = {key: getattr(ws, key) for key in ("cst", "pln", "stf", "dij")}
        state.update(results=ws.results, active_result=ws.active_result)
        if self._current_tab > 0:
            state.update(stf=None, dij=None)
        self._tab_states[self._current_tab] = state
        new_state, self._tab_states[index] = self._tab_states[index], None
        self._current_tab = index
        self._switching_tabs = True
        try:
            ws.set_many(**new_state)
        finally:
            self._switching_tabs = False
        self._place_widgets()

    def _add_plan_tabs(self, plans: list) -> None:
        for plan in plans:
            page = QWidget()
            lay = QVBoxLayout(page)
            lay.setContentsMargins(0, 0, 0, 0)
            index = self._agent_tabs.addTab(page, plan["label"])
            self._agent_tabs.setTabToolTip(index, plan["tooltip"])
            self._tab_states.append(plan["state"])
            self._tab_keys.append(plan.get("key"))

    def _show_plan_tab(self, plan: dict) -> None:
        """Switch to the tab of *plan*'s optimization, adding the tab if there is none."""
        key = plan.get("key")
        if key is None or key not in self._tab_keys:
            self._add_plan_tabs([plan])
            index = self._agent_tabs.count() - 1
        else:
            index = self._tab_keys.index(key)
        self._agent_tabs.setCurrentIndex(index)

    def _clear_plan_tabs(self) -> None:
        """Remove the plan tabs; the workspace's current state becomes the Agent tab's."""
        if self._agent_tabs.count() <= 1:
            return
        self._switching_tabs = True
        try:
            self._agent_tabs.setCurrentIndex(0)
        finally:
            self._switching_tabs = False
        self._current_tab = 0
        self._place_widgets()  # moves the classic column out of the plan tab
        while self._agent_tabs.count() > 1:
            page = self._agent_tabs.widget(1)
            self._agent_tabs.removeTab(1)
            page.deleteLater()
        self._tab_states = [None]
        self._tab_keys = [None]

    def _on_workspace_changed(self, keys: list) -> None:
        # Plans of another patient are meaningless
        if "ct" in keys and not self._switching_tabs:
            self._clear_plan_tabs()

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt override)
        """Confirm quitting during a computation and stop the worker thread.

        Without this, closing the window would destroy a still-running QThread,
        which aborts the whole process (Qt fatal).
        """
        if self._is_busy():
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
            if self.agent_panel is not None:
                self.agent_panel.shutdown()
        close_analysis_windows(self._viewer)
        # The log handler outlives the widget otherwise, and logging into a
        # destroyed QObject bridge would crash on interpreter shutdown.
        self.log_console.detach()
        super().closeEvent(event)

    # ------------------------------------------------------------------
    # Menu bar
    # ------------------------------------------------------------------

    def _build_menu_bar(self) -> None:
        self._file_menu = FileMenu(self.workspace, self.workflow_widget, self)
        self._file_menu.ui_mode_changed.connect(lambda _mode: self.apply_ui_mode_setting())
        log_toggle = QAction("Log Panel", self)
        log_toggle.setCheckable(True)
        log_toggle.setChecked(True)
        log_toggle.toggled.connect(self._log_panel.setVisible)
        self._view_menu = ViewMenu(self, panel_actions=(log_toggle,))
        self._base_data_menu = BaseDataMenu(self)
        self._base_data_menu.base_data_changed.connect(self.plan_widget.refresh_base_data)
        self._settings_menu = SettingsMenu(self)
        self._settings_menu.settings_applied.connect(self.apply_ui_mode_setting)
        self._help_menu = HelpMenu(self)
        for menu in (
            self._file_menu,
            self._view_menu,
            self._base_data_menu,
            self._settings_menu,
            self._help_menu,
        ):
            self.menuBar().addMenu(menu)

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _build_layout(self) -> None:
        # A horizontal splitter of three columns, each itself a vertical splitter,
        # so the user can resize or fully collapse any panel (helpful on small or
        # oddly proportioned displays).
        self._main_splitter = QSplitter(Qt.Horizontal)
        self._main_splitter.addWidget(self._build_left_stack())
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

    def _build_left_stack(self) -> QWidget:
        # Page 0: the classic column; page 1: the agentic tabs. The classic column
        # moves into a plan tab while that is shown (see _place_widgets).
        self._classic_column = self._build_left_column()
        self._classic_page = QWidget()
        lay = QVBoxLayout(self._classic_page)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self._classic_column)
        self._agent_tabs = QTabWidget()
        self._agent_tabs.currentChanged.connect(self._on_agent_tab_changed)
        self._left_stack = QStackedWidget()
        self._left_stack.addWidget(self._classic_page)
        self._left_stack.addWidget(self._agent_tabs)
        self._left_stack.setMinimumWidth(330)
        return self._left_stack

    def _build_left_column(self) -> QWidget:
        # Fixed control panels keep their natural height; the objectives table
        # takes most of the column's slack (mirrors matRad's tall objectives
        # panel), with the log console growing at half its rate below it.
        self._log_panel = _group("Log", self.log_console)
        self._plan_group = _group("Plan", self.plan_widget)
        col = self._vsplit(
            (_group("Workflow", self.workflow_widget), 0),
            (self._plan_group, 0),
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


def launch_main_window(
    workspace: Optional[WorkspaceManager] = None, splash: Optional[StartupSplash] = None
) -> None:
    """Create and show the main window, starting the Qt event loop.

    Parameters
    ----------
    workspace:
        Shared :class:`WorkspaceManager`.  Falls back to the singleton.
    splash:
        Startup splash to advance while the window is built; it closes once the
        window is shown.
    """
    app = QApplication.instance() or QApplication(sys.argv)
    if splash is not None:
        splash.step("Building main window…", BUILD_WINDOW_WEIGHT)
    win = MainWindow(workspace)
    # Sensible restored-window size, but open maximized by default.
    win.resize(1500, 850)
    if splash is not None:
        splash.step("Opening main window…", SHOW_WINDOW_WEIGHT)
    win.showMaximized()
    if splash is not None:
        splash.finish(win)
    app.exec()


if __name__ == "__main__":
    from pyRadPlan import load_tg119, PhotonPlan

    ws = WorkspaceManager.instance()
    _ct, _cst = load_tg119()
    ws.set_many(ct=_ct, cst=_cst, pln=PhotonPlan())
    launch_main_window(ws)
