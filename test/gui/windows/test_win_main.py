import os

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QToolBar

from pyRadPlan.gui.windows._main_win import MainWindow
from pyRadPlan.gui.workspace import WorkspaceManager

# Widget methods are patched on the class: undoing a patch on an instance leaves
# the bound method behind as an attribute, a reference cycle whose garbage
# collection segfaults PySide (Linux, Python 3.14).


def test_main_window_constructs_empty(qapp):
    win = MainWindow(WorkspaceManager())
    assert win.workflow_widget is not None
    assert win.plan_widget is not None
    assert win.optimization_widget is not None
    assert win.logo_widget is not None
    assert win.info_widget is not None


def test_viewer_subwidgets_reparented(qapp):
    win = MainWindow(WorkspaceManager())
    # The decomposed viewer's controls live in the main-window panels now,
    # not inside the ViewingWidget container.
    for w in (
        win._viewer.vis_widget,
        win._viewer.quantity_widget,
        win._viewer.opts_widget,
        win._viewer.vois_widget,
    ):
        assert w.parent() is not None
        assert w.parent() is not win._viewer


def test_shared_workspace_drives_all_widgets(qapp, test_data_photons):
    ct, cst, _ = test_data_photons
    ws = WorkspaceManager()
    win = MainWindow(ws)

    ws.set_many(ct=ct, cst=cst)

    assert win._viewer.quantity_widget._ct is not None
    assert len(win._viewer.vois_widget._voi_checkboxes) == len(cst.vois)
    # loading ct+cst auto-applies the plan widget's defaults as the initial plan
    assert ws.pln is not None
    assert "dose influence" in win.workflow_widget._lbl_status.text().lower()


def test_single_menu_bar_with_expected_menus(qapp):
    win = MainWindow(WorkspaceManager())

    # Everything lives on one menu bar now; the matRad-style toolbar is gone.
    # Nested widget-internal toolbars (e.g. matplotlib navigation) don't count:
    # only toolbars installed in the main window's toolbar areas would be one.
    installed = [
        tb
        for tb in win.findChildren(QToolBar)
        if win.toolBarArea(tb) != Qt.ToolBarArea.NoToolBarArea
    ]
    assert installed == []
    titles = [a.text() for a in win.menuBar().actions()]
    assert titles == ["&File", "&View", "&Base Data", "&Settings", "&Help"]


def test_base_data_menu_add_and_remove_refresh_plan_widget(qapp, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    from pyRadPlan.gui.menus import BaseDataDialog
    from pyRadPlan.machines import find_machine_file

    monkeypatch.setenv("PYRADPLAN_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: QMessageBox.Ok)
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.Yes)
    win = MainWindow(WorkspaceManager())
    win.plan_widget._cmb_radiation.setCurrentText("photons")
    menu = win._base_data_menu

    def listed():
        combo = win.plan_widget._cmb_machine
        return [combo.itemText(i) for i in range(combo.count())]

    bundled = str(find_machine_file("photons", "Generic"))
    # The bundled name is taken, so the menu asks for another one.
    monkeypatch.setattr(
        "pyRadPlan.gui.menus._base_data.QInputDialog.getText",
        lambda *a, **k: ("MyLinac", True),
    )
    target = menu.add("machines", bundled)
    assert target == tmp_path / "machines" / "photons_MyLinac.mat"
    assert "MyLinac" in listed()

    dialog = BaseDataDialog(win)
    dialog.base_data_changed.connect(menu.base_data_changed)
    tab = dialog.kind_tabs["machines"]
    sources = [tab.table.item(row, 2).text() for row in range(tab.table.rowCount())]
    names = [tab.table.item(row, 1).text() for row in range(tab.table.rowCount())]
    assert sources.count("User") == 1 and "pyRadPlan" in sources

    tab.table.selectRow(names.index("Generic"))
    assert not tab.btn_remove.isEnabled()
    tab.table.selectRow(names.index("MyLinac"))
    assert tab.btn_remove.isEnabled()
    assert tab.remove_selected()
    assert not target.exists()
    assert "MyLinac" not in listed()
    assert tab.table.rowCount() == len(names) - 1


def test_base_data_dialog_lists_and_adds_rbe_tables(qapp, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    from pyRadPlan.core import find_base_data
    from pyRadPlan.gui.menus import BaseDataDialog

    monkeypatch.setenv("PYRADPLAN_DATA_DIR", str(tmp_path))
    critical = []
    monkeypatch.setattr(QMessageBox, "critical", lambda *a, **k: critical.append(a[2]))
    dialog = BaseDataDialog()
    tab = dialog.kind_tabs["rbe_tables"]
    assert tab.table.rowCount() == 1
    assert tab.table.item(0, 1).text() == "LEM I"

    shipped = find_base_data("rbe_tables", "RBEtable_LEMI_Scholz06_AX01_BX005.mat").path
    copy = tmp_path / "RBEtable_mine.mat"
    copy.write_bytes(shipped.read_bytes())
    assert tab.add(str(copy)) == tmp_path / "rbe" / "RBEtable_mine.mat"
    assert tab.table.rowCount() == 2

    # The Add button asks for the file instead of taking the clicked flag for it
    second = tmp_path / "RBEtable_clicked.mat"
    second.write_bytes(copy.read_bytes())
    monkeypatch.setattr(
        "pyRadPlan.gui.menus._base_data.QFileDialog.getOpenFileName",
        lambda *a, **k: (str(second), ""),
    )
    tab.btn_add.click()
    assert (tmp_path / "rbe" / "RBEtable_clicked.mat").is_file()
    assert tab.table.rowCount() == 3

    # A machine file is no RBE table and is reported instead of added
    bad = find_base_data("machines", "photons_Generic.mat").path
    assert tab.add(str(bad)) is None
    assert critical and "RBEtable" in critical[0]
    assert dialog.kind_tabs["stopping_power_tables"].table.rowCount() == 1


def test_view_menu_stubs_disabled(qapp):
    win = MainWindow(WorkspaceManager())
    actions = {a.text(): a for a in win._view_menu.actions() if a.text()}

    for stub in ("Screenshot", "Zoom In", "Pan"):
        assert not actions[stub].isEnabled()


@pytest.fixture
def ignored_color_scheme_requests(qapp, monkeypatch):
    """Record scheme requests to a platform ignoring them, reporting none, rendering light."""
    from PySide6.QtGui import QStyleHints

    from pyRadPlan.gui.menus import _view

    requested = []
    monkeypatch.setattr(QStyleHints, "setColorScheme", lambda self, s: requested.append(s))
    monkeypatch.setattr(QStyleHints, "colorScheme", lambda self: Qt.ColorScheme.Unknown)
    monkeypatch.setattr(_view, "_palette_dark", lambda palette: False)
    _view._color_scheme_switch()._follow_platform()
    yield requested
    _view._color_scheme_switch()._follow_platform()


@pytest.fixture
def dark_native_palette(monkeypatch, ignored_color_scheme_requests):
    """Like ``ignored_color_scheme_requests``, with a dark native palette."""
    from pyRadPlan.gui.menus import _view

    monkeypatch.setattr(_view, "_palette_dark", lambda palette: True)
    _view._color_scheme_switch()._follow_platform()
    return ignored_color_scheme_requests


def _is_dark(qapp):
    from PySide6.QtGui import QPalette

    return qapp.palette().color(QPalette.ColorRole.Window).lightness() < 128


def test_view_menu_toggles_dark_mode(qapp, ignored_color_scheme_requests):
    win = MainWindow(WorkspaceManager())
    action = win._view_menu.dark_mode_action

    action.trigger()
    action.trigger()

    assert ignored_color_scheme_requests == [Qt.ColorScheme.Dark, Qt.ColorScheme.Light]


def test_view_menu_follows_os_color_scheme_without_pinning_it(qapp, ignored_color_scheme_requests):
    win = MainWindow(WorkspaceManager())
    action = win._view_menu.dark_mode_action
    hints = qapp.styleHints()

    hints.colorSchemeChanged.emit(Qt.ColorScheme.Dark)
    assert action.isChecked()
    hints.colorSchemeChanged.emit(Qt.ColorScheme.Light)
    assert not action.isChecked()
    assert ignored_color_scheme_requests == []


def test_dark_mode_falls_back_to_fusion_where_the_platform_ignores_it(
    qapp, ignored_color_scheme_requests
):
    from pyRadPlan.gui.menus import ViewMenu

    native_style = qapp.style().name()
    win = MainWindow(WorkspaceManager())
    other_window_menu = ViewMenu()

    win._view_menu.dark_mode_action.trigger()
    assert qapp.style().name() == "fusion"
    assert _is_dark(qapp)
    assert other_window_menu.dark_mode_action.isChecked()

    win._view_menu.dark_mode_action.trigger()
    assert qapp.style().name() == native_style
    assert not other_window_menu.dark_mode_action.isChecked()


def test_dark_mode_fallback_ends_when_the_platform_catches_up(qapp, ignored_color_scheme_requests):
    native_style = qapp.style().name()
    win = MainWindow(WorkspaceManager())
    action = win._view_menu.dark_mode_action
    action.trigger()

    qapp.styleHints().colorSchemeChanged.emit(Qt.ColorScheme.Light)
    assert qapp.style().name() == "fusion"  # the user's choice overrides the platform
    assert action.isChecked()

    qapp.styleHints().colorSchemeChanged.emit(Qt.ColorScheme.Dark)
    assert qapp.style().name() == native_style
    assert action.isChecked()


def test_dark_mode_choice_outlasts_the_platform_changing_back(qapp, ignored_color_scheme_requests):
    native_style = qapp.style().name()
    win = MainWindow(WorkspaceManager())
    action = win._view_menu.dark_mode_action
    action.trigger()
    hints = qapp.styleHints()

    hints.colorSchemeChanged.emit(Qt.ColorScheme.Dark)
    assert qapp.style().name() == native_style
    hints.colorSchemeChanged.emit(Qt.ColorScheme.Light)

    assert qapp.style().name() == "fusion"
    assert _is_dark(qapp)
    assert action.isChecked()


def test_palette_darkness(qapp, dark_palette):
    from PySide6.QtGui import QColor, QPalette

    from pyRadPlan.gui.menus._view import _palette_dark

    light = QPalette()
    light.setColor(QPalette.ColorRole.Window, QColor("#efefef"))
    light.setColor(QPalette.ColorRole.WindowText, QColor("#000000"))

    assert _palette_dark(dark_palette[0])
    assert not _palette_dark(light)


def test_dark_mode_judges_an_unreported_scheme_by_the_native_palette(qapp, dark_native_palette):
    from pyRadPlan.gui.menus import ViewMenu

    native_style = qapp.style().name()
    menu = ViewMenu()
    assert menu.dark_mode_action.isChecked()

    menu.dark_mode_action.trigger()
    assert qapp.style().name() == "fusion"
    assert not _is_dark(qapp)

    menu.dark_mode_action.trigger()
    assert qapp.style().name() == native_style
    assert menu.dark_mode_action.isChecked()


def test_file_menu_opens_phantoms_from_all_folders(qapp, tmp_path, monkeypatch):
    from pyRadPlan._settings import get_settings
    from pyRadPlan.gui.widgets.workflow import WorkflowWidget
    from pyRadPlan.io import phantom_path

    monkeypatch.setenv("PYRADPLAN_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(get_settings(), "base_data_path", str(tmp_path / "course"))
    monkeypatch.setattr(
        WorkflowWidget,
        "_run_in_thread",
        lambda self, fn, *args, on_success=None, **kwargs: on_success(fn(*args)),
    )
    ws = WorkspaceManager()
    win = MainWindow(ws)
    menu = win._file_menu._phantom_menu

    def phantoms():
        menu.aboutToShow.emit()
        return {a.data(): a for a in menu.actions() if a.data()}

    assert list(phantoms()) == ["TG119"]
    assert phantoms()["TG119"].toolTip().startswith("Shipped with pyRadPlan")

    # files dropped into the user's and a shared phantom folder show up without a restart
    (tmp_path / "data" / "phantoms").mkdir(parents=True)
    (tmp_path / "data" / "phantoms" / "Mine.mat").write_bytes(phantom_path("TG119").read_bytes())
    (tmp_path / "course" / "phantoms").mkdir(parents=True)
    (tmp_path / "course" / "phantoms" / "Course.mat").write_bytes(
        phantom_path("TG119").read_bytes()
    )
    listed = phantoms()
    assert list(listed) == ["TG119", "Mine", "Course"]
    assert str(tmp_path / "course" / "phantoms") in listed["Course"].toolTip()

    listed["Course"].trigger()

    assert ws.has("ct", "cst")
    assert {voi.name for voi in ws.cst.vois} >= {"Core", "OuterTarget", "BODY"}


# def test_file_menu_save_actions_track_workspace(qapp, test_data_photons):
#     ct, cst, _ = test_data_photons
#     ws = WorkspaceManager()
#     win = MainWindow(ws)

#     fm = win._file_menu
#     # Loading is always available; save actions gate on their object being present.
#     assert fm._act_load_file.isEnabled()
#     assert fm._act_load_folder.isEnabled()
#     assert not fm._act_save_cst.isEnabled()  # no cst loaded
#     assert not fm._act_save.isEnabled()  # no ct loaded
#     assert not fm._act_save_dij.isEnabled()  # dij not computed
#     assert not fm._act_save_result.isEnabled()  # no result yet

#     # Loading ct+cst makes the PlanWidget seed a default plan, so cst/ct/pln saves enable.
#     ws.set_many(ct=ct, cst=cst)
#     assert fm._act_save_cst.isEnabled()
#     assert fm._act_save.isEnabled()
#     assert fm._act_save_plan.isEnabled()
#     assert not fm._act_save_dij.isEnabled()  # still no dij
#     assert not fm._act_save_result.isEnabled()  # still no result


def test_compact_font_is_smaller(qapp):
    applied = qapp.property("compact_font_applied")
    default = qapp.font().pointSizeF()
    win = MainWindow(WorkspaceManager())
    assert qapp.property("compact_font_applied")
    if not applied and default > 0:
        assert qapp.font().pointSizeF() < default
    # Existing widgets receive the application font change as posted events.
    qapp.processEvents()
    assert win.font() == qapp.font()


def test_style_sheet_keeps_compact_font(qapp):
    win = MainWindow(WorkspaceManager())
    btn = win.plan_widget._btn_apply
    btn.setStyleSheet("font-weight: bold;")
    btn.setStyleSheet("")
    assert btn.font().pointSizeF() == qapp.font().pointSizeF()


def test_settings_dialog_shows_subconfig_tabs(qapp):
    from pyRadPlan.gui.menus._settings import SettingsDialog

    dialog = SettingsDialog()
    titles = [dialog._tabs.tabText(i) for i in range(dialog._tabs.count())]
    # The top-level fields that are no sub-configuration form the General tab.
    assert titles[0] == "General"
    assert list(dialog._general_form._editors) == ["base_data_path"]
    assert "XP (Backend)" in titles
    assert "AI" in titles
    assert "GUI" in titles
    assert "ui_mode" in dialog._sub_forms["gui"]._editors
    assert "prefer_gpu" in dialog._sub_forms["xp"]._editors
    assert "preferred_gpu_array_backend" in dialog._sub_forms["xp"]._editors
    assert "agents_model" in dialog._sub_forms["ai"]._editors
    assert "modelhub_device" in dialog._sub_forms["ai"]._editors


def test_settings_dialog_single_section(qapp):
    from pyRadPlan.gui.menus._settings import SettingsDialog

    dialog = SettingsDialog(section="xp")
    assert dialog._tabs is None
    assert dialog.windowTitle() == "XP (Backend) Settings"
    assert list(dialog._sub_forms) == ["xp"]
    assert "prefer_gpu" in dialog._sub_forms["xp"]._editors

    with pytest.raises(ValueError):
        SettingsDialog(section="nope")


def test_settings_dialog_applies_to_singleton_and_env(qapp, monkeypatch):
    from pyRadPlan._settings import get_settings
    from pyRadPlan.gui.menus._settings import SettingsDialog

    settings = get_settings()
    orig_prefer_gpu = settings.xp.prefer_gpu
    orig_ai_model = settings.ai.agents_model
    monkeypatch.setattr(settings, "base_data_path", "")
    monkeypatch.delenv("PYRADPLAN_BASE_DATA_PATH", raising=False)
    monkeypatch.delenv("PYRADPLAN_XP_PREFER_GPU", raising=False)
    monkeypatch.delenv("PYRADPLAN_AI_AGENTS_MODEL", raising=False)
    monkeypatch.setenv("PYRADPLAN_AI_MODEL", "stale-legacy-value")

    try:
        dialog = SettingsDialog()
        dialog._sub_forms["xp"]._set_value("prefer_gpu", False)
        dialog._sub_forms["ai"]._set_value("agents_model", "test-model")
        dialog._general_form._set_value("base_data_path", "course-cases")
        dialog.apply()

        assert settings.base_data_path == "course-cases"
        assert os.environ["PYRADPLAN_BASE_DATA_PATH"] == "course-cases"

        assert settings.xp.prefer_gpu is False
        assert os.environ["PYRADPLAN_XP_PREFER_GPU"] == "False"
        assert settings.ai.agents_model == "test-model"
        assert os.environ["PYRADPLAN_AI_AGENTS_MODEL"] == "test-model"
        # the legacy alias is cleared so it cannot override the new choice
        assert "PYRADPLAN_AI_MODEL" not in os.environ
    finally:
        os.environ.pop("PYRADPLAN_XP_PREFER_GPU", None)
        os.environ.pop("PYRADPLAN_AI_AGENTS_MODEL", None)
        os.environ.pop("PYRADPLAN_BASE_DATA_PATH", None)
        settings.xp.prefer_gpu = orig_prefer_gpu
        settings.ai.agents_model = orig_ai_model


def test_settings_dialog_writes_the_env_var_that_is_actually_read(qapp, tmp_path, monkeypatch):
    """An edit must be written under the env var name pydantic reads the field from."""
    from pyRadPlan._settings import AiSettings, get_settings
    from pyRadPlan.gui.menus._settings import SettingsDialog

    settings = get_settings()
    orig = settings.ai.modelhub_local_models_dir
    monkeypatch.delenv("PYRADPLAN_AI_MODELHUB_LOCAL_MODELS_DIR", raising=False)

    try:
        dialog = SettingsDialog(section="ai")
        dialog._sub_forms["ai"]._set_value("modelhub_local_models_dir", tmp_path / "chosen")
        dialog.apply()

        assert settings.ai.modelhub_local_models_dir == tmp_path / "chosen"
        # a freshly constructed settings object must see the edit
        assert AiSettings(_env_file=None).modelhub_local_models_dir == tmp_path / "chosen"
    finally:
        os.environ.pop("PYRADPLAN_AI_MODELHUB_LOCAL_MODELS_DIR", None)
        settings.ai.modelhub_local_models_dir = orig


def test_settings_menu_has_quick_links_and_preferences(qapp):
    win = MainWindow(WorkspaceManager())
    labels = [a.text() for a in win._settings_menu.actions() if a.text()]
    assert labels == ["XP (Backend)…", "AI…", "GUI…", "Preferences…"]


def test_inputs_locked_while_busy(qapp):
    win = MainWindow(WorkspaceManager())
    assert win.plan_widget.isEnabled()
    assert win.optimization_widget.isEnabled()

    win.workflow_widget._set_busy(True)
    assert not win.plan_widget.isEnabled()
    assert not win.optimization_widget.isEnabled()

    win.workflow_widget._set_busy(False)
    assert win.plan_widget.isEnabled()
    assert win.optimization_widget.isEnabled()


@pytest.fixture
def gui_settings(monkeypatch):
    """Hand out the GUI settings with a clean environment and restore them afterwards."""
    from pyRadPlan._settings import get_settings

    settings = get_settings().gui
    snapshot = settings.model_dump()
    monkeypatch.delenv("PYRADPLAN_GUI_UI_MODE", raising=False)
    yield settings
    os.environ.pop("PYRADPLAN_GUI_UI_MODE", None)
    for name, value in snapshot.items():
        setattr(settings, name, value)


def test_file_menu_ui_mode_switches_settings_and_plan_widget(qapp, gui_settings):
    from pyRadPlan.gui.widgets import UIMode

    gui_settings.ui_mode = "default"
    win = MainWindow(WorkspaceManager())
    actions = win._file_menu._ui_mode_actions
    assert list(actions) == ["default", "educational", "agentic"]
    assert actions["default"].isChecked()
    assert win.plan_widget.ui_mode() is UIMode.DEFAULT

    actions["educational"].trigger()
    assert gui_settings.ui_mode == "educational"
    assert os.environ["PYRADPLAN_GUI_UI_MODE"] == "educational"
    assert win.plan_widget.ui_mode() is UIMode.EDUCATIONAL
    assert actions["educational"].isChecked()

    actions["default"].trigger()
    assert gui_settings.ui_mode == "default"
    assert win.plan_widget.ui_mode() is UIMode.DEFAULT


def test_main_window_starts_in_configured_ui_mode(qapp, gui_settings):
    from pyRadPlan.gui.widgets import UIMode

    gui_settings.ui_mode = "educational"
    win = MainWindow(WorkspaceManager())
    assert win.plan_widget.ui_mode() is UIMode.EDUCATIONAL
    assert win._file_menu._ui_mode_actions["educational"].isChecked()


def test_settings_menu_applies_ui_mode_after_dialog(qapp, gui_settings, monkeypatch):
    from PySide6.QtWidgets import QDialog

    from pyRadPlan.gui.menus._settings import SettingsDialog
    from pyRadPlan.gui.widgets import UIMode

    gui_settings.ui_mode = "default"
    win = MainWindow(WorkspaceManager())

    def fake_exec(dialog):
        dialog._sub_forms["gui"]._set_value("ui_mode", "educational")
        dialog.apply()
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(SettingsDialog, "exec", fake_exec)
    win._settings_menu._edit_settings("gui")

    assert gui_settings.ui_mode == "educational"
    assert win.plan_widget.ui_mode() is UIMode.EDUCATIONAL
    assert win._file_menu._ui_mode_actions["educational"].isChecked()


def _open_analysis_window(parent=None):
    from pyRadPlan.gui.windows._analysis_win import show_analysis

    mask = np.zeros((4, 4, 4), dtype=bool)
    mask[1:3, 1:3, 1:3] = True
    return show_analysis(
        quantities={"Dose": np.ones((4, 4, 4))}, masks={"Target": mask}, parent=parent
    )


def test_closing_main_window_closes_analysis_windows(qapp):
    from pyRadPlan.gui.windows._analysis_win import _OPEN_WINDOWS, close_all_analysis_windows

    win = MainWindow(WorkspaceManager())
    analysis = _open_analysis_window(parent=win._viewer)
    try:
        assert analysis in _OPEN_WINDOWS
        win.close()
        assert analysis not in _OPEN_WINDOWS
        assert not analysis.isVisible()
    finally:
        close_all_analysis_windows()


def test_cancelled_main_window_close_keeps_analysis_windows(qapp, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    from pyRadPlan.gui.windows._analysis_win import _OPEN_WINDOWS, close_all_analysis_windows

    win = MainWindow(WorkspaceManager())
    analysis = _open_analysis_window(parent=win._viewer)
    monkeypatch.setattr(type(win.workflow_widget), "is_busy", property(lambda self: True))
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: QMessageBox.No)
    try:
        assert not win.close()
        assert analysis in _OPEN_WINDOWS
    finally:
        close_all_analysis_windows()


def test_dragged_beams_are_plan_edits_applied_with_the_plan(qapp, test_data_photons):
    ct, cst, _ = test_data_photons
    ws = WorkspaceManager()
    win = MainWindow(ws)
    ws.set_many(ct=ct, cst=cst)
    applied = list(ws.pln.prop_stf["gantry_angles"])

    win._viewer.quantity_widget.beam_angle_changed.emit(0, 33.0)

    assert ws.pln.prop_stf["gantry_angles"] == applied
    assert win.plan_widget.edited_beam_angles()[0][0] == 33.0
    beam = win._viewer.quantity_widget._beams[0]
    assert beam["gantry"] == 33.0 and beam["pending"]

    win.plan_widget._btn_apply.click()
    assert ws.pln.prop_stf["gantry_angles"][0] == 33.0
    assert not win._viewer.quantity_widget._beams[0]["pending"]


def test_workflow_offers_the_pending_plan_changes(qapp, test_data_photons, monkeypatch):
    ct, cst, _ = test_data_photons
    ws = WorkspaceManager()
    win = MainWindow(ws)
    ws.set_many(ct=ct, cst=cst)
    win.plan_widget.set_gantry_angle(0, 33.0)
    assert win.plan_widget.has_pending_changes()

    monkeypatch.setattr(
        type(win.workflow_widget), "_ask_pending_changes", staticmethod(lambda *a: "apply")
    )
    monkeypatch.setattr(
        type(win.workflow_widget), "_run_in_thread", staticmethod(lambda fn, **kw: None)
    )
    win.workflow_widget._on_calc_dose()

    assert ws.pln.prop_stf["gantry_angles"][0] == 33.0
    assert not win.plan_widget.has_pending_changes()


# ---- agentic mode ------------------------------------------------------------------


@pytest.fixture
def agentic(monkeypatch):
    pytest.importorskip("pydantic_ai")
    from pyRadPlan._settings import get_settings

    monkeypatch.setattr(get_settings().gui, "ui_mode", "agentic")


def test_agentic_mode_shows_agent_tab_with_plan_editor(qapp, agentic):
    from pyRadPlan._settings import get_settings

    win = MainWindow(WorkspaceManager())
    assert win._left_stack.currentIndex() == 1
    assert win._agent_tabs.tabText(0) == "Agent"
    assert win.plan_widget.parent() is win.agent_panel.plan_page

    get_settings().gui.ui_mode = "default"
    win.apply_ui_mode_setting()
    assert win._left_stack.currentIndex() == 0
    assert win.plan_widget.parent() is win._plan_group


def test_plan_tabs_hold_their_own_workspace_state(qapp, agentic, test_data_photons):
    ct, cst, result = test_data_photons
    ws = WorkspaceManager()
    win = MainWindow(ws)
    ws.set_many(ct=ct, cst=cst)
    base_pln = ws.pln
    dose = {"physical_dose": result["physicalDose"]}
    plan_state = {
        "cst": cst,
        "pln": base_pln,
        "stf": None,
        "dij": None,
        "results": {"current": dose, "Plan_1": dose},
        "active_result": "current",
    }
    win._add_plan_tabs([{"label": "Plan 1", "tooltip": "t", "state": plan_state}])

    ws.dij = "base dij"
    win._agent_tabs.setCurrentIndex(1)
    assert list(ws.results) == ["current", "Plan_1"] and ws.dij is None
    assert ws.active_result == "current" and "physical_dose" in ws.result
    assert win._classic_column.parent() is win._agent_tabs.widget(1)
    assert win.plan_widget.parent() is win._plan_group

    ws.dij = "plan dij"
    win._agent_tabs.setCurrentIndex(0)
    assert ws.dij == "base dij" and ws.result is None and not ws.results
    win._agent_tabs.setCurrentIndex(1)
    assert ws.dij is None  # a plan tab's dose influence is dropped when leaving it
    assert list(ws.results) == ["current", "Plan_1"]

    ws.ct = ct  # another patient: the plan tabs go
    assert win._agent_tabs.count() == 1
    assert win._classic_column.parent() is win._classic_page


def _agent_plan(index, cst, dose, hard_ok=True):
    goals = [
        {
            "goal": "Core max <= 1.6",
            "hard": True,
            "value": 1.5,
            "margin": 0.1,
            "rel_margin": 0.06 if hard_ok else -0.06,
            "passed": hard_ok,
        },
        {
            "goal": "PTV D95 >= 1.8",
            "hard": False,
            "value": 1.7,
            "margin": -0.1,
            "rel_margin": -0.05,
            "passed": False,
        },
    ]
    sod = {
        "name": "Squared Overdosing",
        "penalty": 3,
        "quantity": "physical_dose",
        "parameters": {"d_max": 1.0},
    }
    from pyRadPlan.ai.agents import PlanRecord
    from pyRadPlan.result import create_result

    return PlanRecord(
        index=index,
        beams={"gantry_angles": [0, 180], "couch_angles": [0, 0]},
        objectives={"Core": [sod], "opt_ring": [sod]},
        evaluation={
            "all_hard_passed": hard_ok,
            "num_failed": 1 + (not hard_ok),
            "score": float(index),
            "goals": goals,
            "unresolved_goals": ["Rectum max <= 50"],
        },
        result=None if dose is None else create_result({"physical_dose": dose}),
        cst=cst,
    )


def test_agent_result_tab_loads_plans_and_clear_run(qapp, agentic, test_data_photons):
    from types import SimpleNamespace

    from pyRadPlan.ai.agents import PlanningReport

    ct, cst, result = test_data_photons
    ws = WorkspaceManager()
    win = MainWindow(ws)
    ws.set_many(ct=ct, cst=cst)
    panel = win.agent_panel
    dose = result["physicalDose"]
    plans = [_agent_plan(1, cst, dose, hard_ok=False), _agent_plan(2, cst, dose)]
    panel._state = SimpleNamespace(plans=plans, pln=ws.pln, num_optimizations=2)
    for plan in plans:
        panel._on_agent_event("plan", {"plan": plan})
    report = PlanningReport(all_hard_goals_met=True, summary="Done.", trade_offs=["Core vs PTV"])
    panel._on_run_done(SimpleNamespace(output=report), None)
    # the picks are results of their own; the active result is the one before the run
    assert list(ws.results) == ["Plan_1"] and ws.result is None
    assert ws.results["Plan_1"] is plans[1].result  # the plan's own PlanResult

    # the report is in the result tab, not in the log
    text = panel.result_widget.report.toPlainText()
    assert "Core vs PTV" in text and "Rectum max <= 50" in text
    assert "Core vs PTV" not in panel._log.toPlainText()
    table = panel.result_widget.table
    rows = [[table.item(r, c).text() for c in range(1, 6)] for r in range(table.rowCount())]
    assert rows == [
        ["1", "1", "1", "1", "2 beams · 2 obj · 1 helper"],
        ["2 (Plan 1)", "0", "1", "2", "2 beams · 2 obj · 1 helper"],
    ]
    assert panel.progress_plot.legend_entries() == [
        "Core max <= 1.6",
        "PTV D95 >= 1.8 (soft)",
        "Score",
    ]

    # Load plan adds a tab once, and reuses the tab of a selected plan
    panel._on_load_requested(1)
    assert [win._agent_tabs.tabText(i) for i in range(3)] == ["Agent", "Plan 1", "Opt 1"]
    assert win._agent_tabs.currentIndex() == 2
    win._agent_tabs.setCurrentIndex(0)
    panel._on_load_requested(1)
    panel._on_load_requested(2)
    assert win._agent_tabs.count() == 3 and win._agent_tabs.currentIndex() == 1
    win._agent_tabs.setCurrentIndex(0)

    panel.clear_run()
    assert win._agent_tabs.count() == 1 and table.rowCount() == 0
    assert ws.result is None and not ws.results and not panel._log.toPlainText()
    assert ws.pln is not None  # the plan stays

    panel.agentic_plan_widget._spn_opt.setValue(3)
    panel.clear_everything()
    assert panel.agentic_plan_widget.max_optimizations == 8
    assert ws.ct is ct and ws.cst is cst and ws.dij is None


def test_agent_plan_line_names_blends_and_warm_starts(qapp):
    from pyRadPlan.gui.widgets.agent._agent_panel import _plan_line

    plan = _agent_plan(3, None, None)
    plan.evaluation["optimizer"] = {"iterations": 40, "converged": True, "start_from": 2}
    line = _plan_line(plan)
    assert line.startswith("Optimization 3: score 3") and "(converged) from plan 2" in line
    blend = _agent_plan(4, None, None)
    blend.blend = {"plans": [1, 3], "weights": [0.75, 0.25]}
    assert _plan_line(blend).startswith("Blend 4 (75 % plan 1 + 25 % plan 3): score 4")


def test_agent_pareto_line_names_the_compromise(qapp):
    from pyRadPlan.gui.widgets.agent._agent_panel import _pareto_line

    result = {"plans": [2, 5], "num_blends": 95, "front_size": 40}
    assert (
        _pareto_line(result) == "Pareto front of plans 2, 5: no blend of 95 meets all hard goals"
    )
    result["compromise"] = {
        "weights": [0.4, 0.6],
        "score": 11.2,
        "weakest_axes": [{"goal": "PAROTID_LT mean <= 26", "pct_of_range_from_best": 40.0}],
    }
    assert _pareto_line(result) == (
        "Pareto front of plans 2, 5: 40 of 95 blends on the front · compromise 40 % plan 2 + "
        "60 % plan 5 (score 11.2) · weakest PAROTID_LT mean"
    )


def test_agentic_plan_goals_form_the_protocol(qapp):
    pytest.importorskip("pydantic_ai")
    from pyRadPlan.ai.agents import Goal
    from pyRadPlan.gui.widgets.agent import AgenticPlanWidget

    widget = AgenticPlanWidget(WorkspaceManager(), tools={"optimize": "", "set_beams": ""})
    assert widget.protocol().goals == [] and not widget.protocol().locked
    goals = [
        Goal(voi="PTV", metric="D95", op=">=", value=1.9),
        Goal(voi="Cord", metric="max", op="<=", value=1.5, hard=False, source="table"),
    ]
    widget.set_goals(goals)
    protocol = widget.protocol()
    assert protocol.goals == goals and protocol.locked
    assert widget.tools() == ["optimize", "set_beams"]

    assert (widget.max_dij_calcs, widget.beam_range, widget.allow_couch, widget.max_iter) == (
        3,
        (1, 9),
        True,
        5000,
    )
    widget._spn_min_beams.setValue(12)  # the range stays valid
    assert widget.beam_range == (9, 9)
    widget._spn_max_beams.setValue(15)
    widget._spn_min_beams.setValue(12)
    widget._chk_couch.setChecked(False)
    widget.reset()
    assert widget.beam_range == (1, 9) and widget.allow_couch
    widget.set_goals(goals)

    widget._tbl_goals.item(0, 1).setText("D95%")
    with pytest.raises(ValueError, match="row 1"):
        widget.protocol()


def test_agentic_plan_goals_import_export(qapp, tmp_path, monkeypatch):
    pytest.importorskip("pydantic_ai")
    from pyRadPlan.ai.agents import Goal
    from pyRadPlan.gui.widgets.agent import AgenticPlanWidget
    from PySide6.QtWidgets import QMessageBox

    from pyRadPlan.plan import PhotonPlan

    shown = []
    monkeypatch.setattr(QMessageBox, "information", lambda *a: shown.append(a[2]))
    monkeypatch.setattr(QMessageBox, "warning", lambda *a: shown.append(a[2]))
    ws = WorkspaceManager()
    widget = AgenticPlanWidget(ws, tools={"optimize": ""})
    widget.import_goals(str(tmp_path / "missing.json"))
    assert "plan first" in shown.pop()

    ws.set_many(pln=PhotonPlan(num_of_fractions=20, dose_convention="total"))
    goals = [Goal(voi="PTV", metric="D95", op=">=", value=57.0, hard=False)]
    widget.set_goals(goals)
    path = tmp_path / "goals.json"
    widget.export_goals(str(path))
    widget.set_goals([])
    widget.import_goals(str(path))
    assert widget.goals() == goals and not shown

    ws.set_many(pln=PhotonPlan(num_of_fractions=19, dose_convention="per_fraction"))
    widget.import_goals(str(path))
    assert widget.goals()[0].value == 3.0 and "20 fractions" in shown.pop()
    path.write_text("not json")
    widget.import_goals(str(path))
    assert "Cannot read" in shown.pop() and widget.goals()[0].value == 3.0


def test_agent_pause_and_continue(qapp, agentic, test_data_photons, monkeypatch):
    from types import SimpleNamespace

    from PySide6.QtWidgets import QMessageBox
    from pydantic_ai import DeferredToolRequests
    from pydantic_ai.messages import ToolCallPart

    from pyRadPlan.ai.agents import PlanningReport, continue_planning_agent
    from pyRadPlan.gui.widgets.agent import AgentPanel

    ct, cst, result = test_data_photons
    ws = WorkspaceManager()
    win = MainWindow(ws)
    ws.set_many(ct=ct, cst=cst)
    panel = win.agent_panel
    started = []
    monkeypatch.setattr(panel, "_start", lambda *a, **k: started.append((a, k)))
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a: warnings.append(a[2]))

    # Pause asks the running agent to stop before its next optimization
    state = SimpleNamespace(plans=[], pln=ws.pln, num_optimizations=2, pause_requested=False)
    panel._state = state
    monkeypatch.setattr(AgentPanel, "is_busy", property(lambda self: True))
    panel.pause_agent()
    assert state.pause_requested and not panel._btn_pause.isEnabled()
    monkeypatch.setattr(AgentPanel, "is_busy", property(lambda self: False))

    call = ToolCallPart("optimize", {}, tool_call_id="c1")
    paused = SimpleNamespace(
        output=DeferredToolRequests(approvals=[call], metadata={"c1": {"reason": "review"}})
    )
    panel._on_run_done(paused, None)
    assert "review after 2 optimization(s)" in panel._lbl_status.text()
    assert not panel._feedback_box.isHidden() and not panel._btn_redirect.isHidden()
    assert panel.agentic_plan_widget._spn_opt.isEnabled()  # budgets can change now

    panel.agentic_plan_widget._spn_opt.setValue(12)
    panel.agentic_plan_widget._spn_review.setValue(3)
    panel._txt_feedback.setPlainText("Spare the parotids.")
    panel.continue_agent("redirect")
    ((args, kwargs),) = started
    assert args[:4] == (state, continue_planning_agent, paused, "Spare the parotids.")
    assert kwargs["action"] == "redirect" and kwargs["protocol"].goals == []
    assert state.max_optimizations == 12 and state.review_every == 3
    assert "Planner: redirect: Spare the parotids." in panel._log.toPlainText()
    assert not panel._txt_feedback.toPlainText()

    # After the report the box continues the conversation with a new instruction
    panel._state = state
    report = PlanningReport(all_hard_goals_met=True, summary="Done.")
    panel._on_run_done(SimpleNamespace(output=report), None)
    assert not panel._feedback_box.isHidden() and panel._btn_redirect.isHidden()
    panel.continue_agent()
    assert "instruction" in warnings.pop() and len(started) == 1
    panel._txt_feedback.setPlainText("Try seven beams.")
    panel.continue_agent()
    assert started[-1][0][3] == "Try seven beams." and started[-1][1]["action"] == "continue"

    panel.clear_run()
    assert panel._session is None and panel._feedback_box.isHidden()
