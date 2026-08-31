import os

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QToolBar

from pyRadPlan.gui.windows._main_win import MainWindow
from pyRadPlan.gui.workspace import WorkspaceManager


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
    assert "&File" in titles and "&View" in titles and "&Settings" in titles


def test_view_menu_stubs_disabled(qapp):
    win = MainWindow(WorkspaceManager())
    actions = {a.text(): a for a in win._view_menu.actions() if a.text()}

    for stub in ("Screenshot", "Zoom In", "Pan", "Toggle Dark Mode"):
        assert not actions[stub].isEnabled()


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
    win = MainWindow(WorkspaceManager())
    default = qapp.font().pointSizeF()
    if default > 0:
        assert win.font().pointSizeF() < default


def test_settings_dialog_shows_subconfig_tabs(qapp):
    from pyRadPlan.gui.menus._settings import SettingsDialog

    dialog = SettingsDialog()
    titles = [dialog._tabs.tabText(i) for i in range(dialog._tabs.count())]
    # All top-level fields are sub-configurations, so there is no General tab.
    assert dialog._general_form is None
    assert "General" not in titles
    assert "XP (Backend)" in titles
    assert "AI" in titles
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
    monkeypatch.delenv("PYRADPLAN_XP_PREFER_GPU", raising=False)
    monkeypatch.delenv("PYRADPLAN_AI_AGENTS_MODEL", raising=False)
    monkeypatch.setenv("PYRADPLAN_AI_MODEL", "stale-legacy-value")

    try:
        dialog = SettingsDialog()
        dialog._sub_forms["xp"]._set_value("prefer_gpu", False)
        dialog._sub_forms["ai"]._set_value("agents_model", "test-model")
        dialog.apply()

        assert settings.xp.prefer_gpu is False
        assert os.environ["PYRADPLAN_XP_PREFER_GPU"] == "False"
        assert settings.ai.agents_model == "test-model"
        assert os.environ["PYRADPLAN_AI_AGENTS_MODEL"] == "test-model"
        # the legacy alias is cleared so it cannot override the new choice
        assert "PYRADPLAN_AI_MODEL" not in os.environ
    finally:
        os.environ.pop("PYRADPLAN_XP_PREFER_GPU", None)
        os.environ.pop("PYRADPLAN_AI_AGENTS_MODEL", None)
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
    assert labels == ["XP (Backend)…", "AI…", "Preferences…"]


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
