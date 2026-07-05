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
    # ct+cst but no pln yet -> workflow prompts to configure a plan
    assert "no data loaded" not in win.workflow_widget._lbl_status.text().lower()
    assert "plan" in win.workflow_widget._lbl_status.text().lower()


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


def test_file_menu_import_dose_tracks_ct(qapp, test_data_photons):
    ct, cst, _ = test_data_photons
    ws = WorkspaceManager()
    win = MainWindow(ws)

    fm = win._file_menu
    assert fm._act_load_mat.isEnabled()
    assert not fm._act_export_dicom.isEnabled()  # not yet implemented
    assert not fm._act_import_dose.isEnabled()  # no CT loaded

    ws.set_many(ct=ct, cst=cst)
    assert fm._act_import_dose.isEnabled()


def test_compact_font_is_smaller(qapp):
    win = MainWindow(WorkspaceManager())
    default = qapp.font().pointSizeF()
    if default > 0:
        assert win.font().pointSizeF() < default


def test_settings_menu_applies_ai_settings_to_env(qapp, monkeypatch):
    pytest.importorskip("pydantic_ai")
    win = MainWindow(WorkspaceManager())
    menu = win._settings_menu

    # Stub the dialog so the test doesn't open a modal window.
    class _FakeDialog:
        def __init__(self, *a, **k):
            pass

        def exec(self):
            from PySide6.QtWidgets import QDialog

            return QDialog.DialogCode.Accepted

        def values(self):
            return {"model": "test-model", "display_usage": False}

    monkeypatch.setattr("pyRadPlan.gui.widgets.ConfigFormDialog", _FakeDialog)
    monkeypatch.delenv("PYRADPLAN_AI_MODEL", raising=False)

    menu._edit_ai_settings()

    assert os.environ["PYRADPLAN_AI_MODEL"] == "test-model"


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
