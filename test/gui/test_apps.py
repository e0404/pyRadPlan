from unittest.mock import patch

import numpy as np
import pytest
import SimpleITK as sitk
from PySide6.QtWidgets import QApplication

from pyRadPlan.gui import apps
from pyRadPlan.gui.apps import _voi_colors, analysis_viewer, gui, main


@pytest.fixture
def no_event_loop():
    """Keep analysis_viewer from blocking on the Qt event loop."""
    with patch.object(QApplication, "exec", lambda self: 0):
        yield


@pytest.fixture
def dose(test_data_photons):
    _, cst, result = test_data_photons
    shape = sitk.GetArrayFromImage(cst.vois[0].mask).shape
    return np.reshape(np.asarray(result["physicalDose"]).ravel()[: np.prod(shape)], shape)


def test_analysis_viewer_with_result_dict(qapp, no_event_loop, test_data_photons, dose):
    _, cst, _ = test_data_photons
    analysis_viewer(cst=cst, result={"physical_dose": dose})


def test_analysis_viewer_with_raw_array(qapp, no_event_loop, test_data_photons, dose):
    """A bare array is wrapped into a single-quantity result."""
    _, cst, _ = test_data_photons
    analysis_viewer(cst=cst, result=dose)


def test_analysis_viewer_with_sitk_image(qapp, no_event_loop, test_data_photons, dose):
    _, cst, _ = test_data_photons
    analysis_viewer(cst=cst, result={"physical_dose": sitk.GetImageFromArray(dose)})


def test_analysis_viewer_requires_cst(qapp, no_event_loop, dose):
    with pytest.raises(ValueError, match="structure set"):
        analysis_viewer(cst=None, result={"physical_dose": dose})


def test_analysis_viewer_requires_result(qapp, no_event_loop, test_data_photons):
    _, cst, _ = test_data_photons
    with pytest.raises(ValueError, match="result"):
        analysis_viewer(cst=cst, result=None)


def test_analysis_viewer_rejects_result_without_3d_quantity(
    qapp, no_event_loop, test_data_photons
):
    """Non-3D entries (e.g. the fluence vector 'w') are not analyzable."""
    _, cst, _ = test_data_photons
    with pytest.raises(ValueError, match="no 3D quantity"):
        analysis_viewer(cst=cst, result={"w": np.ones(10)})


def test_analysis_viewer_reads_workspace(qapp, no_event_loop, test_data_photons, dose):
    """With no arguments, cst and result come from the shared workspace."""
    from pyRadPlan.gui.workspace import WorkspaceManager

    _, cst, _ = test_data_photons
    workspace = WorkspaceManager.instance()
    try:
        workspace.set_many(cst=cst, result={"physical_dose": dose})
        analysis_viewer()
    finally:
        workspace.clear()


def test_voi_colors_fall_back_to_default_color(test_data_photons):
    """Every VOI gets a color: its visible_color if set, else its type default color."""
    _, cst, _ = test_data_photons
    first = cst.vois[0].model_copy(update={"visible_color": (1, 2, 3)})
    rest = [voi.model_copy(update={"visible_color": None}) for voi in cst.vois[1:]]
    cst = cst.model_copy(update={"vois": [first, *rest]})

    colors = _voi_colors(cst)

    assert set(colors) == {voi.name for voi in cst.vois}
    assert colors[first.name] == (1, 2, 3)
    for voi in rest:
        assert colors[voi.name] == tuple(voi.default_color)


def test_analysis_viewer_passes_voi_colors(qapp, no_event_loop, test_data_photons, dose):
    """The viewer hands the resolved VOI colors to the analysis window."""
    _, cst, _ = test_data_photons
    captured = {}

    def fake_show_analysis(**kwargs):
        captured.update(kwargs)
        return object()

    with patch("pyRadPlan.gui.windows._analysis_win.show_analysis", fake_show_analysis):
        analysis_viewer(cst=cst, result={"physical_dose": dose})

    assert captured["overlay"]["voi_colors"] == _voi_colors(cst)
    assert set(captured["overlay"]["voi_colors"]) == {voi.name for voi in cst.vois}


@pytest.fixture
def captured_main_window(monkeypatch):
    """Replace the main-window launcher and hand back the workspace it received."""
    import pyRadPlan.gui.windows._main_win as main_win
    from pyRadPlan.gui.workspace import WorkspaceManager

    launched = {}
    monkeypatch.setattr(main_win, "launch_main_window", lambda ws: launched.update(ws=ws))
    try:
        yield launched
    finally:
        WorkspaceManager.instance().clear()


def test_gui_resolves_bundled_phantom_name(captured_main_window):
    """`pyRadPlanGUI TG119` loads the bundled phantom instead of failing on the path."""
    gui("TG119")

    workspace = captured_main_window["ws"]
    assert workspace.has("ct", "cst")
    assert len(workspace.cst.vois) > 0


def test_gui_unknown_patient_names_phantoms(captured_main_window):
    with pytest.raises(FileNotFoundError, match="TG119"):
        gui("no_such_patient")
    assert "ws" not in captured_main_window


def test_main_passes_patient_to_gui(monkeypatch):
    received = []
    monkeypatch.setattr(apps, "gui", received.append)
    main(["TG119"])
    main([])
    assert received == ["TG119", None]
