import numpy as np
import pytest
import SimpleITK as sitk

from pyRadPlan.gui.widgets._result_widget import ViewingWidget
from pyRadPlan.gui.workspace import WorkspaceManager
from pyRadPlan.result import create_result


def _make_workspace(ct, cst, result=None):
    ws = WorkspaceManager()
    ws.set_many(ct=ct, cst=cst, result=result)
    return ws


def _result_for(ct):
    """A minimal validated PlanResult on the CT grid for viewer tests."""
    return create_result({"physical_dose": sitk.Cast(ct.cube_hu, sitk.sitkFloat32)})


def _sequenced_result_for(ct):
    """A minimal validated PlanResult with a two-beam sequence attached."""
    from pyRadPlan.machines import create_bld
    from pyRadPlan.result._beam_sequence import BeamSequence
    from pyRadPlan.result import Sequence

    result = _result_for(ct)

    mlc_info = {
        "device_type": "MLCX",
        "device_orientation": "X",
        "leaf_position_boundaries": [-20, -10, 0, 10],
        "leaf_positions": [[-10, 5], [-10, 0], [-5, 0], [0, 0]],
        "leaf_width": 10.0,
        "leaf_leakage": 0.1,
    }
    beam = BeamSequence(
        mlc_sequence=[create_bld(mlc_info)], shapes_weight=[1.0], mlc_window=[-20, 20]
    )
    return result.model_copy(update={"sequence": Sequence(beam=[beam, beam])})


def test_viewing_widget_init(qapp):
    widget = ViewingWidget()
    assert widget is not None
    assert widget.quantity_widget is not None
    assert widget.vis_widget is not None
    assert widget.opts_widget is not None
    assert widget.vois_widget is not None


def test_viewing_widget_reacts_to_workspace(qapp, test_data_photons):
    ct, cst, result = test_data_photons

    # The result overlay is not asserted here; only CT/VOIs/colours are checked.
    ws = _make_workspace(ct, cst)
    widget = ViewingWidget(ws)

    # CT was derived from the workspace and pushed to the renderer
    assert widget.quantity_widget._ct is not None
    # VOIs were populated from the cst
    assert len(widget.vois_widget._voi_checkboxes) == len(cst.vois)
    # Colors propagated to the quantity widget for contour drawing
    assert len(widget.quantity_widget._voi_colors) > 0


def test_viewing_widget_updates_on_change(qapp, test_data_photons):
    ct, cst, result = test_data_photons

    ws = WorkspaceManager()
    widget = ViewingWidget(ws)
    # No CT yet -> renderer has no data
    assert widget.quantity_widget._ct is None

    ws.set_many(ct=ct, cst=cst)
    assert widget.quantity_widget._ct is not None
    assert len(widget.vois_widget._voi_checkboxes) == len(cst.vois)


def test_viewing_widget_signals(qapp):
    widget = ViewingWidget()

    received = []
    widget.overlay_toggled.connect(lambda n, c: received.append((n, c)))

    widget.vis_widget.overlay_toggled.emit("CT", False)

    assert len(received) > 0
    assert received[-1] == ("CT", False)


def test_viewing_widget_set_plane(qapp, test_data_photons):
    ct, cst, result = test_data_photons

    ws = _make_workspace(ct, cst)
    widget = ViewingWidget(ws)

    widget.set_plane("Sagittal")
    assert widget.quantity_widget._plane == "Sagittal"


def test_viewing_widget_single_quantity_result(qapp, test_data_photons):
    ct, cst, _ = test_data_photons

    ws = _make_workspace(ct, cst, _result_for(ct))
    widget = ViewingWidget(ws)

    assert widget.quantity_widget._active_quantity_name is not None
    assert widget.quantity_widget.get_available_quantities()


def test_viewing_widget_clears_on_workspace_clear(qapp, test_data_photons):
    ct, cst, result = test_data_photons

    ws = _make_workspace(ct, cst, _result_for(ct))
    widget = ViewingWidget(ws)
    assert widget.quantity_widget._ct is not None
    assert len(widget.vois_widget._voi_checkboxes) == len(cst.vois)

    ws.clear()

    assert widget.quantity_widget._ct is None
    assert len(widget.vois_widget._voi_checkboxes) == 0
    assert widget.quantity_widget._masks == {}
    assert widget.quantity_widget._quantities == {}


def test_viewing_widget_voi_replaced_writes_back_to_cst(qapp, test_data_photons):
    from pyRadPlan.cst import create_voi

    ct, cst, result = test_data_photons

    ws = _make_workspace(ct, cst)
    widget = ViewingWidget(ws)

    voi = ws.cst.vois[0]
    new_type = next(t for t in ("TARGET", "OAR") if t != voi.voi_type)
    data = dict(voi)
    data.update(voi_type=new_type)
    data.pop("default_color", None)
    new_voi = create_voi(data)

    selected_before = widget.vois_widget.selected_vois()
    widget.vois_widget.voi_replaced.emit(voi.name, new_voi)

    assert ws.cst.vois[0] is new_voi
    assert ws.cst.vois[0].voi_type == new_type
    # The viewer must not rebuild (and reset the selection) from its own write
    assert widget.vois_widget.selected_vois() == selected_before


def test_viewing_widget_deprecated_set_data(qapp):
    widget = ViewingWidget()
    ct_arr = np.zeros((5, 6, 7))

    with pytest.deprecated_call():
        widget.set_data(ct_arr, np.ones((5, 6, 7)))

    assert widget.quantity_widget._ct is not None
    assert widget.quantity_widget.get_available_quantities()


def test_viewing_widget_sequence_button(qapp, test_data_photons):
    ct, cst, _ = test_data_photons

    ws = _make_workspace(ct, cst, _result_for(ct))
    ws_seq = _make_workspace(ct, cst, _sequenced_result_for(ct))
    widget = ViewingWidget(ws)
    widget_seq = ViewingWidget(ws_seq)

    assert not widget.vis_widget.sequence_btn.isEnabled()
    assert widget_seq.vis_widget.sequence_btn.isEnabled()

    ws_seq.clear()
    assert not widget_seq.vis_widget.sequence_btn.isEnabled()


def test_viewing_widget_show_sequence(qapp, test_data_photons):
    ct, cst, _ = test_data_photons

    ws = _make_workspace(ct, cst, _sequenced_result_for(ct))
    widget = ViewingWidget(ws)

    widget._on_show_sequence()

    assert widget._sequence_window is not None
    assert widget._sequence_window.viewer.tabs.count() == 2

    widget.quantity_widget._active_quantity_name = "physical_dose_beam 1"
    widget._on_show_sequence()
    assert widget._sequence_window.viewer.tabs.currentIndex() == 1
