import numpy as np
import pytest
import SimpleITK as sitk

from pyRadPlan.gui.widgets._result_widget import ViewingWidget
from pyRadPlan.gui.workspace import WorkspaceManager


def _beam_labels(quantity_widget):
    """Text of the beam tip labels currently rendered, in drawing order."""
    import pyqtgraph as pg

    return [
        item.toPlainText() for item in quantity_widget._beam_items if isinstance(item, pg.TextItem)
    ]


def _marker_symbols(quantity_widget):
    """Symbols of the ⊙ / ⊗ markers of beams along the viewing direction."""
    import pyqtgraph as pg

    return [
        [str(spot.symbol()) for spot in item.points()]
        for item in quantity_widget._beam_items
        if isinstance(item, pg.ScatterPlotItem) and len(item.points()) == 2
    ]


def _make_workspace(ct, cst, result=None):
    ws = WorkspaceManager()
    ws.set_many(ct=ct, cst=cst, result=result)
    return ws


def test_viewing_widget_init(qapp):
    widget = ViewingWidget()
    assert widget is not None
    assert widget.quantity_widget is not None
    assert widget.vis_widget is not None
    assert widget.opts_widget is not None
    assert widget.vois_widget is not None


def test_viewing_widget_reacts_to_workspace(qapp, test_data_photons):
    ct, cst, result = test_data_photons

    ws = _make_workspace(ct, cst, result if isinstance(result, dict) else None)
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


def test_viewing_widget_raw_array_result(qapp, test_data_photons):
    ct, cst, _ = test_data_photons
    raw = np.ones(sitk.GetArrayFromImage(ct.cube_hu).shape)

    ws = _make_workspace(ct, cst, raw)
    widget = ViewingWidget(ws)

    assert widget.quantity_widget._active_quantity_name is not None
    assert widget.quantity_widget.get_available_quantities()


def test_viewing_widget_clears_on_workspace_clear(qapp, test_data_photons):
    ct, cst, result = test_data_photons

    ws = _make_workspace(ct, cst, result if isinstance(result, dict) else None)
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


def test_beam_overlay_derived_from_plan(qapp, test_data_photons):
    from pyRadPlan.plan import PhotonPlan

    ct, cst, _ = test_data_photons
    ws = _make_workspace(ct, cst)
    widget = ViewingWidget(ws)

    # Without a plan there is nothing to draw and the toggle stays disabled.
    assert widget.quantity_widget._beams == []
    assert not widget.vis_widget.beams_checkbox.isEnabled()

    ws.pln = PhotonPlan(prop_stf={"gantry_angles": [0, 90, 180], "couch_angles": [0, 0, 0]})

    beams = widget.quantity_widget._beams
    assert len(beams) == 3
    assert widget.vis_widget.beams_checkbox.isEnabled()

    spacing = np.array(ct.cube_hu.GetSpacing())
    # Gantry 0 comes from anterior (-y in LPS), 90 from the patient's left (+x).
    anterior = beams[0]["source_direction"] * np.array([spacing[2], spacing[0], spacing[1]])
    left = beams[1]["source_direction"] * np.array([spacing[2], spacing[0], spacing[1]])
    assert np.allclose(anterior, [0.0, 0.0, -1.0], atol=1e-9)
    assert np.allclose(left, [0.0, 1.0, 0.0], atol=1e-9)
    # All beams share the automatic (target-derived) isocenter.
    assert np.allclose(beams[0]["iso_center"], beams[1]["iso_center"])


def test_beam_overlay_accepts_numpy_plan_angles(qapp, test_data_photons):
    from pyRadPlan.plan import PhotonPlan

    ct, cst, _ = test_data_photons
    ws = _make_workspace(ct, cst)
    widget = ViewingWidget(ws)

    # As in the examples: numpy angles, a single couch angle for all beams and
    # an explicit isocenter.
    iso = cst.target_center_of_mass()
    ws.pln = PhotonPlan(
        prop_stf={
            "gantry_angles": np.linspace(0, 360, 4, endpoint=False),
            "couch_angles": np.array([0.0]),
            "iso_center": np.asarray(iso),
        }
    )

    beams = widget.quantity_widget._beams
    assert len(beams) == 4
    assert widget.vis_widget.beams_checkbox.isEnabled()
    assert "gantry 270°" in beams[3]["label"]


def test_beam_overlay_items_follow_the_toggle(qapp, test_data_photons):
    from pyRadPlan.plan import PhotonPlan

    ct, cst, _ = test_data_photons
    ws = _make_workspace(ct, cst)
    ws.pln = PhotonPlan(prop_stf={"gantry_angles": [0, 90], "couch_angles": [0, 0]})
    widget = ViewingWidget(ws)
    quantity_widget = widget.quantity_widget

    assert quantity_widget._beam_items == []

    widget.vis_widget.beams_checkbox.setChecked(True)
    quantity_widget.set_plane("Axial")
    # One line, one source marker and one tip label per beam.
    assert len(quantity_widget._beam_items) == 6
    assert _beam_labels(quantity_widget) == ["#0", "#1"]

    # The overlay is a projection: a beam along the viewing direction has none and
    # is marked at the isocenter instead. Gantry 90 comes from the patient's left,
    # the side the sagittal view is seen from, and gantry 0 from anterior, the
    # side of the coronal view; both sources lie in front, so both beams run into
    # the screen (a cross).
    quantity_widget.set_plane("Sagittal")
    assert _beam_labels(quantity_widget) == ["#0", "#1"]
    assert _marker_symbols(quantity_widget) == [["o", "x"]]
    quantity_widget.set_plane("Coronal")
    assert _beam_labels(quantity_widget) == ["#0", "#1"]
    assert _marker_symbols(quantity_widget) == [["o", "x"]]

    widget.vis_widget.beams_checkbox.setChecked(False)
    assert quantity_widget._beam_items == []


def test_beam_overlay_prefers_the_steering_information(qapp, test_data_photons):
    """Prefer generated steering information over plan-derived beam data."""
    from pyRadPlan.plan import PhotonPlan
    from pyRadPlan.stf import SteeringInformation

    ct, cst, _ = test_data_photons
    ws = _make_workspace(ct, cst)
    ws.pln = PhotonPlan(prop_stf={"gantry_angles": [0, 90], "couch_angles": [0, 0]})
    widget = ViewingWidget(ws)
    assert len(widget.quantity_widget._beams) == 2

    # A generated stf overrides the plan angles; here it holds a single beam.
    ws.stf = SteeringInformation(
        beams=[
            {
                "gantry_angle": 45.0,
                "couch_angle": 0.0,
                "iso_center": cst.target_center_of_mass(),
                "rays": [],
            }
        ]
    )
    beams = widget.quantity_widget._beams
    assert len(beams) == 1
    assert "45" in beams[0]["label"]
