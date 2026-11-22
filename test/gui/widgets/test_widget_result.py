import numpy as np
import pytest
from PySide6.QtGui import QColor
import SimpleITK as sitk

from pyRadPlan.gui.widgets._result_widget import ViewingWidget
from pyRadPlan.gui.workspace import WorkspaceManager
from pyRadPlan.result import create_result


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
    beam = BeamSequence.from_shapes([create_bld(mlc_info)], [1.0])
    return result.model_copy(update={"sequence": Sequence(beams=[beam, beam])})


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


def test_viewing_widget_reuses_unchanged_quantities(qapp, test_data_photons):
    """Unchanged images (also those a snapshot shares) are not converted again."""
    ct, cst, _ = test_data_photons
    ws = _make_workspace(ct, cst, _result_for(ct))
    widget = ViewingWidget(ws)
    before = widget._display_arrays(ws.result_entries(), ct.cube_hu)["physical_dose"]

    ws.add_result("saved", ws.result.model_copy())

    arrays = widget._display_arrays(ws.result_entries(), ct.cube_hu)
    assert list(arrays) == ["physical_dose", "saved/physical_dose"]
    assert arrays["physical_dose"] is before
    assert arrays["saved/physical_dose"] is before
    assert list(widget.quantity_widget._quantities) == list(arrays)


def test_viewing_widget_shows_results_of_other_grids_and_scenarios(qapp, test_data_photons):
    """Each result is listed with its own scenario labels, resampled onto the CT."""
    ct, cst, _ = test_data_photons
    dose = _result_for(ct).physical_dose.values
    ws = _make_workspace(ct, cst, None)
    ws.result = {
        "physical_dose": dose,
        "physical_dose_scenario": [dose, dose],
        "scenario_labels": ("nominal", "shifted"),
    }
    coarse = sitk.Image([s // 2 for s in dose.GetSize()], sitk.sitkFloat32)
    coarse.SetSpacing([2.0 * s for s in dose.GetSpacing()])
    coarse.SetOrigin(dose.GetOrigin())
    coarse.SetDirection(dose.GetDirection())
    ws.add_result(
        "measured",
        {
            "physical_dose": coarse + 1.0,
            "physical_dose_scenario": [coarse, coarse, coarse],
            "scenario_labels": ("a", "b", "c"),
        },
    )
    widget = ViewingWidget(ws)

    quantities = widget.quantity_widget._quantities
    assert "measured/physical_dose" in quantities
    assert "measured/physical_dose_scenario 2" in quantities
    assert quantities["measured/physical_dose"].shape == quantities["physical_dose"].shape
    assert quantities["measured/physical_dose"].max() == pytest.approx(1.0)

    selector = widget.vis_widget.quantity_selector
    texts = {}
    for key in ("physical_dose_scenario 1", "measured/physical_dose_scenario 1"):
        assert selector.set_current_key(key)
        texts[key] = selector.component_combo.currentText()
    assert texts == {
        "physical_dose_scenario 1": "scenario 1: shifted",
        "measured/physical_dose_scenario 1": "scenario 1: b",
    }
    assert selector.quantity_combo.currentText() == "dose (measured)"


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
    data.pop("color", None)
    new_voi = create_voi(data)

    selected_before = widget.vois_widget.selected_vois()
    widget.vois_widget.voi_replaced.emit(voi, new_voi)

    # The session recreates the VOI from the edited fields; the list holds the stored one.
    stored = ws.cst.vois[0]
    assert stored.voi_type == new_type and stored.name == voi.name
    assert widget.vois_widget._voi_by_name[voi.name] is stored
    # The viewer must not rebuild (and reset the selection) from its own write
    assert set(widget.vois_widget.selected_vois()) == set(selected_before)


def test_structure_edits_during_a_run(qapp, test_data_photons, monkeypatch):
    """A colour pick applies at once; a metadata edit waits for the run to end."""
    from pyRadPlan.app import Operation
    from pyRadPlan.cst import recreate_voi

    ct, cst, result = test_data_photons
    ws = _make_workspace(ct, cst)
    widget = ViewingWidget(ws)
    ws.set_many(result={"physical_dose": sitk.Image([2, 2, 2], sitk.sitkFloat32)})
    session = ws.session
    operation = Operation("calc", "Calculating", lambda: None, lambda value: None)
    session.begin(operation)
    voi = ws.cst.vois[0]
    picked = QColor(1, 2, 3)
    monkeypatch.setattr(
        "pyRadPlan.gui.widgets.result.vois_widget.QColorDialog.getColor", lambda *a: picked
    )
    changes = []
    ws.workspace_changed.connect(changes.append)

    widget.vois_widget._pick_color(voi)
    new_type = next(t for t in ("TARGET", "OAR") if t != voi.voi_type)
    widget.vois_widget.voi_replaced.emit(voi, recreate_voi(voi, voi_type=new_type))

    assert tuple(ws.cst.vois[0].color) == (1, 2, 3)
    assert ws.cst.vois[0] is voi and changes == []
    session.finish(operation, None)
    assert ws.cst.vois[0].voi_type == new_type and changes == [["cst"]]
    assert session.stages()["optimization"] == "stale"


def test_voi_tissue_edit_marks_dose_influence_stale(qapp, test_data_photons):
    """A tissue edit in the VOI popup must reach the workflow widget's indicators."""
    from pyRadPlan.gui.widgets.workflow import WorkflowWidget
    from pyRadPlan.gui.widgets.result.vois_widget import VOIMetadataDialog
    from PySide6.QtWidgets import QDialog

    ct, cst, result = test_data_photons

    ws = _make_workspace(ct, cst)
    widget = ViewingWidget(ws)
    workflow = WorkflowWidget(ws)
    ws.set_many(
        pln=object(),
        stf=object(),
        dij=object(),
        result={"physical_dose": sitk.Image([2, 2, 2], sitk.sitkFloat32)},
    )
    assert not workflow.session.tracker.dij_stale

    def fake_exec(self):
        self.alpha_spin.setValue(0.42)
        self.accept()
        return QDialog.DialogCode.Accepted

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(VOIMetadataDialog, "exec", fake_exec)
        widget.vois_widget._edit_voi(ws.cst.vois[0])

    assert ws.cst.vois[0].tissue_properties.alpha_x == pytest.approx(0.42)
    assert workflow.session.tracker.dij_stale
    assert "recalculate" in workflow._lbl_status.text().lower()


def test_voi_metadata_dialog_retype_rebinds_auto_color(qapp, test_data_photons):
    from pyRadPlan.cst import DEFAULT_VOI_COLORS, StructureSet, create_voi
    from pyRadPlan.gui.widgets.result.vois_widget import VOIMetadataDialog

    ct, cst, _ = test_data_photons

    # A VOI that never had a color picked carries its type's auto-assigned one.
    # (The matRad fixture ships explicit visibleColors, so build a fresh VOI.)
    # It must reach the dialog through a StructureSet, as GUI VOIs always do:
    # set_colors() assigns the field there but keeps it out of model_fields_set,
    # which is what tells an auto-assigned color apart from a picked one.
    voi = create_voi(voi_type="TARGET", name="auto", ct_image=ct, mask=cst.vois[0].mask)
    voi = StructureSet(ct_image=ct, vois=[voi]).vois[0]
    assert tuple(voi.color) == DEFAULT_VOI_COLORS["TARGET"][0]
    assert "color" not in voi.model_fields_set
    new_type = "OAR"

    dialog = VOIMetadataDialog(voi)
    dialog.type_combo.setCurrentText(new_type)
    dialog.accept()

    assert dialog.updated_voi.voi_type == new_type
    assert tuple(dialog.updated_voi.color) == DEFAULT_VOI_COLORS[new_type][0]


def test_voi_metadata_dialog_retype_keeps_picked_color(qapp, test_data_photons):
    from pyRadPlan.cst import DEFAULT_VOI_COLORS
    from pyRadPlan.gui.widgets.result.vois_widget import VOIMetadataDialog

    _, cst, _ = test_data_photons

    voi = cst.vois[0]
    picked = (7, 8, 9)
    voi.color = picked
    new_type = next(t for t in ("TARGET", "OAR") if t != voi.voi_type)

    dialog = VOIMetadataDialog(voi)
    dialog.type_combo.setCurrentText(new_type)
    dialog.accept()

    assert dialog.updated_voi.voi_type == new_type
    assert tuple(dialog.updated_voi.color) == picked
    assert picked not in DEFAULT_VOI_COLORS[new_type]


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


def test_show_sequence_reuses_the_same_window(qapp, test_data_photons):
    ct, cst, _ = test_data_photons

    ws = _make_workspace(ct, cst, _sequenced_result_for(ct))
    widget = ViewingWidget(ws)

    widget._on_show_sequence()
    first = widget._sequence_window
    widget._on_show_sequence()

    assert widget._sequence_window is first


def test_sequence_window_is_top_level_for_its_own_taskbar_entry(qapp, test_data_photons):
    ct, cst, _ = test_data_photons

    ws = _make_workspace(ct, cst, _sequenced_result_for(ct))
    widget = ViewingWidget(ws)
    widget._on_show_sequence()

    assert widget._sequence_window.parent() is None


def test_reshowing_the_same_sequence_does_not_rebuild_the_viewer(qapp, test_data_photons):
    ct, cst, _ = test_data_photons

    ws = _make_workspace(ct, cst, _sequenced_result_for(ct))
    widget = ViewingWidget(ws)

    widget._on_show_sequence()
    tabs = widget._sequence_window.viewer._beam_widgets[:]

    widget._on_show_sequence()

    assert widget._sequence_window.viewer._beam_widgets == tabs


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
    """Without a plan, the generated steering information provides the beams."""
    from pyRadPlan.stf import SteeringInformation

    ct, cst, _ = test_data_photons
    ws = _make_workspace(ct, cst)
    iso = np.asarray(cst.target_center_of_mass(), dtype=float).ravel() + [10.0, 0.0, 0.0]
    ws.stf = SteeringInformation(
        beams=[
            {"gantry_angle": angle, "couch_angle": 0.0, "iso_center": iso, "rays": []}
            for angle in (0.0, 90.0)
        ]
    )
    widget = ViewingWidget(ws)

    beams = widget.quantity_widget._beams
    spacing = np.array(ct.cube_hu.GetSpacing())
    origin = np.array(ct.cube_hu.GetOrigin())
    expected = (iso - origin) / spacing
    assert len(beams) == 2
    assert np.allclose(beams[0]["iso_center"], expected[[2, 0, 1]])


def test_isocenter_marker_and_beams_follow_the_plan_isocenter(qapp, test_data_photons):
    """A plan isocenter moves the marker and the beams, also those of an existing stf."""
    from pyRadPlan.plan import PhotonPlan
    from pyRadPlan.stf import SteeringInformation

    ct, cst, _ = test_data_photons
    ws = _make_workspace(ct, cst)
    com = np.asarray(cst.target_center_of_mass(), dtype=float).ravel()
    ws.pln = PhotonPlan(prop_stf={"gantry_angles": [0], "couch_angles": [0]})
    ws.stf = SteeringInformation(
        beams=[{"gantry_angle": 0.0, "couch_angle": 0.0, "iso_center": com, "rays": []}]
    )
    widget = ViewingWidget(ws)
    spacing = np.array(ct.cube_hu.GetSpacing())
    origin = np.array(ct.cube_hu.GetOrigin())
    assert np.allclose(widget.quantity_widget._isocenter, ((com - origin) / spacing)[[2, 0, 1]])

    iso = com + [10.0, 0.0, 0.0]
    ws.pln = PhotonPlan(
        prop_stf={"gantry_angles": [0], "couch_angles": [0], "iso_center": list(iso)}
    )
    expected = ((iso - origin) / spacing)[[2, 0, 1]]
    assert np.allclose(widget.quantity_widget._isocenter, expected)
    assert np.allclose(widget.quantity_widget._beams[0]["iso_center"], expected)


def test_automatic_isocenter_moves_the_beams_of_an_existing_stf(qapp, test_data_photons):
    """In auto mode the beams are drawn at the target center of mass, as Recalculate Dose does."""
    from pyRadPlan.plan import PhotonPlan
    from pyRadPlan.stf import SteeringInformation

    ct, cst, _ = test_data_photons
    ws = _make_workspace(ct, cst)
    com = np.asarray(cst.target_center_of_mass(), dtype=float).ravel()
    ws.pln = PhotonPlan(prop_stf={"gantry_angles": [0], "couch_angles": [0]})
    ws.stf = SteeringInformation(
        beams=[
            {
                "gantry_angle": 0.0,
                "couch_angle": 0.0,
                "iso_center": com + [10.0, 0.0, 0.0],
                "rays": [],
            }
        ]
    )
    widget = ViewingWidget(ws)

    expected = ((com - np.array(ct.cube_hu.GetOrigin())) / np.array(ct.cube_hu.GetSpacing()))[
        [2, 0, 1]
    ]
    assert np.allclose(widget.quantity_widget._isocenter, expected)
    assert np.allclose(widget.quantity_widget._beams[0]["iso_center"], expected)


@pytest.mark.parametrize("iso_center", [[], np.empty((0, 3))], ids=["list", "array"])
def test_empty_plan_isocenter_is_automatic(qapp, test_data_photons, iso_center):
    """An empty isocenter (e.g. from a MATLAB round-trip) selects the target center of mass."""
    from pyRadPlan.plan import PhotonPlan

    ct, cst, _ = test_data_photons
    ws = _make_workspace(ct, cst)
    ws.pln = PhotonPlan(prop_stf={"gantry_angles": [0], "iso_center": iso_center})
    widget = ViewingWidget(ws)

    com = np.asarray(cst.target_center_of_mass(), dtype=float).ravel()
    expected = ((com - np.array(ct.cube_hu.GetOrigin())) / np.array(ct.cube_hu.GetSpacing()))[
        [2, 0, 1]
    ]
    assert np.allclose(widget.quantity_widget._isocenter, expected)
    assert np.allclose(widget.quantity_widget._beams[0]["iso_center"], expected)


def test_per_beam_plan_isocenters(qapp, test_data_photons):
    """Each beam is drawn at its own isocenter; the marker shows the first beam's."""
    from pyRadPlan.plan import PhotonPlan

    ct, cst, _ = test_data_photons
    ws = _make_workspace(ct, cst)
    com = np.asarray(cst.target_center_of_mass(), dtype=float).ravel()
    isos = np.array([com, com + [10.0, 0.0, 0.0]])
    ws.pln = PhotonPlan(prop_stf={"gantry_angles": [0, 90], "iso_center": isos.tolist()})
    widget = ViewingWidget(ws)

    origin = np.array(ct.cube_hu.GetOrigin())
    spacing = np.array(ct.cube_hu.GetSpacing())
    expected = ((isos - origin) / spacing)[:, [2, 0, 1]]
    beams = widget.quantity_widget._beams
    assert np.allclose([beam["iso_center"] for beam in beams], expected)
    assert np.allclose(widget.quantity_widget._isocenter, expected[0])


def test_beam_overlay_ignores_outdated_steering_information(qapp, test_data_photons):
    """An stf whose angles no longer match the plan is outdated; the plan's beams are drawn."""
    from pyRadPlan.plan import PhotonPlan
    from pyRadPlan.stf import SteeringInformation

    ct, cst, _ = test_data_photons
    ws = _make_workspace(ct, cst)
    ws.pln = PhotonPlan(prop_stf={"gantry_angles": [0, 90], "couch_angles": [0, 0]})
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
    widget = ViewingWidget(ws)

    beams = widget.quantity_widget._beams
    assert [beam["gantry"] for beam in beams] == [0.0, 90.0]


class _DragEvent:
    """Stand-in for a pyqtgraph drag event at *pos* (view coordinates)."""

    def __init__(self, pos, start=False, finish=False):
        self._pos, self._start, self._finish = pos, start, finish

    def isStart(self):  # noqa: N802
        return self._start

    def isFinish(self):  # noqa: N802
        return self._finish

    def pos(self):
        from PySide6.QtCore import QPointF

        return QPointF(*self._pos)


def _beam_sources(quantity_widget):
    from pyRadPlan.gui.widgets.result.quantity_widget import _BeamSourceItem

    return [item for item in quantity_widget._beam_items if isinstance(item, _BeamSourceItem)]


def test_dragging_a_beam_changes_its_gantry_angle(qapp, test_data_photons):
    from pyRadPlan.plan import PhotonPlan

    ct, cst, _ = test_data_photons
    ws = _make_workspace(ct, cst)
    ws.pln = PhotonPlan(prop_stf={"gantry_angles": [0, 180], "couch_angles": [0]})
    widget = ViewingWidget(ws)
    quantity_widget = widget.quantity_widget
    widget.vis_widget.beams_checkbox.setChecked(True)
    quantity_widget.set_plane("Axial")
    assert all(source._on_drag is not None for source in _beam_sources(quantity_widget))

    # Gantry 90 has its source at the patient's left, to the right in the axial view.
    h_iso, v_iso = quantity_widget._view_point(quantity_widget._beams[0]["iso_center"])
    tip = quantity_widget._beam_parts[0]["tip"]
    quantity_widget._on_beam_drag(0, _DragEvent(tip, start=True))
    assert quantity_widget._beam_drag is not None  # the orbit is shown
    # Slightly downwards, i.e. posterior, is past 90 towards 180.
    quantity_widget._on_beam_drag(0, _DragEvent((h_iso + 5.0, v_iso + 0.1)))
    assert _beam_labels(quantity_widget)[0] == "#0  91°"
    quantity_widget._on_beam_drag(0, _DragEvent((h_iso + 5.0, v_iso), finish=True))
    assert quantity_widget._beam_drag is None
    source = quantity_widget._beam_parts[0]["source"]
    dropped = (source.data["x"][0], source.data["y"][0])
    qapp.processEvents()

    # Redrawn from the plan, the source stays where it was dropped.
    assert quantity_widget._beam_parts[0]["tip"] == pytest.approx(dropped)
    assert ws.pln.prop_stf["gantry_angles"] == [90.0, 180.0]
    assert [beam["gantry"] for beam in quantity_widget._beams] == [90.0, 180.0]
    assert _beam_labels(quantity_widget) == ["#0", "#1"]


def test_beams_drag_only_where_their_orbit_is_visible(qapp, test_data_photons):
    from pyRadPlan.plan import PhotonPlan

    ct, cst, _ = test_data_photons
    ws = _make_workspace(ct, cst)
    # The gantry of a couch-90 beam rotates in the sagittal plane.
    ws.pln = PhotonPlan(prop_stf={"gantry_angles": [0, 45], "couch_angles": [0, 90]})
    widget = ViewingWidget(ws)
    quantity_widget = widget.quantity_widget
    widget.vis_widget.beams_checkbox.setChecked(True)

    quantity_widget.set_plane("Axial")
    coplanar, rotated = _beam_sources(quantity_widget)
    assert coplanar._on_drag is not None and rotated._on_drag is None
    assert "Drag it in the Sagittal view" in rotated.toolTip()

    # The sagittal view sees the coplanar beam's orbit edge-on.
    quantity_widget.set_plane("Sagittal")
    coplanar, rotated = _beam_sources(quantity_widget)
    assert coplanar._on_drag is None and rotated._on_drag is not None
    assert "Drag it in the Axial view" in coplanar.toolTip()


def test_dragged_beams_go_to_the_beam_editor(qapp, test_data_photons):
    from pyRadPlan.plan import PhotonPlan
    from pyRadPlan.gui.widgets.result.quantity_widget import _BeamWedgeItem

    ct, cst, _ = test_data_photons
    ws = _make_workspace(ct, cst)
    ws.pln = PhotonPlan(prop_stf={"gantry_angles": [0, 180], "couch_angles": [0]})
    widget = ViewingWidget(ws)
    edits = []
    widget.set_beam_editor(lambda index, angle: edits.append((index, angle)))
    quantity_widget = widget.quantity_widget
    widget.vis_widget.beams_checkbox.setChecked(True)
    quantity_widget.set_plane("Axial")

    h_iso, v_iso = quantity_widget._view_point(quantity_widget._beams[0]["iso_center"])
    tip = quantity_widget._beam_parts[0]["tip"]
    quantity_widget._on_beam_drag(0, _DragEvent(tip, start=True))
    quantity_widget._on_beam_drag(0, _DragEvent((h_iso + 5.0, v_iso), finish=True))
    qapp.processEvents()
    assert edits == [(0, 90.0)]
    assert ws.pln.prop_stf["gantry_angles"] == [0, 180]

    # The editor's (not yet applied) angles are drawn, the changed beam marked.
    widget.set_edited_beam_angles(([90.0, 180.0], [0.0, 0.0]))
    assert [beam["gantry"] for beam in quantity_widget._beams] == [90.0, 180.0]
    assert [beam["pending"] for beam in quantity_widget._beams] == [True, False]
    assert _beam_labels(quantity_widget) == ["#0*", "#1"]
    wedges = [i for i in quantity_widget._beam_items if isinstance(i, _BeamWedgeItem)]
    assert [wedge.pending for wedge in wedges] == [True, False]

    widget.set_edited_beam_angles(None)
    assert [beam["gantry"] for beam in quantity_widget._beams] == [0.0, 180.0]
    assert _beam_labels(quantity_widget) == ["#0", "#1"]


def test_escape_cancels_a_beam_drag(qapp, test_data_photons):
    from pyRadPlan.plan import PhotonPlan

    ct, cst, _ = test_data_photons
    ws = _make_workspace(ct, cst)
    ws.pln = PhotonPlan(prop_stf={"gantry_angles": [0], "couch_angles": [0]})
    widget = ViewingWidget(ws)
    quantity_widget = widget.quantity_widget
    widget.vis_widget.beams_checkbox.setChecked(True)
    quantity_widget.set_plane("Axial")

    h_iso, v_iso = quantity_widget._view_point(quantity_widget._beams[0]["iso_center"])
    tip = quantity_widget._beam_parts[0]["tip"]
    quantity_widget._on_beam_drag(0, _DragEvent(tip, start=True))
    quantity_widget._on_beam_drag(0, _DragEvent((h_iso + 5.0, v_iso)))
    assert quantity_widget._cancel_drag_shortcut.isEnabled()
    quantity_widget._cancel_drag_shortcut.activated.emit()

    assert quantity_widget._beam_drag is None
    assert not quantity_widget._cancel_drag_shortcut.isEnabled()
    assert _beam_labels(quantity_widget) == ["#0"]
    source = quantity_widget._beam_parts[0]["source"]
    assert (source.data["x"][0], source.data["y"][0]) == pytest.approx(tip)
    # The rest of the drag is ignored.
    quantity_widget._on_beam_drag(0, _DragEvent((h_iso + 5.0, v_iso), finish=True))
    qapp.processEvents()
    assert ws.pln.prop_stf["gantry_angles"] == [0]


def test_orientation_marker_follows_the_plane_and_toggle(qapp, test_data_photons):
    ct, cst, _ = test_data_photons
    widget = ViewingWidget(_make_workspace(ct, cst))
    quantity_widget = widget.quantity_widget
    marker = quantity_widget._orientation_marker
    assert not marker.isHidden()

    # Axial is seen from the feet with anterior up; sagittal and coronal show the
    # patient lying with the head to the right, seen from the left and from anterior.
    expected = {
        "Axial": {"right": "L", "left": "R", "top": "A", "bottom": "P"},
        "Sagittal": {"right": "S", "left": "I", "top": "A", "bottom": "P"},
        "Coronal": {"right": "S", "left": "I", "top": "R", "bottom": "L"},
    }
    for plane, letters in expected.items():
        quantity_widget.set_plane(plane)
        assert marker.edge_letters() == letters
        assert not marker.grab().isNull()

    widget.vis_widget.orientation_checkbox.setChecked(False)
    assert marker.isHidden()
    widget.vis_widget.orientation_checkbox.setChecked(True)
    assert not marker.isHidden()

    widget.workspace.clear()
    assert marker.isHidden()


@pytest.mark.parametrize("angle", [30.0, 180.0])
def test_orientation_marker_uses_ct_direction(qapp, angle):
    from pyRadPlan.ct import validate_ct

    theta = np.radians(angle)
    c, s = np.cos(theta), np.sin(theta)
    direction = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    image = sitk.GetImageFromArray(np.zeros((5, 6, 7), dtype=np.float32))
    image.SetDirection(direction.ravel().tolist())
    ws = WorkspaceManager()
    widget = ViewingWidget(ws)
    ws.ct = validate_ct({"cube_hu": image})
    viewer = widget.quantity_widget
    marker = viewer._orientation_marker

    for plane, right_axis, down_axis in [("Axial", 0, 1), ("Sagittal", 2, 1), ("Coronal", 2, 0)]:
        viewer.set_plane(plane)
        np.testing.assert_allclose(marker._right, direction[:, right_axis])
        np.testing.assert_allclose(marker._down, direction[:, down_axis])
        assert not marker.grab().isNull()

    viewer.set_plane("Axial")
    if angle == 180.0:
        assert marker.edge_letters() == {"right": "R", "left": "L", "top": "P", "bottom": "A"}

    # Loading another CT must not retain the previous patient's orientation.
    image.SetDirection(np.eye(3).ravel().tolist())
    ws.ct = validate_ct({"cube_hu": image})
    assert marker.edge_letters() == {"right": "L", "left": "R", "top": "A", "bottom": "P"}


def test_orientation_geometry_defaults_to_identity(qapp):
    from pyRadPlan.gui.widgets.result.quantity_widget import QuantityWidget

    viewer = QuantityWidget()
    viewer.set_ct_geometry((0, 0, 0), (1, 1, 1), np.diag([-1, -1, 1]).ravel())
    assert viewer._orientation_marker.edge_letters()["right"] == "R"
    viewer.set_ct_geometry((0, 0, 0), (1, 1, 1))
    assert viewer._orientation_marker.edge_letters()["right"] == "L"


@pytest.mark.parametrize(
    ("keys", "expected"),
    [
        (["physical_dose", "rbe_x_dose", "let"], "rbe_x_dose"),
        (["let", "physical_dose"], "physical_dose"),
        (["let", "alpha_dose"], "let"),
        ([], None),
    ],
)
def test_default_quantity_prefers_rbe_weighted_dose(keys, expected):
    from pyRadPlan.gui.widgets._result_widget import default_quantity_key

    assert default_quantity_key(keys) == expected


def test_viewer_shows_rbe_weighted_dose_of_a_biological_result(qapp, test_data_photons):
    ct, cst, _ = test_data_photons
    dose = np.ones(sitk.GetArrayFromImage(ct.cube_hu).shape)

    widget = ViewingWidget(_make_workspace(ct, cst, {"physical_dose": dose}))
    assert widget.quantity_widget._active_quantity_name == "physical_dose"

    widget = ViewingWidget(
        _make_workspace(ct, cst, {"physical_dose": dose, "rbe_x_dose": 1.1 * dose})
    )
    assert widget.quantity_widget._active_quantity_name == "rbe_x_dose"
    assert widget.vis_widget.quantity_selector.current_key() == "rbe_x_dose"


def test_viewer_lists_results_per_scenario_with_dij_labels(qapp, test_data_photons):
    from types import SimpleNamespace

    ct, cst, _ = test_data_photons
    dose = np.ones(sitk.GetArrayFromImage(ct.cube_hu).shape)
    result = {"physical_dose": dose, "physical_dose_scenario": [dose, None, 1.1 * dose]}
    ws = _make_workspace(ct, cst, result)
    widget = ViewingWidget(ws)

    # The derived scenario 1 has no image; the others keep their flat index.
    available = widget.quantity_widget.get_available_quantities()
    assert "physical_dose_scenario 0" in available and "physical_dose_scenario 2" in available
    assert "physical_dose_scenario 1" not in available

    selector = widget.vis_widget.quantity_selector
    combo = selector.component_combo

    def components():
        return [combo.itemText(i) for i in range(combo.count())]

    assert components() == ["full", "scenario 0", "scenario 2"]
    ws.dij = SimpleNamespace(scenario_labels=lambda: ["nominal", "derived", "range +1.0 mm"])
    assert components() == ["full", "scenario 0: nominal", "scenario 2: range +1.0 mm"]
    ws.dij = None
    assert components() == ["full", "scenario 0", "scenario 2"]


def test_overlay_meta_covers_per_scenario_lists():
    from pyRadPlan.gui.widgets._result_widget import _build_overlay_meta

    arr = np.zeros((2, 2, 2))
    units, labels = _build_overlay_meta(
        {"rbe_x_dose": arr, "rbe_x_dose_scenario": [arr, None], "physical_dose_beam": [arr]}
    )
    assert labels["rbe_x_dose_scenario 0"] == "RBE-weighted Dose"
    assert units["rbe_x_dose_scenario 1"] == "Gy (RBE)"
    assert labels["physical_dose_beam 0"] == "Dose"


def test_overlay_meta_resolves_other_results_and_shares_rbe_axis():
    from pyRadPlan.analysis._units import compare_units
    from pyRadPlan.gui.widgets._result_widget import _build_overlay_meta

    arr = np.zeros((2, 2, 2))
    units, labels = _build_overlay_meta(
        {
            "rbe_x_dose": arr,
            "physical_dose": arr,
            "carbon/rbe_x_dose": arr,
            "carbon/physical_dose": arr,
            "carbon/effect": arr,
            "carbon/rbe_x_dose_beam": [arr],
            "effect": arr,
        }
    )
    assert units["carbon/rbe_x_dose"] == units["rbe_x_dose"] == "Gy (RBE)"
    assert labels["carbon/rbe_x_dose"] == labels["carbon/rbe_x_dose_beam 0"] == "RBE-weighted Dose"
    assert compare_units(units["rbe_x_dose"], units["carbon/rbe_x_dose"]) == (True, 1.0)
    assert compare_units(units["rbe_x_dose"], units["carbon/effect"])[0] is False


def test_beam_axes_are_mapped_to_the_viewer_voxel_space():
    """The shared LPS beam geometry, in the viewer's anisotropic (z, x, y) voxel space."""
    from types import SimpleNamespace

    from pyRadPlan.gui.widgets._result_widget import _viewer_beams

    image = sitk.Image([4, 5, 6], sitk.sitkFloat32)
    image.SetOrigin((10.0, 20.0, 30.0))
    image.SetSpacing((2.0, 4.0, 5.0))
    ct = SimpleNamespace(cube_hu=image)
    stf_beam = {
        "index": 0,
        "gantry": 0.0,
        "couch": 0.0,
        "iso_center": [12.0, 28.0, 45.0],
        "source_direction": [0.0, -1.0, 0.0],
        "name": "#0",
        "label": "Beam #0: gantry 0°, couch 0°",
        "pending": False,
    }
    plan_beam = {
        **stf_beam,
        "index": 1,
        "name": "#1",
        "orbit_u": [0.0, -1.0, 0.0],
        "orbit_w": [1.0, 0.0, 0.0],
        "orbit_axis": [0.0, 0.0, 1.0],
        "pending": True,
    }

    beams = _viewer_beams(ct, {"beams": [stf_beam, plan_beam]})

    np.testing.assert_allclose(beams[0]["iso_center"], [3.0, 1.0, 2.0])
    np.testing.assert_allclose(beams[0]["source_direction"], [0.0, 0.0, -0.25])
    # Only beams with plan angles (an orbit) can be rotated by dragging.
    assert set(beams[0]) == {"iso_center", "source_direction", "name", "label"}
    assert (beams[1]["index"], beams[1]["gantry"]) == (1, 0.0)
    np.testing.assert_allclose(beams[1]["orbit"][1], [0.0, 0.5, 0.0])
    np.testing.assert_allclose(beams[1]["orbit_axis"], [0.2, 0.0, 0.0])
    assert "pending" not in beams[1]
    edited = _viewer_beams(ct, {"beams": [stf_beam, plan_beam]}, edited=True)
    assert [beam["pending"] for beam in edited] == [False, True]


def test_viewer_prefers_the_scenario_labels_of_the_result(qapp, test_data_photons):
    from types import SimpleNamespace

    ct, cst, _ = test_data_photons
    dose = np.ones(sitk.GetArrayFromImage(ct.cube_hu).shape)
    ws = _make_workspace(ct, cst, {"physical_dose": dose, "physical_dose_scenario": [dose, dose]})
    widget = ViewingWidget(ws)
    combo = widget.vis_widget.quantity_selector.component_combo

    ws.dij = SimpleNamespace(scenario_labels=lambda: ["from dij 0", "from dij 1"])
    assert combo.itemText(2) == "scenario 1: from dij 1"

    ws.result = {
        "physical_dose": dose,
        "physical_dose_scenario": [dose, dose],
        "scenario_labels": ("from result 0", "from result 1"),
    }
    assert combo.itemText(2) == "scenario 1: from result 1"
    assert "scenario_labels" not in widget.quantity_widget.get_available_quantities()


def _duplicate_named_workspace(test_data_photons):
    from pyRadPlan.cst import recreate_voi

    ct, cst, result = test_data_photons
    cst.vois = [
        recreate_voi(voi, name=cst.vois[0].name) if i == 1 else voi
        for i, voi in enumerate(cst.vois)
    ]
    ws = _make_workspace(ct, cst)
    return ws, ViewingWidget(ws)


def test_structure_edits_address_structures_sharing_a_name(qapp, test_data_photons, monkeypatch):
    from pyRadPlan.cst import recreate_voi

    ws, widget = _duplicate_named_workspace(test_data_photons)
    first, second = ws.cst.vois[0], ws.cst.vois[1]
    assert first.name == second.name
    first_before = (first.voi_type, tuple(first.color))
    new_type = next(t for t in ("TARGET", "OAR") if t != second.voi_type)
    monkeypatch.setattr(
        "pyRadPlan.gui.widgets.result.vois_widget.QColorDialog.getColor",
        lambda *a: QColor(4, 5, 6),
    )

    widget.vois_widget._pick_color(second)
    widget.vois_widget.voi_replaced.emit(second, recreate_voi(second, voi_type=new_type))

    assert tuple(ws.cst.vois[1].color) == (4, 5, 6)
    assert ws.cst.vois[1].voi_type == new_type
    assert (ws.cst.vois[0].voi_type, tuple(ws.cst.vois[0].color)) == first_before


def test_deferred_structure_edit_is_dropped_when_the_structure_set_changed(
    qapp, test_data_photons
):
    import copy

    from pyRadPlan.app import Operation
    from pyRadPlan.cst import recreate_voi

    ct, cst, result = test_data_photons
    ws = _make_workspace(ct, cst)
    widget = ViewingWidget(ws)
    session = ws.session
    operation = Operation("calc", "Calculating", lambda: None, lambda value: None)
    session.begin(operation)
    voi = ws.cst.vois[0]
    new_type = next(t for t in ("TARGET", "OAR") if t != voi.voi_type)
    widget.vois_widget.voi_replaced.emit(voi, recreate_voi(voi, voi_type=new_type))

    other = copy.deepcopy(cst)
    original_type = other.vois[0].voi_type
    ws.set_many(cst=other)
    session.finish(operation, None)

    assert ws.cst is other
    assert ws.cst.vois[0].voi_type == original_type
