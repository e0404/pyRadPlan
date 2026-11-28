import numpy as np

from pyRadPlan.machines import Jaw, create_bld
from pyRadPlan.result import PlanResult, Sequence
from pyRadPlan.result._beam_sequence import BeamSequence
from pyRadPlan.stf import Beam, Ray, SteeringInformation
from pyRadPlan.stf._beamlet import PhotonBixel

from pyRadPlan.gui.widgets.result.aperture_widget import (
    ApertureViewerWidget,
    BeamApertureWidget,
    SequenceViewContext,
)


def _jaws() -> list[Jaw]:
    return [
        Jaw(device_orientation="X", positions=[-20.0, 20.0], field_width=40.0),
        Jaw(device_orientation="Y", positions=[-20.0, 20.0], field_width=40.0),
    ]


def _make_beam(n_shapes: int = 2) -> BeamSequence:
    mlc_info = {
        "device_type": "MLCX",
        "device_orientation": "X",
        "leaf_position_boundaries": [-20, -10, 0, 10],
        "leaf_positions": [[-10, 5], [-10, 0], [-5, 0], [0, 0]],
        "leaf_width": 10.0,
        "leaf_leakage": 0.1,
    }
    shapes = [create_bld(mlc_info) for _ in range(n_shapes)]
    weights = list(np.linspace(1.0, 0.3, n_shapes))
    return BeamSequence.from_shapes(shapes, weights, jaws=_jaws())


def test_beam_aperture_widget_init(qapp):
    beam = _make_beam(n_shapes=2)
    widget = BeamApertureWidget(beam, "Beam 0")

    assert widget.shape_slider.maximum() == 1
    assert widget.play_button.isEnabled()
    assert widget.shape_label.text() == "1 / 2"


def test_beam_aperture_widget_single_shape_disables_playback(qapp):
    beam = _make_beam(n_shapes=1)
    widget = BeamApertureWidget(beam, "Beam 0")

    assert not widget.shape_slider.isEnabled()
    assert not widget.play_button.isEnabled()


def test_beam_aperture_widget_playback(qapp):
    beam = _make_beam(n_shapes=3)
    widget = BeamApertureWidget(beam, "Beam 0")

    widget.play_button.setChecked(True)
    assert widget._timer.isActive()

    widget.stop()
    assert not widget._timer.isActive()
    assert not widget.play_button.isChecked()


def test_beam_aperture_widget_slider_updates_label(qapp):
    beam = _make_beam(n_shapes=3)
    widget = BeamApertureWidget(beam, "Beam 0")

    widget.shape_slider.setValue(2)
    assert widget.shape_label.text() == "3 / 3"


def test_aperture_viewer_widget_set_sequence(qapp):
    sequence = Sequence(beams=[_make_beam(), _make_beam()])

    viewer = ApertureViewerWidget()
    viewer.set_sequence(sequence)

    assert viewer.tabs.count() == 2
    assert len(viewer._beam_widgets) == 2


def test_aperture_viewer_widget_rebuild_stops_previous_playback(qapp):
    sequence = Sequence(beams=[_make_beam()])

    viewer = ApertureViewerWidget()
    viewer.set_sequence(sequence)
    viewer._beam_widgets[0].play_button.setChecked(True)
    assert viewer._beam_widgets[0]._timer.isActive()

    viewer.set_sequence(sequence)
    assert viewer.tabs.count() == 1


def test_aperture_viewer_widget_shows_sequencer_label(qapp):
    sequence = Sequence(beams=[_make_beam()], sequencer="xia")

    viewer = ApertureViewerWidget()
    viewer.set_sequence(sequence)

    assert not viewer.sequencer_label.isHidden()
    assert viewer.sequencer_label.text() == "Sequencer: xia"


def test_aperture_viewer_widget_hides_sequencer_label_when_unknown(qapp):
    sequence = Sequence(beams=[_make_beam()])

    viewer = ApertureViewerWidget()
    viewer.set_sequence(sequence)

    assert viewer.sequencer_label.isHidden()


def _make_ray(x: float, z: float) -> Ray:
    return Ray(ray_pos_bev=[x, 1000.0, z], ray_pos=[x, 0.0, z], beamlets=[PhotonBixel(energy=6.0)])


def _make_stf(n_beams: int = 1) -> SteeringInformation:
    rays = [_make_ray(x, z) for x, z in ((-2.5, -2.5), (2.5, -2.5), (-2.5, 2.5), (2.5, 2.5))]
    beams = [Beam(iso_center=[0.0, 0.0, 0.0], bixel_width=5.0, rays=rays) for _ in range(n_beams)]
    return SteeringInformation(beams=beams)


def _fluence_context(n_beams: int = 1) -> SequenceViewContext:
    return SequenceViewContext(
        stf=_make_stf(n_beams),
        result=PlanResult(fluence=np.array([1.0, 2.0, 3.0, 4.0] * n_beams)),
    )


def test_no_fluence_panel_without_context(qapp):
    widget = BeamApertureWidget(_make_beam(), "Beam 0")

    assert not widget.has_fluence


def test_fluence_panel_offers_optimized_delivered_and_difference(qapp):
    widget = BeamApertureWidget(
        _make_beam(),
        "Beam 0",
        beam_ix=0,
        context=_fluence_context(),
        delivered_fluence=np.array([4.0, 3.0, 2.0, 1.0]),
    )

    assert widget.has_fluence
    assert set(widget._fluence_grids) == {"optimized", "delivered", "difference"}
    np.testing.assert_allclose(
        widget._fluence_grids["difference"],
        widget._fluence_grids["delivered"] - widget._fluence_grids["optimized"],
    )
    assert widget.fluence_mode == "delivered"


def test_overlay_bar_hidden_when_no_beam_has_fluence(qapp):
    viewer = ApertureViewerWidget()
    viewer.set_sequence(Sequence(beams=[_make_beam()]))

    assert viewer.overlay_bar.isHidden()


def test_fluence_selector_applies_to_every_beam_tab(qapp):
    context = _fluence_context(n_beams=2)
    sequence = Sequence(
        beams=[_make_beam(), _make_beam()],
        fluence=np.array([4.0, 3.0, 2.0, 1.0] * 2),
    )

    viewer = ApertureViewerWidget()
    viewer.set_sequence(sequence, context=context)
    assert all(w.fluence_mode == "delivered" for w in viewer._beam_widgets)

    difference_ix = viewer.fluence_selector.findData("difference")
    viewer.fluence_selector.setCurrentIndex(difference_ix)

    assert all(w.fluence_mode == "difference" for w in viewer._beam_widgets)


def test_overlay_checkboxes_apply_to_every_beam_tab(qapp):
    context = _fluence_context()
    sequence = Sequence(beams=[_make_beam()], fluence=np.array([4.0, 3.0, 2.0, 1.0]))

    viewer = ApertureViewerWidget()
    viewer.set_sequence(sequence, context=context)
    assert viewer._beam_widgets[0].show_mlc_borders
    assert viewer._beam_widgets[0].show_target

    viewer.mlc_borders_checkbox.setChecked(False)
    viewer.target_checkbox.setChecked(False)

    assert not viewer._beam_widgets[0].show_mlc_borders
    assert not viewer._beam_widgets[0].show_target


def test_leaf_coordinate_selector_applies_to_every_beam_tab(qapp):
    sequence = Sequence(beams=[_make_beam(), _make_beam()])

    viewer = ApertureViewerWidget()
    viewer.set_sequence(sequence)
    assert all(w.leaf_coordinate == "leaf_num" for w in viewer._beam_widgets)

    physical_ix = viewer.leaf_coordinate_selector.findData("physical")
    viewer.leaf_coordinate_selector.setCurrentIndex(physical_ix)

    assert all(w.leaf_coordinate == "physical" for w in viewer._beam_widgets)


def test_leaf_coordinate_selector_reflects_caller_chosen_coordinate(qapp):
    sequence = Sequence(beams=[_make_beam()])

    viewer = ApertureViewerWidget()
    viewer.set_sequence(sequence, leaf_coordinate="physical")

    assert viewer.leaf_coordinate_selector.currentData() == "physical"
    assert viewer._beam_widgets[0].leaf_coordinate == "physical"


def test_set_leaf_coordinate_keeps_displayed_shape(qapp):
    widget = BeamApertureWidget(_make_beam(n_shapes=3), "Beam 0")
    widget.shape_slider.setValue(2)

    widget.set_leaf_coordinate("physical")

    assert widget.shape_slider.value() == 2
    assert widget.shape_label.text() == "3 / 3"


def test_target_outline_maps_onto_the_leaf_pair_axis(qapp):
    context = _fluence_context()
    widget = BeamApertureWidget(
        _make_beam(),
        "Beam 0",
        beam_ix=0,
        context=context,
        delivered_fluence=np.array([4.0, 3.0, 2.0, 1.0]),
    )
    # Stand in for a projected target spanning two known leaf rows.
    mlc = widget.beam.segments[0].start.mlc
    boundaries = np.asarray(mlc.leaf_position_boundaries, dtype=float)
    widget._target_points = np.array([[0.0, boundaries[0]], [0.0, boundaries[1]]])

    widget.set_leaf_coordinate("physical")
    np.testing.assert_allclose(widget._target_points_for_axis(0)[:, 1], boundaries[:2])

    widget.set_leaf_coordinate("leaf_num")
    # Row n is drawn from n - 0.5 to n + 0.5 (_leaf_row_bounds): these are lower row edges
    np.testing.assert_allclose(widget._target_points_for_axis(0)[:, 1], [0.5, 1.5])


def test_target_color_is_configurable(qapp):
    context = _fluence_context()
    sequence = Sequence(beams=[_make_beam()], fluence=np.array([4.0, 3.0, 2.0, 1.0]))

    viewer = ApertureViewerWidget()
    viewer.set_sequence(sequence, context=context)
    assert viewer._beam_widgets[0].target_color == "red"

    viewer.target_color = "#00ff00"
    for widget in viewer._beam_widgets:
        widget.set_target_color(viewer.target_color)

    assert viewer._beam_widgets[0].target_color == "#00ff00"


def test_bixel_grid_checkbox_applies_to_every_beam_tab(qapp):
    context = _fluence_context()
    sequence = Sequence(beams=[_make_beam()], fluence=np.array([4.0, 3.0, 2.0, 1.0]))

    viewer = ApertureViewerWidget()
    viewer.set_sequence(sequence, context=context)
    assert not viewer._beam_widgets[0].show_bixel_grid

    viewer.bixel_grid_checkbox.setChecked(True)

    assert viewer._beam_widgets[0].show_bixel_grid


def test_sequencer_label_includes_the_level_count(qapp):
    sequence = Sequence(beams=[_make_beam()], sequencer="xia", num_of_levels=7)

    viewer = ApertureViewerWidget()
    viewer.set_sequence(sequence)

    assert viewer.sequencer_label.text() == "Sequencer: xia (7 levels)"


def test_sequencer_label_omits_levels_when_unknown(qapp):
    sequence = Sequence(beams=[_make_beam()], sequencer="xia")

    viewer = ApertureViewerWidget()
    viewer.set_sequence(sequence)

    assert viewer.sequencer_label.text() == "Sequencer: xia"


def test_view_options_reset_for_a_newly_shown_sequence(qapp):
    context = _fluence_context()
    sequence = Sequence(beams=[_make_beam()], fluence=np.array([4.0, 3.0, 2.0, 1.0]))

    viewer = ApertureViewerWidget()
    viewer.set_sequence(sequence, context=context)

    viewer.fluence_selector.setCurrentIndex(viewer.fluence_selector.findData("optimized"))
    viewer.bixel_grid_checkbox.setChecked(True)
    viewer.mlc_borders_checkbox.setChecked(False)
    viewer.target_color = "#00ff00"

    viewer.set_sequence(sequence, context=context)

    assert viewer.fluence_selector.currentData() == "delivered"
    assert not viewer.bixel_grid_checkbox.isChecked()
    assert viewer.mlc_borders_checkbox.isChecked()
    assert viewer.target_checkbox.isChecked()
    assert viewer.target_color == "red"
    assert viewer._beam_widgets[0].fluence_mode == "delivered"
    assert viewer._beam_widgets[0].target_color == "red"


def test_target_projection_runs_with_a_full_context(qapp, test_data_photons):
    ct, cst, _ = test_data_photons
    context = _fluence_context()
    context.ct = ct
    context.cst = cst

    widget = BeamApertureWidget(
        _make_beam(),
        "Beam 0",
        beam_ix=0,
        context=context,
        delivered_fluence=np.array([4.0, 3.0, 2.0, 1.0]),
    )

    assert widget._target_points is not None
    assert widget._target_points.shape[1] == 2
    assert widget._target_resolution == max(float(v) for v in ct.resolution.values())
