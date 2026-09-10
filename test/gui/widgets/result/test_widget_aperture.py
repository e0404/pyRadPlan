import numpy as np

from pyRadPlan.machines import create_bld
from pyRadPlan.result import Sequence
from pyRadPlan.result._beam_sequence import BeamSequence
from pyRadPlan.gui.widgets.result.aperture_widget import (
    ApertureViewerWidget,
    BeamApertureWidget,
)


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
    return BeamSequence(mlc_sequence=shapes, shapes_weight=weights, mlc_window=[-20, 20])


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
    sequence = Sequence(beam=[_make_beam(), _make_beam()])

    viewer = ApertureViewerWidget()
    viewer.set_sequence(sequence)

    assert viewer.tabs.count() == 2
    assert len(viewer._beam_widgets) == 2


def test_aperture_viewer_widget_rebuild_stops_previous_playback(qapp):
    sequence = Sequence(beam=[_make_beam()])

    viewer = ApertureViewerWidget()
    viewer.set_sequence(sequence)
    viewer._beam_widgets[0].play_button.setChecked(True)
    assert viewer._beam_widgets[0]._timer.isActive()

    viewer.set_sequence(sequence)
    assert viewer.tabs.count() == 1


def test_aperture_viewer_widget_shows_sequencer_label(qapp):
    sequence = Sequence(beam=[_make_beam()], sequencer="xia")

    viewer = ApertureViewerWidget()
    viewer.set_sequence(sequence)

    assert not viewer.sequencer_label.isHidden()
    assert viewer.sequencer_label.text() == "Sequencer: xia"


def test_aperture_viewer_widget_hides_sequencer_label_when_unknown(qapp):
    sequence = Sequence(beam=[_make_beam()])

    viewer = ApertureViewerWidget()
    viewer.set_sequence(sequence)

    assert viewer.sequencer_label.isHidden()
