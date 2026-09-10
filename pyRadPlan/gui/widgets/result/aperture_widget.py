"""MLC aperture shape viewer widget, with a per-beam animate/interactive player."""

from __future__ import annotations

from typing import Optional

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from pyRadPlan.result import Sequence
from pyRadPlan.result._beam_sequence import BeamSequence
from pyRadPlan.stf import SteeringInformation
from pyRadPlan.visualization._plot_aperture_info import (
    _beam_label,
    _draw_aperture,
    _weight_color,
)

# Dwell time for a full-weight shape when playing; other shapes are scaled by
# their weight relative to the beam's heaviest shape (a proxy for timing when
# no VMAT control-point timing is available).
_BASE_INTERVAL_MS = 900
_MIN_INTERVAL_MS = 150


class BeamApertureWidget(QWidget):
    """Single-beam aperture player: one shape at a time, with playback controls."""

    def __init__(
        self,
        beam: BeamSequence,
        label: str,
        leaf_coordinate: str = "leaf_num",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.beam = beam
        self.label = label
        self.leaf_coordinate = leaf_coordinate
        self._n_shapes = beam.num_of_shapes
        self._w_max = float(np.max(beam.shapes_weight)) if self._n_shapes else 0.0

        self.figure = Figure(figsize=(4, 4), dpi=100, layout="constrained")
        self.canvas = FigureCanvas(self.figure)

        self.shape_slider = QSlider(Qt.Orientation.Horizontal)
        self.shape_slider.setMinimum(0)
        self.shape_slider.setMaximum(max(self._n_shapes - 1, 0))
        self.shape_slider.setEnabled(self._n_shapes > 1)
        self.shape_slider.valueChanged.connect(self._on_slider_changed)

        self.shape_label = QLabel()

        self.play_button = QPushButton("Play")
        self.play_button.setCheckable(True)
        self.play_button.setEnabled(self._n_shapes > 1)
        self.play_button.toggled.connect(self._on_play_toggled)

        self.speed_spin = QDoubleSpinBox()
        self.speed_spin.setRange(0.1, 10.0)
        self.speed_spin.setSingleStep(0.1)
        self.speed_spin.setValue(1.0)
        self.speed_spin.setPrefix("Speed: ")
        self.speed_spin.setSuffix("x")

        self.loop_checkbox = QCheckBox("Loop")
        self.loop_checkbox.setChecked(True)

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._advance)

        controls = QHBoxLayout()
        controls.addWidget(self.play_button)
        controls.addWidget(self.shape_slider, 1)
        controls.addWidget(self.shape_label)

        options = QHBoxLayout()
        options.addWidget(self.speed_spin)
        options.addWidget(self.loop_checkbox)
        options.addStretch(1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas, 1)
        layout.addLayout(controls)
        layout.addLayout(options)

        self._draw_shape(0)

    def stop(self) -> None:
        """Stop playback (called when the parent viewer is torn down/rebuilt)."""
        self._timer.stop()
        self.play_button.setChecked(False)

    def _on_play_toggled(self, checked: bool) -> None:
        self.play_button.setText("Pause" if checked else "Play")
        if checked:
            self._schedule_next()
        else:
            self._timer.stop()

    def _schedule_next(self) -> None:
        j = self.shape_slider.value()
        weight = float(self.beam.shapes_weight[j]) if self._n_shapes else 0.0
        frac = (weight / self._w_max) if self._w_max > 0 else 1.0
        interval = max(int(_BASE_INTERVAL_MS * frac / self.speed_spin.value()), _MIN_INTERVAL_MS)
        self._timer.start(interval)

    def _advance(self) -> None:
        if not self.play_button.isChecked():
            return
        j = self.shape_slider.value()
        if j + 1 < self._n_shapes:
            self.shape_slider.setValue(j + 1)
        elif self.loop_checkbox.isChecked():
            self.shape_slider.setValue(0)
        else:
            self.play_button.setChecked(False)
            return
        self._schedule_next()

    def _on_slider_changed(self, value: int) -> None:
        self._draw_shape(value)

    def _draw_shape(self, index: int) -> None:
        self.figure.clear()
        if self._n_shapes == 0:
            self.canvas.draw()
            return
        ax = self.figure.add_subplot(111)
        weight = float(self.beam.shapes_weight[index])
        ax.set_facecolor(_weight_color(weight, self._w_max))
        _draw_aperture(
            ax, self.beam.mlc_sequence[index], self.beam.mlc_window, self.leaf_coordinate
        )
        ax.set_title(f"{self.label} — shape {index + 1}/{self._n_shapes}, w={weight:.2g}")
        self.shape_label.setText(f"{index + 1} / {self._n_shapes}")
        self.canvas.draw()


class ApertureViewerWidget(QWidget):
    """Tabbed MLC aperture shape viewer: one tab per beam.

    Mirrors matRad's ``matRad_visApertureInfo`` "perBeam" view (a shape grid
    per beam) combined with its "animate"/"interactive" player (a single
    aperture stepped through over time). pyRadPlan does not yet support VMAT;
    the dwell-time-by-weight proxy below stands in until VMAT sequencing (and
    its control-point timing) is implemented.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.sequencer_label = QLabel()
        self.sequencer_label.hide()
        self.tabs = QTabWidget()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.sequencer_label)
        layout.addWidget(self.tabs)
        self._beam_widgets: list[BeamApertureWidget] = []

    def set_sequence(
        self,
        sequence: Sequence,
        stf: Optional[SteeringInformation] = None,
        leaf_coordinate: str = "leaf_num",
    ) -> None:
        """Populate one tab per beam from a sequenced ``Sequence``."""
        for widget in self._beam_widgets:
            widget.stop()
        self.tabs.clear()
        self._beam_widgets = []

        if sequence.sequencer:
            self.sequencer_label.setText(f"Sequencer: {sequence.sequencer}")
            self.sequencer_label.show()
        else:
            self.sequencer_label.hide()

        for beam_ix, beam in enumerate(sequence.beam):
            label = _beam_label(beam_ix, stf)
            widget = BeamApertureWidget(beam, label, leaf_coordinate, self)
            self._beam_widgets.append(widget)
            self.tabs.addTab(widget, label)

    def set_current_beam(self, beam_ix: int) -> None:
        """Switch to the tab for ``beam_ix``, if it exists."""
        if 0 <= beam_ix < self.tabs.count():
            self.tabs.setCurrentIndex(beam_ix)
