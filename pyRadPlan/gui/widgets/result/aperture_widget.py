"""MLC aperture shape viewer widget, with a per-beam animate/interactive player."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
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

from pyRadPlan.ct import CT
from pyRadPlan.cst import StructureSet
from pyRadPlan.result import PlanResult, Sequence
from pyRadPlan.result._beam_sequence import BeamSequence
from pyRadPlan.stf import SteeringInformation
from pyRadPlan.visualization._plot_aperture_info import (
    _beam_label,
    _draw_aperture,
    _draw_bixel_grid,
    _draw_fluence_axis,
    _draw_fluence_diff_axis,
    _draw_mlc_borders,
    _draw_target_outline,
    _leaf_num_points,
    _native_fluence_map,
    _target_bev_points,
    _weight_color,
)

# Dwell time for a full-weight shape when playing; other shapes are scaled by
# their weight relative to the beam's heaviest shape (a proxy for timing when
# no VMAT control-point timing is available).
_BASE_INTERVAL_MS = 900
_MIN_INTERVAL_MS = 150

logger = logging.getLogger(__name__)


@dataclass
class SequenceViewContext:
    """Data the sequence viewer needs beyond the ``Sequence`` itself."""

    stf: Optional[SteeringInformation] = None
    result: Optional[PlanResult] = None
    ct: Optional[CT] = None
    cst: Optional[StructureSet] = None


class BeamApertureWidget(QWidget):
    """Single-beam aperture player: one shape at a time, with playback controls."""

    def __init__(
        self,
        beam: BeamSequence,
        label: str,
        leaf_coordinate: str = "leaf_num",
        parent: Optional[QWidget] = None,
        *,
        beam_ix: int = 0,
        context: Optional[SequenceViewContext] = None,
        delivered_fluence: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__(parent)
        self.beam = beam
        self.label = label
        self.leaf_coordinate = leaf_coordinate
        self.beam_ix = beam_ix
        self.context = context
        self.fluence_mode = "delivered"
        self.show_mlc_borders = True
        self.show_target = True
        self.show_bixel_grid = False
        self.target_color = "red"
        self._n_shapes = beam.num_of_shapes
        self._w_max = float(np.max(beam.shapes_weight)) if self._n_shapes else 0.0

        # Computed once: neither the fluence maps nor the target projection depend
        # on which shape is displayed, and the player redraws on every step.
        self._fluence_grids, self._fluence_window = self._build_fluence_grids(delivered_fluence)
        self._target_resolution = 3.0
        self._target_points = self._build_target_points()

        self.figure = Figure(
            figsize=(9, 4) if self.has_fluence else (4, 4), dpi=100, layout="constrained"
        )
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

    def _beam_geometry(self):
        """The steering-information beam for this tab, if the context supplies one."""
        stf = getattr(self.context, "stf", None)
        if stf is None or self.beam_ix >= len(stf.beams):
            return None
        return stf.beams[self.beam_ix]

    def _build_fluence_grids(
        self, delivered_fluence: Optional[np.ndarray]
    ) -> tuple[dict, Optional[list]]:
        """Scatter this beam's optimized/delivered fluence onto a dense BEV grid."""
        beam_geometry = self._beam_geometry()
        if beam_geometry is None or self.beam.bixel_ix is None or not self.beam.mlc_sequence:
            return {}, None

        device_orientation = self.beam.mlc_sequence[0].device_orientation
        grids: dict = {}
        window = None

        result = getattr(self.context, "result", None)
        optimized_fluence = getattr(result, "fluence", None)
        for mode, values in (("optimized", optimized_fluence), ("delivered", delivered_fluence)):
            if values is None:
                continue
            grids[mode], window = _native_fluence_map(
                beam_geometry, np.asarray(values)[self.beam.bixel_ix], device_orientation
            )

        if "optimized" in grids and "delivered" in grids:
            grids["difference"] = grids["delivered"] - grids["optimized"]

        return grids, window

    def _build_target_points(self) -> Optional[np.ndarray]:
        """Project the target into this beam's BEV plane, if the context allows."""
        beam_geometry = self._beam_geometry()
        ct = getattr(self.context, "ct", None)
        cst = getattr(self.context, "cst", None)
        if beam_geometry is None or ct is None or cst is None or not self.beam.mlc_sequence:
            return None
        try:
            self._target_resolution = max(float(v) for v in ct.resolution.values())
            return _target_bev_points(
                ct, cst, beam_geometry, self.beam.mlc_sequence[0].device_orientation
            )
        except Exception:  # noqa: BLE001 - an un-projectable target must not break the viewer
            logger.debug("Could not project the target into BEV", exc_info=True)
            return None

    @property
    def has_fluence(self) -> bool:
        """Whether a fluence panel can be drawn alongside the aperture."""
        return bool(self._fluence_grids)

    def stop(self) -> None:
        """Stop playback (called when the parent viewer is torn down/rebuilt)."""
        self._timer.stop()
        self.play_button.setChecked(False)

    def set_fluence_mode(self, mode: str) -> None:
        """Show the optimized, delivered or difference fluence in the right panel."""
        if mode == self.fluence_mode:
            return
        self.fluence_mode = mode
        self._draw_shape(self.shape_slider.value())

    def set_overlays(self, mlc_borders: bool, target: bool, bixel_grid: bool = False) -> None:
        """Toggle the MLC-boundary, projected-target and bixel-grid overlays."""
        current = (self.show_mlc_borders, self.show_target, self.show_bixel_grid)
        if (mlc_borders, target, bixel_grid) == current:
            return
        self.show_mlc_borders = mlc_borders
        self.show_target = target
        self.show_bixel_grid = bixel_grid
        self._draw_shape(self.shape_slider.value())

    def set_target_color(self, color: str) -> None:
        """Set the colour the projected-target outline is drawn in."""
        if color == self.target_color:
            return
        self.target_color = color
        self._draw_shape(self.shape_slider.value())

    def _target_points_for_axis(self, index: int) -> Optional[np.ndarray]:
        """The projected target on whichever vertical axis is currently shown."""
        if self._target_points is None:
            return None
        if self.leaf_coordinate == "physical":
            return self._target_points
        return _leaf_num_points(self._target_points, self.beam.mlc_sequence[index])

    def _target_bin_size(self, index: int):
        """Outline bin size for the current axes."""
        if self.leaf_coordinate == "physical":
            return self._target_resolution
        leaf_width = getattr(self.beam.mlc_sequence[index], "leaf_width", None)
        if not leaf_width:
            return self._target_resolution
        return (self._target_resolution, self._target_resolution / leaf_width)

    def _bixel_grid_vertical(self, index: int) -> Optional[np.ndarray]:
        """Bixel-boundary positions for the aperture panel's vertical axis."""
        if self._fluence_window is None:
            return None
        beam_geometry = self._beam_geometry()
        bixel_width = getattr(beam_geometry, "bixel_width", None)
        if not bixel_width:
            return None

        lines = np.arange(
            self._fluence_window[2], self._fluence_window[3] + 0.5 * bixel_width, bixel_width
        )
        if self.leaf_coordinate == "physical":
            return lines
        return _leaf_num_points(
            np.column_stack((np.zeros_like(lines), lines)), self.beam.mlc_sequence[index]
        )[:, 1]

    def set_leaf_coordinate(self, leaf_coordinate: str) -> None:
        """Switch the stacking axis between leaf-pair index and physical mm."""
        if leaf_coordinate == self.leaf_coordinate:
            return
        self.leaf_coordinate = leaf_coordinate
        self._draw_shape(self.shape_slider.value())

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

        n_panels = 2 if self.has_fluence else 1
        ax = self.figure.add_subplot(1, n_panels, 1)
        weight = float(self.beam.shapes_weight[index])
        ax.set_facecolor(_weight_color(weight, self._w_max))
        _draw_aperture(
            ax, self.beam.mlc_sequence[index], self.beam.mlc_window, self.leaf_coordinate
        )
        if self.show_bixel_grid and self._fluence_window is not None:
            _draw_bixel_grid(
                ax,
                self._fluence_window,
                getattr(self._beam_geometry(), "bixel_width", 0.0),
                vertical=self._bixel_grid_vertical(index),
                color="0.6",
            )
        if self.show_target:
            _draw_target_outline(
                ax,
                self._target_points_for_axis(index),
                self._target_bin_size(index),
                self.target_color,
            )
        ax.set_title(f"{self.label} — shape {index + 1}/{self._n_shapes}, w={weight:.2g}")

        if self.has_fluence:
            self._draw_fluence_panel(self.figure.add_subplot(1, n_panels, 2), index)

        self.shape_label.setText(f"{index + 1} / {self._n_shapes}")
        self.canvas.draw()

    def _draw_fluence_panel(self, ax, index: int) -> None:
        """Draw the selected fluence for this beam, with the requested overlays."""
        grid = self._fluence_grids.get(self.fluence_mode)
        if grid is None:
            ax.set_axis_off()
            ax.text(0.5, 0.5, f"No {self.fluence_mode} fluence", ha="center", va="center")
            return

        finite = grid[np.isfinite(grid)]
        if self.fluence_mode == "difference":
            vmax_abs = float(np.abs(finite).max()) if finite.size else 0.0
            _draw_fluence_diff_axis(
                ax,
                self.figure,
                grid,
                self._fluence_window,
                vmax_abs or 1.0,
                "Difference fluence (delivered − optimized)",
            )
        else:
            vmax = float(finite.max()) if finite.size else 0.0
            _draw_fluence_axis(
                ax,
                self.figure,
                grid,
                self._fluence_window,
                vmax or 1.0,
                f"{self.fluence_mode.capitalize()} fluence",
            )

        if self.show_bixel_grid:
            _draw_bixel_grid(
                ax,
                self._fluence_window,
                getattr(self._beam_geometry(), "bixel_width", 0.0),
            )
        if self.show_mlc_borders:
            _draw_mlc_borders(ax, self.beam.mlc_sequence[index], self.beam.mlc_window)
        if self.show_target:
            _draw_target_outline(
                ax, self._target_points, self._target_resolution, self.target_color
            )


class ApertureViewerWidget(QWidget):
    """Tabbed MLC aperture shape viewer: one tab per beam.

    Mirrors matRad's ``matRad_visApertureInfo`` "perBeam" view (a shape grid
    per beam) combined with its "animate"/"interactive" player (a single
    aperture stepped through over time). pyRadPlan does not yet support VMAT;
    the dwell-time-by-weight proxy below stands in until VMAT sequencing (and
    its control-point timing) is implemented.
    """

    #: Stacking-axis choices: (combo label, ``leaf_coordinate`` value).
    _LEAF_COORDINATES = (("Leaf pair #", "leaf_num"), ("Physical [mm]", "physical"))

    #: Fluence-panel choices: (combo label, mode).
    _FLUENCE_MODES = (
        ("Delivered", "delivered"),
        ("Optimized", "optimized"),
        ("Difference", "difference"),
    )

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.sequencer_label = QLabel()
        self.sequencer_label.hide()

        self.leaf_coordinate_selector = QComboBox()
        for label, value in self._LEAF_COORDINATES:
            self.leaf_coordinate_selector.addItem(label, value)
        self.leaf_coordinate_selector.currentIndexChanged.connect(self._on_leaf_coordinate_changed)

        self.fluence_selector = QComboBox()
        for label, value in self._FLUENCE_MODES:
            self.fluence_selector.addItem(label, value)
        self.fluence_selector.currentIndexChanged.connect(self._on_fluence_mode_changed)

        self.mlc_borders_checkbox = QCheckBox("MLC boundaries")
        self.mlc_borders_checkbox.setChecked(True)
        self.mlc_borders_checkbox.toggled.connect(self._on_overlays_toggled)

        self.target_checkbox = QCheckBox("Target (BEV)")
        self.target_checkbox.setChecked(True)
        self.target_checkbox.setToolTip("Outline of the target projected into the beam's-eye view")
        self.target_checkbox.toggled.connect(self._on_overlays_toggled)

        self.target_color = "red"
        self.target_color_button = QPushButton()
        self.target_color_button.setFixedSize(12, 12)
        self.target_color_button.setFlat(True)
        self.target_color_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.target_color_button.setToolTip("Colour of the projected-target outline")
        self.target_color_button.clicked.connect(self._on_pick_target_color)
        self._update_target_color_button()

        self.bixel_grid_checkbox = QCheckBox("Bixel grid")
        self.bixel_grid_checkbox.setToolTip("Overlay the ray/bixel grid the fluence is sampled on")
        self.bixel_grid_checkbox.toggled.connect(self._on_overlays_toggled)

        header = QHBoxLayout()
        header.addWidget(self.sequencer_label)
        header.addStretch(1)
        header.addWidget(QLabel("Stacking axis:"))
        header.addWidget(self.leaf_coordinate_selector)

        self.overlay_bar = QWidget()
        overlays = QHBoxLayout(self.overlay_bar)
        overlays.setContentsMargins(0, 0, 0, 0)
        overlays.addWidget(QLabel("Fluence:"))
        overlays.addWidget(self.fluence_selector)
        overlays.addWidget(self.mlc_borders_checkbox)
        overlays.addWidget(self.target_checkbox)
        overlays.addWidget(self.target_color_button)
        overlays.addWidget(self.bixel_grid_checkbox)
        overlays.addStretch(1)

        self.tabs = QTabWidget()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(header)
        layout.addWidget(self.overlay_bar)
        layout.addWidget(self.tabs)
        self._beam_widgets: list[BeamApertureWidget] = []
        self.overlay_bar.hide()

    def set_sequence(
        self,
        sequence: Sequence,
        stf: Optional[SteeringInformation] = None,
        leaf_coordinate: str = "leaf_num",
        context: Optional[SequenceViewContext] = None,
    ) -> None:
        """Populate one tab per beam from a sequenced ``Sequence``.

        Passing a *context* additionally enables the fluence panel and the
        projected-target overlay; without one only the aperture shapes are shown.
        """
        if context is not None and stf is None:
            stf = context.stf
        for widget in self._beam_widgets:
            widget.stop()
        self.tabs.clear()
        self._beam_widgets = []

        if sequence.sequencer:
            levels = sequence.num_of_levels
            suffix = f" ({levels} levels)" if levels else ""
            self.sequencer_label.setText(f"Sequencer: {sequence.sequencer}{suffix}")
            self.sequencer_label.show()
        else:
            self.sequencer_label.hide()

        index = self.leaf_coordinate_selector.findData(leaf_coordinate)
        if index >= 0:
            self.leaf_coordinate_selector.blockSignals(True)
            self.leaf_coordinate_selector.setCurrentIndex(index)
            self.leaf_coordinate_selector.blockSignals(False)

        self._reset_view_options()

        delivered = sequence.fluence
        for beam_ix, beam in enumerate(sequence.beam):
            label = _beam_label(beam_ix, stf)
            widget = BeamApertureWidget(
                beam,
                label,
                leaf_coordinate,
                self,
                beam_ix=beam_ix,
                context=context,
                delivered_fluence=delivered,
            )
            self._beam_widgets.append(widget)
            self.tabs.addTab(widget, label)

        self.overlay_bar.setVisible(any(w.has_fluence for w in self._beam_widgets))
        self._on_fluence_mode_changed()
        self._on_overlays_toggled()
        for widget in self._beam_widgets:
            widget.set_target_color(self.target_color)

    def _reset_view_options(self) -> None:
        """Restore the overlay controls to their defaults for a newly shown sequence."""
        for widget, value in (
            (self.mlc_borders_checkbox, True),
            (self.target_checkbox, True),
            (self.bixel_grid_checkbox, False),
        ):
            widget.blockSignals(True)
            widget.setChecked(value)
            widget.blockSignals(False)

        delivered = self.fluence_selector.findData("delivered")
        if delivered >= 0:
            self.fluence_selector.blockSignals(True)
            self.fluence_selector.setCurrentIndex(delivered)
            self.fluence_selector.blockSignals(False)

        self.target_color = "red"
        self._update_target_color_button()

    def _on_leaf_coordinate_changed(self) -> None:
        leaf_coordinate = self.leaf_coordinate_selector.currentData()
        for widget in self._beam_widgets:
            widget.set_leaf_coordinate(leaf_coordinate)

    def _on_fluence_mode_changed(self) -> None:
        mode = self.fluence_selector.currentData()
        for widget in self._beam_widgets:
            widget.set_fluence_mode(mode)

    def _on_overlays_toggled(self) -> None:
        mlc_borders = self.mlc_borders_checkbox.isChecked()
        target = self.target_checkbox.isChecked()
        bixel_grid = self.bixel_grid_checkbox.isChecked()
        for widget in self._beam_widgets:
            widget.set_overlays(mlc_borders, target, bixel_grid)

    def _update_target_color_button(self) -> None:
        self.target_color_button.setStyleSheet(
            f"background-color: {self.target_color}; border: 1px solid #444;"
        )

    def _on_pick_target_color(self) -> None:
        color = QColorDialog.getColor(
            QColor(self.target_color), self, "Pick the target outline colour"
        )
        if not color.isValid():
            return
        self.target_color = color.name()
        self._update_target_color_button()
        for widget in self._beam_widgets:
            widget.set_target_color(self.target_color)

    def set_current_beam(self, beam_ix: int) -> None:
        """Switch to the tab for ``beam_ix``, if it exists."""
        if 0 <= beam_ix < self.tabs.count():
            self.tabs.setCurrentIndex(beam_ix)
