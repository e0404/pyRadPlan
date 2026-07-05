"""Viewer options widget."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)


class ViewerOptionsWidget(QWidget):
    """Widget for viewer options (window/level, colormap, opacity)."""

    # Signals
    mode_changed = Signal(str)  # "Quantity" | "ct"
    colormap_changed = Signal(str, str)  # colormap name, mode
    window_level_changed = Signal(float, float, str)  # center, width, mode
    opacity_changed = Signal(float)
    reset_requested = Signal()
    local_range_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._updating_from_preset = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        self._build_colormap_row(layout)
        self._build_ct_window_controls(layout)
        self._build_range_controls(layout)
        self._build_opacity_reset(layout)
        layout.addStretch()

    def _build_colormap_row(self, layout: QVBoxLayout) -> None:
        """Add colormap mode toggle and colormap combo to layout."""
        cmap_row = QHBoxLayout()
        self.cmap_mode_btn = QPushButton("Quantity")
        self.cmap_mode_btn.setToolTip("Toggle between Quantity and CT mode")
        self.cmap_mode_btn.setCheckable(True)
        self.cmap_mode_btn.setChecked(True)
        self.cmap_mode_btn.clicked.connect(self._on_cmap_mode_toggled)
        cmap_row.addWidget(self.cmap_mode_btn)

        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems(
            [
                "jet",
                "viridis",
                "magma",
                "plasma",
                "inferno",
                "gray",
                "bone",
                "seismic",
                "coolwarm",
                "RdBu_r",
                "bwr",
            ]
        )
        self.cmap_combo.currentTextChanged.connect(self._on_cmap_changed)
        cmap_row.addWidget(self.cmap_combo)
        layout.addLayout(cmap_row)

    def _build_ct_window_controls(self, layout: QVBoxLayout) -> None:
        """Add CT preset combo and center/width spin+slider pairs to layout."""
        layout.addWidget(QLabel("CT Window:"))
        self.ct_preset_combo = QComboBox()
        self.ct_preset_combo.addItems(["Custom", "Abdomen", "Lung", "Bone", "Brain"])
        self.ct_preset_combo.currentTextChanged.connect(self._on_ct_preset_changed)
        layout.addWidget(self.ct_preset_combo)

        wc_row = QHBoxLayout()
        wc_row.addWidget(QLabel("Center:"))
        self.wc_spin = QDoubleSpinBox()
        self.wc_spin.setRange(-2000, 4000)
        self.wc_spin.setValue(40)
        self.wc_spin.valueChanged.connect(self._on_wc_spin_changed)
        wc_row.addWidget(self.wc_spin)
        layout.addLayout(wc_row)

        self.wc_slider = QSlider(Qt.Orientation.Horizontal)
        self.wc_slider.setRange(-2000, 4000)
        self.wc_slider.setValue(40)
        self.wc_slider.valueChanged.connect(self._on_wc_slider_changed)
        layout.addWidget(self.wc_slider)

        ww_row = QHBoxLayout()
        ww_row.addWidget(QLabel("Width:"))
        self.ww_spin = QDoubleSpinBox()
        self.ww_spin.setRange(0, 4000)
        self.ww_spin.setValue(400)
        self.ww_spin.valueChanged.connect(self._on_ww_spin_changed)
        ww_row.addWidget(self.ww_spin)
        layout.addLayout(ww_row)

        self.ww_slider = QSlider(Qt.Orientation.Horizontal)
        self.ww_slider.setRange(0, 4000)
        self.ww_slider.setValue(400)
        self.ww_slider.valueChanged.connect(self._on_ww_slider_changed)
        layout.addWidget(self.ww_slider)

    def _build_range_controls(self, layout: QVBoxLayout) -> None:
        """Add range min/max spinboxes and local-range button to layout."""
        range_row = QHBoxLayout()
        range_row.addWidget(QLabel("Range:"))
        self.range_min_spin = QDoubleSpinBox()
        self.range_min_spin.setRange(-2000, 10000)
        self.range_min_spin.setDecimals(2)
        self.range_min_spin.valueChanged.connect(self._on_range_min_changed)
        range_row.addWidget(self.range_min_spin)

        range_row.addWidget(QLabel("-"))

        self.range_max_spin = QDoubleSpinBox()
        self.range_max_spin.setRange(-2000, 10000)
        self.range_max_spin.setDecimals(2)
        self.range_max_spin.valueChanged.connect(self._on_range_max_changed)
        range_row.addWidget(self.range_max_spin)
        layout.addLayout(range_row)

        self.local_range_btn = QPushButton("Use local min/max")
        self.local_range_btn.setToolTip("Set range to min/max of current slice")
        self.local_range_btn.clicked.connect(self.local_range_requested)
        layout.addWidget(self.local_range_btn)

    def _build_opacity_reset(self, layout: QVBoxLayout) -> None:
        """Add opacity slider and reset button to layout."""
        layout.addWidget(QLabel("Overlay Opacity:"))
        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(40)
        self.opacity_slider.valueChanged.connect(self._on_opacity_changed)
        layout.addWidget(self.opacity_slider)

        self.reset_btn = QPushButton("Reset Options")
        self.reset_btn.setToolTip("Reset all viewer options to defaults")
        self.reset_btn.clicked.connect(self.reset_requested)
        layout.addWidget(self.reset_btn)

    def sync_ui(
        self,
        mode: str,
        data_range: tuple[float, float],
        window_level: tuple[float, float] | None,
        colormap: str,
    ) -> None:
        """Update UI controls to reflect settings for the given mode."""
        dmin, dmax = data_range

        self.wc_spin.blockSignals(True)
        self.ww_spin.blockSignals(True)
        self.wc_slider.blockSignals(True)
        self.ww_slider.blockSignals(True)
        self.cmap_combo.blockSignals(True)
        self.ct_preset_combo.blockSignals(True)
        self.range_min_spin.blockSignals(True)
        self.range_max_spin.blockSignals(True)

        if mode == "quantity":
            pad_min = min(0.0, dmin * 1.5) if dmin < 0 else 0.0
            pad_max = max(10.0, dmax * 1.5)
            self._apply_quantity_ranges(pad_min, pad_max)
        else:
            self._apply_ct_ranges()

        if window_level is not None:
            c, w = window_level
        elif mode == "ct":
            c, w = (dmax + dmin) * 0.5, dmax - dmin
        else:
            c, w = (dmax + dmin) * 0.5, dmax - dmin

        self.wc_spin.setValue(float(c))
        self.ww_spin.setValue(float(w))
        self.wc_slider.setValue(self._val_to_slider(c, mode))
        self.ww_slider.setValue(self._val_to_slider(w, mode))

        vmin = c - w / 2
        vmax = c + w / 2
        self.range_min_spin.setValue(vmin)
        self.range_max_spin.setValue(vmax)

        self.cmap_combo.setCurrentText(colormap)

        self.wc_spin.blockSignals(False)
        self.ww_spin.blockSignals(False)
        self.wc_slider.blockSignals(False)
        self.ww_slider.blockSignals(False)
        self.cmap_combo.blockSignals(False)
        self.ct_preset_combo.blockSignals(False)
        self.range_min_spin.blockSignals(False)
        self.range_max_spin.blockSignals(False)

    def _apply_quantity_ranges(self, limit_min: float, limit_max: float) -> None:
        """Configure spinbox/slider ranges for quantity mode."""
        width_max = limit_max - limit_min
        self.wc_slider.setRange(int(limit_min * 100), int(limit_max * 100))
        self.ww_slider.setRange(0, int(width_max * 100))
        self.wc_spin.setRange(limit_min, limit_max)
        self.ww_spin.setRange(0.0, width_max)
        self.wc_spin.setSingleStep(0.1)
        self.ww_spin.setSingleStep(0.1)
        self.range_min_spin.setRange(limit_min, limit_max)
        self.range_max_spin.setRange(limit_min, limit_max)
        self.range_min_spin.setSingleStep(0.1)
        self.range_max_spin.setSingleStep(0.1)
        self.ct_preset_combo.setEnabled(False)

    def _apply_ct_ranges(self) -> None:
        """Configure spinbox/slider ranges for CT mode."""
        self.wc_slider.setRange(-2000, 4000)
        self.ww_slider.setRange(0, 4000)
        self.wc_spin.setRange(-2000.0, 4000.0)
        self.ww_spin.setRange(0.0, 4000.0)
        self.wc_spin.setSingleStep(1.0)
        self.ww_spin.setSingleStep(1.0)
        self.range_min_spin.setRange(-2000.0, 4000.0)
        self.range_max_spin.setRange(-2000.0, 4000.0)
        self.range_min_spin.setSingleStep(1.0)
        self.range_max_spin.setSingleStep(1.0)
        self.ct_preset_combo.setEnabled(True)

    def set_range_values(self, vmin: float, vmax: float) -> None:
        """Set the range spinboxes directly."""
        self.range_min_spin.setValue(vmin)
        self.range_max_spin.setValue(vmax)

    def reset_ui(self) -> None:
        """Reset UI to default state."""
        self.cmap_mode_btn.setChecked(True)  # quantity
        self.cmap_mode_btn.setText("Quantity")
        self.opacity_slider.blockSignals(True)
        self.opacity_slider.setValue(40)
        self.opacity_slider.blockSignals(False)

    def _val_to_slider(self, value: float, mode: str) -> int:
        if mode == "quantity":
            return int(value * 100)
        return int(value)

    def _slider_to_val(self, value: int, mode: str) -> float:
        if mode == "quantity":
            return value / 100.0
        return float(value)

    def _on_cmap_mode_toggled(self, checked: bool) -> None:
        mode = "quantity" if checked else "ct"
        self.cmap_mode_btn.setText("Quantity" if checked else "CT")
        self.mode_changed.emit(mode)

    def _on_cmap_changed(self, name: str) -> None:
        mode = "quantity" if self.cmap_mode_btn.isChecked() else "ct"
        self.colormap_changed.emit(name, mode)

    def _on_ct_preset_changed(self, preset: str) -> None:
        presets = {
            "Abdomen": (40, 400),
            "Lung": (-600, 1500),
            "Bone": (400, 1800),
            "Brain": (40, 80),
            "Custom": None,
        }
        val = presets.get(preset)
        if val:
            c, w = val
            self._updating_from_preset = True
            self.wc_spin.setValue(float(c))
            self.ww_spin.setValue(float(w))
            self._updating_from_preset = False

    def _on_wc_spin_changed(self, value: float) -> None:
        mode = "quantity" if self.cmap_mode_btn.isChecked() else "ct"
        self.wc_slider.blockSignals(True)
        self.wc_slider.setValue(self._val_to_slider(value, mode))
        self.wc_slider.blockSignals(False)
        self._emit_window_changed()

    def _on_ww_spin_changed(self, value: float) -> None:
        mode = "quantity" if self.cmap_mode_btn.isChecked() else "ct"
        self.ww_slider.blockSignals(True)
        self.ww_slider.setValue(self._val_to_slider(value, mode))
        self.ww_slider.blockSignals(False)
        self._emit_window_changed()

    def _on_wc_slider_changed(self, value: int) -> None:
        mode = "quantity" if self.cmap_mode_btn.isChecked() else "ct"
        val = self._slider_to_val(value, mode)
        self.wc_spin.blockSignals(True)
        self.wc_spin.setValue(val)
        self.wc_spin.blockSignals(False)
        self._emit_window_changed()

    def _on_ww_slider_changed(self, value: int) -> None:
        mode = "quantity" if self.cmap_mode_btn.isChecked() else "ct"
        val = self._slider_to_val(value, mode)
        self.ww_spin.blockSignals(True)
        self.ww_spin.setValue(val)
        self.ww_spin.blockSignals(False)
        self._emit_window_changed()

    def _on_range_min_changed(self, val: float) -> None:
        """Update Center/Width when Min changes."""
        vmax = self.range_max_spin.value()
        val = min(val, vmax)  # clamp

        width = vmax - val
        center = val + width / 2

        self.wc_spin.setValue(center)
        self.ww_spin.setValue(width)

    def _on_range_max_changed(self, val: float) -> None:
        """Update Center/Width when Max changes."""
        vmin = self.range_min_spin.value()
        val = max(val, vmin)

        width = val - vmin
        center = vmin + width / 2

        self.wc_spin.setValue(center)
        self.ww_spin.setValue(width)

    def _on_opacity_changed(self, value: int) -> None:
        self.opacity_changed.emit(value / 100.0)

    def _emit_window_changed(self) -> None:
        c = self.wc_spin.value()
        w = self.ww_spin.value()
        mode = "quantity" if self.cmap_mode_btn.isChecked() else "ct"

        vmin = c - w / 2
        vmax = c + w / 2
        self.range_min_spin.blockSignals(True)
        self.range_max_spin.blockSignals(True)
        self.range_min_spin.setValue(vmin)
        self.range_max_spin.setValue(vmax)
        self.range_min_spin.blockSignals(False)
        self.range_max_spin.blockSignals(False)

        if (
            mode == "ct"
            and not self._updating_from_preset
            and self.ct_preset_combo.currentText() != "Custom"
        ):
            self.ct_preset_combo.blockSignals(True)
            self.ct_preset_combo.setCurrentText("Custom")
            self.ct_preset_combo.blockSignals(False)

        self.window_level_changed.emit(c, w, mode)
