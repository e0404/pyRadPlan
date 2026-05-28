"""Slice quantity viewing widget using PySide6."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import SimpleITK as sitk
import pyqtgraph as pg
from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
    QComboBox,
)


class _SliceViewBox(pg.ViewBox):
    """ViewBox that scrolls slices on wheel and zooms on Ctrl+wheel."""

    slice_scroll = Signal(int)  # emits +1 or -1

    def wheelEvent(self, ev, axis=None):  # noqa: N802
        if ev.modifiers() & Qt.KeyboardModifier.ControlModifier:
            super().wheelEvent(ev, axis)
        else:
            delta = 1 if ev.delta() > 0 else -1
            self.slice_scroll.emit(delta)
            ev.accept()


class QuantityWidget(QWidget):
    """Interactive slice viewer for volumetric quantities.

    Public API:
    -----------
    set_data(ct_volume, quantity_volume=None, overlay_unit="Gy")
        Provide base CT (or scalar image) and optional quantity volume.
    set_plane(plane: str)
        Change anatomical plane ("Axial"|"Sagittal"|"Coronal"). Resets slider range.
    set_slice(index: int)
        Update the currently displayed slice.
    connect_viewer_signals(viewer)
        Convenience to wire up signals from an instance of ViewerWidget.

    Notes
    -----
    For now VOI highlighting is a placeholder; selected VOIs are listed but not
    rendered until mask data integration is added.
    """

    slice_changed = Signal(int)  # emitted when user moves the slider
    range_changed = Signal(float, float)  # emitted when slice changes, sending min/max

    _PLANE_MAP = {
        "Axial": 0,  # z
        "Sagittal": 1,  # x
        "Coronal": 2,  # y
        "axial": 0,
        "sagittal": 1,
        "coronal": 2,
    }

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._init_state()
        self._setup_ui()

    def _init_state(self) -> None:
        """Initialise all instance-level state attributes."""
        self._ct: Optional[np.ndarray] = None
        self._quantities: dict[str, np.ndarray] = {}
        self._active_quantity_name: Optional[str] = None
        self._overlay_unit: str | dict[str, str] = ""
        self._overlay_units: dict[str, str] = {}
        self._overlay_units_default: str = ""
        self._overlay_labels: dict[str, str] = {}
        self._plane: str = "Axial"
        self._global_max: bool = True
        self._isolines: bool = False
        self._show_ct: bool = True
        self._show_quantity: bool = True
        self._selected_vois: Sequence[str] = []
        self._masks: dict[str, np.ndarray] = {}
        self._voi_colors: dict[str, tuple[int, int, int]] = {}
        self._show_cst: bool = True
        self._last_cmap = None
        self._colorbar_mode: str | None = None  # 'quantity' or 'ct'

        self._isocenter: Optional[np.ndarray] = None
        self._show_isocenter: bool = False
        self._isoline_levels: list[float] = []
        self._quantity_opacity: float = 0.4
        self._ct_window: Optional[tuple[float, float]] = None
        self._quantity_window: Optional[tuple[float, float]] = None
        self._quantity_colormap: str = "jet"
        self._ct_colormap: str = "grey"
        self._active_mode: str = "quantity"

        self._image_item = None
        self._quantity_item = None
        self._iso_items: list = []
        self._isocenter_item = None

    def _setup_ui(self) -> None:
        """Build and wire all UI widgets."""
        layout = QVBoxLayout(self)
        self.setLayout(layout)

        self._plot_widget = pg.GraphicsLayoutWidget()
        self._view_box = _SliceViewBox()
        self._plot_widget.addItem(self._view_box, row=0, col=0)
        self._view_box.setAspectLocked(True)
        self._view_box.invertY(True)
        self._view_box.slice_scroll.connect(self._on_scroll_slice)
        self._plot_widget.viewport().installEventFilter(self)
        layout.addWidget(self._plot_widget)
        self._colorbar = None  # created lazily with quantity

        slider_row = QHBoxLayout()

        self.plane_combobox = QComboBox()
        self.plane_combobox.addItems(["Axial", "Sagittal", "Coronal"])
        self.plane_combobox.currentTextChanged.connect(self.set_plane)
        slider_row.addWidget(self.plane_combobox)

        self.slice_slider = QSlider(Qt.Orientation.Horizontal)
        self.slice_slider.setEnabled(False)
        self.slice_slider.valueChanged.connect(self._on_slider_changed)

        self.slice_spin = QSpinBox()
        self.slice_spin.setEnabled(False)
        self.slice_spin.setMinimum(0)
        self.slice_spin.valueChanged.connect(self._on_spin_changed)

        self._slice_label = QLabel("Slice: -")
        slider_row.addWidget(self.slice_slider, 1)
        slider_row.addWidget(self.slice_spin, 0)
        slider_row.addWidget(self._slice_label, 0)
        layout.addLayout(slider_row)

    # ------------------------------------------------------------------
    # Data & plane management
    # ------------------------------------------------------------------
    def set_data(
        self,
        ct_volume: np.ndarray,
        quantity_volume: Optional[np.ndarray | dict[str, np.ndarray | list[np.ndarray]]] = None,
        overlay_unit: str | dict[str, str] = "Gy",
        isocenter: Optional[np.ndarray] = None,
        overlay_label: dict[str, str] | None = None,
    ) -> None:
        """Assign imaging data.

        Parameters
        ----------
        ct_volume : ndarray
            Base CT or scalar image as 3D array (X,Y,Z) ordering assumed.
        quantity_volume : ndarray or dict, optional
            quantity volume matching ct_volume shape. Can be a single array or a dict of arrays.
            Values can also be lists of arrays (e.g. beams).
        overlay_unit : str
            Unit label for quantity overlay.
        isocenter : ndarray, optional
            Isocenter coordinates in voxel space (z, x, y) matching the viewer's slice orientation.
        """
        self._ct = ct_volume.astype(float, copy=False)
        self._quantities = {}
        self._active_quantity_name = None

        if quantity_volume is not None:
            if isinstance(quantity_volume, dict):
                self._quantities = self._process_dict_quantity(quantity_volume)
                if self._quantities:
                    if "Physical quantity" in self._quantities:
                        self._active_quantity_name = "Physical quantity"
                    elif "physical_quantity" in self._quantities:
                        self._active_quantity_name = "physical_quantity"
                    else:
                        self._active_quantity_name = next(iter(self._quantities))
            else:
                self._quantities["Physical quantity"] = quantity_volume.astype(float)
                self._active_quantity_name = "Physical quantity"

        if isinstance(overlay_unit, dict):
            self._overlay_units = overlay_unit
        else:
            self._overlay_units = {}
            self._overlay_units_default = overlay_unit
        self._overlay_unit = overlay_unit
        self._overlay_labels = overlay_label or {}
        self._isocenter = isocenter

        self._quantity_window = None

        self._configure_slider()
        self.update_slice()

    def _process_dict_quantity(self, quantity_dict: dict) -> dict[str, np.ndarray]:
        """Process a dict of quantity volumes into a flat ``{name: array}`` mapping."""
        result: dict[str, np.ndarray] = {}
        for k, v in quantity_dict.items():
            if isinstance(v, list):
                for i, beam_vol in enumerate(v):
                    try:
                        arr = beam_vol.astype(float)
                    except AttributeError:
                        if isinstance(beam_vol, sitk.Image):
                            arr = sitk.GetArrayFromImage(beam_vol).transpose(2, 1, 0).astype(float)
                        else:
                            continue
                    result[f"{k} {i}"] = arr
            else:
                try:
                    arr = v.astype(float)
                except AttributeError:
                    if isinstance(v, sitk.Image):
                        arr = sitk.GetArrayFromImage(v).transpose(2, 1, 0).astype(float)
                    else:
                        continue
                result[k] = arr
        return result

    def _get_active_unit(self) -> str:
        """Return the unit string for the currently active quantity."""
        if self._overlay_units and self._active_quantity_name:
            return self._overlay_units.get(
                self._active_quantity_name,
                getattr(self, "_overlay_units_default", ""),
            )
        if isinstance(self._overlay_unit, str):
            return self._overlay_unit
        return ""

    def _get_active_label(self) -> str:
        """Return the display label for the currently active quantity."""
        if self._overlay_labels and self._active_quantity_name:
            return self._overlay_labels.get(self._active_quantity_name, "Quantity")
        return "Quantity"

    def get_active_overlay_unit(self) -> str:
        """Return the unit string for the currently active quantity."""
        return self._get_active_unit()

    def get_active_overlay_label(self) -> str:
        """Return the display label for the currently active quantity."""
        return self._get_active_label()

    def set_masks(self, masks: dict[str, np.ndarray]) -> None:
        """Assign VOI masks for contour display.

        Each mask must match CT shape (X,Y,Z) and be boolean or 0/1.
        Displayed as contours with distinct colors (similar to matplotlib 'cool').
        """
        valid: dict[str, np.ndarray] = {}
        for name, arr in masks.items():
            try:
                a = np.asarray(arr)
                if a.shape == self._ct.shape:  # type: ignore[union-attr]
                    valid[name] = (a > 0).astype(np.uint8)
            except Exception:
                continue
        self._masks = valid
        self.update_slice()

    def set_voi_colors(self, colors: dict[str, tuple[int, int, int]]) -> None:
        """Assign RGB colors (0..255) for VOI names.

        Parameters
        ----------
        colors : dict[str, tuple[int,int,int]]
            Mapping from VOI name to RGB tuple.
        """
        self._voi_colors = {
            k: tuple(int(max(0, min(255, c))) for c in v)  # clamp defensively
            for k, v in colors.items()
            if isinstance(v, (tuple, list)) and len(v) == 3
        }
        self.update_slice()

    def set_plane(self, plane: str) -> None:
        """Set the anatomical plane and update slider/slice.

        Parameters
        ----------
        plane : str
            One of Axial, Sagittal, Coronal (case-insensitive).
        """
        self._plane = plane
        self._configure_slider()
        self.update_slice()

    def _configure_slider(self) -> None:
        if self._ct is None:
            self.slice_slider.setEnabled(False)
            self.slice_spin.setEnabled(False)
            return
        axis = self._PLANE_MAP.get(self._plane, 2)
        size = self._ct.shape[axis]
        self.slice_slider.setEnabled(True)
        self.slice_slider.setMinimum(0)
        self.slice_slider.setMaximum(size - 1)
        self.slice_spin.setEnabled(True)
        self.slice_spin.setMinimum(0)
        self.slice_spin.setMaximum(size - 1)
        mid = size // 2
        if self.slice_slider.value() != mid:
            self.slice_slider.setValue(mid)
        if self.slice_spin.value() != self.slice_slider.value():
            self.slice_spin.blockSignals(True)
            self.slice_spin.setValue(self.slice_slider.value())
            self.slice_spin.blockSignals(False)
        self._slice_label.setText(f"/ {size - 1}")

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def update_slice(self) -> None:
        """Render current slice with layered: CT (back) | Quantity (opacity) | masks & isolines (front)."""
        if self._ct is None:
            return

        idx = self.slice_slider.value()
        ct_slice = self._extract_slice(self._ct, self._plane, idx)

        self._render_ct_layer(ct_slice)

        quantity_arr = (
            self._quantities.get(self._active_quantity_name)
            if self._active_quantity_name
            else None
        )

        quantity_slice = self._render_quantity_layer(quantity_arr, idx)
        self._manage_colorbar(idx, quantity_arr, ct_slice, quantity_slice)

        self._update_isolines(idx, quantity_arr)
        self._update_contours(idx)
        self._update_isocenter(idx)

        self._slice_label.setText(f"/ {self.slice_slider.maximum()} ({self._plane})")
        self.slice_changed.emit(idx)

        min_val, max_val = 0.0, 0.0
        if (
            self._colorbar_mode == "quantity"
            and quantity_arr is not None
            and quantity_slice is not None
        ):
            valid = quantity_slice[np.isfinite(quantity_slice)]
            if valid.size > 0:
                min_val, max_val = float(valid.min()), float(valid.max())
        elif self._colorbar_mode == "ct" and self._ct is not None:
            valid = ct_slice[np.isfinite(ct_slice)]
            if valid.size > 0:
                min_val, max_val = float(valid.min()), float(valid.max())

        self.range_changed.emit(min_val, max_val)

        if self.slice_spin.value() != idx:
            self.slice_spin.blockSignals(True)
            self.slice_spin.setValue(idx)
            self.slice_spin.blockSignals(False)

        self.range_changed.emit(self.slice_slider.minimum(), self.slice_slider.maximum())

    def _render_ct_layer(self, ct_slice: np.ndarray) -> None:
        """Create or update the CT base image item."""
        if self._image_item is None:
            self._image_item = pg.ImageItem(ct_slice)
            self._image_item.setZValue(0)
            self._view_box.addItem(self._image_item)
        else:
            self._image_item.setImage(ct_slice, autoLevels=False)

        if self._ct_window is not None:
            center, width = self._ct_window
            self._image_item.setLevels((center - width / 2, center + width / 2))
        elif self._global_max:
            self._image_item.setLevels((float(self._ct.min()), float(self._ct.max())))
        else:
            self._image_item.setLevels((float(ct_slice.min()), float(ct_slice.max())))
        self._image_item.setOpacity(1.0 if self._show_ct else 0.0)

    def _render_quantity_layer(
        self, quantity_arr: Optional[np.ndarray], idx: int
    ) -> Optional[np.ndarray]:
        """Create or update the quantity overlay image item. Returns the rendered slice or None."""
        if not (self._show_quantity and quantity_arr is not None):
            if self._quantity_item is not None:
                self._view_box.removeItem(self._quantity_item)
                self._quantity_item = None
            return None

        quantity_slice = self._extract_slice(quantity_arr, self._plane, idx).astype(float)
        quantity_slice[quantity_slice == 0] = np.nan

        if self._quantity_item is None:
            cmap = pg.colormap.getFromMatplotlib(self._quantity_colormap)
            lut = cmap.getLookupTable(0.0, 1.0, 256)
            self._last_cmap = cmap
            self._quantity_item = pg.ImageItem(
                quantity_slice, lut=lut, opacity=self._quantity_opacity
            )
            self._quantity_item.setZValue(5)
            self._view_box.addItem(self._quantity_item)
        else:
            cmap = pg.colormap.getFromMatplotlib(self._quantity_colormap)
            lut = cmap.getLookupTable(0.0, 1.0, 256)
            self._quantity_item.setLookupTable(lut)
            self._quantity_item.setImage(quantity_slice, autoLevels=not self._global_max)
            self._quantity_item.setOpacity(self._quantity_opacity)

        if self._quantity_window is not None:
            center, width = self._quantity_window
            self._quantity_item.setLevels((center - width / 2, center + width / 2))
        elif self._global_max:
            self._quantity_item.setLevels((float(quantity_arr.min()), float(quantity_arr.max())))

        return quantity_slice

    def _manage_colorbar(
        self,
        idx: int,
        quantity_arr: Optional[np.ndarray],
        ct_slice: np.ndarray,
        quantity_slice: Optional[np.ndarray],
    ) -> None:
        """Show/hide/update colorbar depending on active mode and available data."""
        if self._active_mode == "quantity" and quantity_arr is not None:
            if quantity_slice is None:
                quantity_slice = self._extract_slice(quantity_arr, self._plane, idx).astype(float)
                quantity_slice[quantity_slice == 0] = np.nan
            self._ensure_colorbar("quantity", quantity_slice)
        elif self._active_mode == "ct" and self._ct is not None:
            self._ensure_colorbar("ct", ct_slice)
        elif getattr(self, "_colorbar", None) is not None:
            try:
                self._plot_widget.removeItem(self._colorbar)
            except Exception:
                pass
            self._colorbar = None
            self._colorbar_mode = None

    def _get_colorbar_params(self, mode: str):
        """Return ``(volume, item, label, cmap)`` for the given mode, or ``None`` if invalid."""
        if mode == "quantity" and self._active_quantity_name:
            volume = self._quantities[self._active_quantity_name]
            item = self._quantity_item
            unit = self._get_active_unit()
            name = self._get_active_label()
            label = f"{name} [{unit}]" if unit else name
            cmap = pg.colormap.getFromMatplotlib(self._quantity_colormap)
            self._last_cmap = cmap
            return volume, item, label, cmap
        if mode == "ct" and self._ct is not None:
            volume = self._ct
            item = self._image_item
            label = "CT [HU]"
            if hasattr(pg, "colormap"):
                try:
                    cmap = pg.colormap.get(self._ct_colormap)
                except Exception:
                    cmap = pg.ColorMap([0.0, 1.0], [(0, 0, 0), (255, 255, 255)])
            else:
                cmap = pg.ColorMap([0.0, 1.0], [(0, 0, 0), (255, 255, 255)])
            return volume, item, label, cmap
        return None

    def _ensure_colorbar(self, mode: str, slice_data: np.ndarray) -> None:
        """Ensure a colorbar exists for the given mode ('Quantity'|'ct')."""
        needs_recreate = getattr(self, "_colorbar_mode", None) != mode
        if (
            mode == "quantity"
            and getattr(self, "_colorbar_quantity_name", None) != self._active_quantity_name
        ):
            needs_recreate = True

        if needs_recreate and getattr(self, "_colorbar", None) is not None:
            try:
                self._plot_widget.removeItem(self._colorbar)
            except Exception:
                pass
            self._colorbar = None

        if getattr(self, "_colorbar", None) is not None:
            self._colorbar_mode = mode
            self._update_colorbar_range(mode, slice_data)
            return

        params = self._get_colorbar_params(mode)
        if params is None:
            return

        volume, item, label, cmap = params
        vmin = float(volume.min())
        vmax = float(volume.max())
        self._colorbar = pg.ColorBarItem(values=(vmin, vmax), colorMap=cmap, interactive=False)
        self._plot_widget.addItem(self._colorbar, row=0, col=1)
        if item is not None:
            self._colorbar.setImageItem(item)
        axis = self._colorbar.getAxis("right") if hasattr(self._colorbar, "getAxis") else None
        if axis is not None:
            try:
                axis.setLabel(text=label)
            except Exception:
                pass
        self._colorbar_mode = mode
        self._colorbar_quantity_name = self._active_quantity_name if mode == "quantity" else None
        self._update_colorbar_range(mode, slice_data)

    def _resolve_range(
        self,
        window: tuple[float, float] | None,
        full_arr: np.ndarray,
        slice_data: np.ndarray,
    ) -> tuple[float, float]:
        """Resolve ``(vmin, vmax)`` from an explicit window, global extent, or slice extent."""
        if window is not None:
            center, width = window
            return center - width / 2, center + width / 2
        if self._global_max:
            return float(full_arr.min()), float(full_arr.max())
        valid = slice_data[np.isfinite(slice_data)]
        if valid.size == 0:
            return 0.0, 1e-6
        return float(valid.min()), float(valid.max())

    def _compute_display_range(self, mode: str, slice_data: np.ndarray) -> tuple[float, float]:
        """Compute ``(vmin, vmax)`` for the colorbar based on mode and windowing settings."""
        if mode == "quantity" and self._active_quantity_name:
            arr = self._quantities[self._active_quantity_name]
            return self._resolve_range(self._quantity_window, arr, slice_data)
        if mode == "ct" and self._ct is not None:
            return self._resolve_range(self._ct_window, self._ct, slice_data)
        return 0.0, 1.0

    def _update_colorbar_range(self, mode: str, slice_data: np.ndarray) -> None:
        """Update colorbar range according to global/local scaling for mode."""
        if getattr(self, "_colorbar", None) is None:
            return
        try:
            vmin, vmax = self._compute_display_range(mode, slice_data)
            if vmax <= vmin:
                vmax = vmin + 1e-6
            axis = self._colorbar.getAxis("right")
            axis.setRange(vmin, vmax)
        except Exception:
            pass

    def _update_isolines(self, idx: int, quantity_arr: Optional[np.ndarray]) -> None:
        for iso in self._iso_items:
            self._view_box.removeItem(iso)
        self._iso_items.clear()

        if not (self._isolines and quantity_arr is not None and self._show_quantity):
            return
        quantity_slice = self._extract_slice(quantity_arr, self._plane, idx).astype(float)

        max_quantity_val = quantity_arr.max()
        max_quantity = float(max_quantity_val) if max_quantity_val > 0 else 1.0

        if self._isoline_levels:
            levels = self._isoline_levels
        else:
            max_val = float(max_quantity_val) if self._global_max else float(quantity_slice.max())
            if max_val <= 0:
                return
            levels = [0.5 * max_val, 0.8 * max_val, 0.95 * max_val]

        try:
            cmap = pg.colormap.getFromMatplotlib(self._quantity_colormap)
        except Exception:
            cmap = pg.colormap.getFromMatplotlib("jet")

        if quantity_slice.max() == quantity_slice.min():
            return

        for val in levels:
            if val > quantity_slice.max() or val < quantity_slice.min():
                continue
            norm_val = val / max_quantity
            color = cmap.map(norm_val)
            try:
                iso = pg.IsocurveItem(quantity_slice, level=val, pen=pg.mkPen(color, width=2))
                iso.setZValue(10)
                self._view_box.addItem(iso)
                self._iso_items.append(iso)
            except Exception as e:
                print(f"Error adding isoline/label: {e}")
                continue

    def _update_contours(self, idx: int) -> None:
        if not hasattr(self, "_mask_items"):
            self._mask_items = []
        for item in self._mask_items:
            self._view_box.removeItem(item)
        self._mask_items.clear()
        if not self._masks or not self._show_cst:
            return
        names = [n for n in self._selected_vois if n in self._masks]
        if not names:
            return
        n = len(names)
        for i, name in enumerate(names):
            mask = self._masks[name]
            mask_slice = self._extract_slice(mask, self._plane, idx).astype(np.int32)
            if mask_slice.max() == 0:
                continue
            if name in self._voi_colors:
                r, g, b = self._voi_colors[name]
            else:
                r = int(255 * (i / max(1, n - 1)))
                g = int(255 * (1 - i / max(1, n - 1)))
                b = 255
            pen = pg.mkPen(color=(r, g, b), width=1.5)
            try:
                curve = pg.IsocurveItem(mask_slice, level=0.5, pen=pen)
            except Exception:
                continue
            curve.setZValue(15)
            self._view_box.addItem(curve)
            self._mask_items.append(curve)

    def _update_isocenter(self, idx: int) -> None:
        if self._isocenter_item is not None:
            self._view_box.removeItem(self._isocenter_item)
            self._isocenter_item = None

        if not self._show_isocenter or self._isocenter is None:
            return

        axis = self._PLANE_MAP.get(self._plane, 2)
        iso_slice_idx = int(round(self._isocenter[axis]))

        if idx == iso_slice_idx:
            if axis == 2:  # Axial (Z) -> X, Y
                x, y = self._isocenter[0], self._isocenter[1]
            elif axis == 0:  # Sagittal (X) -> Y, Z
                x, y = self._isocenter[1], self._isocenter[2]
            elif axis == 1:  # Coronal (Y) -> X, Z
                x, y = self._isocenter[0], self._isocenter[2]
            else:
                return

            self._isocenter_item = pg.ScatterPlotItem(
                [x],
                [y],
                symbol="+",
                size=20,
                pen=pg.mkPen("r", width=2),
                brush=pg.mkBrush("r"),
            )
            self._isocenter_item.setZValue(20)
            self._view_box.addItem(self._isocenter_item)

    @staticmethod
    def _extract_slice(volume: np.ndarray, plane: str, index: int) -> np.ndarray:
        axis = QuantityWidget._PLANE_MAP.get(plane, 2)
        if axis == 2:  # axial: X,Y, index in Z
            return volume[:, :, index]
        if axis == 0:  # sagittal: Y,Z
            return volume[index, :, :]
        if axis == 1:  # coronal: X,Z
            return volume[:, index, :]
        return volume[:, :, index]

    # ------------------------------------------------------------------
    # Slots for external signals (ViewerWidget)
    # ------------------------------------------------------------------
    def on_plane_changed(self, plane: str) -> None:
        """Slot: receive plane change from viewer widget."""
        self.set_plane(plane)

    def on_global_max_toggled(self, enabled: bool) -> None:
        """Slot: enable/disable global intensity scaling."""
        self._global_max = enabled
        self.update_slice()

    def on_isolines_toggled(self, enabled: bool) -> None:
        """Slot: toggle quantity isoline overlay rendering."""
        self._isolines = enabled
        self.update_slice()

    def on_overlay_toggled(self, name: str, enabled: bool) -> None:
        """Slot: toggle visibility of CT or quantity overlay.

        Other overlay names are ignored gracefully for now.
        """
        key = name.upper()
        if key == "CT":
            self._show_ct = enabled
            if self._image_item is not None:
                self._image_item.setOpacity(1.0 if enabled else 0.0)
        elif key in {"QUANTITY", "PHYSICAL QUANTITY", "PHYSICAL_QUANTITY"}:
            self._show_quantity = enabled
        elif key == "CST":
            self._show_cst = enabled

        self.update_slice()

    def on_vois_selection_changed(self, vois: Sequence[str]) -> None:
        """Slot: record selected VOI names (rendering to be implemented)."""
        self._selected_vois = vois
        self.update_slice()

    # ------------------------------------------------------------------
    # Internal callbacks
    # ------------------------------------------------------------------
    def _on_slider_changed(self, value: int) -> None:  # value provided by Qt
        self.update_slice()

    def _on_scroll_slice(self, delta: int) -> None:
        new_val = self.slice_slider.value() + delta
        new_val = max(self.slice_slider.minimum(), min(self.slice_slider.maximum(), new_val))
        self.slice_slider.setValue(new_val)

    def eventFilter(self, obj, ev):  # noqa: N802
        """Handle native gesture events for pinch-to-zoom on the viewport."""
        if ev.type() == QEvent.Type.NativeGesture:
            if ev.gestureType() == Qt.NativeGestureType.ZoomNativeGesture:
                factor = 1.0 + ev.value()
                if factor > 0:
                    self._view_box.scaleBy((1.0 / factor, 1.0 / factor))
                ev.accept()
                return True
        return super().eventFilter(obj, ev)

    def _on_spin_changed(self, value: int) -> None:
        if value != self.slice_slider.value():
            self.slice_slider.setValue(value)

    # ------------------------------------------------------------------
    # Convenience wiring
    # ------------------------------------------------------------------
    def connect_viewer_signals(self, viewer: QWidget) -> None:
        """Connect signals from a `ViewerWidget` instance.

        Uses duck typing; only connects existing attributes.
        """
        if hasattr(viewer, "plane_changed"):
            viewer.plane_changed.connect(self.on_plane_changed)
        if hasattr(viewer, "global_max_toggled"):
            viewer.global_max_toggled.connect(self.on_global_max_toggled)
        if hasattr(viewer, "isolines_toggled"):
            viewer.isolines_toggled.connect(self.on_isolines_toggled)
        if hasattr(viewer, "overlay_toggled"):
            viewer.overlay_toggled.connect(self.on_overlay_toggled)
        if hasattr(viewer, "vois_selection_changed"):
            viewer.vois_selection_changed.connect(self.on_vois_selection_changed)

    # ------------------------------------------------------------------
    # New Visualization Methods
    # ------------------------------------------------------------------
    def set_isolines(self, levels: list[float]) -> None:
        """Set custom isoline levels in overlay units (e.g. Gy)."""
        self._isoline_levels = levels
        self.update_slice()

    def set_isocenter_visible(self, visible: bool) -> None:
        """Toggle isocenter visibility."""
        self._show_isocenter = visible
        self.update_slice()

    def set_opacity(self, opacity: float) -> None:
        """Set opacity for the quantity overlay (0.0 to 1.0)."""
        self._quantity_opacity = max(0.0, min(1.0, opacity))
        if self._quantity_item is not None:
            self._quantity_item.setOpacity(self._quantity_opacity)

    def set_window_level(self, center: float, width: float, mode: Optional[str] = None) -> None:
        """Set window/level for the specified mode (or active mode)."""
        target_mode = mode if mode else self._active_mode
        if target_mode == "quantity":
            self._quantity_window = (center, width)
        else:
            self._ct_window = (center, width)
        self.update_slice()

    def get_window_level(self, mode: Optional[str] = None) -> tuple[float, float] | None:
        """Get current window/level for the specified mode."""
        target_mode = mode if mode else self._active_mode
        if target_mode == "quantity":
            return self._quantity_window
        return self._ct_window

    def get_colormap(self, mode: Optional[str] = None) -> str:
        """Get current colormap name for the specified mode."""
        target_mode = mode if mode else self._active_mode
        if target_mode == "quantity":
            return self._quantity_colormap
        return self._ct_colormap

    def set_active_mode(self, mode: str) -> None:
        """Set the active mode for colorbar and editing (quantity or ct)."""
        if mode not in ("quantity", "ct"):
            return
        self._active_mode = mode
        self.update_slice()

    def set_colormap(self, name: str, mode: str = "quantity") -> None:
        """Set colormap for quantity or CT."""
        if mode == "quantity":
            self._quantity_colormap = name
            if self._quantity_item is not None:
                self._view_box.removeItem(self._quantity_item)
                self._quantity_item = None
        elif mode == "ct":
            self._ct_colormap = name

        if getattr(self, "_colorbar", None) is not None:
            try:
                self._plot_widget.removeItem(self._colorbar)
            except Exception:
                pass
            self._colorbar = None

        self.update_slice()

    def set_active_quantity(self, name: str | None) -> None:
        """Set the active quantity to display."""
        if name in self._quantities:
            self._active_quantity_name = name
        else:
            self._active_quantity_name = None
        self.update_slice()

    def get_data_range(self, mode: str) -> tuple[float, float]:
        """Return (min, max) of the full volume for the given mode."""
        if mode == "quantity" and self._active_quantity_name:
            arr = self._quantities.get(self._active_quantity_name)
            if arr is not None:
                return float(arr.min()), float(arr.max())
        elif mode == "ct" and self._ct is not None:
            return float(self._ct.min()), float(self._ct.max())
        return 0.0, 1.0

    def get_current_slice_range(self, mode: str) -> tuple[float, float]:
        """Return (min, max) of the current slice for the given mode."""
        idx = self.slice_slider.value()

        if mode == "quantity" and self._active_quantity_name:
            arr = self._quantities.get(self._active_quantity_name)
            if arr is not None:
                slice_data = self._extract_slice(arr, self._plane, idx).astype(float)
                valid = slice_data[np.isfinite(slice_data)]
                if valid.size > 0:
                    return float(valid.min()), float(valid.max())
                return 0.0, 0.0

        elif mode == "ct" and self._ct is not None:
            slice_data = self._extract_slice(self._ct, self._plane, idx)
            valid = slice_data[np.isfinite(slice_data)]
            if valid.size > 0:
                return float(valid.min()), float(valid.max())

        return 0.0, 1.0

    def recenter_to_isocenter(self) -> None:
        """Set the slice slider to the isocenter position and reset the view."""
        if self._isocenter is None or self._ct is None:
            return

        axis = self._PLANE_MAP.get(self._plane, 2)
        if 0 <= axis < len(self._isocenter):
            slice_idx = int(round(self._isocenter[axis]))
            slice_idx = max(
                self.slice_slider.minimum(), min(self.slice_slider.maximum(), slice_idx)
            )
            self.slice_slider.setValue(slice_idx)
        self._view_box.autoRange()

    def reset_options(self) -> None:
        """Reset visualization options to defaults."""
        self._quantity_window = None
        self._ct_window = None
        self._quantity_colormap = "jet"
        self._ct_colormap = "grey"
        self._quantity_opacity = 0.4

        if self._quantity_item is not None:
            self._view_box.removeItem(self._quantity_item)
            self._quantity_item = None

        if getattr(self, "_colorbar", None) is not None:
            try:
                self._plot_widget.removeItem(self._colorbar)
            except Exception:
                pass
            self._colorbar = None

        self.update_slice()

    def get_available_quantities(self) -> list[str]:
        """Return a list of available quantity/quantity names."""
        return list(self._quantities.keys())
