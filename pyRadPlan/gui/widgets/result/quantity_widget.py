"""Slice quantity viewing widget using PySide6."""

from __future__ import annotations

from functools import partial
from typing import Callable, Optional, Sequence

import numpy as np
import SimpleITK as sitk
import pyqtgraph as pg
from PySide6.QtCore import QEvent, QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QBrush, QColor, QKeySequence, QPainter, QPen, QPolygonF, QShortcut
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
    QComboBox,
)

from .._display_names import result_key_labels
from ._orientation_marker import OrientationMarker

#: Colour of the beam axis overlay; distinct from the red isocenter marker.
_BEAM_COLOR = (255, 210, 0)
#: Screen width of a beam wedge at the isocenter, in pixels.
_BEAM_WIDTH_PX = 3.0
#: Source-to-isocenter width ratio of a beam pointing straight at the viewer; a
#: beam tilted by an angle to the plane gets this ratio to the power of its sine.
_BEAM_PERSPECTIVE = 6.0
#: Beams whose in-plane fraction is below this (about 2 deg from the normal) are
#: drawn as a marker at the isocenter instead of a projected line.
_BEAM_MIN_IN_PLANE = 0.035
#: Beams tilted out of the plane by more than about 5 deg get a ⊙ / ⊗ after their
#: label, since a slight taper alone is easy to miss.
_BEAM_MIN_OUT_OF_PLANE = 0.087
#: A beam can be dragged in a view only while its gantry orbit is tilted from
#: the screen by less than about 72 deg, i.e. not seen (almost) edge-on.
_BEAM_MIN_ORBIT_TILT = 0.3
#: Colour of the dashed gantry orbit shown while dragging a beam.
_ORBIT_COLOR = (200, 200, 200, 170)
#: Margin of the orientation marker from the corner of the plot area, in pixels.
_ORIENTATION_MARGIN = 8
#: Image direction of each viewer axis in (x, y, z); the viewer volume is (z, x, y).
_VIEWER_AXES_XYZ = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


class _BeamWedgeItem(pg.GraphicsObject):
    """Beam axis drawn as a wedge whose width changes from source to isocenter.

    The end points live in data coordinates, the widths are screen pixels so the
    wedge keeps its shape while zooming.  Like perspective, a wider source end
    reads as a beam coming from in front of the displayed plane.  A *pending*
    beam, edited but not applied yet, gets a dashed outline and a fainter fill.
    """

    def __init__(
        self,
        source: tuple[float, float],
        target: tuple[float, float],
        source_width: float,
        target_width: float,
        color: tuple[int, int, int],
        pending: bool = False,
    ) -> None:
        super().__init__()
        self._source = QPointF(*source)
        self._target = QPointF(*target)
        self.widths = (float(source_width), float(target_width))
        self.pending = pending
        self._pen = QPen(QColor(*color))
        self._pen.setCosmetic(True)
        if pending:
            self._pen.setStyle(Qt.PenStyle.DashLine)
        fill = QColor(*color)
        fill.setAlpha(40 if pending else 110)
        self._brush = QBrush(fill)

    def set_source(self, source: tuple[float, float], width: float) -> None:
        """Move the source end, e.g. while the beam is dragged."""
        self.prepareGeometryChange()
        self._source = QPointF(*source)
        self.widths = (float(width), self.widths[1])
        self.update()

    def viewTransformChanged(self) -> None:  # noqa: N802
        super().viewTransformChanged()
        # The pixel widths map to a different data extent after zooming.
        self.prepareGeometryChange()

    def boundingRect(self) -> QRectF:  # noqa: N802
        rect = QRectF(self._source, self._target).normalized()
        pad = max(self.widths) / 2.0 + 1.0
        px_x = self.pixelLength(QPointF(1.0, 0.0))
        px_y = self.pixelLength(QPointF(0.0, 1.0))
        if px_x is None or px_y is None:
            return rect
        return rect.adjusted(-pad * px_x, -pad * px_y, pad * px_x, pad * px_y)

    def paint(self, painter: QPainter, *_args) -> None:
        transform = painter.transform()
        a = transform.map(self._source)
        b = transform.map(self._target)
        dx, dy = b.x() - a.x(), b.y() - a.y()
        length = float(np.hypot(dx, dy))
        if length < 1e-9:
            return
        nx, ny = -dy / length, dx / length
        half_a, half_b = self.widths[0] / 2.0, self.widths[1] / 2.0
        polygon = QPolygonF(
            [
                QPointF(a.x() + nx * half_a, a.y() + ny * half_a),
                QPointF(b.x() + nx * half_b, b.y() + ny * half_b),
                QPointF(b.x() - nx * half_b, b.y() - ny * half_b),
                QPointF(a.x() - nx * half_a, a.y() - ny * half_a),
            ]
        )
        painter.save()
        # Draw in device pixels so the widths do not scale with the zoom.
        painter.resetTransform()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(self._pen)
        painter.setBrush(self._brush)
        painter.drawPolygon(polygon)
        painter.restore()


class _BeamSourceItem(pg.ScatterPlotItem):
    """Source marker of a beam; dragging it calls *on_drag* with the drag event."""

    def __init__(self, *args, on_drag: Optional[Callable] = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._on_drag = on_drag
        if on_drag is not None:
            self.setCursor(Qt.CursorShape.OpenHandCursor)

    def mouseDragEvent(self, ev) -> None:  # noqa: N802
        if self._on_drag is None or ev.button() != Qt.MouseButton.LeftButton:
            ev.ignore()
            return
        # Only a drag starting on the marker itself; elsewhere the view pans.
        if ev.isStart() and not len(self.pointsAt(ev.buttonDownPos())):
            ev.ignore()
            return
        ev.accept()
        self._on_drag(ev)


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
    #: A beam was dragged to a new gantry angle: (index in the plan, angle in degrees).
    beam_angle_changed = Signal(int, float)
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
        """Initialize all instance-level state attributes."""
        self._ct: Optional[np.ndarray] = None
        #: The array last passed to :meth:`set_data`, to tell a new image from a new result.
        self._ct_source: Optional[np.ndarray] = None
        #: Whether the slice still follows the isocenter: true from a new image on until the
        #: user picks a slice, so an isocenter known only later (e.g. once the structures
        #: are loaded) still positions the view.
        self._follow_isocenter: bool = True
        self._setting_slice: bool = False
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
        self._ct_origin: Optional[np.ndarray] = None
        self._ct_spacing: Optional[np.ndarray] = None
        self._ct_direction = np.eye(3)
        self._show_isocenter: bool = False
        #: One entry per beam, in viewer voxel space (see :meth:`set_beams`).
        self._beams: list[dict] = []
        self._show_beams: bool = False
        #: Items of each drawn beam by its position in ``_beams``, for dragging.
        self._beam_parts: dict[int, dict] = {}
        self._beam_drag: Optional[dict] = None
        self._show_orientation: bool = True
        self._isoline_levels: list[float] = []
        self._quantity_opacity: float = 0.4
        self._ct_window: Optional[tuple[float, float]] = None
        self._quantity_window: Optional[tuple[float, float]] = None
        self._quantity_colormap: str = "jet"
        self._ct_colormap: str = "gray"
        self._active_mode: str = "quantity"

        self._image_item = None
        self._quantity_item = None
        self._iso_items: list = []
        self._isocenter_item = None
        self._beam_items: list = []

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

        # Cursor info overlay (lower-left corner of the plot area)
        self._info_label = QLabel(self._plot_widget)
        self._info_label.setStyleSheet(
            "background-color: rgba(0, 0, 0, 150); color: white;"
            "padding: 4px 6px; border-radius: 3px;"
            "font-family: monospace; font-size: 8pt;"
        )
        self._info_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._info_label.hide()
        self._plot_widget.scene().sigMouseMoved.connect(self._on_mouse_moved)

        self._cancel_drag_shortcut = QShortcut(
            QKeySequence(Qt.Key.Key_Escape), self, context=Qt.ShortcutContext.WindowShortcut
        )
        self._cancel_drag_shortcut.setEnabled(False)
        self._cancel_drag_shortcut.activated.connect(self._abort_beam_drag)

        self._orientation_marker = OrientationMarker(self._plot_widget)
        self._orientation_marker.move(_ORIENTATION_MARGIN, _ORIENTATION_MARGIN)
        self._orientation_marker.hide()

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

        A new image shows the isocenter slice (the middle one without an isocenter), and
        follows the isocenter until the user picks a slice; from then on the same
        ``ct_volume`` object again, e.g. with a new result, keeps the current slice.
        """
        new_image = ct_volume is not self._ct_source
        self._ct_source = ct_volume
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
                self._quantities["Physical quantity"] = quantity_volume.astype(float, copy=False)
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

        if new_image:
            self._follow_isocenter = True
        self._configure_slider(reset=self._follow_isocenter)
        self.update_slice()
        self._update_orientation_marker()

    def clear_data(self) -> None:
        """Remove all imaging data and rendered items (e.g. workspace cleared)."""
        self._ct = None
        self._ct_source = None
        self._quantities = {}
        self._active_quantity_name = None
        self._masks = {}
        self._isocenter = None
        self._ct_origin = None
        self._ct_spacing = None
        self._ct_direction = np.eye(3)
        self._beams = []
        self._clear_beam_items()
        self._info_label.hide()
        for attr in ("_image_item", "_quantity_item", "_isocenter_item"):
            item = getattr(self, attr)
            if item is not None:
                self._view_box.removeItem(item)
                setattr(self, attr, None)
        for iso in self._iso_items:
            self._view_box.removeItem(iso)
        self._iso_items.clear()
        for item in getattr(self, "_mask_items", []):
            self._view_box.removeItem(item)
        if hasattr(self, "_mask_items"):
            self._mask_items.clear()
        if self._colorbar is not None:
            try:
                self._plot_widget.removeItem(self._colorbar)
            except Exception:  # noqa: BLE001 - already removed
                pass
            self._colorbar = None
            self._colorbar_mode = None
        self._configure_slider()
        self._slice_label.setText("Slice: -")
        self._update_orientation_marker()

    def set_ct_geometry(
        self,
        origin: Optional[Sequence[float]],
        spacing: Optional[Sequence[float]],
        direction: Optional[Sequence[float]] = None,
    ) -> None:
        """Set CT geometry, with a row-major image-to-LPS direction matrix.

        Origin and spacing use (x, y, z) order. Omitting direction assumes an
        identity matrix, as for a volume already aligned with the LPS axes.
        """
        self._ct_origin = None if origin is None else np.asarray(origin, dtype=float)
        self._ct_spacing = None if spacing is None else np.asarray(spacing, dtype=float)
        self._ct_direction = (
            np.eye(3) if direction is None else np.asarray(direction, dtype=float).reshape(3, 3)
        )
        self._update_orientation_marker()

    # ------------------------------------------------------------------
    # Cursor info overlay
    # ------------------------------------------------------------------
    def _on_mouse_moved(self, scene_pos) -> None:
        if self._ct is None or not self._view_box.sceneBoundingRect().contains(scene_pos):
            self._info_label.hide()
            return
        view_pos = self._view_box.mapSceneToView(scene_pos)
        index = self._view_to_volume_index(view_pos.x(), view_pos.y())
        if index is None:
            self._info_label.hide()
            return
        self._info_label.setText(self._format_cursor_info(index))
        self._info_label.adjustSize()
        self._position_info_label()
        self._info_label.show()

    def _view_to_volume_index(self, vx: float, vy: float) -> Optional[tuple[int, int, int]]:
        """Map view coordinates and slice index to a (iz, ix, iy) viewer-volume index."""
        axis = self._PLANE_MAP.get(self._plane, 2)
        idx = self.slice_slider.value()
        a, b = int(np.floor(vx)), int(np.floor(vy))
        if axis == 0:  # Axial: view = (x, y)
            iz, ix, iy = idx, a, b
        elif axis == 1:  # Sagittal: view = (z, y)
            iz, ix, iy = a, idx, b
        else:  # Coronal: view = (z, x)
            iz, ix, iy = a, b, idx
        nz, nx, ny = self._ct.shape
        if not (0 <= iz < nz and 0 <= ix < nx and 0 <= iy < ny):
            return None
        return iz, ix, iy

    def _format_cursor_info(self, index: tuple[int, int, int]) -> str:
        """Build the overlay text for the voxel at the given (iz, ix, iy) index."""
        iz, ix, iy = index
        lines = []
        if self._ct_origin is not None and self._ct_spacing is not None:
            pos = self._ct_origin + self._ct_spacing * np.array([ix, iy, iz], dtype=float)
            lines.append(f"Pos: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}) mm")
        lines.append(f"Index: [{ix}, {iy}, {iz}] (sitk) / [{iz}, {iy}, {ix}] (numpy)")
        lines.append(f"CT: {self._ct[iz, ix, iy]:.0f} HU")
        labels = result_key_labels(self._quantities)
        for name, arr in self._quantities.items():
            base = name.split()[0] if " " in name else name
            if base.endswith("_beam"):
                continue
            unit = self._overlay_units.get(name, getattr(self, "_overlay_units_default", ""))
            value = f"{labels[name]}: {arr[iz, ix, iy]:.4g}"
            lines.append(f"{value} {unit}" if unit else value)
        return "\n".join(lines)

    def _position_info_label(self) -> None:
        margin = 6
        self._info_label.move(
            margin, self._plot_widget.height() - self._info_label.height() - margin
        )

    def _process_dict_quantity(self, quantity_dict: dict) -> dict[str, np.ndarray]:
        """Process a dict of quantity volumes into a flat ``{name: array}`` mapping."""
        result: dict[str, np.ndarray] = {}
        for k, v in quantity_dict.items():
            if isinstance(v, list):
                for i, beam_vol in enumerate(v):
                    try:
                        arr = beam_vol.astype(float, copy=False)
                    except AttributeError:
                        if isinstance(beam_vol, sitk.Image):
                            arr = sitk.GetArrayFromImage(beam_vol).transpose(2, 1, 0).astype(float)
                        else:
                            continue
                    result[f"{k} {i}"] = arr
            else:
                try:
                    arr = v.astype(float, copy=False)
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
        self._configure_slider(reset=True)
        self.update_slice()
        self._update_orientation_marker()

    def _configure_slider(self, reset: bool = True) -> None:
        """Set the slider range for the current plane; *reset* moves to :meth:`_home_slice`."""
        if self._ct is None:
            self.slice_slider.setEnabled(False)
            self.slice_spin.setEnabled(False)
            return
        axis = self._PLANE_MAP.get(self._plane, 2)
        size = self._ct.shape[axis]
        # A kept slice outside the new range is clamped by setMaximum, which is not a
        # slice the user picked either.
        self._setting_slice = True
        try:
            self.slice_slider.setEnabled(True)
            self.slice_slider.setMinimum(0)
            self.slice_slider.setMaximum(size - 1)
            self.slice_spin.setEnabled(True)
            self.slice_spin.setMinimum(0)
            self.slice_spin.setMaximum(size - 1)
            if reset:
                self.slice_slider.setValue(self._home_slice())
        finally:
            self._setting_slice = False
        if self.slice_spin.value() != self.slice_slider.value():
            self.slice_spin.blockSignals(True)
            self.slice_spin.setValue(self.slice_slider.value())
            self.slice_spin.blockSignals(False)
        self._slice_label.setText(f"/ {size - 1}")

    def _home_slice(self) -> int:
        """Slice through the isocenter, or the middle slice without one."""
        axis = self._PLANE_MAP.get(self._plane, 2)
        if self._isocenter is not None and 0 <= axis < len(self._isocenter):
            idx = int(round(float(self._isocenter[axis])))
            return max(self.slice_slider.minimum(), min(self.slice_slider.maximum(), idx))
        return self._ct.shape[axis] // 2

    def _set_slice(self, idx: int) -> None:
        """Move the slider without counting it as the user's choice of slice."""
        self._setting_slice = True
        try:
            self.slice_slider.setValue(idx)
        finally:
            self._setting_slice = False

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
        self._update_beams(idx)
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

    def _plane_axes(self, plane: Optional[str] = None) -> tuple[int, int]:
        """Volume axes shown horizontally and vertically in *plane* (default: the current one)."""
        axis = self._PLANE_MAP.get(plane or self._plane, 2)
        return tuple(i for i in (0, 1, 2) if i != axis)

    def _update_orientation_marker(self) -> None:
        """Point the orientation marker at the current plane; shown only with an image."""
        h_axis, v_axis = self._plane_axes()
        axes_lps = _VIEWER_AXES_XYZ @ self._ct_direction.T
        # The vertical axis is inverted, so the second viewer axis runs downwards.
        self._orientation_marker.set_view(axes_lps[h_axis], axes_lps[v_axis])
        self._orientation_marker.setVisible(self._show_orientation and self._ct is not None)

    def _viewer_spacing(self) -> np.ndarray:
        """Voxel spacing in mm along the viewer's (z, x, y) axes."""
        return np.ones(3) if self._ct_spacing is None else self._ct_spacing[[2, 0, 1]]

    def _beam_radius(self, h_iso: float, v_iso: float) -> Optional[float]:
        """Source distance in mm of the beams around an isocenter, or None outside the slice.

        The radius of the largest circle around the isocenter that fits in the
        slice, so the beam lines stay inside the image instead of running off into
        empty space, which would also widen the auto-ranged view.  Every beam gets
        the same distance, so its source stays on its gantry orbit while dragged.
        """
        h_axis, v_axis = self._plane_axes()
        spacing = self._viewer_spacing()
        radius = min(
            min(h_iso, self._ct.shape[h_axis] - 1 - h_iso) * spacing[h_axis],
            min(v_iso, self._ct.shape[v_axis] - 1 - v_iso) * spacing[v_axis],
        )
        return float(radius) if radius > 0.0 else None

    @staticmethod
    def _label_anchor(component: float) -> float:
        """Text anchor along one screen axis for a tip label pointing *component*.

        ``0`` puts the text's leading edge on the point so it extends towards
        larger coordinates, ``1`` does the opposite and ``0.5`` centres it.  The
        vertical axis is inverted in the view, which flips "larger" to "further
        down" for both the beam component and the anchor, so one rule serves both
        axes.
        """
        if component > 0.3:
            return 0.0
        if component < -0.3:
            return 1.0
        return 0.5

    def _toward_viewer(self, direction: np.ndarray, plane: Optional[str] = None) -> float:
        """Signed sine of the angle between *direction* and the displayed plane.

        Positive when *direction* points out of the screen towards the viewer.
        *direction* is in viewer voxel space and is scaled to mm first, so the
        value reflects the physical angle.  The slice is shown column-major with
        an inverted vertical axis, making (horizontal, vertical, into-screen) a
        right-handed frame in which ``e_h x e_v`` points into the screen.
        """
        h_axis, v_axis = self._plane_axes(plane)
        n_axis = self._PLANE_MAP.get(plane or self._plane, 2)
        spacing = np.ones(3) if self._ct_spacing is None else self._ct_spacing[[2, 0, 1]]
        physical = np.asarray(direction, dtype=float) * spacing
        norm = float(np.linalg.norm(physical))
        if norm == 0.0:
            return 0.0
        into_screen = np.cross(np.eye(3)[h_axis], np.eye(3)[v_axis])[n_axis]
        return float(-into_screen * physical[n_axis] / norm)

    @staticmethod
    def _out_of_plane_note(toward: float) -> str:
        angle = float(np.degrees(np.arcsin(min(abs(toward), 1.0))))
        if angle < 1.0:
            return "in the displayed plane"
        side = "towards" if toward > 0 else "away from"
        return f"{angle:.0f}° out of plane, source {side} the viewer"

    @staticmethod
    def _direction_symbol(toward: float) -> str:
        """⊗ (the arrow's tail) for a beam going into the screen, ⊙ (its tip) coming out.

        A source in front of the plane (*toward* > 0) means the beam travels into it.
        """
        return "⊗" if toward > 0 else "⊙"

    def _add_beam_item(self, item, z_value: float = 18) -> None:
        item.setZValue(z_value)
        self._view_box.addItem(item)
        self._beam_items.append(item)

    def _add_perpendicular_beam(
        self, i: int, beam: dict, h_iso: float, v_iso: float, toward: float
    ) -> None:
        """Mark a beam running along the view direction with the usual ⊙ / ⊗ symbol.

        ⊙ (a dot, the arrow tip) is a beam coming out of the screen, i.e. a
        source behind the plane; ⊗ (a cross, the tail) one going into it.
        """
        inner = "x" if self._direction_symbol(toward) == "⊗" else "o"
        marker = pg.ScatterPlotItem(
            spots=[
                {"pos": (h_iso, v_iso), "symbol": "o", "size": 16, "brush": None},
                {
                    "pos": (h_iso, v_iso),
                    "symbol": inner,
                    "size": 9 if inner == "x" else 5,
                    "brush": pg.mkBrush(color=_BEAM_COLOR),
                },
            ],
            pen=pg.mkPen(color=_BEAM_COLOR, width=1.5),
        )
        marker.setToolTip(self._beam_tooltip(beam, toward))
        self._add_beam_item(marker)

        label = pg.TextItem(beam.get("name", f"#{i}"), color=_BEAM_COLOR, anchor=(-0.4, 1.1))
        label.setPos(h_iso, v_iso)
        self._add_beam_item(label, 19)

    def _update_beams(self, _idx: int) -> None:
        """Draw one wedge per beam, from the source towards the isocenter.

        The beam axis is a line in space, so it is projected onto the displayed
        plane and shown on every slice rather than only where it intersects the
        current one.  All sources lie at the same distance from the isocenter
        (see :meth:`_beam_radius`), so out-of-plane beams appear shorter.  The
        wedge's source end is wider when the source lies in
        front of the plane and narrower when it lies behind it, so beams that
        are not coplanar with the view stand out, and their label carries ⊙ / ⊗
        for the direction; a beam running (almost) along the view direction
        gets a ⊙ / ⊗ marker at the isocenter instead.
        """
        self._clear_beam_items()

        if not self._show_beams or not self._beams or self._ct is None:
            return

        h_axis, v_axis = self._plane_axes()
        spacing = self._viewer_spacing()

        for i, beam in enumerate(self._beams):
            iso = np.asarray(beam["iso_center"], dtype=float)
            direction = np.asarray(beam["source_direction"], dtype=float)
            # Per mm along the beam, so the source lands at the radius in mm.
            direction = direction / max(float(np.linalg.norm(direction * spacing)), 1e-12)
            h_iso, v_iso = float(iso[h_axis]), float(iso[v_axis])
            toward = self._toward_viewer(direction)
            d_h, d_v = float(direction[h_axis]), float(direction[v_axis])
            norm = float(np.hypot(d_h, d_v))
            if norm < 1e-6 or np.sqrt(max(0.0, 1.0 - toward**2)) < _BEAM_MIN_IN_PLANE:
                self._add_perpendicular_beam(i, beam, h_iso, v_iso, toward)
                continue
            radius = self._beam_radius(h_iso, v_iso)
            if radius is None:
                continue
            h_src = h_iso + direction[h_axis] * radius
            v_src = v_iso + direction[v_axis] * radius
            d_h, d_v = d_h / norm, d_v / norm

            source_width = max(1.0, _BEAM_WIDTH_PX * _BEAM_PERSPECTIVE**toward)
            wedge = _BeamWedgeItem(
                (h_src, v_src),
                (h_iso, v_iso),
                source_width=source_width,
                target_width=_BEAM_WIDTH_PX,
                color=_BEAM_COLOR,
                pending=bool(beam.get("pending")),
            )
            self._add_beam_item(wedge)

            # Marks the source end, so the direction is readable without the
            # isocenter marker being switched on; dragging it rotates the gantry.
            draggable = self._beam_orbit_tilt(beam) >= _BEAM_MIN_ORBIT_TILT
            source = _BeamSourceItem(
                [h_src],
                [v_src],
                symbol="o",
                size=max(7.0, source_width),
                pen=pg.mkPen(color=_BEAM_COLOR),
                brush=pg.mkBrush(color=_BEAM_COLOR),
                on_drag=partial(self._on_beam_drag, i) if draggable else None,
            )
            source.setToolTip(self._beam_tooltip(beam, toward))
            self._add_beam_item(source)

            # Anchored away from the isocenter, so the text sits outside the
            # line instead of on top of it.
            text = beam.get("name", f"#{i}") + ("*" if beam.get("pending") else "")
            if abs(toward) >= _BEAM_MIN_OUT_OF_PLANE:
                text = f"{text} {self._direction_symbol(toward)}"
            label = pg.TextItem(
                text,
                color=_BEAM_COLOR,
                anchor=(self._label_anchor(d_h), self._label_anchor(d_v)),
            )
            label.setPos(h_src, v_src)
            self._add_beam_item(label, 19)
            self._beam_parts[i] = {
                "wedge": wedge,
                "source": source,
                "label": label,
                "tip": (h_src, v_src),
                "radius": radius,
            }

    def _clear_beam_items(self) -> None:
        self._cancel_beam_drag()
        for item in self._beam_items:
            self._view_box.removeItem(item)
        self._beam_items.clear()
        self._beam_parts.clear()

    def _beam_orbit_tilt(self, beam: dict, plane: Optional[str] = None) -> float:
        """How face-on the beam's gantry orbit is seen in *plane*: 1 face-on, 0 edge-on.

        0 also for beams without a plan angle to change.
        """
        if "orbit_axis" not in beam:
            return 0.0
        return abs(self._toward_viewer(beam["orbit_axis"], plane))

    def _beam_tooltip(self, beam: dict, toward: float) -> str:
        lines = [beam.get("label", ""), self._out_of_plane_note(toward)]
        if beam.get("pending"):
            lines.append("Not applied yet: apply or discard it in the plan panel")
        if "orbit_axis" in beam:
            if self._beam_orbit_tilt(beam) >= _BEAM_MIN_ORBIT_TILT:
                lines.append("Drag the source marker to change the gantry angle")
            else:
                best = max(
                    ("Axial", "Sagittal", "Coronal"),
                    key=lambda plane: self._beam_orbit_tilt(beam, plane),
                )
                lines.append(f"Drag it in the {best} view to change the gantry angle")
        return "\n".join(line for line in lines if line)

    def _on_beam_drag(self, i: int, ev) -> None:
        """Move beam *i* along its gantry orbit; release writes the angle back."""
        if ev.isStart():
            self._start_beam_drag(i)
        drag = self._beam_drag
        if drag is None or drag["index"] != i:
            return
        pos = ev.pos()
        offset = np.array([pos.x(), pos.y()]) - drag["iso"]
        if np.any(offset):
            # The orbit point is ``iso + radius * matrix @ (cos g, sin g)``, so the
            # angle whose point lies in the direction of the mouse solves this.
            c = np.linalg.solve(drag["matrix"], offset)
            self._show_dragged_beam(round(float(np.degrees(np.arctan2(c[1], c[0])))) % 360.0)
        if ev.isFinish():
            self._finish_beam_drag()

    def _start_beam_drag(self, i: int) -> None:
        """Show beam *i*'s gantry orbit, as a dashed ellipse through its source marker."""
        self._cancel_beam_drag()
        beam = self._beams[i]
        h_axis, v_axis = self._plane_axes()
        u, w = (np.asarray(vector, dtype=float) for vector in beam["orbit"])
        matrix = np.array([[u[h_axis], w[h_axis]], [u[v_axis], w[v_axis]]])
        iso = np.array([beam["iso_center"][h_axis], beam["iso_center"][v_axis]], dtype=float)
        tip = self._beam_parts[i]["tip"]
        radius = self._beam_parts[i]["radius"]

        t = np.linspace(0.0, 2.0 * np.pi, 241)
        ring = iso[:, None] + radius * matrix @ np.stack([np.cos(t), np.sin(t)])
        orbit = pg.PlotCurveItem(
            ring[0],
            ring[1],
            pen=pg.mkPen(color=_ORBIT_COLOR, width=1.5, style=Qt.PenStyle.DashLine),
        )
        orbit.setZValue(17)
        self._view_box.addItem(orbit, ignoreBounds=True)
        label = self._beam_parts[i]["label"]
        self._beam_drag = {
            "index": i,
            "matrix": matrix,
            "iso": iso,
            "radius": radius,
            "orbit": orbit,
            "gantry": beam["gantry"],
            "restore": (
                (float(tip[0]), float(tip[1])),
                self._beam_parts[i]["wedge"].widths[0],
                label.toPlainText(),
                (float(label.anchor[0]), float(label.anchor[1])),
            ),
        }
        self._cancel_drag_shortcut.setEnabled(True)

    def _show_dragged_beam(self, gantry: float) -> None:
        """Draw the dragged beam at *gantry* degrees, its source on the orbit."""
        drag = self._beam_drag
        i = drag["index"]
        beam, parts = self._beams[i], self._beam_parts[i]
        drag["gantry"] = gantry
        angle = np.radians(gantry)
        u, w = (np.asarray(vector, dtype=float) for vector in beam["orbit"])
        on_orbit = drag["matrix"] @ np.array([np.cos(angle), np.sin(angle)])
        tip = drag["iso"] + drag["radius"] * on_orbit
        toward = self._toward_viewer(np.cos(angle) * u + np.sin(angle) * w)
        width = max(1.0, _BEAM_WIDTH_PX * _BEAM_PERSPECTIVE**toward)

        parts["wedge"].set_source((float(tip[0]), float(tip[1])), width)
        parts["source"].setData([float(tip[0])], [float(tip[1])])
        parts["source"].setSize(max(7.0, width))
        direction = on_orbit / max(float(np.linalg.norm(on_orbit)), 1e-12)
        label = parts["label"]
        label.setText(f"{beam.get('name', f'#{i}')}  {gantry:g}°")
        label.setAnchor((self._label_anchor(direction[0]), self._label_anchor(direction[1])))
        label.setPos(float(tip[0]), float(tip[1]))

    def _finish_beam_drag(self) -> None:
        drag = self._beam_drag
        self._cancel_beam_drag()
        beam = self._beams[drag["index"]]
        # Deferred: the plan update redraws the beams, removing the item whose
        # drag event is still being handled.
        QTimer.singleShot(0, partial(self._commit_beam_angle, beam, drag["gantry"]))

    def _commit_beam_angle(self, beam: dict, gantry: float) -> None:
        if abs((gantry - beam["gantry"] + 180.0) % 360.0 - 180.0) > 1e-9:
            self.beam_angle_changed.emit(int(beam["index"]), float(gantry))
        # Redraws from the beams as they now are: the updated plan, or the old
        # angle if nothing took the change.
        self._update_beams(self.slice_slider.value())

    def _cancel_beam_drag(self) -> None:
        self._cancel_drag_shortcut.setEnabled(False)
        if self._beam_drag is not None:
            self._view_box.removeItem(self._beam_drag["orbit"])
            self._beam_drag = None

    def _abort_beam_drag(self) -> None:
        """Escape: put the dragged beam back and ignore the rest of the drag."""
        drag = self._beam_drag
        if drag is None:
            return
        # Restored in place: the dragged item still receives the drag's events.
        tip, width, text, anchor = drag["restore"]
        parts = self._beam_parts[drag["index"]]
        parts["wedge"].set_source(tip, width)
        parts["source"].setData([tip[0]], [tip[1]])
        parts["source"].setSize(max(7.0, width))
        parts["label"].setText(text)
        parts["label"].setAnchor(anchor)
        parts["label"].setPos(*tip)
        self._cancel_beam_drag()

    def _update_isocenter(self, idx: int) -> None:
        if self._isocenter_item is not None:
            self._view_box.removeItem(self._isocenter_item)
            self._isocenter_item = None

        if not self._show_isocenter or self._isocenter is None:
            return

        axis = self._PLANE_MAP.get(self._plane, 2)
        iso_slice_idx = int(round(self._isocenter[axis]))

        if idx == iso_slice_idx:
            h_axis, v_axis = self._plane_axes()
            x, y = self._isocenter[h_axis], self._isocenter[v_axis]

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
        if not self._setting_slice:
            self._follow_isocenter = False
        self.update_slice()

    def _on_scroll_slice(self, delta: int) -> None:
        new_val = self.slice_slider.value() + delta
        new_val = max(self.slice_slider.minimum(), min(self.slice_slider.maximum(), new_val))
        self.slice_slider.setValue(new_val)

    def eventFilter(self, obj, ev):  # noqa: N802
        """Handle native gestures and keep the cursor-info overlay anchored."""
        if obj is self._plot_widget.viewport():
            if ev.type() == QEvent.Type.Resize:
                self._position_info_label()
            elif ev.type() == QEvent.Type.Leave:
                self._info_label.hide()
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

    def set_isocenter(self, isocenter: Optional[np.ndarray]) -> None:
        """Move the isocenter marker, in voxel space (z, x, y) like :meth:`set_data`."""
        unchanged = (isocenter is None and self._isocenter is None) or (
            isocenter is not None
            and self._isocenter is not None
            and np.array_equal(isocenter, self._isocenter)
        )
        self._isocenter = isocenter
        if unchanged:
            return
        if self._follow_isocenter and self._ct is not None and isocenter is not None:
            home = self._home_slice()
            if home != self.slice_slider.value():
                self._set_slice(home)  # renders the slice through _on_slider_changed
                return
        if self._show_isocenter:
            self.update_slice()

    def set_beams(self, beams: Sequence[dict]) -> None:
        """Assign the beam axes to draw.

        Parameters
        ----------
        beams : sequence of dict
            One entry per beam with ``iso_center`` (the beam's target point) and
            ``source_direction`` (pointing from the isocenter towards the
            source), both in the viewer's voxel space, plus an optional
            ``label`` used as the tooltip of the source marker.
        """
        self._beams = list(beams)
        # Nothing is on screen while the overlay is off, so skip the re-render;
        # the workspace pushes beams on every update, not only on plan changes.
        if self._show_beams:
            self.update_slice()

    def set_beams_visible(self, visible: bool) -> None:
        """Toggle the beam axis overlay."""
        self._show_beams = visible
        self.update_slice()

    def set_orientation_visible(self, visible: bool) -> None:
        """Toggle the patient orientation marker in the corner of the plot."""
        self._show_orientation = visible
        self._update_orientation_marker()

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
            self._set_slice(slice_idx)
            self._follow_isocenter = True
        self._view_box.autoRange()

    def reset_options(self) -> None:
        """Reset visualization options to defaults."""
        self._quantity_window = None
        self._ct_window = None
        self._quantity_colormap = "jet"
        self._ct_colormap = "gray"
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
