"""Viewer settings widget."""

from __future__ import annotations

from typing import Iterable, Optional
import numpy as np

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QVBoxLayout,
    QWidget,
)

from pyRadPlan.cst import VOI
from pyRadPlan.gui.windows._analysis_win import show_analysis

# Local import (module is in the same folder)
from .result.quantity_widget import QuantityWidget
from .result.visualization_widget import VisualizationWidget
from .result.viewer_options_widget import ViewerOptionsWidget
from .result.vois_widget import VOIsWidget


class ViewingWidget(QWidget):
    """Compact settings panel for the viewer with overlays and VOI selection.

    Provides:
    - Global max and isolines toggles
    - Plane selection (Axial/Sagittal/Coronal)
    - Overlay toggles (CT/CST/Quantity)
    - Dynamic VOIs list (variable number) with checkboxes

    Signals expose user choices for integration with the rest of the app.
    """

    # Signals to integrate with the rest of the app
    plane_changed = Signal(str)  # "Axial" | "Sagittal" | "Coronal"
    global_max_toggled = Signal(bool)
    isolines_toggled = Signal(bool)
    overlay_toggled = Signal(str, bool)  # overlay name, checked
    voi_toggled = Signal(str, bool)  # voi name, checked
    vois_selection_changed = Signal(list)  # list[str] of selected VOIs

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        self._vois: list[VOI] = []

        # Root layout: Left (Vis), Center (Image), Right (Options)
        root_layout = QHBoxLayout(self)
        self.setLayout(root_layout)

        # --- Left Panel: Visualization ---
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(6, 6, 6, 6)
        root_layout.addWidget(left_panel, 0)

        self.vis_widget = VisualizationWidget()
        left_layout.addWidget(self.vis_widget)

        # Connect Visualization signals
        self.vis_widget.overlay_toggled.connect(self.overlay_toggled)
        self.vis_widget.isolines_toggled.connect(self.isolines_toggled)
        self.vis_widget.isocenter_toggled.connect(self._on_isocenter_toggled)
        self.vis_widget.quantity_changed.connect(self._on_quantity_changed)
        self.vis_widget.isolines_set.connect(self._on_isolines_set)
        self.vis_widget.recenter_requested.connect(self._on_recenter)
        self.vis_widget.show_analysis_requested.connect(self._on_show_analysis)

        left_layout.addStretch(1)

        # --- Center: Quantity Widget ---
        self.quantity_widget = QuantityWidget(self)
        root_layout.addWidget(self.quantity_widget, 1)

        # --- Right Panel: Viewer Options + VOIs ---
        right_panel = QWidget()
        right_panel.setFixedWidth(320)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(6, 6, 6, 6)
        root_layout.addWidget(right_panel, 0)

        # Viewer Options
        self.opts_widget = ViewerOptionsWidget()
        right_layout.addWidget(self.opts_widget)

        # Connect Viewer Options signals
        self.opts_widget.mode_changed.connect(self._on_mode_changed)
        self.opts_widget.colormap_changed.connect(self._on_colormap_changed)
        self.opts_widget.window_level_changed.connect(self._on_window_level_changed)
        self.opts_widget.opacity_changed.connect(self._on_opacity_changed)
        self.opts_widget.reset_requested.connect(self._on_reset_options)
        self.opts_widget.local_range_requested.connect(self._on_use_local_min_max)

        # VOIs
        self.vois_widget = VOIsWidget()
        right_layout.addWidget(self.vois_widget)

        # Connect VOIs signals
        self.vois_widget.voi_toggled.connect(self._on_voi_toggled)
        self.vois_widget.selection_changed.connect(self.vois_selection_changed)
        self.vois_widget.color_changed.connect(self._on_voi_color_changed)

        # Wire viewer signals to quantity widget
        self.quantity_widget.connect_viewer_signals(self)
        self.quantity_widget.range_changed.connect(self._on_range_changed)

        # Initialize UI state
        # Don't call _sync_ui_to_mode here to avoid overwriting initial state before data is loaded
        # self._sync_ui_to_mode("quantity")
        # self.cmap_mode_btn.setText("Quantity") # Handled in ViewerOptionsWidget default

    # Convenience pass-through API -------------------------------------
    def set_data(
        self,
        ct_volume,
        quantity_volume=None,
        overlay_unit="Gy",
        overlay_label=None,
        isocenter=None,
    ) -> None:  # noqa: D401
        """Forward data assignment to internal quantity widget."""
        self.quantity_widget.set_data(
            ct_volume, quantity_volume, overlay_unit, isocenter, overlay_label=overlay_label
        )

        # Populate quantity selector
        quantities = self.quantity_widget.get_available_quantities()
        active_quantity = None
        if quantities:
            if "Physical quantity" in quantities:
                active_quantity = "Physical quantity"
            elif "physical_quantity" in quantities:
                active_quantity = "physical_quantity"
            else:
                active_quantity = quantities[0]

        self.vis_widget.update_quantity_selector(quantities, active_quantity)

        # Update UI limits based on new data
        # We need to know the current mode from opts_widget
        mode = "quantity" if self.opts_widget.cmap_mode_btn.isChecked() else "ct"
        self._sync_ui_to_mode(mode)

    def set_masks(self, masks: dict[str, np.ndarray]) -> None:
        """Assign VOI masks to underlying quantity widget for contour display."""
        self.quantity_widget.set_masks(masks)

    # --- Public API for VOIs ---
    def set_vois(self, vois: list[VOI], checked: Optional[Iterable[str]] = None) -> None:
        """Populate the VOIs section with a variable number of checkboxes."""
        self._vois = list(vois)
        self.vois_widget.set_vois(vois, checked)

        # Propagate colors to quantity widget for contour drawing
        if hasattr(self.quantity_widget, "set_voi_colors"):
            self.quantity_widget.set_voi_colors(self.vois_widget.get_voi_colors())

    def selected_vois(self) -> list[str]:
        """Return the list of currently checked VOIs by name."""
        return self.vois_widget.selected_vois()

    # --- Helpers ---
    def _on_voi_toggled(self, name: str, checked: bool) -> None:
        self.voi_toggled.emit(name, checked)
        # selection_changed is emitted by vois_widget directly

    def _on_voi_color_changed(self, name: str, rgb: tuple) -> None:
        """Re-propagate updated colors to quantity widget for contour redraw."""
        if hasattr(self.quantity_widget, "set_voi_colors"):
            self.quantity_widget.set_voi_colors(self.vois_widget.get_voi_colors())

    def set_plane(self, plane: str) -> None:
        """Programmatically set the active plane by its display name."""
        self.quantity_widget.set_plane(plane)

    def _on_isocenter_toggled(self, checked: bool) -> None:
        self.quantity_widget.set_isocenter_visible(checked)

    def _on_isolines_set(self, levels: list[float]) -> None:
        self.quantity_widget.set_isolines(levels)

    def _on_recenter(self) -> None:
        self.quantity_widget.recenter_to_isocenter()

    def _on_quantity_changed(self, name: str) -> None:
        """Update the active quantity in the widget."""
        self.quantity_widget.set_active_quantity(name)
        # Only update UI limits if we are already in quantity/Quantity mode
        if self.opts_widget.cmap_mode_btn.isChecked():
            # Just update the range limits
            dmin, dmax = self.quantity_widget.get_data_range("quantity")
            # We need to update the opts widget limits, but not reset everything
            # The sync_ui method does a lot, maybe too much?
            # Let's reuse sync_ui for now as it sets ranges based on data
            wl = self.quantity_widget.get_window_level("quantity")
            cmap = self.quantity_widget.get_colormap("quantity")
            self.opts_widget.sync_ui("quantity", (dmin, dmax), wl, cmap)

    def _on_mode_changed(self, mode: str) -> None:
        # Switch active mode in quantity widget
        self.quantity_widget.set_active_mode(mode)
        # Sync UI controls
        self._sync_ui_to_mode(mode)

    def _on_colormap_changed(self, name: str, mode: str) -> None:
        self.quantity_widget.set_colormap(name, mode)

    def _on_window_level_changed(self, center: float, width: float, mode: str) -> None:
        self.quantity_widget.set_window_level(center, width, mode)

    def _on_opacity_changed(self, value: float) -> None:
        self.quantity_widget.set_opacity(value)

    def _sync_ui_to_mode(self, mode: str) -> None:
        """Update UI controls to reflect settings for the given mode."""
        # Get data range for limits
        dmin, dmax = self.quantity_widget.get_data_range(mode)
        wl = self.quantity_widget.get_window_level(mode)
        cmap = self.quantity_widget.get_colormap(mode)

        self.opts_widget.sync_ui(mode, (dmin, dmax), wl, cmap)

    def _on_range_changed(self, min_val: float, max_val: float) -> None:
        # This signal comes from QuantityWidget with slice data range.
        pass

    def _on_reset_options(self) -> None:
        """Reset all viewer options to defaults."""
        # Reset quantity widget state
        self.quantity_widget.reset_options()

        # Reset UI controls
        self.opts_widget.reset_ui()
        self.quantity_widget.set_active_mode("quantity")

        self._sync_ui_to_mode("quantity")

    def _on_use_local_min_max(self) -> None:
        """Set range to min/max of current data."""
        mode = "quantity" if self.opts_widget.cmap_mode_btn.isChecked() else "ct"
        dmin, dmax = self.quantity_widget.get_current_slice_range(mode)

        # Update range spinboxes in opts widget
        self.opts_widget.set_range_values(dmin, dmax)

    def _on_show_analysis(self) -> None:
        """Show the Analysis window with all quantities and masks."""
        quantities = self.quantity_widget._quantities
        if not quantities:
            return

        masks = self.quantity_widget._masks
        if not masks:
            return

        initial_quantity = self.quantity_widget._active_quantity_name or ""
        initial_vois = list(self.selected_vois())
        voi_colors = self.vois_widget.get_voi_colors()
        voi_types = {v.name: getattr(v, "voi_type", "") for v in self._vois}

        self._analysis_window = show_analysis(
            quantities=quantities,
            masks=masks,
            parent=self,
            overlay={"voi_colors": voi_colors},
            overlay_units=self.quantity_widget._overlay_units,
            overlay_labels=self.quantity_widget._overlay_labels,
            initial_quantity=initial_quantity,
            initial_vois=initial_vois,
            voi_types=voi_types,
        )
