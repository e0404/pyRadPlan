"""Viewer settings widget."""

from __future__ import annotations

import logging
import warnings
from typing import Optional

import numpy as np
import SimpleITK as sitk

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QVBoxLayout,
    QWidget,
)

from pyRadPlan.cst import VOI
from pyRadPlan.ct import CT
from pyRadPlan.cst import StructureSet
from pyRadPlan.gui.windows._analysis_win import show_analysis
from pyRadPlan.gui.workspace import WorkspaceManager

# Local import (module is in the same folder)
from ._base import WorkspaceWidget
from .result.quantity_widget import QuantityWidget
from .result.visualization_widget import VisualizationWidget
from .result.viewer_options_widget import ViewerOptionsWidget
from .result.vois_widget import VOIsWidget

logger = logging.getLogger(__name__)


# --- Data derivation helpers (workspace objects -> viewer-ready arrays) -------


def _process_result(result: dict | np.ndarray | sitk.Image | None) -> dict | np.ndarray | None:
    """Convert a workspace ``result`` into viewer-ready (transposed) arrays."""
    if result is None:
        return None
    if isinstance(result, dict):
        return _process_dict_result(result)
    return _process_raw_result(result)


def _process_dict_result(result_dict: dict) -> dict:
    """Convert each 3-D value in a result dict into a transposed array (or list)."""
    quantities_dict: dict = {}
    for k, v in result_dict.items():
        if isinstance(v, sitk.Image):
            arr = sitk.GetArrayFromImage(v)
            if arr.ndim == 3:
                quantities_dict[k] = np.transpose(arr, (0, 2, 1))
        elif isinstance(v, np.ndarray) and v.ndim == 3:
            quantities_dict[k] = np.transpose(v, (0, 2, 1))
        elif isinstance(v, list):
            processed = _process_list_quantity(v)
            if processed:
                quantities_dict[k] = processed
    return quantities_dict


def _process_list_quantity(items: list) -> list:
    """Transpose each 3-D array in a list of images/arrays."""
    processed = []
    for item in items:
        if isinstance(item, sitk.Image):
            arr = sitk.GetArrayFromImage(item)
        elif isinstance(item, np.ndarray):
            arr = item
        else:
            continue
        if arr.ndim == 3:
            processed.append(np.transpose(arr, (0, 2, 1)))
    return processed


def _process_raw_result(result: object) -> np.ndarray:
    """Convert a single raw image or array to a transposed numpy array."""
    if isinstance(result, sitk.Image):
        result_arr = sitk.GetArrayFromImage(result)
    else:
        result_arr = np.asarray(result)
    return np.transpose(result_arr, (0, 2, 1))


def _build_overlay_meta(
    quantity_data: dict | np.ndarray | None,
) -> tuple[dict[str, str], dict[str, str]]:
    """Build overlay unit and label dicts from the processed quantity data."""
    quantity_meta: dict[str, tuple[str, str]] = {
        "physical_dose": ("Dose", "Gy"),
        "physical_dose_beam": ("Dose", "Gy"),
        "let": ("LET", "keV/µm"),
        "let_beam": ("LET", "keV/µm"),
        "effect": ("Effect", ""),
        "effect_beam": ("Effect", ""),
        "rbe_x_dose": ("RBE-weighted Dose", "Gy (RBE)"),
        "rbe_x_dose_beam": ("RBE-weighted Dose", "Gy (RBE)"),
        "alpha_dose": ("Alpha Dose", "Gy"),
        "alpha_dose_beam": ("Alpha Dose", "Gy"),
        "sqrt_beta_dose": ("Sqrt(Beta) Dose", "Gy\u00bd"),
        "sqrt_beta_dose_beam": ("Sqrt(Beta) Dose", "Gy\u00bd"),
        "let_dose": ("LET\u00b7Dose", "Gy\u00b7keV/\u00b5m"),
        "let_dose_beam": ("LET\u00b7Dose", "Gy\u00b7keV/\u00b5m"),
    }

    overlay_units: dict[str, str] = {}
    overlay_labels: dict[str, str] = {}
    if isinstance(quantity_data, dict):
        for k, v in quantity_data.items():
            base = k.split()[0] if " " in k else k
            qname, unit = quantity_meta.get(base, ("Quantity", ""))
            if isinstance(v, list):
                for i in range(len(v)):
                    expanded = f"{k} {i}"
                    overlay_units[expanded] = unit
                    overlay_labels[expanded] = qname
            else:
                overlay_units[k] = unit
                overlay_labels[k] = qname
    else:
        overlay_units = {"Physical quantity": "Gy"}
        overlay_labels = {"Physical quantity": "Dose"}

    return overlay_units, overlay_labels


def _compute_isocenter_vox(ct: CT, cst: Optional[StructureSet]) -> np.ndarray | None:
    """Compute isocenter voxel coordinates (z, x, y) from the CST target center of mass."""
    if cst is None or not hasattr(cst, "target_center_of_mass"):
        return None
    try:
        iso = cst.target_center_of_mass()
    except Exception:  # noqa: BLE001 - isocenter is optional
        return None
    if iso is None:
        return None

    origin = np.array(ct.cube_hu.GetOrigin())
    spacing = np.array(ct.cube_hu.GetSpacing())
    iso = np.array(iso).flatten()

    # Voxel index = (physical - origin) / spacing; all are (x, y, z)
    isocenter_vox_phys = (iso - origin) / spacing

    # Viewer array is (Z, X, Y) after transpose; isocenter_vox_phys is (x, y, z)
    return np.array([isocenter_vox_phys[2], isocenter_vox_phys[0], isocenter_vox_phys[1]])


class ViewingWidget(WorkspaceWidget):
    """Composite slice viewer bound to a :class:`WorkspaceManager`.

    Derives viewer-ready arrays from the workspace ``ct``, ``cst`` and ``result``
    objects and drives the embedded rendering widgets.  Provides:

    - Plane selection (Axial/Sagittal/Coronal), overlays and isolines
    - Window/level, colormap and opacity controls
    - Dynamic VOI list with visibility and color selection

    The viewer updates automatically whenever the workspace changes -- populate
    the :class:`WorkspaceManager` instead of calling the deprecated
    ``set_data``/``set_vois``/``set_masks`` shims directly.
    """

    # Signals to integrate with the rest of the app
    plane_changed = Signal(str)  # "Axial" | "Sagittal" | "Coronal"
    global_max_toggled = Signal(bool)
    isolines_toggled = Signal(bool)
    overlay_toggled = Signal(str, bool)  # overlay name, checked
    voi_toggled = Signal(str, bool)  # voi name, checked
    vois_selection_changed = Signal(list)  # list[str] of selected VOIs
    voi_metadata_changed = Signal(str)  # voi name whose metadata was edited

    # Only the data and computed result drive the display; pln/stf/dij are
    # upstream planning objects the viewer does not render.
    _watched_keys = ("ct", "cst", "result")

    def __init__(
        self,
        workspace: Optional[WorkspaceManager] = None,
        parent: Optional[QWidget] = None,
        build_layout: bool = True,
    ) -> None:
        super().__init__(workspace, parent)

        self._vois: list[VOI] = []
        self._has_data = False
        # Identity-based caches of derived (transposed) arrays.  The workspace
        # notifies on every write (e.g. per objective edit), but the underlying
        # images rarely change; converting a full CT / all VOI masks per
        # notification would block the GUI thread with large memcopies.
        self._ct_cache: Optional[tuple[object, np.ndarray]] = None
        self._mask_cache: dict[str, tuple[object, np.ndarray]] = {}
        self._quantity_cache: dict[str, tuple[object, object]] = {}

        # Create and wire the four sub-widgets.  Whether they are arranged here
        # (standalone use) or placed individually by a host window (e.g. the
        # MainWindow's matRad-style panels) is controlled by ``build_layout``.
        self.vis_widget = VisualizationWidget()
        self.vis_widget.overlay_toggled.connect(self.overlay_toggled)
        self.vis_widget.isolines_toggled.connect(self.isolines_toggled)
        self.vis_widget.isocenter_toggled.connect(self._on_isocenter_toggled)
        self.vis_widget.quantity_changed.connect(self._on_quantity_changed)
        self.vis_widget.isolines_set.connect(self._on_isolines_set)
        self.vis_widget.recenter_requested.connect(self._on_recenter)
        self.vis_widget.show_analysis_requested.connect(self._on_show_analysis)

        self.quantity_widget = QuantityWidget(self)

        self.opts_widget = ViewerOptionsWidget()
        self.opts_widget.mode_changed.connect(self._on_mode_changed)
        self.opts_widget.colormap_changed.connect(self._on_colormap_changed)
        self.opts_widget.window_level_changed.connect(self._on_window_level_changed)
        self.opts_widget.opacity_changed.connect(self._on_opacity_changed)
        self.opts_widget.reset_requested.connect(self._on_reset_options)
        self.opts_widget.local_range_requested.connect(self._on_use_local_min_max)

        self.vois_widget = VOIsWidget()
        self.vois_widget.voi_toggled.connect(self._on_voi_toggled)
        self.vois_widget.selection_changed.connect(self.vois_selection_changed)
        self.vois_widget.color_changed.connect(self._on_voi_color_changed)
        self.vois_widget.metadata_changed.connect(self.voi_metadata_changed)
        self.vois_widget.voi_replaced.connect(self._on_voi_replaced)

        # Wire viewer signals to quantity widget
        self.quantity_widget.connect_viewer_signals(self)
        self.quantity_widget.range_changed.connect(self._on_range_changed)

        if build_layout:
            self._build_default_layout()

        # First render from the (possibly already populated) workspace
        self.initialize()

    @staticmethod
    def _group(title: str, content: QWidget) -> QGroupBox:
        """Wrap *content* in a titled group box (the sub-widgets are bare)."""
        box = QGroupBox(title)
        lay = QVBoxLayout(box)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.addWidget(content)
        return box

    def _build_default_layout(self) -> None:
        """Arrange the sub-widgets in the standalone 3-panel layout."""
        root_layout = QHBoxLayout(self)

        # --- Left Panel: Visualization ---
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(6, 6, 6, 6)
        left_layout.addWidget(self._group("Visualization", self.vis_widget))
        left_layout.addStretch(1)
        root_layout.addWidget(left_panel, 0)

        # --- Center: Quantity Widget ---
        root_layout.addWidget(self.quantity_widget, 1)

        # --- Right Panel: Viewer Options + VOIs ---
        right_panel = QWidget()
        right_panel.setFixedWidth(320)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(6, 6, 6, 6)
        right_layout.addWidget(self._group("Viewer Options", self.opts_widget))
        right_layout.addWidget(self._group("VOIs", self.vois_widget))
        root_layout.addWidget(right_panel, 0)

    # --- Workspace-driven update --------------------------------------
    def _do_update(self, changed_keys: list) -> None:
        """Derive viewer arrays from the workspace and drive the child widgets."""
        ws = self.workspace
        ct = ws.ct
        if ct is None:
            # Workspace was cleared: drop the previous patient's display state
            # instead of keeping stale images/VOIs on screen.
            self._clear_data()
            return

        full = not changed_keys
        data_changed = full or "ct" in changed_keys or "result" in changed_keys
        cst_changed = full or "cst" in changed_keys or "ct" in changed_keys

        if data_changed:
            self._apply_ct_and_result(ct, ws.cst, ws.result)
            self._has_data = True

        if cst_changed:
            self._apply_vois(ws.cst)

    def _clear_data(self) -> None:
        self._has_data = False
        self._ct_cache = None
        self._quantity_cache.clear()
        self._mask_cache.clear()
        self._vois = []
        self.vois_widget.set_vois([])
        self.quantity_widget.clear_data()
        self.vis_widget.update_quantity_selector([], None)

    def _ct_array(self, ct: CT) -> np.ndarray:
        """Get the transposed CT array, reusing the cache for an unchanged image."""
        if self._ct_cache is not None and self._ct_cache[0] is ct.cube_hu:
            return self._ct_cache[1]
        ct_arr = np.transpose(sitk.GetArrayFromImage(ct.cube_hu), (0, 2, 1))
        self._ct_cache = (ct.cube_hu, ct_arr)
        return ct_arr

    def _process_result_cached(self, result) -> dict | np.ndarray | None:
        """Like :func:`_process_result`, reusing per-key conversions by identity.

        Result dicts routinely share value objects across updates (snapshots,
        carried-forward keys), so unchanged cubes are not converted again.
        """
        if not isinstance(result, dict):
            self._quantity_cache.clear()
            return _process_result(result)
        cache: dict[str, tuple[object, object]] = {}
        processed: dict = {}
        for key, value in result.items():
            cached = self._quantity_cache.get(key)
            if cached is not None and cached[0] is value:
                cache[key] = cached
            else:
                single = _process_dict_result({key: value})
                if key not in single:
                    continue
                cache[key] = (value, single[key])
            processed[key] = cache[key][1]
        self._quantity_cache = cache
        return processed

    def _apply_ct_and_result(self, ct: CT, cst, result) -> None:
        ct_arr = self._ct_array(ct)
        quantity_data = self._process_result_cached(result)
        overlay_units, overlay_labels = _build_overlay_meta(quantity_data)
        isocenter_vox = _compute_isocenter_vox(ct, cst)

        self.quantity_widget.set_ct_geometry(ct.cube_hu.GetOrigin(), ct.cube_hu.GetSpacing())
        self.quantity_widget.set_data(
            ct_arr,
            quantity_data,
            overlay_units,
            isocenter_vox,
            overlay_label=overlay_labels,
        )

        quantities = self.quantity_widget.get_available_quantities()
        active_quantity = None
        if quantities:
            for candidate in ("physical_dose", "Physical quantity", "physical_quantity"):
                if candidate in quantities:
                    active_quantity = candidate
                    break
            else:
                active_quantity = quantities[0]
            self.quantity_widget.set_active_quantity(active_quantity)

        self.vis_widget.update_quantity_selector(quantities, active_quantity)

        mode = "quantity" if self.opts_widget.cmap_mode_btn.isChecked() else "ct"
        self._sync_ui_to_mode(mode)

    def _apply_vois(self, cst: Optional[StructureSet]) -> None:
        vois = list(cst.vois) if cst is not None else []
        self._vois = vois
        self.vois_widget.set_vois(vois)
        self.quantity_widget.set_voi_colors(self.vois_widget.get_voi_colors())
        self.quantity_widget.set_masks({voi.name: self._mask_array(voi) for voi in vois})
        self._prune_mask_cache({voi.name for voi in vois})

    def _mask_array(self, voi: VOI) -> np.ndarray:
        """Get the transposed mask array, reusing the cache for an unchanged mask."""
        cached = self._mask_cache.get(voi.name)
        if cached is not None and cached[0] is voi.mask:
            return cached[1]
        arr = np.transpose(sitk.GetArrayFromImage(voi.mask), (0, 2, 1))
        self._mask_cache[voi.name] = (voi.mask, arr)
        return arr

    def _prune_mask_cache(self, names: set) -> None:
        for name in list(self._mask_cache):
            if name not in names:
                del self._mask_cache[name]

    def selected_vois(self) -> list[str]:
        """Return the list of currently checked VOIs by name."""
        return self.vois_widget.selected_vois()

    # --- Deprecated direct-population API (pre-workspace, kept as shims) ---
    def set_data(
        self,
        ct_volume,
        quantity_volume=None,
        overlay_unit="Gy",
        overlay_label=None,
        isocenter=None,
    ) -> None:
        """Forward data assignment to the internal quantity widget.

        .. deprecated:: 0.5
            Populate the :class:`~pyRadPlan.gui.workspace.WorkspaceManager`
            instead; the viewer derives its display from the workspace.
        """
        warnings.warn(
            "ViewingWidget.set_data is deprecated; populate the WorkspaceManager instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.quantity_widget.set_data(
            ct_volume, quantity_volume, overlay_unit, isocenter, overlay_label=overlay_label
        )
        quantities = self.quantity_widget.get_available_quantities()
        active_quantity = self.quantity_widget._active_quantity_name
        self.vis_widget.update_quantity_selector(quantities, active_quantity)
        mode = "quantity" if self.opts_widget.cmap_mode_btn.isChecked() else "ct"
        self._sync_ui_to_mode(mode)

    def set_masks(self, masks: dict[str, np.ndarray]) -> None:
        """Assign VOI masks for contour display.

        .. deprecated:: 0.5
            Populate the :class:`~pyRadPlan.gui.workspace.WorkspaceManager`
            instead; masks are derived from the workspace ``cst``.
        """
        warnings.warn(
            "ViewingWidget.set_masks is deprecated; populate the WorkspaceManager instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.quantity_widget.set_masks(masks)

    def set_vois(self, vois: list[VOI], checked=None) -> None:
        """Populate the VOIs section with checkboxes.

        .. deprecated:: 0.5
            Populate the :class:`~pyRadPlan.gui.workspace.WorkspaceManager`
            instead; VOIs are derived from the workspace ``cst``.
        """
        warnings.warn(
            "ViewingWidget.set_vois is deprecated; populate the WorkspaceManager instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._vois = list(vois)
        self.vois_widget.set_vois(self._vois, checked)
        self.quantity_widget.set_voi_colors(self.vois_widget.get_voi_colors())

    # --- Helpers ---
    def _on_voi_toggled(self, name: str, checked: bool) -> None:
        self.voi_toggled.emit(name, checked)
        # selection_changed is emitted by vois_widget directly

    def _on_voi_replaced(self, name: str, new_voi) -> None:
        """Write a recreated (type-changed) VOI back to the workspace ``cst``.

        Plain metadata edits mutate the shared VOI objects in place, but a type
        change produces a new instance that must replace the old one in the
        structure set. The write is wrapped in :meth:`hold_updates` so this
        widget does not rebuild (and reset the VOI selection) from its own
        change, while peer widgets refresh.
        """
        self._vois = [new_voi if v.name == name else v for v in self._vois]
        cst = self._ws.cst
        if cst is None:
            return
        try:
            cst.vois = [new_voi if v.name == name else v for v in cst.vois]
        except (ValueError, TypeError):
            logger.exception("Rejected VOI type change for %r", name)
            return
        with self.hold_updates():
            self._ws.cst = cst

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
