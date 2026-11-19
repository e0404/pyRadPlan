"""Viewer settings widget."""

from __future__ import annotations

import logging
import warnings
from typing import Callable, Optional, Sequence

import numpy as np
import SimpleITK as sitk

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QVBoxLayout,
    QWidget,
)

from pyRadPlan.app import PreconditionError
from pyRadPlan.app._results import quantity_meta, scenario_labels
from pyRadPlan.app._views import beam_geometry, plan_beam_angles, plan_iso_centers
from pyRadPlan.app._views import default_quantity as default_quantity_key
from pyRadPlan.cst import VOI
from pyRadPlan.ct import CT
from pyRadPlan.cst import StructureSet
from pyRadPlan.gui.windows._analysis_win import show_analysis
from pyRadPlan.gui.workspace import WorkspaceManager

# Local import (module is in the same folder)
from ._base import WorkspaceWidget
from pyRadPlan.biology._editing import load_machine_quietly, plan_dose_engine
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
    """Transpose each 3-D array in a list of images/arrays.

    Other entries (e.g. ``None`` for scenarios without a result) become ``None``, so that
    list positions keep their meaning (beam or flat scenario index).  A list without any
    3-D entry yields ``[]``.
    """
    processed = []
    for item in items:
        if isinstance(item, sitk.Image):
            arr = sitk.GetArrayFromImage(item)
        elif isinstance(item, np.ndarray):
            arr = item
        else:
            arr = None
        processed.append(
            np.transpose(arr, (0, 2, 1)) if arr is not None and arr.ndim == 3 else None
        )
    return processed if any(p is not None for p in processed) else []


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

    overlay_units: dict[str, str] = {}
    overlay_labels: dict[str, str] = {}
    if isinstance(quantity_data, dict):
        for k, v in quantity_data.items():
            qname, unit = quantity_meta(k.split()[0] if " " in k else k)
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


def _isocenter_vox(ct: CT, iso_centers: np.ndarray | None) -> np.ndarray | None:
    """Viewer voxel coordinates (z, x, y) of the isocenter marker (the first beam's)."""
    if iso_centers is None:
        return None
    origin = np.array(ct.cube_hu.GetOrigin())
    spacing = np.array(ct.cube_hu.GetSpacing())
    return _to_viewer_vox(iso_centers[0], origin, spacing)


def _to_viewer_vox(point_lps: np.ndarray, origin: np.ndarray, spacing: np.ndarray) -> np.ndarray:
    """Convert an LPS point in mm to the viewer's (z, x, y) voxel coordinates."""
    vox = (np.asarray(point_lps, dtype=float).ravel() - origin) / spacing
    return np.array([vox[2], vox[0], vox[1]])


def _to_viewer_direction(direction_lps: np.ndarray, spacing: np.ndarray) -> np.ndarray:
    """Convert an LPS direction to (z, x, y) voxel space (anisotropic spacing)."""
    vec = np.asarray(direction_lps, dtype=float).ravel() / spacing
    return np.array([vec[2], vec[0], vec[1]])


def _viewer_beams(ct: CT, geometry: dict, edited: bool = False) -> list[dict]:
    """Beam axes of :func:`pyRadPlan.app._views.beam_geometry` in the viewer's voxel space.

    Beams with plan angles carry their ``index`` in the plan, ``gantry``/``couch``
    angles and gantry ``orbit`` ``(u, w)`` with its ``orbit_axis``, so the viewer can
    rotate them; *edited* adds the ``pending`` flags of drawn but unapplied angles.
    """
    origin = np.array(ct.cube_hu.GetOrigin())
    spacing = np.array(ct.cube_hu.GetSpacing())
    beams = []
    for entry in geometry["beams"]:
        beam = {
            "iso_center": _to_viewer_vox(entry["iso_center"], origin, spacing),
            "source_direction": _to_viewer_direction(entry["source_direction"], spacing),
            "name": entry["name"],
            "label": entry["label"],
        }
        if "orbit_axis" in entry:
            beam.update(
                index=entry["index"],
                gantry=entry["gantry"],
                couch=entry["couch"],
                orbit=(
                    _to_viewer_direction(entry["orbit_u"], spacing),
                    _to_viewer_direction(entry["orbit_w"], spacing),
                ),
                orbit_axis=_to_viewer_direction(entry["orbit_axis"], spacing),
            )
        if edited:
            beam["pending"] = entry["pending"]
        beams.append(beam)
    return beams


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

    # Only the data, the computed result and the beam geometry drive the
    # display; of the dij, only its scenario labels are shown.
    _watched_keys = ("ct", "cst", "pln", "stf", "dij", "result")

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
        #: Beam angles edited in a plan form but not applied yet (see set_beam_editor).
        self._edited_angles: Optional[tuple[list[float], list[float]]] = None
        self._beam_editor: Optional[Callable[[int, float], None]] = None
        #: Labels of the workspace dij's scenarios, by flat scenario index.
        self._scenario_labels: list[str] = []

        # Create and wire the four sub-widgets.  Whether they are arranged here
        # (standalone use) or placed individually by a host window (e.g. the
        # MainWindow's matRad-style panels) is controlled by ``build_layout``.
        self.vis_widget = VisualizationWidget()
        self.vis_widget.overlay_toggled.connect(self.overlay_toggled)
        self.vis_widget.isolines_toggled.connect(self.isolines_toggled)
        self.vis_widget.isocenter_toggled.connect(self._on_isocenter_toggled)
        self.vis_widget.beams_toggled.connect(self._on_beams_toggled)
        self.vis_widget.orientation_toggled.connect(self._on_orientation_toggled)
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
        self.vois_widget.set_machine_provider(self._plan_machine)
        self.vois_widget.set_bio_model_provider(self._plan_bio_model)
        self.vois_widget.set_engine_provider(self._plan_engine)
        self.vois_widget.set_color_writer(self._write_voi_color)
        self.vois_widget.voi_toggled.connect(self._on_voi_toggled)
        self.vois_widget.selection_changed.connect(self.vois_selection_changed)
        self.vois_widget.color_changed.connect(self._on_voi_color_changed)
        self.vois_widget.metadata_changed.connect(self.voi_metadata_changed)
        self.vois_widget.voi_replaced.connect(self._on_voi_replaced)

        # Wire viewer signals to quantity widget
        self.quantity_widget.connect_viewer_signals(self)
        self.quantity_widget.range_changed.connect(self._on_range_changed)
        self.quantity_widget.beam_angle_changed.connect(self._on_beam_angle_changed)

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
        full = not changed_keys
        if full or "dij" in changed_keys or "result" in changed_keys:
            self._apply_scenario_labels(ws.dij, ws.result)
        ct = ws.ct
        if ct is None:
            # Workspace was cleared: drop the previous patient's display state
            # instead of keeping stale images/VOIs on screen.
            self._clear_data()
            return

        data_changed = full or "ct" in changed_keys or "result" in changed_keys
        cst_changed = full or "cst" in changed_keys or "ct" in changed_keys
        # The beam axes are anchored to the (target-derived) isocenter, so a new
        # structure set moves them as much as a new plan does.
        beams_changed = cst_changed or "pln" in changed_keys or "stf" in changed_keys

        if data_changed:
            self._apply_ct_and_result(ct, ws.cst, ws.result)
            self._has_data = True

        if cst_changed:
            self._apply_vois(ws.cst)

        if data_changed or beams_changed:
            self._apply_beams(ct, ws.cst, ws.pln, ws.stf)

    def _apply_scenario_labels(self, dij, result=None) -> None:
        labels = scenario_labels(result, dij)
        self._scenario_labels = labels
        self.vis_widget.set_scenario_labels(labels)

    def _clear_data(self) -> None:
        self._has_data = False
        self._ct_cache = None
        self._quantity_cache.clear()
        self._mask_cache.clear()
        self._vois = []
        self.vois_widget.set_vois([])
        self.quantity_widget.clear_data()
        self.vis_widget.update_quantity_selector([], None)
        self.vis_widget.set_beams_available(False)

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
        isocenter_vox = _isocenter_vox(ct, plan_iso_centers(self.workspace.pln, cst))

        self.quantity_widget.set_ct_geometry(
            ct.cube_hu.GetOrigin(), ct.cube_hu.GetSpacing(), ct.cube_hu.GetDirection()
        )
        self.quantity_widget.set_data(
            ct_arr,
            quantity_data,
            overlay_units,
            isocenter_vox,
            overlay_label=overlay_labels,
        )

        quantities = self.quantity_widget.get_available_quantities()
        active_quantity = default_quantity_key(quantities)
        if active_quantity is not None:
            self.quantity_widget.set_active_quantity(active_quantity)

        self.vis_widget.update_quantity_selector(quantities, active_quantity)

        mode = "quantity" if self.opts_widget.cmap_mode_btn.isChecked() else "ct"
        self._sync_ui_to_mode(mode)

    def _plan_machine(self) -> Optional[object]:
        """
        Machine base data of the workspace plan, whose tissue classes the editor offers.

        None if there is no plan or the base data cannot be loaded. Resolved on demand,
        since this widget does not watch the plan; a machine named by the plan is reduced
        to what the editor reads off it, see
        :func:`~pyRadPlan.gui.widgets._tissue_params.load_machine_quietly`.
        """
        machine = getattr(self._ws.pln, "machine", None)
        if machine is None:
            return None
        if not isinstance(machine, str):
            return machine  # the plan already carries the loaded machine
        return load_machine_quietly(getattr(self._ws.pln, "radiation_mode", ""), machine)

    def _plan_bio_model(self) -> Optional[object]:
        """Biological model of the workspace plan, whose own tissue classes the editor offers."""
        return getattr(self._ws.pln, "bio_model", None)

    def _plan_engine(self) -> Optional[type]:
        """Dose engine class of the workspace plan (its own or the default), None if none."""
        return plan_dose_engine(self._ws.pln)

    def _apply_beams(self, ct: CT, cst, pln, stf) -> None:
        """Hand the beam axes to the viewer and enable the toggle when there are any.

        Beam angles edited but not applied yet are drawn in place of the plan's,
        each changed beam flagged ``pending``.
        """
        iso_centers = plan_iso_centers(pln, cst)
        self.quantity_widget.set_isocenter(_isocenter_vox(ct, iso_centers))
        edited = self._edited_angles if pln is not None else None
        draft = None
        if edited is not None:
            draft = {"gantry_angles": list(edited[0]), "couch_angles": list(edited[1])}
        try:
            geometry = beam_geometry(pln, stf, cst, draft, iso_centers=iso_centers)
            beams = _viewer_beams(ct, geometry, edited=draft is not None)
        except Exception:  # noqa: BLE001 - the overlay must never break the viewer
            logger.warning("Could not derive the beam geometry for the viewer", exc_info=True)
            beams = []
        self.quantity_widget.set_beams(beams)
        self.vis_widget.set_beams_available(bool(beams))

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

    def _on_voi_replaced(self, old_voi, new_voi) -> None:
        """Write the metadata edited in the VOI editor to the workspace ``cst``.

        The edited type, overlap priority and tissue go through
        :meth:`~pyRadPlan.app.Session.update_structure` (the editor has already asked
        about tissue errors), addressing the structure by its position, which recreates
        the VOI in the structure set; the stored VOI then replaces the editor's in the VOI
        list. While a calculation runs, the session refuses structure edits, so the edit is
        applied once the calculation has ended, unless another structure set has been
        loaded meanwhile.
        The write is wrapped in :meth:`hold_updates` so this widget does not rebuild (and
        reset the VOI selection) from its own change, while peer widgets refresh.
        """
        name = old_voi.name
        position = next((i for i, v in enumerate(self._vois) if v is old_voi), None)
        self._vois = [new_voi if v is old_voi else v for v in self._vois]
        changes = {
            "type": new_voi.voi_type,
            "overlap_priority": new_voi.overlap_priority,
            "tissue": new_voi.tissue_properties,
        }
        queued_cst = self._ws.cst

        def apply() -> None:
            cst = self._ws.cst
            if cst is None or cst is not queued_cst or position is None:
                logger.warning("Dropped the edit of structure %r: the structure set changed", name)
                return
            if not 0 <= position < len(cst.vois) or cst.vois[position].name != name:
                logger.warning("Dropped the edit of structure %r: it is no longer there", name)
                return
            try:
                with self.hold_updates():
                    self._ws.session.update_structure(
                        position, changes, apply_anyway=True, detail=False
                    )
            except (PreconditionError, LookupError, ValueError, TypeError):
                logger.exception("Rejected VOI type change for %r", name)
                return
            stored = cst.vois[position]
            self._vois = [stored if v is new_voi else v for v in self._vois]
            self.vois_widget.replace_voi(name, stored)

        self._ws.session.call_when_idle(apply)

    def _write_voi_color(self, voi: VOI, rgb: tuple) -> None:
        """Set a picked colour through the session (a display edit, also during a run)."""
        cst = self._ws.cst
        position = next((i for i, v in enumerate(self._vois) if v is voi), None)
        if (
            cst is not None
            and position is not None
            and position < len(cst.vois)
            and cst.vois[position].name == voi.name
        ):
            self._ws.session.update_structure(position, {"color": rgb}, detail=False)
        else:
            voi.color = rgb

    def _on_voi_color_changed(self, name: str, rgb: tuple) -> None:
        """Re-propagate updated colors to quantity widget for contour redraw."""
        if hasattr(self.quantity_widget, "set_voi_colors"):
            self.quantity_widget.set_voi_colors(self.vois_widget.get_voi_colors())

    def set_plane(self, plane: str) -> None:
        """Programmatically set the active plane by its display name."""
        self.quantity_widget.set_plane(plane)

    def _on_isocenter_toggled(self, checked: bool) -> None:
        self.quantity_widget.set_isocenter_visible(checked)

    def _on_beams_toggled(self, checked: bool) -> None:
        self.quantity_widget.set_beams_visible(checked)

    def _on_orientation_toggled(self, checked: bool) -> None:
        self.quantity_widget.set_orientation_visible(checked)

    def set_beam_editor(self, editor: Optional[Callable[[int, float], None]]) -> None:
        """Route dragged beam angles to *editor* instead of writing them to the plan.

        *editor* is called with the beam's index and new gantry angle, e.g. a plan
        form's ``set_gantry_angle``, so a drag becomes an edit that is applied with
        the form.  Pass the form's edited angles to :meth:`set_edited_beam_angles`
        to draw them.  ``None`` (the default) writes drags straight to the plan.
        """
        self._beam_editor = editor

    def set_edited_beam_angles(
        self, angles: Optional[tuple[Sequence[float], Sequence[float]]]
    ) -> None:
        """Draw the beams at these ``(gantry, couch)`` angles until they are applied.

        ``None`` draws the plan's beams.
        """
        self._edited_angles = (
            None
            if angles is None
            else ([float(a) for a in angles[0]], [float(a) for a in angles[1]])
        )
        ct = self._ws.ct
        if ct is not None:
            self._apply_beams(ct, self._ws.cst, self._ws.pln, self._ws.stf)

    def _on_beam_angle_changed(self, index: int, gantry: float) -> None:
        """Hand a beam's dragged gantry angle to the beam editor, or write it to the plan."""
        if self._beam_editor is not None:
            self._beam_editor(index, gantry)
            return
        pln = self._ws.pln
        angles = plan_beam_angles(pln)
        if angles is None or not 0 <= index < angles[0].size:
            return
        gantry_angles = [float(angle) for angle in angles[0]]
        gantry_angles[index] = float(gantry)
        prop_stf = {**pln.prop_stf, "gantry_angles": gantry_angles}
        self._ws.pln = pln.model_copy(update={"prop_stf": prop_stf})

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
            scenario_labels=self._scenario_labels,
        )
