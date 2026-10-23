"""DVH and QI analysis widget."""

from __future__ import annotations

import logging
from functools import partial
from typing import Any, Optional

import numpy as np

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from pyRadPlan.gui.widgets._base import format_number_list, parse_number_list
from pyRadPlan.gui.widgets._quantity_selector import QuantitySelector
from pyRadPlan.gui.widgets.analysis._dvh import DVHPlotWidget
from pyRadPlan.gui.widgets.analysis._gamma import GammaWidget
from pyRadPlan.gui.widgets.analysis._qi import QITableWidget
from pyRadPlan.gui.widgets.analysis._qi_point_dialog import PointKind, QIPointDialog
from pyRadPlan.analysis._units import safe_unit
from pyRadPlan.gui.widgets.result._labels import TruncatedCheckBox
from pyRadPlan.analysis import DEFAULT_REF_VOLS, QICollection, format_unit_symbol
from pyRadPlan.analysis._dvh import DVH, ureg

logger = logging.getLogger(__name__)

_TYPE_GROUPS: tuple[tuple[str, str], ...] = (
    ("TARGET", "Targets"),
    ("OAR", "OARs"),
    ("EXTERNAL", "External"),
    ("HELPER", "Helpers"),
)
_OTHER_GROUP = ("OTHER", "Other")

_GROUPBOX_STYLE = (
    "QGroupBox { font-size: 9pt; font-weight: 600; margin-top: 6px; "
    "border: 1px solid palette(mid); border-radius: 4px; padding: 4px 6px 4px 6px; }"
    "QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; }"
)

_NONE_LABEL = "— None —"


class AnalysisWidget(QWidget):
    """Widget displaying DVH plot, QI table, and Gamma analysis.

    Data is set via :meth:`set_data`. The primary quantity, chosen in the
    primary dropdown, drives the solid DVH curves and the QI table. An optional
    secondary quantity from the compare dropdown adds dashed DVH curves for
    comparison, on a second x axis when its unit is incompatible. DVH curves and
    quality indicators are computed on demand and cached, so switching back to a
    quantity or toggling structures only re-renders from the cache.

    The DVH plot and the QI table are linked: selecting a table row or clicking a
    curve highlights the structure in both, and selecting a D_x / V_x cell or
    clicking its marker also shows the point with its value and parameter lines.
    Double-clicking the plot or dragging a highlighted parameter line proposes a
    new D_x / V_x point, which is added to the QI parameters after confirmation.
    """

    # Emitted when a VOI color is changed inside this widget
    color_changed = Signal(str, tuple)  # voi name, new RGB tuple

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        # --- Internal state ---
        self._quantities: dict[str, np.ndarray] = {}
        self._masks: dict[str, np.ndarray] = {}
        self._voi_types: dict[str, str] = {}
        self._voi_colors: dict[str, tuple[int, int, int]] = {}
        self._overlay_units: dict[str, str] = {}
        self._overlay_labels: dict[str, str] = {}
        self._dvh_cache: dict[tuple[str, str], DVH] = {}  # (qty_name, voi_name)
        # Keyed by (qty_name, params signature); filled lazily for the shown
        # quantity only, since the D_x/V_x parameters are user-editable.
        self._qi_cache: dict[tuple[str, str], QICollection] = {}

        # Per-VOI UI state (populated by set_data)
        self._voi_checkboxes: dict[str, TruncatedCheckBox] = {}
        self._voi_color_swatches: dict[str, QPushButton] = {}

        # Metric selection UI state, plus the last computed QI collection so
        # toggling a metric re-renders the table without recomputing it.
        self._metric_checkboxes: dict[str, TruncatedCheckBox] = {}
        self._available_metrics: list[str] = []
        self._syncing_metrics = False
        self._qi_collection: QICollection | None = None
        self._qi_structures: list[str] = []

        #: "Show in DVH" checkbox of the D_x ("dx") and V_x ("vx") points.
        self.show_markers_checkboxes: dict[str, QCheckBox] = {}

        # V_x text typed by the user per quantity. V_x values carry the unit of
        # the shown quantity, so they cannot be shared like the D_x percentages.
        self._shown_quantity = ""
        self._ref_doses_by_quantity: dict[str, str] = {}
        self._synced_ref_doses_text = ""

        # --- Main layout ---
        layout = QVBoxLayout(self)

        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        # ---- Tab 1: DVH & QI ----
        self.dvh_qi_widget = QWidget()
        dvh_qi_layout = QVBoxLayout(self.dvh_qi_widget)
        dvh_qi_layout.setContentsMargins(4, 4, 4, 4)

        # Controls panel (quantity left | structures right)
        self._controls_panel = self._build_controls_panel()
        dvh_qi_layout.addWidget(self._controls_panel)

        splitter = QSplitter(Qt.Orientation.Vertical)
        dvh_qi_layout.addWidget(splitter)

        self.dvh_widget = DVHPlotWidget()
        self.qi_widget = QITableWidget()

        splitter.addWidget(self.dvh_widget)
        splitter.addWidget(self.qi_widget)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 1)

        for kind, checkbox in self.show_markers_checkboxes.items():
            checkbox.toggled.connect(partial(self.dvh_widget.set_markers_visible, kind))
        self.qi_widget.cell_selected.connect(self._on_qi_cell_selected)
        self.dvh_widget.structure_clicked.connect(self._on_structure_clicked)
        self.dvh_widget.marker_clicked.connect(self._on_marker_clicked)
        # Deferred, so the dialogs do not open inside the plot's mouse event handling.
        self.dvh_widget.point_requested.connect(
            lambda dose, volume: QTimer.singleShot(0, partial(self._add_point, dose, volume))
        )
        self.dvh_widget.parameter_moved.connect(
            lambda structure, metric, parameter: QTimer.singleShot(
                0, partial(self._move_point, structure, metric, parameter)
            )
        )

        self.tabs.addTab(self.dvh_qi_widget, "DVH / QI")

        # ---- Tab 2: Gamma ----
        self.gamma_widget = GammaWidget()
        self.tabs.addTab(self.gamma_widget, "Gamma Analysis")

    # ------------------------------------------------------------------
    # Controls panel construction
    # ------------------------------------------------------------------

    def _build_controls_panel(self) -> QWidget:
        """Build the side-by-side Quantity | Structures | QI parameters controls panel."""
        panel = QWidget()
        panel_layout = QHBoxLayout(panel)
        panel_layout.setContentsMargins(0, 0, 0, 0)
        panel_layout.setSpacing(8)

        # ---- Left: Quantities ----
        qty_group = QGroupBox("Quantities")
        form = QFormLayout()
        form.setContentsMargins(8, 8, 8, 8)
        form.setSpacing(6)
        qty_group.setLayout(form)

        self.quantity_selector = QuantitySelector()
        self.quantity_selector.setToolTip(
            "Quantity shown in the DVH plot (solid) and the QI table"
        )
        form.addRow(QLabel("Primary:"), self.quantity_selector)

        self.secondary_selector = QuantitySelector(none_label=_NONE_LABEL)
        self.secondary_selector.setToolTip(
            "Optional second quantity plotted as dashed DVH curves; "
            "incompatible units get a second x axis"
        )
        form.addRow(QLabel("Compare:"), self.secondary_selector)

        panel_layout.addWidget(qty_group, 1)

        # ---- Middle: Structures ----
        voi_group = QGroupBox("Structures")
        voi_outer = QVBoxLayout()
        voi_outer.setContentsMargins(6, 6, 6, 6)
        voi_outer.setSpacing(4)
        voi_group.setLayout(voi_outer)

        # Scroll area — same pattern as VOIsWidget in the main viewer
        self._vois_scroll = QScrollArea()
        self._vois_scroll.setWidgetResizable(True)
        self._vois_scroll.setMinimumHeight(120)
        self._vois_scroll.setMaximumHeight(260)
        self._vois_scroll.setMinimumWidth(320)
        self._vois_container = QWidget()
        self._vois_layout = QVBoxLayout(self._vois_container)
        self._vois_layout.setContentsMargins(2, 2, 2, 2)
        self._vois_layout.setSpacing(4)
        self._vois_scroll.setWidget(self._vois_container)
        voi_outer.addWidget(self._vois_scroll)

        # All / None quick-select buttons
        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(0, 0, 0, 0)
        all_btn = QPushButton("All")
        all_btn.setFixedWidth(48)
        none_btn = QPushButton("None")
        none_btn.setFixedWidth(48)
        all_btn.clicked.connect(self._select_all_vois)
        none_btn.clicked.connect(self._deselect_all_vois)
        btn_row.addWidget(all_btn)
        btn_row.addWidget(none_btn)
        btn_row.addStretch(1)
        voi_outer.addLayout(btn_row)

        panel_layout.addWidget(voi_group, 2)

        # ---- Right: QI parameters ----
        panel_layout.addWidget(self._build_qi_params_group(), 1)

        self.quantity_selector.key_changed.connect(self._replot)
        self.secondary_selector.key_changed.connect(self._replot)

        return panel

    def _build_qi_params_group(self) -> QGroupBox:
        """Build the D_x / V_x inputs and the metric selection for the QI table."""
        group = QGroupBox("QI parameters")
        outer = QVBoxLayout()
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(4)
        group.setLayout(outer)

        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(6)

        self.ref_vols_edit = QLineEdit(format_number_list(DEFAULT_REF_VOLS))
        self.ref_vols_edit.setToolTip(
            "Reference volumes in % for D_x metrics, e.g. '2 50 98'.\n"
            "Invalid input falls back to the defaults."
        )
        form.addRow(QLabel("D_x [%]:"), self._with_markers_checkbox("dx", self.ref_vols_edit))

        self.ref_doses_edit = QLineEdit()
        self.ref_doses_edit.setToolTip(
            "Reference doses for V_x metrics, e.g. '10 20 30'.\n"
            "Clear the field to re-derive five values from the maximum."
        )
        self._ref_doses_label = QLabel("V_x:")
        form.addRow(self._ref_doses_label, self._with_markers_checkbox("vx", self.ref_doses_edit))
        outer.addLayout(form)

        # Metric selection — same scroll + checkbox pattern as the Structures box
        self._metrics_scroll = QScrollArea()
        self._metrics_scroll.setWidgetResizable(True)
        self._metrics_scroll.setMinimumHeight(115)
        self._metrics_container = QWidget()
        self._metrics_layout = QGridLayout(self._metrics_container)
        self._metrics_layout.setContentsMargins(2, 2, 2, 2)
        self._metrics_layout.setHorizontalSpacing(8)
        self._metrics_layout.setVerticalSpacing(2)
        self._metrics_scroll.setWidget(self._metrics_container)
        outer.addWidget(self._metrics_scroll, 1)

        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(0, 0, 0, 0)
        all_btn = QPushButton("All")
        all_btn.setFixedWidth(48)
        none_btn = QPushButton("None")
        none_btn.setFixedWidth(48)
        all_btn.clicked.connect(self._select_all_metrics)
        none_btn.clicked.connect(self._deselect_all_metrics)
        btn_row.addWidget(all_btn)
        btn_row.addWidget(none_btn)
        btn_row.addStretch(1)
        outer.addLayout(btn_row)

        self.ref_vols_edit.editingFinished.connect(self._replot)
        self.ref_doses_edit.editingFinished.connect(self._on_ref_doses_edited)

        return group

    def _with_markers_checkbox(self, kind: str, edit: QLineEdit) -> QHBoxLayout:
        """Return *edit* in a row with a checkbox showing its points in the DVH."""
        name = "D_x" if kind == "dx" else "V_x"
        checkbox = QCheckBox("Show in DVH")
        checkbox.setToolTip(
            f"Mark the {name} points of the table on the DVH curves.\n"
            "Select a D_x / V_x cell or click a marker to show its value and parameter "
            "lines;\ndrag the dashed parameter line or double-click the plot to add a point."
        )
        self.show_markers_checkboxes[kind] = checkbox
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(edit, 1)
        row.addWidget(checkbox)
        return row

    def _sync_metric_checkboxes(self, selectable: list[str]) -> None:
        """Rebuild the metric checkboxes when the selectable metric set changes.

        Only metrics that are *not* already governed by the D_x / V_x inputs
        above get a checkbox; listing them twice in the same box would be
        redundant. Check states of metrics that survive a parameter change are
        preserved; metrics seen for the first time start checked.
        """
        if selectable == list(self._metric_checkboxes):
            return

        previous = {name: cb.isChecked() for name, cb in self._metric_checkboxes.items()}

        for cb in self._metric_checkboxes.values():
            cb.deleteLater()
        self._metric_checkboxes.clear()
        while self._metrics_layout.count():
            item = self._metrics_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        for i, name in enumerate(selectable):
            cb = TruncatedCheckBox(name)
            cb.setChecked(previous.get(name, True))
            cb.stateChanged.connect(self._on_metric_toggled)
            self._metric_checkboxes[name] = cb
            self._metrics_layout.addWidget(cb, *divmod(i, 3))

    def _checked_metrics(self) -> list[str]:
        """Return the metric ids to show as columns, in the collections' order.

        Metrics driven by the D_x / V_x inputs have no checkbox and are always
        included; the inputs themselves decide which of them exist.
        """
        return [
            name
            for name in self._available_metrics
            if name not in self._metric_checkboxes or self._metric_checkboxes[name].isChecked()
        ]

    def _on_metric_toggled(self, *_: Any) -> None:
        """Re-render the table only; the metric set itself did not change."""
        if self._syncing_metrics:
            return
        self._render_qi_table()

    def _set_all_metrics(self, checked: bool) -> None:
        for cb in self._metric_checkboxes.values():
            cb.blockSignals(True)
            cb.setChecked(checked)
            cb.blockSignals(False)
        self._render_qi_table()

    def _select_all_metrics(self) -> None:
        self._set_all_metrics(True)

    def _deselect_all_metrics(self) -> None:
        self._set_all_metrics(False)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_data(
        self,
        quantities: dict[str, np.ndarray],
        masks: dict[str, np.ndarray],
        voi_colors: dict[str, tuple[int, int, int]] | None = None,
        overlay_units: dict[str, str] | None = None,
        overlay_labels: dict[str, str] | None = None,
        initial_quantity: str = "",
        initial_vois: list[str] | None = None,
        voi_types: dict[str, str] | None = None,
    ) -> None:
        """Set all data, populate controls, and trigger the initial plot.

        The secondary (compare) dropdown is repopulated with the same quantities
        and reset to none, so only the primary quantity is plotted initially.

        Parameters
        ----------
        quantities:
            Mapping of quantity name → 3-D numpy array.
        masks:
            Mapping of VOI name → boolean/uint8 mask array matching the
            quantity arrays in shape.
        voi_colors:
            RGB tuples (0–255) per VOI name.
        overlay_units:
            Physical unit string (e.g. ``"Gy"``) per quantity name.
        overlay_labels:
            Display label (e.g. ``"Dose"``) per quantity name.
        initial_quantity:
            Quantity name to preselect in the quantity dropdown.
        initial_vois:
            VOI names to pre-check in the list. Defaults to all VOIs.
        voi_types:
            Optional mapping of VOI name → type (``TARGET``, ``OAR``,
            ``EXTERNAL``, ``HELPER``) used to group the list.
        """
        self._quantities = quantities or {}
        self._masks = masks or {}
        self._voi_types = dict(voi_types) if voi_types else {}
        self._voi_colors = dict(voi_colors) if voi_colors else {}
        self._overlay_units = overlay_units or {}
        self._overlay_labels = overlay_labels or {}
        self._dvh_cache = {}
        self._qi_cache = {}
        self._shown_quantity = ""
        self._ref_doses_by_quantity = {}

        qty_names = list(self._quantities.keys())

        self.quantity_selector.set_keys(qty_names, initial_quantity)
        self.secondary_selector.set_keys(qty_names)

        # ---- Populate VOI rows ----
        self._rebuild_voi_rows(initial_vois)

        self._replot()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _rebuild_voi_rows(self, initial_vois: list[str] | None) -> None:
        """Clear and rebuild the per-VOI checkbox + color-swatch rows."""
        # Remove old widgets
        for cb in self._voi_checkboxes.values():
            cb.deleteLater()
        self._voi_checkboxes.clear()
        self._voi_color_swatches.clear()

        while self._vois_layout.count():
            item = self._vois_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

        checked_set = set(initial_vois) if initial_vois is not None else set(self._masks.keys())

        sorted_names = sorted(self._masks.keys(), key=str.lower)

        groups: dict[str, list[str]] = {}
        for name in sorted_names:
            key = (self._voi_types.get(name, "") or "").upper() or _OTHER_GROUP[0]
            groups.setdefault(key, []).append(name)

        for key, title in (*_TYPE_GROUPS, _OTHER_GROUP):
            members = groups.pop(key, [])
            if not members:
                continue
            self._add_group_box(title, members, checked_set)
        for key, members in groups.items():
            if not members:
                continue
            self._add_group_box(key.title(), members, checked_set)

        self._vois_layout.addStretch(1)

    def _add_group_box(self, title: str, members: list[str], checked_set: set[str]) -> None:
        """Create a small QGroupBox for *title* containing a 2-column grid of VOIs."""
        group = QGroupBox(title)
        group.setStyleSheet(_GROUPBOX_STYLE)
        grid = QGridLayout(group)
        grid.setContentsMargins(6, 4, 6, 4)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(2)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)

        for i, name in enumerate(members):
            row_widget = self._build_voi_row(name, checked_set)
            grid_row, grid_col = divmod(i, 2)
            grid.addWidget(row_widget, grid_row, grid_col)

        self._vois_layout.addWidget(group)

    def _build_voi_row(self, name: str, checked_set: set[str]) -> QWidget:
        """Build a single [swatch | checkbox] row widget for *name*."""
        rgb = self._voi_colors.get(name) or (128, 128, 128)
        self._voi_colors[name] = rgb

        # Color swatch button
        swatch = QPushButton()
        swatch.setFixedSize(14, 14)
        swatch.setFlat(True)
        swatch.setStyleSheet(
            f"background-color: rgb({rgb[0]},{rgb[1]},{rgb[2]}); border: 1px solid #555;"
        )
        swatch.setCursor(Qt.CursorShape.PointingHandCursor)
        swatch.setToolTip(f"Change color for {name}")
        swatch.clicked.connect(lambda _, n=name: self._pick_color(n))
        self._voi_color_swatches[name] = swatch

        # Checkbox
        cb = TruncatedCheckBox(name)
        cb.setChecked(name in checked_set)
        cb.stateChanged.connect(self._replot)
        self._voi_checkboxes[name] = cb

        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 1, 0, 1)
        row_layout.setSpacing(6)
        row_layout.addWidget(swatch)
        row_layout.addWidget(cb, 1)
        return row

    def _pick_color(self, name: str) -> None:
        """Open a color dialog for *name* and apply the result."""
        current = self._voi_colors.get(name, (128, 128, 128))
        color = QColorDialog.getColor(QColor(*current), self, f"Pick color for {name}")
        if color.isValid():
            rgb = (color.red(), color.green(), color.blue())
            self._voi_colors[name] = rgb
            swatch = self._voi_color_swatches[name]
            swatch.setStyleSheet(
                f"background-color: rgb({rgb[0]},{rgb[1]},{rgb[2]}); border: 1px solid #555;"
            )
            self.color_changed.emit(name, rgb)
            self._replot()

    def _qi_params(self) -> tuple[list[float] | None, list[float] | None]:
        """Read the D_x / V_x inputs, falling back to the defaults when invalid.

        D_x volumes must be finite and within [0, 100] %, V_x doses finite and
        non-negative; out-of-range values are treated like unparsable text.
        """
        try:
            ref_vols = parse_number_list(self.ref_vols_edit.text())
            if not all(np.isfinite(v) and 0.0 <= v <= 100.0 for v in ref_vols):
                raise ValueError
        except ValueError:
            logger.debug("Invalid D_x input %r; using defaults", self.ref_vols_edit.text())
            ref_vols = None
        try:
            ref_doses = parse_number_list(self.ref_doses_edit.text())
            if not all(np.isfinite(v) and v >= 0.0 for v in ref_doses):
                raise ValueError
        except ValueError:
            logger.debug("Invalid V_x input %r; using defaults", self.ref_doses_edit.text())
            ref_doses = None

        # An empty field means "use the backend default", not "no metrics".
        return (ref_vols or None, ref_doses or None)

    def _on_ref_doses_edited(self) -> None:
        self._remember_ref_doses()
        self._replot()

    def _remember_ref_doses(self) -> None:
        """Store the V_x text as explicit input for the shown quantity.

        Text equal to what :meth:`_sync_param_fields` wrote back is not user
        input: storing derived defaults would pin them for that quantity.
        Invalid text is dropped rather than stored, so the quantity falls back
        to its defaults instead of re-applying the invalid entry on return.
        """
        text = self.ref_doses_edit.text()
        if not self._shown_quantity or text == self._synced_ref_doses_text:
            return
        if text and self._qi_params()[1] is None:
            self._ref_doses_by_quantity.pop(self._shown_quantity, None)
        else:
            self._ref_doses_by_quantity[self._shown_quantity] = text

    def _show_quantity(self, qty_name: str) -> None:
        """Switch the V_x input and its label over to *qty_name*."""
        self._remember_ref_doses()
        self._shown_quantity = qty_name

        text = self._ref_doses_by_quantity.get(qty_name, "")
        self.ref_doses_edit.blockSignals(True)
        self.ref_doses_edit.setText(text)
        self.ref_doses_edit.blockSignals(False)
        self._synced_ref_doses_text = text

        unit = safe_unit(self._overlay_units.get(qty_name, ""))
        label = "V_x:" if unit == ureg.dimensionless else f"V_x [{format_unit_symbol(unit)}]:"
        self._ref_doses_label.setText(label)

    def _dvh_for(self, qty_name: str, voi_name: str) -> DVH | None:
        """Return (and cache) the DVH of *qty_name* in structure *voi_name*."""
        key = (qty_name, voi_name)
        if key in self._dvh_cache:
            return self._dvh_cache[key]
        if qty_name not in self._quantities or voi_name not in self._masks:
            return None
        try:
            dvh = DVH.compute(
                quantity=self._quantities[qty_name], mask=self._masks[voi_name], name=voi_name
            )
        except (ValueError, TypeError) as exc:
            logger.debug("Could not compute %r DVH for structure %r: %s", qty_name, voi_name, exc)
            return None
        self._dvh_cache[key] = dvh
        return dvh

    def _qis_for(
        self,
        qty_name: str,
        ref_vols: list[float] | None,
        ref_doses: list[float] | None,
    ) -> QICollection | None:
        """Return (and cache) the QI collection for *qty_name*.

        Computed on demand rather than eagerly for every quantity: the metric
        parameters are user-editable, and voxel extraction over large masks
        dominates the cost.
        """
        if qty_name not in self._quantities or not self._masks:
            return None

        dose_unit = safe_unit(self._overlay_units.get(qty_name, ""))
        key = (qty_name, f"{ref_vols}|{ref_doses}|{dose_unit}")
        if key not in self._qi_cache:
            try:
                self._qi_cache[key] = QICollection.from_masks(
                    self._masks,
                    self._quantities[qty_name],
                    ref_vols=ref_vols,
                    ref_doses=ref_doses,
                    dose_unit=dose_unit,
                )
            except (ValueError, TypeError) as exc:
                logger.warning("Could not compute quality indicators for %r: %s", qty_name, exc)
                return None
        return self._qi_cache[key]

    def _update_qi_table(self, qty_name: str, selected_vois: list[str]) -> None:
        """Recompute the QI collection, refresh the metric list, and render."""
        self._qi_structures = selected_vois

        if not qty_name or not selected_vois:
            self._qi_collection = None
            self.qi_widget.set_qis(None)
            return

        ref_vols, ref_doses = self._qi_params()

        # Resolve the defaults here rather than in from_masks, so the values
        # written back to the fields keep the cache key identical.
        if ref_vols is None:
            ref_vols = list(DEFAULT_REF_VOLS)
        if ref_doses is None and qty_name in self._quantities:
            try:
                ref_doses = QICollection.default_ref_doses(self._quantities[qty_name]) or None
            except ValueError:
                ref_doses = None

        qis = self._qis_for(qty_name, ref_vols, ref_doses)
        self._qi_collection = qis
        self._available_metrics = qis.metric_ids(selected_vois) if qis is not None else []

        # D_x / V_x metrics are governed by the inputs above, so they get no
        # checkbox of their own; only the fixed reductions are selectable.
        selectable = [m for m in self._available_metrics if not self._is_parametric(qis, m)]

        self._syncing_metrics = True
        try:
            self._sync_metric_checkboxes(selectable)
            self._sync_param_fields(qis)
        finally:
            self._syncing_metrics = False

        self._render_qi_table()

    @staticmethod
    def _is_parametric(qis: QICollection | None, metric: str) -> bool:
        """Whether *metric* comes from a D_x / V_x reference parameter."""
        if qis is None:
            return False
        for structure in qis:
            qi = structure.metrics.get(metric)
            if qi is not None:
                return qi.qi_type in ("dx", "vx")
        return False

    def _sync_param_fields(self, qis: QICollection | None) -> None:
        """Write the reference values actually used back into the D_x / V_x inputs.

        Keeps the fields showing the real parameters rather than an ``auto``
        placeholder, and makes a fallback from invalid input visible. Without a
        collection (nothing could be computed) the user's input is left intact.
        """
        if qis is None:
            return

        ref_vols: list[float] = []
        ref_doses: list[float] = []
        for structure in qis:
            for qi in structure.values():
                if qi.qi_type == "dx" and qi.ref_vol not in ref_vols:
                    ref_vols.append(qi.ref_vol)
                elif qi.qi_type == "vx" and qi.ref_dose not in ref_doses:
                    ref_doses.append(qi.ref_dose)

        for edit, values in ((self.ref_vols_edit, ref_vols), (self.ref_doses_edit, ref_doses)):
            text = format_number_list(values)
            if edit.text() != text:
                edit.setText(text)
        self._synced_ref_doses_text = self.ref_doses_edit.text()

    def _render_qi_table(self) -> None:
        """Render the table from the cached collection and the checked metrics."""
        if self._qi_collection is None or not self._qi_structures:
            self.qi_widget.set_qis(None)
            return

        self.qi_widget.set_qis(
            self._qi_collection,
            structures=self._qi_structures,
            voi_colors=self._voi_colors,
            metrics=self._checked_metrics(),
        )

    def _checked_vois(self) -> list[str]:
        """Return names of currently checked VOIs."""
        return [n for n, cb in self._voi_checkboxes.items() if cb.isChecked()]

    def _select_all_vois(self) -> None:
        for cb in self._voi_checkboxes.values():
            cb.blockSignals(True)
            cb.setChecked(True)
            cb.blockSignals(False)
        self._replot()

    def _deselect_all_vois(self) -> None:
        for cb in self._voi_checkboxes.values():
            cb.blockSignals(True)
            cb.setChecked(False)
            cb.blockSignals(False)
        self._replot()

    def _secondary_quantity(self) -> str:
        """Return the quantity chosen for comparison, or ``""`` when there is none.

        A secondary equal to the primary quantity would only duplicate the solid
        curves, so it counts as none.
        """
        name = self.secondary_selector.current_key()
        if (
            not name
            or name not in self._quantities
            or name == self.quantity_selector.current_key()
        ):
            return ""
        return name

    def _collect_dvhs(self, qty_name: str, voi_names: list[str]) -> list[DVH]:
        """Return the available DVHs of *qty_name* for *voi_names*."""
        if not qty_name:
            return []
        dvhs = (self._dvh_for(qty_name, voi_name) for voi_name in voi_names)
        return [dvh for dvh in dvhs if dvh is not None]

    def _replot(self, *_: Any) -> None:
        """Replot the DVH and QI table for the shown quantities and the checked VOIs.

        The optional secondary quantity only adds dashed DVH curves; the QI
        table and its parameters follow the primary quantity.
        """
        qty_name = self.quantity_selector.current_key()
        if qty_name != self._shown_quantity:
            self._show_quantity(qty_name)
        selected_vois = self._checked_vois()

        # The QI table first, since the DVH plot marks its D_x / V_x points.
        self._update_qi_table(qty_name, selected_vois)
        dvhs = self._collect_dvhs(qty_name, selected_vois)
        plot_kwargs: dict[str, Any] = {
            "voi_colors": self._voi_colors,
            "overlay_unit": self._overlay_units.get(qty_name, ""),
            "overlay_label": self._overlay_labels.get(qty_name, ""),
            "qis": self._qi_collection,
        }
        secondary = self._secondary_quantity()
        if secondary:
            plot_kwargs.update(
                secondary_dvhs=self._collect_dvhs(secondary, selected_vois),
                secondary_unit=self._overlay_units.get(secondary, ""),
                secondary_label=self._overlay_labels.get(secondary, ""),
                primary_name=self.quantity_selector.current_label(),
                secondary_name=self.secondary_selector.current_label(),
            )
        self.dvh_widget.plot(dvhs, **plot_kwargs)

    # ------------------------------------------------------------------
    # DVH / QI table interaction
    # ------------------------------------------------------------------

    def _on_qi_cell_selected(self, structure: str, metric: str) -> None:
        self.dvh_widget.highlight(structure or None, metric or None)

    def _on_structure_clicked(self, structure: str) -> None:
        self.qi_widget.select(structure or None)
        self.dvh_widget.highlight(structure or None)

    def _on_marker_clicked(self, structure: str, metric: str) -> None:
        self.qi_widget.select(structure, metric)
        self.dvh_widget.highlight(structure, metric)

    def _ask_qi_point(
        self,
        dose: float,
        volume: float,
        kind: Optional[PointKind] = None,
        replace: Optional[str] = None,
    ) -> Optional[tuple[PointKind, float, bool]]:
        """Let the user confirm a D_x / V_x point; ``None`` when cancelled."""
        dose_unit = safe_unit(self._overlay_units.get(self._shown_quantity, ""))
        dialog = QIPointDialog(
            dose,
            volume,
            dose_unit="" if dose_unit == ureg.dimensionless else format_unit_symbol(dose_unit),
            kind=kind,
            replace=replace,
            parent=self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None
        return dialog.chosen()

    def _add_point(self, dose: float, volume: float) -> None:
        """Add a D_x / V_x point picked by a double-click in the DVH plot."""
        choice = self._ask_qi_point(dose, volume)
        if choice is not None:
            kind, parameter, _ = choice
            self._apply_qi_point(kind, parameter)

    def _move_point(self, structure: str, metric: str, parameter: float) -> None:
        """Replace or add the D_x / V_x point whose parameter line was dragged."""
        qis = self._qi_collection
        qi = qis[structure].metrics.get(metric) if qis and structure in qis else None
        if qi is None or qi.qi_type not in ("dx", "vx"):
            return
        old = qi.ref_vol if qi.qi_type == "dx" else qi.ref_dose
        dose, volume = (0.0, parameter) if qi.qi_type == "dx" else (parameter, 0.0)
        choice = self._ask_qi_point(dose, volume, kind=qi.qi_type, replace=metric)
        if choice is None:
            # Snap the dragged lines back to the unchanged point.
            self.dvh_widget.highlight(structure, metric)
            return
        kind, parameter, replace = choice
        self._apply_qi_point(
            kind, parameter, replace=old if replace else None, structure=structure
        )

    def _apply_qi_point(
        self,
        kind: PointKind,
        parameter: float,
        replace: Optional[float] = None,
        structure: Optional[str] = None,
    ) -> None:
        """Put *parameter* into the D_x or V_x input, replacing *replace* if given.

        The table and plot are then recomputed, the points are shown in the plot,
        and the new point is highlighted for *structure* (default: the highlighted
        structure).
        """
        if structure is None:
            structure = self.dvh_widget.highlighted()[0]
        edit = self.ref_vols_edit if kind == "dx" else self.ref_doses_edit
        try:
            values = parse_number_list(edit.text())
        except ValueError:
            values = []
        if replace is not None:
            values = [parameter if np.isclose(v, replace) else v for v in values]
        if not any(np.isclose(v, parameter) for v in values):
            values.append(parameter)
        edit.setText(format_number_list(values))
        if kind == "vx":
            self._remember_ref_doses()

        self.show_markers_checkboxes[kind].setChecked(True)
        self._replot()

        metric = self._metric_for(structure, kind, parameter)
        self.qi_widget.select(structure, metric)
        self.dvh_widget.highlight(structure, metric)

    def _metric_for(self, structure: Optional[str], kind: str, parameter: float) -> Optional[str]:
        """Return the metric id of *structure*'s D_x / V_x QI with *parameter*."""
        qis = self._qi_collection
        if structure is None or qis is None or structure not in qis:
            return None
        for metric, qi in qis[structure].metrics.items():
            if qi.qi_type != kind:
                continue
            if np.isclose(qi.ref_vol if kind == "dx" else qi.ref_dose, parameter):
                return metric
        return None
