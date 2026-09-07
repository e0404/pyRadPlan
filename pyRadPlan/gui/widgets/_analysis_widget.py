"""DVH and QI analysis widget."""

from __future__ import annotations

from typing import Any

import numpy as np

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QColorDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QComboBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from pyRadPlan.gui.widgets.analysis._dvh import DVHPlotWidget
from pyRadPlan.gui.widgets.analysis._gamma import GammaWidget
from pyRadPlan.gui.widgets.analysis._qi import QITableWidget
from pyRadPlan.gui.widgets.result._labels import TruncatedCheckBox
from pyRadPlan.analysis._dvh import DVH

_NONE_LABEL = "— None —"

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


class AnalysisWidget(QWidget):
    """Widget displaying DVH plot, QI table, and Gamma analysis.

    Data is set via :meth:`set_data`. DVH curves are computed eagerly for all
    (quantity x mask) combinations and cached; selection changes only trigger
    a replot from the cache.
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

        # Per-VOI UI state (populated by set_data)
        self._voi_checkboxes: dict[str, TruncatedCheckBox] = {}
        self._voi_color_swatches: dict[str, QPushButton] = {}

        # --- Main layout ---
        layout = QVBoxLayout(self)

        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        # ---- Tab 1: DVH & QI ----
        self.dvh_qi_widget = QWidget()
        dvh_qi_layout = QVBoxLayout(self.dvh_qi_widget)
        dvh_qi_layout.setContentsMargins(4, 4, 4, 4)

        # Controls panel (quantities left | structures right)
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

        self.tabs.addTab(self.dvh_qi_widget, "DVH / QI")

        # ---- Tab 2: Gamma ----
        self.gamma_widget = GammaWidget()
        self.tabs.addTab(self.gamma_widget, "Gamma Analysis")

    # ------------------------------------------------------------------
    # Controls panel construction
    # ------------------------------------------------------------------

    def _build_controls_panel(self) -> QWidget:
        """Build the side-by-side Quantities | Structures controls panel."""
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

        self.q1_combo = QComboBox()
        self.q1_combo.setToolTip("Primary quantity — plotted as solid lines")
        form.addRow(QLabel("Primary:"), self.q1_combo)

        self.q2_combo = QComboBox()
        self.q2_combo.setToolTip("Secondary quantity — plotted as dotted lines (optional)")
        form.addRow(QLabel("Secondary:"), self.q2_combo)

        panel_layout.addWidget(qty_group, 1)

        # ---- Right: Structures ----
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

        # Connect quantity-combo signals
        self.q1_combo.currentTextChanged.connect(self._replot)
        self.q2_combo.currentTextChanged.connect(self._replot)

        return panel

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
        """Set all data, populate controls, cache DVHs, and trigger initial plot.

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
            Quantity name to preselect in the primary combo.
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

        # Pre-compute DVHs for every (quantity × mask) combination
        for qty_name, qty_arr in self._quantities.items():
            for voi_name, mask in self._masks.items():
                try:
                    dvh = DVH.compute(quantity=qty_arr, mask=mask, name=voi_name)
                    self._dvh_cache[(qty_name, voi_name)] = dvh
                except Exception:
                    pass

        # ---- Populate quantity combos ----
        qty_names = list(self._quantities.keys())

        self.q1_combo.blockSignals(True)
        self.q2_combo.blockSignals(True)

        self.q1_combo.clear()
        self.q1_combo.addItems(qty_names)

        self.q2_combo.clear()
        self.q2_combo.addItem(_NONE_LABEL)
        self.q2_combo.addItems(qty_names)

        if initial_quantity and initial_quantity in qty_names:
            self.q1_combo.setCurrentText(initial_quantity)
        elif qty_names:
            self.q1_combo.setCurrentIndex(0)
        self.q2_combo.setCurrentIndex(0)  # default: None

        self.q1_combo.blockSignals(False)
        self.q2_combo.blockSignals(False)

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

    def _replot(self, *_: Any) -> None:
        """Replot DVH using the current combo and VOI selections."""
        q1_name = self.q1_combo.currentText()
        q2_name = self.q2_combo.currentText()
        selected_vois = self._checked_vois()

        if not q1_name or not selected_vois:
            self.dvh_widget.plot([], None, self._voi_colors)
            return

        dvhs_q1 = [
            self._dvh_cache[(q1_name, v)] for v in selected_vois if (q1_name, v) in self._dvh_cache
        ]

        dvhs_q2: list[DVH] | None = None
        if q2_name and q2_name != _NONE_LABEL:
            dvhs_q2 = [
                self._dvh_cache[(q2_name, v)]
                for v in selected_vois
                if (q2_name, v) in self._dvh_cache
            ]
            if not dvhs_q2:
                dvhs_q2 = None

        overlay_unit = self._overlay_units.get(q1_name, "")
        overlay_label = self._overlay_labels.get(q1_name, "")

        self.dvh_widget.plot(
            dvhs_q1,
            dvhs_q2=dvhs_q2,
            voi_colors=self._voi_colors,
            overlay_unit=overlay_unit,
            overlay_label=overlay_label,
            q1_label=q1_name,
            q2_label=q2_name if dvhs_q2 else "",
        )
        self.qi_widget.update(None)
