"""VOIs selection widget."""

from __future__ import annotations

from typing import Iterable, Optional

import SimpleITK as sitk

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QColorDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from pyRadPlan.cst import VOI

from ._labels import TruncatedCheckBox


# Order in which groups are rendered, and their display titles.
_TYPE_GROUPS: tuple[tuple[str, str], ...] = (
    ("TARGET", "Targets"),
    ("OAR", "OARs"),
    ("EXTERNAL", "External"),
    ("HELPER", "Helpers"),
)
_OTHER_GROUP = ("OTHER", "Other")

_GROUPBOX_STYLE = (
    "QGroupBox { font-size: 10pt; font-weight: 600; margin-top: 6px; "
    "border: 1px solid palette(mid); border-radius: 4px; padding: 4px 6px 4px 6px; }"
    "QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; }"
)


class VOIsWidget(QWidget):
    """Widget for VOI selection."""

    # Signals
    voi_toggled = Signal(str, bool)  # voi name, checked
    selection_changed = Signal(list)  # list[str] of selected VOIs
    color_changed = Signal(str, tuple)  # voi name, new RGB tuple

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._voi_checkboxes: dict[str, TruncatedCheckBox] = {}
        self._voi_colors: dict[str, tuple[int, int, int]] = {}
        self._voi_color_swatches: dict[str, QPushButton] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        vois_group = QGroupBox("VOIs")
        vois_group_layout = QVBoxLayout()
        vois_group_layout.setContentsMargins(6, 6, 6, 6)
        vois_group_layout.setSpacing(4)
        vois_group.setLayout(vois_group_layout)
        layout.addWidget(vois_group)

        self._vois_scroll = QScrollArea()
        self._vois_scroll.setWidgetResizable(True)
        self._vois_container = QWidget()
        self._vois_layout = QVBoxLayout(self._vois_container)
        self._vois_layout.setContentsMargins(2, 2, 2, 2)
        self._vois_layout.setSpacing(4)
        self._vois_scroll.setWidget(self._vois_container)
        vois_group_layout.addWidget(self._vois_scroll)

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
        vois_group_layout.addLayout(btn_row)

    def set_vois(self, vois: list[VOI], checked: Optional[Iterable[str]] = None) -> None:
        """Populate the VOIs section with a variable number of checkboxes."""
        for cb in self._voi_checkboxes.values():
            cb.deleteLater()
        self._voi_checkboxes.clear()
        self._voi_color_swatches.clear()

        while self._vois_layout.count():
            item = self._vois_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

        sorted_vois = sorted(vois, key=lambda v: v.name.lower())

        empty_names = self._empty_voi_names(sorted_vois)
        checked_set = self._determine_checked_vois(sorted_vois, checked) - empty_names

        groups: dict[str, list[VOI]] = {}
        for voi in sorted_vois:
            key = getattr(voi, "voi_type", "").upper() or _OTHER_GROUP[0]
            groups.setdefault(key, []).append(voi)

        for key, title in (*_TYPE_GROUPS, _OTHER_GROUP):
            members = groups.pop(key, [])
            if not members:
                continue
            self._add_group_box(title, members, checked_set)

        # Any unknown types that slipped through
        for key, members in groups.items():
            if not members:
                continue
            self._add_group_box(key.title(), members, checked_set)

        self._vois_layout.addStretch(1)

        self.selection_changed.emit(self.selected_vois())

    def selected_vois(self) -> list[str]:
        """Return the list of currently checked VOIs by name."""
        return [name for name, cb in self._voi_checkboxes.items() if cb.isChecked()]

    def get_voi_colors(self) -> dict[str, tuple[int, int, int]]:
        """Return the color mapping for VOIs."""
        return self._voi_colors

    @staticmethod
    def _empty_voi_names(vois: list[VOI]) -> set[str]:
        """Return names of VOIs whose mask contains no voxels."""
        empty: set[str] = set()
        for voi in vois:
            try:
                if not sitk.GetArrayViewFromImage(voi.mask).any():
                    empty.add(voi.name)
            except Exception:
                pass
        return empty

    def _determine_checked_vois(
        self, vois: list[VOI], checked: Optional[Iterable[str]]
    ) -> set[str]:
        """Determine which VOIs should be checked by default."""
        if checked is not None:
            return set(checked)

        with_objectives = {
            voi.name
            for voi in vois
            if any(obj is not None for obj in getattr(voi, "objectives", None) or [])
        }
        if with_objectives:
            return with_objectives

        body_name, target_name, oar_name = self._select_by_voi_type(vois)
        body_name, target_name, oar_name = self._fill_by_name_heuristic(
            vois, body_name, target_name, oar_name
        )
        return {n for n in [body_name, target_name, oar_name] if n}

    def _select_by_voi_type(
        self, vois: list[VOI]
    ) -> tuple[Optional[str], Optional[str], Optional[str]]:
        """Return (body, target, oar) names selected by voi_type attribute."""
        body_name: Optional[str] = None
        target_name: Optional[str] = None
        oar_name: Optional[str] = None
        for voi in vois:
            vtype = getattr(voi, "voi_type", "").upper()
            if body_name is None and vtype == "EXTERNAL":
                body_name = voi.name
            if target_name is None and vtype == "TARGET":
                target_name = voi.name
            if oar_name is None and vtype == "OAR":
                oar_name = voi.name
            if body_name and target_name and oar_name:
                break
        return body_name, target_name, oar_name

    def _fill_by_name_heuristic(
        self,
        vois: list[VOI],
        body_name: Optional[str],
        target_name: Optional[str],
        oar_name: Optional[str],
    ) -> tuple[Optional[str], Optional[str], Optional[str]]:
        """Fill in any ``None`` entry via name substring matching."""
        if body_name is None:
            for voi in vois:
                if "body" in voi.name.lower():
                    body_name = voi.name
                    break
        if target_name is None:
            for voi in vois:
                if "target" in voi.name.lower():
                    target_name = voi.name
                    break
        if oar_name is None:
            for voi in vois:
                if "oar" in voi.name.lower():
                    oar_name = voi.name
                    break
        return body_name, target_name, oar_name

    def _add_group_box(self, title: str, members: list[VOI], checked_set: set[str]) -> None:
        """Create a small QGroupBox for *title* containing a 2-column grid of VOIs."""
        group = QGroupBox(title)
        group.setStyleSheet(_GROUPBOX_STYLE)
        grid = QGridLayout(group)
        grid.setContentsMargins(6, 4, 6, 4)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(2)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)

        for i, voi in enumerate(members):
            row_widget = self._build_voi_row(voi, checked_set)
            grid_row, grid_col = divmod(i, 2)
            grid.addWidget(row_widget, grid_row, grid_col)

        self._vois_layout.addWidget(group)

    def _build_voi_row(self, voi: VOI, checked_set: set[str]) -> QWidget:
        """Build a single [swatch | checkbox] row widget for *voi*."""
        cb = TruncatedCheckBox(voi.name)
        cb.setChecked(voi.name in checked_set)
        cb.stateChanged.connect(lambda s, n=voi.name: self._on_voi_toggled(n, bool(s)))

        rgb = tuple(int(c) for c in voi.visible_color)
        self._voi_colors[voi.name] = rgb

        swatch = QPushButton()
        swatch.setFixedSize(12, 12)
        swatch.setFlat(True)
        swatch.setStyleSheet(
            f"background-color: rgb({rgb[0]}, {rgb[1]}, {rgb[2]}); border: 1px solid #444;"
        )
        swatch.setCursor(Qt.CursorShape.PointingHandCursor)
        swatch.clicked.connect(lambda _, n=voi.name: self._pick_color(n))
        self._voi_color_swatches[voi.name] = swatch

        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(6)
        row_layout.addWidget(swatch, 0)
        row_layout.addWidget(cb, 1)

        self._voi_checkboxes[voi.name] = cb
        return row

    def _pick_color(self, name: str) -> None:
        """Open a color dialog for the given VOI and apply the result."""
        current = self._voi_colors.get(name, (128, 128, 128))
        color = QColorDialog.getColor(QColor(*current), self, f"Pick color for {name}")
        if color.isValid():
            rgb = (color.red(), color.green(), color.blue())
            self._voi_colors[name] = rgb
            swatch = self._voi_color_swatches[name]
            swatch.setStyleSheet(
                f"background-color: rgb({rgb[0]}, {rgb[1]}, {rgb[2]}); border: 1px solid #444;"
            )
            self.color_changed.emit(name, rgb)

    def _on_voi_toggled(self, name: str, checked: bool) -> None:
        self.voi_toggled.emit(name, checked)
        self.selection_changed.emit(self.selected_vois())

    def _select_all_vois(self) -> None:
        for cb in self._voi_checkboxes.values():
            cb.blockSignals(True)
            cb.setChecked(True)
            cb.blockSignals(False)
        self.selection_changed.emit(self.selected_vois())

    def _deselect_all_vois(self) -> None:
        for cb in self._voi_checkboxes.values():
            cb.blockSignals(True)
            cb.setChecked(False)
            cb.blockSignals(False)
        self.selection_changed.emit(self.selected_vois())
