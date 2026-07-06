"""VOIs selection widget."""

from __future__ import annotations

import html
from typing import Iterable, Optional

import SimpleITK as sitk
from pydantic import ValidationError

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QMouseEvent
from PySide6.QtWidgets import (
    QColorDialog,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QStyle,
    QStyleOptionButton,
    QVBoxLayout,
    QWidget,
)

from pyRadPlan.cst import VOI, create_voi

from ._labels import TruncatedCheckBox


# Order in which groups are rendered, and their display titles.
_TYPE_GROUPS: tuple[tuple[str, str], ...] = (
    ("TARGET", "Targets"),
    ("OAR", "OARs"),
    ("EXTERNAL", "External"),
    ("HELPER", "Helpers"),
)
_OTHER_GROUP = ("OTHER", "Other")

# Types instantiable via create_voi, offered in the metadata editor.
_VOI_TYPES = ("TARGET", "OAR", "EXTERNAL", "HELPER")

_GROUPBOX_STYLE = (
    "QGroupBox { font-size: 10pt; font-weight: 600; margin-top: 6px; "
    "border: 1px solid palette(mid); border-radius: 4px; padding: 4px 6px 4px 6px; }"
    "QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; }"
)


class _VOICheckBox(TruncatedCheckBox):
    """Checkbox whose label area emits *label_clicked* instead of toggling.

    Toggling remains available on the indicator box and via keyboard.
    """

    label_clicked = Signal()

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            opt = QStyleOptionButton()
            self.initStyleOption(opt)
            indicator = self.style().subElementRect(
                QStyle.SubElement.SE_CheckBoxIndicator, opt, self
            )
            if event.position().toPoint().x() > indicator.right():
                self.label_clicked.emit()
                event.accept()
                return
        super().mousePressEvent(event)


class VOIMetadataDialog(QDialog):
    """Small popup to edit the basic metadata of a single VOI.

    Values are written back to the VOI model on accept, relying on pydantic's
    assignment validation. The VOI type is bound to the model class, so
    changing it recreates the VOI as the matching class; the result is exposed
    via :attr:`updated_voi` (identical to the input VOI when the type is kept).
    """

    def __init__(self, voi: VOI, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._voi = voi
        self.updated_voi: VOI = voi
        self.setWindowTitle(f"Edit VOI: {voi.name}")

        form = QFormLayout(self)
        form.addRow("Name:", QLabel(voi.name))

        self.type_combo = QComboBox()
        self.type_combo.addItems(list(_VOI_TYPES))
        if voi.voi_type not in _VOI_TYPES:
            self.type_combo.addItem(voi.voi_type)
        self.type_combo.setCurrentText(voi.voi_type)
        form.addRow("Type:", self.type_combo)

        self.alpha_spin = QDoubleSpinBox()
        self.alpha_spin.setDecimals(4)
        self.alpha_spin.setRange(0.0, 100.0)
        self.alpha_spin.setSingleStep(0.01)
        self.alpha_spin.setSuffix(" Gy⁻¹")
        self.alpha_spin.setValue(voi.alpha_x)
        form.addRow("α_x:", self.alpha_spin)

        self.beta_spin = QDoubleSpinBox()
        self.beta_spin.setDecimals(4)
        self.beta_spin.setRange(0.0, 100.0)
        self.beta_spin.setSingleStep(0.01)
        self.beta_spin.setSuffix(" Gy⁻²")
        self.beta_spin.setValue(voi.beta_x)
        form.addRow("β_x:", self.beta_spin)

        self.priority_spin = QSpinBox()
        self.priority_spin.setRange(0, 9999)
        self.priority_spin.setValue(voi.overlap_priority)
        self.priority_spin.setToolTip("Lower priority numbers overlap higher ones")
        form.addRow("Overlap priority:", self.priority_spin)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def accept(self) -> None:
        """Validate and apply the edited values to the VOI before closing."""
        new_type = self.type_combo.currentText()
        try:
            if new_type != self._voi.voi_type:
                data = dict(self._voi)
                data.update(
                    voi_type=new_type,
                    alpha_x=self.alpha_spin.value(),
                    beta_x=self.beta_spin.value(),
                    overlap_priority=self.priority_spin.value(),
                )
                # Let the new type's default_factory provide its bound color
                data.pop("default_color", None)
                self.updated_voi = create_voi(data)
            else:
                self._voi.alpha_x = self.alpha_spin.value()
                self._voi.beta_x = self.beta_spin.value()
                self._voi.overlap_priority = self.priority_spin.value()
        except (ValidationError, ValueError, TypeError) as exc:
            QMessageBox.warning(self, "Invalid value", str(exc))
            return
        super().accept()


class VOIsWidget(QWidget):
    """Widget for VOI selection."""

    # Signals
    voi_toggled = Signal(str, bool)  # voi name, checked
    selection_changed = Signal(list)  # list[str] of selected VOIs
    color_changed = Signal(str, tuple)  # voi name, new RGB tuple
    metadata_changed = Signal(str)  # voi name whose metadata was edited
    voi_replaced = Signal(str, object)  # voi name, new VOI instance (after a type change)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._vois: list[VOI] = []
        self._voi_by_name: dict[str, VOI] = {}
        self._group_mode: str = "type"
        self._voi_checkboxes: dict[str, TruncatedCheckBox] = {}
        self._voi_colors: dict[str, tuple[int, int, int]] = {}
        self._voi_color_swatches: dict[str, QPushButton] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        self._vois_scroll = QScrollArea()
        self._vois_scroll.setWidgetResizable(True)
        self._vois_container = QWidget()
        self._vois_layout = QVBoxLayout(self._vois_container)
        self._vois_layout.setContentsMargins(2, 2, 2, 2)
        self._vois_layout.setSpacing(4)
        self._vois_scroll.setWidget(self._vois_container)
        layout.addWidget(self._vois_scroll)

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

        self._group_combo = QComboBox()
        self._group_combo.addItems(["By Type", "By Overlap"])
        self._group_combo.setToolTip("Group the VOI list by type or by overlap priority")
        self._group_combo.currentIndexChanged.connect(self._on_group_mode_changed)
        btn_row.addWidget(self._group_combo)

        layout.addLayout(btn_row)

    def set_vois(self, vois: list[VOI], checked: Optional[Iterable[str]] = None) -> None:
        """Populate the VOIs section with a variable number of checkboxes."""
        self._vois = list(vois)
        self._voi_by_name = {voi.name: voi for voi in self._vois}

        sorted_vois = sorted(self._vois, key=lambda v: v.name.lower())
        empty_names = self._empty_voi_names(sorted_vois)
        checked_set = self._determine_checked_vois(sorted_vois, checked) - empty_names

        self._render(checked_set)

    def _render(self, checked_set: set[str]) -> None:
        """Rebuild the group boxes for the stored VOIs with the given checked state."""
        for cb in self._voi_checkboxes.values():
            cb.deleteLater()
        self._voi_checkboxes.clear()
        self._voi_color_swatches.clear()

        while self._vois_layout.count():
            item = self._vois_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

        sorted_vois = sorted(self._vois, key=lambda v: v.name.lower())

        if self._group_mode == "overlap":
            self._render_by_overlap(sorted_vois, checked_set)
        else:
            self._render_by_type(sorted_vois, checked_set)

        self._vois_layout.addStretch(1)

        self.selection_changed.emit(self.selected_vois())

    def _render_by_type(self, sorted_vois: list[VOI], checked_set: set[str]) -> None:
        """Render one group box per VOI type."""
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

    def _render_by_overlap(self, sorted_vois: list[VOI], checked_set: set[str]) -> None:
        """Render one group box per overlap priority level, lowest (dominant) first."""
        groups: dict[int, list[VOI]] = {}
        for voi in sorted_vois:
            groups.setdefault(int(voi.overlap_priority), []).append(voi)

        for priority in sorted(groups):
            self._add_group_box(f"Overlap priority {priority}", groups[priority], checked_set)

    def _on_group_mode_changed(self, index: int) -> None:
        self._group_mode = "overlap" if index == 1 else "type"
        self._render(set(self.selected_vois()))

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
        cb = _VOICheckBox(voi.name)
        cb.setChecked(voi.name in checked_set)
        cb.setToolTip(self._voi_tooltip(voi))
        cb.stateChanged.connect(lambda s, n=voi.name: self._on_voi_toggled(n, bool(s)))
        cb.label_clicked.connect(lambda n=voi.name: self._edit_voi(n))

        rgb = tuple(int(c) for c in (voi.visible_color or voi.default_color))
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

    @staticmethod
    def _voi_tooltip(voi: VOI) -> str:
        """Build a rich-text tooltip summarizing the VOI metadata."""
        if voi.beta_x:
            ratio = f"{voi.alpha_x / voi.beta_x:.4g}&nbsp;Gy"
        else:
            ratio = "&#8734;"
        return (
            f"<b>{html.escape(voi.name)}</b> ({html.escape(voi.voi_type)})<br/>"
            f"&alpha;<sub>x</sub> = {voi.alpha_x:g}&nbsp;Gy<sup>&#8722;1</sup>, "
            f"&beta;<sub>x</sub> = {voi.beta_x:g}&nbsp;Gy<sup>&#8722;2</sup><br/>"
            f"&alpha;/&beta; = {ratio}<br/>"
            f"Overlap priority: {voi.overlap_priority}<br/>"
            f"<i>Click the name to edit</i>"
        )

    def _edit_voi(self, name: str) -> None:
        """Open the metadata editor popup for the given VOI."""
        voi = self._voi_by_name.get(name)
        if voi is None:
            return
        dialog = VOIMetadataDialog(voi, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        new_voi = dialog.updated_voi
        if new_voi is not voi:
            self._vois = [new_voi if v is voi else v for v in self._vois]
            self._voi_by_name[name] = new_voi
        self._render(set(self.selected_vois()))
        if new_voi is not voi:
            self.voi_replaced.emit(name, new_voi)
        self.metadata_changed.emit(name)

    def _pick_color(self, name: str) -> None:
        """Open a color dialog for the given VOI and apply the result."""
        current = self._voi_colors.get(name, (128, 128, 128))
        color = QColorDialog.getColor(QColor(*current), self, f"Pick color for {name}")
        if color.isValid():
            rgb = (color.red(), color.green(), color.blue())
            self._voi_colors[name] = rgb
            voi = self._voi_by_name.get(name)
            if voi is not None:
                voi.visible_color = rgb
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
