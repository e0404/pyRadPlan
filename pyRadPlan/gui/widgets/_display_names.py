"""Human-readable names for quantities, result keys and biological models.

The GUI lists quantities and models by their display name and keeps the
identifier as item data.  :func:`show_identifiers_when_open` additionally
prints the identifier faintly next to each entry of an open combo box popup.
"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import (
    QModelIndex,
    QPersistentModelIndex,
    QRect,
    QSize,
    Qt,
)
from PySide6.QtGui import QPainter, QPalette
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
)

from pyRadPlan.quantities._display_names import (  # noqa: F401 - re-exported for the GUI
    FULL_COMPONENT,
    _INDEXED_KEY,
    _BEAM_SUFFIX,
    _SCENARIO_SUFFIX,
    SCENARIO_LABELS_KEY,
    BEAM_COMPONENT,
    SCENARIO_COMPONENT,
    _COMPONENT_SUFFIXES,
    _split_component_suffix,
    quantity_display_name,
    bio_model_display_name,
    robustness_display_name,
    ResultGroup,
    _split_index,
    group_result_keys,
    result_key,
    result_key_labels,
    split_result_key,
)

#: Item data role holding the identifier shown next to an item in an open popup.
IDENTIFIER_ROLE = Qt.ItemDataRole.UserRole + 101


class _IdentifierDelegate(QStyledItemDelegate):
    """Draw the item's :data:`IDENTIFIER_ROLE` faintly in parentheses after its text."""

    _GAP = "  "

    @staticmethod
    def _suffix(index: QModelIndex | QPersistentModelIndex) -> str:
        identifier = index.data(IDENTIFIER_ROLE)
        if not identifier or identifier == index.data(Qt.ItemDataRole.DisplayRole):
            return ""
        return f"({identifier})"

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionViewItem,
        index: QModelIndex | QPersistentModelIndex,
    ) -> None:
        super().paint(painter, option, index)
        suffix = self._suffix(index)
        if not suffix:
            return
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        style = opt.widget.style() if opt.widget is not None else QApplication.style()
        text_rect = style.subElementRect(QStyle.SubElement.SE_ItemViewItemText, opt, opt.widget)
        metrics = opt.fontMetrics
        left = text_rect.left() + metrics.horizontalAdvance(opt.text + self._GAP)
        if left >= text_rect.right():
            return
        selected = bool(opt.state & QStyle.StateFlag.State_Selected)
        color = opt.palette.color(
            QPalette.ColorRole.HighlightedText if selected else QPalette.ColorRole.PlaceholderText
        )
        if selected:
            color.setAlphaF(0.7)
        painter.save()
        painter.setPen(color)
        painter.drawText(
            QRect(left, text_rect.top(), text_rect.right() - left, text_rect.height()),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            suffix,
        )
        painter.restore()

    def sizeHint(  # noqa: N802
        self, option: QStyleOptionViewItem, index: QModelIndex | QPersistentModelIndex
    ) -> QSize:
        size = super().sizeHint(option, index)
        suffix = self._suffix(index)
        if suffix:
            size.setWidth(size.width() + option.fontMetrics.horizontalAdvance(self._GAP + suffix))
        return size


def show_identifiers_when_open(combo: QComboBox) -> None:
    """Show each item's :data:`IDENTIFIER_ROLE` in the open popup of *combo*.

    The closed combo box only shows the item text; the popup is widened to fit
    the identifiers.
    """
    combo.setItemDelegate(_IdentifierDelegate(combo))
    view = combo.view()

    def fit_popup(*_: Any) -> None:
        width = view.sizeHintForColumn(0) + 2 * view.frameWidth()
        width += view.verticalScrollBar().sizeHint().width()
        view.setMinimumWidth(width)

    model = combo.model()
    model.rowsInserted.connect(fit_popup)
    model.modelReset.connect(fit_popup)
    model.dataChanged.connect(fit_popup)


def add_identified_item(combo: QComboBox, text: str, data: Any, identifier: str) -> None:
    """Append an item showing *text*, holding *data* and naming *identifier* when open."""
    combo.addItem(text, data)
    combo.setItemData(combo.count() - 1, identifier, IDENTIFIER_ROLE)


def current_identifier(combo: QComboBox) -> str:
    """Return the identifier the selected item holds as data (``""`` for none)."""
    data = combo.currentData()
    return data if isinstance(data, str) else ""


def select_identifier(combo: QComboBox, identifier: Any) -> bool:
    """Select the item holding *identifier* as data; return whether there is one."""
    index = combo.findData(identifier)
    if index >= 0:
        combo.setCurrentIndex(index)
    return index >= 0
