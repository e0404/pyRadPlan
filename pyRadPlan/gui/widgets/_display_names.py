"""Human-readable names for quantities, result keys and biological models.

The GUI lists quantities and models by their display name and keeps the
identifier as item data.  :func:`show_identifiers_when_open` additionally
prints the identifier faintly next to each entry of an open combo box popup.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable

from PySide6.QtCore import QModelIndex, QPersistentModelIndex, QRect, QSize, Qt
from PySide6.QtGui import QPainter, QPalette
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
)

from pyRadPlan.gui.workspace import RESULT_KEY_SEPARATOR
from pyRadPlan.optimization.robustness import get_robustness_operator_class
from pyRadPlan.quantities import get_available_quantities

#: Item data role holding the identifier shown next to an item in an open popup.
IDENTIFIER_ROLE = Qt.ItemDataRole.UserRole + 101

#: Component label of the total (all beams) entry of a result group.
FULL_COMPONENT = "full"

#: Flattened list entries of the viewer (``"<key> <i>"``).
_INDEXED_KEY = re.compile(r"^(?P<stem>.+) (?P<index>\d+)$")
_BEAM_SUFFIX = "_beam"
_SCENARIO_SUFFIX = "_scenario"
#: Component kinds of a :class:`ResultGroup` besides :data:`FULL_COMPONENT`.
BEAM_COMPONENT = "beam"
SCENARIO_COMPONENT = "scenario"
#: Component suffixes of result keys (``"<identifier><suffix>"``) and their component kind.
_COMPONENT_SUFFIXES = {_BEAM_SUFFIX: BEAM_COMPONENT, _SCENARIO_SUFFIX: SCENARIO_COMPONENT}


def _split_component_suffix(stem: str) -> tuple[str, str]:
    """Split ``"<base>_beam"`` / ``"<base>_scenario"`` into base and kind (``""`` if none)."""
    for suffix, kind in _COMPONENT_SUFFIXES.items():
        if stem.endswith(suffix) and len(stem) > len(suffix):
            return stem[: -len(suffix)], kind
    return stem, ""


def result_key(tag: str, key: str) -> str:
    """Key listing the entry *key* of the result *tag* (``""`` for the active result).

    See :meth:`pyRadPlan.gui.workspace.WorkspaceManager.result_entries`.
    """
    return f"{tag}{RESULT_KEY_SEPARATOR}{key}" if tag else key


def split_result_key(key: str) -> tuple[str, str]:
    """Split a key built by :func:`result_key` into the result tag and the entry key."""
    tag, separator, entry = key.partition(RESULT_KEY_SEPARATOR)
    return (tag, entry) if separator and tag and entry else ("", key)


def quantity_display_name(key: str) -> str:
    """Return the display name of a quantity identifier or result key.

    Registered quantity identifiers map to :attr:`RTQuantity.name`, including
    per-beam and per-scenario result keys (``"<identifier>_beam"``,
    ``"<identifier>_scenario"`` and their flattened ``"<identifier>_beam <i>"``
    entries).  Any other key (``alpha``, imported doses, matRad names) is
    returned unchanged; use :func:`group_result_keys` to also name the result
    a key belongs to.
    """
    stem, sep, index = key.partition(" ")
    base, kind = _split_component_suffix(stem)
    quantity = get_available_quantities().get(base)
    if quantity is None:
        return key
    name = f"{quantity.name} {kind}" if kind else quantity.name
    return f"{name}{sep}{index}"


def bio_model_display_name(model_cls: Any) -> str:
    """Return the display name of a biological model class (or instance)."""
    return getattr(model_cls, "name", None) or model_cls.model


def robustness_display_name(name: str) -> str:
    """Return the label of a robustness operator name; unknown names are returned unchanged."""
    try:
        operator_cls = get_robustness_operator_class(name)
    except (KeyError, ValueError):
        return name
    return operator_cls.label or operator_cls.name


# ----------------------------------------------------------------------
# Result key grouping
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class ResultGroup:
    """Result entries belonging to one quantity of one result.

    Attributes
    ----------
    base:
        Quantity identifier or, for unregistered entries, the key stem.
    tag:
        Name of the result; empty for the active result.
    label:
        Display label, e.g. ``"RBExDose"`` or ``"dose (protons_30fx)"``.
    components:
        ``(label, key)`` pairs: :data:`FULL_COMPONENT` for the total, then
        ``"beam <i>"`` for each per-beam entry and ``"scenario <i>"`` for each
        per-scenario entry (``<i>`` is the flat scenario index).
    kinds:
        Component kind per entry of :attr:`components`: :data:`FULL_COMPONENT`,
        :data:`BEAM_COMPONENT`, :data:`SCENARIO_COMPONENT` or ``""`` for other
        indexed entries.  Empty for groups built without kinds.
    """

    base: str
    tag: str
    label: str
    components: tuple[tuple[str, str], ...]
    kinds: tuple[str, ...] = ()

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(key for _, key in self.components)

    def keys_of(self, kind: str) -> tuple[str, ...]:
        """Keys of the components of *kind* (e.g. :data:`SCENARIO_COMPONENT`)."""
        return tuple(key for (_, key), k in zip(self.components, self.kinds) if k == kind)

    @property
    def identifier(self) -> str:
        """Key of the total entry, or the group stem when there is none."""
        for label, key in self.components:
            if label == FULL_COMPONENT:
                return key
        return result_key(self.tag, self.base)


def _split_index(key: str) -> tuple[str, int | None]:
    match = _INDEXED_KEY.match(key)
    if match is None:
        return key, None
    return match["stem"], int(match["index"])


def group_result_keys(keys: Iterable[str]) -> list[ResultGroup]:
    """Group result keys by result and quantity.

    ``"[<result>/]<base>[_beam|_scenario][ <i>]"`` keys (see :func:`result_key`)
    are grouped by ``(base, result)``: the per-beam and per-scenario entries join
    the total of their quantity (beams first, then scenarios).  Groups keep the
    order of first appearance.
    """
    ranks = {FULL_COMPONENT: 0, BEAM_COMPONENT: 1, "": 1, SCENARIO_COMPONENT: 2}

    entries: dict[tuple[str, str], list[tuple[tuple[int, int], str, str, str]]] = {}
    for key in keys:
        tag, entry = split_result_key(key)
        stem, index = _split_index(entry)
        stem, kind = _split_component_suffix(stem)
        if kind:
            label = kind if index is None else f"{kind} {index}"
        elif index is not None:
            label = f"[{index}]"
        else:
            label = kind = FULL_COMPONENT
        order = (ranks[kind], -1 if index is None else index)
        entries.setdefault((stem, tag), []).append((order, label, key, kind))

    groups = []
    for (base, tag), items in entries.items():
        label = quantity_display_name(base)
        if tag:
            label = f"{label} ({tag})"
        items.sort(key=lambda e: e[0])
        components = tuple((lbl, key) for _, lbl, key, _ in items)
        kinds = tuple(kind for *_, kind in items)
        groups.append(ResultGroup(base, tag, label, components, kinds))
    return groups


def result_key_labels(keys: Iterable[str]) -> dict[str, str]:
    """Map each result key to a display label, e.g. ``"dose (tag), beam 0"``."""
    labels = {}
    for group in group_result_keys(keys):
        for component, key in group.components:
            labels[key] = (
                group.label if component == FULL_COMPONENT else f"{group.label}, {component}"
            )
    return labels


# ----------------------------------------------------------------------
# Identifiers in open combo box popups
# ----------------------------------------------------------------------


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
