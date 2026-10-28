"""Human-readable names for quantities, result keys and biological models.

The GUI lists quantities and models by their display name and keeps the
identifier as item data.  :func:`show_identifiers_when_open` additionally
prints the identifier faintly next to each entry of an open combo box popup.
"""

from __future__ import annotations

import re
from collections import Counter
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

from pyRadPlan.quantities import get_available_quantities

#: Item data role holding the identifier shown next to an item in an open popup.
IDENTIFIER_ROLE = Qt.ItemDataRole.UserRole + 101

#: Component label of the total (all beams) entry of a result group.
FULL_COMPONENT = "full"

#: Flattened list entries of the viewer (``"<key> <i>"``).
_INDEXED_KEY = re.compile(r"^(?P<stem>.+) (?P<index>\d+)$")
_BEAM_SUFFIX = "_beam"


def quantity_display_name(key: str) -> str:
    """Return the display name of a quantity identifier or result key.

    Registered quantity identifiers map to :attr:`RTQuantity.name`, including
    per-beam result keys (``"<identifier>_beam"`` and their flattened
    ``"<identifier>_beam <i>"`` entries).  Any other key (``alpha``, imported
    doses, matRad names) is returned unchanged; use :func:`group_result_keys`
    to also resolve snapshot tags.
    """
    base, sep, index = key.partition(" ")
    per_beam = base.endswith(_BEAM_SUFFIX)
    quantity = get_available_quantities().get(base.removesuffix(_BEAM_SUFFIX))
    if quantity is None:
        return key
    name = f"{quantity.name} beam" if per_beam else quantity.name
    return f"{name}{sep}{index}"


def bio_model_display_name(model_cls: Any) -> str:
    """Return the display name of a biological model class (or instance)."""
    return getattr(model_cls, "name", None) or model_cls.model


# ----------------------------------------------------------------------
# Result key grouping
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class ResultGroup:
    """Result entries belonging to one quantity of one (saved) result.

    Attributes
    ----------
    base:
        Quantity identifier or, for unregistered entries, the key stem.
    tag:
        Snapshot tag of a saved result; empty for the current result.
    label:
        Display label, e.g. ``"RBExDose"`` or ``"dose (protons_30fx)"``.
    components:
        ``(label, key)`` pairs: :data:`FULL_COMPONENT` for the total, then
        ``"beam <i>"`` for each per-beam entry.
    """

    base: str
    tag: str
    label: str
    components: tuple[tuple[str, str], ...]

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(key for _, key in self.components)

    @property
    def identifier(self) -> str:
        """Key of the total entry, or the group stem when there is none."""
        for label, key in self.components:
            if label == FULL_COMPONENT:
                return key
        return f"{self.base}_{self.tag}" if self.tag else self.base


def _split_index(key: str) -> tuple[str, int | None]:
    match = _INDEXED_KEY.match(key)
    if match is None:
        return key, None
    return match["stem"], int(match["index"])


def _snapshot_tags(stems: set[str]) -> list[str]:
    """Deduce snapshot tags (``"<key>_<tag>"``) from the key suffixes.

    A candidate is the remainder after a registered quantity identifier (and
    an optional ``_beam``); it counts as a tag only when at least two keys
    carry it, which tells a saved result (every key is copied) apart from a
    single suffixed entry such as ``physical_dose_var``.
    """
    identifiers = sorted(get_available_quantities(), key=len, reverse=True)
    canonical = {stem for name in identifiers for stem in (name, name + _BEAM_SUFFIX)}
    # A tag such as "dose" must not count the live physical_dose/rbe_x_dose
    # entries as evidence of a snapshot, or rename those entries later.
    stems = stems - canonical
    candidates: set[str] = set()
    for stem in stems:
        for identifier in identifiers:
            if stem == identifier:
                break
            if stem.startswith(identifier + "_"):
                rest = stem[len(identifier) + 1 :]
                if rest.startswith("beam_"):
                    rest = rest[len("beam_") :]
                if rest and rest != "beam":
                    candidates.add(rest)
                break
    counts = Counter(tag for tag in candidates for stem in stems if stem.endswith("_" + tag))
    return sorted((tag for tag, n in counts.items() if n >= 2), key=len, reverse=True)


def group_result_keys(keys: Iterable[str]) -> list[ResultGroup]:
    """Group result keys by quantity and saved result, deduced from their suffixes.

    ``"<base>[_beam][_<tag>][ <i>]"`` keys are grouped by ``(base, tag)``:
    the per-beam entries join the total of their quantity, and snapshot tags
    (see the workflow's *Save Result*) are split off.  Groups keep the order
    of first appearance.
    """
    keys = list(keys)
    tags = _snapshot_tags({_split_index(key)[0] for key in keys})
    quantities = get_available_quantities()

    entries: dict[tuple[str, str], list[tuple[tuple[int, int], str, str]]] = {}
    for key in keys:
        stem, index = _split_index(key)
        tag = ""
        if stem.removesuffix(_BEAM_SUFFIX) not in quantities:
            tag = next(
                (t for t in tags if stem.endswith("_" + t) and len(stem) > len(t) + 1),
                "",
            )
        if tag:
            stem = stem[: -len(tag) - 1]
        per_beam = stem.endswith(_BEAM_SUFFIX) and len(stem) > len(_BEAM_SUFFIX)
        if per_beam:
            stem = stem[: -len(_BEAM_SUFFIX)]
        if per_beam:
            label = "beam" if index is None else f"beam {index}"
        elif index is not None:
            label = f"[{index}]"
        else:
            label = FULL_COMPONENT
        order = (0 if label == FULL_COMPONENT else 1, -1 if index is None else index)
        entries.setdefault((stem, tag), []).append((order, label, key))

    groups = []
    for (base, tag), items in entries.items():
        label = quantity_display_name(base)
        if tag:
            label = f"{label} ({tag})"
        components = tuple((lbl, key) for _, lbl, key in sorted(items, key=lambda e: e[0]))
        groups.append(ResultGroup(base, tag, label, components))
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
