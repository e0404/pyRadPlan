"""Two-level result quantity selector: quantity (per saved result), then full / beam / scenario."""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QWidget

from ._display_names import (
    FULL_COMPONENT,
    SCENARIO_COMPONENT,
    ResultGroup,
    add_identified_item,
    group_result_keys,
    show_identifiers_when_open,
)


class QuantitySelector(QWidget):
    """Pick a result key by quantity group and component.

    The first combo box lists one entry per quantity and saved result (see
    :func:`~pyRadPlan.gui.widgets._display_names.group_result_keys`), the
    second one the total (``"full"``), the individual beam contributions and
    the individual uncertainty scenarios of the selected group.  Scenario
    entries show the labels given to :meth:`set_scenario_labels`.

    Parameters
    ----------
    none_label:
        If given, the quantity combo box starts with this entry, selecting no
        key (:meth:`current_key` returns ``""``).
    parent:
        Optional Qt parent widget.
    """

    #: Emitted with the selected result key ("" for the *none* entry).
    key_changed = Signal(str)

    def __init__(self, none_label: Optional[str] = None, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._none_label = none_label
        self._groups: list[ResultGroup] = []
        self._scenario_labels: list[str] = []
        #: Group whose components the component combo box lists.
        self._shown_group: Optional[ResultGroup] = None

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        self.quantity_combo = QComboBox()
        self.component_combo = QComboBox()
        self.component_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.component_combo.setToolTip(
            "Total of all beams, a single beam contribution or the result of a single "
            "uncertainty scenario"
        )
        for combo in (self.quantity_combo, self.component_combo):
            show_identifiers_when_open(combo)
        layout.addWidget(self.quantity_combo, 1)
        layout.addWidget(self.component_combo)

        self.quantity_combo.currentIndexChanged.connect(self._on_group_changed)
        self.component_combo.currentIndexChanged.connect(self._emit_key)
        self._fill_components(None, "")

    # ------------------------------------------------------------------

    def set_keys(self, keys: Iterable[str], active: Optional[str] = None) -> None:
        """Replace the selectable result keys, selecting *active* if present.

        Without a matching *active* key the first group (or the *none* entry)
        is selected.  Does not emit :attr:`key_changed`.
        """
        self._groups = group_result_keys(keys)
        self.quantity_combo.blockSignals(True)
        self.quantity_combo.clear()
        if self._none_label is not None:
            self.quantity_combo.addItem(self._none_label, -1)
        for position, group in enumerate(self._groups):
            add_identified_item(self.quantity_combo, group.label, position, group.identifier)
        group, component = self._locate(active)
        if group is None and self._none_label is None and self._groups:
            group = self._groups[0]
        self._select_group(group)
        self.quantity_combo.blockSignals(False)
        self._fill_components(group, component)

    def set_scenario_labels(self, labels: Sequence[str]) -> None:
        """Label the scenario components by flat scenario index, e.g. ``"setup (...) mm"``.

        Scenario entries then read ``"scenario <i>: <label>"``; indices without a label
        keep ``"scenario <i>"``.  Keeps the selection and does not emit
        :attr:`key_changed`.
        """
        labels = list(labels)
        if labels == self._scenario_labels:
            return
        self._scenario_labels = labels
        group, component = self._locate(self.current_key())
        if group is not None:
            self._fill_components(group, component)

    def scenario_keys(self) -> tuple[str, ...]:
        """Keys of the scenario components of the selected group (empty if none)."""
        group = self._current_group()
        return () if group is None else group.keys_of(SCENARIO_COMPONENT)

    def current_key(self) -> str:
        """Return the selected result key, or ``""`` if nothing is selected."""
        return self.component_combo.currentData() or ""

    def current_label(self) -> str:
        """Return a display label of the selected key, e.g. ``"dose, beam 0"``."""
        group = self._current_group()
        if group is None:
            return ""
        component = self.component_combo.currentText()
        return group.label if component == FULL_COMPONENT else f"{group.label}, {component}"

    def set_current_key(self, key: str) -> bool:
        """Select *key*; return whether it is available.  Emits :attr:`key_changed`."""
        group, component = self._locate(key)
        if group is None:
            return False
        self._select_group(group)
        self._fill_components(group, component)
        self._emit_key()
        return True

    # ------------------------------------------------------------------

    def _current_group(self) -> Optional[ResultGroup]:
        position = self.quantity_combo.currentData()
        if position is None or position < 0:
            return None
        return self._groups[position]

    def _select_group(self, group: Optional[ResultGroup]) -> None:
        position = -1 if group is None else self._groups.index(group)
        self.quantity_combo.blockSignals(True)
        self.quantity_combo.setCurrentIndex(max(self.quantity_combo.findData(position), 0))
        self.quantity_combo.blockSignals(False)

    def _locate(self, key: Optional[str]) -> tuple[Optional[ResultGroup], str]:
        for group in self._groups:
            for component, candidate in group.components:
                if candidate == key:
                    return group, component
        return None, ""

    def _fill_components(self, group: Optional[ResultGroup], component: str) -> None:
        self.component_combo.blockSignals(True)
        self.component_combo.clear()
        self._shown_group = group
        components = group.components if group is not None else ()
        kinds = group.kinds if group is not None and group.kinds else ("",) * len(components)
        for (label, key), kind in zip(components, kinds):
            add_identified_item(self.component_combo, self._component_text(label, kind), key, key)
        labels = [label for label, _ in components]
        self.component_combo.setCurrentIndex(labels.index(component) if component in labels else 0)
        self.component_combo.setEnabled(len(components) > 1)
        self.component_combo.blockSignals(False)

    def _component_text(self, label: str, kind: str) -> str:
        if kind != SCENARIO_COMPONENT:
            return label
        index = int(label.rpartition(" ")[2]) if label[-1:].isdigit() else -1
        if 0 <= index < len(self._scenario_labels) and self._scenario_labels[index]:
            return f"{label}: {self._scenario_labels[index]}"
        return label

    def _current_component(self) -> str:
        """Component label (e.g. ``"beam 1"``) of the selected key."""
        index = self.component_combo.currentIndex()
        group = self._shown_group
        if group is None or not 0 <= index < len(group.components):
            return ""
        return group.components[index][0]

    def _on_group_changed(self, _index: int) -> None:
        # Keep the chosen component (e.g. "beam 1") when the new group has it.
        self._fill_components(self._current_group(), self._current_component())
        self._emit_key()

    def _emit_key(self, *_: object) -> None:
        self.key_changed.emit(self.current_key())
