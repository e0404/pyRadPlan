"""Modal dialog to choose which result quantities to export to disk."""

from __future__ import annotations

from typing import Optional

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from .._result_widget import QUANTITY_META
from ._labels import TruncatedCheckBox


class SaveResultDialog(QDialog):
    """Let the user pick one or more result quantities to save.

    The dialog lists *quantities* (the image-like keys of the current result) as
    checkboxes.  ``selected_quantities`` returns the checked keys after the dialog
    is accepted; at least one must be checked to accept.

    Parameters
    ----------
    quantities:
        The result quantity keys offered for export.
    parent:
        Optional Qt parent widget.
    """

    def __init__(self, quantities: list[str], parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Save Result")
        self._checkboxes: dict[str, TruncatedCheckBox] = {}

        root = QVBoxLayout(self)
        root.addWidget(QLabel("Select the quantities to export:"))

        # A scroll area keeps the dialog compact when a result has many quantities.
        container = QWidget()
        col = QVBoxLayout(container)
        col.setContentsMargins(4, 4, 4, 4)
        for key in quantities:
            label, _unit = QUANTITY_META.get(key, ("", ""))
            text = f"{label} ({key})" if label else key
            checkbox = TruncatedCheckBox(text)
            self._checkboxes[key] = checkbox
            col.addWidget(checkbox)
        col.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(container)
        root.addWidget(scroll)

        # Quick select-all / clear helpers.
        quick = QHBoxLayout()
        btn_all = QPushButton("All")
        btn_all.clicked.connect(lambda: self._set_all(True))
        btn_none = QPushButton("None")
        btn_none.clicked.connect(lambda: self._set_all(False))
        quick.addWidget(btn_all)
        quick.addWidget(btn_none)
        quick.addStretch()
        root.addLayout(quick)

        self._status = QLabel("")
        self._status.setStyleSheet("color: #c0392b;")
        root.addWidget(self._status)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def _set_all(self, checked: bool) -> None:
        for checkbox in self._checkboxes.values():
            checkbox.setChecked(checked)

    def selected_quantities(self) -> list[str]:
        """Return the keys of the checked quantities."""
        return [key for key, cb in self._checkboxes.items() if cb.isChecked()]

    def accept(self) -> None:  # noqa: D102 (Qt override)
        if not self.selected_quantities():
            self._status.setText("Select at least one quantity to export.")
            return
        super().accept()
