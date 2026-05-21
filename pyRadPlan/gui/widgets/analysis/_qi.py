"""QI table widget."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QHeaderView,
    QTableWidget,
    QVBoxLayout,
    QWidget,
)
from typing import Any


class QITableWidget(QWidget):
    """Widget displaying QI table."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.table = QTableWidget()
        self.table.setColumnCount(7)
        self.table.setHorizontalHeaderLabels(
            ["Structure", "Mean", "Std", "Min", "Max", "Vx", "Dx"]
        )
        header = self.table.horizontalHeader()
        if hasattr(header, "setSectionResizeMode"):
            header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

        layout.addWidget(self.table)

    def update(self, qi: Any) -> None:
        pass
