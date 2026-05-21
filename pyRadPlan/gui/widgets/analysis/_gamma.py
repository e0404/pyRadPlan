"""Gamma analysis widget."""

from __future__ import annotations

from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget


class GammaWidget(QWidget):
    """Widget displaying Gamma analysis (dummy)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        label = QLabel("Gamma Analysis (Not Implemented).")
        label2 = QLabel("(Updated as soon as Viewer supports dose comparison)")
        layout.addWidget(label)
        layout.addWidget(label2)
