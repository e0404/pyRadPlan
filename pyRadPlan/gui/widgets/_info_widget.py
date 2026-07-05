"""Info widget showing the pyRadPlan version and a short about note.

matRad's InfoWidget equivalent.  Static (no workspace listener needed).
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget


def _package_version() -> str:
    try:
        return version("pyRadPlan")
    except PackageNotFoundError:
        return "unknown"


class InfoWidget(QWidget):
    """Display the package version and project link."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(2)

        ver = QLabel(f"pyRadPlan v{_package_version()}")
        ver.setAlignment(Qt.AlignCenter)
        ver.setStyleSheet("font-weight: bold;")

        link = QLabel(
            '<a href="https://github.com/e0404/pyRadPlan">github.com/e0404/pyRadPlan</a>'
        )
        link.setAlignment(Qt.AlignCenter)
        link.setOpenExternalLinks(True)
        link.setTextInteractionFlags(Qt.TextBrowserInteraction)

        layout.addWidget(ver)
        layout.addWidget(link)
