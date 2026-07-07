"""Logo / branding banner widget (matRad's ``matRad_LogoWidget`` equivalent).

Shows the prominent pyRadPlan logo on the right and the smaller DKFZ logo on the
left, with the non-clinical-use disclaimer underneath the DKFZ mark.  Both logos
are rendered from bundled SVG assets so they stay crisp at any size.  The DKFZ
logo switches between its blue (light theme) and white (dark theme) variant
automatically based on the active palette.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QPalette
from PySide6.QtSvgWidgets import QSvgWidget
from PySide6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget

from pyRadPlan.gui.assets import asset_path

_PYRADPLAN_LOGO = "pyradplan_logo_full_landscape.svg"
_DKFZ_LOGO_LIGHT = "dkfz_logo_blue.svg"
_DKFZ_LOGO_DARK = "dkfz_logo_white.svg"

# Reference heights (px) driving each logo's size; pyRadPlan is the larger mark.
_PYRADPLAN_HEIGHT = 64
_DKFZ_HEIGHT = 40


class _SvgLogo(QSvgWidget):
    """SVG logo rendered centered and undistorted at a fixed reference height.

    ``QSvgWidget`` reports the SVG's native size as its size hint, which for the
    logos is far too large; ``ref_height`` pins the logo to a sensible height and
    lets the aspect ratio drive its width.
    """

    def __init__(self, filename: str, ref_height: int, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._ref_height = ref_height
        self.setMinimumHeight(0)
        self.setMaximumHeight(ref_height)
        self.set_logo(filename)

    def set_logo(self, filename: str) -> None:
        """Load *filename* from the logo assets, keeping the aspect ratio."""
        self.load(str(asset_path("logos", filename)))
        self.renderer().setAspectRatioMode(Qt.KeepAspectRatio)
        size = self.renderer().defaultSize()
        self._aspect = size.width() / size.height() if size.height() else 1.0

    def sizeHint(self) -> QSize:  # noqa: N802 (Qt override)
        return QSize(round(self._ref_height * self._aspect), self._ref_height)

    def heightForWidth(self, width: int) -> int:  # noqa: N802 (Qt override)
        return round(width / self._aspect)

    def hasHeightForWidth(self) -> bool:  # noqa: N802 (Qt override)
        return True


class LogoWidget(QWidget):
    """Branding banner shown at the top of the main window."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(16)

        # Left: smaller DKFZ logo (theme variant) with the disclaimer beneath it,
        # pinned to the top-left corner so it stays anchored above the slice
        # viewer even when the banner is stretched wide.
        left = QVBoxLayout()
        left.setContentsMargins(0, 0, 0, 0)
        left.setSpacing(2)

        self._dkfz = _SvgLogo(self._dkfz_variant(), _DKFZ_HEIGHT)
        left.addWidget(self._dkfz, alignment=Qt.AlignLeft | Qt.AlignTop)

        disclaimer = QLabel("NOT FOR CLINICAL USE!")
        disclaimer.setAlignment(Qt.AlignLeft)
        disclaimer.setStyleSheet("color: #c0392b; font-weight: bold;")
        left.addWidget(disclaimer, alignment=Qt.AlignLeft | Qt.AlignTop)
        left.addStretch()

        layout.addLayout(left, 1)

        # Right: prominent pyRadPlan logo.
        self._logo = _SvgLogo(_PYRADPLAN_LOGO, _PYRADPLAN_HEIGHT)
        layout.addWidget(self._logo, 0, alignment=Qt.AlignRight | Qt.AlignTop)

    def _dkfz_variant(self) -> str:
        """Pick the DKFZ logo matching the current palette (dark vs light)."""
        window = self.palette().color(QPalette.Window)
        is_dark = window.lightness() < 128
        return _DKFZ_LOGO_DARK if is_dark else _DKFZ_LOGO_LIGHT

    def changeEvent(self, event) -> None:  # noqa: N802 (Qt override)
        """Swap the DKFZ logo when the application palette/theme changes."""
        if event.type() == event.Type.PaletteChange and hasattr(self, "_dkfz"):
            self._dkfz.set_logo(self._dkfz_variant())
        super().changeEvent(event)
