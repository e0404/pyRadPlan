"""View menu for the pyRadPlan main window.

Collects the slice-viewer and appearance actions that previously lived on the
matRad-style toolbar.  Apart from the dark mode switch they are currently
disabled stubs kept for layout parity with matRad; wire them up as the
corresponding features land.
"""

from __future__ import annotations

from typing import Optional, Sequence

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QGuiApplication
from PySide6.QtWidgets import QMenu, QWidget

#: (label, tooltip) for each not-yet-implemented view action.
_STUBS = (
    ("Screenshot", "Take a screenshot of the current view"),
    ("Zoom In", "Zoom in"),
    ("Zoom Out", "Zoom out"),
    ("Pan", "Pan"),
    ("Data Cursor", "Data cursor"),
    ("Toggle Legend", "Toggle legend"),
    ("Toggle Colorbar", "Toggle colorbar"),
)


class ViewMenu(QMenu):
    """The main window's *View* menu (slice-viewer and appearance actions).

    Parameters
    ----------
    parent:
        Optional Qt parent widget.
    panel_actions:
        Working actions listed before the stubs, e.g. dock-widget toggle
        actions from the main window.
    """

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        panel_actions: Sequence[QAction] = (),
    ) -> None:
        super().__init__("&View", parent)
        for action in panel_actions:
            self.addAction(action)
        if panel_actions:
            self.addSeparator()
        for label, tip in _STUBS:
            action = self.addAction(label)
            action.setToolTip(tip)
            action.setEnabled(False)
        self.addSeparator()
        self.dark_mode_action = self.addAction("Dark Mode")
        self.dark_mode_action.setToolTip("Switch between light and dark mode")
        self.dark_mode_action.setCheckable(True)
        hints = QGuiApplication.styleHints()
        self._sync_dark_mode(hints.colorScheme())
        self.dark_mode_action.toggled.connect(set_dark_mode)
        hints.colorSchemeChanged.connect(self._sync_dark_mode)

    def _sync_dark_mode(self, scheme: Qt.ColorScheme) -> None:
        """Tick the action whenever dark mode is on, set from any window or the OS."""
        self.dark_mode_action.setChecked(scheme == Qt.ColorScheme.Dark)


def set_dark_mode(dark: bool) -> None:
    """Switch the application between the dark and the light color scheme."""
    QGuiApplication.styleHints().setColorScheme(
        Qt.ColorScheme.Dark if dark else Qt.ColorScheme.Light
    )
