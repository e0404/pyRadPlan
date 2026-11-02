"""View menu for the pyRadPlan main window.

Collects the slice-viewer and appearance actions that previously lived on the
matRad-style toolbar.  Apart from the dark mode switch they are currently
disabled stubs kept for layout parity with matRad; wire them up as the
corresponding features land.
"""

from __future__ import annotations

from typing import Optional, Sequence

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QAction, QColor, QGuiApplication, QPalette
from PySide6.QtWidgets import QApplication, QMenu, QWidget

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
        switch = _color_scheme_switch()
        self.dark_mode_action.setChecked(switch.dark)
        # ``triggered`` fires on user clicks only, so following an OS change does not
        # pin the application to an explicit scheme that stops following the OS.
        self.dark_mode_action.triggered.connect(set_dark_mode)
        switch.changed.connect(self.dark_mode_action.setChecked)


#: Palette colors of the Fusion fallback, for dark (True) and light (False) mode.
_FALLBACK_COLORS = {
    True: {
        QPalette.ColorRole.Window: "#353535",
        QPalette.ColorRole.WindowText: "#f0f0f0",
        QPalette.ColorRole.Base: "#2a2a2a",
        QPalette.ColorRole.AlternateBase: "#424242",
        QPalette.ColorRole.Button: "#353535",
        QPalette.ColorRole.ButtonText: "#f0f0f0",
        QPalette.ColorRole.Text: "#f0f0f0",
        QPalette.ColorRole.BrightText: "#ff5555",
        QPalette.ColorRole.PlaceholderText: "#a0a0a0",
        QPalette.ColorRole.ToolTipBase: "#353535",
        QPalette.ColorRole.ToolTipText: "#f0f0f0",
        QPalette.ColorRole.Highlight: "#2a82da",
        QPalette.ColorRole.HighlightedText: "#ffffff",
        QPalette.ColorRole.Link: "#56a0e8",
    },
    False: {
        QPalette.ColorRole.Window: "#efefef",
        QPalette.ColorRole.WindowText: "#000000",
        QPalette.ColorRole.Base: "#ffffff",
        QPalette.ColorRole.AlternateBase: "#f7f7f7",
        QPalette.ColorRole.Button: "#efefef",
        QPalette.ColorRole.ButtonText: "#000000",
        QPalette.ColorRole.Text: "#000000",
        QPalette.ColorRole.BrightText: "#ffffff",
        QPalette.ColorRole.PlaceholderText: "#7f7f7f",
        QPalette.ColorRole.ToolTipBase: "#ffffdc",
        QPalette.ColorRole.ToolTipText: "#000000",
        QPalette.ColorRole.Highlight: "#308cc6",
        QPalette.ColorRole.HighlightedText: "#ffffff",
        QPalette.ColorRole.Link: "#0000ff",
    },
}
_FALLBACK_DISABLED_TEXT = {True: "#7f7f7f", False: "#bebebe"}


def _fallback_palette(dark: bool) -> QPalette:
    colors = _FALLBACK_COLORS[dark]
    # Derives the bevel shades (Light, Mid, Dark, Shadow) Fusion draws with.
    palette = QPalette(
        QColor(colors[QPalette.ColorRole.Button]), QColor(colors[QPalette.ColorRole.Window])
    )
    for role, color in colors.items():
        palette.setColor(role, QColor(color))
    for role in (
        QPalette.ColorRole.WindowText,
        QPalette.ColorRole.Text,
        QPalette.ColorRole.ButtonText,
    ):
        palette.setColor(QPalette.ColorGroup.Disabled, role, QColor(_FALLBACK_DISABLED_TEXT[dark]))
    return palette


def _platform_dark() -> bool:
    """Whether the platform currently renders the dark scheme (never on Qt < 6.5)."""
    hints = QGuiApplication.styleHints()
    return hasattr(hints, "colorScheme") and hints.colorScheme() == Qt.ColorScheme.Dark


class _ColorSchemeSwitch(QObject):
    """Application-wide dark mode, rendered by the platform or by a Fusion fallback.

    The scheme is requested from the platform (Qt >= 6.8).  Platforms without a theme
    plugin that implements the request (bare X11/Wayland sessions, minimal window
    managers, Qt < 6.8) ignore it; then the application switches to the Fusion style
    with a matching palette until the platform renders the wanted scheme itself.
    """

    #: The application turned dark (True) or light (False).
    changed = Signal(bool)

    def __init__(self, app: QApplication) -> None:
        super().__init__(app)
        #: Style the fallback replaced; None while the platform renders the scheme.
        self._native_style: Optional[str] = None
        self._dark = _platform_dark()
        hints = app.styleHints()
        if hasattr(hints, "colorSchemeChanged"):
            hints.colorSchemeChanged.connect(self._on_platform_scheme)

    @property
    def dark(self) -> bool:
        return self._dark

    def set_dark(self, dark: bool) -> None:
        hints = QGuiApplication.styleHints()
        if hasattr(hints, "setColorScheme"):
            hints.setColorScheme(Qt.ColorScheme.Dark if dark else Qt.ColorScheme.Light)
        if _platform_dark() == dark:
            self._restore_native()
        else:
            self._apply_fallback(dark)
        self._set(dark)

    def _on_platform_scheme(self, scheme: Qt.ColorScheme) -> None:
        dark = scheme == Qt.ColorScheme.Dark
        if self._native_style is not None:
            if dark != self._dark:
                return  # the user's choice overrides the platform
            self._restore_native()  # the platform caught up with the request
        self._set(dark)

    def _apply_fallback(self, dark: bool) -> None:
        app = QApplication.instance()
        if self._native_style is None:
            self._native_style = app.style().name()
        app.setStyle("Fusion")
        app.setPalette(_fallback_palette(dark))

    def _restore_native(self) -> None:
        if self._native_style is None:
            return
        app = QApplication.instance()
        app.setStyle(self._native_style)
        # An empty palette resolves to the platform's and keeps following it.
        app.setPalette(QPalette())
        self._native_style = None

    def _set(self, dark: bool) -> None:
        if dark != self._dark:
            self._dark = dark
            self.changed.emit(dark)


def _color_scheme_switch() -> _ColorSchemeSwitch:
    app = QApplication.instance()
    switch = app.findChild(_ColorSchemeSwitch)
    return switch if switch is not None else _ColorSchemeSwitch(app)


def set_dark_mode(dark: bool) -> None:
    """Switch the application between the dark and the light color scheme."""
    _color_scheme_switch().set_dark(dark)
