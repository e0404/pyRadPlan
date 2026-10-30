"""Startup splash screen reporting the GUI's loading progress.

Most of the GUI's startup time goes into importing the scientific stack (array
backends, dose engines, optimization, AI agents) before any window exists.  The
splash is shown as soon as a ``QApplication`` exists and advances through the
remaining startup phases, so a slow (e.g. first-after-reboot) start gives
visible feedback.  It deliberately imports nothing beyond PySide6 and the asset
helper, so it can appear before the heavy modules are loaded.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QGuiApplication, QPainter, QPalette, QPen, QPixmap
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import QApplication, QWidget

from pyRadPlan.gui.assets import asset_path

_LOGO = "pyradplan_logo_full_landscape.svg"

# Logical size (px) of the splash and the layout of its contents.
_WIDTH = 520
_HEIGHT = 230
_MARGIN = 28
_LOGO_HEIGHT = 96
_BAR_HEIGHT = 8

#: Relative weights (roughly warm-start milliseconds) of the main-window phases run by
#: :func:`~pyRadPlan.gui.windows._main_win.launch_main_window`.
BUILD_WINDOW_WEIGHT = 400.0
SHOW_WINDOW_WEIGHT = 400.0


class StartupSplash(QWidget):
    """Splash screen with a status line and a determinate progress bar.

    Startup is split into weighted phases.  Each :meth:`step` completes the
    running phase and starts the next one, so the bar shows the share of the
    expected startup work already done, and the status line counts the phases
    (``n/N``).

    A frameless window rather than a ``QSplashScreen``: on Windows,
    ``QSplashScreen.show()`` blocks for about a second waiting for the window
    to be exposed, adding to the very startup time it is meant to cover.

    Parameters
    ----------
    total_weight:
        Sum of the weights of all phases that will be announced via :meth:`step`.
    total_steps:
        Number of phases that will be announced via :meth:`step`.  When ``None``,
        the status line shows the completed percentage instead of a phase count.
    """

    def __init__(self, total_weight: float = 1.0, total_steps: Optional[int] = None) -> None:
        super().__init__(None, Qt.WindowType.SplashScreen | Qt.WindowType.FramelessWindowHint)
        self._background = _splash_pixmap()
        self._total = max(float(total_weight), 1e-9)
        self._done = 0.0
        self._current = 0.0
        self._message = ""
        self._steps = 0
        self._total_steps = total_steps
        self.setFixedSize(_WIDTH, _HEIGHT)
        screen = QGuiApplication.primaryScreen()
        if screen is not None:
            self.move(screen.availableGeometry().center() - self.rect().center())

    def progress(self) -> float:
        """Return the completed share of the startup work, in ``[0, 1]``."""
        return min(self._done / self._total, 1.0)

    def message(self) -> str:
        """Return the status line of the running phase."""
        return self._message

    def counter(self) -> str:
        """Return the phase count (``n/N``), or the percentage without a step total."""
        if self._total_steps is None:
            return f"{round(100 * self.progress())} %"
        return f"{min(self._steps, self._total_steps)}/{self._total_steps}"

    def set_message(self, message: str) -> None:
        """Update the status line of the running phase, e.g. with its progress."""
        self._message = message
        self._refresh()

    def step(self, message: str, weight: float = 1.0) -> None:
        """Complete the running phase and start a new one.

        Parameters
        ----------
        message:
            Status line shown while the new phase runs.
        weight:
            Relative duration of the new phase, in the units of ``total_weight``.
        """
        self._done += self._current
        self._current = max(float(weight), 0.0)
        self._message = message
        self._steps += 1
        self._refresh()

    def finish(self, window: QWidget) -> None:
        """Fill the bar, then close the splash once the shown *window* is drawn."""
        self._done = self._total
        self._current = 0.0
        self._refresh()
        window.repaint()
        self.close()

    def _refresh(self) -> None:
        # Startup blocks the event loop, so paint (and let the platform show the
        # window at all) right away instead of waiting for the next event pass.
        self.repaint()
        QApplication.processEvents()

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt override)
        """Draw the logo background, the status line and the progress bar."""
        palette = self.palette()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.drawPixmap(0, 0, self._background)

        bar = QRectF(_MARGIN, _HEIGHT - _MARGIN - _BAR_HEIGHT, _WIDTH - 2 * _MARGIN, _BAR_HEIGHT)
        text = QRectF(_MARGIN, bar.top() - 26, bar.width(), 20)

        painter.setPen(palette.color(QPalette.ColorRole.WindowText))
        painter.drawText(
            text, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, self._message
        )
        painter.drawText(
            text,
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
            self.counter(),
        )

        radius = _BAR_HEIGHT / 2
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(
            _blend(
                palette.color(QPalette.ColorRole.WindowText),
                palette.color(QPalette.ColorRole.Window),
                0.15,
            )
        )
        painter.drawRoundedRect(bar, radius, radius)
        if self.progress() > 0:
            filled = QRectF(bar)
            filled.setWidth(max(bar.width() * self.progress(), _BAR_HEIGHT))
            painter.setBrush(palette.color(QPalette.ColorRole.Highlight))
            painter.drawRoundedRect(filled, radius, radius)
        painter.end()


def _blend(foreground: QColor, background: QColor, weight: float) -> QColor:
    """Mix *weight* of *foreground* into *background*."""
    return QColor.fromRgbF(
        *(
            weight * fg + (1.0 - weight) * bg
            for fg, bg in zip(foreground.getRgbF()[:3], background.getRgbF()[:3])
        )
    )


def _splash_pixmap() -> QPixmap:
    """Render the splash background: palette window color, border and the pyRadPlan logo."""
    palette = QApplication.palette()
    screen = QGuiApplication.primaryScreen()
    ratio = screen.devicePixelRatio() if screen is not None else 1.0

    pixmap = QPixmap(round(_WIDTH * ratio), round(_HEIGHT * ratio))
    pixmap.setDevicePixelRatio(ratio)
    pixmap.fill(palette.color(QPalette.ColorRole.Window))

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(QPen(palette.color(QPalette.ColorRole.Mid), 1))
    painter.drawRect(QRectF(0.5, 0.5, _WIDTH - 1, _HEIGHT - 1))

    renderer = QSvgRenderer(str(asset_path("logos", _LOGO)))
    size = renderer.defaultSize()
    aspect = size.width() / size.height() if size.height() else 1.0
    logo_width = min(_WIDTH - 2 * _MARGIN, _LOGO_HEIGHT * aspect)
    renderer.render(
        painter,
        QRectF((_WIDTH - logo_width) / 2, _MARGIN, logo_width, logo_width / aspect),
    )
    painter.end()
    return pixmap
