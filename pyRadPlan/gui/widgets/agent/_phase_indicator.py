"""Status dot that spins while its phase is running."""

from __future__ import annotations

from typing import Literal, Optional

from PySide6.QtCore import QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QWidget

_GRAY = QColor("#888888")
_GREEN = QColor("#27ae60")

PhaseState = Literal["idle", "active", "done"]


class PhaseIndicator(QWidget):
    """A status dot that turns into a spinning circle while its phase is running.

    Gray ring: not run yet. Spinning green arc: running. Green dot: ran at least once.
    """

    _STEP_DEG = 30
    _INTERVAL_MS = 60

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setFixedSize(14, 14)
        self._state: PhaseState = "idle"
        self._angle = 0
        self._timer = QTimer(self)
        self._timer.setInterval(self._INTERVAL_MS)
        self._timer.timeout.connect(self._spin)

    @property
    def state(self) -> PhaseState:
        """``"idle"``, ``"active"`` or ``"done"``."""
        return self._state

    def set_state(self, state: PhaseState) -> None:
        """Show the phase as not run yet, running or done."""
        self._state = state
        if state == "active":
            self._timer.start()
        else:
            self._timer.stop()
        self.update()

    def _spin(self) -> None:
        self._angle = (self._angle + self._STEP_DEG) % 360
        self.update()

    def paintEvent(self, _event) -> None:  # noqa: N802 (Qt override)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(2.5, 2.5, 9, 9)
        if self._state == "active":
            pen = QPen(_GREEN, 2.2)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            painter.drawArc(rect, -self._angle * 16, 270 * 16)
        elif self._state == "done":
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(_GREEN)
            painter.drawEllipse(rect)
        else:
            painter.setPen(QPen(_GRAY, 1.4))
            painter.drawEllipse(rect)
