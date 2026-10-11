"""Patient orientation marker shown in a corner of the slice viewer."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QWidget

#: The mannequin as capsules ``(end, end, radius)`` in patient LPS coordinates
#: (x towards the patient's left, y posterior, z superior), standing about 180
#: units tall.  Equal ends make a sphere.  Nose and toes point anterior (-y).
_FIGURE: tuple[tuple[tuple[float, float, float], tuple[float, float, float], float], ...] = (
    ((0, 0, 75), (0, 0, 75), 11.0),  # head
    ((0, -12, 74), (0, -12, 74), 4.0),  # nose
    ((0, 0, 56), (0, 0, 66), 5.0),  # neck
    ((-7, 0, 12), (-7, 0, 46), 10.0),  # torso, right half
    ((7, 0, 12), (7, 0, 46), 10.0),  # torso, left half
    ((-20, 0, 49), (-24, 0, 6), 4.5),  # right arm
    ((20, 0, 49), (24, 0, 6), 4.5),  # left arm
    ((-8, 0, 2), (-9, 0, -78), 6.0),  # right leg
    ((8, 0, 2), (9, 0, -78), 6.0),  # left leg
    ((-9, 0, -84), (-9, -14, -86), 4.0),  # right foot
    ((9, 0, -84), (9, -14, -86), 4.0),  # left foot
)

#: Patient direction letters for the negative and positive LPS axes.
_LETTERS = (("R", "L"), ("A", "P"), ("I", "S"))

#: Camera distance from the figure's centre, in figure units; the mild
#: perspective draws nearer parts larger, which tells near from far.
_CAMERA_DISTANCE = 280.0
#: Parts centred further than this in front of the figure's centre are cut away,
#: as the displayed slice cuts away the anatomy in front of it.  Seen from the
#: feet this leaves a cross-section of the trunk instead of a view onto the soles.
_CUT_DEPTH = 12.0
#: Parts whose depths differ by less than this are drawn as one silhouette.
_SAME_DEPTH = 1.5
#: On-screen length of the figure's longer side, and the margin holding the letters.
_FIGURE_PX = 104
_LETTER_MARGIN = 13
_NEAR_SHADE = 240
_FAR_SHADE = 120


def _direction_letter(direction: np.ndarray) -> str:
    axis = int(np.argmax(np.abs(direction)))
    return _LETTERS[axis][int(direction[axis] > 0)]


class OrientationMarker(QWidget):
    """Small mannequin showing how the patient lies in the displayed view.

    The figure is projected onto the screen axes of the view, seen from in front
    of the screen with a mild perspective and nearer body parts drawn lighter
    and on top, so a single model serves every plane.  Like the slice, it is cut
    at its centre, so parts well in front of it are left out.  The widget takes the
    figure's on-screen shape, e.g. long and flat for a patient lying in a
    sagittal view.  The letters name the patient direction of each screen edge
    (e.g. ``L`` on the right for the patient's left).
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._parts: list[tuple[float, np.ndarray, np.ndarray, float]] = []
        self.set_view((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))

    def set_view(self, right: Sequence[float], down: Sequence[float]) -> None:
        """Set the patient (LPS) directions of the screen's right and downward axes."""
        self._right = np.asarray(right, dtype=float)
        self._down = np.asarray(down, dtype=float)
        self._parts = self._project()
        self.update()

    def edge_letters(self) -> dict[str, str]:
        """Patient direction letter of each screen edge."""
        return {
            "right": _direction_letter(self._right),
            "left": _direction_letter(-self._right),
            "bottom": _direction_letter(self._down),
            "top": _direction_letter(-self._down),
        }

    def _project(self) -> list[tuple[float, np.ndarray, np.ndarray, float]]:
        """Project the figure into widget pixels and fit the widget around it.

        Returns ``(depth, start, end, radius)`` per part, farthest first.
        """
        into_screen = np.cross(self._right, self._down)
        axes = np.stack([self._right, self._down])
        parts = []
        for start_lps, end_lps, radius in _FIGURE:
            ends = np.array([start_lps, end_lps], dtype=float)
            depths = ends @ into_screen
            if depths.mean() < -_CUT_DEPTH:
                continue
            factors = _CAMERA_DISTANCE / (_CAMERA_DISTANCE + depths)
            start, end = (ends @ axes.T) * factors[:, None]
            parts.append((float(depths.mean()), start, end, radius * float(factors.mean())))
        parts.sort(key=lambda part: -part[0])

        corners = np.array([p + s * r for _, a, b, r in parts for p in (a, b) for s in (-1, 1)])
        low, high = corners.min(axis=0), corners.max(axis=0)
        scale = _FIGURE_PX / float(max(high - low))
        size = np.ceil(scale * (high - low)) + 2 * _LETTER_MARGIN
        self.setFixedSize(int(size[0]), int(size[1]))
        offset = size / 2.0 - scale * (low + high) / 2.0
        return [
            (depth, offset + scale * a, offset + scale * b, scale * r) for depth, a, b, r in parts
        ]

    def paintEvent(self, _event) -> None:  # noqa: N802 (Qt override)
        """Draw the background, the projected mannequin and the edge letters."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(0, 0, 0, 150))
        painter.drawRoundedRect(QRectF(self.rect()), 4, 4)

        depths = [depth for depth, *_ in self._parts]
        span = max(max(depths) - min(depths), 1e-9)
        groups: list[list] = []
        for part in self._parts:
            if groups and groups[-1][0][0] - part[0] < _SAME_DEPTH:
                groups[-1].append(part)
            else:
                groups.append([part])
        for group in groups:
            weight = (max(depths) - group[0][0]) / span
            shade = round(_FAR_SHADE + weight * (_NEAR_SHADE - _FAR_SHADE))
            # Outlines first, so nearer parts stand out from those behind them.
            for _, start, end, radius in group:
                _draw_capsule(painter, start, end, radius + 0.75, QColor(30, 30, 30))
            for _, start, end, radius in group:
                _draw_capsule(painter, start, end, radius, QColor(shade, shade, shade))

        font = QFont(self.font())
        font.setPointSizeF(7.5)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QColor(255, 255, 255))
        letters = self.edge_letters()
        m, w, h = float(_LETTER_MARGIN), float(self.width()), float(self.height())
        rects = {
            "left": QRectF(0, 0, m, h),
            "right": QRectF(w - m, 0, m, h),
            "top": QRectF(0, 0, w, m),
            "bottom": QRectF(0, h - m, w, m),
        }
        for edge, rect in rects.items():
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, letters[edge])
        painter.end()


def _draw_capsule(
    painter: QPainter, start: np.ndarray, end: np.ndarray, radius: float, color: QColor
) -> None:
    a, b = QPointF(*start), QPointF(*end)
    if a == b:
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(color)
        painter.drawEllipse(a, radius, radius)
    else:
        pen = QPen(color, 2.0 * radius)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.drawLine(a, b)
