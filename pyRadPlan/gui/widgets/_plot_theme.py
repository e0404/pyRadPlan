"""Plot colors derived from the Qt palette, so plots follow the GUI's light/dark scheme."""

from __future__ import annotations

from dataclasses import dataclass

import pyqtgraph as pg
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QWidget

#: Share of the foreground color in the grid color (the rest is the axes background).
_GRID_WEIGHT = 0.3


@dataclass(frozen=True)
class PlotColors:
    """Colors of a plot as ``#rrggbb`` strings, as pyqtgraph accepts them."""

    #: Area around the axes (palette ``Window``).
    figure: str
    #: Data area inside the axes (palette ``Base``).
    axes: str
    #: Text, ticks and axis lines (palette ``WindowText``).
    foreground: str
    #: Grid lines, a blend of foreground and axes background.
    grid: str


def plot_colors(widget: QWidget) -> PlotColors:
    """Return the plot colors matching the current palette of *widget*."""
    palette = widget.palette()
    base = palette.color(QPalette.ColorRole.Base)
    text = palette.color(QPalette.ColorRole.WindowText)
    grid = QColor.fromRgbF(
        *(
            _GRID_WEIGHT * fg + (1.0 - _GRID_WEIGHT) * bg
            for fg, bg in zip(text.getRgbF()[:3], base.getRgbF()[:3])
        )
    )
    return PlotColors(
        figure=palette.color(QPalette.ColorRole.Window).name(),
        axes=base.name(),
        foreground=text.name(),
        grid=grid.name(),
    )


def apply_pyqtgraph_colors(layout: pg.GraphicsLayoutWidget, colors: PlotColors) -> None:
    """Restyle *layout* and the plots it contains with *colors*."""
    layout.setBackground(colors.figure)
    for item in layout.ci.items:
        if not isinstance(item, pg.PlotItem):
            continue
        item.getViewBox().setBackgroundColor(colors.axes)
        for name in ("left", "bottom", "right", "top"):
            axis = item.getAxis(name)
            axis.setPen(colors.foreground)
            axis.setTextPen(colors.foreground)
        if item.titleLabel.text:
            item.setTitle(item.titleLabel.text, color=colors.foreground)
