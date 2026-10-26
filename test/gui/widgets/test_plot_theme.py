import pytest

pytest.importorskip("PySide6")

from PySide6.QtGui import QColor
from PySide6.QtWidgets import QWidget

from pyRadPlan.gui.widgets._plot_theme import plot_colors


def test_plot_colors_follow_palette(qapp, dark_palette):
    palette, expected = dark_palette
    widget = QWidget()
    widget.setPalette(palette)

    colors = plot_colors(widget)

    assert (colors.figure, colors.axes, colors.foreground) == (
        expected["window"],
        expected["base"],
        expected["text"],
    )
    # the grid lies between background and text, closer to the background
    base, text = QColor(expected["base"]).lightness(), QColor(expected["text"]).lightness()
    grid = QColor(colors.grid).lightness()
    assert base < grid < text
    assert grid - base < text - grid
