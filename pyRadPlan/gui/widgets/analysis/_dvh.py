"""DVH plotting widget."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtWidgets import QVBoxLayout, QWidget
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

if TYPE_CHECKING:
    from pyRadPlan.analysis._dvh import DVH


class DVHPlotWidget(QWidget):
    """Widget displaying DVH plot."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.figure = Figure(figsize=(5, 4), dpi=100, layout="constrained")
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

    def plot(
        self,
        dvhs_q1: list[DVH],
        dvhs_q2: list[DVH] | None = None,
        voi_colors: dict[str, tuple[int, int, int]] | None = None,
        overlay_unit: str = "",
        overlay_label: str = "",
        q1_label: str = "Primary",
        q2_label: str = "Secondary",
    ) -> None:
        """Plot primary (solid) and optional secondary (dotted) DVH curves.

        The legend is placed outside the axes on the right. It contains one
        colored line per plotted VOI, plus line-style indicator rows for Q1
        (solid black) and, when present, Q2 (dotted black).
        """
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        def _get_color(name: str) -> tuple[float, float, float] | str:
            if voi_colors and name in voi_colors:
                r, g, b = voi_colors[name]
                return (r / 255, g / 255, b / 255)
            return "gray"

        # Track which VOI names were actually plotted (preserving order)
        plotted: list[str] = []

        # --- Q1: solid lines ---
        for dvh in dvhs_q1:
            color = _get_color(dvh.name)
            ax.plot(dvh.bins, dvh.cum_volume, color=color, linewidth=2, linestyle="-")
            if dvh.name not in plotted:
                plotted.append(dvh.name)

        # --- Q2: dotted lines ---
        if dvhs_q2:
            for dvh in dvhs_q2:
                color = _get_color(dvh.name)
                ax.plot(dvh.bins, dvh.cum_volume, color=color, linewidth=2, linestyle=":")
                if dvh.name not in plotted:
                    plotted.append(dvh.name)

        # --- Axes labels and grid ---
        name = overlay_label or "Dose"
        ax.set_xlabel(f"{name} [{overlay_unit}]" if overlay_unit else name)
        ax.set_ylabel("Volume [%]")
        ax.grid(True, which="both", linestyle="--", alpha=0.7)

        # --- Custom external legend ---
        # One colored line per plotted VOI
        voi_handles = [Line2D([0], [0], color=_get_color(n), lw=2, label=n) for n in plotted]

        # Line-style indicator rows
        style_handles = [Line2D([0], [0], color="black", lw=2, linestyle="-", label=q1_label)]
        if dvhs_q2:
            style_handles.append(
                Line2D([0], [0], color="black", lw=2, linestyle=":", label=q2_label)
            )

        all_handles = voi_handles + style_handles
        if all_handles:
            ax.legend(
                handles=all_handles,
                loc="upper left",
                bbox_to_anchor=(1.02, 1),
                borderaxespad=0,
                frameon=True,
            )

        self.canvas.draw()
