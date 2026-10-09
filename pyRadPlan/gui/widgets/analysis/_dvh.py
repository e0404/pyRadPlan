"""DVH plotting widget."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtWidgets import QVBoxLayout, QWidget
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from pyRadPlan.gui.widgets.analysis._units import compare_units

if TYPE_CHECKING:
    from matplotlib.axes import Axes

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

        self.secondary_axes: Axes | None = None

    def plot(
        self,
        dvhs: list[DVH],
        voi_colors: dict[str, tuple[int, int, int]] | None = None,
        overlay_unit: str = "",
        overlay_label: str = "",
        secondary_dvhs: list[DVH] | None = None,
        secondary_unit: str = "",
        secondary_label: str = "",
        primary_name: str = "",
        secondary_name: str = "",
    ) -> None:
        """Plot primary DVH curves solid and optional secondary curves dashed.

        Secondary curves use the same color per structure as the primary ones.
        If the secondary unit is compatible with the primary unit (see
        :func:`compare_units`), the secondary curves share the primary x axis
        and are rescaled to the primary unit. Otherwise they are drawn on a
        twin x axis shown at the top, stored in ``secondary_axes``.

        The legend is placed outside the axes on the right and lists the
        plotted VOI names in plot order. When a secondary quantity is drawn,
        two black rows follow that identify the solid and dashed line styles.
        No legend is drawn when nothing is plotted.

        Parameters
        ----------
        dvhs : list[DVH]
            Primary DVHs to plot as solid lines.
        voi_colors : dict[str, tuple[int, int, int]] | None, optional
            RGB colors (0-255) keyed by VOI name. VOIs without an entry are
            drawn in gray.
        overlay_unit : str, optional
            Unit of the primary quantity, shown in the x-axis label.
        overlay_label : str, optional
            Name of the primary quantity for the x-axis label. Defaults to
            "Dose" when empty.
        secondary_dvhs : list[DVH] | None, optional
            Secondary DVHs to plot as dashed lines. Nothing secondary is drawn
            when None or empty.
        secondary_unit : str, optional
            Unit of the secondary quantity.
        secondary_label : str, optional
            Name of the secondary quantity for the twin x-axis label. Defaults
            to "Dose" when empty. Only used when the units are incompatible.
        primary_name : str, optional
            Legend label of the solid line style. Defaults to "Primary".
        secondary_name : str, optional
            Legend label of the dashed line style. Defaults to "Secondary".
        """
        self.figure.clear()
        self.secondary_axes = None
        ax = self.figure.add_subplot(111)

        def _get_color(name: str) -> tuple[float, float, float] | str:
            if voi_colors and name in voi_colors:
                r, g, b = voi_colors[name]
                return (r / 255, g / 255, b / 255)
            return "gray"

        def _axis_label(label: str, unit: str) -> str:
            name = label or "Dose"
            return f"{name} [{unit}]" if unit else name

        plotted: list[str] = []
        for dvh in dvhs:
            color = _get_color(dvh.name)
            ax.plot(dvh.bins, dvh.cum_volume, color=color, linewidth=2, linestyle="-")
            if dvh.name not in plotted:
                plotted.append(dvh.name)

        ax.set_xlabel(_axis_label(overlay_label, overlay_unit))
        ax.set_ylabel("Volume [%]")
        ax.grid(True, which="both", linestyle="--", alpha=0.7)

        if secondary_dvhs:
            shared, scale = compare_units(overlay_unit, secondary_unit)
            if shared:
                sec_ax = ax
            else:
                sec_ax = ax.twiny()
                sec_ax.set_xlabel(_axis_label(secondary_label, secondary_unit))
                self.secondary_axes = sec_ax
            for dvh in secondary_dvhs:
                bins = dvh.bins * scale if shared else dvh.bins
                color = _get_color(dvh.name)
                sec_ax.plot(bins, dvh.cum_volume, color=color, linewidth=2, linestyle="--")
                if dvh.name not in plotted:
                    plotted.append(dvh.name)

        if plotted:
            handles = [Line2D([0], [0], color=_get_color(n), lw=2, label=n) for n in plotted]
            if secondary_dvhs:
                handles += [
                    Line2D([0], [0], color="black", lw=2, ls="-", label=primary_name or "Primary"),
                    Line2D(
                        [0], [0], color="black", lw=2, ls="--", label=secondary_name or "Secondary"
                    ),
                ]
            ax.legend(
                handles=handles,
                loc="upper left",
                bbox_to_anchor=(1.02, 1),
                borderaxespad=0,
                frameon=True,
            )

        self.canvas.draw()
