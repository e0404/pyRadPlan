"""Plot of the clinical goals over the optimizations of a planning agent run."""

from __future__ import annotations

import html
from typing import Any, Optional, Sequence

import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QGridLayout, QLabel, QScrollArea, QSplitter, QVBoxLayout, QWidget

from .._plot_theme import plot_colors

#: Margins are shown clipped to this range (relative to the goal's limit).
_CLIP = (-1.0, 0.5)
#: Width of one legend column; the number of columns follows the widget's width.
_LEGEND_COLUMN_WIDTH = 280
_GOAL_WIDTH = 3.0
_SCORE_WIDTH = 5.0


class GoalProgressPlot(QWidget):
    """Margin of every goal per optimization; above the 0 line a goal is met.

    A margin is relative to the goal's limit, in percent, and shown clipped to
    [-100 %, +50 %] so goals far inside their limit do not squash the region around 0.
    Hard goals are solid lines, soft goals dashed; the thick line is the plan score.
    Selected plans are marked with vertical lines. The goals are listed below the plot;
    goals whose structure does not exist are not evaluated and not drawn.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        split = QSplitter(Qt.Orientation.Vertical)
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setLabel("bottom", "Optimization")
        self.plot_widget.setLabel("left", "Goal margin [%] (higher is better)")
        self.plot_widget.showGrid(x=False, y=True, alpha=0.2)
        split.addWidget(self.plot_widget)

        self._legend = QWidget()
        self._legend_lay = QGridLayout(self._legend)
        self._legend_lay.setContentsMargins(6, 2, 6, 2)
        self._legend_lay.setHorizontalSpacing(16)
        self._legend_lay.setVerticalSpacing(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self._legend)
        split.addWidget(scroll)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 1)
        lay.addWidget(split)
        self._entries: list[tuple[str, str, bool]] = []
        self._columns = 1
        self.set_plans([])

    def set_plans(self, plans: Sequence[Any], picks: Sequence[Any] = ()) -> None:
        """Draw *plans* (``PlanningState.plans``) and mark *picks* as Plan 1, 2, ..."""
        plot = self.plot_widget
        plot.clear()
        self._apply_colors()
        fg = plot_colors(plot).foreground
        plot.addItem(
            pg.InfiniteLine(
                pos=0,
                angle=0,
                pen=pg.mkPen(QColor(150, 150, 150), width=2.5),
                label="goal met above",
                labelOpts={"position": 0.08, "color": fg},
            )
        )
        entries: list[tuple[str, str, bool]] = []
        if plans:
            entries = self._draw(plans, picks, fg)
        self._set_legend(entries)

    def _draw(self, plans: Sequence[Any], picks: Sequence[Any], fg) -> list:
        plot = self.plot_widget
        rows = [{g["goal"]: g for g in p.evaluation["goals"]} for p in plans]
        goals = list(dict.fromkeys(goal for r in rows for goal in r))
        x = [p.index for p in plans]
        plot.getAxis("bottom").setTicks([[(i, str(i)) for i in x]])
        lo, hi = _CLIP
        entries = []
        for i, goal in enumerate(goals):
            hard = next(r[goal]["hard"] for r in rows if goal in r)
            y = [
                100 * min(max(r[goal]["rel_margin"], lo), hi) if goal in r else float("nan")
                for r in rows
            ]
            color = pg.intColor(i, len(goals))
            style = Qt.PenStyle.SolidLine if hard else Qt.PenStyle.DashLine
            plot.plot(
                x,
                y,
                pen=pg.mkPen(color, width=_GOAL_WIDTH, style=style),
                symbol="o",
                symbolSize=8,
                symbolBrush=color,
            )
            entries.append((color.name(), goal if hard else f"{goal} (soft)", hard))
        plot.plot(
            x,
            [p.evaluation["score"] for p in plans],
            pen=pg.mkPen(fg, width=_SCORE_WIDTH),
            symbol="o",
            symbolSize=9,
            symbolBrush=fg,
        )
        entries.append((QColor(fg).name(), "Score", True))
        for number, p in enumerate(picks, start=1):
            plot.addItem(
                pg.InfiniteLine(
                    pos=p.index,
                    angle=90,
                    pen=pg.mkPen("#e6b800", width=2.5),
                    label=f"Plan {number}",
                    labelOpts={"position": 0.95, "color": fg},
                )
            )
        return entries

    def legend_entries(self) -> list[str]:
        """Return the texts of the legend below the plot."""
        lay = self._legend_lay
        items = (lay.itemAt(i).widget() for i in range(lay.count()))
        return [w.property("goal") for w in items if w is not None]

    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt override)
        """Re-flow the legend when the number of columns that fit changes."""
        super().resizeEvent(event)
        if self._legend_columns() != self._columns:
            self._set_legend(self._entries)

    def _legend_columns(self) -> int:
        return max(1, self.width() // _LEGEND_COLUMN_WIDTH)

    def _set_legend(self, entries: list[tuple[str, str, bool]]) -> None:
        self._entries = entries
        self._columns = columns = self._legend_columns()
        lay = self._legend_lay
        while lay.count():
            widget = lay.takeAt(0).widget()
            if widget is not None:
                widget.deleteLater()
        for row in range(lay.rowCount()):  # a grid never shrinks: drop the old stretch
            lay.setRowStretch(row, 0)
        for n, (color, text, hard) in enumerate(entries):
            line = "━━" if hard else "╍╍"
            label = QLabel(
                f'<span style="color:{color}; font-weight:bold;">{line}</span> {html.escape(text)}'
            )
            label.setProperty("goal", text)
            label.setWordWrap(True)
            lay.addWidget(label, n // columns, n % columns)
        lay.setRowStretch(lay.rowCount(), 1)

    def _apply_colors(self) -> None:
        plot = self.plot_widget
        colors = plot_colors(plot)
        plot.setBackground(colors.figure)
        plot.getPlotItem().getViewBox().setBackgroundColor(colors.axes)
        for name in ("left", "bottom"):
            axis = plot.getAxis(name)
            axis.setPen(colors.foreground)
            axis.setTextPen(colors.foreground)
