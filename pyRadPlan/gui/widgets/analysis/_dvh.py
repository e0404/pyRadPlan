"""DVH plotting widget."""

from __future__ import annotations

from html import escape
from typing import TYPE_CHECKING, Optional

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QSizePolicy, QVBoxLayout, QWidget

from pyRadPlan.analysis._dvh import dose_at_volume
from pyRadPlan.analysis._dvh_plot import (
    DVHCurve,
    DVHPlot,
    LegendEntry,
    QIMarker,
    build_dvh_plot,
    format_marker_value,
)
from pyRadPlan.gui.widgets._plot_theme import PlotColors, apply_pyqtgraph_colors, plot_colors

if TYPE_CHECKING:
    from pyRadPlan.analysis import QICollection
    from pyRadPlan.analysis._dvh import DVH

_LINE_WIDTH = 2
_HIGHLIGHT_WIDTH = 4
_GUIDE_WIDTH = 2
#: Stacking above the curves (the highlighted curve is at 1): guide lines with their
#: labels, markers, the cursor line and, on top, the cursor readout.
_GUIDE_Z, _MARKER_Z, _CURSOR_Z, _READOUT_Z = 5, 10, 20, 30
#: Opacity (0-255) of curves and markers of structures other than the highlighted one.
_FADED_ALPHA = 60
_MARKER_SIZE = 9
_HIGHLIGHT_MARKER_SIZE = 14
#: Distance in pixels within which a click still hits a curve.
_CURVE_CLICK_WIDTH = 8
_LINE_STYLES = {"solid": Qt.PenStyle.SolidLine, "dashed": Qt.PenStyle.DashLine}


def _qcolor(color, alpha: int = 255) -> QColor:
    """Return *color* (``[0, 1]`` RGB tuple or color name) with *alpha*."""
    qcolor = QColor.fromRgbF(*color) if isinstance(color, tuple) else QColor(color)
    qcolor.setAlpha(alpha)
    return qcolor


def _pen(color, line_style: str, width: float = _LINE_WIDTH, alpha: int = 255):
    return pg.mkPen(_qcolor(color, alpha), width=width, style=_LINE_STYLES[line_style])


class DVHPlotWidget(QWidget):
    """Widget displaying an interactive DVH plot, drawn with pyqtgraph.

    What is shown is decided by :func:`~pyRadPlan.analysis._dvh_plot.build_dvh_plot`;
    the description last drawn is kept in :attr:`shown_plot`.  The plot zooms and pans
    with the mouse; its context menu offers "View All" and export to PNG, SVG or CSV.

    Interaction:

    - Hovering shows a data cursor listing every curve's volume at the mouse position.
    - Clicking a curve emits :attr:`structure_clicked`; clicking empty plot area emits
      it with an empty name.
    - Clicking a D_x / V_x marker emits :attr:`marker_clicked`.  Markers are shown with
      per kind with :meth:`set_markers_visible`; the highlighted one is always shown.
    - The highlighted marker (see :meth:`highlight`) gets a solid line at its value and
      a dashed line at its parameter.  Dragging the dashed line moves the point along
      the curve and emits :attr:`parameter_moved` on release.
    - Double-clicking emits :attr:`point_requested` with the dose and volume there.
    """

    #: Structure name of a clicked curve, ``""`` for a click on empty plot area.
    structure_clicked = Signal(str)
    #: Structure and metric id of a clicked D_x / V_x marker.
    marker_clicked = Signal(str, str)
    #: Dose and volume [%] at a double-click on the primary axes.
    point_requested = Signal(float, float)
    #: Structure, metric id and new parameter after dragging a marker's parameter line.
    parameter_moved = Signal(str, str, float)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.view = pg.GraphicsLayoutWidget()
        self.plot_item: pg.PlotItem = self.view.addPlot(row=0, col=0)
        self.plot_item.showGrid(x=True, y=True, alpha=0.3)
        self.legend = pg.LegendItem()
        # Keep the legend at its natural height instead of spreading the rows out.
        self.legend.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        self.view.addItem(self.legend, row=0, col=1)
        self.view.ci.layout.setColumnStretchFactor(0, 1)
        self.view.ci.layout.setAlignment(self.legend, Qt.AlignmentFlag.AlignTop)
        layout.addWidget(self.view)

        #: The description of the plot currently shown.
        self.shown_plot: Optional[DVHPlot] = None
        #: View box of the secondary x axis, present while the plot needs one.
        self.secondary_view: Optional[pg.ViewBox] = None
        self._legend_samples: list[tuple[LegendEntry, pg.PlotDataItem]] = []
        #: The drawn curves, with the description each was drawn from.
        self.curve_items: list[tuple[DVHCurve, pg.PlotDataItem]] = []

        #: Marker kinds (``"dx"``, ``"vx"``) shown in the plot.
        self._visible_marker_kinds: set[str] = set()
        self._highlighted_structure: Optional[str] = None
        self._highlighted_metric: Optional[str] = None
        self.marker_item = pg.ScatterPlotItem(hoverable=True, tip=self._marker_tip)
        self.marker_item.setZValue(_MARKER_Z)
        self.marker_item.sigClicked.connect(self._on_marker_clicked)
        #: Solid line at the highlighted marker's value, dashed (draggable) one at its parameter.
        self.value_line: Optional[pg.InfiniteLine] = None
        self.parameter_line: Optional[pg.InfiniteLine] = None
        #: Position of the highlighted marker while its parameter line is dragged.
        self._dragged: Optional[tuple[float, float]] = None

        self.cursor_line = pg.InfiniteLine(angle=90, movable=False)
        self.cursor_line.setZValue(_CURSOR_Z)
        self.cursor_text = pg.TextItem(anchor=(0, 0))
        self.cursor_text.setZValue(_READOUT_Z)
        scene = self.plot_item.scene()
        scene.sigMouseMoved.connect(self._on_mouse_moved)
        scene.sigMouseClicked.connect(self._on_scene_clicked)

        self._draw(build_dvh_plot([]))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

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
        qis: QICollection | None = None,
    ) -> None:
        """Plot primary DVH curves solid and optional secondary curves dashed.

        See :func:`~pyRadPlan.analysis._dvh_plot.build_dvh_plot` for how the
        curves, axes, legend and D_x / V_x markers (from *qis*) are chosen;
        *overlay_unit* and *overlay_label* are its ``unit`` and ``label``.  A
        highlight survives as long as its structure and metric are still shown.
        """
        self._draw(
            build_dvh_plot(
                dvhs,
                voi_colors=voi_colors,
                unit=overlay_unit,
                label=overlay_label,
                secondary_dvhs=secondary_dvhs,
                secondary_unit=secondary_unit,
                secondary_label=secondary_label,
                primary_name=primary_name,
                secondary_name=secondary_name,
                qis=qis,
            )
        )

    def set_markers_visible(self, kind: str, visible: bool) -> None:
        """Show or hide the D_x (``"dx"``) or V_x (``"vx"``) markers.

        The highlighted marker stays visible either way.
        """
        if visible:
            self._visible_marker_kinds.add(kind)
        else:
            self._visible_marker_kinds.discard(kind)
        self._draw_markers()

    def highlight(self, structure: Optional[str], metric: Optional[str] = None) -> None:
        """Emphasize *structure*'s curves and, with *metric*, its D_x / V_x marker.

        Unknown names clear the respective highlight; ``None`` clears both.
        """
        self._set_highlight(structure, metric)
        self._refresh_highlight()

    def highlighted(self) -> tuple[Optional[str], Optional[str]]:
        """Return the highlighted structure and metric id (each ``None`` if unset)."""
        return self._highlighted_structure, self._highlighted_metric

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _draw(self, plot: DVHPlot) -> None:
        self._clear()
        self.shown_plot = plot
        item = self.plot_item
        for curve in plot.curves:
            data = pg.PlotDataItem(curve.x, curve.y)
            data.setCurveClickable(True, width=_CURVE_CLICK_WIDTH)
            # A bound method, not a closure over self: PySide holds bound receivers
            # weakly, while a closure kept by the item's connection forms a cycle
            # whose garbage collection segfaults (seen on Linux).
            data.sigClicked.connect(self._on_curve_clicked)
            if curve.secondary_axis:
                self._secondary_view().addItem(data)
            else:
                item.addItem(data)
            self.curve_items.append((curve, data))
        item.setLabel("bottom", plot.x_label)
        item.setLabel("left", plot.y_label)
        if plot.secondary_x_label is not None:
            item.setLabel("top", plot.secondary_x_label)

        for entry in plot.legend:
            sample = pg.PlotDataItem()
            self._legend_samples.append((entry, sample))
            self.legend.addItem(sample, entry.label)
        self.legend.setVisible(bool(plot.legend))

        item.addItem(self.marker_item)
        for overlay in (self.cursor_line, self.cursor_text):
            overlay.hide()
            item.addItem(overlay, ignoreBounds=True)

        self._set_highlight(self._highlighted_structure, self._highlighted_metric)
        self._apply_plot_colors()
        item.enableAutoRange()

    def _clear(self) -> None:
        self.plot_item.clear()
        self.legend.clear()
        self._legend_samples = []
        self.curve_items = []
        self.value_line = self.parameter_line = None
        self._dragged = None
        if self.secondary_view is not None:
            self.plot_item.vb.sigResized.disconnect(self._sync_secondary_view)
            self.secondary_view.scene().removeItem(self.secondary_view)
            self.secondary_view = None
        self.plot_item.hideAxis("top")

    def _secondary_view(self) -> pg.ViewBox:
        """Return the view box of the secondary x axis, creating it on first use.

        It overlays the main view with a shared y axis. Mouse zoom/pan and the
        context menu pass through to the main view.
        """
        if self.secondary_view is None:
            view = pg.ViewBox()
            view.setMouseEnabled(x=False, y=False)
            view.setMenuEnabled(False)
            # View boxes default to z=-100, below the plot's opaque background.
            view.setZValue(1)
            self.plot_item.scene().addItem(view)
            self.plot_item.showAxis("top")
            self.plot_item.getAxis("top").linkToView(view)
            view.setYLink(self.plot_item.vb)
            self.plot_item.vb.sigResized.connect(self._sync_secondary_view)
            self.secondary_view = view
            self._sync_secondary_view()
        return self.secondary_view

    def _sync_secondary_view(self) -> None:
        view = self.secondary_view
        view.setGeometry(self.plot_item.vb.sceneBoundingRect())
        view.linkedViewChanged(self.plot_item.vb, view.YAxis)

    def _marker(self, structure: Optional[str], metric: Optional[str]) -> Optional[QIMarker]:
        for marker in self.shown_plot.markers:
            if (marker.structure, marker.metric) == (structure, metric):
                return marker
        return None

    def _set_highlight(self, structure: Optional[str], metric: Optional[str]) -> None:
        """Store the highlight, dropping names the shown plot does not contain."""
        if structure is None or not any(c.name == structure for c in self.shown_plot.curves):
            structure, metric = None, None
        if metric is not None and self._marker(structure, metric) is None:
            metric = None
        self._highlighted_structure = structure
        self._highlighted_metric = metric

    def _alpha(self, structure: str) -> int:
        """Full opacity unless another structure is highlighted."""
        return 255 if self._highlighted_structure in (None, structure) else _FADED_ALPHA

    def _refresh_highlight(self) -> None:
        for curve, item in self.curve_items:
            highlighted = curve.name == self._highlighted_structure
            width = _HIGHLIGHT_WIDTH if highlighted else _LINE_WIDTH
            item.setPen(_pen(curve.color, curve.line_style, width, self._alpha(curve.name)))
            item.setZValue(1 if highlighted else 0)
        self._draw_marker_lines()
        self._draw_markers()

    def _draw_markers(self) -> None:
        foreground = plot_colors(self).foreground
        highlighted = self._marker(self._highlighted_structure, self._highlighted_metric)
        spots = []
        for marker in self.shown_plot.markers:
            is_highlighted = marker is highlighted
            if not (marker.kind in self._visible_marker_kinds or is_highlighted):
                continue
            pos = self._dragged if is_highlighted and self._dragged else (marker.x, marker.y)
            alpha = self._alpha(marker.structure)
            spots.append(
                {
                    "pos": pos,
                    "data": marker,
                    "size": _HIGHLIGHT_MARKER_SIZE if is_highlighted else _MARKER_SIZE,
                    "symbol": "o" if marker.kind == "dx" else "s",
                    "brush": _qcolor(marker.color, alpha),
                    "pen": pg.mkPen(_qcolor(foreground, alpha), width=1),
                }
            )
        self.marker_item.setData(spots)

    def _draw_marker_lines(self) -> None:
        """(Re)draw the value and parameter lines of the highlighted marker."""
        for line in (self.value_line, self.parameter_line):
            if line is not None:
                self.plot_item.removeItem(line)
        self.value_line = self.parameter_line = None
        self._dragged = None
        marker = self._marker(self._highlighted_structure, self._highlighted_metric)
        if marker is None:
            return

        colors = plot_colors(self)
        label_opts = {
            "position": 0.93,
            "color": colors.foreground,
            "fill": _qcolor(colors.axes, 200),
            "border": pg.mkPen(colors.grid),
        }
        is_dx = marker.kind == "dx"
        # D_x: dose (value) on the x axis, volume (parameter) on the y axis; V_x swapped.
        self.value_line = pg.InfiniteLine(
            pos=marker.value,
            angle=90 if is_dx else 0,
            pen=_pen(marker.color, "solid", _GUIDE_WIDTH),
            label=marker.value_text,
            labelOpts=label_opts,
        )
        self.parameter_line = pg.InfiniteLine(
            pos=marker.parameter,
            angle=0 if is_dx else 90,
            movable=True,
            bounds=(0.0, 100.0) if is_dx else (0.0, None),
            pen=_pen(marker.color, "dashed", _GUIDE_WIDTH),
            hoverPen=_pen(marker.color, "dashed", _HIGHLIGHT_WIDTH),
            label=marker.parameter_text(),
            labelOpts=label_opts,
        )
        self.parameter_line.setToolTip(f"Drag to change {marker.metric}")
        self.parameter_line.sigPositionChanged.connect(self._on_parameter_dragged)
        self.parameter_line.sigPositionChangeFinished.connect(self._on_parameter_released)
        for line in (self.value_line, self.parameter_line):
            line.setZValue(_GUIDE_Z)
            self.plot_item.addItem(line, ignoreBounds=True)

    # ------------------------------------------------------------------
    # Interaction
    # ------------------------------------------------------------------

    def _dragged_point(self, marker: QIMarker, parameter: float) -> tuple[float, float]:
        """Return the (parameter, value) of *marker* moved to *parameter* on its curve."""
        curve = self.shown_plot.primary_curve(marker.structure)
        if marker.kind == "dx":
            parameter = float(np.clip(parameter, 0.0, 100.0))
            return parameter, float(dose_at_volume(curve.x, curve.y, parameter))
        parameter = max(parameter, 0.0)
        return parameter, float(np.interp(parameter, curve.x, curve.y))

    def _on_parameter_dragged(self, line: pg.InfiniteLine) -> None:
        marker = self._marker(self._highlighted_structure, self._highlighted_metric)
        if marker is None or self.value_line is None:
            return
        parameter, value = self._dragged_point(marker, line.value())
        self.value_line.setValue(value)
        self._set_label(self.value_line, format_marker_value(value, marker.value_unit))
        self._set_label(line, marker.parameter_text(parameter))
        self._dragged = (value, parameter) if marker.kind == "dx" else (parameter, value)
        self._draw_markers()

    def _on_parameter_released(self, line: pg.InfiniteLine) -> None:
        marker = self._marker(self._highlighted_structure, self._highlighted_metric)
        if marker is None:
            return
        parameter, _ = self._dragged_point(marker, line.value())
        decimals = 1 if marker.kind == "dx" else 2
        self.parameter_moved.emit(marker.structure, marker.metric, round(parameter, decimals))

    @staticmethod
    def _set_label(line: pg.InfiniteLine, text: str) -> None:
        line.label.format = text
        line.label.valueChanged()

    @staticmethod
    def _marker_tip(x: float, y: float, data: QIMarker) -> str:
        return f"{data.structure}: {data.metric} = {data.value_text}"

    def _on_curve_clicked(self, item, _ev) -> None:
        name = next((curve.name for curve, data in self.curve_items if data is item), None)
        if name is not None:
            self.structure_clicked.emit(name)

    def _on_marker_clicked(self, _item, points, _ev) -> None:
        if len(points):
            marker: QIMarker = points[0].data()
            self.marker_clicked.emit(marker.structure, marker.metric)

    def _on_scene_clicked(self, ev) -> None:
        if ev.button() != Qt.MouseButton.LeftButton:
            return
        view = self.plot_item.vb
        if not view.sceneBoundingRect().contains(ev.scenePos()):
            return
        if ev.double():
            point = view.mapSceneToView(ev.scenePos())
            self.point_requested.emit(max(point.x(), 0.0), float(np.clip(point.y(), 0.0, 100.0)))
        elif not ev.isAccepted() and not (
            self.parameter_line is not None and self.parameter_line.mouseHovering
        ):
            self.structure_clicked.emit("")

    def _on_mouse_moved(self, scene_pos: QPointF) -> None:
        """Move the data cursor to the mouse and list each curve's volume there."""
        plot = self.shown_plot
        view = self.plot_item.vb
        if not plot.curves or not view.sceneBoundingRect().contains(scene_pos):
            self._hide_cursor()
            return
        x = view.mapSceneToView(scene_pos).x()
        secondary_x = (
            self.secondary_view.mapSceneToView(scene_pos).x()
            if self.secondary_view is not None
            else None
        )
        colors = plot_colors(self)
        self.cursor_text.setHtml(self._cursor_html(plot, x, secondary_x, colors))

        self.cursor_line.setValue(x)
        (x_min, x_max), (_, y_max) = view.viewRange()
        # Keep the readout on the side of the cursor with more room.
        self.cursor_text.setAnchor((0, 0) if x < (x_min + x_max) / 2 else (1, 0))
        self.cursor_text.setPos(x, y_max)
        self.cursor_line.show()
        self.cursor_text.show()

    @staticmethod
    def _cursor_html(
        plot: DVHPlot, x: float, secondary_x: Optional[float], colors: PlotColors
    ) -> str:
        dashed_name = next(
            (e.label for e in plot.legend if e.color is None and e.line_style == "dashed"), ""
        )
        rows = [f"{escape(plot.x_label)}: {x:.2f}"]
        if secondary_x is not None and plot.secondary_x_label is not None:
            rows.append(f"{escape(plot.secondary_x_label)}: {secondary_x:.2f}")
        for curve in plot.curves:
            at = secondary_x if curve.secondary_axis else x
            volume = float(np.interp(at, curve.x, curve.y))
            symbol = "━" if curve.line_style == "solid" else "┅"
            name = curve.name if curve.line_style == "solid" else f"{curve.name} ({dashed_name})"
            rows.append(
                f'<span style="color:{_qcolor(curve.color).name()}">{symbol}</span> '
                f"{escape(name)}: {volume:.1f} %"
            )
        return f'<div style="color:{colors.foreground}">{"<br>".join(rows)}</div>'

    def _hide_cursor(self) -> None:
        self.cursor_line.hide()
        self.cursor_text.hide()

    def leaveEvent(self, event) -> None:  # noqa: N802 - Qt override name
        """Hide the data cursor when the mouse leaves the plot."""
        self._hide_cursor()
        super().leaveEvent(event)

    # ------------------------------------------------------------------
    # Theming
    # ------------------------------------------------------------------

    def _apply_plot_colors(self) -> None:
        """Match the plot to the current (light or dark) palette."""
        colors = plot_colors(self)
        apply_pyqtgraph_colors(self.view, colors)
        self.legend.setBrush(colors.axes)
        self.legend.setPen(colors.grid)
        self.legend.setLabelTextColor(colors.foreground)
        for _, label in self.legend.items:
            label.setText(label.text)  # labels only re-render their color on setText
        for entry, sample in self._legend_samples:
            sample.setPen(_pen(entry.color or colors.foreground, entry.line_style))
        self.legend.update()

        self.cursor_line.setPen(pg.mkPen(colors.foreground, style=Qt.PenStyle.DotLine))
        self.cursor_text.fill = pg.mkBrush(_qcolor(colors.axes, 220))
        self.cursor_text.border = pg.mkPen(colors.grid)
        self._refresh_highlight()

    def changeEvent(self, event) -> None:  # noqa: N802 - Qt override name
        """Restyle the plot when the application palette/theme changes."""
        if event.type() == event.Type.PaletteChange and hasattr(self, "cursor_text"):
            self._apply_plot_colors()
        super().changeEvent(event)
