import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest

from pyRadPlan.analysis._dvh import DVH
from pyRadPlan.gui.widgets.analysis._dvh import DVHPlotWidget


def _make_dvh(name, scale):
    quantity = np.linspace(0.0, scale, 27).reshape(3, 3, 3)
    mask = np.ones((3, 3, 3), dtype=bool)
    return DVH.compute(quantity=quantity, mask=mask, name=name)


def _styles(items):
    return [item.opts["pen"].style() for item in items]


def _curves(widget):
    """Curve items drawn against the primary x axis."""
    return [item for curve, item in widget.curve_items if not curve.secondary_axis]


def _legend_labels(widget):
    return [label.text for _, label in widget.legend.items]


def test_dvh_plot_widget_init(qapp):
    widget = DVHPlotWidget()
    assert _curves(widget) == []
    assert not widget.legend.isVisible()


def test_dvh_plot_widget_plot(qapp, test_data_photons):
    ct, cst, result = test_data_photons
    dose = np.swapaxes(result["physicalDose"], 0, 1)
    dvhs = [DVH.compute(quantity=dose, mask=voi.mask, name=voi.name) for voi in cst.vois]

    widget = DVHPlotWidget()
    widget.plot(dvhs)

    assert len(_curves(widget)) == len(dvhs)


def test_dvh_plot_widget_draws_curves_labels_and_legend(qapp):
    widget = DVHPlotWidget()
    widget.plot(
        [_make_dvh("PTV", 2.0), _make_dvh("Body", 1.0)],
        voi_colors={"PTV": (255, 0, 0)},
        overlay_unit="Gy",
        overlay_label="Dose",
    )

    items = _curves(widget)
    assert _styles(items) == [Qt.PenStyle.SolidLine] * 2
    assert items[0].opts["pen"].color().name() == "#ff0000"
    assert widget.plot_item.getAxis("bottom").labelText == "Dose [Gy]"
    assert widget.plot_item.getAxis("left").labelText == "Volume [%]"
    assert widget.legend.isVisible()
    assert _legend_labels(widget) == ["PTV", "Body"]


def test_dvh_plot_widget_empty(qapp):
    widget = DVHPlotWidget()
    widget.plot([_make_dvh("PTV", 2.0)])
    widget.plot([])

    assert _curves(widget) == []
    assert _legend_labels(widget) == []
    assert not widget.legend.isVisible()
    assert widget.plot_item.getAxis("bottom").labelText == "Dose"


def test_dvh_plot_widget_compatible_secondary(qapp):
    widget = DVHPlotWidget()
    widget.plot(
        [_make_dvh("PTV", 2.0), _make_dvh("Body", 1.0)],
        overlay_unit="Gy",
        secondary_dvhs=[_make_dvh("PTV", 2.0), _make_dvh("Body", 1.0)],
        secondary_unit="Gy (RBE)",
        primary_name="physicalDose",
        secondary_name="RBExDose",
    )

    assert widget.secondary_view is None
    items = _curves(widget)
    assert _styles(items) == [Qt.PenStyle.SolidLine] * 2 + [Qt.PenStyle.DashLine] * 2
    assert _legend_labels(widget) == ["PTV", "Body", "physicalDose", "RBExDose"]


def test_dvh_plot_widget_incompatible_secondary(qapp):
    secondary = _make_dvh("PTV", 5.0)
    widget = DVHPlotWidget()
    widget.plot(
        [_make_dvh("PTV", 2.0)],
        overlay_unit="Gy",
        secondary_dvhs=[secondary],
        secondary_unit="keV/µm",
        secondary_label="LET",
    )

    assert widget.secondary_view is not None
    top = widget.plot_item.getAxis("top")
    assert top.isVisibleTo(widget.plot_item) and "keV" in top.labelText
    assert _styles(_curves(widget)) == [Qt.PenStyle.SolidLine]
    twin = [i for i in widget.secondary_view.addedItems if isinstance(i, pg.PlotDataItem)]
    assert _styles(twin) == [Qt.PenStyle.DashLine]
    np.testing.assert_allclose(twin[0].xData, secondary.bins)
    # drawn above the main plot's opaque background; zoom/pan act on the main view
    assert widget.secondary_view.zValue() > widget.plot_item.zValue()
    assert widget.secondary_view.state["mouseEnabled"] == [False, False]

    widget.plot([_make_dvh("PTV", 2.0)], overlay_unit="Gy")
    assert widget.secondary_view is None
    assert not widget.plot_item.getAxis("top").isVisibleTo(widget.plot_item)


def test_dvh_plot_follows_palette(qapp, dark_palette):
    palette, colors = dark_palette
    widget = DVHPlotWidget()
    widget.plot(
        [_make_dvh("PTV", 2.0)],
        voi_colors={"PTV": (255, 0, 0)},
        secondary_dvhs=[_make_dvh("PTV", 1.0)],
    )
    widget.setPalette(palette)

    assert widget.view.backgroundBrush().color().name() == colors["window"]
    assert widget.plot_item.getViewBox().background.brush().color().name() == colors["base"]
    assert widget.plot_item.getAxis("bottom").pen().color().name() == colors["text"]
    assert {label.opts["color"].name() for _, label in widget.legend.items} == {colors["text"]}
    # the line-style rows follow the text color, the structure keeps its own color
    pens = [sample.item.opts["pen"].color().name() for sample, _ in widget.legend.items]
    assert pens == ["#ff0000", colors["text"], colors["text"]]


# ---------------------------------------------------------------------------
# Interaction
# ---------------------------------------------------------------------------


def _plot_with_markers(widget):
    """Plot PTV and Body with D98 and V1Gy markers; returns the QI collection."""
    from pyRadPlan.analysis import QICollection

    quantity = np.linspace(0.0, 2.0, 27).reshape(3, 3, 3)
    mask = np.ones((3, 3, 3), dtype=bool)
    qis = QICollection.from_masks(
        {"PTV": mask, "Body": mask}, quantity, ref_vols=[98.0], ref_doses=[1.0]
    )
    dvhs = [DVH.compute(quantity=quantity, mask=mask, name=n) for n in ("PTV", "Body")]
    widget.plot(
        dvhs, voi_colors={"PTV": (255, 0, 0), "Body": (0, 0, 255)}, overlay_unit="Gy", qis=qis
    )
    return qis


def _shown(qapp, widget):
    widget.resize(600, 400)
    widget.show()
    # A real X11 window manager maps the window asynchronously; the view range is only
    # laid out once it is exposed.
    QTest.qWaitForWindowExposed(widget)
    qapp.processEvents()


class _Click:
    """Stand-in for a pyqtgraph mouse click event."""

    def __init__(self, scene_pos, double=False, accepted=False):
        self._pos, self._double, self._accepted = scene_pos, double, accepted

    def button(self):
        return Qt.MouseButton.LeftButton

    def scenePos(self):  # noqa: N802 - pyqtgraph API
        return self._pos

    def double(self):
        return self._double

    def isAccepted(self):  # noqa: N802 - pyqtgraph API
        return self._accepted


def test_markers_hidden_until_shown_or_highlighted(qapp):
    widget = DVHPlotWidget()
    _plot_with_markers(widget)
    assert len(widget.marker_item.points()) == 0

    widget.set_markers_visible("dx", True)
    assert {p.data().metric for p in widget.marker_item.points()} == {"D98"}
    widget.set_markers_visible("vx", True)
    assert len(widget.marker_item.points()) == 4

    widget.set_markers_visible("dx", False)
    widget.set_markers_visible("vx", False)
    widget.highlight("PTV", "D98")
    assert [p.data().metric for p in widget.marker_item.points()] == ["D98"]


def test_highlighted_structure_is_emphasized(qapp):
    widget = DVHPlotWidget()
    _plot_with_markers(widget)
    widget.highlight("PTV")

    pens = {curve.name: item.opts["pen"] for curve, item in widget.curve_items}
    assert pens["PTV"].widthF() > pens["Body"].widthF()
    assert pens["PTV"].color().alpha() == 255
    assert pens["Body"].color().alpha() < 255
    assert widget.highlighted() == ("PTV", None)
    assert widget.value_line is None

    widget.highlight(None)
    assert widget.highlighted() == (None, None)
    assert all(item.opts["pen"].color().alpha() == 255 for _, item in widget.curve_items)


def test_highlighted_dx_marker_shows_value_and_parameter_lines(qapp):
    widget = DVHPlotWidget()
    qis = _plot_with_markers(widget)
    widget.highlight("PTV", "D98")

    # D98: solid vertical line at its dose, dashed horizontal line at 98 %
    assert widget.value_line.angle == 90
    assert widget.value_line.value() == qis["PTV"]["D98"].value
    assert widget.value_line.pen.style() == Qt.PenStyle.SolidLine
    assert widget.parameter_line.angle == 0
    assert widget.parameter_line.value() == 98.0
    assert widget.parameter_line.pen.style() == Qt.PenStyle.DashLine
    assert widget.parameter_line.movable


def test_highlighted_vx_marker_lines_are_swapped(qapp):
    widget = DVHPlotWidget()
    qis = _plot_with_markers(widget)
    widget.highlight("PTV", "V1Gy")

    assert widget.value_line.angle == 0
    assert widget.value_line.value() == qis["PTV"]["V1Gy"].value
    assert widget.parameter_line.angle == 90
    assert widget.parameter_line.value() == 1.0


def test_dragging_the_parameter_line_moves_the_point(qapp):
    from pyRadPlan.analysis._dvh import dose_at_volume

    widget = DVHPlotWidget()
    _plot_with_markers(widget)
    widget.highlight("PTV", "D98")
    moved = []
    widget.parameter_moved.connect(lambda *args: moved.append(args))

    widget.parameter_line.setValue(50.04)

    curve = widget.shown_plot.primary_curve("PTV")
    expected = dose_at_volume(curve.x, curve.y, 50.04)
    assert widget.value_line.value() == expected
    assert widget.parameter_line.label.format == "D50"
    point = widget.marker_item.points()[0].pos()
    assert (point.x(), point.y()) == (expected, 50.04)

    widget.parameter_line.sigPositionChangeFinished.emit(widget.parameter_line)
    assert moved == [("PTV", "D98", 50.0)]


def test_highlight_survives_replot_while_shown(qapp):
    widget = DVHPlotWidget()
    _plot_with_markers(widget)
    widget.highlight("PTV", "D98")

    _plot_with_markers(widget)
    assert widget.highlighted() == ("PTV", "D98")
    assert widget.parameter_line is not None

    widget.plot([_make_dvh("Body", 1.0)])
    assert widget.highlighted() == (None, None)


def test_curve_and_marker_clicks_are_reported(qapp):
    widget = DVHPlotWidget()
    _plot_with_markers(widget)
    widget.set_markers_visible("vx", True)
    structures, markers = [], []
    widget.structure_clicked.connect(structures.append)
    widget.marker_clicked.connect(lambda *args: markers.append(args))

    item = dict((c.name, i) for c, i in widget.curve_items)["Body"]
    item.sigClicked.emit(item, None)
    points = [p for p in widget.marker_item.points() if p.data().metric == "V1Gy"]
    widget.marker_item.sigClicked.emit(widget.marker_item, points[:1], None)

    assert structures == ["Body"]
    assert markers == [(points[0].data().structure, "V1Gy")]


def test_scene_clicks_clear_and_double_click_requests_a_point(qapp):
    from PySide6.QtCore import QPointF

    widget = DVHPlotWidget()
    _plot_with_markers(widget)
    _shown(qapp, widget)
    structures, points = [], []
    widget.structure_clicked.connect(structures.append)
    widget.point_requested.connect(lambda *args: points.append(args))
    scene_pos = widget.plot_item.vb.mapViewToScene(QPointF(1.0, 40.0))

    widget._on_scene_clicked(_Click(scene_pos, accepted=True))  # a curve took it
    widget._on_scene_clicked(_Click(scene_pos))
    widget._on_scene_clicked(_Click(scene_pos, double=True))

    assert structures == [""]
    assert len(points) == 1
    np.testing.assert_allclose(points[0], (1.0, 40.0), atol=1e-6)


def test_data_cursor_lists_volumes_at_the_mouse(qapp):
    from PySide6.QtCore import QPointF

    widget = DVHPlotWidget()
    _plot_with_markers(widget)
    _shown(qapp, widget)

    widget._on_mouse_moved(widget.plot_item.vb.mapViewToScene(QPointF(1.0, 50.0)))

    assert widget.cursor_line.isVisible()
    assert abs(widget.cursor_line.value() - 1.0) < 1e-6
    html = widget.cursor_text.textItem.toHtml()
    curve = widget.shown_plot.primary_curve("PTV")
    assert "PTV" in html and "Body" in html
    assert f"{np.interp(1.0, curve.x, curve.y):.1f} %" in html

    widget._on_mouse_moved(QPointF(-50.0, -50.0))
    assert not widget.cursor_line.isVisible()


def test_plotted_widget_is_freed_without_garbage_collection(qapp):
    # A reference cycle through a signal connection left the widget to the
    # garbage collector, whose teardown order segfaulted PySide on Linux.
    import gc
    import weakref

    widget = DVHPlotWidget()
    _plot_with_markers(widget)
    widget.highlight("Body")
    gc.disable()
    try:
        ref = weakref.ref(widget)
        del widget
        assert ref() is None
    finally:
        gc.enable()
