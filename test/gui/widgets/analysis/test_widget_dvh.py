from pyRadPlan.gui.widgets.analysis._dvh import DVHPlotWidget
from pyRadPlan.analysis._dvh import DVH
import numpy as np


def test_dvh_plot_widget_init(qapp):
    widget = DVHPlotWidget()
    assert widget is not None
    assert widget.figure is not None
    assert widget.canvas is not None


def test_dvh_plot_widget_plot(qapp, test_data_photons):
    ct, cst, result = test_data_photons

    dose = np.swapaxes(result["physicalDose"], 0, 1)

    dvhs = [DVH.compute(quantity=dose, mask=voi.mask, name=voi.name) for voi in cst.vois]

    widget = DVHPlotWidget()
    widget.plot(dvhs)

    # Check if axes were created
    assert len(widget.figure.axes) > 0


def _make_dvh(name, scale):
    quantity = np.linspace(0.0, scale, 27).reshape(3, 3, 3)
    mask = np.ones((3, 3, 3), dtype=bool)
    return DVH.compute(quantity=quantity, mask=mask, name=name)


def test_dvh_plot_widget_legend_lists_structures(qapp):
    widget = DVHPlotWidget()
    widget.plot(
        [_make_dvh("PTV", 2.0), _make_dvh("Body", 1.0)],
        voi_colors={"PTV": (255, 0, 0)},
        overlay_unit="Gy",
        overlay_label="Dose",
    )

    ax = widget.figure.axes[0]
    texts = [t.get_text() for t in ax.get_legend().get_texts()]
    assert texts == ["PTV", "Body"]
    assert len(ax.get_lines()) == 2
    assert all(line.get_linestyle() == "-" for line in ax.get_lines())
    assert ax.get_xlabel() == "Dose [Gy]"


def test_dvh_plot_widget_empty(qapp):
    widget = DVHPlotWidget()
    widget.plot([])

    ax = widget.figure.axes[0]
    assert ax.get_legend() is None
    assert len(ax.get_lines()) == 0
    assert ax.get_xlabel() == "Dose"
    assert ax.get_ylabel() == "Volume [%]"


def test_dvh_plot_widget_compatible_secondary(qapp):
    widget = DVHPlotWidget()
    widget.plot(
        [_make_dvh("PTV", 2.0), _make_dvh("Body", 1.0)],
        overlay_unit="Gy",
        overlay_label="Dose",
        secondary_dvhs=[_make_dvh("PTV", 2.0), _make_dvh("Body", 1.0)],
        secondary_unit="Gy (RBE)",
        secondary_label="RBExDose",
        primary_name="physicalDose",
        secondary_name="RBExDose",
    )

    assert len(widget.figure.axes) == 1
    assert widget.secondary_axes is None
    ax = widget.figure.axes[0]
    lines = ax.get_lines()
    assert [line.get_linestyle() for line in lines] == ["-", "-", "--", "--"]
    assert lines[0].get_color() == lines[2].get_color()
    texts = [t.get_text() for t in ax.get_legend().get_texts()]
    assert texts == ["PTV", "Body", "physicalDose", "RBExDose"]


def test_dvh_plot_widget_rescaled_secondary(qapp):
    primary = _make_dvh("PTV", 2.0)
    secondary = _make_dvh("PTV", 200.0)
    widget = DVHPlotWidget()
    widget.plot(
        [primary],
        overlay_unit="Gy",
        secondary_dvhs=[secondary],
        secondary_unit="cGy",
    )

    assert widget.secondary_axes is None
    dashed = [line for line in widget.figure.axes[0].get_lines() if line.get_linestyle() == "--"]
    assert len(dashed) == 1
    np.testing.assert_allclose(dashed[0].get_xdata(), secondary.bins * 0.01)
    texts = [t.get_text() for t in widget.figure.axes[0].get_legend().get_texts()]
    assert texts == ["PTV", "Primary", "Secondary"]


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

    assert widget.secondary_axes is not None
    assert len(widget.figure.axes) == 2
    ax = widget.figure.axes[0]
    assert "keV" in widget.secondary_axes.get_xlabel()
    assert [line.get_linestyle() for line in ax.get_lines()] == ["-"]
    twin_lines = widget.secondary_axes.get_lines()
    assert [line.get_linestyle() for line in twin_lines] == ["--"]
    np.testing.assert_allclose(twin_lines[0].get_xdata(), secondary.bins)
    assert ax.get_legend() is not None

    widget.plot([_make_dvh("PTV", 2.0)], overlay_unit="Gy")
    assert widget.secondary_axes is None
    assert len(widget.figure.axes) == 1
