import numpy as np

from pyRadPlan.analysis._dvh import DVH
from pyRadPlan.analysis._dvh_plot import DEFAULT_CURVE_COLOR, build_dvh_plot


def _make_dvh(name, scale):
    quantity = np.linspace(0.0, scale, 27).reshape(3, 3, 3)
    mask = np.ones((3, 3, 3), dtype=bool)
    return DVH.compute(quantity=quantity, mask=mask, name=name)


def test_primary_curves_and_legend():
    plot = build_dvh_plot(
        [_make_dvh("PTV", 2.0), _make_dvh("Body", 1.0)],
        voi_colors={"PTV": (255, 0, 0)},
        unit="Gy",
        label="Dose",
    )

    assert [c.name for c in plot.curves] == ["PTV", "Body"]
    assert [c.line_style for c in plot.curves] == ["solid", "solid"]
    assert plot.curves[0].color == (1.0, 0.0, 0.0)
    assert plot.curves[1].color == DEFAULT_CURVE_COLOR
    assert plot.x_label == "Dose [Gy]"
    assert plot.y_label == "Volume [%]"
    assert plot.secondary_x_label is None
    assert [e.label for e in plot.legend] == ["PTV", "Body"]


def test_empty_plot_has_no_legend():
    plot = build_dvh_plot([])
    assert plot.curves == ()
    assert plot.legend == ()
    assert plot.x_label == "Dose"


def test_compatible_secondary_shares_the_axis():
    plot = build_dvh_plot(
        [_make_dvh("PTV", 2.0), _make_dvh("Body", 1.0)],
        unit="Gy",
        secondary_dvhs=[_make_dvh("PTV", 2.0), _make_dvh("Body", 1.0)],
        secondary_unit="Gy (RBE)",
        primary_name="physicalDose",
        secondary_name="RBExDose",
    )

    assert plot.secondary_x_label is None
    assert [c.line_style for c in plot.curves] == ["solid", "solid", "dashed", "dashed"]
    assert not any(c.secondary_axis for c in plot.curves)
    assert plot.curves[0].color == plot.curves[2].color
    assert [(e.label, e.color, e.line_style) for e in plot.legend[2:]] == [
        ("physicalDose", None, "solid"),
        ("RBExDose", None, "dashed"),
    ]
    assert [e.label for e in plot.legend[:2]] == ["PTV", "Body"]


def test_secondary_is_rescaled_to_the_primary_unit():
    secondary = _make_dvh("PTV", 200.0)
    plot = build_dvh_plot(
        [_make_dvh("PTV", 2.0)], unit="Gy", secondary_dvhs=[secondary], secondary_unit="cGy"
    )

    dashed = [c for c in plot.curves if c.line_style == "dashed"]
    np.testing.assert_allclose(dashed[0].x, secondary.bins * 0.01)
    assert [e.label for e in plot.legend] == ["PTV", "Primary", "Secondary"]


def test_incompatible_secondary_gets_its_own_axis():
    secondary = _make_dvh("PTV", 5.0)
    plot = build_dvh_plot(
        [_make_dvh("PTV", 2.0)],
        unit="Gy",
        secondary_dvhs=[secondary],
        secondary_unit="keV/µm",
        secondary_label="LET",
    )

    assert plot.secondary_x_label == "LET [keV/µm]"
    assert [(c.line_style, c.secondary_axis) for c in plot.curves] == [
        ("solid", False),
        ("dashed", True),
    ]
    np.testing.assert_allclose(plot.curves[1].x, secondary.bins)


def test_markers_from_dx_and_vx_quality_indicators():
    from pyRadPlan.analysis import QICollection

    quantity = np.linspace(0.0, 2.0, 27).reshape(3, 3, 3)
    masks = {"PTV": np.ones((3, 3, 3), dtype=bool), "Other": np.ones((3, 3, 3), dtype=bool)}
    qis = QICollection.from_masks(masks, quantity, ref_vols=[98.0], ref_doses=[1.0])
    dvh = DVH.compute(quantity=quantity, mask=masks["PTV"], name="PTV")

    plot = build_dvh_plot([dvh], voi_colors={"PTV": (255, 0, 0)}, unit="Gy", qis=qis)

    # only the plotted structure, only the D_x / V_x metrics
    assert {(m.structure, m.metric, m.kind) for m in plot.markers} == {
        ("PTV", "D98", "dx"),
        ("PTV", "V1Gy", "vx"),
    }
    dx = next(m for m in plot.markers if m.kind == "dx")
    assert (dx.y, dx.x) == (98.0, qis["PTV"]["D98"].value)
    assert dx.value_text.endswith(" Gy")
    vx = next(m for m in plot.markers if m.kind == "vx")
    assert (vx.x, vx.y) == (1.0, qis["PTV"]["V1Gy"].value)
    assert vx.value_text.endswith(" %")
    assert vx.color == (1.0, 0.0, 0.0)
    # the metric names follow the metric ids, also for a moved parameter
    assert (dx.parameter_text(), vx.parameter_text()) == ("D98", "V1Gy")
    assert (dx.parameter_text(95.34), vx.parameter_text(1.256)) == ("D95.3", "V1.26Gy")
    assert plot.primary_curve("PTV") is plot.curves[0]
    assert plot.primary_curve("Other") is None


def test_no_markers_without_quality_indicators():
    assert build_dvh_plot([_make_dvh("PTV", 2.0)]).markers == ()
