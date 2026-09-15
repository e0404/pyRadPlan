import numpy as np
import pytest
import SimpleITK as sitk
import matplotlib

matplotlib.use("Agg")

from pyRadPlan import CT
from pyRadPlan.result import PlanResult, QuantityDistribution
from pyRadPlan.stf import Beam, Ray, SteeringInformation
from pyRadPlan.stf._beamlet import PhotonBixel
from pyRadPlan.visualization import plot_dose_comparison


def _make_dose_result(values: np.ndarray) -> PlanResult:
    return PlanResult(quantities={"physical_dose": QuantityDistribution(values=values)})


def test_plot_dose_comparison(tmp_path):
    ct = CT(cube_hu=sitk.Image(10, 10, 5, sitk.sitkFloat32))
    before = _make_dose_result(np.random.rand(5, 10, 10).astype(np.float32))
    after = _make_dose_result(np.random.rand(5, 10, 10).astype(np.float32))

    fig = plot_dose_comparison(
        before,
        after,
        image_volume=ct,
        show_plot=False,
        save_filename=str(tmp_path / "dose.png"),
    )[0]
    assert len(fig.axes) >= 3
    assert (tmp_path / "dose.png").exists()


def test_diff_panel_uses_diverging_colormap_regardless_of_overlay_cmap():
    ct = CT(cube_hu=sitk.Image(10, 10, 5, sitk.sitkFloat32))
    before = _make_dose_result(np.zeros((5, 10, 10), dtype=np.float32))
    after = _make_dose_result(np.ones((5, 10, 10), dtype=np.float32))

    fig = plot_dose_comparison(
        before, after, image_volume=ct, overlay_cmap="jet", show_plot=False
    )[0]

    diff_qm = fig.axes[2].collections[-1]
    assert diff_qm.get_cmap().name == "RdBu_r"


def test_diff_panel_window_is_scoped_to_displayed_slice():
    ct = CT(cube_hu=sitk.Image(10, 10, 5, sitk.sitkFloat32))

    before_arr = np.zeros((5, 10, 10), dtype=np.float32)
    after_arr = np.zeros((5, 10, 10), dtype=np.float32)
    # Displayed (default middle) slice: a small, genuine difference.
    after_arr[2] = 0.1
    # A different slice: a huge outlier that must not affect the displayed window.
    after_arr[4] = 1000.0

    before = _make_dose_result(before_arr)
    after = _make_dose_result(after_arr)

    fig = plot_dose_comparison(before, after, image_volume=ct, show_plot=False)[0]

    diff_qm = fig.axes[2].collections[-1]
    vmin, vmax = diff_qm.get_clim()
    assert vmax == pytest.approx(0.1)
    assert vmin == pytest.approx(-0.1)


def test_diff_panel_shows_negative_values():
    ct = CT(cube_hu=sitk.Image(10, 10, 5, sitk.sitkFloat32))

    before_arr = np.ones((5, 10, 10), dtype=np.float32)
    after_arr = np.ones((5, 10, 10), dtype=np.float32)
    after_arr[2, :5, :] = 1.5  # +0.5 on the displayed (middle) slice
    after_arr[2, 5:, :] = 0.5  # -0.5 on the displayed (middle) slice

    before = _make_dose_result(before_arr)
    after = _make_dose_result(after_arr)

    fig = plot_dose_comparison(before, after, image_volume=ct, show_plot=False)[0]

    diff_qm = fig.axes[2].collections[-1]
    alphas = diff_qm.get_facecolors()[:, 3]
    assert np.all(alphas > 0), "negative-difference voxels were made fully transparent"


def test_title_before_after_diff_are_customizable():
    ct = CT(cube_hu=sitk.Image(10, 10, 5, sitk.sitkFloat32))
    before = _make_dose_result(np.zeros((5, 10, 10), dtype=np.float32))
    after = _make_dose_result(np.ones((5, 10, 10), dtype=np.float32))

    fig = plot_dose_comparison(
        before,
        after,
        title_before="Sequenced (dij)",
        title_after="Forward (field-based)",
        image_volume=ct,
        show_plot=False,
    )[0]

    assert fig.axes[0].get_title() == "Sequenced (dij)"
    assert fig.axes[1].get_title() == "Forward (field-based)"
    assert fig.axes[2].get_title() == "Difference (Forward (field-based) − Sequenced (dij))"


def test_title_diff_explicit_override():
    ct = CT(cube_hu=sitk.Image(10, 10, 5, sitk.sitkFloat32))
    before = _make_dose_result(np.zeros((5, 10, 10), dtype=np.float32))
    after = _make_dose_result(np.ones((5, 10, 10), dtype=np.float32))

    fig = plot_dose_comparison(
        before,
        after,
        title_before="A",
        title_after="B",
        title_diff="Custom diff title",
        image_volume=ct,
        show_plot=False,
    )[0]

    assert fig.axes[2].get_title() == "Custom diff title"


def test_default_titles_are_generic():
    ct = CT(cube_hu=sitk.Image(10, 10, 5, sitk.sitkFloat32))
    before = _make_dose_result(np.zeros((5, 10, 10), dtype=np.float32))
    after = _make_dose_result(np.ones((5, 10, 10), dtype=np.float32))

    fig = plot_dose_comparison(before, after, image_volume=ct, show_plot=False)[0]

    assert fig.axes[0].get_title() == "Before"
    assert fig.axes[1].get_title() == "After"
    assert fig.axes[2].get_title() == "Difference (After − Before)"


def _rays() -> list[Ray]:
    return [
        Ray(ray_pos_bev=[x, 1000.0, z], ray_pos=[x, 0.0, z], beamlets=[PhotonBixel(energy=6.0)])
        for x, z in ((-2.5, -2.5), (2.5, 2.5))
    ]


def _make_beam_dose_result(total: np.ndarray, beams: list[np.ndarray]) -> PlanResult:
    return PlanResult(
        quantities={"physical_dose": QuantityDistribution(values=total)},
        beam_quantities={"physical_dose": beams},
    )


@pytest.fixture
def per_beam_setup():
    ct = CT(cube_hu=sitk.Image(10, 10, 5, sitk.sitkFloat32))
    rng = np.random.default_rng(0)

    def _result():
        beams = [rng.random((5, 10, 10)).astype(np.float32) for _ in range(2)]
        return _make_beam_dose_result(sum(beams), beams)

    return ct, _result(), _result()


def test_per_beam_figures_are_added(per_beam_setup):
    ct, before, after = per_beam_setup

    figures = plot_dose_comparison(
        before, after, image_volume=ct, beam_indices="all", show_plot=False
    )

    assert len(figures) == 3
    assert figures[0]._suptitle is None
    assert [fig._suptitle.get_text() for fig in figures[1:]] == ["Beam 0", "Beam 1"]
    for fig in figures:
        assert [ax.get_title() for ax in fig.axes[:3]] == [
            "Before",
            "After",
            "Difference (After − Before)",
        ]


def test_per_beam_single_index(per_beam_setup):
    ct, before, after = per_beam_setup

    figures = plot_dose_comparison(before, after, image_volume=ct, beam_indices=1, show_plot=False)

    assert len(figures) == 2
    assert figures[1]._suptitle.get_text() == "Beam 1"


def test_per_beam_leaves_total_figure_unchanged(per_beam_setup):
    ct, before, after = per_beam_setup

    total_only = plot_dose_comparison(before, after, image_volume=ct, show_plot=False)[0]
    with_beams = plot_dose_comparison(
        before, after, image_volume=ct, beam_indices=[0], show_plot=False
    )[0]

    assert len(with_beams.axes) == len(total_only.axes)
    for i in range(3):
        assert with_beams.axes[i].get_title() == total_only.axes[i].get_title()
        assert with_beams.axes[i].collections[-1].get_clim() == pytest.approx(
            total_only.axes[i].collections[-1].get_clim()
        )


def test_per_beam_diff_windowed_per_figure(per_beam_setup):
    ct, before, after = per_beam_setup

    figures = plot_dose_comparison(
        before, after, image_volume=ct, beam_indices=[0], show_plot=False
    )

    total_diff = figures[0].axes[2].collections[-1]
    beam_diff = figures[1].axes[2].collections[-1]
    assert beam_diff.get_clim()[1] < total_diff.get_clim()[1]


def test_per_beam_titles_use_gantry_angle(per_beam_setup):
    ct, before, after = per_beam_setup
    stf = SteeringInformation(
        beams=[
            Beam(iso_center=[0.0, 0.0, 0.0], bixel_width=5.0, gantry_angle=90.0, rays=_rays()),
            Beam(iso_center=[0.0, 0.0, 0.0], bixel_width=5.0, gantry_angle=270.0, rays=_rays()),
        ]
    )

    figures = plot_dose_comparison(
        before, after, image_volume=ct, beam_indices="all", stf=stf, show_plot=False
    )

    assert [fig._suptitle.get_text() for fig in figures[1:]] == ["Beam 0 (90°)", "Beam 1 (270°)"]


def test_per_beam_figures_are_saved(per_beam_setup, tmp_path):
    ct, before, after = per_beam_setup

    plot_dose_comparison(
        before,
        after,
        image_volume=ct,
        beam_indices="all",
        show_plot=False,
        save_filename=str(tmp_path / "dose.png"),
    )

    assert (tmp_path / "dose.png").exists()
    assert (tmp_path / "dose_beam0.png").exists()
    assert (tmp_path / "dose_beam1.png").exists()


def test_per_beam_requires_per_beam_dose():
    ct = CT(cube_hu=sitk.Image(10, 10, 5, sitk.sitkFloat32))
    before = _make_dose_result(np.zeros((5, 10, 10), dtype=np.float32))
    after = _make_dose_result(np.ones((5, 10, 10), dtype=np.float32))

    with pytest.raises(ValueError):
        plot_dose_comparison(before, after, image_volume=ct, beam_indices=0, show_plot=False)


def test_per_beam_index_out_of_range(per_beam_setup):
    ct, before, after = per_beam_setup

    with pytest.raises(IndexError):
        plot_dose_comparison(before, after, image_volume=ct, beam_indices=5, show_plot=False)
