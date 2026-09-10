import numpy as np
import pytest
import SimpleITK as sitk
import matplotlib

matplotlib.use("Agg")

from pyRadPlan import CT
from pyRadPlan.result import PlanResult, QuantityDistribution
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
    )
    assert len(fig.axes) >= 3
    assert (tmp_path / "dose.png").exists()


def test_diff_panel_uses_diverging_colormap_regardless_of_overlay_cmap():
    ct = CT(cube_hu=sitk.Image(10, 10, 5, sitk.sitkFloat32))
    before = _make_dose_result(np.zeros((5, 10, 10), dtype=np.float32))
    after = _make_dose_result(np.ones((5, 10, 10), dtype=np.float32))

    fig = plot_dose_comparison(before, after, image_volume=ct, overlay_cmap="jet", show_plot=False)

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

    fig = plot_dose_comparison(before, after, image_volume=ct, show_plot=False)

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

    fig = plot_dose_comparison(before, after, image_volume=ct, show_plot=False)

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
    )

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
    )

    assert fig.axes[2].get_title() == "Custom diff title"


def test_default_titles_are_generic():
    ct = CT(cube_hu=sitk.Image(10, 10, 5, sitk.sitkFloat32))
    before = _make_dose_result(np.zeros((5, 10, 10), dtype=np.float32))
    after = _make_dose_result(np.ones((5, 10, 10), dtype=np.float32))

    fig = plot_dose_comparison(before, after, image_volume=ct, show_plot=False)

    assert fig.axes[0].get_title() == "Before"
    assert fig.axes[1].get_title() == "After"
    assert fig.axes[2].get_title() == "Difference (After − Before)"
