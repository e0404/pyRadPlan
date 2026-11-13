import pytest

import numpy as np
import SimpleITK as sitk
import matplotlib

from pyRadPlan import CT, StructureSet
from pyRadPlan import plot_slice

# Use Agg backend for matplotlib, so plots are not displayed during testing. (even if plt_show=False)
# When running tests locally, you can find the files in
# %appdata%\local\Temp\pytest-of-<username>\pytest-<number>.
matplotlib.use("Agg")


@pytest.fixture
def sample_ct():
    cube_hu = sitk.Image(100, 100, 10, sitk.sitkFloat32)
    return CT(cube_hu=cube_hu)


@pytest.fixture
def sample_cst(sample_ct):
    mask_np = np.zeros((10, 100, 100), dtype=np.uint8)
    mask_np[1:8, 20:80, 20:80] = 1
    mask = sitk.GetImageFromArray(mask_np)
    return StructureSet(
        vois=[{"mask": mask, "voi_type": "TARGET", "name": "test", "ct_image": sample_ct}],
        ct_image=sample_ct,
    )


@pytest.fixture
def sample_overlay():
    rand_image = np.random.rand(10, 100, 100).astype(np.float32)
    overlay = sitk.GetImageFromArray(rand_image)
    return overlay


def test_plot_slice_noargs():
    with pytest.raises(ValueError):
        plot_slice()


def test_plot_slice_ct(sample_ct, tmp_path):
    plot_slice(
        image_volume=sample_ct, save_filename=str(tmp_path / "slice_ct.png"), show_plot=False
    )


def test_plot_slice_ct_alias_deprecated(sample_ct, tmp_path):
    with pytest.warns(DeprecationWarning, match="ct"):
        plot_slice(
            ct=sample_ct, save_filename=str(tmp_path / "slice_ct_alias.png"), show_plot=False
        )


def test_plot_slice_ct_window_alias_deprecated(sample_ct, tmp_path):
    with pytest.warns(DeprecationWarning, match="ct_window"):
        plot_slice(
            image_volume=sample_ct,
            ct_window=(-100, 100),
            save_filename=str(tmp_path / "slice_ct_window_alias.png"),
            show_plot=False,
        )


def test_plot_slice_ct_and_image_volume_conflict(sample_ct):
    with pytest.raises(TypeError):
        plot_slice(image_volume=sample_ct, ct=sample_ct, show_plot=False)


def test_plot_slice_cst(sample_cst, tmp_path):
    plot_slice(cst=sample_cst, save_filename=str(tmp_path / "slice_cst.png"), show_plot=False)


def test_plot_slice_ct_cst(sample_ct, sample_cst, tmp_path):
    plot_slice(
        image_volume=sample_ct,
        cst=sample_cst,
        save_filename=str(tmp_path / "slice_ct_cst.png"),
        show_plot=False,
    )


def test_plot_slice_with_overlay(sample_ct, sample_cst, sample_overlay, tmp_path):
    plot_slice(
        image_volume=sample_ct,
        cst=sample_cst,
        overlay=sample_overlay,
        overlay_unit="Gy",
        view_slice=5,
        save_filename=str(tmp_path / "slice_with_overlay.png"),
        show_plot=False,
    )


def test_plot_slice_parameters_coronal(sample_ct, sample_cst, sample_overlay, tmp_path):
    plot_slice(
        image_volume=sample_ct,
        cst=sample_cst,
        overlay=sample_overlay,
        view_slice=[50, 2, 5],
        plane="coronal",
        overlay_unit="Gy",
        contour_line_width=2,
        overlay_alpha=0.2,
        overlay_rel_threshold=0.5,
        save_filename=str(tmp_path / "slice_parameters_coronal.png"),
        show_plot=False,
    )


def test_plot_slice_parameters_sagittal(sample_ct, sample_cst, sample_overlay, tmp_path):
    plot_slice(
        image_volume=sample_ct,
        cst=sample_cst,
        overlay=sample_overlay,
        plane="sagittal",
        save_filename=str(tmp_path / "parameters_sagittal.png"),
        show_plot=False,
    )


def test_plot_slice_parameters_invalid_plane(sample_ct, sample_cst, sample_overlay, tmp_path):
    with pytest.raises(ValueError):
        plot_slice(
            image_volume=sample_ct,
            cst=sample_cst,
            overlay=sample_overlay,
            plane="invalid_plane",
            show_plot=False,
        )


def test_plot_slice_multiple_slices_ct(sample_ct, tmp_path):
    plot_slice(
        image_volume=sample_ct,
        view_slice=[2, 5, 8],
        save_filename=str(tmp_path / "slice_multiple_slices_ct.png"),
        show_plot=False,
    )


def test_global_max(sample_ct, sample_cst, sample_overlay, tmp_path):
    sample_overlay = sitk.GetArrayFromImage(sample_overlay)
    sample_overlay[2] = sample_overlay[2] * 3
    plot_slice(
        image_volume=sample_ct,
        cst=sample_cst,
        overlay=sample_overlay,
        use_global_max=True,
        view_slice=[2, 5, 8],
        show_plot=False,
        save_filename=str(tmp_path / "global_max.png"),
    )


def test_plot_multiple_slices(sample_ct, sample_cst, sample_overlay, tmp_path):
    sample_overlay = sitk.GetArrayFromImage(sample_overlay)
    plot_slice(
        image_volume=sample_ct,
        cst=sample_cst,
        overlay=sample_overlay,
        view_slice=[2, 5, 8],
        show_plot=False,
        save_filename=str(tmp_path / "plot_multiple_slices.png"),
    )


def test_plot_multiple_slices_numpy(sample_ct, sample_cst, sample_overlay, tmp_path):
    sample_overlay = sitk.GetArrayFromImage(sample_overlay)
    plot_slice(
        image_volume=sample_ct,
        cst=sample_cst,
        overlay=sample_overlay,
        view_slice=np.array([2, 5, 8]),
        show_plot=False,
        save_filename=str(tmp_path / "plot_multiple_slices.png"),
    )


@pytest.mark.parametrize("offset", [0.0, -0.7])
def test_overlay_with_negative_values(sample_ct, sample_overlay, offset):
    import matplotlib.pyplot as plt

    overlay = sitk.GetArrayFromImage(sample_overlay) + offset
    overlay_2d = overlay[5]
    _, ax = plt.subplots()
    plot_slice(image_volume=sample_ct, overlay=overlay, view_slice=5, ax=ax, show_plot=False)

    mesh = ax.collections[-1]
    max_abs = np.max(np.abs(overlay_2d))
    np.testing.assert_allclose(mesh.norm.vmax, max_abs)
    if offset < 0:
        np.testing.assert_allclose(mesh.norm.vmin, -max_abs)
        assert mesh.get_cmap().name == "RdBu_r"
        # strongly negative voxels are drawn, not hidden by the threshold
        alpha = np.asarray(mesh.get_alpha()).reshape(overlay_2d.shape)
        assert np.all(alpha[overlay_2d < -0.5 * max_abs] > 0)
    else:
        assert mesh.norm.vmin == 0
        assert mesh.get_cmap().name == "jet"
    plt.close("all")


def test_image_windowing(sample_ct, sample_cst, sample_overlay, tmp_path):
    plot_slice(
        image_volume=sample_ct,
        cst=sample_cst,
        overlay=sample_overlay,
        image_window=(-200, 200),
        window_mode="minmax",
        view_slice=5,
        save_filename=str(tmp_path / "image_windowing_minmax.png"),
        show_plot=False,
    )
    plot_slice(
        image_volume=sample_ct,
        cst=sample_cst,
        overlay=sample_overlay,
        image_window=(-200, 200),
        window_mode="centerwidth",
        view_slice=5,
        save_filename=str(tmp_path / "image_windowing_centerwidth.png"),
        show_plot=False,
    )
