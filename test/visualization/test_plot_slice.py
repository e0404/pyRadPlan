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
    plot_slice(ct=sample_ct, save_filename=str(tmp_path / "slice_ct.png"), show_plot=False)


def test_plot_slice_cst(sample_cst, tmp_path):
    plot_slice(cst=sample_cst, save_filename=str(tmp_path / "slice_cst.png"), show_plot=False)


def test_plot_slice_ct_cst(sample_ct, sample_cst, tmp_path):
    plot_slice(
        ct=sample_ct,
        cst=sample_cst,
        save_filename=str(tmp_path / "slice_ct_cst.png"),
        show_plot=False,
    )


def test_plot_slice_with_overlay(sample_ct, sample_cst, sample_overlay, tmp_path):
    plot_slice(
        ct=sample_ct,
        cst=sample_cst,
        overlay=sample_overlay,
        overlay_unit="Gy",
        view_slice=5,
        save_filename=str(tmp_path / "slice_with_overlay.png"),
        show_plot=False,
    )


def test_plot_slice_parameters_coronal(sample_ct, sample_cst, sample_overlay, tmp_path):
    plot_slice(
        ct=sample_ct,
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
        ct=sample_ct,
        cst=sample_cst,
        overlay=sample_overlay,
        plane="sagittal",
        save_filename=str(tmp_path / "parameters_sagittal.png"),
        show_plot=False,
    )


def test_plot_slice_parameters_invalid_plane(sample_ct, sample_cst, sample_overlay, tmp_path):
    with pytest.raises(ValueError):
        plot_slice(
            ct=sample_ct,
            cst=sample_cst,
            overlay=sample_overlay,
            plane="invalid_plane",
            show_plot=False,
        )


def test_plot_slice_multiple_slices_ct(sample_ct, tmp_path):
    plot_slice(
        ct=sample_ct,
        view_slice=[2, 5, 8],
        save_filename=str(tmp_path / "slice_multiple_slices_ct.png"),
        show_plot=False,
    )


def test_global_max(sample_ct, sample_cst, sample_overlay, tmp_path):
    sample_overlay = sitk.GetArrayFromImage(sample_overlay)
    sample_overlay[2] = sample_overlay[2] * 3
    plot_slice(
        ct=sample_ct,
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
        ct=sample_ct,
        cst=sample_cst,
        overlay=sample_overlay,
        view_slice=[2, 5, 8],
        show_plot=False,
        save_filename=str(tmp_path / "plot_multiple_slices.png"),
    )


def test_plot_multiple_slices_numpy(sample_ct, sample_cst, sample_overlay, tmp_path):
    sample_overlay = sitk.GetArrayFromImage(sample_overlay)
    plot_slice(
        ct=sample_ct,
        cst=sample_cst,
        overlay=sample_overlay,
        view_slice=np.array([2, 5, 8]),
        show_plot=False,
        save_filename=str(tmp_path / "plot_multiple_slices.png"),
    )
