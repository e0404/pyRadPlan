import pytest

import numpy as np
import SimpleITK as sitk
import matplotlib

from pyRadPlan import CT, StructureSet
from pyRadPlan import plot_multiple_slices

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
def sample_overlays():
    # Create two different overlays for testing
    overlay1 = np.random.rand(10, 100, 100).astype(np.float32)
    overlay2 = np.random.rand(10, 100, 100).astype(np.float32) * 0.5
    return [sitk.GetImageFromArray(overlay1), sitk.GetImageFromArray(overlay2)]


@pytest.fixture
def sample_overlays_numpy():
    # Create two different numpy overlays for testing
    overlay1 = np.random.rand(10, 100, 100).astype(np.float32)
    overlay2 = np.random.rand(10, 100, 100).astype(np.float32) * 0.5
    return [overlay1, overlay2]


def test_plot_multiple_slices_noargs():
    with pytest.raises(ValueError):
        plot_multiple_slices()


def test_plot_multiple_slices_no_overlays(sample_ct, tmp_path):
    plot_multiple_slices(
        ct=sample_ct,
        save_filename=str(tmp_path / "distributions_no_overlays.png"),
        show_plot=False,
    )


def test_plot_multiple_slices_empty_overlays(sample_ct):
    with pytest.raises(ValueError):
        plot_multiple_slices(ct=sample_ct, overlays=[])


def test_plot_multiple_slices_ct_single_overlay(sample_ct, sample_overlays, tmp_path):
    plot_multiple_slices(
        ct=sample_ct,
        overlays=[sample_overlays[0]],
        save_filename=str(tmp_path / "distributions_ct_single.png"),
        show_plot=False,
    )


def test_plot_multiple_slices_ct_multiple_overlays(sample_ct, sample_overlays, tmp_path):
    plot_multiple_slices(
        ct=sample_ct,
        overlays=sample_overlays,
        save_filename=str(tmp_path / "distributions_ct_multiple.png"),
        show_plot=False,
    )


def test_plot_multiple_slices_ct_cst_overlays(sample_ct, sample_cst, sample_overlays, tmp_path):
    plot_multiple_slices(
        ct=sample_ct,
        cst=sample_cst,
        overlays=sample_overlays,
        save_filename=str(tmp_path / "distributions_ct_cst_overlays.png"),
        show_plot=False,
    )


def test_plot_multiple_slices_with_parameters(sample_ct, sample_cst, sample_overlays, tmp_path):
    plot_multiple_slices(
        ct=sample_ct,
        cst=sample_cst,
        overlays=sample_overlays,
        view_slice=[2, 5],
        plane="axial",
        overlay_unit=["Gy", "dimensionless"],
        overlay_alpha=0.7,
        overlay_rel_threshold=0.05,
        contour_line_width=2.0,
        use_global_max=True,
        overlay_titles=["Physical Dose", "Biological Effect"],
        save_filename=str(tmp_path / "distributions_with_parameters.png"),
        show_plot=False,
    )


def test_plot_multiple_slices_coronal_plane(sample_ct, sample_cst, sample_overlays, tmp_path):
    plot_multiple_slices(
        ct=sample_ct,
        cst=sample_cst,
        overlays=sample_overlays,
        view_slice=[30, 40, 50],
        plane="coronal",
        overlay_unit=["Gy", "dimensionless"],
        save_filename=str(tmp_path / "distributions_coronal.png"),
        show_plot=False,
    )


def test_plot_multiple_slices_sagittal_plane(sample_ct, sample_cst, sample_overlays, tmp_path):
    plot_multiple_slices(
        ct=sample_ct,
        cst=sample_cst,
        overlays=sample_overlays,
        view_slice=[30, 40],
        plane="sagittal",
        save_filename=str(tmp_path / "distributions_sagittal.png"),
        show_plot=False,
    )


def test_plot_multiple_slices_invalid_plane(sample_ct, sample_overlays):
    with pytest.raises(ValueError):
        plot_multiple_slices(
            ct=sample_ct,
            overlays=sample_overlays,
            plane="invalid_plane",
            show_plot=False,
        )


def test_plot_multiple_slices_numpy_overlays(
    sample_ct, sample_cst, sample_overlays_numpy, tmp_path
):
    plot_multiple_slices(
        ct=sample_ct,
        cst=sample_cst,
        overlays=sample_overlays_numpy,
        save_filename=str(tmp_path / "distributions_numpy_overlays.png"),
        show_plot=False,
    )


def test_plot_multiple_slices_mixed_overlays(
    sample_ct, sample_overlays, sample_overlays_numpy, tmp_path
):
    # Mix SimpleITK and numpy overlays
    mixed_overlays = [sample_overlays[0], sample_overlays_numpy[1]]
    plot_multiple_slices(
        ct=sample_ct,
        overlays=mixed_overlays,
        save_filename=str(tmp_path / "distributions_mixed_overlays.png"),
        show_plot=False,
    )


def test_plot_multiple_slices_single_slice(sample_ct, sample_cst, sample_overlays, tmp_path):
    plot_multiple_slices(
        ct=sample_ct,
        cst=sample_cst,
        overlays=sample_overlays,
        view_slice=5,  # Single slice instead of list
        save_filename=str(tmp_path / "distributions_single_slice.png"),
        show_plot=False,
    )


def test_plot_multiple_slices_numpy_slice_array(sample_ct, sample_cst, sample_overlays, tmp_path):
    plot_multiple_slices(
        ct=sample_ct,
        cst=sample_cst,
        overlays=sample_overlays,
        view_slice=np.array([2, 5, 8]),
        save_filename=str(tmp_path / "distributions_numpy_slice_array.png"),
        show_plot=False,
    )


def test_plot_multiple_slices_mismatched_units(sample_ct, sample_overlays):
    # Test with mismatched number of units vs overlays
    with pytest.raises(ValueError):
        plot_multiple_slices(
            ct=sample_ct,
            overlays=sample_overlays,  # 2 overlays
            overlay_unit=["Gy"],  # Only 1 unit
            show_plot=False,
        )


def test_plot_multiple_slices_mismatched_titles(sample_ct, sample_overlays):
    # Test with mismatched number of titles vs overlays
    plot_multiple_slices(
        ct=sample_ct,
        overlays=sample_overlays,  # 2 overlays
        overlay_titles=["Physical Dose"],  # Only 1 title
        show_plot=False,
    )


def test_plot_multiple_slices_three_overlays(
    sample_ct, sample_cst, sample_overlays_numpy, tmp_path
):
    # Test with three overlays
    overlay3 = np.random.rand(10, 100, 100).astype(np.float32) * 0.3
    three_overlays = sample_overlays_numpy + [overlay3]

    plot_multiple_slices(
        ct=sample_ct,
        cst=sample_cst,
        overlays=three_overlays,
        overlay_unit=["Gy", "dimensionless", "dimensionless"],
        overlay_titles=["Physical Dose", "Biological Dose", "Effect"],
        save_filename=str(tmp_path / "distributions_three_overlays.png"),
        show_plot=False,
    )


def test_plot_multiple_slices_global_max_single_overlay(
    sample_ct, sample_overlays_numpy, tmp_path
):
    # Test global_max with single overlay (should not affect anything)
    overlay_modified = sample_overlays_numpy[0].copy()
    overlay_modified[2] = overlay_modified[2] * 3  # Make one slice much brighter

    plot_multiple_slices(
        ct=sample_ct,
        overlays=[overlay_modified],
        view_slice=[2, 5, 8],
        use_global_max=True,
        save_filename=str(tmp_path / "distributions_global_max_single.png"),
        show_plot=False,
    )


def test_plot_multiple_slices_no_ct_only_overlays(sample_overlays, tmp_path):
    with pytest.raises(ValueError):
        plot_multiple_slices(
            overlays=sample_overlays,
            save_filename=str(tmp_path / "distributions_no_ct.png"),
            show_plot=False,
        )


def test_plot_multiple_slices_only_cst(sample_cst, tmp_path):
    # Test with only CST and no overlays (should raise error)
    with pytest.raises(ValueError):
        plot_multiple_slices(
            cst=sample_cst,
            show_plot=False,
        )


def test_plot_multiple_slices_custom_threshold(sample_ct, sample_overlays, tmp_path):
    plot_multiple_slices(
        ct=sample_ct,
        overlays=sample_overlays,
        overlay_rel_threshold=0.1,  # Higher threshold
        save_filename=str(tmp_path / "distributions_custom_threshold.png"),
        show_plot=False,
    )


def test_plot_multiple_slices_high_alpha(sample_ct, sample_overlays, tmp_path):
    plot_multiple_slices(
        ct=sample_ct,
        overlays=sample_overlays,
        overlay_alpha=0.9,  # High transparency
        save_filename=str(tmp_path / "distributions_high_alpha.png"),
        show_plot=False,
    )
