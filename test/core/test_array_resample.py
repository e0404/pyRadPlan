"""Backend image resampling agrees with SimpleITK, including invalid depths."""

import array_api_compat
import numpy as np
import pytest
import SimpleITK as sitk

from pyRadPlan.core import Grid
from pyRadPlan.core.resample import ArrayBackendGridResampler
from pyRadPlan.core.resample import resample_image
from pyRadPlan.core.xp_utils import to_numpy
from pyRadPlan.core.xp_utils._image_interpolation import interpolate_image_linear


@pytest.fixture(params=["numpy", "array_api_strict", "cupy"])
def xp(request):
    """Provide a real backend or the strict Array API reference."""
    module = pytest.importorskip(request.param)
    if request.param == "cupy" and not module.cuda.is_available():
        pytest.skip("CUDA unavailable")
    return array_api_compat.get_namespace(module.asarray([0.0]))


@pytest.mark.parametrize("shape", [(5, 6, 7), (1, 6, 7)])
@pytest.mark.parametrize("with_nan", [False, True])
@pytest.mark.parametrize("oblique", [False, True])
def test_resampler_matches_sitk(xp, shape, with_nan, oblique):
    """Compare finite values and NaN footprints on different physical grids."""
    values = np.random.default_rng(23).normal(size=shape).astype(np.float32)
    if with_nan:
        values.ravel()[::7] = np.nan
    image = sitk.GetImageFromArray(values)
    image.SetOrigin((3.2, -5.7, 0.3))
    image.SetSpacing((1.3, 2.1, 3.4))
    if oblique:
        a = 0.3
        image.SetDirection((np.cos(a), -np.sin(a), 0.0, np.sin(a), np.cos(a), 0.0, 0.0, 0.0, 1.0))
    source = Grid.from_sitk_image(image)
    target = Grid(
        resolution=dict(x=0.9, y=1.7, z=2.3),
        dimensions=(13, 10, 9),
        origin=np.array([-0.5, -7.1, -2.1]),
    )
    expected = sitk.GetArrayFromImage(
        resample_image(image, interpolator=sitk.sitkLinear, target_grid=target)
    ).ravel()
    # Includes arbitrary order, duplicate indices and small chunk boundaries.
    indices = np.concatenate((np.arange(target.num_voxels)[::-1], [0, 3, 0]))
    sample = ArrayBackendGridResampler(source, target, xp.asarray(indices), chunk_size=37)
    actual = to_numpy(sample(xp.asarray(values)))
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected[indices]))
    np.testing.assert_allclose(actual, expected[indices], rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize(
    "point",
    [
        (1.0, 1.0, 1.0),
        (0.0, 0.0, 0.0),
        (-0.5, 1.2, 1.3),
        (-0.5001, 1.2, 1.3),
        (2.5, 1.2, 1.3),
        (2.4999, 1.2, 1.3),
        (1.0, 0.2, 0.3),
    ],
)
def test_nan_neighbors_and_half_voxel_boundaries(xp, point):
    """Preserve exact samples and nearest extrapolation at image boundaries."""
    values = np.arange(27, dtype=np.float32).reshape(3, 3, 3)
    values[2, 2, 2] = np.nan
    image = sitk.GetImageFromArray(values)
    target = Grid(
        resolution=dict(x=1.0, y=1.0, z=1.0), dimensions=(1, 1, 1), origin=np.array(point)
    )
    expected = sitk.GetArrayFromImage(
        resample_image(image, interpolator=sitk.sitkLinear, target_grid=target)
    ).ravel()
    actual = interpolate_image_linear(xp.asarray(values), xp.asarray([point], dtype=xp.float64))
    np.testing.assert_allclose(to_numpy(actual), expected, rtol=1e-6, atol=1e-6)


def test_matching_grid_and_empty_selection(xp):
    """Gather exact samples on an unchanged grid, including empty selections."""
    grid = Grid(resolution=dict(x=1.0, y=1.0, z=1.0), dimensions=(3, 3, 3))
    values = xp.reshape(xp.arange(27, dtype=xp.float32), (3, 3, 3))
    for indices in ([26, 0, 3, 3], []):
        sample = ArrayBackendGridResampler(grid, grid, xp.asarray(indices, dtype=xp.int64))
        np.testing.assert_array_equal(to_numpy(sample(values)), np.array(indices))


@pytest.mark.parametrize("indices", [[-1], [27]])
def test_resampler_rejects_invalid_indices(xp, indices):
    """Do not let invalid target offsets wrap into a different physical location."""
    grid = Grid(resolution=dict(x=1.0, y=1.0, z=1.0), dimensions=(3, 3, 3))
    with pytest.raises(ValueError, match="outside"):
        ArrayBackendGridResampler(grid, grid, xp.asarray(indices, dtype=xp.int64))


@pytest.mark.parametrize("angle", [0.2, 0.7])
def test_resampled_oblique_grid_preserves_nan_footprint(xp, angle):
    """Grid-aligned queries must retain the image resampler's NaN boundary behavior."""
    values = np.random.default_rng(11).normal(size=(7, 8, 9)).astype(np.float32)
    values.ravel()[::13] = np.nan
    image = sitk.GetImageFromArray(values)
    image.SetSpacing((1.3, 2.1, 1.7))
    image.SetOrigin((-3.0, 8.0, -2.0))
    image.SetDirection(
        (np.cos(angle), -np.sin(angle), 0.0, np.sin(angle), np.cos(angle), 0.0, 0.0, 0.0, 1.0)
    )
    source = Grid.from_sitk_image(image)
    target = source.resample(dict(x=2.1, y=2.9, z=3.1))
    expected = sitk.GetArrayFromImage(
        resample_image(image, interpolator=sitk.sitkLinear, target_grid=target)
    ).ravel()
    sample = ArrayBackendGridResampler(source, target, xp.arange(target.num_voxels))
    np.testing.assert_allclose(
        to_numpy(sample(xp.asarray(values))), expected, rtol=2e-6, atol=2e-6
    )
