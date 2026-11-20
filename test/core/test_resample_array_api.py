"""Public array resampling preserves enums, geometry, namespace and compatibility."""

import importlib

import array_api_compat
import numpy as np
import pytest
import SimpleITK as sitk

from pyRadPlan.core import Grid, resample_array, ArrayBackendGridResampler
from pyRadPlan.core.resample import resample_image, resample_numpy_array
from pyRadPlan.core.xp_utils import to_numpy


@pytest.fixture(params=["numpy", "array_api_strict", "torch", "cupy", "jax.numpy"])
def xp(request):
    """Exercise host, device, strict and immutable Array API execution."""
    module = pytest.importorskip(request.param)
    if request.param == "cupy" and not module.cuda.is_available():
        pytest.skip("CUDA unavailable")
    if request.param == "jax.numpy":
        jax = importlib.import_module("jax")
        previous = jax.config.jax_enable_x64
        jax.config.update("jax_enable_x64", True)
        try:
            yield array_api_compat.array_namespace(module.asarray([0.0]))
        finally:
            jax.config.update("jax_enable_x64", previous)
    else:
        yield array_api_compat.array_namespace(module.asarray([0.0]))


def images(dtype=np.float32):
    """Use mismatched grids including points outside the input image."""
    values = np.arange(5 * 6 * 7, dtype=dtype).reshape(5, 6, 7)
    image = sitk.GetImageFromArray(values)
    image.SetOrigin((3.4, -2.8, 7.1))
    image.SetSpacing((1.3, 1.7, 2.3))
    target = Grid(
        resolution=dict(x=1.1, y=2.1, z=2.9),
        dimensions=(10, 8, 7),
        origin=np.array([2.1, -3.3, 5.7]),
    )
    return values, image, target


@pytest.mark.parametrize(
    "method", [sitk.sitkLinear, sitk.sitkNearestNeighbor, sitk.sitkBSpline, sitk.sitkGaussian]
)
@pytest.mark.parametrize("extrapolate", ["nearest", -3.25, False, True])
def test_enums_extrapolation_and_backend(xp, method, extrapolate):
    """All enum requests retain native semantics, including host fallback modes."""
    values, image, target = images()
    expected = sitk.GetArrayFromImage(
        resample_image(image, interpolator=method, target_grid=target, extrapolate=extrapolate)
    )
    input_array = xp.asarray(values)
    actual = resample_array(
        input_array, image, interpolator=method, target_grid=target, extrapolate=extrapolate
    )
    assert array_api_compat.array_namespace(actual) == xp
    assert array_api_compat.device(actual) == array_api_compat.device(input_array)
    assert actual.dtype == input_array.dtype
    np.testing.assert_allclose(to_numpy(actual), expected, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("dtype", [np.int16, np.int64, np.float64])
@pytest.mark.parametrize("extrapolate", ["nearest", -5.5, False])
def test_nearest_dtypes_and_prepared_interface(xp, dtype, extrapolate):
    """Nearest-neighbor interpolation retains values and output dtype."""
    values, image, target = images(dtype)
    expected = sitk.GetArrayFromImage(
        resample_image(
            image,
            interpolator=sitk.sitkNearestNeighbor,
            target_grid=target,
            extrapolate=extrapolate,
        )
    )
    source = Grid.from_sitk_image(image)
    prepared = ArrayBackendGridResampler(
        source,
        target,
        xp.arange(target.num_voxels),
        interpolator=sitk.sitkNearestNeighbor,
        extrapolate=extrapolate,
    )
    actual = prepared(xp.asarray(values))
    np.testing.assert_array_equal(to_numpy(actual), expected.ravel())
    assert actual.dtype == xp.asarray(values).dtype
    full = resample_array(
        xp.asarray(values),
        reference_grid=source,
        interpolator=sitk.sitkNearestNeighbor,
        target_grid=target,
        extrapolate=extrapolate,
    )
    np.testing.assert_array_equal(to_numpy(full), expected)


def test_supported_device_mode_does_not_download(xp, monkeypatch):
    """An explicit backend request does not invoke the native fallback."""
    values, image, target = images()
    monkeypatch.setattr(
        "pyRadPlan.core.resample.to_numpy",
        lambda *a, **kw: pytest.fail("Unexpected host round trip"),
    )
    resample_array(
        xp.asarray(values),
        image,
        interpolator=sitk.sitkLinear,
        target_grid=target,
        prefer_native=False,
    )


def test_geometry_alternatives_and_deprecated_positional_call():
    """The legacy positional signature and all geometry specifications remain supported."""
    values, image, target = images()
    expected_image = resample_image(image, target_grid=target)
    expected = sitk.GetArrayFromImage(expected_image)
    with pytest.warns(DeprecationWarning, match="resample_array"):
        old = resample_numpy_array(
            values, image, None, sitk.sitkBSpline, None, None, target, "nearest"
        )
    np.testing.assert_array_equal(old, expected)
    actual = resample_array(
        values, reference_grid=Grid.from_sitk_image(image), target_image=expected_image
    )
    np.testing.assert_array_equal(actual, expected)
    by_spacing = resample_array(values, image, target_grid_spacing=(2.0, 2.0, 3.0))
    expected = sitk.GetArrayFromImage(resample_image(image, target_grid_spacing=(2.0, 2.0, 3.0)))
    np.testing.assert_array_equal(by_spacing, expected)


@pytest.mark.parametrize(
    "kwargs", [{}, {"reference_image": True, "reference_grid": True}, {"reference_image": True}]
)
def test_ambiguous_or_missing_geometry_raises(kwargs):
    """Require exactly one source and one target specification."""
    values, image, target = images()
    kwargs = {key: image if key.endswith("image") else target for key in kwargs}
    with pytest.raises(TypeError, match="Only one"):
        resample_array(values, **kwargs)


def test_small_bspline_fallback_is_preserved(xp):
    """The existing native small-image fallback still works on device inputs."""
    values = np.arange(27, dtype=np.float32).reshape(3, 3, 3)
    image = sitk.GetImageFromArray(values)
    grid = Grid.from_sitk_image(image).resample(dict(x=0.8, y=0.7, z=0.9))
    actual = resample_array(xp.asarray(values), image, target_grid=grid)
    expected = sitk.GetArrayFromImage(resample_image(image, target_grid=grid))
    np.testing.assert_array_equal(to_numpy(actual), expected)


def test_explicit_backend_request_rejects_unsupported_mode(xp):
    """Do not silently substitute a supported interpolator for the requested mode."""
    values, image, target = images()
    with pytest.raises(NotImplementedError, match="SimpleITK fallback"):
        resample_array(xp.asarray(values), image, target_grid=target, prefer_native=False)


@pytest.mark.parametrize("point", [-0.5001, -0.5, 0.5, 2.4999, 2.5])
def test_nearest_half_voxel_ties_and_padding(xp, point):
    """Half-voxel rounding and the upper exclusive image boundary match SimpleITK."""
    values = np.arange(27, dtype=np.int16).reshape(3, 3, 3)
    image = sitk.GetImageFromArray(values)
    target = Grid(
        resolution=dict(x=1.0, y=1.0, z=1.0),
        dimensions=(1, 1, 1),
        origin=np.array([point, 1.0, 1.0]),
    )
    actual = resample_array(
        xp.asarray(values),
        image,
        interpolator=sitk.sitkNearestNeighbor,
        target_grid=target,
        extrapolate=-7,
        prefer_native=False,
    )
    expected = sitk.GetArrayFromImage(
        resample_image(
            image, interpolator=sitk.sitkNearestNeighbor, target_grid=target, extrapolate=-7
        )
    )
    np.testing.assert_array_equal(to_numpy(actual), expected)
