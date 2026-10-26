"""Reusable interpolation contract across native and Array API implementations."""

import importlib
from unittest.mock import patch

import array_api_compat
import numpy as np
import pytest

from pyRadPlan.core.xp_utils import ArrayBackendGridInterpolator, cupy_available, to_numpy
from pyRadPlan.core.xp_utils import _interpolation
from pyRadPlan.core.xp_utils.compat import interpnd


@pytest.fixture(params=["numpy", "cupy", "array_api_strict", "torch", "jax.numpy"])
def xp(request):
    """Load optional backends only for their own cases."""
    if request.param == "cupy" and not cupy_available():
        pytest.skip("CuPy device unavailable")
    module = pytest.importorskip(request.param)
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


@pytest.mark.parametrize("prefer_native", [True, False])
@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
def test_repeated_queries(xp, prefer_native, ndim, dtype_name):
    """Nonuniform, descending grids reproduce a multilinear field and clip queries."""
    dtype = getattr(xp, dtype_name)
    points = (np.array([3.0, 1.0, -2.0]),) + (np.array([-1.0, 0.5, 2.0]),) * (ndim - 1)
    grids = np.meshgrid(*points, indexing="ij")
    values = sum((i + 1) * g for i, g in enumerate(grids)) + np.prod(grids, axis=0)
    coords = np.array([[0.25] * ndim, [3.0] * ndim, [-2.0] * ndim])
    clipped = np.column_stack(
        [np.clip(coords[:, i], p.min(), p.max()) for i, p in enumerate(points)]
    )
    expected = sum((i + 1) * clipped[:, i] for i in range(ndim)) + np.prod(clipped, axis=1)
    points = tuple(xp.asarray(p, dtype=dtype) for p in points)
    values = xp.asarray(values, dtype=dtype)
    queries = xp.asarray(coords, dtype=dtype)
    interpolator = ArrayBackendGridInterpolator(points, values, prefer_native=prefer_native)
    if not prefer_native:
        assert interpolator.implementation == "array_api"
    for _ in range(2):
        result = interpolator(queries)
        np.testing.assert_allclose(to_numpy(result), expected, rtol=2e-6, atol=2e-6)
        assert array_api_compat.device(result) == array_api_compat.device(values)
        assert array_api_compat.array_namespace(result) == xp
    np.testing.assert_array_equal(
        to_numpy(interpnd(queries, points, values, prefer_native=prefer_native)), to_numpy(result)
    )
    assert interpolator(xp.empty((0, ndim), dtype=dtype)).shape == (0,)


@pytest.mark.parametrize("prefer_native", [True, False])
def test_bounds_and_query_validation(xp, prefer_native):
    """Boundary points are accepted, out-of-grid coordinates and bad shapes rejected."""
    p = xp.asarray([0.0, 1.0])
    values = xp.asarray([[0.0, 1.0], [1.0, 2.0]])
    interpolator = ArrayBackendGridInterpolator(
        (p, p), values, bounds_error=True, prefer_native=prefer_native
    )
    np.testing.assert_allclose(
        to_numpy(interpolator(xp.asarray([[0.0, 0.0], [1.0, 1.0]]))), [0, 2]
    )
    for coords in [[-0.01, 0.5], [0.5, 1.01]]:
        with pytest.raises(ValueError):
            interpolator(xp.asarray([coords]))
    for shape in [(2,), (1, 3), (2, 1, 2)]:
        with pytest.raises(ValueError, match="Queries must have shape"):
            interpolator(xp.zeros(shape))
    assert interpolator(xp.empty((0, 2))).shape == (0,)


@pytest.mark.parametrize("prefer_native", [True, False])
@pytest.mark.parametrize("coords", [[0, 0], [0, 2, 1]])
def test_nonmonotonic_grid_rejected(xp, prefer_native, coords):
    """Ambiguous grids fail at construction, before any queries are evaluated."""
    p = xp.asarray(coords, dtype=xp.float32)
    with pytest.raises(ValueError, match="ascending or descending"):
        ArrayBackendGridInterpolator(
            (p, p), xp.ones((len(coords),) * 2), prefer_native=prefer_native
        )


@pytest.mark.parametrize("prefer_native", [True, False])
def test_grid_shape_validation(prefer_native):
    """Reject empty axes, non-axes, and a mismatched scalar-value grid."""
    for points, values in [
        ((), np.array(1.0)),
        ((np.array([]),), np.array([])),
        ((np.ones((2, 2)),), np.ones((2,))),
        ((np.arange(2), np.arange(3)), np.ones((2, 2))),
    ]:
        with pytest.raises(ValueError):
            ArrayBackendGridInterpolator(points, values, prefer_native=prefer_native)


@pytest.mark.parametrize("prefer_native", [True, False])
def test_four_dimensions(xp, prefer_native):
    """Native backends retain 4D support; generic support remains explicitly 2D/3D."""
    points = (xp.asarray([0.0, 1.0]),) * 4
    values = xp.ones((2, 2, 2, 2))
    generic = not prefer_native or not (
        array_api_compat.is_numpy_namespace(xp)
        or array_api_compat.is_cupy_namespace(xp)
        or array_api_compat.is_jax_namespace(xp)
    )
    if generic:
        with pytest.raises(NotImplementedError, match="Only 2D and 3D"):
            ArrayBackendGridInterpolator(points, values, prefer_native=prefer_native)
    else:
        interpolator = ArrayBackendGridInterpolator(points, values)
        np.testing.assert_allclose(to_numpy(interpolator(xp.asarray([[0.5] * 4]))), [1.0])


@pytest.mark.parametrize("backend", ["numpy", "cupy"])
def test_native_construction_is_reused(backend):
    """Query calls do not reconstruct the native object or overwrite another grid."""
    if backend == "cupy" and not cupy_available():
        pytest.skip("CuPy device unavailable")
    xp = importlib.import_module(backend)
    native = importlib.import_module(
        "scipy.interpolate" if backend == "numpy" else "cupyx.scipy.interpolate"
    )
    p = xp.asarray([0.0, 1.0])
    values = xp.asarray([[0.0, 1.0], [1.0, 2.0]])
    with patch.object(
        native, "RegularGridInterpolator", wraps=native.RegularGridInterpolator
    ) as ctor:
        first = ArrayBackendGridInterpolator((p, p), values)
        second = ArrayBackendGridInterpolator((p, p), 2 * values)
        for _ in range(3):
            q = xp.asarray([[0.25, 0.5]])
            np.testing.assert_allclose(to_numpy(first(q)), [0.75])
            np.testing.assert_allclose(to_numpy(second(q)), [1.5])
        assert ctor.call_count == 2


def test_generic_preparation_is_reused(monkeypatch):
    """Generic evaluation also avoids repeated grid preparation."""
    p = np.array([1.0, 0.0])
    interpolator = ArrayBackendGridInterpolator((p, p), np.ones((2, 2)), prefer_native=False)

    def unexpected(*args):
        pytest.fail("Grid normalization repeated during evaluation")

    monkeypatch.setattr(_interpolation, "_normalize_grid", unexpected)
    for _ in range(2):
        np.testing.assert_array_equal(interpolator(np.array([[0.25, 0.75]])), [1.0])


def test_native_unavailable_falls_back(monkeypatch):
    """Library availability does not remove the generic implementation."""
    monkeypatch.setattr(_interpolation, "_native_interpolator_class", lambda xp: None)
    p = np.array([0.0, 1.0])
    interpolator = ArrayBackendGridInterpolator((p, p), np.ones((2, 2)))
    assert interpolator.implementation == "array_api"
    np.testing.assert_array_equal(interpolator(np.array([[0.5, 0.5]])), [1.0])


def test_mixed_query_namespace_rejected():
    """Evaluation must not silently transfer query arrays to another backend."""
    strict = pytest.importorskip("array_api_strict")
    p = np.array([0.0, 1.0])
    interpolator = ArrayBackendGridInterpolator((p, p), np.ones((2, 2)))
    with pytest.raises(TypeError):
        interpolator(strict.asarray([[0.5, 0.5]]))


def test_mixed_devices_rejected():
    """Device mismatches fail before data access, without requiring a second GPU."""
    torch = pytest.importorskip("torch")
    p = torch.asarray([0.0, 1.0])
    values = torch.ones((2, 2))
    with pytest.raises(ValueError, match="share a device"):
        ArrayBackendGridInterpolator((p, torch.empty(2, device="meta")), values)
    interpolator = ArrayBackendGridInterpolator((p, p), values)
    with pytest.raises(ValueError, match="interpolator's device"):
        interpolator(torch.empty((1, 2), device="meta"))


def test_generic_singleton_axis_rejected():
    """The generic linear kernels need a cell with two points on each axis."""
    with pytest.raises(ValueError, match="at least two points"):
        ArrayBackendGridInterpolator(
            (np.array([0.0]), np.array([0.0, 1.0])), np.ones((1, 2)), prefer_native=False
        )


def test_query_subclass_uses_compatible_namespace():
    """A different Python array type is allowed when its namespace still matches."""

    class QueryArray(np.ndarray):
        pass

    p = np.array([0.0, 1.0])
    interpolator = ArrayBackendGridInterpolator((p, p), np.ones((2, 2)))
    np.testing.assert_array_equal(interpolator(np.array([[0.5, 0.5]]).view(QueryArray)), [1.0])
