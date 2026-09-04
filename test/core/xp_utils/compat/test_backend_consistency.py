"""Cross-backend tests for compatibility helpers."""

import numpy as np
import pytest
import array_api_strict as strictxp

try:
    import torch
except ImportError:
    torch = None

try:
    import jax.numpy as jnp
except ImportError:
    jnp = None

try:
    import cupy as cp

    try:
        cupy_available = cp.cuda.is_available()
    except cp.cuda.runtime.CUDARuntimeError:
        cupy_available = False
except ImportError:
    cp = None
    cupy_available = False

from pyRadPlan.core.xp_utils import to_numpy
from pyRadPlan.core.xp_utils.compat import array_meshgrid, interp1d, interpnd, _fft2, _ifft2


BACKENDS = [
    pytest.param(np, id="numpy"),
    pytest.param(strictxp, id="array_api_strict"),
    pytest.param(
        torch,
        id="torch",
        marks=pytest.mark.skipif(torch is None, reason="PyTorch is not available"),
    ),
    pytest.param(
        jnp,
        id="jax",
        marks=pytest.mark.skipif(jnp is None, reason="JAX is not available"),
    ),
    pytest.param(
        cp,
        id="cupy",
        marks=pytest.mark.skipif(not cupy_available, reason="CuPy requires an available GPU"),
    ),
]


@pytest.mark.parametrize("xp", BACKENDS)
def test_interp1d_2d_boundaries(xp):
    """Use the same 2D boundary semantics on every backend."""
    x = xp.asarray([0.0, 1.0, 2.0], dtype=xp.float32)
    y = xp.asarray([[0.0, 1.0, 2.0], [10.0, 11.0, 12.0]], dtype=xp.float32)
    xq = xp.asarray([-1.0, 0.5, 3.0], dtype=xp.float32)

    default_result = interp1d(xq, x, y)
    scalar_result = interp1d(xq, x, y, left=-5.0, right=99.0)
    row_result = interp1d(
        xq,
        x,
        y,
        left=xp.asarray([-5.0, -6.0], dtype=xp.float32),
        right=xp.asarray([98.0, 99.0], dtype=xp.float32),
    )

    np.testing.assert_allclose(to_numpy(default_result), [[0.0, 0.5, 2.0], [10.0, 10.5, 12.0]])
    np.testing.assert_allclose(to_numpy(scalar_result), [[-5.0, 0.5, 99.0], [-5.0, 10.5, 99.0]])
    np.testing.assert_allclose(to_numpy(row_result), [[-5.0, 0.5, 98.0], [-6.0, 10.5, 99.0]])


@pytest.mark.parametrize("xp", BACKENDS)
def test_interp1d_container_boundaries(xp):
    """Allow scalar but reject row-wise boundaries for y containers."""
    x = xp.asarray([0.0, 1.0, 2.0], dtype=xp.float32)
    xq = xp.asarray([-1.0, 0.5, 3.0], dtype=xp.float32)
    y1 = xp.asarray([0.0, 1.0, 2.0], dtype=xp.float32)
    y2 = xp.asarray([10.0, 11.0, 12.0], dtype=xp.float32)
    row_left = xp.asarray([-5.0, -6.0], dtype=xp.float32)

    containers = ([y1, y2], (y1, y2), {"first": y1, "second": y2})
    for y in containers:
        result = interp1d(xq, x, y, left=-5.0, right=99.0)
        result_values = result.values() if isinstance(result, dict) else result
        for actual, expected in zip(
            result_values,
            ([-5.0, 0.5, 99.0], [-5.0, 10.5, 99.0]),
            strict=True,
        ):
            np.testing.assert_allclose(to_numpy(actual), expected)

        with pytest.raises(ValueError, match="require y to be a single 2D array"):
            interp1d(xq, x, y, left=row_left)


@pytest.mark.parametrize("xp", BACKENDS)
def test_interp1d_multidimensional_queries(xp):
    """Preserve multidimensional query shapes for 2D y arrays."""
    x = xp.asarray([0.0, 1.0, 2.0], dtype=xp.float32)
    y = xp.asarray([[0.0, 1.0, 2.0], [10.0, 11.0, 12.0]], dtype=xp.float32)
    xq = xp.asarray([[-1.0, 0.5], [1.5, 3.0]], dtype=xp.float32)
    left = xp.asarray([-5.0, -6.0], dtype=xp.float32)
    right = xp.asarray([98.0, 99.0], dtype=xp.float32)

    result = interp1d(xq, x, y, left=left, right=right)

    expected = [[[-5.0, 0.5], [1.5, 98.0]], [[-6.0, 10.5], [11.5, 99.0]]]
    np.testing.assert_allclose(to_numpy(result), expected)


@pytest.mark.parametrize("xp", BACKENDS)
def test_interpnd_bounds_and_multiple_queries(xp):
    """Interpolate multiple points and apply the requested bounds behavior."""
    gx = xp.asarray([0.0, 1.0, 3.0], dtype=xp.float32)
    gy = xp.asarray([0.0, 2.0, 3.0], dtype=xp.float32)
    y = xp.asarray(
        [[0.0, 20.0, 30.0], [1.0, 21.0, 31.0], [3.0, 23.0, 33.0]],
        dtype=xp.float32,
    )
    xq = xp.asarray([[0.5, 1.0], [-1.0, 1.0], [2.0, 4.0]], dtype=xp.float32)

    result = interpnd(xq, (gx, gy), y)

    np.testing.assert_allclose(to_numpy(result), [10.5, 10.0, 32.0], rtol=1e-6)
    with pytest.raises(ValueError, match="outside the grid bounds"):
        interpnd(xq, (gx, gy), y, bounds_error=True)


@pytest.mark.parametrize("xp", BACKENDS)
def test_array_meshgrid_shapes(xp):
    """Return the expected xy and ij meshgrid shapes."""
    x = xp.asarray([0.0, 1.0], dtype=xp.float32)
    y = xp.asarray([0.0, 1.0, 2.0], dtype=xp.float32)

    xy = array_meshgrid(x, y, indexing="xy")
    ij = array_meshgrid(x, y, indexing="ij")

    assert [grid.shape for grid in xy] == [(3, 2), (3, 2)]
    assert [grid.shape for grid in ij] == [(2, 3), (2, 3)]


@pytest.mark.parametrize("xp", BACKENDS)
def test_fft2_round_trip(xp):
    """Recover the original array after a forward and inverse 2D FFT."""
    x = xp.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=xp.float32)

    result = _ifft2(_fft2(x, (2, 2)))

    np.testing.assert_allclose(to_numpy(xp.real(result)), to_numpy(x), rtol=1e-5, atol=1e-6)
