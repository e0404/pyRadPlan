"""Tests for rectilinear n-dimensional interpolation."""

from types import SimpleNamespace

import array_api_strict as strictxp
import numpy as np
import pytest

from pyRadPlan.core.xp_utils import cupy_available, jax_available, pytorch_available
from pyRadPlan.core.xp_utils.compat import interpnd


# %% Input helpers
def _as_numpy(x):
    if hasattr(x, "get"):  # CuPy
        return x.get()
    if hasattr(x, "detach"):  # PyTorch
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _interp_input_2d(xp):
    xq = xp.asarray([[0.25, 0.5]], dtype=xp.float64)

    gx = xp.asarray([0.0, 1.0], dtype=xp.float64)
    gy = xp.asarray([0.0, 1.0], dtype=xp.float64)

    x = (gx, gy)

    # f(x,y) = x + 10*y
    y = xp.asarray(
        [
            [0.0, 10.0],  # x=0
            [1.0, 11.0],  # x=1
        ],
        dtype=xp.float64,
    )

    expected = xp.asarray([0.25 + 10 * 0.5], dtype=xp.float64)

    return xq, x, y, expected


def _interp_input_3d(xp):
    xq = xp.asarray([[0.25, 0.5, 0.75]], dtype=xp.float64)

    gx = xp.asarray([0.0, 1.0], dtype=xp.float64)
    gy = xp.asarray([0.0, 1.0], dtype=xp.float64)
    gz = xp.asarray([0.0, 1.0], dtype=xp.float64)

    x = (gx, gy, gz)

    # f(x,y,z) = x + 10*y + 100*z
    y = xp.asarray(
        [
            [
                [0.0, 100.0],
                [10.0, 110.0],
            ],
            [
                [1.0, 101.0],
                [11.0, 111.0],
            ],
        ],
        dtype=xp.float64,
    )

    expected = xp.asarray([0.25 + 10 * 0.5 + 100 * 0.75], dtype=xp.float64)

    return xq, x, y, expected


def _interp_input_4d(xp):
    xq = xp.asarray([[0.25, 0.5, 0.75, 0.125]], dtype=xp.float64)

    gx = xp.asarray([0.0, 1.0], dtype=xp.float64)
    gy = xp.asarray([0.0, 1.0], dtype=xp.float64)
    gz = xp.asarray([0.0, 1.0], dtype=xp.float64)
    gw = xp.asarray([0.0, 1.0], dtype=xp.float64)

    x = (gx, gy, gz, gw)

    # f(x,y,z,w) = x + 10*y + 100*z + 1000*w
    y = xp.asarray(
        [
            [
                [[0.0, 1000.0], [100.0, 1100.0]],
                [[10.0, 1010.0], [110.0, 1110.0]],
            ],
            [
                [[1.0, 1001.0], [101.0, 1101.0]],
                [[11.0, 1011.0], [111.0, 1111.0]],
            ],
        ],
        dtype=xp.float64,
    )

    expected = xp.asarray(
        [0.25 + 10 * 0.5 + 100 * 0.75 + 1000 * 0.125],
        dtype=xp.float64,
    )

    return xq, x, y, expected


# %% Test generic fallback (2D & 3D)
@pytest.mark.parametrize(
    "xq, x, y, expected",
    [_interp_input_2d(strictxp), _interp_input_3d(strictxp)],
    ids=["2d", "3d"],
)
def test_interpnd_generic_returns(xq, x, y, expected):
    result = interpnd(xq, x, y)

    np.testing.assert_allclose(_as_numpy(result), _as_numpy(expected))


# %% Test generic fallback - flipped axes (2D)
def test_interpnd_generic_2d_handles_descending_axes():
    xq, _, __, expected = _interp_input_2d(strictxp)
    gx = strictxp.asarray([1.0, 0.0], dtype=strictxp.float64)
    gy = strictxp.asarray([1.0, 0.0], dtype=strictxp.float64)
    x = (gx, gy)
    y = strictxp.asarray(
        [
            [11.0, 1.0],  # x = 1
            [10.0, 0.0],  # x = 0
        ],
        dtype=strictxp.float64,
    )

    result = interpnd(xq, x, y)

    np.testing.assert_allclose(_as_numpy(result), _as_numpy(expected))


# %% Test generic fallback - unsupported dimension (4D)
def test_interpnd_generic_unsupported_dimension_raises():
    xq, x, y, _ = _interp_input_4d(strictxp)
    with pytest.raises(
        NotImplementedError,
        match="Only 2D and 3D interpolation is currently implemented for the generic fallback. Note, that torch does use the generic fallback.",
    ):
        interpnd(xq, x, y)


# %% Test Numpy Backend
@pytest.mark.parametrize(
    "xq, x, y, expected",
    [
        _interp_input_2d(np),
        _interp_input_3d(np),
        _interp_input_4d(np),
    ],
    ids=["2d_numpy", "3d_numpy", "4d_numpy"],
)
def test_interpnd_different_array_api(xq, x, y, expected):
    result = interpnd(xq, x, y)

    np.testing.assert_allclose(_as_numpy(result), _as_numpy(expected))


# Test Jax Backend
@pytest.mark.skipif(not jax_available(), reason="JAX is not available")
def test_interpnd_jax():
    import jax.numpy as jnp

    for xq, x, y, expected in [
        _interp_input_2d(jnp),
        _interp_input_3d(jnp),
        _interp_input_4d(jnp),
    ]:
        result = interpnd(xq, x, y)
        np.testing.assert_allclose(_as_numpy(result), _as_numpy(expected))


# Test CuPy Backend
@pytest.mark.skipif(not cupy_available(), reason="CuPy is not available")
def test_interpnd_cupy():
    import cupy as cp

    for xq, x, y, expected in [
        _interp_input_2d(cp),
        _interp_input_3d(cp),
        _interp_input_4d(cp),
    ]:
        result = interpnd(xq, x, y)
        np.testing.assert_allclose(_as_numpy(result), _as_numpy(expected))


# Test PyTorch Backend
@pytest.mark.skipif(not pytorch_available(), reason="PyTorch is not available")
def test_interpnd_pytorch():
    import torch

    for xq, x, y, expected in [
        _interp_input_2d(torch),
        _interp_input_3d(torch),
    ]:
        result = interpnd(xq, x, y)
        np.testing.assert_allclose(_as_numpy(result), _as_numpy(expected))
