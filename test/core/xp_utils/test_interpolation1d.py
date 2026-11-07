"""Prepared 1D interpolation retains one-shot dtype, shape and boundary contracts."""

import array_api_compat
import numpy as np
import pytest

from pyRadPlan.core.xp_utils import ArrayBackend1DInterpolator, interp1d, to_numpy


@pytest.fixture(params=["numpy", "torch", "array_api_strict"])
def xp(request):
    backend = pytest.importorskip(request.param)
    return array_api_compat.array_namespace(backend.asarray([0.0]))


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("kind", ["array", "rows", "fields", "mixed_rows", "mixed_dtype"])
@pytest.mark.parametrize("boundary", [None, -3.5])
@pytest.mark.parametrize("offset", [None, -0.1723, 13.234])
def test_prepared_matches_one_shot(xp, dtype, kind, boundary, offset):
    x = xp.asarray([0.0, 0.3, 2.4, 5.0], dtype=getattr(xp, dtype))
    y = xp.asarray([1.0, -2.0, 4.0, 3.0], dtype=getattr(xp, dtype))
    if kind == "rows":
        y = xp.stack([y, y * 2], axis=0)
    elif kind != "array":
        y = {"one": y, "two": y * 2}
        if kind == "mixed_rows":
            y["two"] = xp.stack([y["one"] * 2, y["one"] * 3], axis=0)
        if kind == "mixed_dtype":
            y["two"] = xp.astype(y["two"], xp.float64 if dtype == "float32" else xp.float32)
    obj = ArrayBackend1DInterpolator(x, y, left=boundary, right=boundary)
    grid = x if offset is None else x + offset
    q = xp.concat(
        [grid, xp.asarray([-9.0, 0.11, 2.2, 40.0, np.nan, np.inf, -np.inf], dtype=x.dtype)]
    )
    for queries in (q, q[:0], xp.flip(q, axis=0)):
        actual = obj(queries, grid_offset=offset)
        expected = interp1d(queries, grid, y, left=boundary, right=boundary)
        if not isinstance(actual, dict):
            actual, expected = {None: actual}, {None: expected}
        assert tuple(actual) == tuple(expected)
        for key in actual:
            assert actual[key].dtype == expected[key].dtype
            assert actual[key].shape == expected[key].shape
            assert array_api_compat.device(actual[key]) == array_api_compat.device(x)
            np.testing.assert_allclose(to_numpy(actual[key]), to_numpy(expected[key]), rtol=2e-6)


@pytest.mark.parametrize("prefer_native", [None, True, False])
def test_dispatch_preference(xp, prefer_native):
    x = xp.asarray([0.0, 1.0, 3.0], dtype=xp.float64)
    y = {"a": x * 2, "b": xp.stack([x * 3, x * 4])}
    q = xp.asarray([-1.0, 0.5, 2.0, 4.0], dtype=xp.float64)
    actual = ArrayBackend1DInterpolator(x, y, prefer_native=prefer_native)(q)
    expected = np.clip(to_numpy(q), 0, 3)
    np.testing.assert_allclose(to_numpy(actual["a"]), expected * 2)
    np.testing.assert_allclose(to_numpy(actual["b"]), np.stack([expected * 3, expected * 4]))


def test_invalid_inputs_and_queries(xp):
    x = xp.asarray([0.0, 1.0, 2.0])
    for invalid in (x[:0], xp.flip(x, axis=0), xp.zeros_like(x)):
        with pytest.raises(ValueError):
            ArrayBackend1DInterpolator(invalid, invalid)
    with pytest.raises(ValueError, match="shape"):
        ArrayBackend1DInterpolator(x, x[:2])
    with pytest.raises(ValueError, match="field"):
        ArrayBackend1DInterpolator(x, {})
    with pytest.raises(TypeError, match="scalar"):
        ArrayBackend1DInterpolator(x, x, left=x)
    with pytest.raises(ValueError, match="one-dimensional"):
        ArrayBackend1DInterpolator(x, x)(xp.reshape(x, (3, 1)))


@pytest.mark.parametrize("boundary", [None, -4.0])
def test_singleton_axis(xp, boundary):
    x = xp.asarray([2.0], dtype=xp.float64)
    q = xp.asarray([0.0, 2.0, 5.0, np.nan], dtype=xp.float64)
    fields = {"a": x * 3, "b": xp.stack([x, x * 2])}
    obj = ArrayBackend1DInterpolator(x, fields, left=boundary, right=boundary)
    actual = obj(q)
    for key, value in fields.items():
        expected = np.apply_along_axis(
            lambda row: np.interp(to_numpy(q), to_numpy(x), row, left=boundary, right=boundary),
            -1,
            to_numpy(value),
        )
        np.testing.assert_array_equal(to_numpy(actual[key]), expected)
