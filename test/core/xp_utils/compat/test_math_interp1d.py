import pytest
import array_api_strict as xp
import numpy as np
import array_api_compat
from pyRadPlan.core.xp_utils.compat import interp1d
from pyRadPlan.core.xp_utils import from_numpy, to_numpy


@pytest.mark.parametrize("dtype", [xp.float64, xp.float32])
def test_interp1d_basic(dtype):
    x = xp.asarray([0.0, 1.0, 2.0, 3.0], dtype=dtype)
    y = xp.asarray([0.0, 1.0, 4.0, 9.0], dtype=dtype)
    xq = xp.asarray([0.5, 1.5, 2.5], dtype=dtype)

    result = interp1d(xq, x, y)
    expected = np.interp(to_numpy(xq), to_numpy(x), to_numpy(y))
    assert np.allclose(to_numpy(result), expected, rtol=1e-6 if dtype == xp.float32 else 1e-7)


@pytest.mark.parametrize("dtype", [xp.float64, xp.float32])
def test_interp1d_on_points(dtype):
    x = xp.asarray([0.0, 1.0, 2.0], dtype=dtype)
    y = xp.asarray([10.0, 20.0, 30.0], dtype=dtype)
    xq = x

    result = interp1d(xq, x, y)
    assert np.allclose(to_numpy(result), to_numpy(y), rtol=1e-6 if dtype == xp.float32 else 1e-7)


@pytest.mark.parametrize("dtype", [xp.float64, xp.float32])
def test_interp1d_2darray(dtype):
    x = xp.asarray([0.0, 1.0, 2.0], dtype=dtype)
    y = xp.asarray([[0.0, 1.0, 4.0], [0.0, 1.0, 4.0]], dtype=dtype)
    xq = xp.asarray([0.5, 1.5], dtype=dtype)

    result = interp1d(xq, x, y)
    expected = np.interp(to_numpy(xq), to_numpy(x), to_numpy(y)[0])
    assert np.allclose(
        to_numpy(result),
        np.stack((expected, expected), axis=0),
        rtol=1e-6 if dtype == xp.float32 else 1e-7,
    )


@pytest.mark.parametrize("dtype", [xp.float64, xp.float32])
def test_interp1d_multiple_y_stack_sequence(dtype):
    x = xp.asarray([0.0, 1.0, 2.0], dtype=dtype)
    y1 = xp.asarray([1.0, 2.0, 3.0], dtype=dtype)
    y2 = xp.asarray([10.0, 20.0, 30.0], dtype=dtype)
    xq = xp.asarray([0.5, 1.5], dtype=dtype)

    result = interp1d(xq, x, [y1, y2], stack=True)
    expected1 = np.interp(to_numpy(xq), to_numpy(x), to_numpy(y1))
    expected2 = np.interp(to_numpy(xq), to_numpy(x), to_numpy(y2))
    assert np.allclose(
        to_numpy(result),
        np.stack((expected1, expected2), axis=0),
        rtol=1e-6 if dtype == xp.float32 else 1e-7,
    )

    result = interp1d(xq, x, (y1, y2), stack=True)
    assert np.allclose(
        to_numpy(result),
        np.stack((expected1, expected2), axis=0),
        rtol=1e-6 if dtype == xp.float32 else 1e-7,
    )


@pytest.mark.parametrize("dtype", [xp.float64, xp.float32])
def test_interp1_multiple_y_sequence(dtype):
    x = xp.asarray([0.0, 1.0, 2.0], dtype=dtype)
    y1 = xp.asarray([1.0, 2.0, 3.0], dtype=dtype)
    y2 = xp.asarray([10.0, 20.0, 30.0], dtype=dtype)
    xq = xp.asarray([0.5, 1.5], dtype=dtype)

    result = interp1d(xq, x, [y1, y2], stack=False)
    assert isinstance(result, list)
    result1 = result[0]
    result2 = result[1]
    expected1 = np.interp(to_numpy(xq), to_numpy(x), to_numpy(y1))
    expected2 = np.interp(to_numpy(xq), to_numpy(x), to_numpy(y2))
    assert np.allclose(to_numpy(result1), expected1, rtol=1e-6 if dtype == xp.float32 else 1e-7)
    assert np.allclose(to_numpy(result2), expected2, rtol=1e-6 if dtype == xp.float32 else 1e-7)

    result = interp1d(xq, x, [y1, y2])
    assert isinstance(result, list)
    result1 = result[0]
    result2 = result[1]
    assert np.allclose(to_numpy(result1), expected1, rtol=1e-6 if dtype == xp.float32 else 1e-7)
    assert np.allclose(to_numpy(result2), expected2, rtol=1e-6 if dtype == xp.float32 else 1e-7)


@pytest.mark.parametrize("dtype", [xp.float64, xp.float32])
def test_interp1d_mixed_1d_2d_sequence(dtype):
    x = xp.asarray([0.0, 1.0, 2.0], dtype=dtype)
    y1 = xp.asarray([1.0, 2.0, 3.0], dtype=dtype)  # 1D
    y2 = xp.asarray([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], dtype=dtype)  # 2D
    xq = xp.asarray([0.5, 1.5], dtype=dtype)

    # stack=False: should return a list, with shapes matching y1 and y2
    result = interp1d(xq, x, [y1, y2], stack=False)
    assert isinstance(result, list)
    assert to_numpy(result[0]).shape == (2,)  # y1: 1D, so shape (2,)
    assert to_numpy(result[1]).shape == (2, 2)  # y2: 2D, so shape (2, 2)
    expected1 = np.interp(to_numpy(xq), to_numpy(x), to_numpy(y1))
    expected2 = np.stack(
        [
            np.interp(to_numpy(xq), to_numpy(x), to_numpy(y2)[0]),
            np.interp(to_numpy(xq), to_numpy(x), to_numpy(y2)[1]),
        ],
        axis=0,
    )
    assert np.allclose(to_numpy(result[0]), expected1, rtol=1e-6 if dtype == xp.float32 else 1e-7)
    assert np.allclose(to_numpy(result[1]), expected2, rtol=1e-6 if dtype == xp.float32 else 1e-7)

    with pytest.raises(ValueError):
        # stack=True: should raise ValueError due to shape mismatch
        result = interp1d(xq, x, [y1, y2], stack=True)


@pytest.mark.parametrize("dtype", [xp.float64, xp.float32])
def test_interp1d_dict_input(dtype):
    x = xp.asarray([0.0, 1.0, 2.0], dtype=dtype)
    y_dict = {
        "a": xp.asarray([1.0, 2.0, 3.0], dtype=dtype),
        "b": xp.asarray([10.0, 20.0, 30.0], dtype=dtype),
    }
    xq = xp.asarray([0.5, 1.5], dtype=dtype)

    result = interp1d(xq, x, y_dict)
    assert isinstance(result, dict)
    assert set(result.keys()) == set(y_dict.keys())
    expected_a = np.interp(to_numpy(xq), to_numpy(x), to_numpy(y_dict["a"]))
    expected_b = np.interp(to_numpy(xq), to_numpy(x), to_numpy(y_dict["b"]))
    assert np.allclose(
        to_numpy(result["a"]), expected_a, rtol=1e-6 if dtype == xp.float32 else 1e-7
    )
    assert np.allclose(
        to_numpy(result["b"]), expected_b, rtol=1e-6 if dtype == xp.float32 else 1e-7
    )

    result = interp1d(xq, x, y_dict, stack=True)
    assert result.shape == (len(y_dict), array_api_compat.size(xq))
    assert np.allclose(to_numpy(result)[0], expected_a, rtol=1e-6 if dtype == xp.float32 else 1e-7)
    assert np.allclose(to_numpy(result)[1], expected_b, rtol=1e-6 if dtype == xp.float32 else 1e-7)


@pytest.mark.parametrize("dtype", [xp.float64, xp.float32])
def test_interp1d_out_of_bounds(dtype):
    x = xp.asarray([0.0, 1.0, 2.0], dtype=dtype)
    y = xp.asarray([0.0, 1.0, 4.0], dtype=dtype)
    xq = xp.asarray([-1.0, 0.0, 2.0, 3.0], dtype=dtype)

    result = interp1d(xq, x, y)
    expected = np.interp(to_numpy(xq), to_numpy(x), to_numpy(y))
    assert np.allclose(to_numpy(result), expected, rtol=1e-6 if dtype == xp.float32 else 1e-7)
