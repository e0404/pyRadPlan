import pytest
from pyRadPlan.core.xp_utils.compat import quantile
from pyRadPlan.core.xp_utils import to_numpy, from_numpy

# Use array_api_strict to ensure Array API compatibility
import timeit
import array_api_strict as xp
import numpy as np

# Relative import from the same package


def _assert_close(actual, expected, atol=1e-6):
    assert xp.allclose(actual, expected, atol=atol)


def test_quantile_nearest_1d():
    x = xp.asarray([1, 2, 3, 4, 5], dtype=xp.float32)

    # From top of distribution: p=0 -> max, p=1 -> min
    np.all(to_numpy(quantile(x, 0.0, method="nearest")) == np.asarray(1.0, dtype=np.float32))
    np.all(to_numpy(quantile(x, 0.5, method="nearest")) == np.asarray(3.0, dtype=np.float32))
    np.all(to_numpy(quantile(x, 1.0, method="nearest")) == np.asarray(5.0, dtype=np.float32))
    np.all(to_numpy(quantile(x, 0.2, method="nearest")) == np.asarray(4.0, dtype=np.float32))


def test_quantile_nearest_2d_axis():
    x = xp.asarray([[1, 2, 3, 4, 5], [10, 20, 30, 40, 50]], dtype=xp.float32)

    # Extreme points
    y0 = to_numpy(quantile(x, 0.0, axis=1, method="nearest"))
    y1 = to_numpy(quantile(x, 1.0, axis=1, method="nearest"))
    np.all(y0 == np.asarray([1.0, 10.0], dtype=np.float32))
    np.all(y1 == np.asarray([5.0, 50.0], dtype=np.float32))

    # median
    y_med = to_numpy(quantile(x, 0.5, axis=1, method="nearest"))
    np.all(y_med == np.asarray([3.0, 30.0], dtype=np.float32))

    # Other points
    y_other = to_numpy(quantile(x, 0.2, axis=1, method="nearest"))
    np.all(y_other == np.asarray([4.0, 40.0], dtype=np.float32))


def test_quantile_linear_1d_interpolation():
    x = from_numpy(xp, np.random.standard_normal((10,)).astype(np.float32))

    y = quantile(x, 0.3, method="linear")
    ynp = np.quantile(to_numpy(x), 0.3, method="linear")
    assert np.isclose(to_numpy(y), ynp, atol=1e-6)

    xnp = to_numpy(x)
    timeit.timeit(lambda: np.quantile(xnp, 0.3, method="linear"), number=10000)


def test_quantile_linear_2d_axis():
    x = from_numpy(xp, np.random.standard_normal((5, 10)).astype(np.float32))

    y = to_numpy(quantile(x, 0.3, axis=1, method="linear"))
    ynp = np.quantile(to_numpy(x), 0.3, axis=1, method="linear")
    assert np.allclose(y, ynp, atol=1e-6)


def test_quantile_is_sorted_flag_equivalence():
    x_unsorted = xp.asarray([3, 1, 5, 2, 4], dtype=xp.float32)
    x_sorted = xp.asarray([1, 2, 3, 4, 5], dtype=xp.float32)

    # Nearest
    y_unsorted = quantile(x_unsorted, 0.6, method="nearest", is_sorted=False)
    y_sorted = quantile(x_sorted, 0.6, method="nearest", is_sorted=True)
    assert xp.all(y_unsorted == y_sorted)

    # Linear
    y_unsorted_lin = quantile(x_unsorted, 0.25, method="linear", is_sorted=False)
    y_sorted_lin = quantile(x_sorted, 0.25, method="linear", is_sorted=True)
    assert np.allclose(to_numpy(y_unsorted_lin), to_numpy(y_sorted_lin), atol=1e-6)
