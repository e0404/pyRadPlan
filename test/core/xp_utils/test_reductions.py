"""Grouped maximum semantics for native reductions and Array API fallback."""

import array_api_compat
import numpy as np
import pytest

from pyRadPlan.core.xp_utils import cupy_available, grouped_max, to_numpy


@pytest.fixture(params=["numpy", "cupy", "array_api_strict", "torch", "jax.numpy"])
def xp(request):
    """Use the available backend on its default device."""
    if request.param == "cupy" and not cupy_available():
        pytest.skip("CuPy device unavailable")
    module = pytest.importorskip(request.param)
    return array_api_compat.array_namespace(module.asarray([0.0]))


@pytest.mark.parametrize(
    "dtype_name", ["float16", "float32", "float64", "int16", "int32", "int64"]
)
def test_grouped_max(xp, dtype_name):
    """Unsorted, repeated groups, negatives and empty groups retain their maxima."""
    if not hasattr(xp, dtype_name):
        pytest.skip("Dtype is not supported by this namespace")
    values = xp.asarray([-7, -2, 3, -1, 5, 5], dtype=getattr(xp, dtype_name))
    groups = xp.asarray([3, 0, 3, 0, 2, 2], dtype=xp.int32)
    result = grouped_max(values, groups, 6, initial=-10)
    np.testing.assert_array_equal(to_numpy(result), [-1, -10, 5, 3, -10, -10])
    assert result.dtype == values.dtype
    assert array_api_compat.device(result) == array_api_compat.device(values)
    np.testing.assert_array_equal(to_numpy(values), [-7, -2, 3, -1, 5, 5])


def test_nan_and_initial(xp):
    """NaNs propagate within groups, and initial participates in the maximum."""
    values = xp.asarray([float("nan"), 2.0, -3.0, float("inf")])
    groups = xp.asarray([1, 1, 0, 2], dtype=xp.int32)
    np.testing.assert_array_equal(
        to_numpy(grouped_max(values, groups, 4, initial=0)), [0, np.nan, np.inf, 0]
    )


def test_empty(xp):
    """An empty input works both with and without output groups."""
    values = xp.asarray([], dtype=xp.float32)
    groups = xp.asarray([], dtype=xp.int32)
    assert grouped_max(values, groups, 0, initial=0).shape == (0,)
    np.testing.assert_array_equal(to_numpy(grouped_max(values, groups, 3, initial=-1)), [-1] * 3)


def test_invalid_groups(xp):
    """Reject invalid shapes, dtypes and indices consistently before native dispatch."""
    values = xp.asarray([1.0, 2.0])
    for groups in ([-1, 0], [0, 2]):
        with pytest.raises(ValueError, match="Group indices"):
            grouped_max(values, xp.asarray(groups, dtype=xp.int32), 2, initial=0)
    with pytest.raises(ValueError, match="equal shapes"):
        grouped_max(values, xp.asarray([0]), 2, initial=0)
    with pytest.raises(TypeError, match="integer dtype"):
        grouped_max(values, xp.asarray([0.0, 1.0]), 2, initial=0)


def test_nan_initial(xp):
    """A NaN initial value propagates to empty and nonempty groups alike."""
    result = grouped_max(
        xp.asarray([2.0]), xp.asarray([0], dtype=xp.int32), 2, initial=float("nan")
    )
    np.testing.assert_array_equal(to_numpy(result), [np.nan, np.nan])


def test_invalid_values_and_count(xp):
    """Reject nonnumeric values and invalid output lengths before dispatch."""
    groups = xp.asarray([0], dtype=xp.int32)
    with pytest.raises(TypeError, match="real numeric"):
        grouped_max(xp.asarray([True]), groups, 1, initial=0)
    with pytest.raises(ValueError, match="nonnegative"):
        grouped_max(xp.asarray([1.0]), groups, -1, initial=0)


def test_mixed_devices():
    """A mismatched device fails before reading the group indices."""
    torch = pytest.importorskip("torch")
    with pytest.raises(ValueError, match="share a device"):
        grouped_max(torch.ones(2), torch.empty(2, device="meta", dtype=torch.int64), 2, initial=0)
