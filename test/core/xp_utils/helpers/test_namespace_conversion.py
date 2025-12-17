import pytest

import array_api_compat
import array_api_strict
from array_api_compat import numpy as np
from pyRadPlan.core.xp_utils import to_namespace


@pytest.fixture
def numpy_array():
    return np.array([[1, 2, 3], [4, 5, 6]])


@pytest.fixture
def array_api_array(numpy_array):
    return array_api_strict.asarray(numpy_array)


def test_to_namespace_numpy_to_array_api(numpy_array):
    result = to_namespace(array_api_strict, numpy_array)
    assert array_api_compat.is_array_api_strict_namespace(result.__array_namespace__())
    assert np.array_equal(result, numpy_array)


def test_to_namespace_array_api_to_numpy(array_api_array):
    result = to_namespace(np, array_api_array)
    assert array_api_compat.is_numpy_array(result)
    assert np.array_equal(result, array_api_array)


def test_to_namespace_no_conversion(array_api_array):
    result = to_namespace(array_api_strict, array_api_array)
    assert result is array_api_array  # No conversion should occur
