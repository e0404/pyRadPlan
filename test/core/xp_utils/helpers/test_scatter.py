"""Tests for the backend-agnostic scatter helper."""

import numpy as np
import pytest
import array_api_strict

from pyRadPlan.core.xp_utils import scatter, to_numpy

try:
    import jax.numpy as jnp

    has_jax = True
except ImportError:
    jnp = None
    has_jax = False


def test_scatter_numpy_in_place():
    arr = np.zeros(5)
    out = scatter(arr, np.array([1, 3]), np.array([1.0, 2.0]))
    assert out is arr
    np.testing.assert_array_equal(out, [0.0, 1.0, 0.0, 2.0, 0.0])


def test_scatter_integer_indices_strict_out_of_place():
    # array-api-strict forbids integer-array __setitem__; scatter must fall back
    arr = array_api_strict.zeros(5)
    out = scatter(arr, array_api_strict.asarray([1, 3]), array_api_strict.asarray([1.0, 2.0]))
    np.testing.assert_array_equal(to_numpy(out), [0.0, 1.0, 0.0, 2.0, 0.0])


def test_scatter_boolean_mask_strict():
    arr = array_api_strict.zeros(4)
    mask = array_api_strict.asarray([True, False, True, False])
    out = scatter(arr, mask, array_api_strict.asarray([1.0, 2.0]))
    np.testing.assert_array_equal(to_numpy(out), [1.0, 0.0, 2.0, 0.0])


@pytest.mark.skipif(not has_jax, reason="jax not installed")
def test_scatter_jax_out_of_place():
    arr = jnp.zeros(5)
    out = scatter(arr, jnp.array([0, 4]), jnp.array([1.0, 2.0]))
    assert out is not arr
    np.testing.assert_array_equal(to_numpy(out), [1.0, 0.0, 0.0, 0.0, 2.0])
    np.testing.assert_array_equal(to_numpy(arr), np.zeros(5))
