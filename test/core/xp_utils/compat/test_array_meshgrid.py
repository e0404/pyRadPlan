import pytest
import numpy as np
from pyRadPlan.core.xp_utils.compat import array_meshgrid

try:
    import cupy as cp
except ImportError:
    cp = None

try:
    import torch
except ImportError:
    torch = None


def test_array_meshgrid_numpy():
    x = np.array([1, 2, 3])
    y = np.array([4, 5])

    # Test xy indexing (default)
    xx, yy = array_meshgrid(x, y, indexing="xy")
    expected_xx_xy = np.array([[1, 2, 3], [1, 2, 3]])
    expected_yy_xy = np.array([[4, 4, 4], [5, 5, 5]])

    np.testing.assert_array_equal(xx, expected_xx_xy)
    np.testing.assert_array_equal(yy, expected_yy_xy)

    # Test ij indexing
    xx, yy = array_meshgrid(x, y, indexing="ij")
    expected_xx_ij = np.array([[1, 1], [2, 2], [3, 3]])
    expected_yy_ij = np.array([[4, 5], [4, 5], [4, 5]])

    np.testing.assert_array_equal(xx, expected_xx_ij)
    np.testing.assert_array_equal(yy, expected_yy_ij)


def test_array_meshgrid_torch():
    if torch is None:
        pytest.skip("Torch not installed")

    x = torch.tensor([1, 2, 3])
    y = torch.tensor([4, 5])

    # Test xy indexing
    xx, yy = array_meshgrid(x, y, indexing="xy")
    expected_xx_xy = torch.tensor([[1, 2, 3], [1, 2, 3]])
    expected_yy_xy = torch.tensor([[4, 4, 4], [5, 5, 5]])

    assert torch.equal(xx, expected_xx_xy)
    assert torch.equal(yy, expected_yy_xy)

    # Test ij indexing
    xx, yy = array_meshgrid(x, y, indexing="ij")
    expected_xx_ij = torch.tensor([[1, 1], [2, 2], [3, 3]])
    expected_yy_ij = torch.tensor([[4, 5], [4, 5], [4, 5]])

    assert torch.equal(xx, expected_xx_ij)
    assert torch.equal(yy, expected_yy_ij)


def test_array_meshgrid_cupy():
    if cp is None:
        pytest.skip("Cupy not installed")

    x = cp.array([1, 2, 3])
    y = cp.array([4, 5])

    # Test xy indexing
    xx, yy = array_meshgrid(x, y, indexing="xy")
    expected_xx_xy = cp.array([[1, 2, 3], [1, 2, 3]])
    expected_yy_xy = cp.array([[4, 4, 4], [5, 5, 5]])

    cp.testing.assert_array_equal(xx, expected_xx_xy)
    cp.testing.assert_array_equal(yy, expected_yy_xy)

    # Test ij indexing
    xx, yy = array_meshgrid(x, y, indexing="ij")
    expected_xx_ij = cp.array([[1, 1], [2, 2], [3, 3]])
    expected_yy_ij = cp.array([[4, 5], [4, 5], [4, 5]])

    cp.testing.assert_array_equal(xx, expected_xx_ij)
    cp.testing.assert_array_equal(yy, expected_yy_ij)


def test_array_meshgrid_array_api_strict():
    try:
        import array_api_strict as nps
    except Exception:
        pytest.skip("array_api_strict not installed")

    from pyRadPlan.core.xp_utils.helpers import to_numpy

    x = nps.asarray([1, 2, 3])
    y = nps.asarray([4, 5])

    # Test xy indexing
    xx, yy = array_meshgrid(x, y, indexing="xy")
    expected_xx_xy = nps.asarray([[1, 2, 3], [1, 2, 3]])
    expected_yy_xy = nps.asarray([[4, 4, 4], [5, 5, 5]])

    np.testing.assert_array_equal(xx, expected_xx_xy)
    np.testing.assert_array_equal(yy, expected_yy_xy)

    # Test ij indexing
    xx, yy = array_meshgrid(x, y, indexing="ij")
    expected_xx_ij = nps.asarray([[1, 1], [2, 2], [3, 3]])
    expected_yy_ij = nps.asarray([[4, 5], [4, 5], [4, 5]])

    np.testing.assert_array_equal(xx, expected_xx_ij)
    np.testing.assert_array_equal(yy, expected_yy_ij)
