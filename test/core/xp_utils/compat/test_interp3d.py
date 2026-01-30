import pytest
import numpy as np
from pyRadPlan.core.xp_utils.compat import interp3d

try:
    import cupy as cp
except ImportError:
    cp = None

try:
    import torch
except ImportError:
    torch = None


def test_interp3d_numpy():
    # Define grid
    x = np.linspace(0, 1, 2)  # 0, 1
    y = np.linspace(0, 1, 2)  # 0, 1
    z = np.linspace(0, 1, 2)  # 0, 1
    points = (x, y, z)

    # Define values: f(x,y,z) = x + 2*y + 3*z
    # Shape (2, 2, 2)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    values = X + 2 * Y + 3 * Z

    # Create interpolator
    interp = interp3d(points, values, method="linear")

    # Query points
    # Midpoint: (0.5, 0.5, 0.5) -> 0.5 + 1 + 1.5 = 3.0
    query = np.array([[0.5, 0.5, 0.5], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    result = interp(query)

    expected = np.array([3.0, 0.0, 6.0])
    np.testing.assert_allclose(result, expected, atol=1e-5)

    # Test bounds error
    query_out = np.array([[1.5, 0.5, 0.5]])
    with pytest.raises(ValueError):
        interp(query_out)

    # Test fill value
    interp_fill = interp3d(points, values, method="linear", bounds_error=False, fill_value=-1.0)
    result_fill = interp_fill(query_out)
    np.testing.assert_allclose(result_fill, [-1.0], atol=1e-5)


def test_interp3d_torch():
    if torch is None:
        pytest.skip("Torch not installed")

    # Define grid
    x = torch.linspace(0, 1, 2)
    y = torch.linspace(0, 1, 2)
    z = torch.linspace(0, 1, 2)
    points = (x, y, z)

    # Define values: f(x,y,z) = x + 2*y + 3*z
    X, Y, Z = torch.meshgrid(x, y, z, indexing="ij")
    values = X + 2 * Y + 3 * Z

    # Create interpolator
    interp = interp3d(points, values, method="linear")

    # Query points
    query = torch.tensor([[0.5, 0.5, 0.5], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    result = interp(query)

    expected = torch.tensor([3.0, 0.0, 6.0])
    assert torch.allclose(result, expected, atol=1e-5)

    # Test bounds error
    query_out = torch.tensor([[1.5, 0.5, 0.5]])
    with pytest.raises(ValueError):
        interp(query_out)

    # Test fill value
    interp_fill = interp3d(points, values, method="linear", bounds_error=False, fill_value=-1.0)
    result_fill = interp_fill(query_out)
    assert torch.allclose(result_fill, torch.tensor([-1.0]), atol=1e-5)


def test_interp3d_cupy():
    if cp is None:
        pytest.skip("Cupy not installed")

    # Define grid
    x = cp.linspace(0, 1, 2)
    y = cp.linspace(0, 1, 2)
    z = cp.linspace(0, 1, 2)
    points = (x, y, z)

    # Define values
    X, Y, Z = cp.meshgrid(x, y, z, indexing="ij")
    values = X + 2 * Y + 3 * Z

    # Create interpolator
    interp = interp3d(points, values, method="linear")

    # Query points
    query = cp.array([[0.5, 0.5, 0.5], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    result = interp(query)

    expected = cp.array([3.0, 0.0, 6.0])
    cp.testing.assert_allclose(result, expected, atol=1e-5)

    # Test bounds error
    query_out = cp.array([[1.5, 0.5, 0.5]])
    # Cupy/Scipy might raise ValueError for bounds
    with pytest.raises(ValueError):
        interp(query_out)

    # Test fill value
    interp_fill = interp3d(points, values, method="linear", bounds_error=False, fill_value=-1.0)
    result_fill = interp_fill(query_out)
    cp.testing.assert_allclose(result_fill, cp.array([-1.0]), atol=1e-5)


def test_interp3d_torch_batched():
    if torch is None:
        pytest.skip("Torch not installed")

    # Define grid
    x = torch.linspace(0, 1, 2)
    y = torch.linspace(0, 1, 2)
    z = torch.linspace(0, 1, 2)
    points = (x, y, z)

    # Define values for Batch 1: f1(x,y,z) = x + 2*y + 3*z
    X, Y, Z = torch.meshgrid(x, y, z, indexing="ij")
    v1 = X + 2 * Y + 3 * Z

    # Define values for Batch 2: f2(x,y,z) = 2*x + y + z
    v2 = 2 * X + Y + Z

    # Stack to create (2, 2, 2, 2) tensor
    values = torch.stack([v1, v2], dim=0)

    # Create interpolator
    interp = interp3d(points, values, method="linear")

    # Query points
    # P1: (0.5, 0.5, 0.5)
    # P2: (0.0, 0.0, 0.0)
    # P3: (1.0, 1.0, 1.0)
    query = torch.tensor([[0.5, 0.5, 0.5], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])

    result = interp(query)

    # Expected results
    # Batch 1 -> [3.0, 0.0, 6.0]
    # Batch 2 -> [2.0, 0.0, 4.0]
    expected = torch.tensor([[3.0, 0.0, 6.0], [2.0, 0.0, 4.0]])

    assert result.shape == (2, 3)
    assert torch.allclose(result, expected, atol=1e-5)
