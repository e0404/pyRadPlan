import pytest
from pyRadPlan.core import Grid
from pydantic import ValidationError
import numpy as np
import SimpleITK as sitk


def test_grid_invalid_resolution():
    with pytest.raises(ValueError):
        grid = Grid(resolution={"x": -1, "y": 1, "z": 1})


def test_grid_invalid_dimensions():
    with pytest.raises(ValueError):
        grid = Grid(dimensions=(0, 10, 10))


def test_grid_resolution_keys():
    grid = Grid(resolution={"x": 1, "y": 1, "z": 1}, dimensions=(10, 10, 10))
    assert grid.resolution.keys() == {"x", "y", "z"}


def test_grid_resolution_values():
    grid = Grid(resolution={"x": 1, "y": 1, "z": 1}, dimensions=(10, 10, 10))
    assert all(isinstance(value, float) for value in grid.resolution.values())


def test_grid_resolution_positive():
    grid = Grid(resolution={"x": 1, "y": 1, "z": 1}, dimensions=(10, 10, 10))
    assert all(value > 0 for value in grid.resolution.values())


def test_grid_resolution_vector():
    grid = Grid(resolution={"x": 1, "y": 1, "z": 1}, dimensions=(10, 10, 10))
    assert isinstance(grid.resolution_vector, np.ndarray)
    assert grid.resolution_vector.shape == (3,)
    assert all(isinstance(value, float) for value in grid.resolution_vector)
    assert np.isclose(
        grid.resolution_vector, np.array([v for v in grid.resolution.values()])
    ).all()


def test_grid_invalid_resolution_keys():
    with pytest.raises(ValidationError):
        grid = Grid(dimensions=(10, 10, 10), resolution={"a": 1, "b": 1, "c": 1})


def test_grid_invalid_resolution_negative():
    with pytest.raises(ValidationError):
        grid = Grid(dimensions=(10, 10, 10), resolution={"x": -1, "y": 1, "z": 1})


def test_grid_dimensions():
    grid = Grid(dimensions=(10, 10, 10), resolution={"x": 1, "y": 1, "z": 1})
    assert isinstance(grid.dimensions, tuple)
    assert len(grid.dimensions) == 3
    assert all(isinstance(value, int) for value in grid.dimensions)


def test_grid_num_voxels():
    grid = Grid(dimensions=(10, 10, 10), resolution={"x": 1, "y": 1, "z": 1})
    assert isinstance(grid.num_voxels, int)
    assert grid.num_voxels == 1000


def test_grid_to_matrad():
    grid = Grid(resolution={"x": 1, "y": 1, "z": 1}, dimensions=(10, 10, 10))
    matrad_dict = grid.to_matrad()
    assert isinstance(matrad_dict, dict)
    assert "dimensions" in matrad_dict
    assert "numOfVoxels" in matrad_dict
    assert isinstance(matrad_dict["dimensions"], tuple)
    assert isinstance(matrad_dict["numOfVoxels"], float)
    assert matrad_dict["numOfVoxels"] == 1000.0


def test_grid_from_sitk_image():
    # Create a SimpleITK image
    image = sitk.Image(10, 10, 10, sitk.sitkFloat32)
    image.SetSpacing((1.0, 1.0, 1.0))
    image.SetOrigin((0.0, 0.0, 0.0))
    image.SetDirection(np.eye(3).flatten())

    # Create a Grid object from the SimpleITK image
    grid = Grid.from_sitk_image(image)

    # Check the Grid object properties
    assert grid.resolution == {"x": 1.0, "y": 1.0, "z": 1.0}
    assert grid.dimensions == (10, 10, 10)
    assert all(grid.origin == (0.0, 0.0, 0.0))
    assert np.array_equal(grid.direction, np.eye(3))


@pytest.fixture
def sample_grid():
    """Create a sample Grid object for testing."""
    resolution = {"x": 1.0, "y": 1.0, "z": 2.0}
    dimensions = (50, 100, 100)
    origin = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    direction = np.eye(3, dtype=np.float64)
    return Grid(resolution=resolution, dimensions=dimensions, origin=origin, direction=direction)


def test_resample_grid_with_dict(sample_grid):
    target_resolution = {"x": 2.0, "y": 2.0, "z": 2.0}
    resampled_grid = sample_grid.resample(target_resolution)
    assert resampled_grid.resolution == target_resolution
    assert resampled_grid.dimensions == (25, 50, 100)
    assert np.allclose(resampled_grid.origin, [0.5, 0.5, 0.0])


def test_resample_grid_with_ndarray(sample_grid):
    target_resolution = np.array([2.0, 2.0, 2.0])
    resampled_grid = sample_grid.resample(target_resolution)
    assert resampled_grid.resolution == {"x": 2.0, "y": 2.0, "z": 2.0}
    assert resampled_grid.dimensions == (25, 50, 100)
    assert np.allclose(resampled_grid.origin, [0.5, 0.5, 0.0])


def test_resample_grid_irregular_resolution(sample_grid):
    target_resolution = {"x": 2.0, "y": 5.0, "z": 3.0}
    resampled_grid = sample_grid.resample(target_resolution)
    assert resampled_grid.resolution == target_resolution
    assert resampled_grid.dimensions == (25, 20, 67)
    assert np.allclose(resampled_grid.origin, [0.5, 2.0, 0.0])


def test_resample_grid_with_shifted_rotated_grid(sample_grid):
    sample_grid.origin = np.array([10.0, 10.0, 10.0], dtype=np.float64)
    angle = np.pi / 4  # 45 degrees
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    sample_grid.direction = np.array(
        [[cos_angle, -sin_angle, 0], [sin_angle, cos_angle, 0], [0, 0, 1]], dtype=np.float64
    )
    target_resolution = {"x": 2.0, "y": 2.0, "z": 2.0}
    resampled_grid = sample_grid.resample(target_resolution)
    assert resampled_grid.resolution == target_resolution
    assert resampled_grid.dimensions == (25, 50, 100)
    assert np.allclose(resampled_grid.origin, np.array([10.0, 10.0 + 0.5 * np.sqrt(2.0), 10.0]))


def _grid_center(grid):
    extent = (np.array(grid.dimensions) - 1) * grid.resolution_vector
    return grid.origin + grid.direction @ (extent / 2.0)


@pytest.mark.parametrize(
    "target_resolution",
    [
        {"x": 2.0, "y": 2.0, "z": 2.0},
        {"x": 2.0, "y": 5.0, "z": 3.0},
        {"x": 0.5, "y": 0.7, "z": 1.0},
        {"x": 3.3, "y": 0.4, "z": 7.0},
    ],
)
def test_resample_grid_preserves_center(target_resolution):
    angle = np.pi / 5
    direction = np.array(
        [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
    )
    grid = Grid(
        resolution={"x": 1.0, "y": 1.5, "z": 2.5},
        dimensions=(31, 40, 17),
        origin=np.array([-12.0, 5.0, 30.0]),
        direction=direction,
    )
    resampled_grid = grid.resample(target_resolution)
    assert np.allclose(_grid_center(resampled_grid), _grid_center(grid))
    assert np.allclose(resampled_grid.direction, grid.direction)


def test_resample_grid_invalid_resolution(sample_grid):
    with pytest.raises(ValueError):
        sample_grid.resample([2.0, 2.0])  # Invalid shape

    with pytest.raises(ValueError):
        sample_grid.resample("invalid")  # Invalid type


# test the custom implementation of the eq/neq operator on grids
def test_grids_equal():
    grid = Grid(resolution={"x": 1, "y": 1, "z": 1}, dimensions=(10, 10, 10))
    grid2 = Grid(resolution={"x": 1, "y": 1, "z": 1}, dimensions=(10, 10, 10))

    assert grid == grid2


def test_grids_not_equal():
    grid = Grid(resolution={"x": 1, "y": 1, "z": 1}, dimensions=(10, 10, 10))
    grid2 = Grid(resolution={"x": 1, "y": 1, "z": 2}, dimensions=(10, 10, 10))
    grid3 = Grid(resolution={"x": 1, "y": 1, "z": 1}, dimensions=(10, 10, 20))

    assert grid != grid2
    assert grid != grid3


def test_grids_not_equal_types():
    grid = Grid(resolution={"x": 1, "y": 1, "z": 1}, dimensions=(10, 10, 10))
    assert grid != 1
    assert grid != "string"
    assert grid != [1, 2, 3]

    # revert
    assert 1 != grid
    assert "string" != grid
    assert [1, 2, 3] != grid
