from typing import Any, Union
from typing_extensions import Self
from pydantic import (
    Field,
    field_validator,
    computed_field,
)
from numpydantic import NDArray, Shape
import numpy as np
import SimpleITK as sitk

from .datamodel import PyRadPlanBaseModel


class Grid(PyRadPlanBaseModel):
    """
    Class representing image grids in the LPS world system.

    Attributes
    ----------
    resolution : dict[str, float]
        The resolution of the grid in the x, y, and z directions.
    dimensions : tuple[int, int, int]
        The dimensions of the grid in the x, y, and z directions.
    origin : np.ndarray
        The origin of the grid in the LPS world system.
    direction : np.ndarray
        The direction cosines of the grid in the LPS world system.
    """

    resolution: dict[str, float]
    dimensions: tuple[int, int, int]
    origin: NDArray[Shape["3"], np.float64] = Field(
        default=np.array([0.0, 0.0, 0.0], dtype=np.float64), alias="cubeCoordOffset"
    )
    direction: NDArray[Shape["3, 3"], np.float64] = Field(default=np.eye(3, dtype=np.float64))

    @computed_field(alias="numOfVoxels")
    @property
    def num_voxels(self) -> int:
        """Number of voxels in the grid."""
        return int(np.prod(self.dimensions))

    @computed_field
    @property
    def x(self) -> float:
        """Return the x coordinates in the LPS world system."""
        x = np.arange(self.dimensions[0]) * self.resolution["x"] + self.origin[0]
        return x

    @computed_field
    @property
    def y(self) -> float:
        """Return the y coordinates in the LPS world system."""
        y = np.arange(self.dimensions[1]) * self.resolution["y"] + self.origin[1]
        return y

    @computed_field
    @property
    def z(self) -> float:
        """Return the z coordinates in the LPS world system."""
        z = np.arange(self.dimensions[2]) * self.resolution["z"] + self.origin[2]
        return z

    @property
    def resolution_vector(self) -> np.ndarray:
        """Return the resolution as a vector."""
        return np.array([self.resolution["x"], self.resolution["y"], self.resolution["z"]])

    @field_validator("resolution", mode="after")
    @classmethod
    def _check_resolution(cls, value: dict[str, float]) -> dict[str, float]:
        """Check if resolution has the correct structure and values."""
        # Check if resolution has the correct structure (dict with keys 'x', 'y', 'z')
        if not all(key in value for key in ["x", "y", "z"]):
            raise ValueError("resolution must have keys 'x', 'y', 'z'")

        # Check if all resolution values are positive floats. If not, try to cast them to float
        for _, v in value.items():
            if v <= 0:
                raise ValueError(f"resolution values must be positive floats, got {v}")

        return value

    @field_validator("dimensions")
    @classmethod
    def _check_dimensions(cls, value: tuple[int, int, int]) -> tuple[int, int, int]:
        """Check if dimensions has the correct structure and values."""
        # Check if dimensions has exactly 3 elements
        if len(value) != 3:
            raise ValueError("dimensions must have exactly 3 elements")

        # Check if all dimensions values are positive integers. If not, try to cast them to int
        for dim in value:
            try:
                tmpdim = int(dim)
            except ValueError:
                raise ValueError(f"dimension value could not be casted into int, got {dim}")

            if tmpdim <= 0:
                raise ValueError(f"dimension values must be positive integers, got {tmpdim}")

        return value

    @field_validator("origin", mode="before")
    @classmethod
    def _check_origin(cls, value: Any) -> Any:
        """Check if origin has the correct shape (3,) and values."""
        # Check if origin has the correct shape (3,)
        try:
            value = np.asarray(value, dtype=np.float64).reshape((3,))
        except ValueError as exc:
            raise ValueError("origin must be convertible to a 1D numpy array of length 3") from exc
        return value

    @field_validator("direction", mode="before")
    @classmethod
    def _check_direction(cls, value: Any) -> Any:
        """Check if direction has the correct shape (3x3)."""

        try:
            value = np.asarray(value, dtype=np.float64).reshape((3, 3))
        except ValueError as exc:
            raise ValueError("direction must be convertible to a 3x3 numpy matrix") from exc
        return value

    # TODO: validate for additional fields if, e.g., loaded from matRad

    def to_matrad(self, context: str = "mat-file") -> Any:
        grid_dict4matrad = super().to_matrad(context=context)
        grid_dict4matrad["dimensions"] = tuple(map(float, grid_dict4matrad["dimensions"]))
        grid_dict4matrad["numOfVoxels"] = float(grid_dict4matrad["numOfVoxels"])
        return grid_dict4matrad

    @classmethod
    def from_sitk_image(cls, sitk_image: sitk.Image) -> Self:
        """
        Create a Grid object from a SimpleITK image.

        Parameters
        ----------
        sitk_image : sitk.Image
            The SimpleITK image to create the Grid object from.

        Returns
        -------
        Grid
            The Grid object created from the SimpleITK image.
        """
        keys = ["x", "y", "z"]
        resolution = dict(zip(keys, sitk_image.GetSpacing()))
        dimensions = sitk_image.GetSize()
        origin = sitk_image.GetOrigin()
        direction = sitk_image.GetDirection()
        return cls(
            resolution=resolution, dimensions=dimensions, origin=origin, direction=direction
        )

    def resample(
        self,
        target_resolution: Union[
            dict[str, float], np.ndarray, tuple[float, float, float], list[float]
        ],
    ) -> Self:
        """
        Create a resampled grid covering the original grid in new resolution.

        Parameters
        ----------
        target_resolution : Union[dict[str,float], np.ndarray]
            The target resolution of the resampled grid.

        Returns
        -------
        Grid
            The resampled grid object.
        """

        if isinstance(target_resolution, dict):
            target_resolution = np.array(
                [target_resolution["x"], target_resolution["y"], target_resolution["z"]]
            )
        else:
            try:
                target_resolution = np.asarray(target_resolution).reshape(
                    3,
                )
            except ValueError as exc:
                raise ValueError(
                    "target_resolution must be convertible to an ndarray of shape (3,)"
                ) from exc

        # Calculate the new dimensions
        new_dimensions = np.ceil(
            np.array(self.dimensions) * self.resolution_vector / target_resolution
        ).astype(int)

        # Now calculate the width of the grid in the new resolution
        old_grid_size = np.array(self.dimensions) * self.resolution_vector
        new_grid_size = new_dimensions * target_resolution
        diff_size = new_grid_size - old_grid_size

        # find the origin of the new image such that it is centered above the old image. The origin
        # is in the center of the first voxel
        origin_shift = diff_size / 2.0

        # now we need to respect the origin and direction to correctly shift the new grid into
        # position
        origin_shift = np.matmul(self.direction, origin_shift)
        new_origin = self.origin - origin_shift

        return Grid(
            resolution=dict(zip(["x", "y", "z"], target_resolution)),
            dimensions=tuple(new_dimensions),
            origin=new_origin,
            direction=self.direction,
        )


if __name__ == "__main__":
    # Example usage
    try:
        grid_dict = {
            "resolution": {"x": 1, "y": 1, "z": 1},
            "dimensions": (10, 10, 10),
        }

        grid_information = Grid.model_validate(grid_dict)
        print(grid_information.to_matrad())
    except Exception as e:
        print("Error creating grid from dose_information:", e)
