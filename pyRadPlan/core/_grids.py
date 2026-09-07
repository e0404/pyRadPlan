from typing import Any, Union
from typing_extensions import Self
from pydantic import (
    Field,
    field_validator,
    model_validator,
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
        The resolution of the grid in the x, y, z (and t for 4D) directions.
    dimensions : tuple[int, ...]
        The dimensions of the grid in the x, y, z (and t for 4D) directions.
        Can be 3D (x, y, z) or 4D (x, y, z, t).
    origin : np.ndarray
        The origin of the grid in the LPS world system. Shape (3,) for 3D or (4,) for 4D.
    direction : np.ndarray
        The direction cosines of the grid in the LPS world system.
        Shape (3, 3) for 3D or (4, 4) for 4D.
    """

    resolution: dict[str, float]
    dimensions: tuple
    origin: Union[NDArray[Shape["3"], np.float64], NDArray[Shape["4"], np.float64]] = Field(
        default=np.array([0.0, 0.0, 0.0], dtype=np.float64), alias="cubeCoordOffset"
    )
    direction: Union[NDArray[Shape["3,3"], np.float64], NDArray[Shape["4,4"], np.float64]] = Field(
        default=np.eye(3, dtype=np.float64)
    )

    @computed_field(alias="numOfVoxels")
    @property
    def num_voxels(self) -> int:
        """Number of voxels in the grid."""
        return int(np.prod(self.dimensions))

    @computed_field
    @property
    def x(self) -> NDArray:
        """Return the x coordinates in the LPS world system."""
        x = np.arange(self.dimensions[0]) * self.resolution["x"] + self.origin[0]
        return x

    @computed_field
    @property
    def y(self) -> NDArray:
        """Return the y coordinates in the LPS world system."""
        y = np.arange(self.dimensions[1]) * self.resolution["y"] + self.origin[1]
        return y

    @computed_field
    @property
    def z(self) -> NDArray:
        """Return the z coordinates in the LPS world system."""
        z = np.arange(self.dimensions[2]) * self.resolution["z"] + self.origin[2]
        return z

    @computed_field
    @property
    def t(self) -> np.ndarray:
        """Return the t coordinates in the LPS world system.

        Returns
        -------
        np.ndarray
            Array of t coordinates for each time point if 4D grid,
            empty array if 3D grid.
        """
        if len(self.dimensions) == 4:
            t = np.arange(self.dimensions[3]) * self.resolution["t"] + self.origin[3]
            return t
        else:
            return np.array([])  # None

    @property
    def resolution_vector(self) -> np.ndarray:
        """Return the resolution as a vector.

        Returns
        -------
        np.ndarray
            Resolution vector [x, y, z] for 3D grids or [x, y, z, t] for 4D grids.
        """
        if len(self.dimensions) == 4:
            return np.array(
                [
                    self.resolution["x"],
                    self.resolution["y"],
                    self.resolution["z"],
                    self.resolution["t"],
                ]
            )

        return np.array([self.resolution["x"], self.resolution["y"], self.resolution["z"]])

    @property
    def direction_vector(self) -> np.ndarray:
        """Return the direction matrix as a flattened vector.

        Returns
        -------
        np.ndarray
            Flattened direction matrix as a 1D array.
        """

        return self.direction.ravel()

    @model_validator(mode="after")
    def _check_dimensionality(self):
        """
        Check if the dimensionality of parameters are consistent.

        Raises
        ------
        ValueError
            If the parameter dimensions are inconsistent with grid dimensionality.
        """

        # 3D:
        if len(self.dimensions) == 3:
            if (
                len(self.resolution) != 3
                or len(self.origin) != 3
                or self.direction.shape != (3, 3)
            ):
                raise ValueError(
                    "For 3D grids, resolution, origin, and direction must have 3 elements."
                )

        # 4D:
        if len(self.dimensions) == 4:
            if (
                len(self.resolution) != 4
                or len(self.origin) != 4
                or self.direction.shape != (4, 4)
            ):
                raise ValueError(
                    "For 4D grids, resolution, origin, and direction must have 4 elements."
                )
        return self

    @field_validator("resolution", mode="after")
    @classmethod
    def _check_resolution(cls, value: dict[str, float]) -> dict[str, float]:
        """Check if resolution has the correct structure and values."""

        expected_keys_3d = {"x", "y", "z"}
        expected_keys_4d = {"x", "y", "z", "t"}

        if not (
            expected_keys_3d.issubset(value.keys()) or expected_keys_4d.issubset(value.keys())
        ):
            raise ValueError("resolution must have keys 'x', 'y', 'z' (and 't' for 4D grids).")

        # Check if all resolution values are positive floats.
        for _, v in value.items():
            if v <= 0:
                raise ValueError(f"resolution values must be positive floats, got {v}")

        return value

    @field_validator("dimensions")
    @classmethod
    def _check_dimensions(cls, value: tuple) -> tuple:
        """Check if dimensions has the correct structure and values."""
        # Check if dimensions has exactly 3 elements
        if not 3 <= len(value) < 5:
            raise ValueError("dimensions must have 3 or 4 (in case of multi-scenario) elements.")

        # Check if all dimensions values are positive integers. If not, try to cast them to int
        converted_dims = []
        for dim in value:
            try:
                tmpdim = int(dim)
            except ValueError:
                raise ValueError(f"dimension value could not be casted into int, got {dim}")

            if tmpdim <= 0:
                raise ValueError(f"dimension values must be positive integers, got {tmpdim}")

            converted_dims.append(tmpdim)

        return tuple(converted_dims)

    @field_validator("origin", mode="before")
    @classmethod
    def _check_origin(cls, value: Any) -> Any:
        """Check if origin has the correct shape (3,) or (4,) for 4D and values."""
        # Convert to numpy array first
        try:
            value = np.asarray(value, dtype=np.float64)
        except ValueError as exc:
            raise ValueError("origin must be convertible to a numpy array") from exc

        # Check if origin has the correct shape (3,) or (4,)
        if value.size == 3:
            value = value.reshape((3,))
        elif value.size == 4:
            value = value.reshape((4,))
        else:
            raise ValueError("origin must be convertible to a 1D numpy array of length 3 or 4")

        return value

    @field_validator("direction", mode="before")
    @classmethod
    def _check_direction(cls, value: Any) -> Any:
        """Check if direction has the correct shape (3x3) or (4x4) for 4D."""

        # Convert to numpy array first
        try:
            value = np.asarray(value, dtype=np.float64)
        except ValueError as exc:
            raise ValueError("direction must be convertible to a numpy matrix") from exc

        # Check if direction has the correct shape (3x3) or (4x4)
        if value.size == 9:
            value = value.reshape((3, 3))
        elif value.size == 16:
            value = value.reshape((4, 4))
        else:
            raise ValueError("direction must be convertible to a 3x3 or 4x4 numpy matrix")

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
        if sitk_image.GetDimension() == 3:
            keys = ["x", "y", "z"]
            resolution = dict(zip(keys, sitk_image.GetSpacing()))
            dimensions = sitk_image.GetSize()
            origin = np.array(sitk_image.GetOrigin(), dtype=np.float64)
            direction = np.array(sitk_image.GetDirection(), dtype=np.float64).reshape(3, 3)

        elif sitk_image.GetDimension() == 4:
            keys = ["x", "y", "z", "t"]
            resolution = dict(zip(keys, sitk_image.GetSpacing()))
            dimensions = sitk_image.GetSize()
            origin = np.array(sitk_image.GetOrigin(), dtype=np.float64)
            direction = np.array(sitk_image.GetDirection(), dtype=np.float64).reshape(4, 4)
        else:
            raise ValueError(
                f"Unsupported image dimensionality: {sitk_image.GetDimension()}. Expected 3D or 4D."
            )

        return cls(
            resolution=resolution, dimensions=dimensions, origin=origin, direction=direction
        )

    @classmethod
    def from_numpy_array(cls, array: np.ndarray, resolution: Union[dict, float] = 1.0) -> Self:
        """
        Create a Grid object from a numpy array.

        Parameters
        ----------
        array : np.ndarray
            The numpy array to create the Grid object from.
        resolution : Union[dict, float], optional
            The resolution for the grid. If float, uses same resolution for all dimensions.
            If dict, should contain keys 'x', 'y', 'z' (and 't' for 4D). Defaults to 1.0.

        Returns
        -------
        Grid
            The Grid object created from the numpy array.
        """
        # numpy arrays are in (z, y, x) or (t, z, y, x) format
        # but Grid dimensions should be in (x, y, z) or (x, y, z, t) format
        shape = array.shape

        if len(shape) == 3:
            # Convert (z,y,x) to (x,y,z)
            dimensions = (shape[2], shape[1], shape[0])
            keys = ["x", "y", "z"]
            origin = np.array([0.0, 0.0, 0.0], dtype=np.float64)
            direction = np.eye(3, dtype=np.float64)
        elif len(shape) == 4:
            # Convert (t,z,y,x) to (x,y,z,t)
            dimensions = (shape[3], shape[2], shape[1], shape[0])
            keys = ["x", "y", "z", "t"]
            origin = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
            direction = np.eye(4, dtype=np.float64)
        else:
            raise ValueError(f"Unsupported array dimensionality: {len(shape)}. Expected 3D or 4D.")

        # Handle resolution parameter
        if isinstance(resolution, dict):
            # Validate that all required keys are present
            required_keys = set(keys)
            if not required_keys.issubset(resolution.keys()):
                missing_keys = required_keys - resolution.keys()
                raise ValueError(f"Resolution dict missing keys: {missing_keys}")
            resolution_dict = {key: resolution[key] for key in keys}
        else:
            # Use same resolution for all dimensions
            resolution_dict = {key: float(resolution) for key in keys}

        return cls(
            resolution=resolution_dict, dimensions=dimensions, origin=origin, direction=direction
        )

    def resample(
        self,
        target_resolution: Union[
            dict[str, float],
            np.ndarray,
            tuple[float, float, float],
            tuple[float, float, float, float],
            list[float],
        ],
    ) -> Self:
        """
        Create a resampled grid covering the original grid in new resolution.

        Parameters
        ----------
        target_resolution : Union[dict[str, float], np.ndarray, tuple, list]
            The target resolution of the resampled grid. Can be:
            - dict with keys 'x', 'y', 'z' (and 't' for 4D)
            - numpy array of shape (3,) or (4,)
            - tuple of 3 or 4 float values
            - list of 3 or 4 float values

        Returns
        -------
        Grid
            The resampled grid object with new resolution while covering
            the same spatial extent as the original grid.
        """

        # Determine if this is a 3D or 4D grid
        is_4d = len(self.dimensions) == 4
        expected_size = 4 if is_4d else 3
        keys = ["x", "y", "z", "t"] if is_4d else ["x", "y", "z"]

        if isinstance(target_resolution, dict):
            # Extract resolution values in the correct order
            target_resolution = np.array([target_resolution[key] for key in keys])
        else:
            try:
                target_resolution = np.asarray(target_resolution).reshape(expected_size)
            except ValueError as exc:
                raise ValueError(
                    f"target_resolution must be convertible to an ndarray of shape ({expected_size},)"
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
            resolution=dict(zip(keys, target_resolution)),
            dimensions=tuple(new_dimensions),
            origin=new_origin,
            direction=self.direction,
        )
