"""Module for storing a given quantity cube."""

import os
from typing import Annotated, Any, Union
from typing_extensions import Self
from pydantic import (
    Field,
    computed_field,
    SerializationInfo,
    model_validator,
    field_serializer,
)
import SimpleITK as sitk
import numpy as np
from pyRadPlan.core import resample_image

from pyRadPlan.core import PyRadPlanBaseModel, Grid


class QuantityCube(PyRadPlanBaseModel):
    """
    A class representing the result of a dose calculation.

    This class is used to store and do further sequencing/analysis of the dose cubes.
    """

    cube: Annotated[sitk.Image, Field(init=False)]  # ,alias="cube")]

    @model_validator(mode="before")
    @classmethod
    def validate_cube(cls, data: Any) -> Any:
        # Early return for cube input (assuming cube is defined/imported elsewhere)
        if isinstance(data, QuantityCube):
            return data

        if isinstance(data, dict):
            # Try to get the cube using our internal key or its alias
            cube = data.get("cube")
            if cube is None:
                cube_fieldname = cls.model_fields.get("cube").validation_alias
                cube = data.pop(cube_fieldname, None)
            if cube is None:
                raise ValueError("Cube not present in dictionary.")

            if isinstance(cube, np.ndarray):
                # Determine if a permutation is required (e.g. matRad data has 'cubeDim')
                permute = data.get("cubeDim") is not None
                # TODO: Remove ct_scenarios!
                # If the numpy array holds multiple scenarios (dtype object), then form a 4D image
                if cube.dtype == object:
                    num_cubes = len(cube)
                    if permute:
                        cube_scenarios = [
                            sitk.GetImageFromArray(np.transpose(cube[i], (2, 0, 1)), False)
                            for i in range(num_cubes)
                        ]
                    else:
                        cube_scenarios = [
                            sitk.GetImageFromArray(cube[i], False) for i in range(num_cubes)
                        ]
                    cube = sitk.JoinSeries(cube_scenarios)
                elif permute:
                    cube = sitk.GetImageFromArray(np.transpose(cube, (2, 0, 1)), False)
                else:
                    cube = sitk.GetImageFromArray(cube, False)

            # Save the result under the key used in this model
            data["cube"] = cube

            # Validate that the cube is indeed a SimpleITK image.
            if not isinstance(data["cube"], sitk.Image):
                raise ValueError(f"Unsupported format of HU cube: {type(data['cube'])}")

            is4d = cube.GetDimension() == 4

            # Set image origin if available from the dictionary.
            if "origin" in data:
                data["cube"].SetOrigin(data["origin"])
            elif all(key in data for key in ("x", "y", "z")):
                origin = np.array([data["x"][0], data["y"][0], data["z"][0]], dtype=float)
                if is4d:
                    origin = np.append(origin, [0.0])
                data["cube"].SetOrigin(origin)

            # Set direction if provided.
            if "direction" in data:
                data["cube"].SetDirection(data["direction"])

            # Set spacing (resolution) if available.
            if "resolution" in data and all(key in data["resolution"] for key in ("x", "y", "z")):
                resolution = data["resolution"]
                if is4d:
                    data["cube"].SetSpacing(
                        [resolution["x"], resolution["y"], resolution["z"], 1.0]
                    )
                else:
                    data["cube"].SetSpacing([resolution["x"], resolution["y"], resolution["z"]])

        return data

    @field_serializer("cube")
    def custom_cube_serializer(self, v: sitk.Image, info: SerializationInfo) -> Any:
        """Serialize the cube field in various contexts, particularly for matRad format."""
        context = info.context
        if context and context.get("matRad") == "mat-file":
            # Convert the SimpleITK image to a NumPy array and flip axes for matRad format
            numpy_array = np.transpose(sitk.GetArrayFromImage(v), (1, 2, 0))

            # Create a dictionary for matRad format
            cube_dict = {
                "cube": np.ndarray(shape=(1,), dtype=object),
                "cubeDim": np.array(v.GetSize(), dtype=float),
            }
            cube_dict["cube"][0] = numpy_array
            return cube_dict

        # Default serialization (e.g., return the SimpleITK image as-is)
        return v

    @computed_field
    @property
    def resolution(self) -> dict:
        """
        Get the resolution of the image.

        Returns
        -------
        dict
            A dictionary containing the resolution in x, y, and z directions.
        """
        spacing = self.cube.GetSpacing()
        return {"x": spacing[0], "y": spacing[1], "z": spacing[2]}

    @computed_field
    @property
    def size(self) -> tuple:
        """
        Get the size of the image.

        Returns
        -------
        tuple
            A tuple containing the size in x, y, and z directions.
        """
        return self.cube.GetSize()[0:3]

    @computed_field
    @property
    def cube_dim(self) -> tuple:
        """
        Get the cube dimensions in x, y, and z.

        Returns
        -------
        tuple
            A tuple containing the cube dimensions.
        """
        return self.size[0:3]

    @computed_field
    @property
    def origin(self) -> tuple:
        """
        Get the origin of the image.

        Returns
        -------
        tuple
            A tuple containing the origin coordinates.
        """
        orig = self.cube.GetOrigin()
        return orig[0:3]

    @computed_field
    @property
    def direction(self) -> tuple:
        """
        Get the direction cosines of the image.

        Returns
        -------
        tuple
            A tuple containing the direction cosines.
        """
        return self.cube.GetDirection()

    # @computed_field
    # @property
    # def num_of_ct_scen(self) -> int:
    #     """
    #     Get the number of scenarios stored in the image.

    #     Returns
    #     -------
    #     int
    #         The number of scenarios.
    #     """
    #     if self.cube.GetDimension() == 4:
    #         return self.cube.GetSize()[3]
    #     return 1

    @computed_field
    @property
    def x(self) -> np.ndarray:
        """
        Calculate and get the x-vector of the image.

        Returns
        -------
        np.ndarray
            The corresponding x-vector.
        """
        x_spacing = self.cube.GetSpacing()[0]
        x_origin = self.cube.GetOrigin()[0]
        n_x = self.size[0]
        return np.arange(x_origin, x_origin + n_x * x_spacing, x_spacing)

    @computed_field
    @property
    def y(self) -> np.ndarray:
        """
        Calculate and get the y-vector of the image.

        Returns
        -------
        np.ndarray
            The corresponding y-vector.
        """
        y_spacing = self.cube.GetSpacing()[1]
        y_origin = self.cube.GetOrigin()[1]
        n_y = self.size[1]
        return np.arange(y_origin, y_origin + n_y * y_spacing, y_spacing)

    @computed_field
    @property
    def z(self) -> np.ndarray:
        """
        Calculate and get the z-vector of the image.

        Returns
        -------
        np.ndarray
            The corresponding z-vector.
        """
        z_spacing = self.cube.GetSpacing()[2]
        z_origin = self.cube.GetOrigin()[2]
        n_z = self.size[2]
        return np.arange(z_origin, z_origin + n_z * z_spacing, z_spacing)

    @property
    def grid(self) -> Grid:
        """
        Get the grid of the image.

        Returns
        -------
        Grid
            The grid constructed from the SimpleITK image.
        """
        return Grid.from_sitk_image(self.cube)

    def world_to_cube_coords(self, world_coords: np.ndarray) -> np.ndarray:
        """
        Convert world coordinates to cube indices.

        Parameters
        ----------
        world_coords : np.ndarray
            The world coordinates to convert.

        Returns
        -------
        np.ndarray
            The converted cube indices.
        """
        return np.array(self.cube.TransformPhysicalPointToIndex(world_coords))

    def resample_to_grid(self, grid: Grid) -> Self:
        """
        Resample the image to match the specified grid.

        Parameters
        ----------
        grid : Grid
            The grid to resample the image to.

        Returns
        -------
        Self
            The resampled QuantityCube object.
        """
        resampler = sitk.ResampleImageFilter()
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetOutputDirection(grid.direction.ravel())
        resampler.SetOutputOrigin(grid.origin)
        resampler.SetOutputSpacing(grid.resolution_vector)
        resampler.SetSize(grid.dimensions)

        # if self.num_of_ct_scen > 1:
        #     new_images = []
        #     for i in range(self.cube.GetSize()[-1]):
        #         new_images.append(resampler.Execute(self.cube[:, :, :, i]))
        #     new_cube = sitk.JoinSeries(new_images)
        # else:
        new_cube = resampler.Execute(self.cube)

        return self.model_validate({"cube": new_cube})


def resample_ct(cube: QuantityCube, interpolator=sitk.sitkBSpline, **kwargs) -> QuantityCube:
    """
    Resample QuantityCube to a different grid.

    Returns
    -------
    QuantityCube
        A resampled QuantityCube object
    """

    cube = QuantityCube.cube

    resampled_cube = resample_image(input_image=cube, interpolator=interpolator, **kwargs)

    return QuantityCube(cube_hu=resampled_cube)


def create_cube(
    data: Union[dict[str, Any], QuantityCube, os.PathLike, str, None] = None, **kwargs
) -> QuantityCube:
    """
    Create a QuantityCube object from various input types.

    Parameters
    ----------
    data : Union[dict[str, Any], QuantityCube, os.PathLike, str, None], optional
        The input data to create the QuantityCube object from. Can be a dictionary,
        existing QuantityCube object, file path, or None.
    **kwargs
        Additional keyword arguments to create the QuantityCube object.

    Returns
    -------
    QuantityCube
        A QuantityCube object created from the input data or keyword arguments.
    """
    if data:
        if isinstance(data, QuantityCube):
            return data
        return QuantityCube.model_validate(data)
    return QuantityCube(**kwargs)


def validate_cube(
    cube: Union[dict[str, Any], QuantityCube, os.PathLike, None] = None, **kwargs
) -> QuantityCube:
    """
    Validate the cube input and return a QuantityCube object.

    Parameters
    ----------
    cube : Union[dict[str, Any], QuantityCube, os.PathLike, None]
        The input cube to validate.

    Returns
    -------
    QuantityCube
        A validated QuantityCube object.
    """

    return create_cube(cube, **kwargs)
