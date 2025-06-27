"""Module for storing a given quantity values."""

import os
from typing import Annotated, Any, Union
from typing_extensions import Self
from pydantic import (
    Field,
    computed_field,
    model_validator,
    ValidationInfo,
)
import SimpleITK as sitk
import numpy as np
from pyRadPlan.core import resample_image

from pyRadPlan.core import PyRadPlanBaseModel, Grid


class QuantityDistribution(PyRadPlanBaseModel):
    """
    A class representing the result of a dose calculation.

    This class is used to store and do further sequencing/analysis of the dose valuess.
    """

    values: Annotated[sitk.Image, Field(init=False)]  # ,alias="values")]

    @classmethod
    def _retrieve_data(cls, data: Any, info: ValidationInfo) -> dict:
        """Search for data coming from validation."""
        if isinstance(data, Union[sitk.Image, np.ndarray]):
            # If the input is a SimpleITK image or numpy array, set it as the values
            data = {"values": data}
        if not isinstance(data, dict):
            raise ValueError("Invalid type for dose values input.")
        # Try to get the values using our internal key or its alias
        if "cube" in data:
            values = data.get("cube")
        elif "values" in data:
            values = data.get("values")
        if values is None:
            values_fieldname = cls.model_fields.get("cube").validation_alias
            values = data.pop(values_fieldname, None)
        if values is None:
            raise ValueError("Dose values not present in dictionary.")
        return values, data

    @model_validator(mode="before")
    @classmethod
    def validate_values(cls, data: Any, info: ValidationInfo) -> Any:
        # Early return for values input (assuming values is defined/imported elsewhere)
        if isinstance(data, QuantityDistribution):
            return data

        values, data = cls._retrieve_data(data, info)
        if isinstance(values, np.ndarray):
            # Determine if a permutation is required (e.g. matRad data has 'valuesDim')
            permute = data.get("valuesDim") or (
                info.context and info.context.get("from_matRad", False)
            )
            if values.dtype == object:
                values = sitk.JoinSeries(
                    [
                        sitk.GetImageFromArray(
                            np.transpose(values[i], (2, 0, 1)) if permute else values[i],
                            False,
                        )
                        for i in range(len(values))
                    ]
                )
            else:
                arr = np.transpose(values, (2, 0, 1)) if permute else values
                values = sitk.GetImageFromArray(arr, False)

        # Save the result under the key used in this model
        data["values"] = values

        # Validate that the values is indeed a SimpleITK image.
        if not isinstance(data["values"], sitk.Image):
            raise ValueError(f"Unsupported format of HU values: {type(data['values'])}")

        is4d = values.GetDimension() == 4

        # Set image origin if available from the dictionary.
        if "origin" in data:
            data["values"].SetOrigin(data["origin"])
        elif all(key in data for key in ("x", "y", "z")):
            origin = np.array([data["x"][0], data["y"][0], data["z"][0]], dtype=float)
            if is4d:
                origin = np.append(origin, [0.0])
            data["values"].SetOrigin(origin)

        # Set direction if provided.
        if "direction" in data:
            data["values"].SetDirection(data["direction"])

        # Set spacing (resolution) if available.
        if "resolution" in data and all(k in data["resolution"] for k in ("x", "y", "z")):
            res = data["resolution"]
            spacing = (
                [res["x"], res["y"], res["z"], 1.0] if is4d else [res["x"], res["y"], res["z"]]
            )
            data["values"].SetSpacing(spacing)

        return data

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
        spacing = self.values.GetSpacing()
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
        return self.values.GetSize()[0:3]

    @computed_field
    @property
    def values_dim(self) -> tuple:
        """
        Get the values dimensions in x, y, and z.

        Returns
        -------
        tuple
            A tuple containing the values dimensions.
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
        orig = self.values.GetOrigin()
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
        return self.values.GetDirection()

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
    #     if self.values.GetDimension() == 4:
    #         return self.values.GetSize()[3]
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
        x_spacing = self.values.GetSpacing()[0]
        x_origin = self.values.GetOrigin()[0]
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
        y_spacing = self.values.GetSpacing()[1]
        y_origin = self.values.GetOrigin()[1]
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
        z_spacing = self.values.GetSpacing()[2]
        z_origin = self.values.GetOrigin()[2]
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
        return Grid.from_sitk_image(self.values)

    def world_to_values_coords(self, world_coords: np.ndarray) -> np.ndarray:
        """
        Convert world coordinates to values indices.

        Parameters
        ----------
        world_coords : np.ndarray
            The world coordinates to convert.

        Returns
        -------
        np.ndarray
            The converted values indices.
        """
        return np.array(self.values.TransformPhysicalPointToIndex(world_coords))

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
            The resampled Quantityvalues object.
        """
        resampler = sitk.ResampleImageFilter()
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetOutputDirection(grid.direction.ravel())
        resampler.SetOutputOrigin(grid.origin)
        resampler.SetOutputSpacing(grid.resolution_vector)
        resampler.SetSize(grid.dimensions)

        # if self.num_of_ct_scen > 1:
        #     new_images = []
        #     for i in range(self.values.GetSize()[-1]):
        #         new_images.append(resampler.Execute(self.values[:, :, :, i]))
        #     new_values = sitk.JoinSeries(new_images)
        # else:
        new_values = resampler.Execute(self.values)

        return self.model_validate({"values": new_values})


def resample_ct(
    values: QuantityDistribution, interpolator=sitk.sitkBSpline, **kwargs
) -> QuantityDistribution:
    """
    Resample QuantityDistribution to a different grid.

    Returns
    -------
    QuantityDistribution
        A resampled QuantityDistribution object
    """

    values = QuantityDistribution.values

    resampled_values = resample_image(input_image=values, interpolator=interpolator, **kwargs)

    return QuantityDistribution(values_hu=resampled_values)


def create_distribution(
    data: Union[dict[str, Any], QuantityDistribution, os.PathLike, str, None] = None, **kwargs
) -> QuantityDistribution:
    """
    Create a QuantityDistribution object from various input types.

    Parameters
    ----------
    data : Union[dict[str, Any], QuantityDistribution, os.PathLike, str, None], optional
        The input data to create the QuantityDistribution object from. Can be a dictionary,
        existing QuantityDistribution object, file path, or None.
    **kwargs
        Additional keyword arguments to create the QuantityDistribution object.

    Returns
    -------
    QuantityDistribution
        A QuantityDistribution object created from the input data or keyword arguments.
    """
    if data:
        if isinstance(data, QuantityDistribution):
            return data
        return QuantityDistribution.model_validate(data)
    return QuantityDistribution(**kwargs)


def validate_distribution(
    values: Union[dict[str, Any], QuantityDistribution, os.PathLike, None] = None, **kwargs
) -> QuantityDistribution:
    """
    Validate the values input and return a QuantityDistribution object.

    Parameters
    ----------
    values : Union[dict[str, Any], QuantityDistribution, os.PathLike, None]
        The input values to validate.

    Returns
    -------
    QuantityDistribution
        A validated QuantityDistribution object.
    """

    return create_distribution(values, **kwargs)
