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
from pyRadPlan.core import PyRadPlanBaseModel, Grid


class QuantityDistribution(PyRadPlanBaseModel):
    """
    A class representing the result of a dose calculation.

    This class is used to store and do further sequencing/analysis of the dose values.
    """

    values: Annotated[sitk.Image, Field(init=False)]  # ,alias="values")]

    @classmethod
    def _retrieve_data(cls, data: Any, info: ValidationInfo) -> tuple:
        """Extract values from input data, normalizing various input formats."""
        if isinstance(data, (sitk.Image, np.ndarray)):
            data = {"values": data}
        if not isinstance(data, dict):
            raise ValueError(f"Invalid type for quantity values input: {type(data).__name__}")

        if "cube" in data:
            values = data.get("cube")
        elif "values" in data:
            values = data.get("values")
        else:
            raise ValueError(
                "Quantity values not present in dictionary. Expected 'cube' or 'values' key."
            )
        return values, data

    @model_validator(mode="before")
    @classmethod
    def validate_values(cls, data: Any, info: ValidationInfo) -> Any:
        """
        Validate and convert input data to SimpleITK image format.

        This method checks if the input data is in the correct format and
        converts numpy arrays to SimpleITK images if necessary. It also applies
        any specified image properties (origin, direction, spacing) to the
        SimpleITK image.

        Parameters
        ----------
        data : Any
            The input data to be validated and converted.
        info : ValidationInfo
            Validation context information.

        Returns
        -------
        Any
            The validated and converted data.

        Raises
        ------
        ValueError
            If the values are not present or in unsupported format.
        """
        # Return QuantityDistribution objects unchanged
        if isinstance(data, QuantityDistribution):
            return data

        # Extract and process values from input data
        values, processed_data = cls._extract_and_process_values(data, info)

        # Update processed_data with converted values
        processed_data["values"] = values

        # Validate final format
        cls._validate_sitk_format(processed_data)

        # Apply image properties
        cls._apply_image_properties(processed_data, values)

        return processed_data

    @classmethod
    def _extract_and_process_values(
        cls, data: Any, info: ValidationInfo
    ) -> tuple[sitk.Image, dict]:
        """Extract values from data and convert numpy arrays to SimpleITK images."""
        values, processed_data = cls._retrieve_data(data, info)

        if isinstance(values, np.ndarray):
            values = cls._convert_numpy_to_sitk(processed_data, values, info)

        return values, processed_data

    @classmethod
    def _convert_numpy_to_sitk(
        cls, data: dict, values: np.ndarray, info: ValidationInfo
    ) -> sitk.Image:
        """Convert numpy array(s) to SimpleITK image(s)."""
        # Determine if data needs axis permutation (matRad format)
        should_permute = cls._should_permute_axes(data, info)

        # Handle multiple scenarios (object array)
        if values.dtype == object:
            return cls._convert_multiple_scenarios(values, should_permute)

        # Handle single scenario
        return cls._convert_single_scenario(values, should_permute)

    @classmethod
    def _should_permute_axes(cls, data: dict, info: ValidationInfo) -> bool:
        """Determine if axis permutation is required for matRad data."""
        return data.get("valuesDim") or (info.context and info.context.get("from_matRad", False))

    @classmethod
    def _convert_multiple_scenarios(cls, values: np.ndarray, should_permute: bool) -> sitk.Image:
        """Convert multiple scenarios to joined SimpleITK image."""
        return sitk.JoinSeries(
            [
                sitk.GetImageFromArray(
                    np.transpose(values[i], (2, 0, 1)) if should_permute else values[i],
                    False,
                )
                for i in range(len(values))
            ]
        )

    @classmethod
    def _convert_single_scenario(cls, values: np.ndarray, should_permute: bool) -> sitk.Image:
        """Convert single scenario to SimpleITK image."""
        arr = np.transpose(values, (2, 0, 1)) if should_permute else values
        return sitk.GetImageFromArray(arr, False)

    @classmethod
    def _validate_sitk_format(cls, data: dict) -> None:
        """Validate that values is a SimpleITK image."""
        if not isinstance(data["values"], sitk.Image):
            raise ValueError(f"Unsupported format of quantity values: {type(data['values'])}")

    @classmethod
    def _apply_image_properties(cls, data: dict, values: sitk.Image) -> None:
        """Apply origin, direction, and spacing properties to the SimpleITK image."""
        is4d = values.GetDimension() == 4

        # Apply origin (with two different strategies)
        cls._apply_origin(data, is4d)

        # Apply direction
        if "direction" in data:
            data["values"].SetDirection(data["direction"])

        # Apply spacing from resolution
        cls._apply_spacing(data, is4d)

    @classmethod
    def _apply_origin(cls, data: dict, is4d: bool) -> None:
        """Apply origin to the SimpleITK image using two different strategies."""
        if "origin" in data:
            # Strategy 1: Use explicitly provided origin
            data["values"].SetOrigin(data["origin"])
        elif all(key in data for key in ("x", "y", "z")):
            # Strategy 2: Calculate origin from x, y, z coordinate vectors
            cls._apply_origin_from_coordinate_vectors(data, is4d)

    @classmethod
    def _apply_origin_from_coordinate_vectors(cls, data: dict, is4d: bool) -> None:
        """Calculate and apply origin from x, y, z coordinate vectors."""
        origin = np.array([data["x"][0], data["y"][0], data["z"][0]], dtype=float)
        if is4d:
            origin = np.append(origin, [0.0])
        data["values"].SetOrigin(origin)

    @classmethod
    def _apply_spacing(cls, data: dict, is4d: bool) -> None:
        """Apply spacing from resolution data."""
        if "resolution" not in data:
            return

        resolution = data["resolution"]
        if not all(k in resolution for k in ("x", "y", "z")):
            return

        spacing = (
            [resolution["x"], resolution["y"], resolution["z"], 1.0]
            if is4d
            else [resolution["x"], resolution["y"], resolution["z"]]
        )
        data["values"].SetSpacing(spacing)

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
        return np.linspace(x_origin, x_origin + (n_x - 1) * x_spacing, n_x)

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
        return np.linspace(y_origin, y_origin + (n_y - 1) * y_spacing, n_y)

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
        return np.linspace(z_origin, z_origin + (n_z - 1) * z_spacing, n_z)

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

        new_values = resampler.Execute(self.values)

        return self.model_validate({"values": new_values})


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
    if isinstance(data, QuantityDistribution):
        return data

    # Arrays and images are validated before the truthiness test below, because
    # ``if <ndarray>:`` raises an ambiguous-truth error.
    if isinstance(data, (sitk.Image, np.ndarray)):
        return QuantityDistribution.model_validate(data)

    if data:
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
