"""Module for storing a given quantity distribution."""

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

    This class is used to store and do further sequencing/analysis of the dose distributions.
    """

    distribution: Annotated[sitk.Image, Field(init=False)]  # ,alias="distribution")]

    @classmethod
    def _retrieve_data(cls, data: Any, info: ValidationInfo) -> dict:
        """Search for data coming from validation."""
        if isinstance(data, Union[sitk.Image, np.ndarray]):
            # If the input is a SimpleITK image or numpy array, set it as the distribution
            data = {"distribution": data}
        if not isinstance(data, dict):
            raise ValueError("Invalid type for dose distribution input.")
        # Try to get the distribution using our internal key or its alias
        if "cube" in data:
            distribution = data.get("cube")
        elif "distribution" in data:
            distribution = data.get("distribution")
        if distribution is None:
            distribution_fieldname = cls.model_fields.get("cube").validation_alias
            distribution = data.pop(distribution_fieldname, None)
        if distribution is None:
            raise ValueError("Dose distribution not present in dictionary.")
        return distribution, data

    @model_validator(mode="before")
    @classmethod
    def validate_distribution(cls, data: Any, info: ValidationInfo) -> Any:
        # Early return for distribution input (assuming distribution is defined/imported elsewhere)
        if isinstance(data, QuantityDistribution):
            return data

        distribution, data = cls._retrieve_data(data, info)
        if isinstance(distribution, np.ndarray):
            # Determine if a permutation is required (e.g. matRad data has 'distributionDim')
            permute = data.get("distributionDim") or (
                info.context and info.context.get("from_matRad", False)
            )
            if distribution.dtype == object:
                distribution = sitk.JoinSeries(
                    [
                        sitk.GetImageFromArray(
                            np.transpose(distribution[i], (2, 0, 1))
                            if permute
                            else distribution[i],
                            False,
                        )
                        for i in range(len(distribution))
                    ]
                )
            else:
                arr = np.transpose(distribution, (2, 0, 1)) if permute else distribution
                distribution = sitk.GetImageFromArray(arr, False)

        # Save the result under the key used in this model
        data["distribution"] = distribution

        # Validate that the distribution is indeed a SimpleITK image.
        if not isinstance(data["distribution"], sitk.Image):
            raise ValueError(
                f"Unsupported format of HU distribution: {type(data['distribution'])}"
            )

        is4d = distribution.GetDimension() == 4

        # Set image origin if available from the dictionary.
        if "origin" in data:
            data["distribution"].SetOrigin(data["origin"])
        elif all(key in data for key in ("x", "y", "z")):
            origin = np.array([data["x"][0], data["y"][0], data["z"][0]], dtype=float)
            if is4d:
                origin = np.append(origin, [0.0])
            data["distribution"].SetOrigin(origin)

        # Set direction if provided.
        if "direction" in data:
            data["distribution"].SetDirection(data["direction"])

        # Set spacing (resolution) if available.
        if "resolution" in data and all(k in data["resolution"] for k in ("x", "y", "z")):
            res = data["resolution"]
            spacing = (
                [res["x"], res["y"], res["z"], 1.0] if is4d else [res["x"], res["y"], res["z"]]
            )
            data["distribution"].SetSpacing(spacing)

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
        spacing = self.distribution.GetSpacing()
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
        return self.distribution.GetSize()[0:3]

    @computed_field
    @property
    def distribution_dim(self) -> tuple:
        """
        Get the distribution dimensions in x, y, and z.

        Returns
        -------
        tuple
            A tuple containing the distribution dimensions.
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
        orig = self.distribution.GetOrigin()
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
        return self.distribution.GetDirection()

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
    #     if self.distribution.GetDimension() == 4:
    #         return self.distribution.GetSize()[3]
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
        x_spacing = self.distribution.GetSpacing()[0]
        x_origin = self.distribution.GetOrigin()[0]
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
        y_spacing = self.distribution.GetSpacing()[1]
        y_origin = self.distribution.GetOrigin()[1]
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
        z_spacing = self.distribution.GetSpacing()[2]
        z_origin = self.distribution.GetOrigin()[2]
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
        return Grid.from_sitk_image(self.distribution)

    def world_to_distribution_coords(self, world_coords: np.ndarray) -> np.ndarray:
        """
        Convert world coordinates to distribution indices.

        Parameters
        ----------
        world_coords : np.ndarray
            The world coordinates to convert.

        Returns
        -------
        np.ndarray
            The converted distribution indices.
        """
        return np.array(self.distribution.TransformPhysicalPointToIndex(world_coords))

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
            The resampled QuantityDistribution object.
        """
        resampler = sitk.ResampleImageFilter()
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetOutputDirection(grid.direction.ravel())
        resampler.SetOutputOrigin(grid.origin)
        resampler.SetOutputSpacing(grid.resolution_vector)
        resampler.SetSize(grid.dimensions)

        # if self.num_of_ct_scen > 1:
        #     new_images = []
        #     for i in range(self.distribution.GetSize()[-1]):
        #         new_images.append(resampler.Execute(self.distribution[:, :, :, i]))
        #     new_distribution = sitk.JoinSeries(new_images)
        # else:
        new_distribution = resampler.Execute(self.distribution)

        return self.model_validate({"distribution": new_distribution})


def resample_ct(
    distribution: QuantityDistribution, interpolator=sitk.sitkBSpline, **kwargs
) -> QuantityDistribution:
    """
    Resample QuantityDistribution to a different grid.

    Returns
    -------
    QuantityDistribution
        A resampled QuantityDistribution object
    """

    distribution = QuantityDistribution.distribution

    resampled_distribution = resample_image(
        input_image=distribution, interpolator=interpolator, **kwargs
    )

    return QuantityDistribution(distribution_hu=resampled_distribution)


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
    distribution: Union[dict[str, Any], QuantityDistribution, os.PathLike, None] = None, **kwargs
) -> QuantityDistribution:
    """
    Validate the distribution input and return a QuantityDistribution object.

    Parameters
    ----------
    distribution : Union[dict[str, Any], QuantityDistribution, os.PathLike, None]
        The input distribution to validate.

    Returns
    -------
    QuantityDistribution
        A validated QuantityDistribution object.
    """

    return create_distribution(distribution, **kwargs)
