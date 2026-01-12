"""Module for handling CT images in pyRadPlan.

Defines a CT pydantic model and factories.
"""

import os
from abc import ABC
from typing import Annotated, Any, Union
from typing_extensions import Self
from pydantic import (
    Field,
    computed_field,
    model_validator,
)
import SimpleITK as sitk
import numpy as np
from pyRadPlan.core import resample_image

from pyRadPlan.core import PyRadPlanBaseModel, Grid


class CT(PyRadPlanBaseModel, ABC):
    """
    A class representing a CT (Computed Tomography) image.

    This class extends PyRadPlanBaseModel and provides functionality to
    handle CT images, including their properties like resolution, size, origin,
    and direction.

    Attributes
    ----------
    cube_hu : sitk.Image
        The CT image data in Hounsfield Units (HU).
    resolution : dict
        The resolution of the CT image in x, y, and z directions.
    size : tuple
        The size of the CT image in x, y, and z directions.
    origin : tuple
        The origin coordinates of the CT image.
    direction : tuple
        The direction cosines of the CT image.

    Methods
    -------
    to_matrad
        Converts the CT object to a dictionary compatible with matRad format.
    """

    cube_hu: Annotated[sitk.Image, Field(init=False, alias="cubeHU")]

    @model_validator(mode="before")
    @classmethod
    def validate_cube_hu(cls, data: Any) -> Any:
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

        Returns
        -------
        Any
            The validated and converted data.

        Raises
        ------
        ValueError
            If the HU cube is not present in the input dictionary.
        """
        # Return CT objects unchanged
        if isinstance(data, CT):
            return data

        # Process dictionary input
        if not isinstance(data, dict):
            return data

        # Extract cube_hu from data using field name or alias
        cube_hu = cls._extract_cube_hu(data)

        # Auto-extract origin from SimpleITK images if not provided
        cls._extract_origin_from_sitk_image(data, cube_hu)

        # Make sure that we have a SimpleITK image
        if isinstance(cube_hu, np.ndarray):
            cube_hu = cls._convert_numpy_to_sitk(data, cube_hu)

        # Update data with processed cube_hu
        data["cube_hu"] = cube_hu

        # Validate final format (if some other type than numpy/sitk is provided)
        cls._validate_sitk_format(data)

        # Apply image properties
        cls._apply_image_properties(data, cube_hu)

        return data

    @classmethod
    def _extract_cube_hu(cls, data: dict) -> Any:
        """Extract cube_hu from data dictionary using field name or alias."""
        cube_hu = data.get("cube_hu", None)
        if cube_hu is None:
            cube_fieldname = cls.model_fields.get("cube_hu").validation_alias
            cube_hu = data.pop(cube_fieldname, None)

        if cube_hu is None:
            raise ValueError("HU cube not present in dictionary.")

        return cube_hu

    @classmethod
    def _extract_origin_from_sitk_image(cls, data: dict, cube_hu: Any) -> None:
        """Extract origin from SimpleITK image if not already provided."""
        if isinstance(cube_hu, sitk.Image) and "origin" not in data:
            data["origin"] = cube_hu.GetOrigin()

    @classmethod
    def _convert_numpy_to_sitk(cls, data: dict, cube_hu: np.ndarray) -> sitk.Image:
        """Convert numpy array(s) to SimpleITK image(s)."""
        # Determine if data needs axis permutation (matRad format)
        should_permute = data.get("cubeDim", None) is not None

        # Handle multiple CT scenarios (object array)
        if cube_hu.dtype == object:
            return cls._convert_multiple_scenarios(cube_hu, should_permute)

        # Handle single CT scenario
        return cls._convert_single_scenario(cube_hu, should_permute)

    @classmethod
    def _convert_multiple_scenarios(cls, cube_hu: np.ndarray, should_permute: bool) -> sitk.Image:
        """Convert multiple CT scenarios to joined SimpleITK image."""
        num_cubes = len(cube_hu)

        if should_permute:
            ct_scenarios = [
                sitk.GetImageFromArray(np.transpose(cube_hu[i], (2, 0, 1)), False)
                for i in range(num_cubes)
            ]
        else:
            ct_scenarios = [sitk.GetImageFromArray(cube_hu[i], False) for i in range(num_cubes)]

        return sitk.JoinSeries(ct_scenarios)

    @classmethod
    def _convert_single_scenario(cls, cube_hu: np.ndarray, should_permute: bool) -> sitk.Image:
        """Convert single CT scenario to SimpleITK image."""
        if should_permute:
            return sitk.GetImageFromArray(np.transpose(cube_hu, (2, 0, 1)), False)
        else:
            return sitk.GetImageFromArray(cube_hu, False)

    @classmethod
    def _validate_sitk_format(cls, data: dict) -> None:
        """Validate that cube_hu is a SimpleITK image."""
        if not isinstance(data["cube_hu"], sitk.Image):
            raise ValueError(f"Unsupported format of HU cube: {type(data['cube_hu'])}")

    @classmethod
    def _apply_image_properties(cls, data: dict, cube_hu: sitk.Image) -> None:
        """Apply direction, spacing, and origin properties to the SimpleITK image."""
        is4d = cube_hu.GetDimension() == 4

        # Apply direction
        if "direction" in data:
            data["cube_hu"].SetDirection(data["direction"])

        # Apply spacing from resolution
        cls._apply_spacing(data, cube_hu, is4d)

        # Apply origin (with three different strategies)
        cls._apply_origin(data, cube_hu, is4d)

    @classmethod
    def _apply_spacing(cls, data: dict, cube_hu: sitk.Image, is4d: bool) -> None:
        """Apply spacing from resolution data."""
        if "resolution" not in data:
            return

        resolution = data["resolution"]
        if not all(key in resolution for key in ("x", "y", "z")):
            return

        if is4d:
            spacing = [resolution["x"], resolution["y"], resolution["z"], 1.0]
        else:
            spacing = [resolution["x"], resolution["y"], resolution["z"]]

        cube_hu.SetSpacing(spacing)

    @classmethod
    def _apply_origin(cls, data: dict, cube_hu: sitk.Image, is4d: bool) -> None:
        """Apply origin to the SimpleITK image using three different strategies."""
        if "origin" in data:
            # Strategy 1: Use explicitly provided origin
            data["cube_hu"].SetOrigin(data["origin"])
        elif all(key in data for key in ("x", "y", "z")):
            # Strategy 2: Calculate origin from x, y, z coordinate vectors
            cls._apply_origin_from_coordinate_vectors(data, is4d)
        else:
            # Strategy 3: Calculate centered origin based on image geometry
            cls._apply_centered_origin(data, is4d)

    @classmethod
    def _apply_origin_from_coordinate_vectors(cls, data: dict, is4d: bool) -> None:
        """Calculate and apply origin from x, y, z coordinate vectors."""
        origin = np.array([data["x"][0], data["y"][0], data["z"][0]], dtype=float)
        if is4d:
            origin = np.append(origin, [0.0])
        data["cube_hu"].SetOrigin(origin)

    @classmethod
    def _apply_centered_origin(cls, data: dict, is4d: bool) -> None:
        """Calculate and apply centered origin based on image geometry."""
        centered_origin = (
            -np.array(data["cube_hu"].GetSize()) / 2.0 * np.array(data["cube_hu"].GetSpacing())
        )

        if is4d:
            centered_origin = np.append(centered_origin, [0.0])

        data["cube_hu"].SetOrigin(centered_origin)

    @computed_field
    @property
    def resolution(self) -> dict:
        """
        Get the resolution of the CT image.

        Returns
        -------
        dict
            A dictionary containing the resolution in x, y, and z directions.
        """

        resolution = self.cube_hu.GetSpacing()
        res = {}
        res["x"] = resolution[0]
        res["y"] = resolution[1]
        res["z"] = resolution[2]
        return res

    @computed_field
    @property
    def size(self) -> tuple:
        """
        Get the size of the CT image.

        Returns
        -------
        tuple
            A tuple containing the size in x, y, and z directions.
        """
        return self.cube_hu.GetSize()[0:3]

    @computed_field
    @property
    def cube_dim(self) -> tuple:
        """
        Get the size of the CT image.

        Returns
        -------
        tuple
            A tuple containing the size in x, y, and z directions.
        """
        return self.size[0:3]

    @computed_field
    @property
    def origin(self) -> tuple:
        """
        Get the origin of the CT image.

        Returns
        -------
        tuple
            A tuple containing the origin coordinates.
        """
        origin = self.cube_hu.GetOrigin()
        return origin[0:3]

    @computed_field
    @property
    def direction(self) -> tuple:
        """
        Get the direction of the CT image.

        Returns
        -------
        tuple
            A tuple containing the direction cosines.
        """
        direction = self.cube_hu.GetDirection()
        return direction

    @computed_field
    @property
    def num_of_ct_scen(self) -> int:
        """
        Get the number of CT scenarios.

        Returns
        -------
        int
            The number of CT scenarios.
        """
        if self.cube_hu.GetDimension() == 4:
            return self.cube_hu.GetSize()[3]
        return 1

    @computed_field
    @property
    def x(self) -> np.ndarray:
        """
        Calculate and get the x-vector of the CT image.

        Returns
        -------
        int
            The corresponding x-vector of the CT image.
        """

        x_spacing = self.cube_hu.GetSpacing()[0]
        origin = self.cube_hu.GetOrigin()[0]
        x = np.arange(origin, (self.size[0] * x_spacing + origin), x_spacing)
        return x

    @computed_field
    @property
    def y(self) -> np.ndarray:
        """
        Calculate and get the y-vector of the CT image.

        Returns
        -------
        int
            The corresponding y-vector of the CT image.
        """

        y_spacing = self.cube_hu.GetSpacing()[1]
        origin = self.cube_hu.GetOrigin()[1]
        y = np.arange(origin, (self.size[1] * y_spacing + origin), y_spacing)
        return y

    @computed_field
    @property
    def z(self) -> np.ndarray:
        """
        Calculate and get the z-vector of the CT image.

        Returns
        -------
        int
            The corresponding z-vector of the CT image.
        """

        z_spacing = self.cube_hu.GetSpacing()[2]
        origin = self.cube_hu.GetOrigin()[2]
        z = np.arange(origin, (self.size[2] * z_spacing + origin), z_spacing)
        return z

    @property
    def grid(self) -> Grid:
        """
        Get the grid of the CT image.

        Returns
        -------
        Grid
            The grid of the CT image.
        """
        return Grid.from_sitk_image(self.cube_hu)

    def world_to_cube_coords(self, world_coords: np.ndarray) -> np.ndarray:
        """
        Convert world coordinates to cube coordinates.

        Parameters
        ----------
        world_coords : np.ndarray
            The world coordinates to convert.

        Returns
        -------
        np.ndarray
            The converted cube coordinates.
        """

        return np.array(self.cube_hu.TransformPhysicalPointToIndex(world_coords))

    def compute_wet(self, hlut: np.ndarray) -> sitk.Image:
        """
        Compute the water equivalent thickness (WET).

        Uses a provided appropriate Hounsfield Look-Up Table (HLUT).

        Parameters
        ----------
        hlut : np.ndarray
            The Hounsfield Look-Up Table (HLUT) to compute the WET.

        Returns
        -------
        sitk.Image
            The water equivalent thickness (WET) image.
        """

        if hlut.shape[1] != 2 and hlut.shape[0] <= 1:
            raise ValueError(
                "HLUT must have 2 columns of values, with the first column being the HU values "
                "and the second column being the WET values with each at least 2 values."
            )

        # Sort the HLUT by HU values
        ix = np.argsort(hlut[:, 0])
        xp = hlut[ix, 0]
        fp = hlut[ix, 1]

        # Get the np view of the image
        ct_array = sitk.GetArrayViewFromImage(self.cube_hu)

        # Convert the WET array back to a SimpleITK image
        wet_image = sitk.GetImageFromArray(np.interp(ct_array, xp, fp))
        wet_image.SetOrigin(self.cube_hu.GetOrigin())
        wet_image.SetSpacing(self.cube_hu.GetSpacing())
        wet_image.SetDirection(self.cube_hu.GetDirection())

        return wet_image

    def to_matrad(self, context: str = "mat-file") -> Any:
        """
        Convert the CT object to a dictionary compatible with matRad format.

        Returns
        -------
        Any
            A dictionary containing the CT data in matRad format.
        """
        ct_dict = super().to_matrad(context=context)
        # as cubeHU for the matRad format
        sitk_image = ct_dict["cubeHU"]
        # Flip axes for matRad format
        numpy_array = np.transpose(sitk.GetArrayFromImage(sitk_image), (1, 2, 0))
        # We also need to put the image array into a ndarray of objects
        ct_dict["cubeHU"] = np.ndarray(shape=(1,), dtype=object)
        ct_dict["cubeHU"][0] = numpy_array
        ct_dict["cubeDim"] = np.array(ct_dict["cubeDim"], dtype=float)
        return ct_dict

    def resample_to_grid(self, grid: Grid) -> Self:
        """
        Resample the CT image to match the specified grid.

        Parameters
        ----------
        grid : Grid
            The grid to resample the CT image to.

        Returns
        -------
        Self
            The resampled CT object.
        """

        resampler = sitk.ResampleImageFilter()
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetOutputDirection(grid.direction.ravel())
        resampler.SetOutputOrigin(grid.origin)
        resampler.SetOutputSpacing(grid.resolution_vector)
        resampler.SetSize(grid.dimensions)

        if self.num_of_ct_scen > 1:
            new_ct = []
            for i in range(self.cube_hu.GetSize()[-1]):
                new_ct.append(resampler.Execute(self.cube_hu[:, :, :, i]))
            new_ct = sitk.JoinSeries(new_ct)
        else:
            new_ct = resampler.Execute(self.cube_hu)

        return self.model_validate({"cube_hu": new_ct})


def resample_ct(ct: CT, interpolator=sitk.sitkBSpline, **kwargs) -> CT:
    """
    Resample CT to a different grid.

    Returns
    -------
    CT
        A resampled CT object
    """

    ct_cube = ct.cube_hu

    resampled_ct_cube = resample_image(input_image=ct_cube, interpolator=interpolator, **kwargs)

    return validate_ct(cube_hu=resampled_ct_cube)


def ct_from_file(file_path: Union[str, os.PathLike]) -> CT:
    """
    Create a CT object from a file.

    Parameters
    ----------
    file_path : Union[str, os.PathLike]
        The path to the CT image file.

    Returns
    -------
    CT
        A CT object created from the file.

    Raises
    ------
    ValueError
        If reading DICOM series from folders is attempted (not implemented).
    """
    if os.path.isfile(file_path):
        sitk_image = sitk.ReadImage(file_path)
        # GetArrayFromImage accounts for the difference between numpy (z, x, y)
        # and SITK (x, y, z) standards

    else:
        raise ValueError("Reading dicom series from folders not implemented")

    return create_ct(cube_hu=sitk_image)


def create_ct(data: Union[dict[str, Any], CT, os.PathLike, str, None] = None, **kwargs) -> CT:
    """
    Create a CT object from various input types.

    Parameters
    ----------
    data : Union[dict[str, Any], CT, os.PathLike, str, None], optional
        The input data to create the CT object from. Can be a dictionary,
        existing CT object, file path, or None.
    **kwargs
        Additional keyword arguments to create the CT object.

    Returns
    -------
    CT
        A CT object created from the input data or keyword arguments.
    """
    if data:  # If data is not None
        # If data is already a CT object return it directly
        if isinstance(data, CT):
            return data
        # If data is in a file get CT object from file
        if isinstance(data, str) or isinstance(data, os.PathLike):
            return ct_from_file(data)
        # If data is in a dictionary create an SimpleITK image and then
        # the CT object
        return CT.model_validate(data)
    # If neither CT object nor dictionary try to
    # get model from keyword arguments
    return CT(**kwargs)


def validate_ct(ct: Union[dict[str, Any], CT, os.PathLike, None] = None, **kwargs) -> CT:
    """
    Validate and create a CT object.

    This function is a wrapper around create_ct and ensures the returned object
    is a valid CT instance.

    Parameters
    ----------
    ct : Union[dict[str, Any], CT, os.PathLike, None], optional
        The input data to validate and create the CT object from.
    **kwargs
        Additional keyword arguments to create the CT object.

    Returns
    -------
    CT
        A validated CT object.
    """
    return create_ct(ct, **kwargs)
