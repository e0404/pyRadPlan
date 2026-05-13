"""Image / Grid Resampling."""

from typing import Literal, Union

import numpy as np
import SimpleITK as sitk
from ._grids import Grid


def resample_image(
    input_image: sitk.Image,
    interpolator=sitk.sitkBSpline,
    target_image: sitk.Image = None,
    target_grid_spacing: tuple[float, float, float] = None,
    target_grid: Grid = None,
    extrapolate: Union[float, int, bool, Literal["nearest"]] = "nearest",
) -> sitk.Image:  # also accept sitk images and grid points. Can be dealt with with kwargs
    """
    Resample an sitk Image.

    Use target resolution, reference image, dimensions, or Grid as input.

    Parameters
    ----------
    input_image : sitk.Image
        The input image to be resampled.
    interpolator : sitk.InterpolatorEnum, optional
        The interpolator to use for resampling. Default is sitk.sitkBSpline.
    extrapolate : Union[float, int, bool, Literal["nearest"], None], optional
        The value to use for extrapolation outside the image boundaries. Default is None.
    target_image : sitk.Image, optional
        The reference image to resample to. Default is None.
    target_grid_spacing : tuple[float,float,float], optional
        The target grid spacing to resample to. Default is None.
    target_grid : Grid, optional
        The target grid to resample to. Default is None.

    Returns
    -------
    sitk.Image
        The resampled image.

    Notes
    -----
    Exactly one of target_image, target_grid_spacing, or target_grid must be provided.
    """

    if [target_image, target_grid_spacing, target_grid].count(None) != 2:
        raise TypeError(
            "Only one of target_image, target_grid_spacing, or target_grid must be provided."
        )

    image_grid = Grid.from_sitk_image(input_image)

    if target_image is not None:
        target_grid = Grid.from_sitk_image(target_image)
    elif target_grid_spacing is not None:
        target_grid = image_grid.resample(target_grid_spacing)

    resample = sitk.ResampleImageFilter()
    resample.UseNearestNeighborExtrapolatorOff()
    if extrapolate == "nearest":
        resample.UseNearestNeighborExtrapolatorOn()
    elif isinstance(extrapolate, (int, float)):
        resample.SetDefaultPixelValue(extrapolate)
    elif extrapolate is False:
        resample.SetDefaultPixelValue(0)
    elif extrapolate is True:
        resample.SetDefaultPixelValue(1)
    else:
        raise ValueError("Invalid value for 'extrapolate' parameter.")

    resample.SetOutputSpacing(target_grid.resolution_vector)
    resample.SetSize(target_grid.dimensions)
    resample.SetOutputDirection(target_grid.direction.ravel())
    resample.SetOutputOrigin(target_grid.origin)
    resample.SetInterpolator(interpolator)
    resampled_image = resample.Execute(input_image)

    return resampled_image


def resample_numpy_array(
    input_array: np.ndarray,
    reference_image: sitk.Image = None,
    reference_grid: Grid = None,
    interpolator=sitk.sitkBSpline,
    target_image: sitk.Image = None,
    target_grid_spacing: tuple[float, float, float] = None,
    target_grid: Grid = None,
    extrapolate: Union[float, int, bool, Literal["nearest"]] = "nearest",
) -> np.ndarray:
    """
    Resample a numpy grid.

    Use target resolution, reference image, dimensions, or Grid as input.

    We convert the numpy array to an sitk.Image, so the dimensions will be
    switched. More exactly,
    the numpy array will be index i,j,k <-> z,y,x, which will converted to
    j,i,k <-> y,x,z in sitk.
    This is to be considered when supplying the reference grid  or image.

    Parameters
    ----------
    input_image : sitk.Image
        The input image to be resampled.
    reference_image : sitk.Image, optional
        The reference image providing spatial information of the array. Default is None.
        Be wary of sitk <-> numpy indexing conventions.
    reference_grid : Grid, optional
        The reference grid of the resampled array. Default is None.
        Be wary of sitk <-> numpy indexing conventions.
    interpolator : sitk.InterpolatorEnum, optional
        The interpolator to use for resampling. Default is sitk.sitkBSpline.
    target_image : sitk.Image, optional
        The reference image to resample to. Default is None.
    target_grid_spacing : tuple[float,float,float], optional
        The target grid spacing to resample to. Default is None.
    target_grid : Grid, optional
        The target grid to resample to. Default is None.

    Returns
    -------
    sitk.Image
        The resampled image.

    Notes
    -----
    Exactly one of target_image, target_grid_spacing, or target_grid must be provided.
    """

    if [reference_image, reference_grid].count(None) != 1:
        raise TypeError("Only one of reference_image or reference_grid must be provided.")

    if reference_image is not None:
        reference_grid = Grid.from_sitk_image(reference_image)

    # I think there is an issue here. The axes are flipped with GetImageFromArray
    array_sitk = sitk.GetImageFromArray(input_array)
    array_sitk.SetOrigin(reference_grid.origin)
    array_sitk.SetSpacing(reference_grid.resolution_vector)
    array_sitk.SetDirection(reference_grid.direction.ravel())

    resampled_array_sitk = resample_image(
        input_image=array_sitk,
        interpolator=interpolator,
        target_image=target_image,
        target_grid_spacing=target_grid_spacing,
        target_grid=target_grid,
        extrapolate=extrapolate,
    )

    return sitk.GetArrayFromImage(resampled_array_sitk)
