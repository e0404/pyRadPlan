"""Image / Grid Resampling."""

from typing import Literal, Union

import warnings

import array_api_compat
import numpy as np
import SimpleITK as sitk
from ._grids import Grid
from .xp_utils.helpers import to_numpy, to_namespace
from .xp_utils.typing import Array
from .xp_utils._image_interpolation import interpolate_image_linear, interpolate_image_nearest


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
        The value to use for extrapolation outside the image boundaries. Default is "nearest".
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

    # BSpline (order 3) requires >= 4 voxels per dimension; fall back to linear for small images
    if interpolator == sitk.sitkBSpline and any(s < 4 for s in input_image.GetSize()):
        interpolator = sitk.sitkLinear

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


def resample_array(
    input_array: Array,
    reference_image: sitk.Image = None,
    reference_grid: Grid = None,
    interpolator=sitk.sitkBSpline,
    target_image: sitk.Image = None,
    target_grid_spacing: tuple[float, float, float] = None,
    target_grid: Grid = None,
    extrapolate: Union[float, int, bool, Literal["nearest"]] = "nearest",
    *,
    prefer_native: bool = True,
) -> Array:
    """Resample an array while preserving its namespace, device and dtype.

    Parameters
    ----------
    input_array : Array
        Scalar image in NumPy axis order, (z, y, x) for a 3D image.
    reference_image : SimpleITK.Image, optional
        Image supplying the input array's spatial metadata.
    reference_grid : Grid, optional
        Input spatial metadata, as an alternative to reference_image.
    interpolator : SimpleITK interpolator enum, optional
        Interpolation method, default sitkBSpline. Enums select semantics,
        independently of which library executes the interpolation.
    target_image : SimpleITK.Image, optional
        Image supplying the output geometry.
    target_grid_spacing : tuple[float, float, float], optional
        Output spacing; derive the remaining geometry from the input grid.
    target_grid : Grid, optional
        Explicit output geometry.
    extrapolate : float, int, bool or "nearest", optional
        Nearest-pixel extrapolation (default), or a constant outside the image.
    prefer_native : bool, optional
        Prefer SimpleITK for full NumPy/CuPy volumes (default True), including
        transfers for CuPy. False requests the Array API implementation and
        raises NotImplementedError for unsupported modes/dtypes/dimensions.

    Returns
    -------
    Array
        Resampled volume on the input namespace/device, in reversed grid-axis
        order and with the input dtype.

    Notes
    -----
    Supply exactly one reference and one target specification. With
    prefer_native=True, NumPy/CuPy use SimpleITK; other namespaces evaluate
    supported 3D linear/nearest interpolation on their own device and use
    SimpleITK for unsupported cases. A SimpleITK fallback involves a host round
    trip for device arrays. The requested interpolation mode is never replaced
    with linear, except for the existing small-image BSpline fallback in
    resample_image.

    For repeated geometry or selected target voxels, use
    :class:`ArrayBackendGridResampler` to retain backend mapping preparation.
    """
    if sum(x is not None for x in (reference_image, reference_grid)) != 1:
        raise TypeError("Only one of reference_image or reference_grid must be provided.")
    if sum(x is not None for x in (target_image, target_grid_spacing, target_grid)) != 1:
        raise TypeError(
            "Only one of target_image, target_grid_spacing, or target_grid must be provided."
        )
    if reference_image is not None:
        reference_grid = Grid.from_sitk_image(reference_image)
    xp = array_api_compat.array_namespace(input_array)
    device = array_api_compat.device(input_array)
    # Native full-volume execution is faster in the measured NumPy/CuPy workloads.
    # Prepared selected-voxel evaluation remains backend-resident.
    supported = (
        input_array.ndim == 3
        and len(reference_grid.dimensions) == 3
        and (target_image is None or target_image.GetDimension() == 3)
        and (target_grid is None or len(target_grid.dimensions) == 3)
        and _backend_interpolation_supported(xp, input_array.dtype, interpolator, extrapolate)
    )
    if not prefer_native and not supported:
        raise NotImplementedError(
            "Backend resampling does not support this mode/dtype/dimensionality; "
            "use prefer_native=True for SimpleITK fallback."
        )
    native_namespace = array_api_compat.is_numpy_namespace(
        xp
    ) or array_api_compat.is_cupy_namespace(xp)
    if prefer_native and (native_namespace or not supported):
        result = _resample_numpy_sitk(
            to_numpy(input_array),
            reference_grid=reference_grid,
            interpolator=interpolator,
            target_image=target_image,
            target_grid_spacing=target_grid_spacing,
            target_grid=target_grid,
            extrapolate=extrapolate,
        )
        return to_namespace(xp, result, device=device)
    if target_image is not None:
        target_grid = Grid.from_sitk_image(target_image)
    elif target_grid_spacing is not None:
        target_grid = reference_grid.resample(target_grid_spacing)
    indices = xp.arange(target_grid.num_voxels, dtype=xp.int64, device=device)
    prepared = ArrayBackendGridResampler(
        reference_grid, target_grid, indices, interpolator=interpolator, extrapolate=extrapolate
    )
    return xp.reshape(prepared(input_array), tuple(reversed(target_grid.dimensions)))


def _backend_interpolation_supported(xp, dtype, interpolator, extrapolate):
    """Select only image semantics implemented exactly by the Array API evaluator."""
    if interpolator not in (sitk.sitkLinear, sitk.sitkNearestNeighbor):
        return False
    if xp.isdtype(dtype, "real floating"):
        return dtype in (xp.float32, xp.float64)
    # Integer nearest-neighbor sampling must not convert labels through floats.
    # Constant casting outside the integer range is delegated to SimpleITK.
    if interpolator == sitk.sitkNearestNeighbor and xp.isdtype(dtype, "integral"):
        if extrapolate == "nearest":
            return True
        if isinstance(extrapolate, (int, float)):
            info = xp.iinfo(dtype)
            return info.min <= extrapolate <= info.max
    return False


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
    """Resample a NumPy volume; deprecated in favor of :func:`resample_array`.

    All positional arguments and SimpleITK interpolation semantics are retained.
    """
    warnings.warn(
        "resample_numpy_array is deprecated; use resample_array instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return resample_array(
        input_array,
        reference_image,
        reference_grid,
        interpolator,
        target_image,
        target_grid_spacing,
        target_grid,
        extrapolate,
    )


def _resample_numpy_sitk(
    input_array: np.ndarray,
    reference_image: sitk.Image = None,
    reference_grid: Grid = None,
    interpolator=sitk.sitkBSpline,
    target_image: sitk.Image = None,
    target_grid_spacing: tuple[float, float, float] = None,
    target_grid: Grid = None,
    extrapolate: Union[float, int, bool, Literal["nearest"]] = "nearest",
) -> np.ndarray:
    """Execute native image resampling on a host array."""
    if [reference_image, reference_grid].count(None) != 1:
        raise TypeError("Only one of reference_image or reference_grid must be provided.")

    if reference_image is not None:
        reference_grid = Grid.from_sitk_image(reference_image)

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


class ArrayBackendGridResampler:
    """Prepare backend image resampling, optionally only at selected target voxels.

    Parameters
    ----------
    source_grid, target_grid : Grid
        Physical image geometry. Only 3D scalar images are supported.
    target_indices : Array
        Flat C-order indices in the target (z, y, x) volume. Their namespace and
        device determine execution placement. Use all target indices for a full
        volume, or a selection to avoid resampling unused voxels.
    interpolator : SimpleITK interpolator enum, optional
        sitkLinear (default) or sitkNearestNeighbor. Other methods are supported
        through the one-shot resample_array interface's native fallback.
    extrapolate : float, int, bool or "nearest", optional
        Nearest-pixel extrapolation (default), or a constant outside the image.
    chunk_size : int, optional
        Maximum query count per interpolation call, bounding temporary storage.

    Notes
    -----
    Retains selected indices and scanline endpoint maps, not full coordinate grids
    or eight neighbor arrays. Grid metadata is snapshotted at creation. Values
    must use the same namespace/device. Do not mutate retained target indices.
    Output is a 1D array in selection order.
    Boundary and extrapolation semantics match SimpleITK image resampling.
    Float64 coordinate arithmetic preserves physical geometry independently of
    the image value precision.
    """

    def __init__(
        self,
        source_grid: Grid,
        target_grid: Grid,
        target_indices,
        *,
        interpolator=sitk.sitkLinear,
        extrapolate="nearest",
        chunk_size=262144,
    ):
        if interpolator not in (sitk.sitkLinear, sitk.sitkNearestNeighbor):
            raise NotImplementedError(
                "Prepared backend resampling supports linear and nearest neighbor."
            )
        if extrapolate != "nearest" and not isinstance(extrapolate, (int, float)):
            raise ValueError("Invalid value for 'extrapolate' parameter.")
        self.interpolator = interpolator
        self.extrapolate = extrapolate
        if len(source_grid.dimensions) != 3 or len(target_grid.dimensions) != 3:
            raise ValueError("Array image resampling requires 3D grids.")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")
        xp = array_api_compat.array_namespace(target_indices)
        if target_indices.ndim != 1 or not xp.isdtype(target_indices.dtype, "integral"):
            raise ValueError("Target indices must be a one-dimensional integer array.")
        if bool(xp.any((target_indices < 0) | (target_indices >= target_grid.num_voxels))):
            raise ValueError("Target indices are outside the target grid.")
        self.xp = xp
        self.device = array_api_compat.device(target_indices)
        self.indices = target_indices
        self.chunk_size = chunk_size
        self.source_shape = tuple(reversed(source_grid.dimensions))
        self.target_dimensions = target_grid.dimensions
        self.same_grid = source_grid == target_grid
        source_matrix = source_grid.direction * source_grid.resolution_vector
        target_matrix = target_grid.direction * target_grid.resolution_vector
        inverse = xp.asarray(np.linalg.inv(source_matrix), dtype=xp.float64, device=self.device)
        matrix = xp.asarray(target_matrix, dtype=xp.float64, device=self.device)
        origin = xp.asarray(target_grid.origin, dtype=xp.float64, device=self.device)
        source_origin = xp.asarray(source_grid.origin, dtype=xp.float64, device=self.device)
        nx, ny, nz = self.target_dimensions
        rows = xp.arange(ny * nz, dtype=xp.int64, device=self.device)
        y = xp.astype(rows % ny, xp.float64)[:, None]
        z = xp.astype(rows // ny, xp.float64)[:, None]

        def transform(x):
            # Preserve image-transform and scanline arithmetic order. Collapsing
            # the affine map can change tiny weights at exact voxel boundaries,
            # which matters when neighboring radiological depths are NaN.
            point = origin + x * matrix[None, :, 0]
            point = point + y * matrix[None, :, 1]
            point = point + z * matrix[None, :, 2]
            delta = point - source_origin
            return (
                delta[:, 0:1] * inverse[None, :, 0]
                + delta[:, 1:2] * inverse[None, :, 1]
                + delta[:, 2:3] * inverse[None, :, 2]
            )

        self.start = transform(0.0)
        self.vector = transform(float(nx)) - self.start

    def __call__(self, values):
        """Resample a backend array without constructing or downloading an image."""
        xp = self.xp
        if (
            array_api_compat.array_namespace(values) != xp
            or array_api_compat.device(values) != self.device
        ):
            raise ValueError("Image and target indices must share namespace and device.")
        if values.shape != self.source_shape:
            raise ValueError("Expected an image with the source grid shape.")
        if not _backend_interpolation_supported(
            xp, values.dtype, self.interpolator, self.extrapolate
        ):
            raise ValueError(
                "Unsupported image dtype or extrapolation; use resample_array for native fallback."
            )
        if self.same_grid:
            return xp.take(xp.reshape(values, (-1,)), self.indices, axis=0)
        result = xp.empty(self.indices.shape, dtype=values.dtype, device=self.device)
        nx = self.target_dimensions[0]
        for start in range(0, self.indices.shape[0], self.chunk_size):
            stop = min(start + self.chunk_size, self.indices.shape[0])
            indices = self.indices[start:stop]
            row = indices // nx
            fraction = xp.astype(indices % nx, xp.float64)[:, None] / float(nx)
            queries = xp.take(self.start, row, axis=0) + fraction * xp.take(
                self.vector, row, axis=0
            )
            evaluate = (
                interpolate_image_linear
                if self.interpolator == sitk.sitkLinear
                else interpolate_image_nearest
            )
            result[start:stop] = evaluate(values, queries, extrapolate=self.extrapolate)
        return result
