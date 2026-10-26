"""Prepared backend-resident resampling between physical image grids."""

import array_api_compat
import numpy as np

from ._grids import Grid
from .xp_utils._image_interpolation import interpolate_image_linear


class ArrayBackendGridResampler:
    """Prepare linear image resampling, optionally only at selected target voxels.

    Parameters
    ----------
    source_grid, target_grid : Grid
        Physical image geometry. Only 3D scalar images are supported.
    target_indices : Array
        Flat C-order indices in the target (z, y, x) volume. Their namespace and
        device determine execution placement. Use all target indices for a full
        volume, or a selection to avoid resampling unused voxels.
    chunk_size : int, optional
        Maximum query count per interpolation call, bounding temporary storage.

    Notes
    -----
    Retains selected indices and scanline endpoint maps, not full coordinate grids
    or eight neighbor arrays. Grid metadata is snapshotted at creation. Values
    must use the same namespace/device. Do not mutate retained target indices.
    Output is a 1D array in selection order.
    Image boundaries use nearest-pixel extrapolation, matching the dose engine's
    SimpleITK linear resampling. Float64 coordinate arithmetic preserves physical
    geometry independently of the image value precision.
    """

    def __init__(self, source_grid: Grid, target_grid: Grid, target_indices, *, chunk_size=262144):
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
        if values.shape != self.source_shape or not xp.isdtype(values.dtype, "real floating"):
            raise ValueError("Expected a floating image with the source grid shape.")
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
            result[start:stop] = interpolate_image_linear(values, queries)
        return result
