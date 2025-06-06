"""Siddon Ray Tracing Algorithm for Voxelized Geometry."""

from typing import Union
import time
import logging

import numpy as np
import SimpleITK as sitk

from pyRadPlan.raytracer._base import RayTracerBase

# from ._perf import _fast_compute_all_alphas, _fast_compute_plane_alphas
logger = logging.getLogger(__name__)


class RayTracerSiddon(RayTracerBase):
    """Siddon Ray Tracing Algorithm through voxelized geometry."""

    debug_core_performance: bool

    def __init__(self, cubes: Union[sitk.Image, list[sitk.Image]]):
        self.debug_core_performance = False
        super().__init__(cubes)

    # @jit(nopython=True)
    def trace_ray(
        self,
        isocenter: Union[list, np.ndarray],
        source_points: Union[list, np.ndarray],
        target_points: Union[list, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], np.ndarray, np.ndarray]:
        """Trace an individual ray."""

        # Check type and dimension of target_point (should be nx3)
        # Check type and dimension of source_point (should be nx3)
        # Check type and dimension of cubes (should be either a list of or a single 3D numpy array)
        # Check type and dimension of resolution (should be a dictionary with keys 'x', 'y', 'z'
        # and values as floats)
        isocenter = np.asarray(isocenter)
        source_points = np.asarray(source_points)
        target_points = np.asarray(target_points)

        if target_points.size != 3 or source_points.size != 3 or isocenter.size != 3:
            raise ValueError(
                "Number of target Points and source points needs to be equal to one! If you want "
                "to trace multiple rays at once, use trace_rays instead!"
            )
        alphas, lengths, rho, d12, ix = self.trace_rays(
            isocenter, source_points.reshape((1, 3)), target_points.reshape((1, 3))
        )

        # Squeeze Dimensions
        alphas = alphas.squeeze()
        lengths = lengths.squeeze()
        rho = [r.squeeze() for r in rho]
        ix = ix.squeeze()

        return alphas, lengths, rho, d12, ix

    def trace_rays(
        self,
        isocenter: Union[list, np.ndarray],
        source_points: Union[list, np.ndarray],
        target_points: Union[list, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], np.ndarray, np.ndarray]:
        """
        Vectorized Implementation of RayTracing.

        Uses padding to create matrices of ray information.

        Notes
        -----
        Currently, the vectorized implementation uses padding with NaN values. This is not the most
        efficient way to handle the different lengths of the rays. A more efficient way would be to
        use more performant padding values (e.g. an unrealistically large value like the respective
        maximum floating point value)
        """

        num_rays = target_points.shape[0]
        num_sources = source_points.shape[0]

        if num_sources not in (num_rays, 1):
            raise ValueError(
                f"Number of source points ({num_sources}) needs to be one or equal to number of "
                f"target points ({num_rays})!"
            )
        if num_sources == 1:
            source_points = np.tile(source_points, (num_rays, 1))
            num_sources = num_rays

        self._source_points = (source_points + isocenter).astype(self.precision)
        self._target_points = (target_points + isocenter).astype(self.precision)
        self._ray_vec = self._target_points - self._source_points

        t_allalphas_start = time.perf_counter()
        alphas = self._compute_all_alphas()
        t_allalphas_end = time.perf_counter()
        t_allalphas_elapsed = t_allalphas_end - t_allalphas_start

        d12 = np.linalg.norm(self._ray_vec, axis=1, keepdims=True)
        tmp_diff = np.diff(alphas, axis=1)

        lengths = d12 * tmp_diff
        alphas_mid = alphas[:, :-1] + 0.5 * tmp_diff

        val_ix, ijk = self._compute_indices_from_alpha(alphas_mid)

        t_indices_end = time.perf_counter()
        t_indices_elapsed = t_indices_end - t_allalphas_end

        valid_indices = np.nonzero(val_ix)

        if all(arr.size == 0 for arr in valid_indices):
            alphas = np.empty((num_rays, 0), dtype=self.precision)
            lengths = np.empty((num_rays, 0), dtype=self.precision)
            rho = [np.empty((num_rays, 0), dtype=self.precision) for _ in self._cubes]
            ix = np.empty((num_rays, 0), dtype=np.int64)

        else:
            rho, ix = self._get_rho_and_indices(val_ix, valid_indices, ijk)

        t_finalization_end = time.perf_counter()
        t_finalization_elapsed = t_finalization_end - t_indices_end
        if self.debug_core_performance:
            logger.debug(
                f"Trace Ray: {num_rays} rays, {num_sources} sources, "
                f"compute_all_alphas: {t_allalphas_elapsed:.4f}s, "
                f"compute_indices: {t_indices_elapsed:.4f}s, "
                f"finalization: {t_finalization_elapsed:.4f}s"
            )

        return alphas, lengths, rho, d12, ix

    def _compute_all_alphas(self) -> np.ndarray:
        """
        Compute all rays' alpha values (length to plane intersections).

        Here we setup grids to enable logical indexing when computing
        the alphas along each dimension. All alphas between the
        minimum and maximum index will be computed, with additional
        exclusion of singular plane occurrences (max == min)
        All values out of scope will be set to NaN.
        """

        alpha_limits = self._compute_alpha_limits()

        i_min, i_max, j_min, j_max, k_min, k_max = self._compute_entry_and_exit(alpha_limits)

        # Compute alphas for each plane and merge parametric sets

        alpha_x = self._compute_plane_alphas(
            i_min,
            i_max,
            self._x_planes,
            self._source_points[:, 0],
            self._ray_vec[:, 0],
        )
        alpha_y = self._compute_plane_alphas(
            j_min,
            j_max,
            self._y_planes,
            self._source_points[:, 1],
            self._ray_vec[:, 1],
        )
        alpha_z = self._compute_plane_alphas(
            k_min,
            k_max,
            self._z_planes,
            self._source_points[:, 2],
            self._ray_vec[:, 2],
        )

        alphas = np.concatenate((alpha_limits, alpha_x, alpha_y, alpha_z), axis=1)

        # Vectorized unique operation across rows
        # Sort alphas row-wise and remove duplicates
        alphas.sort(axis=1)  # Sort each row ascendingly
        mask = np.diff(alphas, axis=1) == 0  # Identify duplicates
        alphas[:, 1:][mask] = np.nan  # Replace duplicates with NaN
        alphas.sort(axis=1)

        # Alternative for loop to the last sorting operation in O(n)
        # Could be numba'd to squeeze out more performance
        # for row in alphas:
        #     mask = ~np.isnan(row)
        #     k = np.sum(mask)  # how many non-NaN in this row
        #     row[:k] = row[mask]      # move them to the front
        #     row[k:] = np.nan         # fill the rest with NaN

        # Size Reduction
        max_num_columns = np.max(np.sum(~np.isnan(alphas), axis=1))
        alphas = alphas[:, :max_num_columns]

        return alphas

    def _compute_plane_alphas(
        self,
        dim_min: np.ndarray,
        dim_max: np.ndarray,
        planes: np.ndarray,
        source: np.ndarray,
        ray: np.ndarray,
    ) -> np.ndarray:
        """
        Compute the alphas for a given plane.

        Parameters
        ----------
        dim_min : np.ndarray
            The minimum dimension of the plane.
        dim_max : np.ndarray
            The maximum dimension of the plane.
        planes : np.ndarray
            The planes to compute the alphas for.
        source : np.ndarray
            The source points.
        ray : np.ndarray
            The ray vectors.

        Returns
        -------
        alphas : np.ndarray
            The computed alphas for the given plane.
        """

        # ensure 1-D

        # 1) make a (1, P) index row, and compare to (N, 1) dim_min/max → (N, P) mask
        plane_ix = np.arange(planes.shape[0])[None, :]  # shape (1, P)
        low = plane_ix < dim_min[:, None]  # before entry
        high = plane_ix > dim_max[:, None]  # after exit
        deg = (plane_ix == dim_min[:, None]) & (plane_ix == dim_max[:, None])
        nanm = np.isnan(dim_min)[:, None] | np.isnan(dim_max)[:, None]

        mask_invalid = low | high | deg | nanm  # shape (N, P)

        # 2) compute all alphas in one shot (broadcasted): (planes[None,:] - source[:,None]) / ray[:,None]
        #    guard against divide-by-zero warnings if you like via errstate or inv_ray trick
        with np.errstate(divide="ignore", invalid="ignore"):
            alphas = (planes[None, :] - source[:, None]) / ray[:, None]

        # 3) mask out invalid entries
        alphas[mask_invalid] = np.nan

        return alphas

    def _compute_alpha_limits(self):
        """
        Compute the alpha limits for the ray tracing.

        This is a helper function to compute the alpha limits for the ray tracing.
        It is used in the trace_rays function to compute the alpha limits for each ray.
        """

        # Draft for faster alpha calculation
        # # plane limits
        p_min = np.asarray(
            [self._x_planes[0], self._y_planes[0], self._z_planes[0]], dtype=self.precision
        )
        p_max = np.asarray(
            [self._x_planes[-1], self._y_planes[-1], self._z_planes[-1]], dtype=self.precision
        )

        # 1) raw alpha to the two planes per axis, shape (N, 3, 2)

        with np.errstate(divide="ignore", invalid="ignore"):
            alpha_planes = np.stack(
                (
                    (p_min - self._source_points) / self._ray_vec,  # alpha to "near" plane
                    (p_max - self._source_points) / self._ray_vec,
                ),  # alpha to "far"  plane
                axis=-1,
            )

        # 2) final [alpha_min, alpha_max] per axis with ray_vec != 0 mask
        alpha_axis_min = np.nanmin(alpha_planes, axis=-1)  # (N, 3)
        alpha_axis_max = np.nanmax(alpha_planes, axis=-1)  # (N, 3)

        zero_mask = (self._ray_vec == 0.0).all(axis=1)  # (N,)
        alpha_axis_min[zero_mask] = 0.0
        alpha_axis_max[zero_mask] = 1.0

        alpha_min_values = np.maximum(0.0, np.nanmax(alpha_axis_min, axis=1))  # (N,)
        alpha_max_values = np.minimum(1.0, np.nanmin(alpha_axis_max, axis=1))  # (N,)

        alpha_limits = np.stack((alpha_min_values, alpha_max_values), axis=1)  # (N, 2)

        return alpha_limits

    def _compute_indices_from_alpha(self, alphas_mid: np.ndarray):
        cube_origin = np.asarray(self._cubes[0].GetOrigin())

        # Compute coordinates
        sp_scaled = (self._source_points - cube_origin) / self._resolution
        rv_scaled = self._ray_vec / self._resolution

        ijk = sp_scaled[:, :, None] + rv_scaled[:, :, None] * alphas_mid[:, None, :]
        ijk[~np.isfinite(ijk)] = -1.0

        # Round in place
        np.round(ijk, out=ijk)
        ijk = ijk.astype(np.int32, copy=False)

        cube_dim_brd = self._cube_dim[None, :, None]
        val_ix = ((ijk >= 0) & (ijk < cube_dim_brd)).all(axis=1)

        return val_ix, ijk

    def _get_rho_and_indices(self, val_ix: np.ndarray, valid_indices: np.ndarray, ijk: np.ndarray):
        """
        Finalize the output of densities and indices.

        Returns
        -------
        rho : list[np.ndarray]
            The rho values for each cube.
        ix : np.ndarray
            The indices within the cubes.
        """
        stride_j = self._cube_dim[2]
        stride_i = self._cube_dim[1] * self._cube_dim[2]

        ix = ijk[:, 2, :].astype(np.int64, copy=True)
        ix += stride_j * ijk[:, 1, :]
        ix += stride_i * ijk[:, 0, :]

        ix[~val_ix] = -1

        rho = [np.full(val_ix.shape, np.nan, dtype=self.precision) for _ in self._cubes]
        for s, cube in enumerate(self._cubes):
            # Views SimpleITKs image buffer as a numpy array, preserving dimension ordering of
            # sitk
            cube_linear = sitk.GetArrayViewFromImage(cube).ravel(order="F")
            rho[s][valid_indices] = cube_linear[ix[valid_indices]]

        return rho, ix

    def _compute_entry_and_exit(self, alpha_limits: np.ndarray):
        """
        Compute the entry and exit points for the ray tracing.

        This is a helper function to compute the entry and exit points for the ray tracing.
        It is used in the trace_rays function to compute the entry and exit points for each ray.
        """

        ray_direction_positive = self._ray_vec > 0
        alpha_limits_reverse = alpha_limits[:, ::-1]

        alpha_axis = np.where(
            ray_direction_positive[:, :, None],
            alpha_limits[:, None, :],
            alpha_limits_reverse[:, None, :],
        )

        lower_planes = np.array(
            [self._x_planes[0], self._y_planes[0], self._z_planes[0]], dtype=self.precision
        )
        upper_planes = np.array(
            [self._x_planes[-1], self._y_planes[-1], self._z_planes[-1]], dtype=self.precision
        )

        nplanes = np.asarray(self._num_planes, dtype=self.precision)

        dim_min = (
            nplanes[None, :]
            - (upper_planes - alpha_axis[:, :, 0] * self._ray_vec - self._source_points)
            / self._resolution[None, :]
            - 1
        )
        dim_max = (
            self._source_points + alpha_axis[:, :, 1] * self._ray_vec - lower_planes
        ) / self._resolution[None, :]

        # Rounding
        dim_min = np.ceil(np.round(1000 * dim_min) / 1000)
        dim_max = np.floor(np.round(1000 * dim_max) / 1000)

        # unpack the dimensions to i, j, k
        i_min, j_min, k_min = dim_min.T
        i_max, j_max, k_max = dim_max.T

        return i_min, i_max, j_min, j_max, k_min, k_max

    def _initialize_geometry(self):
        """
        Initialize the geometry for the ray tracing.

        Notes
        -----
        For a detailed description of the variables, see Siddon 1985 Medical Physics.
        """

        ref_cube = self._cubes[0]

        if ref_cube.GetDimension() != 3:
            raise ValueError("Only 3D cubes are supported by RayTracerSiddon!")

        origin = np.asarray(ref_cube.GetOrigin()).astype(self.precision)
        self._resolution = np.asarray(ref_cube.GetSpacing()).astype(self.precision)
        direction = np.asarray(ref_cube.GetDirection()).reshape(3, 3).astype(self.precision)
        self._cube_dim = np.asarray(ref_cube.GetSize())

        increment = np.zeros_like(origin)
        increment[0] = (direction @ np.array([1, 0, 0], dtype=self.precision))[
            0
        ] * self._resolution[0]
        increment[1] = (direction @ np.array([0, 1, 0], dtype=self.precision))[
            1
        ] * self._resolution[1]
        increment[2] = (direction @ np.array([0, 0, 1], dtype=self.precision))[
            2
        ] * self._resolution[2]

        self._x_planes = (
            origin[0]
            + (np.arange(self._cube_dim[0] + 1, dtype=self.precision) - 0.5) * increment[0]
        )
        self._y_planes = (
            origin[1]
            + (np.arange(self._cube_dim[1] + 1, dtype=self.precision) - 0.5) * increment[1]
        )
        self._z_planes = (
            origin[2]
            + (np.arange(self._cube_dim[2] + 1, dtype=self.precision) - 0.5) * increment[2]
        )

        self._num_planes = [len(self._x_planes), len(self._y_planes), len(self._z_planes)]
