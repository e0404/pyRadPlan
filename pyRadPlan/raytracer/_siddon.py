"""Siddon Ray Tracing Algorithm for Voxelized Geometry."""

from typing import Union

import logging
from contextlib import nullcontext

import numpy as np
import SimpleITK as sitk
import array_api_compat

from ..core.xp_utils.typing import Array, ArrayNamespace

from ._base import RayTracerBase
from ..core import xp_utils


# from ._perf import _fast_compute_all_alphas, _fast_compute_plane_alphas
logger = logging.getLogger(__name__)


class RayTracerSiddon(RayTracerBase):
    """Siddon Ray Tracing Algorithm through voxelized geometry."""

    debug_core_performance: bool
    use_gpu: bool

    def __init__(self, cubes: Union[sitk.Image, list[sitk.Image]]):
        self.debug_core_performance = False
        self.use_gpu = True
        super().__init__(cubes)

    # @jit(nopython=True)
    def trace_ray(
        self,
        isocenter: Union[list, np.ndarray],
        source_points: Union[list, np.ndarray],
        target_points: Union[list, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], np.ndarray, np.ndarray]:
        """Trace an individual ray."""

        xp = xp_utils.choose_array_api_namespace()

        target_points = xp.asarray(target_points)
        source_points = xp.asarray(source_points)
        isocenter = xp.asarray(isocenter)

        if target_points.size != 3 or source_points.size != 3 or isocenter.size != 3:
            raise ValueError(
                "Number of target Points and source points needs to be equal to one! If you want "
                "to trace multiple rays at once, use trace_rays instead!"
            )
        alphas, lengths, rho, d12, ix = self.trace_rays(
            isocenter, xp.reshape(source_points, (1, 3)), xp.reshape(target_points, (1, 3))
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

        # xp = torch
        # xp = array_api_strict
        # xp = np
        xp = xp_utils.choose_array_api_namespace()

        target_points = xp.asarray(target_points)
        source_points = xp.asarray(source_points)
        isocenter = xp.asarray(isocenter)

        self._array_api_precision = getattr(xp, np.dtype(self.precision).name)

        xp: ArrayNamespace = array_api_compat.array_namespace(
            isocenter, source_points, target_points
        )

        num_rays = target_points.shape[0]
        num_sources = source_points.shape[0]

        if num_sources not in (num_rays, 1):
            raise ValueError(
                f"Number of source points ({num_sources}) needs to be one or equal to number of "
                f"target points ({num_rays})!"
            )
        if num_sources == 1:
            source_points = xp.tile(source_points, (num_rays, 1))
            num_sources = num_rays

        self._source_points = xp.astype(source_points + isocenter, self._array_api_precision)
        self._target_points = xp.astype(target_points + isocenter, self._array_api_precision)
        self._ray_vec = self._target_points - self._source_points

        s = xp_utils.get_current_stream(xp)

        t_allalphas_start = xp_utils.record_event(xp, s)
        alphas = self._compute_all_alphas()
        t_allalphas_end = xp_utils.record_event(xp, s)

        if hasattr(xp, "linalg"):
            d12 = xp.linalg.vector_norm(self._ray_vec, axis=1, keepdims=True)
        else:
            d12 = xp.sqrt(xp.sum(self._ray_vec**2, axis=1, keepdims=True))

        tmp_diff = xp.diff(alphas, axis=1)

        lengths = d12 * tmp_diff
        alphas_mid = alphas[:, :-1] + 0.5 * tmp_diff

        val_ix, ijk = self._compute_indices_from_alpha(alphas_mid)

        t_indices_end = xp_utils.record_event(xp, s)

        if xp.count_nonzero(val_ix) == 0:
            alphas = xp.empty((num_rays, 0), dtype=self._array_api_precision)
            lengths = xp.empty((num_rays, 0), dtype=self._array_api_precision)
            rho = [xp.empty((num_rays, 0), dtype=self._array_api_precision) for _ in self._cubes]
            ix = xp.empty((num_rays, 0), dtype=xp.int64)

        else:
            rho, ix = self._get_rho_and_indices(val_ix, ijk)

        alphas = xp_utils.to_numpy(alphas)
        lengths = xp_utils.to_numpy(lengths)
        rho = [xp_utils.to_numpy(r) for r in rho]
        d12 = np.atleast_1d(xp_utils.to_numpy(d12).squeeze())
        ix = xp_utils.to_numpy(ix)

        t_finalization_end = xp_utils.record_event(xp, s)
        xp_utils.synchronize(xp, s)
        t_allalphas_elapsed = xp_utils.elapsed_time(xp, t_allalphas_start, t_allalphas_end)
        t_indices_elapsed = xp_utils.elapsed_time(xp, t_allalphas_end, t_indices_end)
        t_finalization_elapsed = xp_utils.elapsed_time(xp, t_indices_end, t_finalization_end)
        if self.debug_core_performance:
            logger.debug(
                f"Trace Ray: {num_rays} rays, {num_sources} sources, "
                f"compute_all_alphas: {t_allalphas_elapsed:.4f}s, "
                f"compute_indices: {t_indices_elapsed:.4f}s, "
                f"finalization: {t_finalization_elapsed:.4f}s"
            )

        return alphas, lengths, rho, d12, ix

    def _compute_all_alphas(self) -> Array:
        """
        Compute all rays' alpha values (length to plane intersections).

        Here we setup grids to enable logical indexing when computing
        the alphas along each dimension. All alphas between the
        minimum and maximum index will be computed, with additional
        exclusion of singular plane occurrences (max == min)
        All values out of scope will be set to NaN.
        """
        xp: ArrayNamespace = array_api_compat.array_namespace(self._source_points, self._ray_vec)

        s = xp_utils.get_current_stream(xp)

        t_limits_start = xp_utils.record_event(xp, s)

        alpha_limits = self._compute_alpha_limits()

        t_entry_exit_start = xp_utils.record_event(xp, s)

        i_min, i_max, j_min, j_max, k_min, k_max = self._compute_entry_and_exit(alpha_limits)

        t_planes_start = xp_utils.record_event(xp, s)

        # Compute alphas for each plane and merge parametric sets
        s1 = xp_utils.create_stream(xp)
        with s1:
            x_planes = xp.asarray(self._x_planes, dtype=self._array_api_precision)
            alpha_x = self._compute_plane_alphas(
                i_min,
                i_max,
                x_planes,
                self._source_points[:, 0],
                self._ray_vec[:, 0],
            )
        s2 = xp_utils.create_stream(xp)
        with s2:
            y_planes = xp.asarray(self._y_planes, dtype=self._array_api_precision)
            alpha_y = self._compute_plane_alphas(
                j_min,
                j_max,
                y_planes,
                self._source_points[:, 1],
                self._ray_vec[:, 1],
            )
        s3 = xp_utils.create_stream(xp)
        with s3:
            z_planes = xp.asarray(self._z_planes, dtype=self._array_api_precision)
            alpha_z = self._compute_plane_alphas(
                k_min,
                k_max,
                z_planes,
                self._source_points[:, 2],
                self._ray_vec[:, 2],
            )

        xp_utils.synchronize(xp)  # Synchronize Device

        t_merge_start = xp_utils.record_event(xp, s)

        alphas = xp.concat((alpha_limits, alpha_x, alpha_y, alpha_z), axis=1)

        # Vectorized unique operation across rows
        # Sort alphas row-wise and remove duplicates
        alphas = xp.sort(alphas, axis=1)
        # alphas.sort(axis=1)  # Sort each row ascendingly
        mask = (
            xp.diff(
                alphas, axis=1, prepend=xp.full((alphas.shape[0], 1), xp.inf, dtype=alphas.dtype)
            )
            == 0
        )  # Identify duplicates
        alphas = xp.sort(xp.where(mask, xp.inf, alphas), axis=1)  # Set duplicates to NaN
        # alphas.sort(axis=1)

        # Size Reduction
        t_size_reduction_start = xp_utils.record_event(xp, s)
        max_num_columns = xp.max(xp.count_nonzero(xp.isfinite(alphas), axis=1))
        alphas = alphas[:, :max_num_columns]

        t_end = xp_utils.record_event(xp, s)

        if self.debug_core_performance:
            xp_utils.synchronize(xp, s)
            logger.debug(
                f"  compute_alpha_limits: {xp_utils.elapsed_time(xp, t_limits_start, t_entry_exit_start):.4f}s, "
                f"compute_entry_exit: {xp_utils.elapsed_time(xp, t_entry_exit_start, t_planes_start):.4f}s, "
                f"compute_plane_alphas: {xp_utils.elapsed_time(xp, t_planes_start, t_merge_start):.4f}s, "
                f"merge: {xp_utils.elapsed_time(xp, t_merge_start, t_size_reduction_start):.4f}s, "
                f"size_reduction: {xp_utils.elapsed_time(xp, t_size_reduction_start, t_end):.4f}s"
            )
        return alphas

    def _compute_plane_alphas(
        self,
        dim_min: Array,
        dim_max: Array,
        planes: Array,
        source: Array,
        ray: Array,
    ) -> Array:
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

        # get / validate array namespace
        xp = array_api_compat.array_namespace(dim_min, dim_max, planes, source, ray)

        # ensure 1-D
        # 1) make a (1, P) index row, and compare to (N, 1) dim_min/max → (N, P) mask
        plane_ix = xp.arange(planes.shape[0], dtype=self._array_api_precision)[
            None, :
        ]  # shape (1, P)
        low = plane_ix < dim_min[:, None]  # before entry
        high = plane_ix > dim_max[:, None]  # after exit
        deg = (plane_ix == dim_min[:, None]) & (plane_ix == dim_max[:, None])
        nanm = xp.isnan(dim_min)[:, None] | xp.isnan(dim_max)[:, None]

        mask_invalid = low | high | deg | nanm  # shape (N, P)

        # 2) compute all alphas in one shot (broadcasted): (planes[None,:] - source[:,None]) / ray[:,None]
        #    guard against divide-by-zero warnings if you like via errstate or inv_ray trick
        if array_api_compat.is_numpy_array(ray):
            errstate = np.errstate(divide="ignore", invalid="ignore")
        else:
            errstate = nullcontext()
        with errstate:
            alphas = (planes[None, :] - source[:, None]) / ray[:, None]

        # 3) mask out invalid entries
        alphas[mask_invalid] = xp.nan

        return alphas

    def _compute_alpha_limits(self):
        """
        Compute the alpha limits for the ray tracing.

        This is a helper function to compute the alpha limits for the ray tracing.
        It is used in the trace_rays function to compute the alpha limits for each ray.
        """

        # get / validate array namespace
        xp = array_api_compat.array_namespace(self._ray_vec, api_version="2024.12")

        s = xp_utils.get_current_stream(xp)
        t_init_start = xp_utils.record_event(xp, s)

        # Draft for faster alpha calculation
        # # plane limits
        p_min = xp.asarray(
            [self._x_planes[0], self._y_planes[0], self._z_planes[0]],
            dtype=self._array_api_precision,
        )
        p_max = xp.asarray(
            [self._x_planes[-1], self._y_planes[-1], self._z_planes[-1]],
            dtype=self._array_api_precision,
        )

        # 1) raw alpha to the two planes per axis, shape (N, 3, 2)
        alpha_planes = xp.stack(
            (
                (p_min - self._source_points) / self._ray_vec,  # alpha to "near" plane
                (p_max - self._source_points) / self._ray_vec,
            ),  # alpha to "far"  plane
            axis=-1,
        )
        alpha_nans = xp.isnan(alpha_planes)

        t_mask_start = xp_utils.record_event(xp, s)
        # zero_mask = cp.all(self._ray_vec == 0.0,axis=1)  # (N,)
        zero_mask = xp.max(xp.abs(self._ray_vec), axis=1) <= 0.0  # (N,)
        t_mask_end = xp_utils.record_event(xp, s)

        # alpha_min_values = cp.maximum(0.0, cp.max(alpha_axis_min, axis=1))  # (N,)
        # alpha_max_values = cp.minimum(1.0, cp.min(alpha_axis_max, axis=1))  # (N,)

        # alpha_min_values = cp.where(zero_mask, 0.0, alpha_min_values)
        # alpha_max_values = cp.where(zero_mask, 1.0, alpha_max_values)

        xp_utils.synchronize(xp, s)

        s1 = xp_utils.create_stream(xp)
        with s1:
            alpha_axis_min = xp.min(xp.where(alpha_nans, -xp.inf, alpha_planes), axis=-1)  # (N, 3)
            alpha_min_values = xp.maximum(alpha_axis_min[:, 0], alpha_axis_min[:, 1])
            alpha_min_values = xp.maximum(alpha_min_values, alpha_axis_min[:, 2])
            alpha_min_values = xp.clip(alpha_min_values, 0.0, None)
            alpha_min_values = xp.where(zero_mask, 0.0, alpha_min_values)

        s2 = xp_utils.create_stream(xp)
        with s2:
            alpha_axis_max = xp.max(xp.where(alpha_nans, -xp.inf, alpha_planes), axis=-1)  # (N, 3)
            alpha_max_values = xp.minimum(alpha_axis_max[:, 0], alpha_axis_max[:, 1])
            alpha_max_values = xp.minimum(alpha_max_values, alpha_axis_max[:, 2])
            alpha_max_values = xp.clip(alpha_max_values, None, 1.0)
            alpha_max_values = xp.where(zero_mask, 1.0, alpha_max_values)

        xp_utils.synchronize(xp)

        # pytorch does not support maximum/minimum with scalar and array
        # alpha_min_values = xp.maximum(alpha_min_values, 0.0)
        # alpha_max_values = xp.minimum(alpha_max_values, 1.0)

        t_final_limits_end = xp_utils.record_event(xp, s)

        alpha_limits = xp.stack((alpha_min_values, alpha_max_values), axis=1)  # (N, 2)

        t_end = xp_utils.record_event(xp, s)

        if self.debug_core_performance:
            xp_utils.synchronize(xp, s)
            logger.debug(
                f"    init: {xp_utils.elapsed_time(xp, t_init_start, t_mask_start):.4f}s, "
                f"mask: {xp_utils.elapsed_time(xp, t_mask_start, t_mask_end):.4f}s, "
                f"minmax: {xp_utils.elapsed_time(xp, t_mask_end, t_final_limits_end):.4f}s"
                f"finalize: {xp_utils.elapsed_time(xp, t_final_limits_end, t_end):.4f}s"
            )

        return alpha_limits

    def _compute_indices_from_alpha(self, alphas_mid: Array):
        xp = array_api_compat.array_namespace(alphas_mid)

        cube_origin = xp.asarray(self._cubes[0].GetOrigin())

        res = xp.asarray(self._resolution)

        # Compute coordinates
        sp_scaled = (self._source_points - cube_origin) / res
        rv_scaled = self._ray_vec / res

        ijk = sp_scaled[:, :, None] + rv_scaled[:, :, None] * alphas_mid[:, None, :]
        ijk[~xp.isfinite(ijk)] = -1.0

        # Round in place
        ijk = xp.astype(xp.round(ijk), xp.int32)

        cube_dim_brd = xp.asarray(self._cube_dim)[None, :, None]
        val_ix = xp.all((ijk >= 0) & (ijk < cube_dim_brd), axis=1)

        return val_ix, ijk

    def _get_rho_and_indices(self, val_ix: Array, ijk: Array):
        """
        Finalize the output of densities and indices.

        Returns
        -------
        rho : list[np.ndarray]
            The rho values for each cube.
        ix : np.ndarray
            The indices within the cubes.
        """

        xp = array_api_compat.array_namespace(val_ix, ijk)

        stride_j = self._cube_dim[2]
        stride_i = self._cube_dim[1] * self._cube_dim[2]

        ix = xp.astype(ijk[:, 2, :], xp.int64)
        ix += stride_j * ijk[:, 1, :]
        ix += stride_i * ijk[:, 0, :]

        ix[~val_ix] = -1

        rho = [xp.full(val_ix.shape, xp.nan, dtype=self._array_api_precision) for _ in self._cubes]
        for s, cube in enumerate(self._cubes):
            # Views SimpleITKs image buffer as a numpy array, preserving dimension ordering of
            # sitk
            cube_linear = xp_utils.from_numpy(
                xp, sitk.GetArrayViewFromImage(cube).ravel(order="F")
            )
            # cube_values = cube_linear[ix[val_ix]]
            # cube_mask   = xp.arange(len(cube_linear),ix.dtype)[None,:] == ix[val_ix][:,None]

            # rho[s] = xp.where(val_ix,cube_linear[ix[val_ix]],rho[s])

            rho[s][val_ix] = xp.astype(cube_linear[ix[val_ix]], self._array_api_precision)

        return rho, ix

    def _compute_entry_and_exit(self, alpha_limits: Array):
        """
        Compute the entry and exit points for the ray tracing.

        This is a helper function to compute the entry and exit points for the ray tracing.
        It is used in the trace_rays function to compute the entry and exit points for each ray.
        """

        xp = array_api_compat.array_namespace(self._ray_vec, alpha_limits)

        ray_direction_positive = self._ray_vec > 0

        # alpha_limits_reverse = alpha_limits[:, ::-1]
        alpha_limits_reverse = xp.flip(alpha_limits, axis=1)

        alpha_axis = xp.where(
            ray_direction_positive[:, :, None],
            alpha_limits[:, None, :],
            alpha_limits_reverse[:, None, :],
        )

        lower_planes = xp.asarray(
            (self._x_planes[0], self._y_planes[0], self._z_planes[0]),
            dtype=self._array_api_precision,
        )
        upper_planes = xp.asarray(
            (self._x_planes[-1], self._y_planes[-1], self._z_planes[-1]),
            dtype=self._array_api_precision,
        )

        nplanes = xp.asarray(self._num_planes, dtype=self._array_api_precision)
        resolution = xp.asarray(self._resolution, dtype=self._array_api_precision)

        dim_min = (
            nplanes[None, :]
            - (upper_planes - alpha_axis[:, :, 0] * self._ray_vec - self._source_points)
            / resolution[None, :]
            - 1
        )
        dim_max = (
            self._source_points + alpha_axis[:, :, 1] * self._ray_vec - lower_planes
        ) / resolution[None, :]

        # Rounding
        dim_min = xp.ceil(xp.round(1000 * dim_min) / 1000)
        dim_max = xp.floor(xp.round(1000 * dim_max) / 1000)

        # unpack the dimensions to i, j, k
        i_min = dim_min[:, 0]
        j_min = dim_min[:, 1]
        k_min = dim_min[:, 2]
        i_max = dim_max[:, 0]
        j_max = dim_max[:, 1]
        k_max = dim_max[:, 2]

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
        self._resolution = np.asarray(ref_cube.GetSpacing()).astype(self.precision).tolist()
        direction = (
            np.asarray(ref_cube.GetDirection()).reshape(3, 3).astype(self.precision).tolist()
        )
        self._cube_dim = np.asarray(ref_cube.GetSize()).tolist()

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
        ).tolist()
        self._y_planes = (
            origin[1]
            + (np.arange(self._cube_dim[1] + 1, dtype=self.precision) - 0.5) * increment[1]
        ).tolist()
        self._z_planes = (
            origin[2]
            + (np.arange(self._cube_dim[2] + 1, dtype=self.precision) - 0.5) * increment[2]
        ).tolist()

        self._num_planes = [len(self._x_planes), len(self._y_planes), len(self._z_planes)]
