"""Siddon Ray Tracing Algorithm for Voxelized Geometry."""

import logging
from timeit import default_timer as timer
from typing import Union

import array_api_compat
import numpy as np
import SimpleITK as sitk

from ..core import xp_utils
from ..core.xp_utils.typing import Array, ArrayNamespace
from . import _kernels
from ._base import RayTracerBase

# from ._perf import _fast_compute_all_alphas, _fast_compute_plane_alphas
logger = logging.getLogger(__name__)


class RayTracerSiddon(RayTracerBase):
    """Siddon Ray Tracing Algorithm through voxelized geometry."""

    debug_core_performance: bool
    use_gpu: bool

    def __init__(self, cubes: Union[sitk.Image, list[sitk.Image]]):
        self.debug_core_performance = False
        self.use_gpu = True
        self.device = xp_utils.choose_device()
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

        target_points = xp.asarray(target_points, device=self.device)
        source_points = xp.asarray(source_points, device=self.device)
        isocenter = xp.asarray(isocenter, device=self.device)

        # array_api_compat.size instead of .size: on torch tensors .size is a method
        if (
            array_api_compat.size(target_points) != 3
            or array_api_compat.size(source_points) != 3
            or array_api_compat.size(isocenter) != 3
        ):
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
        alphas, lengths, rho, d12, ix = self._trace_rays_core(
            isocenter, source_points, target_points
        )

        t_finalization_start = timer()
        alphas = xp_utils.to_numpy(alphas)
        lengths = xp_utils.to_numpy(lengths)
        rho = [xp_utils.to_numpy(r) for r in rho]
        d12 = np.atleast_1d(xp_utils.to_numpy(d12).squeeze())
        ix = xp_utils.to_numpy(ix).astype(np.int64, copy=False)

        if self.debug_core_performance:
            # to_numpy synchronizes the device, so host timestamps are accurate here
            logger.debug(f"Trace Ray: finalization: {timer() - t_finalization_start:.4f}s")

        return alphas, lengths, rho, d12, ix

    def _trace_rays_device(
        self,
        isocenter: Union[list, np.ndarray],
        source_points: Union[list, np.ndarray],
        target_points: Union[list, np.ndarray],
    ) -> tuple[Array, list[Array], Array]:
        """Trace rays keeping ``(lengths, rho, ix)`` on the compute device.

        Skips the host conversion of :meth:`trace_rays` (and the unused alphas/d12
        transfers) for consumers like :meth:`trace_cubes` that continue on the device.
        """
        _, lengths, rho, _, ix = self._trace_rays_core(isocenter, source_points, target_points)
        return lengths, rho, ix

    def _trace_rays_core(
        self,
        isocenter: Union[list, np.ndarray],
        source_points: Union[list, np.ndarray],
        target_points: Union[list, np.ndarray],
    ) -> tuple[Array, Array, list[Array], Array, Array]:
        """Run the vectorized Siddon trace, returning backend arrays on the device."""

        # xp = torch
        # xp = array_api_strict
        # xp = np
        xp = xp_utils.choose_array_api_namespace()

        target_points = xp.asarray(target_points, device=self.device)
        source_points = xp.asarray(source_points, device=self.device)
        isocenter = xp.asarray(isocenter, device=self.device)

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
            source_points = xp.broadcast_to(source_points, (num_rays, 3))
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

        if self.debug_core_performance:
            xp_utils.synchronize(xp, s)
            t_allalphas_elapsed = xp_utils.elapsed_time(xp, t_allalphas_start, t_allalphas_end)
            t_indices_elapsed = xp_utils.elapsed_time(xp, t_allalphas_end, t_indices_end)
            logger.debug(
                f"Trace Ray: {num_rays} rays, {num_sources} sources, "
                f"compute_all_alphas: {t_allalphas_elapsed:.4f}s, "
                f"compute_indices: {t_indices_elapsed:.4f}s"
            )

        return alphas, lengths, rho, d12, ix

    def _get_device_arrays(self, xp: ArrayNamespace) -> dict:
        """
        Per-(namespace, device, precision) cache of constant geometry and cube arrays.

        Avoids re-converting the Python-list plane coordinates and re-uploading the
        (potentially large) cubes on every trace call. Invalidated when the cubes
        change (see :meth:`_initialize_geometry`).
        """
        key = (xp.__name__, str(self.device), np.dtype(self.precision).name)
        cached = self._device_cache.get(key)
        if cached is None:
            dtype = self._array_api_precision
            first_planes = [self._x_planes[0], self._y_planes[0], self._z_planes[0]]
            last_planes = [self._x_planes[-1], self._y_planes[-1], self._z_planes[-1]]

            cubes_linear = []
            for cube in self._cubes:
                # SimpleITK exposes a read-only, C-contiguous (z, y, x) view. DLPack cannot
                # export a read-only NumPy array, and Torch would otherwise alias it without
                # write protection.
                cube_np = sitk.GetArrayViewFromImage(cube).ravel()
                if not cube_np.flags.writeable:
                    cube_np = cube_np.copy()
                cubes_linear.append(xp_utils.to_namespace(xp, cube_np, device=self.device))

            cached = {
                "x_planes": xp.asarray(self._x_planes, dtype=dtype, device=self.device),
                "y_planes": xp.asarray(self._y_planes, dtype=dtype, device=self.device),
                "z_planes": xp.asarray(self._z_planes, dtype=dtype, device=self.device),
                "p_min": xp.asarray(first_planes, dtype=dtype, device=self.device),
                "p_max": xp.asarray(last_planes, dtype=dtype, device=self.device),
                "num_planes": xp.asarray(self._num_planes, dtype=dtype, device=self.device),
                "resolution": xp.asarray(self._resolution, dtype=dtype, device=self.device),
                # default dtypes on purpose: these feed float64/int promotion paths
                "cube_origin": xp.asarray(self._cubes[0].GetOrigin(), device=self.device),
                "resolution_default": xp.asarray(self._resolution, device=self.device),
                "cube_dim": xp.asarray(self._cube_dim, device=self.device),
                "plane_ix": xp.arange(max(self._num_planes), dtype=dtype, device=self.device),
                "cubes_linear": cubes_linear,
            }
            self._device_cache[key] = cached
        return cached

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

        dev = self._get_device_arrays(xp)

        # Compute alphas for each plane and merge parametric sets
        s1 = xp_utils.create_stream(xp)
        with s1:
            alpha_x = self._compute_plane_alphas(
                i_min,
                i_max,
                dev["x_planes"],
                self._source_points[:, 0],
                self._ray_vec[:, 0],
            )
        s2 = xp_utils.create_stream(xp)
        with s2:
            alpha_y = self._compute_plane_alphas(
                j_min,
                j_max,
                dev["y_planes"],
                self._source_points[:, 1],
                self._ray_vec[:, 1],
            )
        s3 = xp_utils.create_stream(xp)
        with s3:
            alpha_z = self._compute_plane_alphas(
                k_min,
                k_max,
                dev["z_planes"],
                self._source_points[:, 2],
                self._ray_vec[:, 2],
            )

        # Order the merge after the plane-alpha streams device-side instead of
        # blocking the host with a full synchronize
        for si in (s1, s2, s3):
            xp_utils.stream_wait_event(xp, s, xp_utils.record_event(xp, si))

        t_merge_start = xp_utils.record_event(xp, s)

        alphas = _kernels.merge_sorted_unique(alpha_limits, alpha_x, alpha_y, alpha_z)

        # Size Reduction
        t_size_reduction_start = xp_utils.record_event(xp, s)
        # Slicing with a data-dependent width syncs the host on accelerators, but
        # benchmarks show the reduced downstream width more than pays for the stall
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

        xp = array_api_compat.array_namespace(dim_min, dim_max, planes, source, ray)
        plane_ix = self._get_device_arrays(xp)["plane_ix"]
        return _kernels.compute_plane_alphas(dim_min, dim_max, planes, source, ray, plane_ix)

    def _compute_alpha_limits(self):
        """
        Compute the alpha limits for the ray tracing.

        This is a helper function to compute the alpha limits for the ray tracing.
        It is used in the trace_rays function to compute the alpha limits for each ray.
        """

        # get / validate array namespace
        xp = array_api_compat.array_namespace(self._ray_vec)

        s = xp_utils.get_current_stream(xp)
        t_init_start = xp_utils.record_event(xp, s)

        dev = self._get_device_arrays(xp)
        p_min = dev["p_min"]
        p_max = dev["p_max"]

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

        # The (N, 3) arrays here are too small for stream parallelism to pay off, so
        # everything runs sequentially on the current stream
        alpha_axis_min = xp.min(xp.where(alpha_nans, -xp.inf, alpha_planes), axis=-1)  # (N, 3)
        alpha_min_values = xp.maximum(alpha_axis_min[:, 0], alpha_axis_min[:, 1])
        alpha_min_values = xp.maximum(alpha_min_values, alpha_axis_min[:, 2])
        alpha_min_values = xp.clip(alpha_min_values, 0.0, None)
        alpha_min_values = xp.where(zero_mask, 0.0, alpha_min_values)

        alpha_axis_max = xp.max(xp.where(alpha_nans, -xp.inf, alpha_planes), axis=-1)  # (N, 3)
        alpha_max_values = xp.minimum(alpha_axis_max[:, 0], alpha_axis_max[:, 1])
        alpha_max_values = xp.minimum(alpha_max_values, alpha_axis_max[:, 2])
        alpha_max_values = xp.clip(alpha_max_values, None, 1.0)
        alpha_max_values = xp.where(zero_mask, 1.0, alpha_max_values)

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

        dev = self._get_device_arrays(xp)
        return _kernels.compute_indices_from_alpha(
            self._source_points,
            self._ray_vec,
            alphas_mid,
            dev["cube_origin"],
            dev["resolution_default"],
            dev["cube_dim"],
        )

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

        i = ijk[:, 0, :]
        j = ijk[:, 1, :]
        k = ijk[:, 2, :]

        # Public indices use Fortran order on the (z, y, x) SimpleITK array view. Keep that
        # contract, but gather cube values through the view's native C order to avoid making a
        # complete Fortran-order copy of every cube on every call.
        ix = k + self._cube_dim[2] * j + self._cube_dim[1] * self._cube_dim[2] * i

        ix = xp.where(val_ix, ix, -1)

        cube_ix = i + self._cube_dim[0] * j + self._cube_dim[0] * self._cube_dim[1] * k
        cube_ix = xp.where(val_ix, cube_ix, 0)

        rho = [
            xp.full(val_ix.shape, xp.nan, dtype=self._array_api_precision, device=self.device)
            for _ in self._cubes
        ]
        cubes_linear = self._get_device_arrays(xp)["cubes_linear"]
        for s, cube_linear in enumerate(cubes_linear):
            rho[s] = xp.where(
                val_ix, xp.astype(cube_linear[cube_ix], self._array_api_precision), rho[s]
            )

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

        dev = self._get_device_arrays(xp)
        lower_planes = dev["p_min"]
        upper_planes = dev["p_max"]
        nplanes = dev["num_planes"]
        resolution = dev["resolution"]

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

        self._device_cache = {}
