"""Interface for voxel geometry ray tracers."""

from abc import ABC, abstractmethod
from typing import Union, Any, Optional
import logging
import time

import numpy as np
import SimpleITK as sitk
import array_api_compat

from ..core import xp_utils
from ..core.xp_utils.typing import Array
from . import _kernels
from ..core.np2sitk import linear_indices_to_image_coordinates
from ..geometry import lps
from ..stf._beam import Beam

logger = logging.getLogger(__name__)


class RayTracerBase(ABC):
    """Base class for all ray tracers."""

    lateral_cut_off: float
    precision: np.dtype
    fixed_ray_spacing_range: Optional[float]

    @property
    def cubes(self):
        """CT or other arbitrary cubes of similar resolution to be traced."""
        return self._cubes

    @cubes.setter
    def cubes(self, cubes: Union[sitk.Image, list[sitk.Image]]):
        if not isinstance(cubes, list):
            cubes = [cubes]
        self._cubes = cubes
        self._initialize_geometry()

    def __init__(self, cubes: Union[sitk.Image, list[sitk.Image]]):
        self.lateral_cut_off = 50.0
        self.precision = np.float32
        self.fixed_ray_spacing_length = None
        self.cubes = cubes

    def trace_rays(
        self,
        isocenter: Union[list, np.ndarray],
        source_points: Union[list, np.ndarray],
        target_points: Union[list, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], np.ndarray, np.ndarray]:
        """
        Trace multiple rays through a cube.

        Parameters
        ----------
        isocenter : Union[list, np.ndarray]
            Isocenter coordinates (1x3) array or list
        source_points : Union[list, np.ndarray]
            Source points coordinates. (nx3) array or list
        target_points : Union[list, np.ndarray]
            Target points coordinates. (nx3) array or list

        Returns
        -------
        alphas : ndarray
            Array of alpha values for each ray
        lengths : ndarray
            Array of lengths for each ray
        rho : list[ndarray]
            Array of rho values for each ray and each cube
        d12 : ndarray
            Array of full length of each ray
        ix : ndarray
            Linear indices (in numpy ordering) of the voxels intersected by each ray

        Notes
        -----
        The default implementation loops over the trace_ray function. The separate implementation is
        here to enable more performant implementations for specific ray tracers, e.g. through
        vectorization.
        """

        # Assuming size function equivalent is numpy's shape attribute.
        num_rays = target_points.shape[0]
        num_sources = source_points.shape[0]

        if num_sources not in (num_rays, 1):
            # MatRad_Config.instance() and dispError equivalent in Python needs handling.
            raise (
                f"Number of source points ({num_sources}) needs to be one "
                f"or equal to number of target points ({num_rays})!"
            )
        if num_sources == 1:
            source_points = np.tile(source_points, (num_rays, 1))
            num_sources = num_rays

        alphas, lengths, rho, d12, ix = [], [], [], [], []
        for r in range(num_rays):
            alpha, l_val, rho_val, d12_val, ix_val = self.trace_ray(
                isocenter, source_points[r, :], target_points[r, :]
            )
            alphas.append(alpha)
            lengths.append(l_val)
            rho.append(rho_val)
            d12.append(d12_val)
            ix.append(ix_val)

        # Padding with NaN values
        maxnumval = max(len(x) for x in ix)

        def nanpad(x):
            return np.pad(x, (0, maxnumval - len(x)), constant_values=np.nan)

        alphas = [nanpad(alpha) for alpha in alphas]
        lengths = [nanpad(l_val) for l_val in lengths]
        ix = [nanpad(ix_val) for ix_val in ix]

        for c in range(len(self.cubes)):
            rho[c] = [nanpad(rho_val) for rho_val in rho[c]]

        return np.array(alphas), np.array(lengths), rho, np.array(d12), np.array(ix)

    def _trace_rays_device(
        self,
        isocenter: Union[list, np.ndarray],
        source_points: Union[list, np.ndarray],
        target_points: Union[list, np.ndarray],
    ) -> tuple[Array, list[Array], Array]:
        """Trace rays returning ``(lengths, rho, ix)`` on the compute backend and device.

        Default implementation wraps :meth:`trace_rays` and uploads its numpy outputs.
        Subclasses that already compute on a device can override this to skip the
        host round trip (and the conversion of the unused alphas and d12).
        """
        _, lengths, rho, _, ix = self.trace_rays(isocenter, source_points, target_points)

        xp = xp_utils.choose_array_api_namespace()
        device = getattr(self, "device", None)
        ix = xp_utils.to_namespace(xp, ix, device=device)
        lengths = xp_utils.to_namespace(xp, lengths, device=device)
        rho = [xp_utils.to_namespace(xp, r, device=device) for r in rho]
        return lengths, rho, ix

    @abstractmethod
    def trace_ray(
        self,
        isocenter: Union[list, np.ndarray],
        source_points: Union[list, np.ndarray],
        target_points: Union[list, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], np.ndarray, np.ndarray]:
        """
        Trace a single ray through cubes.

        Abstract Method to be implemented in subclasses.
        """

    def trace_cubes(self, beam: Union[dict[str, Any], Beam]) -> list[sitk.Image]:
        """
        Automatically calculate depth by tracing rays through cubes.

        Set up ray matrix with appropriate spacing to trace through
        all cubes, resulting in a cumulative sum of values in every voxel
        relative to the source. Will calculate cumulative sum on all of
        the supplied images.
        """

        if not isinstance(beam, Beam):
            beam = Beam.model_validate(beam)

        t_trace_start = time.perf_counter()
        logger.debug("Computing coordinates...")

        # obtain rotation matrix
        rot_mat = lps.get_beam_rotation_matrix(beam.gantry_angle, beam.couch_angle)

        def rotate_to_bev(voxel_linear_ix):
            coords = linear_indices_to_image_coordinates(
                voxel_linear_ix, self.cubes[0], index_type="sitk", dtype=self.precision
            )
            return (coords - beam.iso_center) @ rot_mat - beam.source_point_bev

        # The BEV y extent is the maximum of an affine map over the voxel-center box,
        # which is attained at one of the 8 corner voxels — no need to rotate all voxels
        nx, ny, nz = self.cubes[0].GetSize()
        # "sitk" linear index layout is z-fastest: z + nz * (y + ny * x)
        corner_lin_ix = np.array(
            [k + nz * (j + ny * i) for i in (0, nx - 1) for j in (0, ny - 1) for k in (0, nz - 1)],
            dtype=np.int64,
        )
        corners_bev = rotate_to_bev(corner_lin_ix)
        t_trace_end = time.perf_counter()
        logger.debug("took %s seconds!", t_trace_end - t_trace_start)

        # central_ray_vector = np.array(iso_center) - np.array(source_point).reshape
        logger.debug("Setting up Ray matrix...")
        t_trace_start = time.perf_counter()

        ray_spacing = np.min(self._resolution) / np.sqrt(2.0, dtype=self.precision)
        ray_matrix_bev_y = (
            np.max(corners_bev[:, 1]) + np.max(self._resolution) + beam.source_point_bev[1]
        )
        ray_matrix_scale = 1 + ray_matrix_bev_y / beam.sad

        # If we have reference positions, we use them to restrict the raytracing region
        reference_positions_bev = ray_matrix_scale * np.array(
            [ray.ray_pos_bev for ray in beam.rays]
        )

        if self.fixed_ray_spacing_length is not None:
            ray_extent = self.fixed_ray_spacing_length
        else:
            # look at max ray_positions in bev and add lateral cutoff
            ray_extent = 2.0 * (
                np.max(np.abs(reference_positions_bev[:, [0, 2]])) + self.lateral_cut_off
            )

        spacing_range = ray_spacing * np.arange(
            np.floor(-ray_extent / ray_spacing),
            np.ceil(ray_extent / ray_spacing) + 1,
            dtype=self.precision,
        )

        candidate_ray_mx = self._get_candidate_ray_matrix(spacing_range, reference_positions_bev)

        ray_idx_z, ray_idx_x = np.nonzero(candidate_ray_mx)

        ray_matrix_bev = np.column_stack(
            (
                spacing_range[ray_idx_x],
                np.full(ray_idx_x.shape[0], ray_matrix_bev_y, dtype=self.precision),
                spacing_range[ray_idx_z],
            )
        )

        if xp_utils.openblas_has_gemm_race():
            # elementwise fallback: this OpenBLAS corrupts tall-skinny (N, 3) @ (3, 3)
            ray_matrix_lps = (
                ray_matrix_bev[:, 0:1] * rot_mat[None, :, 0]
                + ray_matrix_bev[:, 1:2] * rot_mat[None, :, 1]
                + ray_matrix_bev[:, 2:3] * rot_mat[None, :, 2]
            )
        else:
            ray_matrix_lps = ray_matrix_bev @ rot_mat.T

        t_trace_end = time.perf_counter()
        logger.debug("took %s seconds!", t_trace_end - t_trace_start)

        logger.debug("Tracing %d rays through the cubes", np.count_nonzero(candidate_ray_mx))

        t_trace_start = time.perf_counter()
        lengths, rho, ix = self._trace_rays_device(
            beam.iso_center.reshape(1, 3), beam.source_point.reshape(1, 3), ray_matrix_lps
        )
        t_trace_end = time.perf_counter()

        logger.debug("Cube ray tracing took %s seconds...", t_trace_end - t_trace_start)

        xp = array_api_compat.array_namespace(ix)

        # Now we compute which rays will respectively give the voxel value for radiological depth
        ix_remember_from_tracing = self._select_rad_depth_segments(
            beam, ix, ray_matrix_bev, ray_matrix_bev_y, ray_spacing
        )
        t_remember_end = time.perf_counter()

        if logger.isEnabledFor(logging.DEBUG):
            # guarded: int() of the device count would sync the host even with logging off
            logger.debug(
                "Found %d ray indices for radiological depth calculation (took %s seconds)",
                int(xp.count_nonzero(ix_remember_from_tracing)),
                t_remember_end - t_trace_end,
            )
        rad_depth_cubes = self._fill_rad_depth_cubes(lengths, rho, ix, ix_remember_from_tracing)

        t_createcubes_end = time.perf_counter()

        logger.debug(
            "Radiological depth cube filling took %s seconds",
            t_createcubes_end - t_remember_end,
        )

        return rad_depth_cubes
        # scale_factor[valid_ix] = lengths[valid_ix] / d12[valid_ix]

    def _select_rad_depth_segments(
        self,
        beam: Beam,
        ix: Array,
        ray_matrix_bev: np.ndarray,
        ray_matrix_bev_y: float,
        ray_spacing: float,
    ) -> Array:
        """Mark the traced segments whose voxel receives its radiological depth value.

        A segment is selected when its voxel center, projected to the ray-matrix plane,
        falls within half a ray spacing of the segment's ray position. Runs on the
        backend and device of ``ix``.
        """
        rot_mat = lps.get_beam_rotation_matrix(beam.gantry_angle, beam.couch_angle)
        num_voxels = self.cubes[0].GetNumberOfPixels()
        xp = array_api_compat.array_namespace(ix)
        device = array_api_compat.device(ix)

        # Index decode is linear and the image and beam transforms are affine, so
        # voxel index -> BEV coordinates is a single affine map in working precision
        direction = np.asarray(self.cubes[0].GetDirection()).reshape(3, 3)
        index_to_bev = ((direction * np.asarray(self._resolution)).T @ rot_mat).astype(
            self.precision
        )
        bev_offset = (
            (np.asarray(self.cubes[0].GetOrigin()) - beam.iso_center) @ rot_mat
            - beam.source_point_bev
        ).astype(self.precision)

        _, ny, nz = self.cubes[0].GetSize()
        ray_matrix_bev = xp.asarray(ray_matrix_bev, device=device)

        return _kernels.select_rad_depth_segments(
            ix,
            xp.asarray(index_to_bev, device=device),
            xp.asarray(bev_offset, device=device),
            ray_matrix_bev[:, 0],
            ray_matrix_bev[:, 2],
            float(ray_matrix_bev_y + beam.sad),
            # builtin float: array-api-strict rejects numpy scalar operands
            float(ray_spacing) / 2.0,
            num_voxels,
            ny,
            nz,
        )

    def _fill_rad_depth_cubes(
        self,
        lengths: Array,
        rho: list[Array],
        ix: Array,
        ix_remember_from_tracing: Array,
    ) -> list[sitk.Image]:
        """Accumulate segment depths and scatter them into radiological depth cubes.

        Runs on the backend and device of the inputs; only the finished cubes are
        transferred back to the host.
        """
        xp = array_api_compat.array_namespace(ix)
        device = array_api_compat.device(ix)
        precision = getattr(xp, np.dtype(self.precision).name)

        nx, ny, nz = self.cubes[0].GetSize()
        # The selection mask is only set at indices already validated against the cube
        # bounds, so no out-of-range recovery is needed. Public "sitk" linear indices
        # are z-fastest (Fortran order); convert to C-order offsets on the (z, y, x)
        # array view for the flat scatter
        selected_ix = ix[ix_remember_from_tracing]
        z_ix = selected_ix % nz
        tmp = selected_ix // nz
        flat_c_ix = (z_ix * ny + tmp % ny) * nx + tmp // ny

        rad_depth_cubes = []
        for cube, rho_cube in zip(self.cubes, rho):
            segment_depths = lengths * rho_cube
            # Replace NaN with 0 before cumsum to prevent a single invalid voxel
            # np.cumsum([[0.5, 0.33, NaN, 0.18, 0.22]]) -> [0.5, 0.83, NaN, NaN, NaN], which is bad
            segment_depths = xp.where(xp.isfinite(segment_depths), segment_depths, 0.0)
            rel_depths = xp.cumulative_sum(segment_depths, axis=1) - segment_depths / 2.0

            flat_cube = xp.full((nx * ny * nz,), xp.nan, dtype=precision, device=device)
            flat_cube = xp_utils.scatter(
                flat_cube, flat_c_ix, rel_depths[ix_remember_from_tracing]
            )
            cube_np = np.reshape(xp_utils.to_numpy(flat_cube), (nz, ny, nx))

            rad_depth_cube = sitk.GetImageFromArray(cube_np)
            rad_depth_cube.CopyInformation(cube)
            rad_depth_cubes.append(rad_depth_cube)

        return rad_depth_cubes

    def _get_candidate_ray_matrix(self, spacing_range: Array, ref_pos_bev: Array) -> Array:
        """Get candidate ray matrix for given ray spacing and reference positions."""

        xp = array_api_compat.array_namespace(spacing_range, ref_pos_bev)

        # The candidate matrix is a union of discs, and per grid
        # row z each disc covers one contiguous x-interval, so membership reduces to
        # interval-stabbing: count starts/ends up to each grid point via sort + searchsorted
        # in O((N*M + N^2) log(N*M)) instead of evaluating all N^2 * M pairwise distances
        n = array_api_compat.size(spacing_range)
        device = array_api_compat.device(spacing_range)

        # interval bounds in float64 so boundary membership matches the distance test
        grid = xp.astype(spacing_range, xp.float64)
        ref_x = xp.astype(ref_pos_bev[:, 0], xp.float64)[:, None]
        ref_z = xp.astype(ref_pos_bev[:, 2], xp.float64)[:, None]

        half_width_sq = float(self.lateral_cut_off) ** 2 - (grid[None, :] - ref_z) ** 2  # (M, N)
        # keep only the (disc, row) pairs the disc actually reaches
        inside = xp.reshape(half_width_sq >= 0, (-1,))
        half_width = xp.sqrt(xp.reshape(half_width_sq, (-1,))[inside])
        center = xp.reshape(xp.broadcast_to(ref_x, half_width_sq.shape), (-1,))[inside]

        lo_ix = xp.searchsorted(grid, center - half_width, side="left")
        hi_ix = xp.searchsorted(grid, center + half_width, side="right")

        # separate the per-row intervals by a row offset so one flat sorted array serves
        # all rows: full rows of earlier offsets cancel in the start/end count difference
        row_offset = xp.arange(n, device=device) * (n + 1)
        interval_offset = xp.reshape(
            xp.broadcast_to(row_offset[None, :], half_width_sq.shape), (-1,)
        )[inside]
        starts = xp.sort(lo_ix + interval_offset)
        ends = xp.sort(hi_ix + interval_offset)

        queries = xp.reshape(row_offset[:, None] + xp.arange(n, device=device)[None, :], (-1,))
        coverage = xp.searchsorted(starts, queries, side="right") - xp.searchsorted(
            ends, queries, side="right"
        )

        return xp.reshape(coverage > 0, (n, n))

    @abstractmethod
    def _initialize_geometry(self):
        """
        Initialize geometry of the ray tracer.

        Will be automatically called when the cubes are set.
        """
