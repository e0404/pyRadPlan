"""Interface for voxel geometry ray tracers."""

from abc import ABC, abstractmethod
from typing import Union, Any
import logging
import time

import numpy as np
import SimpleITK as sitk

from pyRadPlan.core.np2sitk import linear_indices_to_image_coordinates
from pyRadPlan.geometry import lps
from pyRadPlan.stf._beam import Beam

from ._perf import fast_spatial_circle_lookup

logger = logging.getLogger(__name__)


class RayTracerBase(ABC):
    """Base class for all ray tracers."""

    lateral_cut_off: float
    precision: np.dtype

    @property
    def cubes(self):
        """CT or other abritrary cubes of similar resolution to be traced."""
        return self._cubes

    @cubes.setter
    def cubes(self, cubes: Union[sitk.Image, list[sitk.Image]]):
        if not isinstance(cubes, list):
            cubes = [cubes]
        self._cubes = cubes
        self._initialize_geometry()
        self._coords = None

    def __init__(self, cubes: Union[sitk.Image, list[sitk.Image]]):
        self.lateral_cut_off = 50.0
        self.precision = np.float32
        self.cubes = cubes
        self._coords = None

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
        The default implementation loops over the trace_ray function. The separte implementation is
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
        if self._coords is None:
            # Obtain coordinates
            cube_ix = np.arange(self.cubes[0].GetNumberOfPixels(), dtype=np.int64)
            self._coords = linear_indices_to_image_coordinates(
                cube_ix, self.cubes[0], index_type="sitk", dtype=self.precision
            )

        # obtain rotation matrix
        rot_mat = lps.get_beam_rotation_matrix(beam.gantry_angle, beam.couch_angle)

        # rotate coordinates
        coords = (self._coords - beam.iso_center) @ rot_mat - beam.source_point_bev
        t_trace_end = time.perf_counter()
        logger.debug("took %s seconds!", t_trace_end - t_trace_start)

        # central_ray_vector = np.array(iso_center) - np.array(source_point).reshape
        logger.debug("Setting up Ray matrix...")
        t_trace_start = time.perf_counter()

        ray_spacing = np.min(self._resolution) / np.sqrt(2.0, dtype=self.precision)
        ray_matrix_bev_y = (
            np.max(coords[:, 1]) + np.max(self._resolution) + beam.source_point_bev[1]
        )
        ray_matrix_scale = 1 + ray_matrix_bev_y / beam.sad

        # num_candidate_rays = 2 * np.ceil(500.0 / ray_spacing).astype(np.int64) + 1

        spacing_range = ray_spacing * np.arange(
            np.floor(-500.0 / ray_spacing), np.ceil(500.0 / ray_spacing) + 1, dtype=self.precision
        )
        candidate_ray_coords_x, candidate_ray_coords_z = np.meshgrid(spacing_range, spacing_range)

        # If we have reference positions, we use them to restrict the raytracing region
        reference_positions_bev = ray_matrix_scale * np.array(
            [ray.ray_pos_bev for ray in beam.rays]
        )

        # use a precompiled numba function to speed up the spatial lookup
        candidate_ray_mx = fast_spatial_circle_lookup(
            candidate_ray_coords_x,
            candidate_ray_coords_z,
            reference_positions_bev,
            self.lateral_cut_off,
        )

        # candidate_ray_mx = np.full(candidate_ray_coords_x.shape, False, dtype=np.bool)

        # for i in range(reference_positions_bev.shape[0]):
        #     notix = (candidate_ray_coords_x - reference_positions_bev[i, 0]) ** 2 + (
        #         candidate_ray_coords_z - reference_positions_bev[i, 2]
        #     ) ** 2 <= self.lateral_cut_off**2
        #     candidate_ray_mx[notix] = True

        ray_matrix_bev = np.hstack(
            (
                candidate_ray_coords_x[candidate_ray_mx].reshape(-1, 1),
                ray_matrix_bev_y
                * np.ones(np.sum(candidate_ray_mx), dtype=self.precision).reshape(-1, 1),
                candidate_ray_coords_z[candidate_ray_mx].reshape(-1, 1),
            )
        )

        ray_matrix_lps = ray_matrix_bev @ rot_mat.T

        t_trace_end = time.perf_counter()
        logger.debug("took %s seconds!", t_trace_end - t_trace_start)

        logger.debug("Tracing %d rays through the cubes", np.count_nonzero(candidate_ray_mx))

        t_trace_start = time.perf_counter()
        _, lengths, rho, d12, ix = self.trace_rays(
            beam.iso_center.reshape(1, 3), beam.source_point.reshape(1, 3), ray_matrix_lps
        )
        t_trace_end = time.perf_counter()

        logger.debug("Cube ray tracing took %s seconds...", t_trace_end - t_trace_start)

        # Now we compute which rays will respectively give the voxel value for radiological depth
        valid_ix = np.isfinite(ix)

        scale_factor = np.zeros_like(ix, dtype=self.precision)
        scale_factor[valid_ix] = (ray_matrix_bev_y + beam.sad) / coords[ix[valid_ix], 1]

        x_dist = np.full_like(ix, np.nan, dtype=self.precision)
        z_dist = np.full_like(ix, np.nan, dtype=self.precision)

        x_dist[valid_ix] = coords[ix[valid_ix], 0] * scale_factor[valid_ix]
        x_dist = x_dist - ray_matrix_bev[:, 0, np.newaxis]

        z_dist[valid_ix] = coords[ix[valid_ix], 2] * scale_factor[valid_ix]
        z_dist = z_dist - ray_matrix_bev[:, 2, np.newaxis]

        ray_selection = ray_spacing / 2.0

        ix_remember_from_tracing = (
            (x_dist > -ray_selection)
            & (x_dist <= ray_selection)
            & (z_dist > -ray_selection)
            & (z_dist <= ray_selection)
        )
        t_remember_end = time.perf_counter()

        logger.debug(
            "Found %d ray indices for radiological depth calculation (took %s seconds)",
            np.count_nonzero(ix_remember_from_tracing),
            t_remember_end - t_trace_end,
        )
        rad_depth_cubes = [
            np.nan * np.ones_like(sitk.GetArrayViewFromImage(cube), dtype=self.precision)
            for cube in self.cubes
        ]

        for i, cube in enumerate(rad_depth_cubes):
            rel_depths = lengths * rho[i]
            rel_depths = np.cumsum(rel_depths, axis=1) - rel_depths / 2.0

            try:
                ix_assign = np.unravel_index(ix[ix_remember_from_tracing], cube.shape, order="F")
            except (ValueError, IndexError):
                logger.error(
                    "Error in unraveling indices from raytracing. Trying to recover...",
                    exc_info=True,
                )
                tmp_ix = ix[ix_remember_from_tracing]
                rel_depths = rel_depths[ix_remember_from_tracing]

                wrong_values = np.logical_or(tmp_ix < 0, tmp_ix >= cube.size)
                tmp_ix = tmp_ix[~wrong_values]
                rel_depths = rel_depths[~wrong_values]

                # Remove the wrong values
                ix_assign = np.unravel_index(tmp_ix, cube.shape, order="F")
                logger.info(
                    "Recovered %d indices for radiological depth cube",
                    np.count_nonzero(wrong_values),
                )
                cube[ix_assign] = rel_depths
            else:
                cube[ix_assign] = rel_depths[ix_remember_from_tracing]

            rad_depth_cubes[i] = sitk.GetImageFromArray(cube)
            rad_depth_cubes[i].CopyInformation(self.cubes[i])

        t_createcubes_end = time.perf_counter()

        logger.debug(
            "Radiological depth cube filling took %s seconds",
            t_createcubes_end - t_remember_end,
        )

        return rad_depth_cubes
        # scale_factor[valid_ix] = lengths[valid_ix] / d12[valid_ix]

    @abstractmethod
    def _initialize_geometry(self):
        """
        Initialize geometry of the ray tracer.

        Will be automatically called when the cubes are set.
        """
