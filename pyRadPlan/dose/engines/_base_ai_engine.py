"""dummy engine for ai models"""

import logging
from typing import ClassVar


from typing import Any

import SimpleITK as sitk
import numpy as np


import array_api_compat
from ...core.xp_utils.typing import Array

from pyRadPlan.ct import CT
from pyRadPlan.cst import StructureSet
from pyRadPlan.stf import SteeringInformation
from pyRadPlan.core import Grid
from pyRadPlan.core.np2sitk import linear_indices_to_grid_coordinates
from pyRadPlan.core.xp_utils.compat import interp3d, array_meshgrid
from pyRadPlan.core.xp_utils import to_namespace

from ._base_pencilbeam import PencilBeamEngineAbstract

from pyRadPlan.dose.ai_models._base_ai_models import get_model

logger = logging.getLogger(__name__)

USE_BATCH = True
USE_BATCH_INTERP = True


class AIEngine(PencilBeamEngineAbstract):
    """AI-driven pencil-beam dose engine.

    What it does
    - Builds a beamlet-aligned regular sampling grid (depth, lateral, lateral) for each ray.
    - Samples CT on that grid and feeds it, along with the bixel energy, into an AI model
        to predict a local dose cuboid. This is done as a batch per ray.
    - Maps world dose-grid voxel centers into the beamlet-local frame and interpolates the
        predicted cuboid at those points.
    - Emits sparse bixel contributions (linear indices + `physical_dose`) that the base
        pencil-beam engine accumulates into the dose influence matrix.

    How it works
    - Coordinate frames: world → BEV → beamlet-local via rotation/translation from the beam.
    - Grid construction: uniform spacing in depth and lateral with configurable ranges; axis
        order is (depth, lateral, lateral) to match model conventions.
    - Model inference: plug-in architecture via `get_model`; backend-agnostic arrays using
        the Array API (`self.xp`).
    - Interpolation: 3D linear interpolation over (depth, lateral, lateral) axes to evaluate
        predicted dose at world dose-grid locations.
    - Scenarios: geometry and CT sampling are done per-ray; scenario adaptation follows the
        base class hooks if needed.

    Notes
    -----
    - Physics-specific behavior is delegated to the selected models `_predict_dose`.
    - Interop with SimpleITK/NumPy occurs where required; otherwise operations follow
        the configured array backend (`self.xp`).
    """

    short_name = "AI"
    name = "AI"
    possible_radiation_modes = ["protons", "VHEE"]
    model_name: ClassVar[str] = None

    def __init__(self, pln=None):
        """Initialize the AI engine.

        Parameters
        ----------
        pln : Any, optional
            Plan-like object providing engine configuration (e.g., selected model via
            `prop_dose_calc`), by default ``None``. If provided, engine properties are
            derived from it and the AI model is selected accordingly at init time.

        Returns
        -------
        None
        """
        super().__init__(pln=None)
        self.pln = pln

    def _compute_bixel(self, ray: dict, k: int) -> dict:
        """Compute dose contribution for a single bixel.

        Method
        - Predicts a local dose cuboid for the current beamlet using the AI model.
        - Builds a regular-grid interpolator in beamlet-local coordinates with axes
          ordered as (depth, lateral, lateral) == (y, x, z).
        - Maps the world dose-grid coordinates into the beamlet-local frame and
          interpolates only the inside-of-cuboid points.
        - Returns a sparse-like bixel payload with linear indices and non-zero dose.

        Parameters
        ----------
        ray : dict
            Ray dictionary produced during beam initialization containing geometry,
            grid construction metadata, and per-ray beamlets.
        k : int
            Index of the beamlet (bixel) within ``ray['beamlets']``.

        Returns
        -------
        dict
            Bixel dictionary augmented with
            - `ix` (Array[int64]): linear voxel indices in the dose grid where dose is non-zero,
            - `physical_dose` (Array[float32]): corresponding dose values,
            along with bookkeeping fields (beam/ray/bixel indices, energy indices, etc.).
        """
        # Use the engine's array backend consistently
        xp = self.xp

        # Initialize bixel metadata and compute the beamlet-local cuboid values
        bixel = self._init_bixel(ray, k)

        if USE_BATCH and getattr(self.model, "HAS_BATCHED_FORWARD", False):
            precomp = ray.get("_precomputed_bixels")
            if precomp is not None:
                # res = precomp[k]
                # bixel_out = bixel.copy()
                bixel["ix"] = precomp[k]["ix"]
                bixel["physical_dose"] = precomp[k]["physical_dose"]
                precomp[k] = None

                if all(x is None for x in precomp):
                    del ray["_precomputed_bixels"]
                    del ray["ct_cuboid"]
                    self.grid_information = None
                return bixel

        cuboid = self._calc_particle_bixel(ray["ct_cuboid"], bixel["energy"])

        depth = self.grid_information["depth_range"]
        lateral = self.grid_information["lateral_range"]

        interpolator = interp3d(
            (depth, lateral, lateral),
            cuboid,
            method="linear",
            bounds_error=False,
            fill_value=0.0,
        )

        local_points_masked = self.grid_information["local_points_masked"]
        mask = self.grid_information["ix_mask"]

        pts_eval = xp.stack(
            (local_points_masked[:, 1], local_points_masked[:, 0], local_points_masked[:, 2]),
            axis=1,
        )

        vals_in = xp.asarray(interpolator(pts_eval), dtype=xp.float32, device=self.device)

        vdose_sub = self._vdose_grid_mask[mask]
        vals_in = xp.where(vdose_sub, vals_in, 0.0)

        # keep only non zero
        nz_mask = vals_in > 0
        if xp.any(nz_mask):
            bixel["ix"] = mask[nz_mask]
            bixel["physical_dose"] = vals_in[nz_mask]
        else:
            bixel["ix"] = xp.asarray([])
            bixel["physical_dose"] = xp.asarray([])

        return bixel

    def _calc_particle_bixel(self, ct_cuboid, energy: float):
        """Predict a beamlet-local dose cuboid using the AI model.

        Parameters
        ----------
        ct_cuboid : Array
            CT sampling values in the beamlet-local grid with axes ordered as
            (depth, lateral, lateral).
        energy : float
            Nominal beamlet energy (in machine units), used by the model.

        Returns
        -------
        Array
            Predicted dose cuboid on the same regular grid and axis order as the
            provided ``ct_cuboid`` (depth, lateral, lateral).
        """
        cuboid = self.model._predict_dose(ct_cuboid, energy, self.grid_information)

        return cuboid

    def _init_bixel(self, curr_ray: dict, k: int) -> dict:
        """Initialize bixel bookkeeping and energy indexing.

        Parameters
        ----------
        curr_ray : dict
            Ray dictionary with entries such as ``beamlets``, energy, MU constraints, etc.
        k : int
            Index of the bixel inside ``curr_ray['beamlets']``.

        Returns
        -------
        dict
            The bixel dictionary augmented with indices (beam/ray/bixel), MU limits,
            particles-per-MU, and the machine energy index used for lookups.
        """
        bixel = curr_ray["beamlets"][k]
        bixel["beam_index"] = curr_ray["beam_index"]
        bixel["ray_index"] = curr_ray["ray_index"]
        bixel["bixel_index"] = k

        bixel["min_mu"] = 0
        if "min_mu" in curr_ray:
            bixel["min_mu"] = curr_ray["min_mu"][k]

        bixel["max_mu"] = float("inf")
        if "max_mu" in curr_ray:
            bixel["max_mu"] = curr_ray["max_mu"][k]

        bixel["num_particles_per_mu"] = 1e6
        if "num_particles_per_mu" in curr_ray:
            bixel["num_particles_per_mu"] = curr_ray["num_particles_per_mu"][k]

        # Find energy index in base data
        energy = curr_ray["beamlets"][k]["energy"]
        energy_ix = self._machine.get_energy_index(energy, 4)

        if energy_ix.size > 1:
            raise ValueError("Multiple energies found in base data for one bixel!")

        xp = self.xp
        energy_ix = xp.asarray(energy_ix, dtype=xp.int64, device=self.device)

        bixel["energy_ix"] = energy_ix

        return bixel

    def _extract_ct_cuboid(self, ray: dict[str, Any], beam_info) -> None:
        """Sample CT on the beamlet-local grid constructed for this ray.

        Method
        - Constructs a regular local grid (depth, lateral, lateral) aligned with the current
          beamlet geometry and transforms it to world coordinates.
        - Interpolates the CT on this transformed grid.
        - Stores the resulting cuboid into ``ray['ct_cuboid']`` for downstream model use.

        Parameters
        ----------
        ray : dict[str, Any]
            Ray dictionary to be augmented with the sampled CT cuboid.
        beam_info : dict
            Per-beam precomputed geometry and ray-tracing information used to build the grid.

        Returns
        -------
        None
            The function mutates ``ray`` in place by inserting ``ray['ct_cuboid']`` and
            updates ``self.grid_information`` with the constructed grids and transforms.
        """

        grid_information = self._construct_grid(ray, beam_info)

        interpolated_values_ct = self._extract_sitk_grid_and_interpolate(
            grid_information, ct_interpolation=True
        )
        ray["ct_cuboid"] = interpolated_values_ct

    def _construct_grid(self, ray: dict, beam_info: dict) -> dict:
        """Construct a beamlet-local regular sampling grid and transforms.

        Parameters
        ----------
        ray : dict
            Current ray dictionary containing rotation matrices, BEV positions and isocenter.
        beam_info : dict
            Beam-level geometry and traced coordinate arrays (e.g., BEV coordinates).

        Returns
        -------
        dict
            A dictionary with
            - `lateral_range` (Array[float]): lateral coordinates used for X/Z axes,
            - `depth_range` (Array[float]): longitudinal coordinates (Y axis),
            - `spacing` (float): grid spacing used for all axes,
            - `grid` (Array[N,3]): beamlet-local grid points, (x=lateral, y=depth, z=lateral),
            - `grid_rotated` (Array[N,3]): corresponding world coordinates,
            - `R_world_to_BEV`, `R_BEV_to_beamlet` (Array[3,3]): transforms between frames,
            - `iso_center`, `source_point_bev` (Array[3]): translation references.
        """
        rot_mat = ray["rotation_mat"]

        rot_coords_bev = beam_info["bev_coords"]
        source_point_bev = ray["source_point_bev"]
        target_point_bev = ray["target_point_bev"]
        # sad = ray["sad"]
        rad_depth_mask = beam_info["valid_coords_all"]

        rot_coords_bev = to_namespace(self.xp, rot_coords_bev, device=self.device)
        source_point_bev = to_namespace(self.xp, source_point_bev, device=self.device)
        target_point_bev = to_namespace(self.xp, target_point_bev, device=self.device)
        rad_depth_mask = to_namespace(self.xp, rad_depth_mask, device=self.device)
        xp = self.xp
        # Load grid parameters from model config

        grid_params = self.model_config["grid_parameters"]
        spacing = grid_params["spacing"]
        lateral_range = grid_params["lateral_range"]
        range_coord = lateral_range / spacing
        depth_offset_start = grid_params["depth_offset_start"]
        depth_range = grid_params["depth_range"]

        # coordinates in bev
        y_start = depth_offset_start
        y_end = depth_offset_start + depth_range

        lateral_range = xp.arange(
            -range_coord + spacing / 2,
            range_coord + spacing / 2,
            spacing,
            dtype=xp.float32,
            device=self.device,
        )
        depth_range = xp.arange(y_start, y_end, spacing, dtype=xp.float32, device=self.device)

        depth_vals, lateral_x_vals, lateral_z_vals = array_meshgrid(
            depth_range, lateral_range, lateral_range, indexing="ij"
        )

        # Assign world coordinates: x=lateral_x, y=depth, z=lateral_z
        x = lateral_x_vals
        y = depth_vals
        z = lateral_z_vals

        grid = xp.stack(
            (xp.reshape(x, (-1,)), xp.reshape(y, (-1,)), xp.reshape(z, (-1,))),
            axis=-1,
        )

        if hasattr(xp, "linalg"):
            norm = xp.linalg.vector_norm
            cross = xp.linalg.cross
        else:

            def norm(arr: Array) -> Array:
                return xp.sqrt(xp.sum(arr**2, axis=-1))

            def cross(x: Array, y: Array) -> Array:
                return xp.stack(
                    [
                        x[..., 1] * y[..., 2] - x[..., 2] * y[..., 1],
                        x[..., 2] * y[..., 0] - x[..., 0] * y[..., 2],
                        x[..., 0] * y[..., 1] - x[..., 1] * y[..., 0],
                    ],
                    axis=-1,
                )

        # We later need a copy of the index, and we do it here to do some
        # compatibility management with the array API standard
        # ix = xp.asarray(rad_depth_mask, copy=True)
        rad_depth_ix = xp.nonzero(rad_depth_mask)[0]

        # Make sure that the source-point is a 2D array
        source_point_bev = xp.reshape(source_point_bev, (3,))
        target_point_bev = xp.reshape(target_point_bev, (3,))

        # Put [0 0 0] position in the source point for beamlet who passes through isocenter
        a = -source_point_bev

        # Normalize the vector
        a /= norm(a)

        # Put [0 0 0] position in the source point for a single beamlet
        b = target_point_bev - source_point_bev

        # Normalize the vector
        b /= norm(b)

        # Define rotation matrix
        rot_coords_temp = xp.take(rot_coords_bev, rad_depth_ix, axis=0)
        if not xp.all(a == b):
            # Cross product
            cross_ab = cross(a, b)

            if self._eps_ijk is None:
                self._eps_ijk = xp.asarray(
                    [
                        [[0, 0, 0], [0, 0, -1], [0, 1, 0]],
                        [[0, 0, 1], [0, 0, 0], [-1, 0, 0]],
                        [[0, -1, 0], [1, 0, 0], [0, 0, 0]],
                    ],
                    dtype=cross_ab.dtype,
                    device=self.device,
                )

            ssc_matrix = xp.tensordot(cross_ab, self._eps_ijk, axes=1)

            derived_rot_mat = (
                xp.eye(3, dtype=a.dtype, device=self.device)
                + ssc_matrix
                + ssc_matrix @ ssc_matrix * (1 - xp.vecdot(a, b)) / (norm(cross_ab) ** 2)
            )
        else:
            derived_rot_mat = xp.eye(3, dtype=a.dtype, device=self.device)

        # Compose local-beamlet -> BEV -> world transform so that (x=0,z=0) lies on the ray
        R_beamlet_to_BEV = (derived_rot_mat).T
        # inverse of BEV->beamlet
        R_BEV_to_world = (rot_mat).T
        # translation terms
        iso_center = to_namespace(self.xp, ray["iso_center"], device=self.device)
        # 1) local (beamlet) -> BEV (relative to iso) via rotation
        grid_bev_rel_to_src = (grid - source_point_bev) @ R_beamlet_to_BEV
        # 2) shift from source-relative to BEV absolute by adding source_point_bev
        grid_bev = grid_bev_rel_to_src + source_point_bev
        # 3) BEV -> world, then add iso_center
        grid_rotated = grid_bev @ R_BEV_to_world + iso_center

        # We reduce computation by first computing a bounding box of the beamlet in world coordinate,
        # then filtering world points before the full transformation.

        # Define 8 corners of the cuboid in local beamlet frame
        lx_min, lx_max = lateral_range[0] - spacing / 2, lateral_range[-1] + spacing / 2
        ly_min, ly_max = depth_range[0] - spacing / 2, depth_range[-1] + spacing / 2
        lz_min, lz_max = lateral_range[0] - spacing / 2, lateral_range[-1] + spacing / 2

        corners_local = xp.asarray(
            [
                [lx_min, ly_min, lz_min],
                [lx_min, ly_min, lz_max],
                [lx_min, ly_max, lz_min],
                [lx_min, ly_max, lz_max],
                [lx_max, ly_min, lz_min],
                [lx_max, ly_min, lz_max],
                [lx_max, ly_max, lz_min],
                [lx_max, ly_max, lz_max],
            ],
            dtype=lateral_range.dtype,
            device=self.device,
        )

        # Transform corners to world
        corners_bev_rel = (corners_local - source_point_bev) @ R_beamlet_to_BEV
        corners_world = (corners_bev_rel + source_point_bev) @ R_BEV_to_world + iso_center

        # Find world bounding box
        # Add a small buffer
        # TODO: check buffer size
        min_world = xp.min(corners_world, axis=0) - 2 * spacing
        max_world = xp.max(corners_world, axis=0) + 2 * spacing

        # Filter world coordinates
        bbox_mask = (
            (self.coords_all[:, 0] >= min_world[0])
            & (self.coords_all[:, 0] <= max_world[0])
            & (self.coords_all[:, 1] >= min_world[1])
            & (self.coords_all[:, 1] <= max_world[1])
            & (self.coords_all[:, 2] >= min_world[2])
            & (self.coords_all[:, 2] <= max_world[2])
        )

        candidate_indices = xp.nonzero(bbox_mask)[0]
        candidate_points = self.coords_all[candidate_indices]

        # Use candidate points for precise transformation and check
        local_points_subset = (
            (candidate_points - iso_center) @ rot_mat - source_point_bev
        ) @ derived_rot_mat + source_point_bev

        # Compute inside mask
        mask_subset = (
            (local_points_subset[:, 1] >= ly_min)
            & (local_points_subset[:, 1] <= ly_max)
            & (local_points_subset[:, 0] >= lx_min)
            & (local_points_subset[:, 0] <= lx_max)
            & (local_points_subset[:, 2] >= lz_min)
            & (local_points_subset[:, 2] <= lz_max)
        )

        valid_subset_indices = xp.nonzero(mask_subset)[0]

        ix_mask = candidate_indices[valid_subset_indices]
        local_points_masked = local_points_subset[valid_subset_indices]

        grid_information = {
            "lateral_range": lateral_range,
            "depth_range": depth_range,
            "spacing": spacing,
            # "grid": grid,
            "grid_rotated": grid_rotated,
            # Store transforms for world->beamlet mapping used by RegularGridInterpolator
            # "R_world_to_BEV": rot_mat,  # inverse of R_BEV_to_world
            # "R_BEV_to_beamlet": derived_rot_mat,  # inverse of R_beamlet_to_BEV
            "iso_center": ray["iso_center"],
            "source_point_bev": ray["source_point_bev"],
            "local_points_masked": local_points_masked,
            "ix_mask": ix_mask,
        }

        self.grid_information = grid_information

        return grid_information

    def _extract_sitk_grid_and_interpolate(
        self, grid_information: dict, ct_interpolation: bool = False
    ):
        """Interpolate a volume (CT) on a provided world-space sampling grid.

        Parameters
        ----------
        grid_information : dict
            Dictionary returned by ``_construct_grid`` containing `grid_rotated`,
            `lateral_range` and `depth_range`.
        ct_interpolation : bool, optional
            If ``True``, assigns HU air value (-1000) to out-of-volume samples; otherwise
            zeros are kept. Defaults to ``False``.

        Returns
        -------
        Array
            Interpolated values reshaped to (depth, lateral, lateral), matching the
            regular local grid axis order for downstream model consumption.
        """
        xp = array_api_compat.array_namespace(grid_information["grid_rotated"])

        lateral_range = grid_information["lateral_range"]
        depth_range = grid_information["depth_range"]
        grid_rotated = grid_information["grid_rotated"]

        values = self.ct_interpolator(grid_rotated)
        # Meshgrid now built with shape (depth, lateral, lateral); direct reshape yields desired ordering
        values_reshaped = xp.reshape(
            values, (len(depth_range), len(lateral_range), len(lateral_range))
        )

        if ct_interpolation is True:
            ix_zeros = xp.nonzero(values_reshaped == 0)
            values_reshaped[ix_zeros] = -1000  # hu for air

        # debug: visualize ct cuboid (network input)
        # import matplotlib.pyplot as plt
        # z_slice = int(np.round(len(grid_information["lateral_range"]) / 2))
        # plt.imshow(values_reshaped[:, :, z_slice], cmap="gray")

        return xp.asarray(values_reshaped, device=self.device)

    def _init_ray(self, beam_info: dict[str], j: int) -> dict[str]:
        """Initialize per-ray geometry and CT sampling for the AI engine.
        Performs forward pass for all bixel of current ray and stores the outputs

        Parameters
        ----------
        beam_info : dict[str]
            Beam information dictionary produced in beam initialization, containing
            rotation matrices, BEV coordinates, and ray metadata.
        j : int
            Index of the ray to initialize.

        Returns
        -------
        dict[str]
            Ray dictionary augmented with indices, geometry, an attached
            `ct_cuboid` sampled on the local grid for later model inference
            and saves the precomputed batch for this ray.
        """
        ray = beam_info["beam"]["rays"][j]
        ray["beam_index"] = beam_info["beam_index"]
        ray["ray_index"] = j
        ray["iso_center"] = beam_info["beam"]["iso_center"]

        if "num_of_bixels_per_ray" in beam_info["beam"]:
            ray["num_of_bixels"] = beam_info["beam"]["num_of_bixels_per_ray"][j]
        else:
            # Fallback: use the actual number of beamlets on this ray
            # This is needed for machines like "Focused" that don't precompute counts.
            ray["num_of_bixels"] = len(ray.get("beamlets", [])) or 0

        ray["source_point_bev"] = beam_info["beam"]["source_point_bev"]
        ray["sad"] = beam_info["beam"]["sad"]
        ray["bixel_width"] = beam_info["beam"]["bixel_width"]

        # self._get_ray_geometry_from_beam(ray, beam_info)

        ray["rotation_mat"] = beam_info["rot_mat_system_T"]
        ray["radiation_mode"] = beam_info["beam"]["radiation_mode"]
        ray["source_point"] = beam_info["beam"]["source_point"]
        ray["machine"] = beam_info["beam"]["machine"]
        self._extract_ct_cuboid(ray, beam_info)

        xp = self.xp
        if USE_BATCH and getattr(self.model, "HAS_BATCHED_FORWARD", False):
            local_points_masked = self.grid_information["local_points_masked"]
            ix_mask = self.grid_information["ix_mask"]
            depth = self.grid_information["depth_range"]
            lateral = self.grid_information["lateral_range"]
            pts_eval = xp.stack(
                (local_points_masked[:, 1], local_points_masked[:, 0], local_points_masked[:, 2]),
                axis=1,
            )
            vdose_sub = self._vdose_grid_mask[ix_mask]
            # Gather all bixel energies for this ray
            energies = [b.get("energy") for b in ray.get("beamlets", [])]
            B = len(energies)

            assert B == ray["num_of_bixels"], "Inconsistent bixel count in ray!"
            assert B > 0, "No bixels found in ray!"

            # Batched model inference: one forward pass for all bixels of the ray
            energies_arr = xp.asarray(energies, device=self.device)
            cuboid_batch = self.model._predict_dose(
                ray["ct_cuboid"], energies_arr, self.grid_information
            )
            assert cuboid_batch is not None, "Batched model prediction returned None!"

            precomputed = []

            if USE_BATCH_INTERP and array_api_compat.is_torch_namespace(xp):
                interpolator = interp3d(
                    (depth, lateral, lateral),
                    cuboid_batch,
                    method="linear",
                    bounds_error=False,
                    fill_value=0.0,
                )
                vals_batch = xp.asarray(
                    interpolator(pts_eval), dtype=xp.float32, device=self.device
                )
            else:
                vals_batch = []
                for i in range(B):
                    cuboid_i = cuboid_batch[i]
                    interpolator = interp3d(
                        (depth, lateral, lateral),
                        cuboid_i,
                        method="linear",
                        bounds_error=False,
                        fill_value=0.0,
                    )
                    vals_in = xp.asarray(
                        interpolator(pts_eval), dtype=xp.float32, device=self.device
                    )
                    vals_batch.append(vals_in)

            for i in range(B):
                vals_in = vals_batch[i]
                # Apply VOI mask only to inside points
                vals_in = xp.where(vdose_sub, vals_in, 0.0)

                nz_mask = vals_in > 0
                if xp.any(nz_mask):
                    ix_nz = xp.asarray(ix_mask[nz_mask], dtype=xp.int64, device=self.device)
                    vals_nz = xp.asarray(vals_in[nz_mask], dtype=xp.float32, device=self.device)
                else:
                    ix_nz = xp.asarray([], dtype=xp.int64, device=self.device)
                    vals_nz = xp.asarray([], dtype=xp.float32, device=self.device)

                precomputed.append({"ix": ix_nz, "physical_dose": vals_nz})

            ray["_precomputed_bixels"] = precomputed

        return ray

    def _extract_single_scenario_ray(self, ray: dict, scen_idx: int) -> dict:
        """Return the ray for a specific scenario without modifying depths.

        Parameters
        ----------
        ray : dict
            Base ray dictionary.
        scen_idx : int
            Scenario index in the multi-scenario handler.

        Returns
        -------
        dict
            A shallow copy of the input ray. Unlike the base PB engine, AI engine
            does not adapt radiological depths per scenario here.
        """
        # Gets number of scenario
        scen_num = self.mult_scen.scen_num(scen_idx)
        ct_scen = self.mult_scen.linear_mask[0][scen_num]

        scen_ray = ray.copy()
        # scen_ray["ix"] = scen_ray["ix"][ct_scen]

        return scen_ray

    def _init_dose_calc(self, ct: CT, cst: StructureSet, stf: SteeringInformation):
        """Initialize CT grid context and the AI model for dose calculation.

        Method
        - Calls the base initializer to set up common state and the dose grid.
        - Extracts SimpleITK image arrays and builds a compatible Grid for interpolation.
        - Loads the configured AI model and wires the current array backend to it.
        - Precomputes world coordinates for all dose-grid voxels (as linear indices) used
          later by interpolation in `_compute_bixel`.

        Parameters
        ----------
        ct : CT
            The CT object (SimpleITK-backed) providing voxel data and geometry.
        cst : StructureSet
            Structure set; unused here but required by the interface.
        stf : SteeringInformation
            Steering information including beams and rays for the plan.

        Returns
        -------
        dict
            The dose influence matrix scaffolding as produced by the base initializer.
        """
        dij = super()._init_dose_calc(ct, cst, stf)
        xp = self.xp

        self.ct_cube_hu = ct.cube_hu
        self._vdose_grid_mask = xp.asarray(self._vdose_grid_mask, device=self.device)

        self.sitk_xp = xp.asarray(sitk.GetArrayFromImage(ct.cube_hu), device=self.device)
        sitk_grid = Grid.from_sitk_image(ct.cube_hu)
        self.ct_grid_dimensions = (
            xp.asarray(sitk_grid.x, device=self.device),
            xp.asarray(sitk_grid.y, device=self.device),
            xp.asarray(sitk_grid.z, device=self.device),
        )

        self.ct_interpolator = interp3d(
            self.ct_grid_dimensions,
            xp.permute_dims(self.sitk_xp, (2, 1, 0)),
            method="linear",
            bounds_error=False,
            fill_value=0,
        )

        self.model = get_model(self.pln)
        self.model.xp = xp
        self.model.USE_BATCH = USE_BATCH

        # Store model config for grid construction
        self.model_config = getattr(self.model, "config", None)

        # leave numpy here for now, since linear_indices_to_grid_coordinates moves to numpy anyway
        all_idx = np.arange(self.dose_grid.num_voxels, dtype=int)
        self.coords_all = xp.asarray(
            linear_indices_to_grid_coordinates(all_idx, self.dose_grid, index_type="numpy"),
            device=self.device,
        )

        return dij
