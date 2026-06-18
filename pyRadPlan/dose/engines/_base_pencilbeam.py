"""Base class for pencil beam dose calculation algorithms."""

from abc import abstractmethod
from typing import Any, ClassVar, Literal
import warnings
import logging
import time
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

import SimpleITK as sitk
import numpy as np
from scipy import sparse
import array_api_compat

from pyRadPlan.core import resample_image, np2sitk
from pyRadPlan.ct import CT, default_hlut
from pyRadPlan.cst import StructureSet
from pyRadPlan.stf import SteeringInformation
from pyRadPlan.geometry import get_beam_rotation_matrix
from pyRadPlan.raytracer import RayTracerSiddon

from ._base import DoseEngineBase
from ...core.xp_utils.typing import Array
from ...core.xp_utils import cupy_available, pytorch_gpu_available
from ...core.xp_utils import to_namespace, to_numpy, free_gpu_memory
from .kernels import _calc_geo_dists_cupy_kernel, _calc_geo_dists_torch_kernel


logger = logging.getLogger(__name__)


class PencilBeamEngineAbstract(DoseEngineBase):
    """
    An abstract class representing the Pencil Beam Engine.

    This class extends DoseEngineBase and provides the foundational structure for implementing a
    pencil beam dose calculation engine. It includes methods for initializing dose calculations,
    setting defaults, and various helper methods required for the dose calculation process.

    Attributes
    ----------
    keep_rad_depth_cubes : bool
        Flag to keep radiation depth cubes.
    geometric_lateral_cutoff : float
        Lateral geometric cut-off in mm, used for raytracing and geometry.
    dosimetric_lateral_cutoff : float
        Relative dosimetric cut-off (in fraction of values calculated).
    ssd_density_threshold : float
        Threshold for SSD computation.
    use_given_eq_density_cube : bool
        Use the given density cube ct.cube and omit conversion from cubeHU.
    ignore_outside_densities : bool
        Ignore densities outside of cst contours.
    num_of_dij_fill_steps : int
        Number of times during dose calculation the temporary containers are moved to a sparse
        matrix.
    cube_wed : sitk.Image
        Relative electron density / stopping power cube.
    hlut : np.ndarray
        Hounsfield lookup table to create relative electron density cube.

    Methods
    -------
    __init__()
        Initializes the PencilBeamEngineAbstract class.
    set_defaults()
        Sets default values for the attributes.
    _compute_bixel(curr_ray, k)
        Abstract method to compute bixel.
    _calc_dose(ct, cst, stf)
        Calculates the dose.
    _init_dose_calc(ct, cst, stf)
        Initializes dose calculation.
    _allocate_quantity_matrices(dij, names)
        Allocates quantity matrix containers.
    _init_beam(curr_beam, ct, cst, stf, i)
        Initializes the beam.
    _init_ray(curr_beam, j)
        Initializes the ray.
    _extract_single_scenario_ray(ray, scen_idx)
        Extracts a single scenario ray.
    _get_ray_geometry_from_beam(ray, curr_beam)
        Gets ray geometry from beam.
    _get_lateral_distance_from_dose_cutoff_on_ray(ray)
        Gets lateral distance from dose cutoff on ray.
    _fill_dij(bixel, dij, stf, scen_idx, curr_beam_idx, curr_ray_idx, curr_bixel_idx, counter)
        Fills the dose influence matrix (dij).
    _finalize_dose(dij)
        Finalizes the dose.
    calcGeoDists(rot_coords_bev, sourcePoint_bev, targetPoint_bev, SAD, radDepthIx, lateralCutOff)
        Calculates geometric distances.
    geometric_cutoff()
        Property for geometric cutoff.
    warn_deprecated_engine_property(old_name, new_name, internal_name)
        Warns about deprecated engine properties.
    """

    keep_rad_depth_cubes: bool
    geometric_lateral_cutoff: float
    dosimetric_lateral_cutoff: float
    ssd_density_threshold: float
    use_given_eq_density_cube: bool
    ignore_outside_densities: bool
    trace_on_dose_grid: bool
    cube_wed: sitk.Image
    hlut: np.ndarray

    # Every derived engine must declare both flags.
    _dij_guarantee_canonical: ClassVar[bool]
    """``True`` iff the engine guarantees that every assembled CSC column already has
    sorted, unique row indices (no duplicate entries, ascending order)"""

    _dij_guarantee_nonzero: ClassVar[bool]
    """``True`` iff the engine guarantees that it never writes structural zero
    entries into the dose influence matrix"""

    def __init__(self, pln=None):
        self.keep_rad_depth_cubes = False
        self.geometric_lateral_cutoff: float = 50
        self.dosimetric_lateral_cutoff: float = 0.9950
        self.ssd_density_threshold: float = 0.0500
        self.use_given_eq_density_cube: bool = False
        self.ignore_outside_densities: bool = True
        self.trace_on_dose_grid: bool = True
        self.cube_wed = None
        self.hlut = None
        self.assumed_sparsity = 5e-4

        self._computed_quantities = []
        self._effective_lateral_cutoff = None
        self._num_of_bixels_container = None
        self._rad_depth_cubes = []
        self._raytracer = None
        self._eps_ijk: Array = None
        self._cuda_kernel_used = False

        super().__init__(pln)

    @abstractmethod
    def _compute_bixel(self, curr_ray, k):
        raise NotImplementedError("Method _compute_bixel must be implemented in derived class.")

    def _calc_dose(self, ct: CT, cst: StructureSet, stf: SteeringInformation):
        """
        Calculate the dose using the pencil beam method.

        Parameters
        ----------
        ct : CT
            The CT object.
        cst : StructureSet
            The structure set object.
        stf : SteeringInformation
            The steering information object.

        Returns
        -------
        dict
            The dose influence matrix dictionary.
        """

        # Initialize
        dij = self._init_dose_calc(ct, cst, stf)

        xp = self.xp
        device = self.device
        self._vox_world_coords = xp.asarray(
            self._vox_world_coords,
            dtype=xp.float32,
            device=device,
        )
        self._vox_world_coords_dose_grid = xp.asarray(
            self._vox_world_coords_dose_grid,
            dtype=xp.float32,
            device=device,
        )
        self._vdose_grid = to_namespace(self.xp, self._vdose_grid, device=self.device)

        # We loop over scenario in the scenario model
        # TODO: we need to correctly work out scenarios
        for shift_scen in range(self.mult_scen.tot_num_shift_scen):
            # Find first instance of the shift to select the shift values
            # TODO!: Check for more than one Scenario
            ix_shift_scen = np.where(self.mult_scen.linear_mask[:, 1] == shift_scen)

            scen_stf = stf
            # Manipulate isocenter
            for beam in scen_stf.beams:
                beam.iso_center += self.mult_scen.iso_shift[ix_shift_scen, :].reshape(-1)

            if self.mult_scen.tot_num_shift_scen > 1:
                logger.info(
                    f"Shift scenario {shift_scen} of {self.mult_scen.tot_num_shift_scen}: \n"
                )

            bixel_counter = 0

            # Loop over all beams
            with logging_redirect_tqdm():
                for i in tqdm(range(dij["num_of_beams"]), desc="Beam", unit="b", leave=False):
                    # Initialize Beam Geometry
                    t = time.time()
                    curr_beam = self._init_beam(dij, ct, cst, scen_stf, i)
                    logger.info("Beam %d initialized in %f seconds.", i + 1, time.time() - t)

                    # Keep tabs on bixels computed in this beam
                    bixel_beam_counter = 0
                    # Ray calculation
                    for j in tqdm(
                        range(curr_beam["beam"]["num_of_rays"]), desc="Ray", unit="r", leave=False
                    ):
                        # Initialize Ray Geometry
                        curr_ray = self._init_ray(curr_beam, j)

                        # Even if the ray hits nothing, we still emit empty bixel columns
                        # so that CSC structures and bookkeeping stay consistent.
                        # The following block might be re-enabled in future to skip empty rays.

                        # check if ray hit anything. If so, skip the computation
                        # if all(not arr.size for arr in curr_ray["rad_depths"]):
                        #     continue #continue

                        # TODO: incorporate scenarios correctly
                        for ct_scen in range(self.mult_scen.num_of_ct_scen):
                            for range_scen in range(self.mult_scen.tot_num_range_scen):
                                # Obtain scenario index
                                full_scen_idx = self.mult_scen.sub2scen_ix(
                                    ct_scen, shift_scen, range_scen
                                )

                                if self.mult_scen.scen_mask[full_scen_idx]:
                                    # Extract single scenario ray
                                    scen_ray = self._extract_single_scenario_ray(
                                        curr_ray, full_scen_idx
                                    )

                                    for k in range(curr_ray["num_of_bixels"]):
                                        # Bixel Computation
                                        curr_bixel = self._compute_bixel(scen_ray, k)
                                        # fill the current bixel in the sparse dose influence
                                        # matrix
                                        self._fill_dij(
                                            curr_bixel,
                                            dij,
                                            scen_stf,
                                            full_scen_idx,
                                            i,
                                            j,
                                            k,
                                            bixel_counter + k,
                                        )
                        # Progress Update & Bookkeeping
                        bixel_counter += curr_ray["num_of_bixels"]
                        bixel_beam_counter += curr_ray["num_of_bixels"]

                    # Free GPU memory from curr_beam
                    del curr_beam["bev_coords"]
                    del curr_beam["rad_depths"]
                    del curr_beam["geo_depths"]
                    del curr_beam["valid_coords"]
                    del curr_beam["valid_coords_all"]
                    del curr_beam["rot_mat_system_T"]
                    curr_beam = None

                    # Force GPU memory cleanup
                    free_gpu_memory(self.xp)

        # Finalize dose calculation
        logger.info("Finalizing dose calculation...")
        t_start = time.time()
        dij = self._finalize_dose(dij)
        t_end = time.time()
        logger.info("Done in %f seconds.", t_end - t_start)

        return dij

    def _init_dose_calc(self, ct: CT, cst: StructureSet, stf: SteeringInformation):
        """
        Initialize the dose calculation.

        Modified inherited method of the superclass DoseEngine,
        containing initialization which is specifically needed for
        pencil beam calculation and not for other engines.
        """

        dij = super()._init_dose_calc(ct, cst, stf)

        # calculate rED or rSP from HU or take provided wedCube
        if self.use_given_eq_density_cube and not hasattr(ct, "cube_hu"):
            logging.warning(
                "HU Conversion requested to be omitted but no ct.cube exists! "
                "Will override and do the conversion anyway!"
            )
            self.use_given_eq_density_cube = False

        if self.use_given_eq_density_cube:
            ct_wed = ct.cube_hu
            logging.info("Omitting HU to rED/rSP conversion and using existing ct.cube!\n")
        else:
            # TODO: obtain correct hlut
            if self.hlut is None:
                self.hlut = default_hlut()
            ct_wed = ct.compute_wet(self.hlut)

        self.cube_wed = ct_wed

        # Ignore densities outside of contours
        if self.ignore_outside_densities:  # TODO: default = None (not tested yet)
            mask_image = np2sitk.linear_indices_to_sitk_mask(
                self._vct_grid, self.cube_wed, order="numpy"
            )

            # # TODO: ct does not yet support multi_scen
            # for i in range(ct.num_of_ct_scen):
            #     self.cube_wed.ToScalarImage()
            #     self.cube_wed[erase_ct_dens_mask == 1] = 0

            # Apply the mask to the cube_wed image
            self.cube_wed = sitk.Mask(self.cube_wed, mask_image, outsideValue=0)

        # Allocate memory for quantity containers
        dij = self._allocate_quantity_matrices(dij, ["physical_dose"])

        # Initialize ray-tracer
        if self.trace_on_dose_grid:
            wed_cube_trace = resample_image(
                input_image=self.cube_wed,
                interpolator=sitk.sitkLinear,
                target_grid=self.dose_grid,
            )
        else:
            wed_cube_trace = self.cube_wed

        self._raytracer = RayTracerSiddon([wed_cube_trace])
        self._raytracer.lateral_cut_off = self.geometric_lateral_cutoff

        return dij

    def _allocate_quantity_matrices(self, dij: dict[str, Any], names: list[str]):
        device = "cpu"
        xp = np  # self.xp
        # initialize shared structure container on first call
        if "shared_structure" not in dij:
            dij["shared_structure"] = np.empty(self.mult_scen.scen_mask.shape, dtype=object)

        # Loop over all requested quantities
        for q_name in names:
            # Create dij list for each quantity
            dij[q_name] = np.empty(self.mult_scen.scen_mask.shape, dtype=object)

            # Loop over all scenarios and preallocate quantity containers
            # TODO: write test for this
            for i in range(self.mult_scen.scen_mask.size):
                # Only if there is a scenario we will allocate
                if self.mult_scen.scen_mask.flat[i]:
                    if self._calc_dose_direct:
                        dij[q_name].flat[i] = xp.zeros(
                            (self.dose_grid.num_voxels, dij["num_of_beams"]),
                            dtype=xp.float32,
                            device=device,
                        )
                    else:
                        # We allocate raw csc sparse matrix structures using
                        # a rough estimat ebased on number of voxels and
                        # beamlets

                        if dij["shared_structure"].flat[i] is None:
                            est_nnz = np.fix(
                                self.assumed_sparsity
                                * self.dose_grid.num_voxels
                                * self._num_of_columns_dij
                            ).astype(np.int64)
                            dij["shared_structure"].flat[i] = {
                                "indices": xp.empty((est_nnz,), dtype=xp.int64, device=device),
                                "indptr": xp.zeros(
                                    (self._num_of_columns_dij + 1,), dtype=xp.int64, device=device
                                ),
                                "nnz": 0,
                            }
                        else:
                            # If shared structure already exists, we use its size
                            est_nnz = dij["shared_structure"].flat[i]["indices"].size

                        dij[q_name].flat[i] = {
                            "data": xp.empty((est_nnz,), dtype=xp.float32, device=device),
                        }

            self._computed_quantities.append(q_name)

        self._effective_lateral_cutoff = self.geometric_lateral_cutoff

        return dij

    def _init_beam(
        self, _dij: dict, ct: CT, _cst: StructureSet, stf: SteeringInformation, i
    ) -> dict:
        """
        Initialize the beam for pencil beam dose calculation.

        Parameters
        ----------
        dij : dict
            The dose influence matrix dictionary.
        ct : CT
            The CT object.
        _cst : StructureSet
            The structure set object. Unused here
        stf : SteeringInformation
            The steering information object.
        i : int
            Index of the beam.

        Returns
        -------
        dict
            Beam Information dictionary
        """
        # TODO: here .model_dump() is used to get beam as dict. Its possible to change...
        # ... the ray_tracing to handle this as the model
        beam_info = {"beam": stf.beams[i].model_dump(), "beam_index": i}

        xp = array_api_compat.array_namespace(self._vox_world_coords)

        for key in [
            "iso_center",
            "source_point_bev",
            "source_point",
            "num_of_bixels_per_ray",
            "gantry_angle",
            "couch_angle",
        ]:
            if array_api_compat.is_array_api_obj(beam_info["beam"][key]):
                beam_info["beam"][key] = to_namespace(
                    xp, beam_info["beam"][key], device=self.device
                )
            else:
                beam_info["beam"][key] = xp.asarray(
                    beam_info["beam"][key], dtype=xp.float32, device=self.device
                )
        for ray in beam_info["beam"]["rays"]:
            for key in ["ray_pos", "ray_pos_bev", "target_point", "target_point_bev"]:
                if ray[key] is None:
                    continue
                if array_api_compat.is_array_api_obj(ray[key]):
                    ray[key] = to_namespace(xp, ray[key], device=self.device)
                else:
                    ray[key] = xp.asarray(ray[key], dtype=xp.float32, device=self.device)

        # Convert voxel indices to real coordinates using iso center of beam i
        iso_center = xp.asarray(
            beam_info["beam"]["iso_center"], dtype=xp.float32, device=self.device
        )

        coords_v = self._vox_world_coords - iso_center
        coords_vdose_grid = self._vox_world_coords_dose_grid - iso_center

        # Get Rotation Matrix
        rot_mat = get_beam_rotation_matrix(
            beam_info["beam"]["gantry_angle"], beam_info["beam"]["couch_angle"]
        )
        beam_info["rot_mat_system_T"] = xp.asarray(rot_mat, dtype=xp.float32, device=self.device)

        # Rotate coordinates (1st couch around Y axis, 2nd gantry movement)
        rot_coords_v = xp.matmul(coords_v, beam_info["rot_mat_system_T"])
        rot_coords_vdose_grid = xp.matmul(coords_vdose_grid, beam_info["rot_mat_system_T"])

        source_point_bev = xp.asarray(
            beam_info["beam"]["source_point_bev"], dtype=xp.float32, device=self.device
        )
        rot_coords_v -= source_point_bev
        rot_coords_vdose_grid -= source_point_bev

        # Calculate geometric distances
        geo_dist_vdose_grid = [
            xp.sqrt(xp.sum(rot_coords_vdose_grid**2, axis=1)) for _ in range(ct.num_of_ct_scen)
        ]

        # Calculate radiological depth cube
        logger.info("Calculating radiological depth cube...")

        start_time = time.time()

        self._raytracer.debug_core_performance = True
        self._raytracer.lateral_cut_off = beam_info["beam"].get(
            "effective_lateral_cut_off", self._effective_lateral_cutoff
        )
        rad_depth_cubes = self._raytracer.trace_cubes(beam_info["beam"])
        self._raytracer.debug_core_performance = False

        # TODO: add universal time-debugging method
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info("Elapsed time for rayTracing per Beam: %f seconds", elapsed_time)

        beam_info["valid_coords"] = [None] * len(rad_depth_cubes)
        beam_info["rad_depths"] = [None] * len(rad_depth_cubes)

        for c, rad_depth_cube in enumerate(rad_depth_cubes):
            if self.trace_on_dose_grid:
                rad_depth_cube_dose_grid = rad_depth_cube
            else:
                rad_depth_cube_dose_grid = resample_image(
                    input_image=rad_depth_cube,
                    interpolator=sitk.sitkLinear,
                    target_grid=self.dose_grid,
                )
            if self.keep_rad_depth_cubes:
                self._rad_depth_cubes.append(rad_depth_cube_dose_grid)

            rad_depth_cube_dosegrid = sitk.GetArrayViewFromImage(rad_depth_cube_dose_grid)
            rad_depth_arr = xp.asarray(
                rad_depth_cube_dosegrid, dtype=self._vox_world_coords.dtype, device=self.device
            )
            rad_depth_vdose_grid = xp.take(
                xp.reshape(rad_depth_arr, (-1,)), self._vdose_grid, axis=0
            )

            # Find valid coordinates
            coord_is_valid = xp.isfinite(rad_depth_vdose_grid)

            beam_info["valid_coords"][c] = coord_is_valid

            # TODO: !remove brackets once multscen is implemented
            beam_info["rad_depths"][c] = rad_depth_vdose_grid

        beam_info["geo_depths"] = geo_dist_vdose_grid
        beam_info["bev_coords"] = rot_coords_vdose_grid

        beam_info["valid_coords_all"] = xp.any(xp.stack(beam_info["valid_coords"]), axis=0)

        # Check existence of target_points
        if any(r["target_point"] is None for r in beam_info["beam"]["rays"]):
            logger.debug("Missing target_point in rays. Calculating target points...")
            for ray in beam_info["beam"]["rays"]:
                ray["target_point"] = 2 * ray["ray_pos"] - beam_info["beam"]["source_point"]
                ray["target_point_bev"] = (
                    2 * ray["ray_pos_bev"] - beam_info["beam"]["source_point_bev"]
                )

        # Compute SSDs
        self._compute_ssd(beam_info, ct, density_threshold=self.ssd_density_threshold)

        logger.info("Done.")

        return beam_info

    def _init_ray(self, beam_info: dict[str], j: int) -> dict[str]:
        """
        Initialize a ray for pencil beam dose calculation.

        Parameters
        ----------
        curr_beam : dict
            The current beam data.
        j : int
            The ray index.

        Returns
        -------
        dict
            The initialized ray.
        """
        # NOTE!: we copy the ray here
        # This ensures that data is not accumulated for every ray
        ray = beam_info["beam"]["rays"][j].copy()
        ray["beam_index"] = beam_info["beam_index"]
        ray["ray_index"] = j
        ray["iso_center"] = beam_info["beam"]["iso_center"]

        if "num_of_bixels_per_ray" in beam_info["beam"]:
            ray["num_of_bixels"] = int(beam_info["beam"]["num_of_bixels_per_ray"][j])
        else:
            # Fallback: use the actual number of beamlets on this ray
            # This is needed for machines like "Focused" that don't precompute counts.
            ray["num_of_bixels"] = len(ray.get("beamlets", [])) or 0

        ray["source_point_bev"] = beam_info["beam"]["source_point_bev"]
        ray["sad"] = beam_info["beam"]["sad"]
        ray["bixel_width"] = beam_info["beam"]["bixel_width"]

        self._get_ray_geometry_from_beam(ray, beam_info)

        return ray

    def _compute_ssd(
        self,
        beam_info: dict,
        _ct: CT,
        mode: Literal["first"] = "first",
        density_threshold: float = 0.05,
        show_warning: bool = True,
    ):
        """
        Compute SSD (Source to Surface Distance) for each ray.

        Parameters
        ----------
        beam_info : dict
            Beam Information, will be modified to include SSD.
        ct : CT
            The CT object.
        mode : Literal["first"], optional
            Mode for handling multiple cubes to compute one SSD. Only 'first' is implemented.
        density_threshold : float, optional
            Value determining the skin threshold.
        show_warning : bool, optional
            Flag to show warnings.
        """

        beam = beam_info["beam"]

        # TODO: Add MultiScenario Support - remove this line
        if mode == "first":
            ssd = [None] * beam["num_of_rays"]

            xp = array_api_compat.array_namespace(beam["iso_center"], beam["source_point"])

            # Build stacked arrays using the backend's stack (works for numpy, cupy, torch)
            ray_pos_bev = xp.stack([xp.asarray(ray["ray_pos_bev"]) for ray in beam["rays"]])
            target_points = xp.stack([xp.asarray(ray["target_point"]) for ray in beam["rays"]])

            alpha, _, rho, d12, _ = self._raytracer.trace_rays(
                beam["iso_center"],
                xp.reshape(beam["source_point"], (1, 3)),
                target_points,
            )

            alpha = to_namespace(xp, alpha, device=self.device)
            rho = [to_namespace(xp, r, device=self.device) for r in rho]
            d12 = to_namespace(xp, d12.squeeze(), device=self.device)
            # prevent d12 from being 0d array which causes issues with indexing later on
            if d12.ndim == 0:
                d12 = xp.reshape(d12, (1,))
            # find rays that do not hit patients
            ix_nan = xp.all(xp.isnan(rho[0]), axis=1)
            if xp.any(ix_nan):
                msg = f"{xp.count_nonzero(ix_nan)} rays do not hit patient. Trying to fix afterwards..."
                if show_warning:
                    warnings.warn(msg)
                else:
                    logger.warning(msg)

            ix_ssd = -1 * xp.ones_like(ix_nan, dtype=xp.int64)

            # Work with explicit row indices to avoid unsupported boolean argmax on some backends
            row_idx = xp.nonzero(~ix_nan)[0]

            # Compute boolean condition, convert to integers for argmax
            cond = xp.take(rho[0], row_idx, axis=0) > density_threshold
            cond_int = xp.where(cond, xp.asarray(1, dtype=xp.int64), xp.asarray(0, dtype=xp.int64))
            arg = xp.argmax(cond_int, axis=1)

            ix_ssd = xp.where(~ix_nan, xp.astype(arg, xp.int64), ix_ssd)

            ssd = beam["sad"] * xp.ones_like(ix_ssd, dtype=xp.float32)

            d12_sel = xp.reshape(xp.take(d12, row_idx, axis=0), (-1,))
            alpha_rows = xp.take(alpha, row_idx, axis=0)
            idx = xp.reshape(xp.astype(arg, xp.int64), (-1, 1))
            alpha_sel = xp.reshape(xp.take_along_axis(alpha_rows, idx, axis=1), (-1,))

            ssd_vals = xp.astype(d12_sel, xp.float32) * xp.astype(alpha_sel, xp.float32)
            ssd = xp.where(~ix_nan, ssd_vals, ssd)

            # Now assign the ssd to all rays
            for j, ray in enumerate(beam["rays"]):
                if ix_nan[j]:  # Try to fix SSD by using SSD of closest neighboring ray
                    ray["SSD"] = self._closest_neighbor_ssd(ray_pos_bev, ssd, ray["ray_pos_bev"])
                else:
                    ray["SSD"] = float(ssd[j])

        else:
            raise ValueError(f"Invalid mode {mode} for SSD calculation")

        beam_info["beam"] = beam

    def _closest_neighbor_ssd(self, ray_pos_bev: Array, ssd: Array, curr_pos: Array) -> float:
        """
        Find the closest neighboring ray's SSD.

        Parameters
        ----------
        ray_pos_bev : np.ndarray
            Array of ray positions.
        curr_pos : np.ndarray
            Current ray position.

        Returns
        -------
        float
            SSD value of the closest neighboring ray.
        """
        xp = array_api_compat.array_namespace(ray_pos_bev, ssd, curr_pos)
        distances = xp.sum((ray_pos_bev - curr_pos) ** 2, axis=1)
        sorted_indices = xp.argsort(distances)
        for ix in sorted_indices:
            if ssd[ix] is not None:
                return float(ssd[ix])
        raise ValueError(
            "Error in SSD calculation: Could not fix SSD calculation by using closest neighboring "
            "ray."
        )

        #
        # """Find the closest neighbor to ray j and return its SSD."""
        # distances = np.linalg.norm(ray_pos_bev - ray_pos_bev[j], axis=1)
        # distances[j] = np.inf
        # return SSD[np.argmin(distances)]

    def _extract_single_scenario_ray(self, ray: dict, scen_idx: int):
        """
        Extract a single scenario ray and adapt radiological depths.

        Parameters
        ----------
        ray (dict):
            The ray data.
        scen_idx (int):
            The scenario index.

        Returns
        -------
        dict:
            The scenario ray with adapted radiological depths.
        """
        # Gets number of scenario
        xp = array_api_compat.array_namespace(ray["rad_depths"][0])
        scen_num = self.mult_scen.scen_num(scen_idx)
        ct_scen = self.mult_scen.linear_mask[0][scen_num]

        # First, create a ray of the specific scenario to adapt rad depths
        scen_ray = ray.copy()
        scen_ray["rad_depths"] = scen_ray["rad_depths"][ct_scen]
        scen_ray["rad_depths"] = (
            1 + xp.asarray(self.mult_scen.rel_range_shift[scen_num], device=self.device)
        ) * scen_ray["rad_depths"] + xp.asarray(
            self.mult_scen.abs_range_shift[scen_num], device=self.device
        )
        scen_ray["radial_dist_sq"] = scen_ray["radial_dist_sq"][ct_scen]
        scen_ray["ix"] = scen_ray["ix"][ct_scen]

        if self.mult_scen.abs_range_shift[scen_num] < 0:
            # TODO: better way to handle this?
            scen_ray["rad_depths"][scen_ray["rad_depths"] < 0] = 0

        if "geo_depths" in scen_ray:
            scen_ray["geo_depths"] = scen_ray["geo_depths"][ct_scen]

        if "lat_dists" in scen_ray:
            scen_ray["lat_dists"] = scen_ray["lat_dists"][ct_scen]

        if "iso_lat_dists" in scen_ray:
            scen_ray["iso_lat_dists"] = scen_ray["iso_lat_dists"][ct_scen]

        return scen_ray

    def _get_ray_geometry_from_beam(self, ray: dict, beam_info: dict):
        raise NotImplementedError(
            "Method _get_ray_geometry_from_beam must be implemented in derived class."
        )

    def _get_ray_geometry_from_beam(self, ray: dict[str], beam_info: dict[str]):
        lateral_ray_cutoff = self._get_lateral_distance_from_dose_cutoff_on_ray(ray)

        xp = array_api_compat.array_namespace(beam_info["bev_coords"])

        # Ray tracing for beam i and ray j
        ix, radial_dist_sq, lat_dists, iso_lat_dists = self.calc_geo_dists(
            beam_info["bev_coords"],
            ray["source_point_bev"],
            ray["target_point_bev"],
            ray["sad"],
            beam_info["valid_coords_all"],
            float(lateral_ray_cutoff),
        )
        # Subindex given the relevant indices from the geometric distance calculation
        ray["valid_coords"] = [beam_ix & ix for beam_ix in beam_info["valid_coords"]]
        ray["ix"] = [self._vdose_grid[ix_in_grid] for ix_in_grid in ray["valid_coords"]]

        ray["radial_dist_sq"] = [radial_dist_sq[beam_ix[ix]] for beam_ix in ray["valid_coords"]]

        if lat_dists is not None:
            ray["lat_dists"] = [lat_dists[beam_ix[ix]] for beam_ix in beam_info["valid_coords"]]

        ray["iso_lat_dists"] = [iso_lat_dists[beam_ix[ix]] for beam_ix in ray["valid_coords"]]

        ray["valid_coords_all"] = xp.any(xp.stack(ray["valid_coords"], axis=0), axis=1)

        ray["geo_depths"] = [
            rD[ix] for rD, ix in zip(beam_info["geo_depths"], ray["valid_coords"])
        ]  # usually not needed for particle beams
        ray["rad_depths"] = [
            rD[ix] for rD, ix in zip(beam_info["rad_depths"], ray["valid_coords"])
        ]

    def _get_lateral_distance_from_dose_cutoff_on_ray(self, ray: dict) -> float:
        """
        Obtain the maximum lateral cutoff on a a ray.

        Distance will computed from dosimetric cutoff setting.

        Parameters
        ----------
        _ray : dict
            The ray data. Unused in this base implementation.

        Returns
        -------
        float
            The lateral distance from the dose cutoff on the ray.
        """

        return ray.get("effective_lateral_cut_off", self._effective_lateral_cutoff)

    def _fill_dij(
        self,
        bixel: dict,
        dij: dict,
        _stf: SteeringInformation,
        scen_idx: int,
        curr_beam_idx: int,
        curr_ray_idx: int,
        curr_bixel_idx: int,
        bixel_counter: int,
    ):
        # Debugging timing

        bixel["physical_dose"] = to_numpy(bixel["physical_dose"])

        ix_size = 0
        if bixel and "ix" in bixel:
            ix_val = bixel["ix"]
            ix_size = ix_val.shape[0]

        sub_scen_idx = tuple(np.unravel_index(scen_idx, self.mult_scen.scen_mask.shape))

        if not self._calc_dose_direct:
            shared_struct = dij["shared_structure"][sub_scen_idx]
            # Advance column pointer even when ix_size == 0 to keep CSC valid
            start = shared_struct["indptr"][bixel_counter]
            end = start + ix_size
            shared_struct["indptr"][bixel_counter + 1] = end

            if ix_size > 0:
                need_nnz = shared_struct["nnz"] + ix_size
                if shared_struct["indices"].size < need_nnz:
                    logger.debug("Resizing data and indices arrays for all quantities...")
                    # heuristic growth factor
                    grow = max(ix_size, (self._num_of_columns_dij - bixel_counter) * (ix_size + 1))
                    new_size = shared_struct["indices"].size + grow
                    shared_struct["indices"].resize((new_size,), refcheck=False)

                    # Trigger resize for all quantities associated with this scenario
                    for q in self._computed_quantities:
                        dij[q][sub_scen_idx]["data"].resize((new_size,), refcheck=False)

                bixel["ix"] = to_numpy(bixel["ix"])
                shared_struct["indices"][start:end] = bixel["ix"]
                shared_struct["nnz"] += ix_size

        for q_name in self._computed_quantities:
            if self._calc_dose_direct:
                if ix_size > 0:
                    bixel["ix"] = to_numpy(bixel["ix"])
                    bixel[q_name] = to_numpy(bixel[q_name])
                    dij[q_name][sub_scen_idx][bixel["ix"], curr_beam_idx] += (
                        bixel["weight"] * bixel[q_name]
                    )
            else:
                data_dict = dij[q_name][sub_scen_idx]

                if ix_size > 0:
                    start = dij["shared_structure"][sub_scen_idx]["indptr"][bixel_counter]
                    end = start + ix_size
                    # Fill values into the allocated slice
                    bixel[q_name] = to_numpy(bixel[q_name])
                    data_dict["data"][start:end] = bixel[q_name]

        # Bookkeeping of bixel numbers
        # remember beam and bixel number
        if self._calc_dose_direct:
            dij["beam_num"][curr_beam_idx] = curr_beam_idx
            dij["ray_num"][curr_beam_idx] = curr_beam_idx
            dij["bixel_num"][curr_beam_idx] = curr_beam_idx
        else:
            dij["beam_num"][bixel_counter] = curr_beam_idx
            dij["ray_num"][bixel_counter] = curr_ray_idx
            dij["bixel_num"][bixel_counter] = curr_bixel_idx

    def _finalize_dose(self, dij: dict):
        """
        Finalize the dose influence matrix.

        Pruning the matrix and concatenating the containers to a compressed
        sparse matrix.

        Parameters
        ----------
        dij : dict
            The dose influence matrix.

        Returns
        -------
        Dij
            The finalized dose influence matrix.
        """

        # Loop over all scenarios and remove dose influence for voxels outside of segmentations
        for i in range(self.mult_scen.scen_mask.size):
            # Only if there is a scenario we will allocate
            if self.mult_scen.scen_mask.flat[i]:
                shared_struct = None
                if not self._calc_dose_direct:
                    shared_struct = dij["shared_structure"].flat[i]
                    # Resize shared indices to actual NNZ
                    shared_struct["indices"].resize((shared_struct["nnz"],), refcheck=False)

                # Loop over all used quantities
                for q_name in self._computed_quantities:
                    if not self._calc_dose_direct:
                        # tmp_matrix = cast(sparse.lil_matrix, dij[q_name].flat[i])
                        # tmp_matrix = tmp_matrix.tocsr().T
                        data_dict = dij[q_name].flat[i]

                        indices = shared_struct["indices"]
                        indptr = shared_struct["indptr"]
                        nnz = shared_struct["nnz"]

                        # Resize to the actual number of non-zero elements
                        data_dict["data"].resize((nnz,), refcheck=False)

                        # Create the matrix, avoid copies
                        tmp_matrix = sparse.csc_array(
                            (data_dict["data"], indices, indptr),
                            shape=(self.dose_grid.num_voxels, self._num_of_columns_dij),
                            dtype=np.float32,
                            copy=False,
                        )

                        if not self._dij_guarantee_nonzero:
                            tmp_matrix.eliminate_zeros()

                        if not self._dij_guarantee_canonical:
                            # make sure indices are sorted and matrix is canonical
                            if not tmp_matrix.has_sorted_indices:
                                logger.debug("Sorting indices for %s...", q_name)
                                tmp_matrix.sort_indices()

                            if not tmp_matrix.has_canonical_format:
                                logger.debug("Matrix is not in canonical format for %s...", q_name)
                                tmp_matrix.sum_duplicates()

                        dij[q_name].flat[i] = tmp_matrix

        if self.keep_rad_depth_cubes and self._rad_depth_cubes:
            dij["rad_depth_cubes"] = self._rad_depth_cubes

        # Call the finalizeDose method from the base class
        return super()._finalize_dose(dij)

    def calc_geo_dists(
        self,
        rot_coords_bev: Array,
        source_point_bev: Array,
        target_point_bev: Array,
        sad: float,
        rad_depth_mask: Array,
        lateral_cutoff: float,
    ):
        """
        Calculate geometric distances for dose calculation.

        This function relies on FOUR different ways to compute distances:
        1) GPU with cupy kernel
        2) GPU with torch kernel
        3) Base path (CPU/GPU) - backend supports fancy indexing regarding __setitem__
            -> numpy/cupy/torch
        4) Fallback path - backend does not support fancy indexing regarding __setitem__
            -> jax/strict

        Parameters
        ----------
        rot_coords_bev : Array
            Coordinates in beam's eye view (BEV) of the voxels where ray tracing results are
            available.
        source_point_bev : Array
            Source point in voxel coordinates in BEV.
        target_point_bev : Array
            Target point in voxel coordinates in BEV.
        sad : float
            Source-to-axis distance.
        rad_depth_mask : Array
            Masks the voxels for which radiological depth calculations are available.
        lateral_cutoff : float
            Lateral cutoff specifying the neighborhood for dose calculations.

        Returns
        -------
        ix : Array
            Indices of voxels where dose influence is computed.
        rad_distances_sq : Array
            Squared radial distances to the central ray.
        lat_dists : Array
            Lateral distances to the central ray (in X & Z).
        iso_lat_dists : Array
            Lateral distances to the central ray projected onto the isocenter plane.
        """

        # xp = array_api_compat.array_namespace(
        #     rot_coords_bev, source_point_bev, target_point_bev, rad_depth_mask
        # )
        xp = array_api_compat.array_namespace(
            rot_coords_bev, source_point_bev, target_point_bev, rad_depth_mask
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
        ix = xp.asarray(rad_depth_mask, copy=True)
        rad_depth_ix = xp.nonzero(rad_depth_mask)[0]

        # Make sure that the source-point is a 1D array with matching dtype
        source_point_bev = xp.astype(xp.reshape(source_point_bev, (3,)), rot_coords_bev.dtype)
        target_point_bev = xp.astype(xp.reshape(target_point_bev, (3,)), rot_coords_bev.dtype)

        # Put [0 0 0] position in the source point for beamlet who passes through isocenter
        a = -source_point_bev

        # Normalize the vector
        a /= norm(a)

        # Put [0 0 0] position in the source point for a single beamlet
        b = target_point_bev - source_point_bev

        # Normalize the vector
        b /= norm(b)

        # Calculate Rotation Matrix
        derived_rot_mat = None
        # Check for alignment with tolerance (avoid numerical instability/noise)
        if not xp.all(xp.abs(a - b) < 1e-7):
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

        # GPU Path (CuPy)
        if (
            cupy_available()
            and hasattr(xp, "cuda")
            and xp.cuda.is_available()
            and "cupy" in xp.__name__
        ):
            if not self._cuda_kernel_used:
                # necessary to avoid multiple logging
                logger.info("Using Cupy kernel for geometric distance calculation.")
                self._cuda_kernel_used = True

            rot_coords_subset = xp.take(rot_coords_bev, rad_depth_ix, axis=0)
            num_elements = rot_coords_subset.shape[0]

            if num_elements > 0:
                if derived_rot_mat is None:
                    derived_rot_mat = xp.eye(3, dtype=a.dtype, device=self.device)

                # Ensure derived_rot_mat is correct type
                derived_rot_mat = xp.astype(derived_rot_mat, rot_coords_subset.dtype)

                lateral_cutoff_sq_over_sad_sq = (lateral_cutoff / sad) ** 2

                # NOTE!: I tried using the RawKernel. However saw no increase in performance.
                # Thus, we stick to ElementwiseKernel for now (easier to maintain)

                # CuPy ElementwiseKernel: call as function, returns values
                # Inputs must be flattened for raw indexing
                # size= is required since all inputs are 'raw'
                (
                    out_coords_x,
                    out_coords_y,
                    out_coords_z,
                    out_lat_x,
                    out_lat_z,
                    out_rad_dist_sq,
                    out_mask,
                ) = _calc_geo_dists_cupy_kernel(
                    rot_coords_subset.ravel(),
                    derived_rot_mat.ravel(),
                    source_point_bev,
                    lateral_cutoff_sq_over_sad_sq,
                    size=num_elements,
                )

                # Stack outputs to match expected format
                out_coords_temp = xp.stack((out_coords_x, out_coords_y, out_coords_z), axis=1)
                out_lat_dists = xp.stack((out_lat_x, out_lat_z), axis=1)

                ix[rad_depth_ix] = out_mask

                # Filter results
                # Optimization: sync once for sub_ix
                sub_ix = xp.nonzero(out_mask)[0]
                rad_distances_sq = xp.take(out_rad_dist_sq, sub_ix, axis=0)
                lat_dists = xp.take(out_lat_dists, sub_ix, axis=0)

                if array_api_compat.size(sub_ix) > 0:
                    iso_lat_dists = lat_dists / out_coords_temp[sub_ix, 1][:, None] * sad
                else:
                    iso_lat_dists = xp.empty_like(lat_dists)

                return ix, rad_distances_sq, lat_dists, iso_lat_dists

        # GPU Path (PyTorch)
        if pytorch_gpu_available() and "torch" in xp.__name__:
            if not self._cuda_kernel_used:
                logger.info("Using PyTorch kernel for geometric distance calculation.")
                self._cuda_kernel_used = True

            rot_coords_subset = xp.take(rot_coords_bev, rad_depth_ix, axis=0)
            num_elements = rot_coords_subset.shape[0]

            if num_elements > 0:
                if derived_rot_mat is None:
                    derived_rot_mat = xp.eye(3, dtype=a.dtype, device=self.device)

                # Ensure derived_rot_mat is correct type
                derived_rot_mat = xp.astype(derived_rot_mat, rot_coords_subset.dtype)

                lateral_cutoff_sq_over_sad_sq = (lateral_cutoff / sad) ** 2

                out_coords_temp, out_lat_dists, out_rad_dist_sq, out_mask = (
                    _calc_geo_dists_torch_kernel(
                        rot_coords_subset,
                        derived_rot_mat,
                        source_point_bev,
                        float(lateral_cutoff_sq_over_sad_sq),
                    )
                )
                # TODO: GPU synch happens here. Maybe this can be avoided?
                ix[rad_depth_ix] = out_mask

                # Filter results
                # Optimization: sync once for sub_ix
                sub_ix = xp.nonzero(out_mask)[0]
                rad_distances_sq = xp.take(out_rad_dist_sq, sub_ix, axis=0)
                lat_dists = xp.take(out_lat_dists, sub_ix, axis=0)

                if array_api_compat.size(sub_ix) > 0:
                    iso_lat_dists = lat_dists / out_coords_temp[sub_ix, 1][:, None] * sad
                else:
                    iso_lat_dists = xp.empty_like(lat_dists)

                return ix, rad_distances_sq, lat_dists, iso_lat_dists

        # Vectorized CPU/NumPy path (Fallback)
        if not self._cuda_kernel_used:
            if not getattr(self, "_fallback_logged", False):
                logger.info("Using vectorized path for geometric distance calculation.")
                self._fallback_logged = True

        rot_coords_subset = xp.take(rot_coords_bev, rad_depth_ix, axis=0)

        # 1. Matrix Multiplication (N,3) @ (3,3) -> (N,3)
        if derived_rot_mat is None:
            out_coords_temp = rot_coords_subset
        else:
            derived_rot_mat = xp.astype(derived_rot_mat, rot_coords_subset.dtype)
            if hasattr(xp, "matmul"):
                out_coords_temp = xp.matmul(rot_coords_subset, derived_rot_mat)
            else:
                out_coords_temp = rot_coords_subset @ derived_rot_mat

        # 2. Calculate Lateral Distances
        lx = out_coords_temp[:, 0] + source_point_bev[0]
        lz = out_coords_temp[:, 2] + source_point_bev[2]

        out_lat_dists = xp.stack((lx, lz), axis=1)

        # 3. Radial Distance Squared
        out_rad_dist_sq = lx * lx + lz * lz

        # 4. Mask
        lateral_cutoff_sq_over_sad_sq = (lateral_cutoff / sad) ** 2
        ry = out_coords_temp[:, 1]
        limit = lateral_cutoff_sq_over_sad_sq * ry * ry
        out_mask = out_rad_dist_sq <= limit

        # Check if backend supports fancy indexing (JAX and array_api_strict don't)
        xp_name = getattr(xp, "__name__", "")
        supports_fancy_setitem = "jax" not in xp_name and "strict" not in xp_name

        # Fancy indexing is supported for cupy/numpy/torch!
        # in some cases the kernels above are not compiled
        # So cupy and torch might rely on this path too.
        if supports_fancy_setitem:
            # Fast path: use fancy indexing scatter
            ix[rad_depth_ix] = out_mask
        else:
            # NOTE! This fallback is necessary for jax/strict. Above can be called for numpy/torch/cupy.
            # -> Above is approx 20~30% faster (proton dose calc)
            # Slow path for immutable arrays: compute mask on full array
            if derived_rot_mat is None:
                out_coords_full = rot_coords_bev
            elif hasattr(xp, "matmul"):
                out_coords_full = xp.matmul(rot_coords_bev, derived_rot_mat)
            else:
                out_coords_full = rot_coords_bev @ derived_rot_mat
            lx_full = out_coords_full[:, 0] + source_point_bev[0]
            lz_full = out_coords_full[:, 2] + source_point_bev[2]
            rad_dist_sq_full = lx_full * lx_full + lz_full * lz_full
            ry_full = out_coords_full[:, 1]
            limit_full = lateral_cutoff_sq_over_sad_sq * ry_full * ry_full
            cutoff_mask_full = rad_dist_sq_full <= limit_full
            ix = rad_depth_mask & cutoff_mask_full

        # for array API compatible indexing
        sub_ix = xp.nonzero(out_mask)[0]

        # Return radial distances squared
        rad_distances_sq = xp.take(out_rad_dist_sq, sub_ix, axis=0)

        # Lateral distances in X & Z
        lat_dists = xp.take(out_lat_dists, sub_ix, axis=0)

        if array_api_compat.size(sub_ix) > 0:
            # Lateral distances projected onto isocenter
            iso_lat_dists = lat_dists / out_coords_temp[sub_ix, 1][:, None] * sad
        else:
            iso_lat_dists = xp.empty_like(lat_dists)

        return ix, rad_distances_sq, lat_dists, iso_lat_dists
