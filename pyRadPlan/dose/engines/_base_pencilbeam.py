"""Base class for pencil beam dose calculation algorithms."""

from abc import abstractmethod
from typing import Any, Literal
import warnings
import logging
import time
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

import SimpleITK as sitk
import numpy as np
from scipy import sparse

from pyRadPlan.core import resample_image, np2sitk
from pyRadPlan.ct import CT, default_hlut
from pyRadPlan.cst import StructureSet
from pyRadPlan.stf import SteeringInformation
from pyRadPlan.geometry import get_beam_rotation_matrix
from pyRadPlan.raytracer import RayTracerSiddon

from ._base import DoseEngineBase


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

        self._computed_quantities = []
        self._effective_lateral_cutoff = None
        self._num_of_bixels_container = None
        self._rad_depth_cubes = []
        self._raytracer = None

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

                        # check if ray hit anything. If so, skip the computation
                        if all(not arr.size for arr in curr_ray["rad_depths"]):
                            continue

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

        return dij

    def _allocate_quantity_matrices(self, dij: dict[str, Any], names: list[str]):
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
                        dij[q_name].flat[i] = np.zeros(
                            (self.dose_grid.num_voxels, dij["num_of_beams"]), dtype=np.float32
                        )
                    else:
                        # We allocate raw csc sparse matrix structures using
                        # a rough estimat ebased on number of voxels and
                        # beamlets
                        est_nnz = np.fix(
                            5e-4 * self.dose_grid.num_voxels * self._num_of_columns_dij
                        ).astype(np.int64)
                        dij[q_name].flat[i] = {
                            "data": np.empty((est_nnz,), dtype=np.float32),
                            "indices": np.empty((est_nnz,), dtype=np.int64),
                            "indptr": np.zeros((self._num_of_columns_dij + 1,), dtype=np.int64),
                            "nnz": 0,
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

        # Convert voxel indices to real coordinates using iso center of beam i
        coords_v = self._vox_world_coords - beam_info["beam"]["iso_center"]
        coords_vdose_grid = self._vox_world_coords_dose_grid - beam_info["beam"]["iso_center"]

        # Get Rotation Matrix
        beam_info["rot_mat_system_T"] = get_beam_rotation_matrix(
            beam_info["beam"]["gantry_angle"], beam_info["beam"]["couch_angle"]
        )

        # Rotate coordinates (1st couch around Y axis, 2nd gantry movement)
        rot_coords_v = np.dot(coords_v, beam_info["rot_mat_system_T"])
        rot_coords_vdose_grid = np.dot(coords_vdose_grid, beam_info["rot_mat_system_T"])

        rot_coords_v -= beam_info["beam"]["source_point_bev"]
        rot_coords_vdose_grid -= beam_info["beam"]["source_point_bev"]

        # Calculate geometric distances
        geo_dist_vdose_grid = [
            np.sqrt(np.sum(rot_coords_vdose_grid**2, axis=1)) for _ in range(ct.num_of_ct_scen)
        ]

        # Calculate radiological depth cube
        logger.info("Calculating radiological depth cube...")

        start_time = time.time()

        self._raytracer.debug_core_performance = True
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
            rad_depth_vdose_grid = rad_depth_cube_dosegrid.ravel()[self._vdose_grid]

            # Find valid coordinates
            coord_is_valid = np.isfinite(rad_depth_vdose_grid)

            beam_info["valid_coords"][c] = coord_is_valid

            # TODO: !remove brakets once mutliscen is implemented
            beam_info["rad_depths"][c] = rad_depth_vdose_grid

        beam_info["geo_depths"] = geo_dist_vdose_grid
        beam_info["bev_coords"] = rot_coords_vdose_grid

        beam_info["valid_coords_all"] = np.any(np.vstack(beam_info["valid_coords"]), axis=0)

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
        ray = beam_info["beam"]["rays"][j]
        ray["beam_index"] = beam_info["beam_index"]
        ray["ray_index"] = j
        ray["iso_center"] = beam_info["beam"]["iso_center"]

        if "num_of_bixels_per_ray" not in beam_info["beam"]:
            ray["num_of_bixels"] = 1
        else:
            ray["num_of_bixels"] = beam_info["beam"]["num_of_bixels_per_ray"][j]

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

            ray_pos_bev = np.array([ray["ray_pos_bev"] for ray in beam["rays"]])
            target_points = np.array([ray["target_point"] for ray in beam["rays"]])

            alpha, _, rho, d12, _ = self._raytracer.trace_rays(
                beam["iso_center"],
                beam["source_point"].reshape((1, 3)),
                target_points,
            )

            # find rays that do not hit patients
            ix_nan = np.all(np.isnan(rho[0]), axis=1)
            if np.any(ix_nan):
                msg = f"{ix_nan.sum()} rays do not hit patient. Trying to fix afterwards..."
                if show_warning:
                    warnings.warn(msg)
                else:
                    logger.warning(msg)

            ix_ssd = -1 * np.ones_like(ix_nan, dtype=np.int64)
            ix_ssd[~ix_nan] = np.argmax(rho[0][~ix_nan] > density_threshold, axis=1)

            ssd = beam["sad"] * np.ones_like(ix_ssd, dtype=np.float32)

            ssd[~ix_nan] = d12[~ix_nan].squeeze() * alpha[~ix_nan, ix_ssd[~ix_nan]]

            # Now assign the ssd to all rays
            for j, ray in enumerate(beam["rays"]):
                if ix_nan[j]:  # Try to fix SSD by using SSD of closest neighboring ray
                    ray["SSD"] = self._closest_neighbor_ssd(ray_pos_bev, ssd, ray["ray_pos_bev"])
                else:
                    ray["SSD"] = float(ssd[j])

        else:
            raise ValueError(f"Invalid mode {mode} for SSD calculation")

        beam_info["beam"] = beam

    def _closest_neighbor_ssd(
        self, ray_pos_bev: np.ndarray, ssd: np.ndarray, curr_pos: np.ndarray
    ) -> float:
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
        distances = np.sum((ray_pos_bev - curr_pos) ** 2, axis=1)
        sorted_indices = np.argsort(distances)
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
        scen_num = self.mult_scen.scen_num(scen_idx)
        ct_scen = self.mult_scen.linear_mask[0][scen_num]

        # First, create a ray of the specific scenario to adapt rad depths
        scen_ray = ray.copy()
        scen_ray["rad_depths"] = scen_ray["rad_depths"][ct_scen]
        scen_ray["rad_depths"] = (1 + self.mult_scen.rel_range_shift[scen_num]) * scen_ray[
            "rad_depths"
        ] + self.mult_scen.abs_range_shift[scen_num]
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

    def _get_ray_geometry_from_beam(self, ray: dict[str], beam_info: dict[str]):
        ray["effective_lateral_cut_off"] = beam_info.get(
            "effective_lateral_cut_off", self._effective_lateral_cutoff
        )
        lateral_ray_cutoff = self._get_lateral_distance_from_dose_cutoff_on_ray(ray)

        # Ray tracing for beam i and ray j
        ix, radial_dist_sq, lat_dists, iso_lat_dists = self.calc_geo_dists(
            beam_info["bev_coords"],
            ray["source_point_bev"],
            ray["target_point_bev"],
            ray["sad"],
            beam_info["valid_coords_all"],
            lateral_ray_cutoff,
        )

        # Subindex given the relevant indices from the geometric distance calculation
        ray["valid_coords"] = [beam_ix & ix for beam_ix in beam_info["valid_coords"]]
        ray["ix"] = [self._vdose_grid[ix_in_grid] for ix_in_grid in ray["valid_coords"]]

        ray["radial_dist_sq"] = [radial_dist_sq[beam_ix[ix]] for beam_ix in ray["valid_coords"]]
        ray["lat_dists"] = [lat_dists[beam_ix[ix]] for beam_ix in ray["valid_coords"]]
        ray["iso_lat_dists"] = [iso_lat_dists[beam_ix[ix]] for beam_ix in ray["valid_coords"]]

        ray["valid_coords_all"] = np.any(np.vstack(ray["valid_coords"]), axis=0)

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
        """
        Fill the dose influence matrix (dij) with bixel contents.

        This is the last step in bixel dose calculation. It will fill all
        the computed quantities into sparse matrix containers.
        If forward calculation is active, accumulation into dense vectors
        will be performed instead.

        Parameters
        ----------
        bixel : dict
            The bixel data.
        dij : dict
            The dose influence matrix.
        stf : SteeringInformation
            The structure containing beam information.
            Unused in this base implementation.
        scen_idx : int
            The scenario index.
        curr_beam_idx : int
            The current beam index.
        curr_ray_idx : int
            The current ray index.
        curr_bixel_idx : int
            The current bixel index.
        bixel_counter : int
            The counter for the bixels.

        Returns
        -------
        dict
            The updated dose influence matrix.
        """
        # Only fill if we actually had bixel (indices) to compute
        if bixel and "ix" in bixel and bixel["ix"].any():
            sub_scen_idx = [
                np.unravel_index(scen_idx, self.mult_scen.scen_mask.shape)[i]
                for i in range(self.mult_scen.scen_mask.ndim)
            ]
            sub_scen_idx = tuple(sub_scen_idx)

            for q_name in self._computed_quantities:
                if self._calc_dose_direct:
                    # We accumulate the current bixel to the container
                    dij[q_name][sub_scen_idx][bixel["ix"], curr_beam_idx] += (
                        bixel["weight"] * bixel[q_name]
                    )
                else:
                    # first we fill the index pointer array
                    dij[q_name][sub_scen_idx]["indptr"][bixel_counter + 1] = (
                        dij[q_name][sub_scen_idx]["indptr"][bixel_counter] + bixel["ix"].size
                    )

                    # If we haven't preallocated enough, we need to expand the arrays
                    if (
                        dij[q_name][sub_scen_idx]["data"].size
                        < dij[q_name][sub_scen_idx]["nnz"] + bixel["ix"].size
                    ):
                        logger.debug("Resizing data and indices arrays for %s...", q_name)

                        # We estimate the required size by the remaining
                        # number of beamlets / columns in the sparse matrix
                        # 1 additional beamlet is added
                        shape_resize = (self._num_of_columns_dij - bixel_counter) * (
                            bixel["ix"].size + 1
                        )
                        shape_resize += dij[q_name][sub_scen_idx]["data"].size
                        dij[q_name][sub_scen_idx]["data"].resize((shape_resize,), refcheck=False)
                        dij[q_name][sub_scen_idx]["indices"].resize(
                            (shape_resize,), refcheck=False
                        )

                    # Fill the corresponding values and indices using indptr
                    dij[q_name][sub_scen_idx]["data"][
                        dij[q_name][sub_scen_idx]["indptr"][bixel_counter] : dij[q_name][
                            sub_scen_idx
                        ]["indptr"][bixel_counter + 1]
                    ] = bixel[q_name]
                    dij[q_name][sub_scen_idx]["indices"][
                        dij[q_name][sub_scen_idx]["indptr"][bixel_counter] : dij[q_name][
                            sub_scen_idx
                        ]["indptr"][bixel_counter + 1]
                    ] = bixel["ix"]

                    # Store how many nnzs we actually have in the matrix
                    dij[q_name][sub_scen_idx]["nnz"] += bixel["ix"].size

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
                # Loop over all used quantities
                for q_name in self._computed_quantities:
                    if not self._calc_dose_direct:
                        # tmp_matrix = cast(sparse.lil_matrix, dij[q_name].flat[i])
                        # tmp_matrix = tmp_matrix.tocsr().T
                        data_dict = dij[q_name].flat[i]

                        # Resize to the actual number of non-zero elements
                        data_dict["data"].resize((data_dict["nnz"],), refcheck=False)
                        data_dict["indices"].resize((data_dict["nnz"],), refcheck=False)

                        # Create the matrix, avoid copies
                        tmp_matrix = sparse.csc_array(
                            (data_dict["data"], data_dict["indices"], data_dict["indptr"]),
                            shape=(self.dose_grid.num_voxels, self._num_of_columns_dij),
                            dtype=np.float32,
                            copy=False,
                        )

                        # Do we need this?
                        tmp_matrix.eliminate_zeros()

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

    @staticmethod
    def calc_geo_dists(
        rot_coords_bev, source_point_bev, target_point_bev, sad, rad_depth_ix, lateral_cutoff
    ):
        """
        Calculate geometric distances for dose calculation.

        Parameters
        ----------
        rot_coords_bev : ndarray
            Coordinates in beam's eye view (BEV) of the voxels where ray tracing results are
            available.
        source_point_bev : ndarray
            Source point in voxel coordinates in BEV.
        target_point_bev : ndarray
            Target point in voxel coordinates in BEV.
        sad : float
            Source-to-axis distance.
        rad_depth_ix : ndarray
            Subset of voxels for which radiological depth calculations are available.
        lateral_cutoff : float
            Lateral cutoff specifying the neighborhood for dose calculations.

        Returns
        -------
        ix : ndarray
            Indices of voxels where dose influence is computed.
        rad_distances_sq : ndarray
            Squared radial distances to the central ray.
        lat_dists : ndarray
            Lateral distances to the central ray (in X & Z).
        iso_lat_dists : ndarray
            Lateral distances to the central ray projected onto the isocenter plane.
        rot_coords_bev, source_point_bev, target_point_bev, sad, rad_depth_ix, lateral_cutoff
        """
        # Put [0 0 0] position in the source point for beamlet who passes through isocenter
        a = -source_point_bev.T

        # Normalize the vector
        a = a / np.linalg.norm(a)

        # Put [0 0 0] position in the source point for a single beamlet
        b = (target_point_bev - source_point_bev).T

        # Normalize the vector
        b = b / np.linalg.norm(b)

        # Define rotation matrix
        if np.all(a == b):
            rot_coords_temp = rot_coords_bev[rad_depth_ix, :]
        else:

            def ssc(v: np.ndarray) -> np.ndarray:
                """Skew-symmetric cross product matrix."""
                return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

            derived_rot_mat = (
                np.eye(3)
                + ssc(np.cross(a, b))
                + np.dot(ssc(np.cross(a, b)), ssc(np.cross(a, b)))
                * (1 - np.dot(a, b))
                / (np.linalg.norm(np.cross(a, b)) ** 2)
            )
            rot_coords_temp = np.dot(rot_coords_bev[rad_depth_ix, :], derived_rot_mat)

        # Put [0 0 0] position CT in center of the beamlet
        lat_dists = rot_coords_temp[:, [0, 2]] + source_point_bev[[0, 2]]

        # Check if radial distance exceeds lateral cutoff (projected to iso center)
        rad_distances_sq = np.sum(lat_dists**2, axis=1)
        subset_mask = rad_distances_sq <= (lateral_cutoff / sad) ** 2 * rot_coords_temp[:, 1] ** 2

        ix = rad_depth_ix.copy()
        # Apply mask for return quantities
        ix[rad_depth_ix] = subset_mask

        # Return radial distances squared
        rad_distances_sq = rad_distances_sq[subset_mask]

        # Lateral distances in X & Z
        lat_dists = lat_dists[subset_mask, :]

        # Lateral distances projected onto isocenter
        iso_lat_dists = lat_dists / rot_coords_temp[subset_mask, 1][:, np.newaxis] * sad

        return ix, rad_distances_sq, lat_dists, iso_lat_dists
