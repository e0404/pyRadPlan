"""Base class for pencil beam dose calculation algorithms."""

from abc import abstractmethod
from contextlib import nullcontext
from typing import Annotated, Any, ClassVar, Optional
import warnings
import logging
import time
from tqdm.contrib.logging import logging_redirect_tqdm

import SimpleITK as sitk
import numpy as np
from pydantic import Field
from scipy import sparse
import array_api_compat

from pyRadPlan.core import (
    AlgorithmParameterMetadata,
    ArrayBackendGridResampler,
    Grid,
    resample_image,
    np2sitk,
)
from pyRadPlan.ct import CT, default_hlut
from pyRadPlan.cst import StructureSet
from pyRadPlan.stf import SteeringInformation
from pyRadPlan.geometry import get_beam_rotation_matrix
from pyRadPlan.raytracer import RayTracerSiddon

from pyRadPlan.scenarios import AFFECTS_GEOMETRY, Scenario

from ._base import DoseEngineBase, ct_phase_images
from ...core.xp_utils.typing import Array
from ...core.xp_utils import cupy_available, pytorch_gpu_available
from ...core.xp_utils import to_namespace, to_numpy, free_gpu_memory
from ._sparse_output import sparse_index_dtype, ensure_index_capacity
from ._batching import BixelBatchProcessor, BixelBatchScheduler, BixelDelivery
from .kernels import _calc_geo_dists_cupy_kernel, _calc_geo_dists_torch_kernel


logger = logging.getLogger(__name__)


class PencilBeamEngineAbstract(DoseEngineBase):
    """
    Abstract class representing a Pencil Beam Engine.

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

        Both flags mirror the properties of the same name on the ion stf generators and are
        checked against the stf in ``_check_stf_density_consistency``.
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
    _apply_scenario_to_ray(ray, scenario)
        Selects a scenario's CT phase on a ray and applies its range shift.
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

    keep_rad_depth_cubes: bool = False
    geometric_lateral_cutoff: Annotated[
        float, Field(gt=0.0, description="Lateral geometric cut-off [mm]")
    ] = 50.0
    dosimetric_lateral_cutoff: Annotated[
        float, Field(gt=0.0, le=1.0, description="Relative dosimetric cut-off")
    ] = 0.9950
    ssd_density_threshold: Annotated[
        float, Field(gt=0.0, description="Density threshold for SSD computation")
    ] = 0.0500
    use_given_eq_density_cube: bool = False
    ignore_outside_densities: bool = False
    supported_uncertainties: ClassVar[tuple[str, ...]] = ("ct_phase", "setup", "range")
    # The density settings are adjusted per calculation (no ct.cube, default HLUT)
    _per_calculation_settings: ClassVar[tuple[str, ...]] = (
        *DoseEngineBase._per_calculation_settings,
        "use_given_eq_density_cube",
        "hlut",
    )
    # Radiation modes whose stf generators select energies by tracing the densities
    _density_dependent_stf_modes: ClassVar[frozenset[str]] = frozenset(
        {"protons", "helium", "carbon", "oxygen"}
    )
    trace_on_dose_grid: bool = True
    assumed_sparsity: Annotated[float, Field(gt=0.0, le=1.0)] = 5e-4
    cube_wed: Annotated[
        Optional[sitk.Image], AlgorithmParameterMetadata(configurable=False, kind="data")
    ] = None
    hlut: Annotated[
        Optional[np.ndarray], AlgorithmParameterMetadata(configurable=False, kind="data")
    ] = None

    # Every derived engine must declare both flags.
    _dij_guarantee_canonical: ClassVar[bool]
    """``True`` iff the engine guarantees that every assembled CSC column already has
    sorted, unique row indices (no duplicate entries, ascending order)"""

    _dij_guarantee_nonzero: ClassVar[bool]
    """``True`` iff the engine guarantees that it never writes structural zero
    entries into the dose influence matrix"""

    def __init__(self, pln=None):
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

    def _create_bixel_batch_processor(self, dij, stf) -> Optional[BixelBatchProcessor]:
        """Opt into staged execution for the initialized beam, or retain scalar execution.

        Called after beam preparation and before ray traversal. A fresh processor
        is used for each beam and geometry group. Its pending work is drained
        before beam geometry or converted kernels can be released/replaced.
        Existing/custom engines retain `_compute_bixel` unless they opt in here.
        """
        return None

    def _collect_bixel_batch(self, batch, dij, stf):
        """Deliver packet results through the existing sparse/forward collector."""
        for work in batch:
            identity = work.delivery
            self._fill_dij(
                work.payload,
                dij,
                stf,
                identity.scenario,
                identity.beam,
                identity.ray,
                identity.bixel,
                identity.column,
            )

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

        groups = self._geometry_groups()
        num_computed = sum(len(scen_ix) for _, scen_ix in groups)
        scenario_progress = (
            self.progress("Scenarios", total=num_computed, console=True, unit="scen")
            if num_computed > 1
            else nullcontext(None)
        )
        with scenario_progress as scenarios_done:
            for setup_shift, scen_ix in groups:
                scen_stf = self._shift_stf(stf, setup_shift)
                self._init_geometry_group()

                if len(self._scenarios) > 1:
                    logger.info(
                        "Geometry scenario group with setup shift %s: scenarios %s",
                        setup_shift.tolist(),
                        scen_ix,
                    )

                bixel_counter = 0

                # Loop over all beams
                with logging_redirect_tqdm():
                    for i in self.track(range(dij["num_of_beams"]), name="Beam", unit="b"):
                        # Initialize Beam Geometry
                        t = time.time()
                        curr_beam = self._init_beam(dij, ct, cst, scen_stf, i)
                        logger.info("Beam %d initialized in %f seconds.", i + 1, time.time() - t)

                        processor = self._create_bixel_batch_processor(dij, scen_stf)
                        context = (
                            nullcontext(None)
                            if processor is None
                            else BixelBatchScheduler(processor)
                        )
                        with context as batcher:
                            # Empty supports still emit columns and delivery identities.
                            for j in self.track(
                                range(curr_beam["beam"]["num_of_rays"]), name="Ray", unit="r"
                            ):
                                curr_ray = self._init_ray(curr_beam, j)
                                for s in scen_ix:
                                    scen_ray = self._apply_scenario_to_ray(
                                        curr_ray, self._scenarios[s]
                                    )
                                    for k in range(curr_ray["num_of_bixels"]):
                                        if batcher is None:
                                            curr_bixel = self._compute_bixel(scen_ray, k)
                                            self._fill_dij(
                                                curr_bixel,
                                                dij,
                                                scen_stf,
                                                s,
                                                i,
                                                j,
                                                k,
                                                bixel_counter + k,
                                            )
                                        else:
                                            batcher.submit(
                                                scen_ray,
                                                k,
                                                BixelDelivery(s, i, j, k, bixel_counter + k),
                                            )
                                bixel_counter += curr_ray["num_of_bixels"]

                        # Release the processor's per-beam state before GPU cleanup
                        # and before the next beam starts allocating geometry.
                        del batcher, context, processor

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

                if scenarios_done is not None:
                    scenarios_done.advance(len(scen_ix))

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

        # Start from scratch when the engine is reused for another calculation
        self._rad_depth_cubes = []

        dij = super()._init_dose_calc(ct, cst, stf)
        self._depth_resampler = None

        # calculate rED or rSP from HU or take provided wedCube
        if self.use_given_eq_density_cube and getattr(ct, "cube", None) is None:
            logging.warning(
                "HU Conversion requested to be omitted but no ct.cube exists! "
                "Will override and do the conversion anyway!"
            )
            self.use_given_eq_density_cube = False

        # The stf's geometry (for ions: its energy layers) was placed on a density cube of
        # its own. Warn before tracing a differently built one.
        self._check_stf_density_consistency(stf)

        if self.use_given_eq_density_cube:
            wed_cubes = ct_phase_images(ct.cube)
            logging.info("Omitting HU to rED/rSP conversion and using existing ct.cube!\n")
        else:
            # TODO: obtain correct hlut
            if self.hlut is None:
                self.hlut = default_hlut()
            if ct.num_of_ct_scen == 1:
                wed_cubes = [ct.compute_wet(self.hlut)]
            else:
                wed_cubes = [
                    CT(cube_hu=phase).compute_wet(self.hlut)
                    for phase in ct_phase_images(ct.cube_hu)
                ]

        # Ignore densities outside of contours
        if self.ignore_outside_densities:  # TODO: default = None (not tested yet)
            mask_image = np2sitk.linear_indices_to_sitk_mask(
                self._vct_grid, wed_cubes[0], order="numpy"
            )
            wed_cubes = [sitk.Mask(cube, mask_image, outsideValue=0) for cube in wed_cubes]

        self.cube_wed = wed_cubes[0] if len(wed_cubes) == 1 else sitk.JoinSeries(wed_cubes)

        # Allocate memory for quantity containers
        dij = self._allocate_quantity_matrices(dij, ["physical_dose"])

        # Initialize ray-tracer with one cube per CT phase
        if self.trace_on_dose_grid:
            wed_cubes_trace = [
                resample_image(
                    input_image=cube,
                    interpolator=sitk.sitkLinear,
                    target_grid=self.dose_grid,
                )
                for cube in wed_cubes
            ]
        else:
            wed_cubes_trace = wed_cubes

        self._raytracer = RayTracerSiddon(wed_cubes_trace)
        self._raytracer.lateral_cut_off = self.geometric_lateral_cutoff

        return dij

    def _check_stf_density_consistency(self, stf: SteeringInformation) -> None:
        """
        Warn if the stf's density assumptions are unknown or differ from the engine's.

        The ion stf generators record which density cube they selected the energy layers
        on. If the engine now traces a differently built cube, the Bragg peaks no longer
        land where the energy selection put them.

        Parameters
        ----------
        stf : SteeringInformation
            The steering information to check against the engine configuration.

        Notes
        -----
        Only beams of radiation modes in ``_density_dependent_stf_modes`` are checked,
        since photon and VHEE beamlet placement does not depend on densities. Beams not
        recording a setting (``None``, e.g. an stf imported from matRad) cause a separate
        warning, as their consistency cannot be verified.
        """

        beams = [
            beam for beam in stf.beams if beam.radiation_mode in self._density_dependent_stf_modes
        ]
        if not beams:
            return

        names = ("use_given_eq_density_cube", "ignore_outside_densities")
        unknown = [name for name in names if any(getattr(b, name) is None for b in beams)]
        if unknown:
            settings = ", ".join(f"{name}={getattr(self, name)}" for name in unknown)
            warnings.warn(
                f"stf does not record {' and '.join(unknown)} (e.g. because it was imported), "
                f"so its energy selection cannot be verified against the dose engine "
                f"({settings}). Make sure the stf was generated with the same settings."
            )

        for name in names:
            engine_value = getattr(self, name)
            stf_values = {getattr(beam, name) for beam in beams} - {None}
            if stf_values and stf_values != {engine_value}:
                warnings.warn(
                    f"stf was generated with {name}="
                    f"{'/'.join(str(value) for value in sorted(stf_values))}, but the dose "
                    f"engine uses {name}={engine_value}. Beamlet placement and dose "
                    "calculation will see different densities."
                )

    def _allocate_quantity_matrices(self, dij: dict[str, Any], names: list[str]):
        device = "cpu"
        xp = np  # self.xp
        # initialize shared structure container on first call
        num_scenarios = len(self._scenarios)
        if "shared_structure" not in dij:
            dij["shared_structure"] = np.empty((num_scenarios,), dtype=object)

        # Loop over all requested quantities
        for q_name in names:
            # Create dij list for each quantity
            dij[q_name] = np.empty((num_scenarios,), dtype=object)

            # Derived scenarios are not computed and keep an empty slot
            for i in self._scenarios.computed_indices():
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
                        index_dtype = sparse_index_dtype(
                            self.dose_grid.num_voxels, self._num_of_columns_dij, int(est_nnz)
                        )
                        dij["shared_structure"].flat[i] = {
                            "indices": xp.empty((est_nnz,), dtype=index_dtype, device=device),
                            "indptr": xp.zeros(
                                (self._num_of_columns_dij + 1,), dtype=index_dtype, device=device
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

    def _sample_radiological_depths(self, rad_depth_cubes):
        """Select dose-grid depths on the backend; export images only when requested."""
        selected = []
        xp = self.xp
        for rad_depth_cube in rad_depth_cubes:
            if self.trace_on_dose_grid:
                rad_depth_arr = rad_depth_cube
                rad_depth_vdose_grid = xp.take(
                    xp.reshape(rad_depth_arr, (-1,)), self._vdose_grid, axis=0
                )
            else:
                if self._depth_resampler is None:
                    indices = (
                        xp.arange(self.dose_grid.num_voxels, dtype=xp.int64, device=self.device)
                        if self.keep_rad_depth_cubes
                        else self._vdose_grid
                    )
                    self._depth_resampler = ArrayBackendGridResampler(
                        Grid.from_sitk_image(self._raytracer.cubes[0]), self.dose_grid, indices
                    )
                sampled = self._depth_resampler(rad_depth_cube)
                if self.keep_rad_depth_cubes:
                    rad_depth_arr = xp.reshape(sampled, tuple(reversed(self.dose_grid.dimensions)))
                    rad_depth_vdose_grid = xp.take(sampled, self._vdose_grid, axis=0)
                else:
                    rad_depth_vdose_grid = sampled
            if self.keep_rad_depth_cubes:
                image = sitk.GetImageFromArray(to_numpy(rad_depth_arr))
                image.SetSpacing(tuple(self.dose_grid.resolution_vector))
                image.SetOrigin(tuple(self.dose_grid.origin))
                image.SetDirection(tuple(self.dose_grid.direction.ravel()))
                self._rad_depth_cubes.append(image)
            rad_depth_vdose_grid = xp.astype(
                rad_depth_vdose_grid, self._vox_world_coords.dtype, copy=False
            )

            selected.append(rad_depth_vdose_grid)
        return selected

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
        patient_coords = rot_coords_vdose_grid if self.trace_on_dose_grid else rot_coords_v
        rad_depth_cubes = self._raytracer.trace_cubes_array(
            beam_info["beam"], patient_coords=patient_coords
        )
        self._raytracer.debug_core_performance = False

        # TODO: add universal time-debugging method
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info("Elapsed time for rayTracing per Beam: %f seconds", elapsed_time)

        beam_info["valid_coords"] = [None] * len(rad_depth_cubes)
        beam_info["rad_depths"] = [None] * len(rad_depth_cubes)

        for c, rad_depth_vdose_grid in enumerate(
            self._sample_radiological_depths(rad_depth_cubes)
        ):
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
        density_threshold: float = 0.05,
        show_warning: bool = True,
    ):
        """
        Compute the SSD (Source to Surface Distance) of each ray for every CT phase.

        Parameters
        ----------
        beam_info : dict
            Beam Information, will be modified to include SSD: ``ray["SSD"]`` is a list with
            one value per CT phase, in the order of the CT's phases.
        ct : CT
            The CT object.
        density_threshold : float, optional
            Value determining the skin threshold.
        show_warning : bool, optional
            Flag to show warnings.
        """

        beam = beam_info["beam"]

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
        d12 = xp.reshape(to_namespace(xp, d12, device=self.device), (-1,))
        # prevent d12 from being 0d array which causes issues with indexing later on
        if d12.ndim == 0:
            d12 = xp.reshape(d12, (1,))

        for ray in beam["rays"]:
            ray["SSD"] = []

        sad = xp.asarray(beam["sad"], dtype=xp.float32, device=self.device)
        for rho_phase in rho:
            # rays that do not hit the patient in this phase have no density along them
            ix_nan = xp.all(xp.isnan(rho_phase), axis=1)
            hit = ~ix_nan

            if xp.any(ix_nan):
                msg = (
                    f"{xp.count_nonzero(ix_nan)} rays do not hit patient. "
                    "Trying to fix afterwards..."
                )
                if show_warning:
                    warnings.warn(msg)
                else:
                    logger.warning(msg)

            if not xp.any(hit):
                raise ValueError(
                    "Error in SSD calculation: no ray of the beam hits the patient. Check the "
                    "beam geometry and the setup shifts of the scenarios."
                )

            # first voxel above the skin threshold along every ray (NaN compares False);
            # integers for argmax, since some backends have no boolean argmax
            cond = xp.where(
                rho_phase > density_threshold,
                xp.asarray(1, dtype=xp.int64),
                xp.asarray(0, dtype=xp.int64),
            )
            idx = xp.reshape(xp.astype(xp.argmax(cond, axis=1), xp.int64), (-1, 1))
            alpha_sel = xp.reshape(xp.take_along_axis(alpha, idx, axis=1), (-1,))
            ssd_vals = xp.astype(d12, xp.float32) * xp.astype(alpha_sel, xp.float32)
            ssd = xp.where(hit, ssd_vals, sad)

            # Transfer Python metadata once per CT phase, rather than once per ray.
            missing_ssd = to_numpy(ix_nan)
            host_ssd = to_numpy(ssd)
            for j, ray in enumerate(beam["rays"]):
                if missing_ssd[j]:  # take the SSD of the closest ray that hits the patient
                    ray["SSD"].append(
                        self._closest_neighbor_ssd(ray_pos_bev, ssd, hit, ray["ray_pos_bev"])
                    )
                else:
                    ray["SSD"].append(float(host_ssd[j]))

        beam_info["beam"] = beam

    def _closest_neighbor_ssd(
        self, ray_pos_bev: Array, ssd: Array, hit: Array, curr_pos: Array
    ) -> float:
        """
        SSD of the closest ray that hits the patient.

        Parameters
        ----------
        ray_pos_bev : Array
            ``(num_rays, 3)`` ray positions in the beam's eye view.
        ssd : Array
            SSD per ray.
        hit : Array
            Boolean per ray, whether it hits the patient (its SSD is valid).
        curr_pos : Array
            Position of the ray whose SSD is missing.

        Returns
        -------
        float
            SSD value of the closest ray hitting the patient.

        Raises
        ------
        ValueError
            No ray of the beam hits the patient.
        """
        xp = array_api_compat.array_namespace(ray_pos_bev, ssd, curr_pos)
        distances = xp.sum((ray_pos_bev - xp.asarray(curr_pos)) ** 2, axis=1)
        for ix in xp.argsort(distances):
            if bool(hit[ix]):
                return float(ssd[ix])
        raise ValueError(
            "Error in SSD calculation: no ray of the beam hits the patient, so there is no "
            "neighboring ray to take the SSD from."
        )

    def _geometry_groups(self) -> list[tuple[np.ndarray, list[int]]]:
        """
        Group the computed scenarios by the beam geometry their ray tracing needs.

        CT phases are not grouped on, since every beam traces all phases at once and
        :meth:`_apply_scenario_to_ray` selects a scenario's phase from the ray.

        Returns
        -------
        list[tuple[np.ndarray, list[int]]]
            Per group the setup (isocenter) shift in mm and the flat indices of its computed
            scenarios. Derived scenarios are left out.
        """
        scen_set = self._scenarios
        keys = tuple(
            name for name in scen_set.dimensions_affecting(AFFECTS_GEOMETRY) if name != "ct_phase"
        )
        groups = []
        for values, indices in scen_set.group_by(*keys):
            computed = [i for i in indices if scen_set.bases[i] is None]
            if computed:
                shift = values.get("setup", np.zeros(3))
                groups.append((np.asarray(shift, dtype=np.float64), computed))
        return groups

    def _init_geometry_group(self) -> None:
        """Reset state derived from the beam geometry of a previous geometry group."""

    @staticmethod
    def _shift_stf(stf: SteeringInformation, shift: np.ndarray) -> SteeringInformation:
        """
        Copy of ``stf`` with every beam's isocenter moved by ``shift``.

        Parameters
        ----------
        stf : SteeringInformation
            The steering information; it is not modified.
        shift : np.ndarray
            Isocenter shift in mm.

        Returns
        -------
        SteeringInformation
            The shifted deep copy.
        """
        scen_stf = stf.model_copy(deep=True)
        for beam in scen_stf.beams:
            beam.iso_center = beam.iso_center + shift
        return scen_stf

    def _apply_scenario_to_ray(self, ray: dict, scenario: Scenario) -> dict:
        """
        Select a scenario's CT phase from a ray and apply its range shift.

        Parameters
        ----------
        ray : dict
            The ray data with one entry per CT phase for the geometric quantities.
        scenario : Scenario
            The scenario; a missing ``range`` dimension counts as no range shift.

        Returns
        -------
        dict
            A shallow copy of the ray for the scenario, with radiological depths scaled as
            ``(1 + rel) * d + abs`` and clipped at zero.
        """
        ct_phase = scenario.ct_phase
        scen_ray = ray.copy()
        for key in (
            "rad_depths",
            "radial_dist_sq",
            "ix",
            "geo_depths",
            "lat_dists",
            "iso_lat_dists",
            "SSD",
        ):
            if key in scen_ray:
                scen_ray[key] = scen_ray[key][ct_phase]

        range_shift = scenario.values.get("range")
        if range_shift is None:
            abs_shift, rel_shift = np.float64(0.0), np.float64(0.0)
        else:
            abs_shift, rel_shift = range_shift[0], range_shift[1]

        rad_depths = scen_ray["rad_depths"]
        xp = array_api_compat.array_namespace(rad_depths)
        rad_depths = (1 + xp.asarray(rel_shift, device=self.device)) * rad_depths + xp.asarray(
            abs_shift, device=self.device
        )
        if abs_shift < 0:
            rad_depths = xp.where(rad_depths < 0, xp.zeros_like(rad_depths), rad_depths)
        scen_ray["rad_depths"] = rad_depths

        return scen_ray

    def _prepare_ray_transforms(self, source: Array, targets: Array) -> dict:
        """Prepare backend rotations together and download only alignment flags.

        Inputs have shapes (3,) and (N, 3), already in the coordinate dtype/device.
        The caller owns this state for one beam/shift. Aligned rays keep the exact
        identity path, including the existing 1e-7 directional tolerance.
        """
        xp = array_api_compat.array_namespace(source, targets)
        device = array_api_compat.device(source)
        if hasattr(xp, "linalg"):
            norm = xp.linalg.vector_norm
            cross = xp.linalg.cross
        else:

            def norm(arr, *, axis=-1, keepdims=False):
                return xp.sqrt(xp.sum(arr**2, axis=axis, keepdims=keepdims))

            def cross(x, y):
                return xp.stack(
                    [
                        x[..., 1] * y[..., 2] - x[..., 2] * y[..., 1],
                        x[..., 2] * y[..., 0] - x[..., 0] * y[..., 2],
                        x[..., 0] * y[..., 1] - x[..., 1] * y[..., 0],
                    ],
                    axis=-1,
                )

        a = -source
        a = a / norm(a)
        b = targets - source
        b = b / norm(b, axis=-1, keepdims=True)
        aligned = xp.all(xp.abs(a - b) < 1e-7, axis=-1)
        cross_ab = cross(xp.broadcast_to(a, b.shape), b)
        eps = xp.asarray(
            [
                [[0, 0, 0], [0, 0, -1], [0, 1, 0]],
                [[0, 0, 1], [0, 0, 0], [-1, 0, 0]],
                [[0, -1, 0], [1, 0, 0], [0, 0, 0]],
            ],
            dtype=source.dtype,
            device=device,
        )
        skew = xp.tensordot(cross_ab, eps, axes=1)
        denominator = norm(cross_ab, axis=-1) ** 2
        # The aligned case never evaluated this division in the per-ray path.
        denominator = xp.where(aligned, xp.ones_like(denominator), denominator)
        identity = xp.eye(3, dtype=source.dtype, device=device)
        rotations = (
            identity
            + skew
            + (skew @ skew) * (1 - xp.vecdot(a, b))[:, None, None] / denominator[:, None, None]
        )
        rotations = xp.where(aligned[:, None, None], identity, rotations)
        return {"source": source, "rotations": rotations, "aligned": to_numpy(aligned)}

    def _prepare_beam_geometry(self, beam_info: dict) -> dict:
        """Compact the final beam-valid coordinates once, after modality depth limits.

        Indices are ordered positions in the engine's selected dose voxels, not
        flattened dose-grid rows. The original beam arrays remain available to
        modality-specific initialization. Cache lifetime is one beam/shift.
        """
        xp = array_api_compat.array_namespace(beam_info["bev_coords"])
        indices = xp.nonzero(beam_info["valid_coords_all"])[0]
        coords = beam_info["bev_coords"]
        beam = beam_info["beam"]
        source = xp.astype(xp.reshape(beam["source_point_bev"], (3,)), coords.dtype)
        targets = (
            xp.astype(
                xp.stack([xp.reshape(r["target_point_bev"], (3,)) for r in beam["rays"]]),
                coords.dtype,
            )
            if beam["rays"]
            else xp.empty((0, 3), dtype=coords.dtype, device=array_api_compat.device(coords))
        )
        return {
            "transforms": self._prepare_ray_transforms(source, targets),
            "indices": indices,
            "coords": xp.take(beam_info["bev_coords"], indices, axis=0),
            "scenario_valid": [
                xp.take(mask, indices, axis=0) for mask in beam_info["valid_coords"]
            ]
            if len(beam_info["valid_coords"]) > 1
            else None,
        }

    def _get_ray_geometry_from_beam(self, ray: dict[str], beam_info: dict[str]):
        # Lazy preparation runs after the complete subclass _init_beam, including
        # particle depth limits. It is shared by all subsequent rays in this beam.
        if "_ray_geometry" not in beam_info:
            beam_info["_ray_geometry"] = self._prepare_beam_geometry(beam_info)
        prepared = beam_info["_ray_geometry"]
        xp = array_api_compat.array_namespace(prepared["coords"])
        lateral_ray_cutoff = self._get_lateral_distance_from_dose_cutoff_on_ray(ray)
        transforms = prepared["transforms"]
        ray_index = ray["ray_index"]
        transform = (
            transforms["source"],
            transforms["rotations"][ray_index, ...],
            transforms["aligned"][ray_index],
        )
        selected, radial_dist_sq, lat_dists, iso_lat_dists = self._calc_geo_dists_on_valid_coords(
            prepared["coords"],
            ray["source_point_bev"],
            ray["target_point_bev"],
            ray["sad"],
            float(lateral_ray_cutoff),
            prepared_transform=transform,
        )
        positions = xp.take(prepared["indices"], selected, axis=0)
        for key in (
            "ix",
            "radial_dist_sq",
            "lat_dists",
            "iso_lat_dists",
            "geo_depths",
            "rad_depths",
        ):
            ray[key] = []
        for scenario in range(len(beam_info["rad_depths"])):
            scenario_positions = positions
            radial, lateral, projected = radial_dist_sq, lat_dists, iso_lat_dists
            if prepared["scenario_valid"] is not None:
                valid = xp.take(prepared["scenario_valid"][scenario], selected, axis=0)
                within_ray = xp.nonzero(valid)[0]
                scenario_positions = xp.take(positions, within_ray, axis=0)
                radial = xp.take(radial_dist_sq, within_ray, axis=0)
                lateral = xp.take(lat_dists, within_ray, axis=0)
                projected = xp.take(iso_lat_dists, within_ray, axis=0)
            ray["ix"].append(xp.take(self._vdose_grid, scenario_positions, axis=0))
            ray["radial_dist_sq"].append(radial)
            ray["lat_dists"].append(lateral)
            ray["iso_lat_dists"].append(projected)
            for key in ("geo_depths", "rad_depths"):
                ray[key].append(xp.take(beam_info[key][scenario], scenario_positions, axis=0))

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

        if not self._calc_dose_direct:
            shared_struct = dij["shared_structure"][scen_idx]
            # Advance column pointer even when ix_size == 0 to keep CSC valid
            # Add in Python precision, then promote BEFORE storing the next pointer.
            start = int(shared_struct["indptr"][bixel_counter])
            end = start + ix_size
            ensure_index_capacity(shared_struct, end)
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
                        dij[q][scen_idx]["data"].resize((new_size,), refcheck=False)
                bixel["ix"] = to_numpy(bixel["ix"])
                shared_struct["indices"][start:end] = bixel["ix"]
                shared_struct["nnz"] += ix_size

        for q_name in self._computed_quantities:
            if self._calc_dose_direct:
                if ix_size > 0:
                    bixel["ix"] = to_numpy(bixel["ix"])
                    bixel[q_name] = to_numpy(bixel[q_name])
                    dij[q_name][scen_idx][bixel["ix"], curr_beam_idx] += (
                        bixel["weight"] * bixel[q_name]
                    )
            else:
                data_dict = dij[q_name][scen_idx]

                if ix_size > 0:
                    start = dij["shared_structure"][scen_idx]["indptr"][bixel_counter]
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
        for i in self._scenarios.computed_indices():
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
            dij["rad_depth_cubes"] = [
                sitk.GetArrayFromImage(cube) for cube in self._rad_depth_cubes
            ]

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

        This compatibility interface returns a full boolean selection mask.
        Dose engines use compact beam coordinates and integer selections internally.
        Both paths share the existing CuPy/PyTorch kernels or Array API arithmetic.

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
            Boolean mask of voxels where dose influence is computed.
        rad_distances_sq : Array
            Squared radial distances to the central ray.
        lat_dists : Array
            Lateral distances to the central ray (in X & Z).
        iso_lat_dists : Array
            Lateral distances to the central ray projected onto the isocenter plane.
        """

        xp = array_api_compat.array_namespace(rot_coords_bev, rad_depth_mask)
        valid = xp.nonzero(rad_depth_mask)[0]
        selected, radial, lateral, projected = self._calc_geo_dists_on_valid_coords(
            xp.take(rot_coords_bev, valid, axis=0),
            source_point_bev,
            target_point_bev,
            sad,
            lateral_cutoff,
        )
        selected = xp.take(valid, selected, axis=0)
        # Rebuild the historical full boolean mask only for callers of this API.
        # A sentinel permits empty selections without mutable/fancy scatter indexing.
        rows = xp.arange(
            rad_depth_mask.shape[0], dtype=valid.dtype, device=array_api_compat.device(valid)
        )
        padded = xp.concat(
            (
                selected,
                xp.asarray(
                    [rad_depth_mask.shape[0]],
                    dtype=valid.dtype,
                    device=array_api_compat.device(valid),
                ),
            )
        )
        mask = xp.take(padded, xp.searchsorted(padded, rows), axis=0) == rows
        return mask, radial, lateral, projected

    def _calc_geo_dists_on_valid_coords(
        self,
        rot_coords_subset: Array,
        source_point_bev: Array,
        target_point_bev: Array,
        sad: float,
        lateral_cutoff: float,
        *,
        prepared_transform: tuple | None = None,
    ):
        """Return ordered local indices and distances for compact beam coordinates."""
        xp = array_api_compat.array_namespace(
            rot_coords_subset, source_point_bev, target_point_bev
        )
        if prepared_transform is None:
            source = xp.astype(xp.reshape(source_point_bev, (3,)), rot_coords_subset.dtype)
            targets = xp.astype(xp.reshape(target_point_bev, (1, 3)), rot_coords_subset.dtype)
            prepared = self._prepare_ray_transforms(source, targets)
            prepared_transform = (source, prepared["rotations"][0, ...], prepared["aligned"][0])
        source_point_bev, rotation, aligned = prepared_transform
        derived_rot_mat = None if aligned else rotation

        lateral_cutoff_sq_over_sad_sq = (lateral_cutoff / sad) ** 2
        use_cupy = "cupy" in xp.__name__ and cupy_available() and xp.cuda.is_available()
        use_torch = "torch" in xp.__name__ and pytorch_gpu_available()
        if (use_cupy or use_torch) and rot_coords_subset.shape[0] > 0:
            if not self._cuda_kernel_used:
                logger.info("Using %s kernel for geometric distance calculation.", xp.__name__)
                self._cuda_kernel_used = True
            if derived_rot_mat is None:
                derived_rot_mat = rotation
            if use_cupy:
                # Retain the existing fused kernel and its arithmetic order.
                _, out_y, _, out_lat_x, out_lat_z, out_rad_dist_sq, out_mask = (
                    _calc_geo_dists_cupy_kernel(
                        rot_coords_subset.ravel(),
                        derived_rot_mat.ravel(),
                        source_point_bev,
                        lateral_cutoff_sq_over_sad_sq,
                        size=rot_coords_subset.shape[0],
                    )
                )
                out_lat_dists = xp.stack((out_lat_x, out_lat_z), axis=1)
            else:
                out_coords_temp, out_lat_dists, out_rad_dist_sq, out_mask = (
                    _calc_geo_dists_torch_kernel(
                        rot_coords_subset,
                        derived_rot_mat,
                        source_point_bev,
                        float(lateral_cutoff_sq_over_sad_sq),
                    )
                )
                out_y = out_coords_temp[:, 1]
        else:
            if not getattr(self, "_fallback_logged", False):
                logger.info("Using vectorized path for geometric distance calculation.")
                self._fallback_logged = True
            if derived_rot_mat is None:
                out_coords_temp = rot_coords_subset
            else:
                out_coords_temp = xp.matmul(rot_coords_subset, derived_rot_mat)
            lx = out_coords_temp[:, 0] + source_point_bev[0]
            lz = out_coords_temp[:, 2] + source_point_bev[2]
            out_lat_dists = xp.stack((lx, lz), axis=1)
            out_rad_dist_sq = lx * lx + lz * lz
            out_y = out_coords_temp[:, 1]
            limit = lateral_cutoff_sq_over_sad_sq * out_y * out_y
            out_mask = out_rad_dist_sq <= limit

        # for array API compatible indexing
        sub_ix = xp.nonzero(out_mask)[0]

        # Return radial distances squared
        rad_distances_sq = xp.take(out_rad_dist_sq, sub_ix, axis=0)

        # Lateral distances in X & Z
        lat_dists = xp.take(out_lat_dists, sub_ix, axis=0)

        if array_api_compat.size(sub_ix) > 0:
            # Lateral distances projected onto isocenter
            iso_lat_dists = lat_dists / xp.take(out_y, sub_ix, axis=0)[:, None] * sad
        else:
            iso_lat_dists = xp.empty_like(lat_dists)

        return sub_ix, rad_distances_sq, lat_dists, iso_lat_dists
