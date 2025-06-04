"""Base class for all dose engines."""

import logging

import sys

if sys.version_info < (3, 10):
    import importlib_resources as resources  # Backport for older versions
else:
    from importlib import resources  # Standard from Python 3.9+

import warnings
import time
from typing import ClassVar, Union
from abc import ABC, abstractmethod

import SimpleITK as sitk
import numpy as np
from pydantic import ValidationError

from pyRadPlan.core import Grid
from pyRadPlan.core.np2sitk import linear_indices_to_grid_coordinates
from pyRadPlan.core.resample import resample_numpy_array
from pyRadPlan.ct import CT, resample_ct
from pyRadPlan.cst import StructureSet
from pyRadPlan.stf import SteeringInformation, validate_stf
from pyRadPlan.plan import Plan, validate_pln
from pyRadPlan.dij import Dij, validate_dij
from pyRadPlan.scenarios import create_scenario_model, ScenarioModel
from pyRadPlan.machines import load_machine_from_mat, validate_machine

logger = logging.getLogger(__name__)

# from dose.calcDoseInit import init_dose_calc


class DoseEngineBase(ABC):
    """
    Abstract Interface for all dose engines.

    All dose engines should inherit from this class.

    Parameters
    ----------
    pln : Plan
        The Plan object to assign properties from.

    Attributes
    ----------
    short_name : str
        The short name of the dose engine.
    name : str
        The name of the dose engine.
    possible_radiation_modes : list[str]
        The possible radiation modes for the dose engine.
    is_dose_engine : bool = True
        Helper class variable telling you that this is a dose engine
    select_voxels_in_scenarios : bool
        Whether to select voxels in scenarios.
    mult_scen : Union[str, ScenarioModel]
        The scenario model.
    bio_model : Union[str, dict]
        The biological model.
    dose_grid : Union[Grid,dict]
        The dose grid to use (struct with at least doseGrid.resolution.x/y/z set).
    """

    # Constant, Abstract properties
    short_name: ClassVar[str]
    name: ClassVar[str]
    possible_radiation_modes: ClassVar[list[str]] = NotImplemented
    is_dose_engine: ClassVar[bool] = True  # Helper variable

    mult_scen: Union[str, ScenarioModel]
    bio_model: Union[str, dict]
    dose_grid: Union[Grid, dict]

    select_voxels_in_scenarios: bool

    # Public properties
    def __init__(self, pln: Union[Plan, dict] = None):
        # Assign default parameters from Matrad_Config or manually
        self.mult_scen = "nomScen"
        self.select_voxels_in_scenarios = None
        self.voxel_sub_ix = None  # selection of where to calculate / store dose, empty by default
        self.dose_grid = None

        if pln is not None:
            self.assign_properties_from_pln(pln, True)

        self._ct_grid = None

        # Protected properties with public get access
        self._machine = None  # base data defined in machine file
        self._timers = None  # timers of dose calc
        self._num_of_columns_dij = None  # number of columns in the dij struct
        self._vox_world_coords = None  # ct voxel coordinates in world
        self._vox_world_coords_dose_grid = None  # dose grid voxel coordinates in world
        self._vct_grid = None  # voxel grid inside patient
        self._vdose_grid = None  # voxel dose grid
        self._vct_grid_scen_ix = None  # logical subindexes of scenarios in ct grid
        self._vdose_grid_scen_ix = None  # logical subindexes of scenarios in dose grid
        self._vct_grid_mask = None  # voxel grid inside patient as logical mask
        self._vdose_grid_mask = None  # voxel dose grid inside patient as logical mask
        self._robust_voxels_on_grid = None  # voxels to be computed in robustness scenarios

        # Protected properties with private get access
        self._last_progress_update = None
        self._calc_dose_direct = False

        # initializing super() class (here: ABC)
        super().__init__()

    # Public methods
    def assign_properties_from_pln(self, pln: Plan, warn_when_property_changed: bool = False):
        """
        Assign properties from a Plan object to the Dose Engine.

        This includes the Scenario Model and the Biological Model, and any
        other properties that
        can be stored in the prop_dose_calc dictionary within the Plan object.
        This function will check if a property exists for the dose engine and,
        if yes, set it.

        Parameters
        ----------
        pln : Plan
            The Plan object to assign properties from.
        warn_when_property_changed : bool
            Whether to warn when properties are changed.
        """

        pln = validate_pln(pln)

        # Set Scenario Model
        if hasattr(pln, "mult_scen"):
            self.mult_scen = pln.mult_scen

        # Assign Biologival Model
        if hasattr(pln, "bio_param"):
            self.bio_param = pln.bio_param  # TODO: No bio_param yet

        if not isinstance(warn_when_property_changed, bool):
            warn_when_property_changed = False

        # Overwrite default properties within the engine
        # with the ones given in the prop_dose_calc dict
        if pln.prop_dose_calc is not None:
            prop_dict = pln.prop_dose_calc
            if (
                "engine" in prop_dict
                and prop_dict["engine"]
                and prop_dict["engine"] != self.short_name
            ):
                raise ValueError(
                    f"Inconsistent dose engines given! pln asks for '{prop_dict['engine']}', "
                    f"but you are using '{self.short_name}'!"
                )
            prop_dict.pop("engine", None)
        else:
            prop_dict = {}

        fields = prop_dict.keys()

        # Set up warning message
        if warn_when_property_changed:
            warning_msg = "Property in Dose Engine overwritten from pln.propDoseCalc"
        else:
            warning_msg = None

        for field in fields:
            if not hasattr(self, field):
                warnings.warn('Property "{}" not found in Dose Engine!'.format(field))
            elif warn_when_property_changed and warning_msg:
                logger.warning(warning_msg + f": {field}")

            setattr(self, field, prop_dict[field])

    def calc_dose_forward(
        self, ct: CT, cst: StructureSet, stf: SteeringInformation, w: np.ndarray
    ) -> dict:
        """
        Perform a forward dose calculation.

        Compute a dose cube by directly applying a set of weights
        during dose calculation.

        Parameters
        ----------
        ct : CT
            The CT scan data.
        cst : StructureSet
            The structure set containing volumes of interest (VOIs).
        stf : SteeringInformation
            The steering information containing beam configurations.
        w : np.ndarray
            The weights to apply during dose calculation.

        Returns
        -------
        Dij
            The dose information

        Notes
        -----
        pyRadPlan handles the forward dose calculation by setting a switch in
        the dose engine (_calc_dose_direct) to True. This facilitates reusing
        of algorithms in both forward and influence matrix calculations.
        """
        time_start = time.time()

        self._calc_dose_direct = True

        stf = validate_stf(stf)

        if w is None:
            logger.info("No weights given. Using weights stored in stf.")
        else:
            logger.info("Using provided weights for forward dose calculation.")

            if w.size != stf.total_number_of_bixels:
                raise ValueError("Number of weights does not match the number of bixels in stf.")

            # Assign the weights to the stf
            bixel_num = 0
            for beam in stf.beams:
                for ray in beam.rays:
                    for bixel in ray.beamlets:
                        bixel.weight = w[bixel_num]
                        bixel_num += 1

        # Run dose calculation (direct flag is on)
        dij = self._calc_dose(ct, cst, stf)

        # Now do the forward weighting with w
        # This is done because the engine might store the individual fields
        result = dij.compute_result_ct_grid(np.ones(dij.total_num_of_bixels, dtype=np.float32))

        time_elapsed = time.time() - time_start
        logger.info("Forward dose calculation done in %.2f seconds.", time_elapsed)

        return result

    def calc_dose_influence(self, ct: CT, cst: StructureSet, stf: SteeringInformation) -> Dij:
        """
        Calculate the set of dose/quantity influence matrices.

        These are the matrices that map a fluence vector to a dose/quantity
        distribution.

        Parameters
        ----------
        ct : CT
            The CT scan data.
        cst : StructureSet
            The structure set containing volumes of interest (VOIs).
        stf : SteeringInformation
            The steering information containing beam configurations.

        Returns
        -------
        Dij
            The dose influence matrix collection

        Notes
        -----
        pyRadPlan handles the forward dose calculation by setting a switch in
        the dose engine (_calc_dose_direct) to True. This facilitates reusing
        of algorithms in both forward and influence matrix calculations.
        """

        time_start = time.time()
        self._calc_dose_direct = False
        dij = self._calc_dose(ct, cst, stf)
        time_elapsed = time.time() - time_start
        logger.info("Dose influence matrix calculation done in %.2f seconds.", time_elapsed)

        return dij

    def set_overlap_priorities(self, cst: StructureSet, ct_dim=None) -> StructureSet:
        """Set overlap priorities for the structures in the CST."""

        logger.info("Adjusting structures for overlap... ")
        t_start = time.time()

        num_of_ct_scenarios = np.unique([x.num_of_scenarios for x in cst.vois])

        # Sanity check
        if len(num_of_ct_scenarios) > 1:
            raise ValueError("Inconsistent number of scenarios in cst struct.")

        new_cst = cst.apply_overlap_priorities()

        t_end = time.time()

        logger.info("Done in %.2f seconds.", t_end - t_start)

        return new_cst

    def select_voxels_from_cst(self, cst, dose_grid, selection_mode):
        """
        Get mask of the voxels (on dose grid).

        Gets the mask from the cst structures specified by selection_mode.
        """
        self.set_overlap_priorities(self, cst, dose_grid["dimensions"])

        selected_cst_structs = []

        include_mask = [
            np.zeros(np.prod(dose_grid["dimensions"]), dtype=bool) for _ in range(len(cst[:, 4]))
        ]

        if selection_mode == "all":
            for ct_scen_idx in range(len(include_mask)):
                include_mask[ct_scen_idx][:] = True
        else:
            if isinstance(selection_mode, str):
                if selection_mode == "targetOnly":
                    selected_cst_structs = [i for i, x in enumerate(cst[:, 2]) if x == "TARGET"]
                elif selection_mode == "objectivesOnly":
                    for i in range(len(cst)):
                        if len(cst[i, 5]) > 0:
                            selected_cst_structs.append(i)
                elif selection_mode == "oarsOnly":
                    selected_cst_structs = [i for i, x in enumerate(cst[:, 2]) if x == "OAR"]
                elif selection_mode == "robustnessOnly":
                    for i in range(len(cst)):
                        for j in range(len(cst[i, 5])):
                            if (
                                "robustness" in cst[i, 5][j]
                                and cst[i, 5][j]["robustness"] != "none"
                            ):
                                selected_cst_structs.append(i)
                else:
                    raise ValueError(f"Unrecognized voxel selection mode: {selection_mode}")
            elif isinstance(selection_mode, (list, np.ndarray)):
                selected_cst_structs = np.unique(
                    np.intersect1d(selection_mode, np.arange(len(cst)) + 1)
                )
                if not np.array_equal(selected_cst_structs, np.unique(selection_mode)):
                    warnings.warn(
                        "Specified structures are not compatible with cst structures. "
                        f"Only performing calculation on structures: {selected_cst_structs}"
                    )

            # Loop over all cst structures
            for i in range(len(cst)):
                if len(cst[i, 4][0]) > 0:
                    if len(cst[i, 5]) > 0:
                        # Loop over obj/constraint functions
                        for j in range(len(cst[i, 5])):
                            obj = cst[i, 5][j]

                            if not isinstance(obj, dict):
                                try:
                                    pass
                                    # TODO: obj = matRad_DoseOptimizationFunction.
                                    # createInstanceFromStruct(obj)
                                except ValueError:
                                    logger.error(
                                        f"cst[{i},5][{j}] is not a valid Objective/constraint! "
                                        "Remove or Replace and try again!"
                                    )

                            robustness = obj.get("robustness", "none")

                            if i in selected_cst_structs:
                                for ct_idx in range(len(cst[i, 4])):
                                    include_mask[ct_idx][cst[i, 4][ct_idx]] = 1

                                if robustness == "none":
                                    logger.info(
                                        f"Including cst structure {cst[i, 1]} even though this "
                                        "structure has no robustness."
                                    )
                            else:
                                logger.info(
                                    f"Excluding cst structure {cst[i, 1]} even though this "
                                    "structure has an objective or constraint."
                                )

                                if robustness != "none":
                                    logger.info(
                                        f"Excluding cst structure {cst[i, 1]} even though this "
                                        "structure has robustness."
                                    )
                    elif i in selected_cst_structs:
                        logger.info(
                            f"Including cst structure {cst[i, 1]} even though this structure "
                            "does not have any objective or constraint"
                        )

        return include_mask

    def resize_cst_to_grid(self, cst: StructureSet, dij, new_ct: CT) -> StructureSet:
        """Resize the CST to the dose cube resolution."""

        logger.info("Resampling structure set... ")

        t_start = time.time()

        cst.resample_on_new_ct(new_ct)

        t_end = time.time()

        logger.info("Done in  %.2f seconds.", t_end - t_start)
        return cst

    # private methods
    def _init_dose_calc(self, ct: CT, cst: StructureSet, stf: SteeringInformation) -> dict:
        """
        Initialize the dose calculation.

        Parameters
        ----------
        self : DoseEngineBase
            The instance of the dose engine.
        ct : CT
            The CT scan data.
        cst : StructureSet
            The structure set containing volumes of interest (VOIs).
        stf : SteeringInformation
            The steering information containing beam configurations.

        Returns
        -------
        dict
            A dictionary containing the dose influence matrix (dij) and related information.

        Description
        -----------
        This method sets up the necessary grids and parameters for dose calculation. It initializes
        the CT and dose grids, checks machine and radiation mode consistency, creates a scenario
        model, and sets up arrays for bookkeeping. It also adjusts the isocenter internally for
        different dose grids and converts CT subscripts to world coordinates. Additionally, it
        loads the machine file from the base data folder and performs voxel selection for dose
        calculation.
        """

        self._ct_grid = Grid.from_sitk_image(ct.cube_hu)

        dij = {}

        # Default: dose influence matrix computation
        if self._calc_dose_direct:
            logger.info("Forward dose calculation using '%s' Dose Engine...", self.name)
        else:
            logger.info("Dose influence matrix calculation using '%s' Dose Engine...", self.name)

        # Check if machine and radiation_mode are consistent
        machine = list({beam.machine for beam in stf.beams})
        radiation_mode = list({beam.radiation_mode for beam in stf.beams})

        if len(machine) != 1 or len(radiation_mode) != 1:
            raise ValueError("machine and radiation mode need to be unique within supplied stf!")

        machine = machine[0]
        radiation_mode = radiation_mode[0]

        # ScenarioModel
        if not isinstance(self.mult_scen, ScenarioModel):
            self.mult_scen = create_scenario_model(self.mult_scen, ct)

        # TODO: Add BioModel Support
        # if not np.isnan(self.bio_param["RBE"]):
        #     dij['RBE'] = self.bio_param['RBE']

        # store CT grid
        dij["ct_grid"] = ct.grid

        if self.dose_grid is None:
            logger.info("Dose Grid not set. Using default resolution.")
            self.dose_grid = ct.grid.resample({"x": 5.0, "y": 5.0, "z": 5.0})

        if not isinstance(self.dose_grid, Grid):
            try:
                self.dose_grid = Grid.model_validate(self.dose_grid)
            except ValidationError:
                self.dose_grid = ct.grid.resample(self.dose_grid["resolution"])

        logger.info(
            "Dose Grid has Dimensions (%d,%d,%d) with resolution x=%f, y=%f, z=%f.",
            self.dose_grid.dimensions[0],
            self.dose_grid.dimensions[1],
            self.dose_grid.dimensions[2],
            self.dose_grid.resolution["x"],
            self.dose_grid.resolution["y"],
            self.dose_grid.resolution["z"],
        )

        dij["dose_grid"] = self.dose_grid

        # Meta information for dij #TODO: Change dij Model in future
        dij["num_of_beams"] = len(stf.beams)
        dij["num_of_scenarios"] = 1  # TODO: Add support for multiple scenarios
        dij["num_of_rays_per_beam"] = [stf.beams[i].num_of_rays for i in range(len(stf.beams))]
        dij["total_num_of_bixels"] = int(
            sum([stf.beams[i].total_number_of_bixels for i in range(len(stf.beams))])
        )
        dij["total_num_of_rays"] = int(sum(dij["num_of_rays_per_beam"]))

        # Check if full dose influence data is required
        self._num_of_columns_dij = (
            len(stf.beams) if self._calc_dose_direct else dij["total_num_of_bixels"]
        )

        # Set up arrays for book keeping
        dij["bixel_num"] = np.nan * np.ones(self._num_of_columns_dij)
        dij["ray_num"] = np.nan * np.ones(self._num_of_columns_dij)
        dij["beam_num"] = np.nan * np.ones(self._num_of_columns_dij)

        # Default MU calibration
        dij["min_mu"] = np.zeros(self._num_of_columns_dij)
        dij["max_mu"] = np.ones(self._num_of_columns_dij) * np.inf
        dij["num_of_particles_per_mu"] = np.ones(self._num_of_columns_dij) * 1e6

        if self.voxel_sub_ix is None:
            self.voxel_sub_ix = np.unique(
                np.concatenate([cst.vois[c].indices_numpy for c in range(0, len(cst.vois))])
            )

        tmp_vct_grid_scen = [np.asarray(self.voxel_sub_ix)] * ct.num_of_ct_scen

        self._vct_grid = np.unique(np.concatenate(tmp_vct_grid_scen))

        # Find subindexes for the individual scenarios. This helps with subselection later.
        self._vct_grid_scen_ix = [np.isin(self._vct_grid, c) for c in tmp_vct_grid_scen]

        # Initialize tmp_vdose_grid_scen
        tmp_vdose_grid_scen = [None] * ct.num_of_ct_scen

        if self._ct_grid == self.dose_grid:
            tmp_vdose_grid_scen = tmp_vct_grid_scen
            resampled_ct = ct
        else:
            resampled_ct = resample_ct(
                ct=ct,
                interpolator=sitk.sitkNearestNeighbor,
                target_grid=self.dose_grid,
            )
            self.dose_grid = Grid.from_sitk_image(resampled_ct.cube_hu)

            for s in range(ct.num_of_ct_scen):
                # Receive linear indices and grid locations from the dose grid
                tmp_cube = np.zeros(ct.cube_dim[::-1], dtype=np.float32)
                tmp_cube.ravel()[tmp_vct_grid_scen[s]] = 1

                interpolated_tmp_cube = resample_numpy_array(
                    tmp_cube,
                    reference_image=ct.cube_hu,
                    interpolator=sitk.sitkNearestNeighbor,
                    target_image=resampled_ct.cube_hu,
                )

                tmp_vdose_grid_scen[s] = np.where(
                    np.abs(np.round(interpolated_tmp_cube)).ravel().astype(bool)
                )

        # Get unique dose grid
        self._vdose_grid = np.unique(np.concatenate(tmp_vdose_grid_scen, axis=0))

        # Find subindexes for the dose grid scenarios
        self._vdose_grid_scen_ix = [np.isin(self._vdose_grid, c) for c in tmp_vdose_grid_scen]

        # Convert CT subscripts to world coordinates.
        self._vox_world_coords = linear_indices_to_grid_coordinates(
            self._vct_grid, self._ct_grid, index_type="numpy"
        )

        self._vox_world_coords_dose_grid = linear_indices_to_grid_coordinates(
            self._vdose_grid, self.dose_grid, index_type="numpy"
        )

        # Create helper masks
        self._vdose_grid_mask = np.zeros(self.dose_grid.num_voxels, dtype=bool)
        self._vdose_grid_mask[self._vdose_grid] = True

        self._vct_grid_mask = np.zeros(np.prod(ct.cube_dim), dtype=bool)
        self._vct_grid_mask[self._vct_grid] = True

        # Load machine file from base data folder
        self._machine = self.load_machine(radiation_mode, machine)

        # Voxel selection for dose calculation
        # TODO: set overlap priorites - Skips most of the function
        cst = self.set_overlap_priorities(cst, ct_dim=None)

        # resizing cst to dose cube resolution
        cst = self.resize_cst_to_grid(cst, dij, resampled_ct)

        # TODO: this function has not yet been implemented
        # structures that are selected here will be included in dose calculation over
        # the robust scenarios
        # self._robust_voxels_on_grid = select_voxels_from_cst(cst, dij["ct_grid"])

        return dij

    def _finalize_dose(self, dij: dict) -> Dij:
        return validate_dij(dij)

    def _progress_update(self, pos, total):
        raise NotImplementedError

    # Private and abstract methods

    @abstractmethod
    def _calc_dose(self, ct: CT, cst: StructureSet, stf: SteeringInformation) -> Dij:
        raise NotImplementedError("Method '_calc_dose' must be implemented.")

    # static methods
    @staticmethod
    def is_available(pln, machine):
        available, msg = [], []
        return available, msg

    @staticmethod
    def load_machine(radiation_mode, machine_name):
        file_name = radiation_mode + "_" + machine_name + ".mat"
        machines_path = resources.files("pyRadPlan.data.machines")
        path = machines_path.joinpath(file_name)
        machine = validate_machine(load_machine_from_mat(path))
        return machine
