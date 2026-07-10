from typing import Any, Union, ClassVar
import warnings
import numpy as np

import SimpleITK as sitk

from pyRadPlan.core import np2sitk, ProgressReporter
from pyRadPlan.stf._exceptions import GeometryError
from pyRadPlan.plan import validate_pln
from pyRadPlan.ct import validate_ct, CT
from pyRadPlan.cst import validate_cst, StructureSet
from pyRadPlan.scenarios import ScenarioModel, validate_scenario_model
from pyRadPlan.machines import Machine, load_from_name, validate_machine


from abc import ABC, abstractmethod


class StfGeneratorBase(ProgressReporter, ABC):
    """Base class for steering information generators."""

    # Class constants
    is_stf_generator: ClassVar[bool] = True
    name: ClassVar[str]
    short_name: ClassVar[str]
    possible_radiation_modes: ClassVar[list[str]]

    @property
    def radiation_mode(self):
        """Radiation Mode."""
        return self._radiation_mode

    @radiation_mode.setter
    def radiation_mode(self, value):
        if value in self.possible_radiation_modes:
            self._radiation_mode = value
        else:
            raise ValueError(
                f"Invalid radiation mode. Possible modes are: {self.possible_radiation_modes}"
            )

    def _computed_target_margin(self) -> float:
        """Margin to be applied to the union of targets for stf generation."""
        return 0.0

    # Initialization method for non-constant properties
    def __init__(self, pln=None):
        super().__init__()  # initialize the ProgressReporter mixin state

        # Defaults
        self.vis_mode: int = 0  # Visualization Options
        self.add_margin: bool = (
            True  # Add margins to target (for numerical stability and robustness)
        )

        self.mult_scen: Any = None  # Scenario Model, #TODO: Merge Scenario Model Developments
        self.bio_param: Any = None  # Biological Model #TODO: Merge Biological Model Developments
        self.machine: Union[str, Machine] = None  # Machine name
        self._ct: CT = None  # CT Object
        self._cst: StructureSet = None  # CST Object, TODO: Merge CST development
        self._mult_scen: ScenarioModel = None  # Scenario Model Object validated from mult_scen
        self._target_voxels: np.ndarray = None  # Target Voxels
        self._patient_voxels: np.ndarray = None  # Patient Voxels
        self._target_voxel_coordinates: np.ndarray = None  # Target Voxel Coordinates
        self._patient_voxel_coordinates: np.ndarray = None  # Patient Voxel Coordinates
        self._target_mask: sitk.Image = None  # Target Mask
        self._patient_mask: sitk.Image = None  # Patient Mask
        self._cube_hu: sitk.Image = None  # HU Cube pre-processed for stf generation

        # Validate plan to make sure it is a Plan instance
        if pln is not None:
            pln = validate_pln(pln)

            self._assign_parameters_from_pln(pln)

    def generate(self, ct, cst):
        """Generate the STF for the given CT and CST."""

        self._ct = validate_ct(ct)
        self._cst = validate_cst(cst)

        self._initialize()

        self._initialize_patient_geometry()
        stf = self._generate_source_geometry()

        # TODO: stf = validate_stf(stf)
        return stf

    def _initialize(self):
        """Initializion Procedure."""

        # Validate the Machine
        if isinstance(self.machine, str):
            tmp_machine = load_from_name(self.radiation_mode, self.machine)
        else:
            tmp_machine = self.machine

        tmp_machine = validate_machine(tmp_machine)

        if tmp_machine.radiation_mode != self.radiation_mode:
            raise ValueError(
                f"Machine radiation mode {tmp_machine.radiation_mode} does not match the stf "
                f"generators radiation mode {self.radiation_mode}"
            )

        self.machine = tmp_machine

        # Validate the scenario model
        self._mult_scen = validate_scenario_model(self.mult_scen, self._ct)

    def _initialize_patient_geometry(self):
        """Initialize patient geometry."""

        # Get all Patient Voxels
        self._patient_voxels = self._cst.patient_voxels(order="numpy")

        if not self._patient_voxels.any():
            raise GeometryError("Contours do not contain any voxels")

        self._patient_mask = self._cst.patient_mask()

        # now process target

        self._target_mask = self._cst.target_union_mask()
        # Add margin if desired, otherwise take the indices directly
        if not self.add_margin:
            self._target_voxels = self._cst.target_union_voxels(order="numpy")
        if self.add_margin:
            added_margin = self._computed_target_margin()  # pbMargin in matRad

            max_iso_shift = np.max(self._mult_scen.iso_shift, axis=0)
            range_margin = (
                self._mult_scen.max_abs_range_shift + self._mult_scen.max_rel_range_shift
            )

            constant_margin = added_margin + range_margin

            # Dimension specific margins
            dim_margins = (
                np.max((self._ct.resolution["x"], max_iso_shift[0], constant_margin)),
                np.max((self._ct.resolution["y"], max_iso_shift[1], constant_margin)),
                np.max((self._ct.resolution["z"], max_iso_shift[2], constant_margin)),
            )

            voxel_margin = np.ceil(
                np.array(dim_margins)
                / (self._ct.resolution["x"], self._ct.resolution["y"], self._ct.resolution["z"])
            ).astype(int)

            dilation = sitk.BinaryDilateImageFilter()
            dilation.SetKernelType(sitk.sitkBox)
            dilation.SetKernelRadius(voxel_margin.astype(int).tolist())

            self._target_mask = dilation.Execute(self._target_mask)

            # Update target voxels
            self._target_voxels = np2sitk.sitk_mask_to_linear_indices(
                self._target_mask, order="numpy"
            )

        if not self._target_voxels.any():
            raise GeometryError("No target found in CST")

        # pre-process CT
        self._cube_hu = self._ct.cube_hu

        # Now obtain the target voxel coordinates
        # TODO: 4D issues? And can this be done more

        self._target_voxel_coordinates = np2sitk.linear_indices_to_grid_coordinates(
            self._target_voxels, self._ct.grid, index_type="numpy"
        ).T

        # Multiply with target mask to ignore densities outside of target
        apply_mask = sitk.MaskImageFilter()
        self._cube_hu = apply_mask.Execute(self._cube_hu, self._patient_mask)

    @abstractmethod
    def _generate_source_geometry(self):
        """Generate the source geometry for the STF."""

    def _assign_parameters_from_pln(self, pln):
        """Assign parameters from the plan prop_stf to the generator."""

        pln = validate_pln(pln)

        self._radiation_mode = pln.radiation_mode

        # pln.prop_stf is a dictionary. For every key we try to assign the value
        # If the key is not existing as an attribute, we throw a warning
        for key in pln.prop_stf:
            try:
                setattr(self, key, pln.prop_stf[key])
            except AttributeError:
                warnings.warn(f"Attribute {key} not existing in {self.name} generator")

        self.machine = pln.machine

        if "mult_scen" in pln:
            self.mult_scen = (
                pln.mult_scen
            )  # Validation will be performed as soon as we have the ct
        elif self.mult_scen is None:
            self.mult_scen = "nomScen"
        else:
            pass

        # TODO: self.bio_param = pln.bio_param
