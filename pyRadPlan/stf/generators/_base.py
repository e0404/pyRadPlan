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

        self.mult_scen: Any = None  # Scenario model, its name / definition, or a ScenarioSet
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

        self._target_mask = self._cst.optimization_target_mask()
        # Add margin if desired, otherwise take the indices directly
        if not self.add_margin:
            self._target_voxels = self._cst.optimization_target_voxels(order="numpy")
        if self.add_margin:
            dim_margins = self._target_margins()

            voxel_margin = np.ceil(
                np.array(dim_margins)
                / (self._ct.resolution["x"], self._ct.resolution["y"], self._ct.resolution["z"])
            ).astype(int)

            dilation = sitk.BinaryDilateImageFilter()
            dilation.SetKernelType(sitk.sitkBox)
            dilation.SetKernelRadius(voxel_margin.astype(int).tolist())

            # matRad_addMargin only grows into voxels that belong to some VOI
            self._target_mask = sitk.And(
                sitk.Cast(dilation.Execute(self._target_mask), sitk.sitkUInt8),
                sitk.Cast(self._patient_mask, sitk.sitkUInt8),
            )

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

    def _target_margins(self) -> tuple[float, float, float]:
        """
        Margin per axis [mm] by which the target is grown for beamlet placement.

        As in matRad: the largest setup shift magnitude along the axis plus the range and
        beamlet margins, at least one CT voxel.
        """
        added_margin = self._computed_target_margin()  # pbMargin in matRad
        max_iso_shift, range_margin = self._scenario_margins()
        return tuple(
            float(max(self._ct.resolution[axis], max_iso_shift[i] + range_margin + added_margin))
            for i, axis in enumerate(("x", "y", "z"))
        )

    def _scenario_margins(self) -> tuple[np.ndarray, float]:
        """
        Target margins required by the scenarios.

        Returns
        -------
        tuple[np.ndarray, float]
            The largest setup shift magnitude per direction in mm, and the range margin, the
            sum of the largest absolute (mm) and relative range shift magnitudes. Dimensions
            the scenario set does not realize contribute zero.
        """
        scen_set = self._mult_scen.scenarios(self._ct)

        max_iso_shift = np.zeros(3)
        if scen_set.has_dimension("setup"):
            max_iso_shift = np.asarray(scen_set.extent("setup"), dtype=float)

        range_margin = 0.0
        if scen_set.has_dimension("range"):
            range_extent = scen_set.extent("range")
            range_margin = float(range_extent[0] + range_extent[1])

        return max_iso_shift, range_margin

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
            attr = "mult_scen" if key == "scenarios" else key
            try:
                setattr(self, attr, pln.prop_stf[key])
            except AttributeError:
                warnings.warn(f"Attribute {key} not existing in {self.name} generator")

        self.machine = pln.machine

        # The plan's scenario model, unless prop_stf names one; validated once the ct is known
        if not {"mult_scen", "scenarios"} & set(pln.prop_stf):
            self.mult_scen = pln.scenarios
        if self.mult_scen is None:
            self.mult_scen = "nomScen"

        # TODO: self.bio_param = pln.bio_param
