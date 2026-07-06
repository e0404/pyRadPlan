from abc import ABC, abstractmethod
from typing import Any, Union, ClassVar
import warnings
import logging

from ...core.xp_utils.typing import Array, ArrayNamespace

import numpy as np
import array_api_compat

from pyRadPlan.plan import Plan, validate_pln
from pyRadPlan.ct import CT, validate_ct
from pyRadPlan.cst import StructureSet, validate_cst
from pyRadPlan.stf import SteeringInformation, validate_stf
from pyRadPlan.dij import Dij, validate_dij
from pyRadPlan.scenarios import ScenarioModel
from pyRadPlan.quantities import RTQuantity, QuantityResolver

from ..objectives import get_objective
from ..solvers import get_available_solvers, get_solver, SolverBase
from ...core import ProgressReporter, xp_utils


logger = logging.getLogger(__name__)


class PlanningProblem(ProgressReporter, ABC):
    """
    Abstract class for all planning problems.

    Parameters
    ----------
    pln : Union[Plan, dict], optional
        Plan object or dictionary to initialize the problem with.

    Attributes
    ----------
    short_name : ClassVar[str]
        Short name of the optimization problem.
    name : ClassVar[str]
        Name of the optimization problem.
    apply_overlap : bool, default=True
        Whether to apply overlap priorities to the StructureSet
    solver : Union[str, dict, SolverBase], default="ipopt"
        The solver to use for optimization.
    """

    # Constant, Abstract properties are realized as ClassVars
    short_name: ClassVar[str]
    name: ClassVar[str]
    possible_radiation_modes: list[str] = [
        "photons",
        "protons",
        "helium",
        "carbon",
        "oxygen",
        "VHEE",
    ]
    default_quantities: dict[str, str] = {
        "photons": "physical_dose",
        "protons": "physical_dose",
        "helium": "physical_dose",
        "carbon": "rbe_x_dose",
        "oxygen": "physical_dose",
        "VHEE": "physical_dose",
    }
    # right now only kernel based rbe model which is only standard in the carbon machine

    apply_overlap: bool
    solver: Union[str, dict, SolverBase]

    # Private properties
    _ct: CT
    _cst: StructureSet
    _stf: SteeringInformation
    _dij: Dij
    _mult_scen: ScenarioModel

    _objective_list: list[tuple]
    _constraint_list: list

    _quantities: list[RTQuantity]
    _q_cache_index: list[int]
    _objectives_per_quantity: dict[str, int]
    _num_objectives: int

    _array_backend: ArrayNamespace

    def __init__(self, pln: Union[Plan, dict] = None):
        super().__init__()

        self._scenario_model = None

        self.solver = "ipopt"
        self.apply_overlap = True

        self.convert_dose_objectives = True

        if pln is not None:
            pln = validate_pln(pln)
            self.assign_properties_from_pln(pln)

        solvers = get_available_solvers()
        if self.solver not in solvers:
            solver_names = list(solvers.keys())

            if len(solver_names) == 0:
                raise ValueError("No solver found!")

            warnings.warn(
                f"Solver {self.solver} not available. Choose from {solver_names}"
                ", and we will choose the first available one for you!"
            )

            self.solver = solver_names[0]

    def assign_properties_from_pln(self, pln: Plan, warn_when_property_changed: bool = False):
        """
        Assign properties from a Plan object to the Planning Problem.

        This function will check if a property exists for the PlanningProblem
        and, if yes, set it.

        Parameters
        ----------
        pln : Plan
            The Plan object to assign properties from.
        warn_when_property_changed : bool
            Whether to warn when properties are changed.
        """

        # Set Scenario Model
        self._mult_scen = pln.mult_scen

        # Assign Biologival Model
        if hasattr(pln, "bio_param"):
            self.bio_param = pln.bio_param  # TODO: No bio_param yet

        if not isinstance(warn_when_property_changed, bool):
            warn_when_property_changed = False

        # Overwrite default properties within the opti_prob
        # with the ones given in the prop_opt dict
        if hasattr(pln, "prop_opt") and isinstance(
            pln.prop_opt, dict
        ):  # TODO: This is not tested yet
            prop_dict = pln.prop_opt
            if (
                "opti_prob" in prop_dict
                and prop_dict["opti_prob"]
                and prop_dict["opti_prob"] != self.short_name
            ):
                raise ValueError(
                    f"Inconsistent dose opti_probs given! pln asks for '{prop_dict['opti_prob']}'"
                    f", but you are using '{self.short_name}'!"
                )
            prop_dict.pop("opti_prob", None)
        else:
            prop_dict = {}

        fields = prop_dict.keys()

        # Set up warning message
        if warn_when_property_changed:
            warning_msg = "Property in Optimization Problem overwritten from pln.prop_opt"
        else:
            warning_msg = None

        for field in fields:
            if not hasattr(self, field):
                warnings.warn(f"Property {field} not found in Problem!")
            elif warn_when_property_changed and warning_msg:
                logger.warning(warning_msg + f": {field}")

            setattr(self, field, prop_dict[field])

    @abstractmethod
    def _solve(self) -> tuple[Array, dict]:
        """Solve the planning problem."""

    def _collect_objectives(self) -> tuple[list[tuple], list[str]]:
        """Parse VOI objectives into (mask, objectives) pairs and collect quantity identifiers."""
        default_quantity = self.default_quantities.get(
            self._stf.beams[0].radiation_mode, "physical_dose"
        )
        objectives: list[tuple] = []
        quantity_ids: list[str] = []

        if self.convert_dose_objectives:
            logger.info(
                "Converting all objectives to use quantity: "
                + self.default_quantities.get(self._stf.beams[0].radiation_mode, "physical_dose")
            )

        for voi in self._cst.vois:
            valid_objectives = [
                obj
                for obj in voi.objectives
                if not (
                    obj is None
                    or (isinstance(obj, (list, tuple)) and len(obj) == 0)
                    or (isinstance(obj, np.ndarray) and obj.size == 0)
                )
            ]
            if not valid_objectives:
                continue

            cube_ix = voi.indices_numpy
            linear_mask = np.zeros(voi.mask.GetNumberOfPixels(), dtype=np.bool_)
            linear_mask[cube_ix] = True
            objs = [get_objective(obj) for obj in valid_objectives]

            if self.convert_dose_objectives:
                for obj in objs:
                    obj.quantity = default_quantity

            for obj in objs:
                obj.preprocess_image_reference_parameters(
                    target_grid=self._dij.dose_grid, index_list=cube_ix
                )

            objectives.append((linear_mask, objs))
            quantity_ids.extend([obj.quantity for obj in objs])

        return objectives, quantity_ids

    def _initialize(self):
        """Initialize the data for the planning problem."""

        # resampling to dose-grid
        self._ct = self._ct.resample_to_grid(self._dij.dose_grid)

        # apply overlap priorities
        if self.apply_overlap:
            self._cst = self._cst.apply_overlap_priorities()

        self._cst = self._cst.resample_on_new_ct(self._ct)

        # sanitize objectives and constraints and manage required quantities
        objectives, quantity_ids = self._collect_objectives()

        self._objective_list = objectives
        # unique quantities
        quantity_ids = list(set(quantity_ids))

        # Resolve all requested quantities (plus their transitive dependencies) through a
        # shared resolver so that any quantity referenced by multiple roots is instantiated
        # exactly once.
        resolver = QuantityResolver(self._dij)
        resolver.resolve(quantity_ids)
        self._quantities = list(resolver.instances.values())

        if len(set([q.array_backend for q in self._quantities])) == 1:
            self._array_backend = self._quantities[0].array_backend
        else:
            raise TypeError(
                "Inconsistent array backends used in quantities. Decide on one (e.g. numpy)"
            )

        # obtain cache info to match quantities with objectives
        self._q_cache_index = []
        self._objectives_per_quantity = {q.identifier: [] for q in self._quantities}
        obj_ix = 0

        for obj_info in self._objective_list:
            for obj in obj_info[1]:
                for q in self._quantities:
                    if q.identifier == obj.quantity:
                        self._q_cache_index.append(
                            len(self._objectives_per_quantity[q.identifier])
                        )
                        self._objectives_per_quantity[q.identifier].append(obj_ix)
                    obj_ix += 1

        self._num_objectives = obj_ix

        # set solver options
        self.solver = get_solver(self.solver)

        # Let the solver push status (and honour pause/stop) through this problem,
        # which is the top-level workflow step observed by callers (e.g. the GUI).
        self.solver.status_callback = self._emit_solver_status

        # initial point

    def _emit_solver_status(self, message: str = "", **data: Any) -> bool:
        """Forward arbitrary solver status upward and report whether to continue.

        Generic on purpose: the problem does not assume *how* the solver produced the
        status (iterative or not).  It emits a :class:`~pyRadPlan.core.StatusReport`
        with whatever *data* the solver supplied and returns the cooperative
        pause/stop decision from :meth:`~pyRadPlan.core.ProgressReporter.checkpoint`.
        """
        self.report_status(message=message, **data)
        return self.checkpoint()

    def solve(
        self,
        ct: Union[CT, dict],
        cst: Union[StructureSet, dict],
        stf: Union[SteeringInformation, dict],
        dij: Union[Dij, dict],
    ) -> tuple[np.ndarray, dict]:
        """
        Solves the planning problem.

        Will perform initialization & validation and call the desired Solver.

        Parameters
        ----------
        ct : Union[CT, dict]
            The CT object or compatible dictionary.
        cst : Union[StructureSet, dict]
            The StructureSet object or compatible dictionary.
        stf : Union[SteeringInformation, dict]
            The SteeringInformation object or compatible dictionary.
        dij : Union[Dij, dict]
            The Dij object or compatible dictionary.

        Returns
        -------
        tuple[np.ndarray,dict]
            The optimized result and additional solver-specific result information as dictionary.
        """

        self._ct = validate_ct(ct)
        self._cst = validate_cst(cst)
        self._stf = validate_stf(stf)
        self._dij = validate_dij(dij)

        self._initialize()
        x, info = self._solve()
        x = xp_utils.to_numpy(x)
        return x, info


class NonLinearPlanningProblem(PlanningProblem):
    """Abstract Class for all Treatment Planning Problems."""

    @abstractmethod
    def _objective_functions(self, x: Array) -> Array:
        """Define the objective functions."""

    @abstractmethod
    def _objective_jacobian(self, x: Array) -> Array:
        """Define the objective jacobian."""

    def _objective_hessian(self, x: Array) -> Array:
        """Define the objective hessian."""
        return {}

    def _constraint_functions(self, x: Array) -> Array:
        """Define the constraint functions."""
        return None

    def _constraint_jacobian(self, x: Array) -> Array:
        """Define the constraint jacobian."""
        return None

    def _constraint_jacobian_structure(self) -> Array:
        """Define the constraint jacobian structure."""
        return None

    def _variable_bounds(self, x: Array) -> Array:
        """Define the variable bounds."""
        xp = array_api_compat.array_namespace(x)
        return xp.asarray([0.0, xp.inf], dtype=xp.float64)
