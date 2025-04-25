from abc import ABC, abstractmethod
from typing import Union, ClassVar
import warnings
import logging

import numpy as np

# from numpy.typing import ArrayLike

from pyRadPlan.plan import Plan, validate_pln
from pyRadPlan.ct import CT, validate_ct
from pyRadPlan.cst import StructureSet, validate_cst
from pyRadPlan.stf import SteeringInformation, validate_stf
from pyRadPlan.dij import Dij, validate_dij
from pyRadPlan.scenarios import ScenarioModel
from pyRadPlan.quantities import FluenceDependentQuantity, get_quantity

from ..objectives import get_objective
from ..solvers import get_available_solvers, get_solver, SolverBase


logger = logging.getLogger(__name__)


class PlanningProblem(ABC):
    """
    Abstrac class for all planning problems.

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
    possible_radiation_modes: list[str] = ["photons", "protons", "helium", "carbon", "oxygen"]

    apply_overlap: bool
    solver: Union[str, dict, SolverBase]

    # Private properties
    _ct: CT
    _cst: StructureSet
    _stf: SteeringInformation
    _dij: Dij
    _mult_scen: ScenarioModel

    _objective_list: list
    _constraint_list: list

    _quantities: list[FluenceDependentQuantity]
    _q_cache_index: list[int]
    _objectives_per_quantity: dict[str, int]

    def __init__(self, pln: Union[Plan, dict] = None):
        self._scenario_model = None

        self.solver = "ipopt"
        self.apply_overlap = True

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
    def _solve(self) -> tuple[np.ndarray, dict]:
        """Solve the planning problem."""

    def _initialize(self):
        """Initialize the data for the planning problem."""

        # resampling to dose-grid
        self._ct = self._ct.resample_to_grid(self._dij.dose_grid)

        # apply overlap priorities
        if self.apply_overlap:
            self._cst = self._cst.apply_overlap_priorities().resample_on_new_ct(self._ct)
        else:
            self._cst = self._cst.resample_on_new_ct(self._ct)

        # sanitize objectives and constraints and manage required quantities
        objectives = []
        quantity_ids = []
        for voi in self._cst.vois:
            if len(voi.objectives) > 0:
                # get the index list
                cube_ix = voi.indices_numpy
                objs = [get_objective(obj) for obj in voi.objectives]

                objectives.append((cube_ix, objs))

                quantity_ids.extend([obj.quantity for obj in objs])

        self._objective_list = objectives

        # unique quantities
        quantity_ids = list(set(quantity_ids))
        # get the quantities and check if they are fluence dependent
        quantities = [get_quantity(qid) for qid in quantity_ids]
        for q in quantities:
            if not issubclass(q, FluenceDependentQuantity):
                raise ValueError(
                    f"Quantity {q} is not fluence dependent! Currently only fluence dependent "
                    "quantities can be used in inverse planning!"
                )

        # TODO: manage scenarios

        # Manage quantites by getting them from the objective quantities
        self._quantities = [q(self._dij) for q in quantities]

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

        # Alternative code idea when storing tuples
        # quantity_obj_info = []
        # for q in quantities:
        #     q_instance = q(self._dij)

        #     # find objective indices with quantity]
        #     obj_ixs = []
        #     for ix, obj in enumerate(self._objective_list):
        #         if q_instance.identifier == obj.quantity:
        #             obj_ixs.append(ix)
        #     quantity_obj_info.append((q_instance, obj_ixs))
        # self._quantities = quantity_obj_info

        # set solver options
        self.solver = get_solver(self.solver)

        # initial point

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
        return self._solve()


class NonLinearPlanningProblem(PlanningProblem):
    """Abstract Class for all Treatment Planning Problems."""

    @abstractmethod
    def _objective_functions(self, x: np.ndarray) -> np.ndarray:
        """Define the objective functions."""

    @abstractmethod
    def _objective_jacobian(self, x: np.ndarray) -> np.ndarray:
        """Define the objective jacobian."""

    def _objective_hessian(self, x: np.ndarray) -> np.ndarray:
        """Define the objective hessian."""
        return {}

    def _constraint_functions(self, x: np.ndarray) -> np.ndarray:
        """Define the constraint functions."""
        return None

    def _constraint_jacobian(self, x: np.ndarray) -> np.ndarray:
        """Define the constraint jacobian."""
        return None

    def _constraint_jacobian_structure(self) -> np.ndarray:
        """Define the constraint jacobian structure."""
        return None

    def _variable_bounds(self, x: np.ndarray) -> np.ndarray:
        """Define the variable bounds."""
        return np.array([0.0, np.inf], dtype=np.float64)
