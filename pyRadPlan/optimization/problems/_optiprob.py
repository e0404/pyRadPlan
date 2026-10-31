from abc import ABC, abstractmethod
from typing import Annotated, Any, Optional, Union, ClassVar
import logging

from ...core.xp_utils.typing import Array, ArrayNamespace

import numpy as np
import array_api_compat

from pyRadPlan.plan import Plan, validate_pln
from pyRadPlan.ct import CT, validate_ct
from pyRadPlan.cst import StructureSet, validate_cst
from pyRadPlan.stf import SteeringInformation, validate_stf
from pyRadPlan.dij import Dij, validate_dij
from pyRadPlan.scenarios import NominalScenarioModel, ScenarioModel, ScenarioSet
from pyRadPlan.quantities import (
    RBExDose,
    RTQuantity,
    QuantityResolver,
    get_available_quantities,
)

from ..objectives import get_objective
from ..robustness import (
    RobustnessOptions,
    RobustTerm,
    build_robust_terms,
    get_robustness_operator_class,
    voi_linear_mask,
)
from ..solvers import get_available_solvers, get_solver, SolverBase
from ...core import AlgorithmParameterMetadata, ConfigurableAlgorithm, ProgressReporter, xp_utils


logger = logging.getLogger(__name__)


class PlanningProblem(ConfigurableAlgorithm, ProgressReporter, ABC):
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
        The solver to use for optimization. A string only selects solver with default options,
        a dict contains specific solver options and the corresponding solver 'name' key.
        Only the default falls back to the first registered solver when unavailable; a solver
        requested via ``pln.prop_opt`` raises instead, as substituting it would silently
        discard its configuration and change the result.
    display : bool, optional
        Toggle propagated to the solver. Left unset, the solver keeps whatever it was
        configured with (its own default, or a value from the ``solver`` dict).
    max_iter : int, optional
        Iteration cap propagated to the solver. Left unset, the solver keeps whatever it was
        configured with (its own default, or a value from the ``solver`` dict).
    convert_dose_objectives : bool, default=True
        Legacy plan-wide conversion of physical-dose objectives to the RBE-weighted dose
        when the dose influence data carries a biological model, skipped when objectives
        already use a biological quantity. Set to ``False`` to use every objective's
        quantity literally.
    evaluate_all_scenarios : bool, default=False
        Evaluate every scenario of the dose influence data's set in every quantity. By default
        a quantity evaluates only the scenarios its objectives need (a ``robustness="none"``
        objective the nominal scenario, a worst-case objective all of them).
    """

    # Constant, Abstract properties are realized as ClassVars
    short_name: ClassVar[str]
    name: ClassVar[str]

    #: Names of the generic knobs forwarded to the solver by :meth:`_propagate_solver_properties`
    #: when set on the problem. Not every solver supports all of them; a subclass tied to a
    #: particular family of solvers can extend the list.
    solver_properties: ClassVar[tuple[str, ...]] = ("max_iter", "display")

    #: Whether this problem appends every objective evaluation to :attr:`obj_history`. Declared
    #: rather than assumed, so that a problem which does not record leaves
    #: ``opt_info["obj_history"]`` absent instead of handing back an empty list that reads as
    #: "the solver never evaluated the objective".
    records_obj_history: ClassVar[bool] = False
    possible_radiation_modes: ClassVar[list[str]] = [
        "photons",
        "protons",
        "helium",
        "carbon",
        "oxygen",
        "VHEE",
    ]
    # right now only kernel based rbe model which is only standard in the carbon machine

    apply_overlap: bool = True
    # Unset by default so that a value configured on the solver itself - including one
    # coming from a prop_opt["solver"] dict - is not overwritten by a top-level default.
    # Only an explicitly assigned top-level value takes precedence over the solver's.
    display: Optional[bool] = None
    max_iter: Optional[int] = None
    solver: Union[str, dict, SolverBase] = "ipopt"
    obj_history: Annotated[Optional[list], AlgorithmParameterMetadata(configurable=False)] = None
    convert_dose_objectives: bool = True
    evaluate_all_scenarios: bool = False

    # Private properties
    _ct: CT
    _cst: StructureSet
    _stf: SteeringInformation
    _dij: Dij
    _scenarios: ScenarioModel

    _objective_list: list[tuple]
    _constraint_list: list
    _terms: list[RobustTerm]
    _scenario_set: ScenarioSet

    _quantities: list[RTQuantity]
    _q_cache_index: list[dict[str, int]]
    _objectives_per_quantity: dict[str, list[int]]
    _num_objectives: int
    _solver_explicit: bool

    _array_backend: ArrayNamespace

    _num_of_fractions: int = 1
    _dose_convention: str = "per_fraction"

    def __init__(self, pln: Union[Plan, dict] = None):
        super().__init__()

        self._scenario_model = None
        self._solver_explicit = False
        self._init_config_defaults()

        if pln is not None:
            pln = validate_pln(pln)
            self.assign_properties_from_pln(pln)

        # Validate the chosen solver name (string or "name" key inside dict).
        solvers = get_available_solvers()
        if isinstance(self.solver, str):
            solver_name = self.solver
        elif isinstance(self.solver, dict):
            solver_name = self.solver.get("name")
        else:
            solver_name = None  # SolverBase instance — skip name check

        if solver_name is not None and solver_name not in solvers:
            solver_names = list(solvers.keys())
            if len(solver_names) == 0:
                raise ValueError("No solver registered!")

            # An explicitly requested solver is never silently substituted: a different
            # solver ignores the given configuration and yields a different result.
            if self._solver_explicit:
                raise ValueError(
                    f"Solver '{solver_name}' was explicitly requested but is not available "
                    f"(registered: {solver_names}). Install its backend (e.g. "
                    f"'pip install ipyopt' for 'ipopt') or pick a registered solver and "
                    f"restate the configuration using that solver's own option names."
                )

            fallback = solver_names[0]
            logger.warning(
                "Default solver '%s' is not available (registered: %s), falling back to '%s'. "
                "Install the backend of '%s' (e.g. 'pip install ipyopt' for 'ipopt') or "
                "configure '%s' explicitly in prop_opt['solver'] using its own option names.",
                solver_name,
                solver_names,
                fallback,
                solver_name,
                fallback,
            )
            self.solver = fallback

    def _get_legacy_default_quantity(self, radiation_mode: str) -> str:
        """
        Return the quantity used by the legacy plan-wide objective conversion.

        Ion objectives are converted to ``rbe_x_dose`` only if the dij can provide it, as
        decided by :meth:`RBExDose.choose_path`; a dij computed without (applicable)
        biological model keeps its physical-dose objectives.
        """
        has_effect = (
            getattr(self._dij, "alpha_dose", None) is not None
            and getattr(self._dij, "sqrt_beta_dose", None) is not None
        )
        use_rbe = RBExDose.choose_path(self._dij, has_effect=has_effect, strict=False) is not None
        default_quantities = {
            "photons": "physical_dose",
            "protons": "rbe_x_dose" if use_rbe else "physical_dose",
            "helium": "rbe_x_dose" if use_rbe else "physical_dose",
            "carbon": "rbe_x_dose" if use_rbe else "physical_dose",
            "oxygen": "rbe_x_dose" if use_rbe else "physical_dose",
            "VHEE": "physical_dose",
        }
        return default_quantities.get(radiation_mode, "physical_dose")

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

        self._num_of_fractions = pln.num_of_fractions
        self._dose_convention = pln.dose_convention

        # Set Scenario Model
        self._scenarios = pln.scenarios

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
            # A copy, so that popping the selector keys leaves the plan untouched.
            prop_dict = dict(pln.prop_opt)
            # "problem" selects the class in get_problem_from_pln, which already warns when
            # it cannot honour it.
            prop_dict.pop("problem", None)
            requested = prop_dict.pop("opti_prob", None)
            if requested and requested != self.short_name:
                raise ValueError(
                    f"Inconsistent dose opti_probs given! pln asks for '{requested}'"
                    f", but you are using '{self.short_name}'!"
                )
        else:
            prop_dict = {}

        fields = prop_dict.keys()

        if "solver" in fields:
            self._solver_explicit = True

        # Set up warning message
        if warn_when_property_changed:
            warning_msg = "Property in Optimization Problem overwritten from pln.prop_opt"
        else:
            warning_msg = None

        for field in fields:
            if not hasattr(self, field):
                logger.warning("Property %s not found in Problem!", field)
            elif warn_when_property_changed and warning_msg:
                logger.warning(warning_msg + f": {field}")

            setattr(self, field, prop_dict[field])

    @abstractmethod
    def _solve(self) -> tuple[Array, dict]:
        """Solve the planning problem."""

    def _collect_objectives(self) -> tuple[list[tuple], list[tuple], list[str]]:
        """
        Parse the VOI objectives.

        Returns
        -------
        voi_objectives : list[tuple]
            ``(voi, objectives)`` pairs holding the problem's converted, preprocessed copies.
        objectives : list[tuple]
            The same as ``(mask, objectives)`` pairs, ``mask`` being the VOI's linear mask on
            the dose grid.
        quantity_ids : list[str]
            The quantity of every objective.
        """
        voi_objectives: list[tuple] = []
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
            # Copy so that quantity conversion / fraction normalization below do not
            # alter the objectives stored in the user's StructureSet.
            objs = [get_objective(obj).model_copy() for obj in valid_objectives]
            voi_objectives.append((voi, objs))

        if self.convert_dose_objectives:
            self._convert_dose_objectives([obj for _, objs in voi_objectives for obj in objs])

        objectives: list[tuple] = []
        quantity_ids: list[str] = []
        for voi, objs in voi_objectives:
            cube_ix = voi.indices_numpy
            linear_mask = voi_linear_mask(voi)
            for obj in objs:
                obj.preprocess_image_reference_parameters(
                    target_grid=self._dij.dose_grid, index_list=cube_ix
                )
            if self._dose_convention == "total":
                for obj in objs:
                    obj.normalize_to_fraction_num(self._num_of_fractions)
            objectives.append((linear_mask, objs))
            quantity_ids.extend([obj.quantity for obj in objs])

        return voi_objectives, objectives, quantity_ids

    def _robustness_options(self) -> RobustnessOptions:
        """Options handed to every robustness operator of the problem."""
        return RobustnessOptions()

    def _problem_scenarios(self, voi_objectives: list[tuple]) -> ScenarioSet:
        """
        Scenario set the robustness operators evaluate.

        The dij's set; without one, an implicit nominal set whose single scenario is slot 0.
        A dij with several scenario slots but no set is accepted only for non-robust
        objectives, which then use slot 0 as their nominal scenario.
        """
        scenarios = getattr(self._dij, "scenarios", None)
        if scenarios is not None:
            return scenarios
        slots = RTQuantity.default_scenarios(self._dij)
        if len(slots) > 1:
            names = {
                get_robustness_operator_class(
                    "scenario_stack" if obj.scenario_aware else obj.robustness
                ).name
                for _, objs in voi_objectives
                for obj in objs
            }
            robust = sorted(names - {"none"})
            if robust:
                raise ValueError(
                    f"The dose influence data has {len(slots)} scenario slots but no scenario "
                    f"set, which the robust objectives ({', '.join(robust)}) need for the "
                    "scenarios' meaning and weights. Attach one with Dij.attach_scenarios()."
                )
            message = (
                f"The dose influence data has {len(slots)} scenario slots but no scenario set; "
                "the non-robust objectives use slot 0 as the nominal scenario. Attach a set "
                "with Dij.attach_scenarios() to optimize robustly."
            )
            logger.warning(message)
            self.report_warning(message)
        return NominalScenarioModel().scenarios()

    def _convert_dose_objectives(self, objectives: list) -> None:
        """Legacy plan-wide conversion of physical-dose objectives to the biological dose.

        Physical-dose objectives are converted only when no objective already uses a
        biological quantity; otherwise the mix is taken as intended and kept.  Objectives
        on any other quantity (e.g. LET) are used literally.  Either outcome is reported
        as a warning.
        """
        # TODO: Replace this plan-wide legacy mode with explicit ``dose_auto``
        # objective intent; see docs/development/optimization_quantity_api.md.
        default_quantity = self._get_legacy_default_quantity(self._stf.beams[0].radiation_mode)
        if default_quantity == "physical_dose":
            return
        physical = [obj for obj in objectives if obj.quantity == "physical_dose"]
        if not physical:
            return

        quantities = get_available_quantities()
        biological = sorted(
            {
                obj.quantity
                for obj in objectives
                if getattr(quantities.get(obj.quantity), "biological", False)
            }
        )
        if biological:
            message = (
                f"{len(physical)} physical-dose objective(s) are not converted to "
                f"{default_quantity!r} because other objectives already use the biological "
                f"quantities {biological}; they are optimized on physical dose."
            )
        else:
            for obj in physical:
                obj.quantity = default_quantity
            message = (
                f"{len(physical)} physical-dose objective(s) were converted to "
                f"{default_quantity!r} for the biological model of the dose influence data. "
                "Set convert_dose_objectives=False to optimize physical dose instead."
            )
        logger.warning(message)
        self.report_warning(message)

    def _initialize(self):
        """Initialize the data for the planning problem."""

        # resampling to dose-grid
        self._ct = self._ct.resample_to_grid(self._dij.dose_grid)

        # apply overlap priorities
        if self.apply_overlap:
            self._cst = self._cst.apply_overlap_priorities()

        self._cst = self._cst.resample_on_new_ct(self._ct)

        # check consistency of bio models and dij
        if getattr(self._dij, "alphax", None) is not None:
            [ax, bx] = self._cst.get_reference_lq_params(True, self._dij.dose_grid)
            if not np.array_equal(ax, self._dij.alphax) or not np.array_equal(bx, self._dij.betax):
                logger.error(
                    "Inconsistency found in bio parameters between biologocal reference alpha and beta parameters in CST and Dij."
                )
        if self._dose_convention == "total":
            logger.info(
                "Objective references are interpreted for the total course "
                "(dose_convention='total'); converting them to one of %d fractions for the "
                "per-fraction optimization (doses and effects / n, sqrt_beta_dose / sqrt(n), "
                "LET unchanged).",
                self._num_of_fractions,
            )
        else:
            logger.info(
                "Objective doses are interpreted per fraction (dose_convention='per_fraction')."
            )
        # sanitize objectives and constraints and manage required quantities
        voi_objectives, objectives, _ = self._collect_objectives()
        self._objective_list = objectives
        self._scenario_set = self._problem_scenarios(voi_objectives)
        self._terms = build_robust_terms(
            voi_objectives,
            self._scenario_set,
            self._robustness_options(),
            masks=[mask for mask, _ in objectives],
        )

        # Every quantity computes the scenarios its terms need (a dependency the union of its
        # consumers'), so a non-robust objective evaluates the nominal scenario only.
        quantity_scenarios: dict[str, set[int]] = {}
        for term in self._terms:
            for identifier in term.quantities:
                quantity_scenarios.setdefault(identifier, set()).update(term.scenario_indices)
        if self.evaluate_all_scenarios:
            for indices in quantity_scenarios.values():
                indices.update(range(len(self._scenario_set)))
        if len(self._scenario_set) > 1:
            logger.info(
                "Scenarios evaluated per quantity: %s (set has %d scenario(s))",
                {k: sorted(v) for k, v in quantity_scenarios.items()},
                len(self._scenario_set),
            )

        # Resolve all requested quantities (plus their transitive dependencies) through a
        # shared resolver so that any quantity referenced by multiple roots is instantiated
        # exactly once. Only the quantities the requested ones actually use are evaluated.
        resolver = QuantityResolver(
            self._dij, scenarios={k: sorted(v) for k, v in quantity_scenarios.items()}
        )
        self._quantities = resolver.reachable(resolver.resolve(list(quantity_scenarios)))

        if len(set([q.array_backend for q in self._quantities])) == 1:
            self._array_backend = self._quantities[0].array_backend
        else:
            raise TypeError(
                "Inconsistent array backends used in quantities. Decide on one (e.g. numpy)"
            )

        # Match quantities with terms: per quantity the terms reading it (one cache row each,
        # also for a composite term with several items on it), per term its row per quantity.
        self._q_cache_index = []
        self._objectives_per_quantity = {q.identifier: [] for q in self._quantities}
        for term_ix, term in enumerate(self._terms):
            rows = {}
            for identifier in term.quantities:
                rows[identifier] = len(self._objectives_per_quantity[identifier])
                self._objectives_per_quantity[identifier].append(term_ix)
            self._q_cache_index.append(rows)

        self._num_objectives = len(self._terms)

        # set solver options
        self.solver = get_solver(self.solver)
        self._propagate_solver_properties()

        # Let the solver push status (and honour pause/stop) through this problem,
        # which is the top-level workflow step observed by callers (e.g. the GUI).
        self.solver.status_callback = self._emit_solver_status

        # initial point

    def _propagate_solver_properties(self):
        """Forward the top-level solver knobs that were explicitly set on this problem.

        Only assigned values are forwarded, so a solver configured through a
        ``prop_opt["solver"]`` dictionary keeps that configuration. A knob the solver does not
        define is reported rather than applied: not every solver is iterative (``max_iter``) or
        produces output (``display``), and silently attaching the attribute would leave the
        caller believing a setting took effect.
        """
        for prop in self.solver_properties:
            value = getattr(self, prop)
            if value is None:
                continue

            if not hasattr(self.solver, prop):
                logger.warning(
                    "Solver '%s' has no '%s' property, so the value configured on the planning "
                    "problem is ignored.",
                    getattr(self.solver, "short_name", type(self.solver).__name__),
                    prop,
                )
                continue

            setattr(self.solver, prop, value)

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

        if self.obj_history is not None:
            # Cleared in place rather than replaced, so a caller holding the list still sees
            # this solve's history and not the previous one's appended to it.
            self.obj_history.clear()

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
