from typing import Literal, Union
import logging

import numpy as np
from numpy.typing import NDArray

import array_api_compat

from ...core.xp_utils.typing import Array

from ...plan import Plan
from ._optiprob import NonLinearPlanningProblem
from ..solvers import NonLinearOptimizer
from ..robustness import RobustnessOptions, RobustTerm
from ..robustness._utils import scenario_value
from ...core import xp_utils

logger = logging.getLogger(__name__)


class NonLinearFluencePlanningProblem(NonLinearPlanningProblem):
    """
    Non-linear fluence-based planning problem.

    Parameters
    ----------
    pln : Union[Plan, dict], optional
        Plan object or dictionary to initialize the problem with.

    Attributes
    ----------
    bypass_objective_jacobian : bool, optional, default=True
        Whether to bypass the objective jacobian calculation. This is useful for scalarized
        optimization (e.g. weighted sum of objectives) as it will minimize storage to only a single
        gradient vector per quantity.
    max_approximation : {"logsumexp", "pnorm", "none"}, default="logsumexp"
        Smooth approximation of the maximum over scenarios used by the worst-case robustness
        operators (``"none"``: exact maximum).
    p_norm : float, default=30.0
        Exponent of the ``"pnorm"`` maximum approximation.
    """

    name = "Non-Linear Fluence Planning Problem"
    short_name = "nonlin_fluence"
    records_obj_history = True

    bypass_objective_jacobian: bool = True
    max_approximation: Literal["logsumexp", "pnorm", "none"] = "logsumexp"
    p_norm: float = 30.0

    def __init__(self, pln: Union[Plan, dict] = None):
        self._target_voxels = None
        self._patient_voxels = None

        self._grad_cache_intermediate = None
        self._grad_cache = None
        self._obj_times = []
        self._deriv_times = []
        self._solve_time = None

        super().__init__(pln)

    def _initialize(self):
        """Initialize this problem."""
        super()._initialize()

        # Their shapes follow the terms' scenarios and quantities, which can change between runs
        self._grad_cache_intermediate = None
        self._grad_cache = None

        # Check if the solver is adequate to solve this problem
        # TODO: check that it can do constraints
        if not isinstance(self.solver, NonLinearOptimizer):
            raise ValueError(
                f"Problem '{self.short_name}' is non-linear and needs a NonLinearOptimizer, "
                f"but solver '{getattr(self.solver, 'short_name', self.solver)}' is not one."
            )

        # Propagate device from quantities to solver and problem
        self._device = getattr(self._quantities[0], "device", None)
        self.solver.device = self._device

        # Pure dependencies (e.g. physical_dose below rbe_x_dose) are evaluated through the
        # quantities using them; computing them here would cost an extra matrix product.
        self._objective_quantities = [
            q for q in self._quantities if self._objectives_per_quantity[q.identifier]
        ]
        # Position of each flat scenario index in a quantity's cotangent blocks.
        self._scenario_positions = {
            q.identifier: {int(s): pos for pos, s in enumerate(q.scenarios)}
            for q in self._objective_quantities
        }

        self.solver.objective = self._objective_function
        self.solver.gradient = self._objective_gradient
        self.solver.bounds = (0.0, float("inf"))

    def _robustness_options(self) -> RobustnessOptions:
        return RobustnessOptions(max_approximation=self.max_approximation, p_norm=self.p_norm)

    def _term_values(self, xp, q_vectors: dict, term: RobustTerm) -> tuple[list, list]:
        """
        Device masks of the term's items per scenario and the items' values.

        The values are stacked over the term's scenarios, or a list per scenario if the
        contours differ between them.
        """
        masks = []
        values = []
        for item, item_masks in zip(term.items, term.scenario_masks):
            device = {id(m): xp_utils.from_numpy(xp, m, device=self._device) for m in item_masks}
            item_ix = [device[id(m)] for m in item_masks]
            scenario_values = [
                q_vectors[item.quantity].flat[s][ix]
                for s, ix in zip(term.scenario_indices, item_ix)
            ]
            masks.append(item_ix)
            values.append(xp.stack(scenario_values) if len(device) == 1 else scenario_values)
        return masks, values

    def _objective_functions(self, x: Array) -> Array:
        """Define the objective functions: one value per robust term."""

        xp = array_api_compat.array_namespace(x)

        q_vectors = {q.identifier: q(x) for q in self._objective_quantities}

        f_vals = xp.zeros(self._num_objectives, dtype=x.dtype, device=self._device)

        for term_ix, term in enumerate(self._terms):
            _, values = self._term_values(xp, q_vectors, term)
            value = xp.asarray(term.operator.evaluate(term.entries, values))
            f_vals = xp_utils.scatter(f_vals, term_ix, xp.astype(value, f_vals.dtype))

        return f_vals

    def _objective_function(self, x: Array) -> Array:
        xp = array_api_compat.array_namespace(x)
        t = xp_utils.record_event(xp)
        f = xp.sum(self._objective_functions(x))
        evt = xp_utils.record_event(xp)
        xp_utils.synchronize(xp)
        self._obj_times.append(xp_utils.elapsed_time(xp, t, evt))
        if self.obj_history is not None:
            self.obj_history.append(float(f))
        return f

    def _set_quantity_imtermediate_gradient_cache(self):
        """Initialize or reset the intermediate gardient caches for all quantities."""

        if self._grad_cache_intermediate is None:
            initialize_cache = True
            self._grad_cache_intermediate = {}

        else:
            initialize_cache = False

        xp = self._array_backend

        # One cotangent block per scenario the quantity evaluates, so that each scenario's
        # objective gradient is back-projected through its own influence matrix.
        for q in self._objective_quantities:
            if initialize_cache:
                if self.bypass_objective_jacobian:
                    cache_rows = 1
                else:
                    cache_rows = len(self._objectives_per_quantity[q.identifier])

                self._grad_cache_intermediate[q.identifier] = xp.zeros(
                    (
                        len(q.scenarios),
                        cache_rows,
                        self._dij.dose_grid.num_voxels,
                    ),
                    dtype=xp.float32,
                    device=self._device,
                )
            else:
                # The trailing ellipsis here is needed for array_api compatibility
                self._grad_cache_intermediate[q.identifier] = xp_utils.scatter(
                    self._grad_cache_intermediate[q.identifier], (slice(None), ...), 0.0
                )

    def _update_objective_grad_cache(self, x: Array, num_objectives: int, q_scenarios: dict):
        """Update the objective gradient / jacobian cache by computing the chain rule."""

        xp = array_api_compat.array_namespace(x)

        # perform chain rule and store in cache
        if self._grad_cache is None:
            if self.bypass_objective_jacobian:
                n_grad_caches = 1
            else:
                n_grad_caches = num_objectives

            self._grad_cache = xp.zeros(
                (n_grad_caches, self._dij.total_num_of_bixels),
                dtype=xp.float64,
                device=self._device,
            )
        else:
            self._grad_cache = xp_utils.scatter(self._grad_cache, (slice(None), ...), 0.0)

        for q in self._objective_quantities:
            if self.bypass_objective_jacobian:
                cache_ix = [0]
            else:
                cache_ix = self._objectives_per_quantity[q.identifier]

            cotangents = np.empty_like(q._q_cache)
            for pos, scen_ix in enumerate(q_scenarios[q.identifier]):
                cotangents.flat[scen_ix] = self._grad_cache_intermediate[q.identifier][pos, ...]
            # Derived scenarios are folded onto their basis slots by the quantity, so the
            # non-empty slots sum to the full derivative.
            derivatives = q.compute_chain_derivative(cotangents, x)
            for derivative in derivatives.flat:
                if derivative is None:
                    continue
                # Row by row, as the Array API has no integer-array (scatter) indexing.
                for row, obj_ix in enumerate(cache_ix):
                    self._grad_cache = xp_utils.scatter(
                        self._grad_cache,
                        (obj_ix, slice(None)),
                        self._grad_cache[obj_ix, :] + derivative[row, :],
                    )

    def _objective_jacobian(self, x: Array) -> Array:
        """Define the objective jacobian."""

        xp = array_api_compat.array_namespace(x)

        self._set_quantity_imtermediate_gradient_cache()

        q_vectors = {q.identifier: q(x) for q in self._objective_quantities}
        q_scenarios = {q.identifier: q.scenarios for q in self._objective_quantities}

        for term_ix, term in enumerate(self._terms):
            masks, values = self._term_values(xp, q_vectors, term)
            grads = term.operator.gradient(term.entries, values)
            for item, item_ix, item_grad in zip(term.items, masks, grads):
                row = (
                    0
                    if self.bypass_objective_jacobian
                    else self._q_cache_index[term_ix][item.quantity]
                )
                positions = self._scenario_positions[item.quantity]
                cache = self._grad_cache_intermediate[item.quantity]
                for k, scen_ix in enumerate(term.scenario_indices):
                    pos = positions[scen_ix]
                    ix = item_ix[k]
                    grad = xp.astype(scenario_value(item_grad, k), cache.dtype, copy=False)
                    # A view on mutable backends, so the scatter below writes through to cache;
                    # where it cannot (JAX, out-of-place fallback) the block is written back.
                    block = cache[pos, row, ...]
                    updated = xp_utils.scatter(block, ix, block[ix] + grad)
                    if updated is not block:
                        cache = xp_utils.scatter(cache, (pos, row, ...), updated)
                self._grad_cache_intermediate[item.quantity] = cache

        self._update_objective_grad_cache(
            x, num_objectives=self._num_objectives, q_scenarios=q_scenarios
        )

        return self._grad_cache

    def _objective_gradient(self, x: Array) -> Array:
        xp = array_api_compat.array_namespace(x)
        t1 = xp_utils.record_event(xp)
        jac = xp.sum(self._objective_jacobian(x), axis=0)
        t2 = xp_utils.record_event(xp)
        xp_utils.synchronize(xp)
        self._deriv_times.append(xp_utils.elapsed_time(xp, t1, t2))
        return jac

    def _objective_hessian(self, x: Array) -> Array:
        """Define the objective hessian."""
        return None

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
        return {}

    def _solve(self) -> tuple[NDArray, dict]:
        """Solve the problem."""

        self._deriv_times = []
        self._obj_times = []

        xp = self._array_backend

        x0 = xp.zeros((self._dij.total_num_of_bixels,), dtype=xp.float64, device=self._device)
        t_start = xp_utils.record_event(xp)
        result = self.solver.solve(x0)
        t_end = xp_utils.record_event(xp)
        xp_utils.synchronize(xp)
        self._solve_time = xp_utils.elapsed_time(xp, t_start, t_end)

        logger.info(
            "%d Objective function evaluations, avg. time: %g +/- %g s",
            len(self._obj_times),
            np.mean(self._obj_times),
            np.std(self._obj_times),
        )
        logger.info(
            "%d Derivative evaluations, avg. time: %g +/- %g s",
            len(self._deriv_times),
            np.mean(self._deriv_times),
            np.std(self._deriv_times),
        )
        logger.info("Solver time: %g s", self._solve_time)

        return result
