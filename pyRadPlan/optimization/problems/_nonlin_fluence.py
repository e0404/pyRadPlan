from typing import Union, cast
import logging

import numpy as np
from numpy.typing import NDArray

import array_api_compat

from ...core.xp_utils.typing import Array

from ...plan import Plan
from ._optiprob import NonLinearPlanningProblem
from ..solvers import NonLinearOptimizer
from ..objectives import Objective
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
        Whether to bypass the objective jacobian calculation. This is usefull for scalarized
        optimization (e.g. weighted sum of objectives) as it will minimize storage to only a single
        gradient vector per quantity.
    """

    name = "Non-Linear Fluence Planning Problem"
    short_name = "nonlin_fluence"

    bypass_objective_jacobian: bool

    def __init__(self, pln: Union[Plan, dict] = None):
        self.bypass_objective_jacobian = True

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

        # Check if the solver is adequate to solve this problem
        # TODO: check that it can do constraints
        if not isinstance(self.solver, NonLinearOptimizer):
            raise ValueError("Solver must be an instance of SolverBase")

        self.solver.objective = self._objective_function
        self.solver.gradient = self._objective_gradient
        self.solver.bounds = (0.0, float("inf"))
        self.solver.max_iter = 500

    def _objective_functions(self, x: Array) -> Array:
        """Define the objective functions."""

        xp = array_api_compat.array_namespace(x)

        q_vectors = {}
        q_scenarios = {}

        # Check & get Caches
        for q in self._quantities:
            q_vectors[q.identifier] = q(x)
            q_scenarios[q.identifier] = q.scenarios

        # Loop over all objectives

        f_vals = xp.zeros(self._num_objectives, dtype=x.dtype)

        obj_ix = 0
        for obj_info in self._objective_list:
            ix = xp_utils.from_numpy(xp, obj_info[0])
            tmp_obj_list = cast(list[Objective], obj_info[1])
            for obj in tmp_obj_list:
                f_tmp_scen = [
                    obj.priority * obj.compute_objective(q_vectors[obj.quantity].flat[scen_ix][ix])
                    for scen_ix in q_scenarios[obj.quantity]
                ]
                f_vals[obj_ix] = xp.sum(xp.stack(f_tmp_scen))
                obj_ix += 1

        return f_vals

    def _objective_function(self, x: Array) -> Array:
        xp = array_api_compat.array_namespace(x)
        t = xp_utils.record_event(xp)
        f = xp.sum(self._objective_functions(x))
        evt = xp_utils.record_event(xp)
        xp_utils.synchronize(xp)
        self._obj_times.append(xp_utils.elapsed_time(xp, t, evt))
        return f

    def _set_quantity_imtermediate_gradient_cache(self):
        """Initialize or reset the intermediate gardient caches for all quantities."""

        if self._grad_cache_intermediate is None:
            initialize_cache = True
            self._grad_cache_intermediate = {}

        else:
            initialize_cache = False

        xp = self._array_backend

        # Check & get Caches
        for q in self._quantities:
            if initialize_cache:
                if self.bypass_objective_jacobian:
                    cache_rows = 1
                else:
                    cache_rows = len(self._objectives_per_quantity[q.identifier])

                self._grad_cache_intermediate[q.identifier] = xp.zeros(
                    (
                        cache_rows,
                        self._dij.dose_grid.num_voxels,
                    ),
                    dtype=xp.float32,
                )
            else:
                # The trailing ellipsis here is needed for array_api compatibility
                self._grad_cache_intermediate[q.identifier][:, ...] = 0.0

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
                (n_grad_caches, self._dij.total_num_of_bixels), dtype=xp.float64
            )
        else:
            self._grad_cache[:, ...] = 0.0

        for q in self._quantities:
            for scen_ix in q_scenarios[q.identifier]:
                if self.bypass_objective_jacobian:
                    cache_ix = 0
                else:
                    cache_ix = self._objectives_per_quantity[q.identifier]

                self._grad_cache[cache_ix, :] += xp.squeeze(
                    q.compute_chain_derivative(
                        self._grad_cache_intermediate[q.identifier], x
                    ).flat[scen_ix],
                    axis=0,
                )

    def _objective_jacobian(self, x: Array) -> Array:
        """Define the objective jacobian."""

        xp = array_api_compat.array_namespace(x)

        self._set_quantity_imtermediate_gradient_cache()

        q_vectors = {q.identifier: q(x) for q in self._quantities}
        q_scenarios = {q.identifier: q.scenarios for q in self._quantities}

        cnt = 0
        for obj_info in self._objective_list:
            ix = xp_utils.from_numpy(xp, obj_info[0])
            tmp_obj_list = cast(list[Objective], obj_info[1])
            for obj in tmp_obj_list:
                if self.bypass_objective_jacobian:
                    q_cache_index = 0
                else:
                    q_cache_index = self._q_cache_index[cnt]
                for scen_ix in q_scenarios[obj.quantity]:
                    self._grad_cache_intermediate[obj.quantity][q_cache_index, ...][ix] += (
                        xp.astype(
                            obj.priority
                            * obj.compute_gradient(q_vectors[obj.quantity].flat[scen_ix][ix]),
                            self._grad_cache_intermediate[obj.quantity][q_cache_index, ...][
                                ix
                            ].dtype,
                            copy=False,
                        )
                    )
                cnt += 1

        self._update_objective_grad_cache(x, num_objectives=cnt, q_scenarios=q_scenarios)

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

        x0 = xp.zeros((self._dij.total_num_of_bixels,), dtype=xp.float64)
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
