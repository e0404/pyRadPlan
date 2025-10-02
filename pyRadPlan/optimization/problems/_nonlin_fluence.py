from typing import Union, cast
import time
import logging

import numpy as np
from numpy.typing import NDArray


from ...plan import Plan
from ._optiprob import NonLinearPlanningProblem
from ..objectives import Objective

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
        self.bypass_objective_jacobian = False

        self._target_voxels = None
        self._patient_voxels = None

        self._grad_cache_intermediate = None
        self._grad_cache = None
        self._solve_time = None

        super().__init__(pln)

    def _initialize(self):
        """Initialize this problem."""
        super()._initialize()

        # # Check if the solver is adequate to solve this problem
        # # TODO: check that it can do constraints
        # # if not isinstance(self.solver, NonLinearOptimizer):
        # #     raise ValueError("Solver must be an instance of SolverBase")
        # self.solver = get_solver(self.solver)
        # self.solver.objective = self._objective_function
        # self.solver.gradient = self._objective_gradient
        # self.solver.bounds = (0.0, np.inf)
        # self.solver.max_iter = 500

    def _evaluate_objective_functions(self, x: NDArray) -> NDArray:
        """Define the objective functions."""

        q_vectors = {}
        q_scenarios = {}

        # Check & get Caches
        for q in self._quantities:
            q_vectors[q.identifier] = q(x)
            q_scenarios[q.identifier] = q.scenarios

        # Loop over all objectives
        f_vals = []

        for obj_info in self._objective_list:
            ix = obj_info[0]
            tmp_obj_list = cast(list[Objective], obj_info[1])
            for obj in tmp_obj_list:
                f_vals.append(
                    sum(
                        [
                            obj.compute_objective(q_vectors[obj.quantity].flat[scen_ix][ix])
                            for scen_ix in q_scenarios[obj.quantity]
                        ]
                    )
                )

        # return as numpy array
        return np.asarray(f_vals, dtype=np.float64)

    # def _objective_function(self, x: NDArray) -> np.float64:
    #     f = np.sum(self._evaluate_objective_functions(x))
    #     return f

    def _evaluate_objective_jacobian(self, x: NDArray) -> NDArray:
        """Define the objective jacobian."""

        q_vectors = {}
        q_scenarios = {}
        if self._grad_cache_intermediate is None:
            initialize_cache = True
            self._grad_cache_intermediate = {}
        else:
            initialize_cache = False

        # Check & get Caches
        for q in self._quantities:
            q_vectors[q.identifier] = q(x)
            q_scenarios[q.identifier] = q.scenarios
            if initialize_cache:
                if self.bypass_objective_jacobian:
                    cache_rows = 1
                else:
                    cache_rows = len(self._objectives_per_quantity[q.identifier])
                self._grad_cache_intermediate[q.identifier] = np.zeros(
                    (
                        cache_rows,
                        self._dij.dose_grid.num_voxels,
                    ),
                    dtype=np.float32,
                )
            else:
                self._grad_cache_intermediate[q.identifier].fill(0.0)

        cnt = 0
        for obj_info in self._objective_list:
            ix = obj_info[0]
            tmp_obj_list = cast(list[Objective], obj_info[1])
            for obj in tmp_obj_list:
                if self.bypass_objective_jacobian:
                    q_cache_index = 0
                else:
                    q_cache_index = self._q_cache_index[cnt]
                for scen_ix in q_scenarios[obj.quantity]:
                    self._grad_cache_intermediate[obj.quantity][q_cache_index, ix] += (
                        obj.compute_gradient(q_vectors[obj.quantity].flat[scen_ix][ix])
                    )
                cnt += 1

        # perform chain rule and store in cache
        if self._grad_cache is None:
            if self.bypass_objective_jacobian:
                n_grad_caches = 1
            else:
                n_grad_caches = cnt

            self._grad_cache = np.zeros(
                (n_grad_caches, self._dij.total_num_of_bixels), dtype=np.float64
            )
        else:
            self._grad_cache.fill(0.0)

        for q in self._quantities:
            for scen_ix in q_scenarios[q.identifier]:
                if self.bypass_objective_jacobian:
                    cache_ix = 0
                else:
                    cache_ix = self._objectives_per_quantity[q.identifier]

                self._grad_cache[cache_ix, :] += (
                    q.compute_chain_derivative(self._grad_cache_intermediate[q.identifier], x)
                    .flat[scen_ix]
                    .squeeze()
                )

        return self._grad_cache

    def _evaluate_objective_hessian(self, x: NDArray) -> NDArray:
        """Define the objective hessian."""
        return {}

    def _evaluate_constraint_functions(self, x: NDArray) -> NDArray:
        """Define the constraint functions."""
        return None

    def _evaluate_constraint_jacobian(self, x: NDArray) -> NDArray:
        """Define the constraint jacobian."""
        return None

    def _evaluate_constraint_jacobian_structure(self) -> NDArray:
        """Define the constraint jacobian structure."""
        return None

    def _get_variable_bounds(self, x: NDArray) -> NDArray:
        """Define the variable bounds."""
        return {}

    def _solve(self) -> tuple[NDArray, dict]:
        """Solve the problem."""

        x0 = np.zeros((self._dij.total_num_of_bixels,), dtype=np.float64)
        t = time.time()
        result = self.tradeoff_strategy.solve(x0)
        # result = self.solver.solve(x0)
        self._solve_time = time.time() - t

        logger.info("Solver time: %g s", self._solve_time)

        return result
