from typing import Union
import numpy as np
import time
import logging


from ._base_scalarization_strategies import ScalarizationStrategyBase

logger = logging.getLogger(__name__)


class WeightedSum(ScalarizationStrategyBase):
    name = "Weighted sum scalarization strategy"
    short_name = "weighted_sum"
    weights: Union[np.ndarray[float], list[float], None]

    def __init__(
        self, scalarization_model_params, callbacks: dict[str, callable], solver: Union[str, dict]
    ):
        #                 weights: Union[np.ndarray[float],list[float]] = None):
        # functions in the planning problem that are required for the actual optimization
        super().__init__(scalarization_model_params, callbacks, solver)
        # self.weights = np.asarray(weights)

    def variable_lower_bounds(self):
        return self.callbacks["get_variable_bounds"]()[0]

    def variable_upper_bounds(self):
        return self.callbacks["get_variable_bounds"]()[1]

    def get_linear_constraints(self):
        pass

    def get_nonlinear_constraints(self):
        pass

    def evaluate_objective(x):
        pass

    def evaluate_objective_jacobian(x):
        pass

    def _evaluate_objective_function(self, x: np.ndarray) -> np.float64:
        t = time.time()
        f = self.scalarization_model_params @ self.callbacks["evaluate_objective_functions"](x)
        self._obj_times.append(time.time() - t)

        # self.weights@self.callbacks["evaluate_objective_functions"](x)
        return f

    def _evaluate_objective_gradient(self, x: np.ndarray) -> np.ndarray:
        t = time.time()

        jac = np.sum(
            self.callbacks["evaluate_objective_jacobian"](x)
            * self.scalarization_model_params[:, None],
            axis=0,
        )
        self._deriv_times.append(time.time() - t)
        return jac

    def evaluate_constraints(self, x):
        return self.callbacks["evaluate_constraints"](x)

    def _solve(self, x: np.ndarray[float]) -> list[np.ndarray[float]]:
        self.solver.objective = self._evaluate_objective_function
        self.solver.gradient = self._evaluate_objective_gradient
        self.solver.bounds = (0.0, np.inf)
        self.solver.max_iter = 500

        result = self.solver.solve(x)

        # needs to be moved
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

        return result
