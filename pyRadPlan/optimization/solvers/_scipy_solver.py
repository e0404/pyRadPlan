"""SciPy solver Class."""

from typing import Callable, Union

import array_api_compat
import numpy as np

from ...core.xp_utils.typing import Array

from scipy.optimize import minimize, Bounds

from ._base_solvers import NonLinearOptimizer
from ...core import xp_utils


class OptimizerSciPy(NonLinearOptimizer):
    """
    SciPy solver configuration class.

    Attributes
    ----------
    options : dict
        Options for the solver
    method : Union[str, Callable]
        The solver method
    """

    name = "SciPy minimize"
    short_name = "scipy"
    gpu_compatible = False

    options: dict[str]
    method: Union[str, Callable]

    def __init__(self):
        self.options = {
            "disp": True,
            "ftol": 1e-5,
            "gtol": 1e-5,
        }

        self.method = "L-BFGS-B"

        super().__init__()

    def solve(self, x0: Array) -> tuple[Array, dict]:
        """
        Solve the problem.

        Parameters
        ----------
        x0 : np.ndarray
            Initial guess for the decision variables.

        Returns
        -------
        result : dict
        """

        self.options.update({"maxiter": self.max_iter})

        if isinstance(x0, list):
            x0 = np.asarray(x0)

        xp = array_api_compat.array_namespace(x0)

        x0 = xp_utils.to_numpy(x0)
        bounds = [xp_utils.to_numpy(xp.asarray(b)) for b in self.bounds]
        bounds = Bounds(lb=bounds[0], ub=bounds[1])

        def scipy_objective(x: Array):
            return xp_utils.to_numpy(self.objective(xp_utils.from_numpy(xp, x)))

        def scipy_gradient(x: Array):
            return xp_utils.to_numpy(self.gradient(xp_utils.from_numpy(xp, x)))

        # Initialize the SciPy solution function and its arguments
        result = minimize(
            x0=x0,
            fun=scipy_objective,
            method=self.method,
            jac=scipy_gradient,
            # constraints=self.constraints,
            # hess=self.hessian,
            tol=self.abs_obj_tol,
            bounds=bounds,
            options=self.options,
        )

        return xp_utils.from_numpy(xp, result["x"]), result
