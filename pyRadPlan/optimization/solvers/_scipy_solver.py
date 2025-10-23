"""SciPy solver Class."""

from typing import Callable, Union

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import minimize, Bounds

from ._base_solvers import NonLinearOptimizer


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

    def solve(self, x0: ArrayLike) -> tuple[np.ndarray, dict]:
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

        x0 = np.asarray(x0)

        bounds = Bounds(lb=self.bounds[0], ub=self.bounds[1])

        # Initialize the SciPy solution function and its arguments
        result = minimize(
            x0=x0,
            fun=self.objective,
            method=self.method,
            jac=self.gradient,
            # constraints=self.constraints,
            # hess=self.hessian,
            tol=self.abs_obj_tol,
            bounds=bounds,
            options=self.options,
        )

        return result["x"], result
