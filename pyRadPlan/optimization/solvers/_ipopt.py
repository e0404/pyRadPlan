"""
Ipopt solver for non-linear optimization problems.

Notes
-----
Not installed by default. Uses ipyopt because it provides linux wheels
"""

from ipyopt import Problem

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._base_solvers import NonLinearOptimizer


class OptimizerIpopt(NonLinearOptimizer):
    """
    IPOPT solver interface.

    Attributes
    ----------
    options : dict
        Options for IPOPT
    """

    name = "Interior Point Optimizer"
    short_name = "ipopt"

    options: dict[str]

    def __init__(self):
        self.result = None

        super().__init__()

        self.options = {
            "print_level": 5,
            "print_user_options": "no",
            "print_options_documentation": "no",
            "tol": 1e-10,
            "dual_inf_tol": 1e-4,
            "constr_viol_tol": 1e-4,
            "compl_inf_tol": 1e-4,
            "acceptable_iter": 5,
            "acceptable_tol": self.abs_obj_tol,
            "acceptable_constr_viol_tol": 1e-2,
            "acceptable_dual_inf_tol": 1e10,
            "acceptable_compl_inf_tol": 1e10,
            "acceptable_obj_change_tol": 1e-4,
            "max_iter": self.max_iter,
            "max_cpu_time": float(self.max_time),
            "mu_strategy": "adaptive",
            "hessian_approximation": "limited-memory",
            "limited_memory_max_history": 20,
            "limited_memory_initialization": "scalar2",
            "linear_solver": "mumps",
            "print_timing_statistics": "yes",
        }

    def solve(self, x0: ArrayLike) -> tuple[np.ndarray, dict]:
        """
        Solve the problem.

        Parameters
        ----------
        x0 : ArrayLike
            Initial guess for the decision variables.

        Returns
        -------
        result : dict
        """

        self.options.update(
            {
                "max_iter": self.max_iter,
                "max_cpu_time": float(self.max_time),
                "acceptable_tol": self.abs_obj_tol,
            }
        )

        x0 = np.asarray(x0)

        eval_jac_g_sparsity_indices = (np.array([]), np.array([]))
        eval_h_sparsity_indices = (np.array([]), np.array([]))

        def ipopt_derivative(x: NDArray[np.float64], out: NDArray[np.float64]):
            out[()] = self.gradient(x).astype(np.float64)
            return out

        # Set the optimization function
        nlp = Problem(
            n=x0.size,
            x_l=np.zeros_like(x0),
            x_u=np.inf * np.ones_like(x0),
            m=0,
            g_l=np.empty((0,)),
            g_u=np.empty((0,)),
            sparsity_indices_jac_g=eval_jac_g_sparsity_indices,
            sparsity_indices_h=eval_h_sparsity_indices,
            eval_f=self.objective,
            eval_grad_f=ipopt_derivative,
            eval_g=lambda _x, _out: None,
            eval_jac_g=lambda _x, _out: None,
            eval_h=None,
            ipopt_options=self.options,
        )

        x, _, status = nlp.solve(x0=x0)

        return x, status
