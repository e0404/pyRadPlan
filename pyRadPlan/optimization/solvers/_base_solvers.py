"""Solver Base Classes for Planning Problems."""

from typing import ClassVar, Callable
from abc import ABC, abstractmethod

from ...core.xp_utils.typing import Array

import numpy as np
# from numpy.typing import ArrayLike


class SolverBase(ABC):
    """
    Abstract Base Class for Solver Implementations / Interfaces.

    Attributes
    ----------
    name : ClassVar[str]
        Full name of the solver
    short_name : ClassVar[str]
        Short name of the solver
    max_time : float, default=3600
        Maximum time for the solver to run in seconds
    bounds : Array, default=[0.0, np.inf]
        Bounds for the variables
    """

    name = ClassVar[str]
    short_name = ClassVar[str]
    gpu_compatible = ClassVar[bool]

    # properties
    max_time: float
    bounds: Array

    def __init__(self):
        self.max_time = 3600
        self.bounds = [0.0, np.inf]

    def __repr__(self) -> str:
        return f"Solver {self.name} ({self.short_name})"

    @abstractmethod
    def solve(self, x0: Array) -> tuple[Array, dict]:
        """
        Interface method to solve the problem.

        Parameters
        ----------
        x0 : Array
            Initial guess for the solution

        Returns
        -------
        tuple[Array, dict]
            Solution vector and additional information as dictionary
        """


class NonLinearOptimizer(SolverBase):
    """
    Non-Linear Optimization Solver Base Class.

    Attributes
    ----------
    max_iter : int
        Maximum number of iterations
    abs_obj_tol : float
        Absolute objective tolerance
    objective : Callable
        Objective function handle
    gradient : Callable
        Gradient function handle
    hessian : Callable, default=None
        Hessian function handle
    constraints : Callable, default=None
        Constraints function handle
    constraints_jac : Callable, default=None
        Constraints Jacobian function handle
    supply_iter_func : bool
        Whether to supply an iteration callback function
    """

    max_iter: int
    abs_obj_tol: float

    objective: Callable
    gradient: Callable
    hessian: Callable
    constraints: Callable
    constraints_jac: Callable

    supply_iter_func: bool

    def __init__(self):
        super().__init__()
        self.max_iter = 500
        self.abs_obj_tol = 1e-6

        self.objective = None
        self.gradient = None
        self.hessian = None
        self.constraints = None
        self.constraints_jac = None

    def iter_func(self, *args, **kwargs) -> bool:
        """
        Get or set solver information as iteration callback.

        Agnostic signature with *args and **kwargs to be able to accomodate
        various solvers.

        Parameters
        ----------
        *args
            Additional arguments
        **kwargs
            Additional keyword arguments

        Returns
        -------
        bool
            Whether to continue the optimization
        """
        return True
