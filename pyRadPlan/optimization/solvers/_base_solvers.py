"""Solver Base Classes for Planning Problems."""

from typing import ClassVar, Callable, Any
from abc import ABC, abstractmethod

from ...core.xp_utils.typing import Array
from ...util.keyboard_listener import KeyboardListener

import numpy as np
from numpy.typing import ArrayLike
import logging

logger = logging.getLogger(__name__)


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

    allow_keyboard_cancel: bool = False
    cancel_key: str = "q"
    allow_esc_cancel: bool = True  # Also allow ESC (\x1b) to cancel
    # properties
    max_time: float
    bounds: Array

    def __init__(self):
        self.max_time = 3600
        self.bounds = [0.0, np.inf]
        # Keyboard listener utility (manages platform specifics internally)
        self._keyboard_listener = KeyboardListener(
            allow_keyboard_cancel=self.allow_keyboard_cancel,
            cancel_key=self.cancel_key,
            allow_esc_cancel=self.allow_esc_cancel,
            component_name=lambda: getattr(self, "name", "Solver"),
        )
        # Reflect effective capability (may be disabled by platform checks)
        self.allow_keyboard_cancel = self._keyboard_listener.allow_keyboard_cancel

    def __repr__(self) -> str:
        return f"Solver {self.name} ({self.short_name})"

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

        self._keyboard_listener.initialize()

        # Important!: Everything between start and end kb_thread must be
        # inside try/finally to ensure thread is stopped on error!.
        # Or terminal stays in weird state.
        try:
            logger.info(f"Starting optimization using {self.name}")
            x, status = self._solve_problem(x0)

        # Ensure thread is stopped.
        finally:
            self._keyboard_listener.finalize()

        return x, status

    @abstractmethod
    def _solve_problem(self, x0: ArrayLike) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Solve the problem.

        Parameters
        ----------
        x0 : Array
            Initial guess for the solution

        Returns
        -------
        tuple[Array, dict]
            Solution vector and additional information as dictionary
        """

    @abstractmethod
    def _callback(self, *args: Any, **kwargs: Any) -> Any:
        """
        Check for early stopping during solver callback.

        This method should handle both early stopping checks and
        user-provided callback functions if the optimizer supports them.
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
