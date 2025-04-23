"""Scalarization Strategy Base Classes for Planning Problems."""

from typing import ClassVar, Callable, Union
from abc import ABC, abstractmethod
import numpy as np
from ...solvers import get_solver, SolverBase


class ScalarizationStrategyBase(ABC):
    """
    Abstract Base Class for Scalarization Strategy Implementations / Interfaces.

    Parameters
    ----------
    callbacks : dict[str, Callable]
        Functions in the planning problem that are required for the actual optimization

    Attributes
    ----------
    name : ClassVar[str]
        Full name of the scalarization strategy
    short_name : ClassVar[str]
        Short name of the scalarization strategy
    callbacks : dict[str, Callable]
        Functions in the planning problem that are required for the actual optimization
    solver : Union[str, dict, SolverBase]
        Solver to be used for optimization
    """

    name: ClassVar[str]
    short_name: ClassVar[str]
    # scalarization_params: dict
    callbacks: dict[str, Callable]
    solver: Union[str, dict, SolverBase]

    # properties
    # TODO

    def __init__(
        self,
        scalarization_model_params,  # TODO: Define type
        callbacks: dict[str, Callable],
        solver: Union[str, dict],
    ):
        # Implementations of class should also manage the concatenating the additional variables and separating them
        self.scalarization_model_params = scalarization_model_params
        self.callbacks = callbacks
        self.solver = solver
        self._obj_times = []
        self._deriv_times = []

    def __repr__(self) -> str:
        return f"Scalarization Strategy {self.name} ({self.short_name})"

    def _initialize(self):
        self.solver = get_solver(self.solver)

    @abstractmethod
    def variable_upper_bounds() -> np.ndarray[float]:
        pass

    @abstractmethod
    def variable_lower_bounds() -> np.ndarray[float]:
        pass

    @abstractmethod
    def get_linear_constraints(self):  # -> dict[Index, LinearConstraint]:
        pass

    @abstractmethod
    def get_nonlinear_constraints(self):  # -> dict[Index, NonlinearConstraint]:
        pass

    @abstractmethod
    def evaluate_objective(x: np.ndarray) -> float:
        print("This is not the same as evaluating objectives. E.g. make weighted sum")

    @abstractmethod
    def evaluate_constraints(x: np.ndarray) -> np.ndarray:
        print(
            "Most of the time this will be the objective constraints and the constraints from the scalarization method"
        )

    # def solve(self, x: np.ndarray[float]) -> np.ndarray[float]:
    #     print("Do something")
    #     self._call_solver_interface(self.solver, self.solver_params)

    def is_objective_convex() -> bool:
        pass

    def solve(self, x: np.ndarray[float]) -> np.ndarray[float]:
        self._initialize()
        self._obj_times = []
        self._deriv_times = []

        return self._solve(x)

    @abstractmethod
    def _solve(self, x: np.ndarray[float]) -> np.ndarray[float]:
        pass

    def _call_solver_interface(solver: str, params: dict) -> np.ndarray[float]:
        pass
