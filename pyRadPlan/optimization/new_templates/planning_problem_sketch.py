import numpy as np
from enum import Enum


class PlanningProblem:
    def __init__(
        self,
        scalarization_method: str | Enum,
        tradeoff_exploration_method: str | Enum | None,
        method_parameters: dict,
    ):
        pass

    def evaluate_objectives(x: np.ndarray[float]) -> list[np.ndarray[float]]:
        pass

    def evaluate_objective_gradients(x: np.ndarray[float]) -> list[np.ndarray[float]]:
        # returns list of 2d arrays
        pass

    def evaluate_objective_hessian(x: np.ndarray[float]) -> list[np.ndarray[float]]:
        # returns list of 3d arrays
        pass

    def evaluate_constraints(x: np.ndarray[float]) -> list[np.ndarray[float]]:
        pass

    def evaluate_constraints_gradients(x: np.ndarray[float]) -> list[np.ndarray[float]]:
        # returns list of 2d arrays
        pass

    def evaluate_constraints_hessian(x: np.ndarray[float]) -> list[np.ndarray[float]]:
        # returns list of 3d arrays
        pass

    def are_objectives_convex() -> bool:
        pass

    def are_objectives_linear() -> bool:
        pass

    def solve(y: np.ndarray[float]) -> list[np.ndarray[float]]:
        pass

    def make_tradeoff_exploration_method_instance(
        evaluate_objectives: callable[np.ndarray[float], list[np.ndarray[float]]],
        evaluate_constraints: callable[np.ndarray[float], list[np.ndarray[float]]],
        evaluate_x_gradients,
        etc,
        scalarization_method: str,
        method_parameters: dict,
    ) -> TradeoffExplorationMethod:
        pass
