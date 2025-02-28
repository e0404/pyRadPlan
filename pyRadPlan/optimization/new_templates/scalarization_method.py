import numpy as np


class ScalarizationMethod:
    def __init__(
        self,
        evaluate_objectives: callable[np.ndarray[float], list[np.ndarray[float]]],
        evaluate_constraints: callable[np.ndarray[float], list[np.ndarray[float]]],
        evaluate_x_gradients,
        etc,
        parameters: dict,
    ):
        # Implementations of class should also manage the concatenating the additional variables and separating them
        pass

    def variable_upper_bounds() -> np.ndarray[float]:
        pass

    def variable_lower_bounds() -> np.ndarray[float]:
        pass

    def get_linear_constrains(self) -> dict[Index, LinearConstraint]:
        pass

    def get_nonlinear_constraints(self) -> dict[Index, NonlinearConstraint]:
        pass

    def evaluate_objective(x: np.ndarray) -> float:
        print("This is not the same as evaluating objectives. E.g. make weighted sum")

    def evaluate_constraints(x: np.ndarray) -> np.ndarray:
        print(
            "Most of the time this will be the objective constraints and the constraints from the scalarization method"
        )

    def solve(self, x: np.ndarray[float]) -> np.ndarray[float]:
        print("Do something")
        self._call_solver_interface(self.solver, self.solver_params)

    def is_objective_convex() -> bool:
        pass

    def _call_solver_interface(solver: str, params: dict) -> np.ndarray[float]:
        pass
