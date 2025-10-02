from typing import Union
import numpy as np
from numpy.typing import ArrayLike

from ...plan import Plan
from ._optiprob import NonLinearPlanningProblem
from ..solvers import NonLinearOptimizer


class SimpleLeastSquaresFluenceOptimization(NonLinearPlanningProblem):
    """Simple least squares fluence-based planning problem."""

    penalties: ArrayLike
    result: dict[str]

    def __init__(self, pln: Union[Plan, dict] = None):
        self.penalties = np.asarray([1000, 1])
        self.target_prescription = 2.0
        self._target_voxels = None
        self._patient_voxels = None

        self._grad_cache_intermediate = None
        self._grad_cache = None
        self._result_cache = None
        self._w_cache = None

        super().__init__(pln)

    def _initialize(self):
        super()._initialize()
        self._target_voxels = self._cst.target_union_voxels(order="numpy")
        self._patient_voxels = self._cst.patient_voxels(order="numpy")

        # Check if the solver is adequate to solve this problem
        # TODO: check that it can do constraints
        if not isinstance(self.solver, NonLinearOptimizer):
            raise ValueError("Solver must be an instance of SolverBase")

        self.solver.objective = self._objective_function
        self.solver.gradient = self._objective_gradient
        self.solver.bounds = (0.0, np.inf)
        self.solver.max_iter = 500
        self.solver.options = {
            "disp": True,
            "ftol": 1e-4,
        }

    def _evaluate_objective_functions(self, x: np.ndarray) -> np.ndarray:
        """Define the objective functions."""

        if not np.array_equal(x, self._w_cache):
            self._result_cache = self._dij.get_result_arrays_from_intensity(x)
            self._w_cache = x.copy()

        dose = self._result_cache["physical_dose"]
        target_fun = (
            np.sum((dose[self._target_voxels] - self.target_prescription) ** 2)
            / self._target_voxels.size
        )
        patient_fun = np.sum((dose[self._patient_voxels]) ** 2) / self._patient_voxels.size
        return np.array([target_fun, patient_fun])

    def _objective_function(self, x: np.ndarray) -> np.float64:
        return np.dot(np.asarray(self.penalties), self._evaluate_objective_functions(x))

    def _evaluate_objective_jacobian(self, x: np.ndarray) -> np.ndarray:
        """Define the objective jacobian."""

        if not np.array_equal(x, self._w_cache):
            self._result_cache = self._dij.get_result_arrays_from_intensity(x)
            self._w_cache = x.copy()

        dose = self._result_cache["physical_dose"]

        if self._grad_cache_intermediate is None:
            self._grad_cache_intermediate = np.zeros((dose.size, 2))

        # We do no sanity checks here, i.e., the grids need to be the same

        self._grad_cache_intermediate[self._target_voxels, 0] = (
            2.0 / self._target_voxels.size * (dose[self._target_voxels] - self.target_prescription)
        )
        self._grad_cache_intermediate[self._patient_voxels, 1] = (
            2.0 / self._patient_voxels.size * (dose[self._patient_voxels])
        )

        self._grad_cache = self._dij.physical_dose.flat[0].T @ self._grad_cache_intermediate

        return self._grad_cache

    def _objective_gradient(self, x: np.ndarray) -> np.ndarray:
        return np.sum(self._evaluate_objective_jacobian(x) * self.penalties, axis=1)

    def _evaluate_objective_hessian(self, x: np.ndarray) -> np.ndarray:
        """Define the objective hessian."""
        return {}

    def _evaluate_constraint_functions(self, x: np.ndarray) -> np.ndarray:
        """Define the constraint functions."""
        return None

    def _evaluate_constraint_jacobian(self, x: np.ndarray) -> np.ndarray:
        """Define the constraint jacobian."""
        return None

    def _evaluate_constraint_jacobian_structure(self) -> np.ndarray:
        """Define the constraint jacobian structure."""
        return None

    def _get_variable_bounds(self, x: np.ndarray) -> np.ndarray:
        """Define the variable bounds."""
        return {}

    def _solve(self) -> tuple[np.ndarray, dict]:
        """Solve the problem."""

        x0 = np.zeros((self._dij.total_num_of_bixels,), dtype=np.float64)
        return self.solver.solve(x0)
