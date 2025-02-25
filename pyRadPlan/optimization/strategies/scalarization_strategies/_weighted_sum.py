from typing import Union
import numpy as np
from ._base_scalarization_strategies import ScalarizationStrategyBase
class WeightedSum(ScalarizationStrategyBase):
    name = "Weighted sum scalarization strategy"
    short_name = "weighted_sum"

    def __init__(self,
                 callbacks,
                 weights: Union[np.ndarray[float],list[float],None] = None):
        #functions in the planning problem that are required for the actual optimization
        super().__init__(callbacks)
        self.weights = np.asarray(weights)

    def variable_lower_bounds(self):
        return self.callbacks['get_variable_bounds']()[0]
    def variable_upper_bounds(self):
        return self.callbacks['get_variable_bounds']()[1]

    def get_linear_constraints(self):
        pass

    def get_nonlinear_constraints(self):
        pass

    def evaluate_objective(self,x):
        return self.weights @ self.callbacks['evaluate_objectives'](x)

    def evaluate_constraints(self,x):
        return self.callbacks['evaluate_constraints'](x)
    
    def _solve(self,x: np.ndarray[float]) -> list[np.ndarray[float]]:
        print('Hello World!')
        return x
