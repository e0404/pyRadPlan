import numpy as np


class abstract_objective():
    def __init__(self):
        pass

    def evaluate(x: np.ndarray[float]) -> float:
        pass

    def evaluate_gradient(x: np.ndarray[float]) -> np.ndarray[float]:
        pass

    def evaluate_hessian(x: np.ndarray[float]) -> np.ndarray[float]:
        pass

    def is_linear() -> bool:
        pass

    def is_convex() -> bool:
        pass
