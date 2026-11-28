"""Logistic-regression outcome objective."""

from typing import Any, Literal

import numpy as np
from pydantic import Field

from ._feature_outcome import FeatureOutcomeObjective


class OutcomeLR(FeatureOutcomeObjective):
    """Logistic-regression outcome objective on dose and geometry features.

    The model is fully described by its coefficients, ``p = sigmoid(w . x + b)`` with
    ``dp/dx = p (1 - p) w``, so evaluation needs no scikit-learn.
    """

    name: Literal["Outcome LR"] = "Outcome LR"

    coef: Any = Field(default=None, exclude=True)
    intercept: float = Field(default=0.0, exclude=True)

    @classmethod
    def _check_estimator(cls, estimator: Any, data: dict) -> None:
        coef = getattr(estimator, "coef_", None)
        if type(estimator).__name__ != "LogisticRegression" or coef is None or coef.shape[0] != 1:
            raise ValueError(
                f"OutcomeLR needs a binary LogisticRegression, got {type(estimator).__name__}."
            )
        data["coef"] = np.asarray(coef[0], dtype=np.float64)
        data["intercept"] = float(estimator.intercept_[0])

    def _probability(self, features: np.ndarray) -> float:
        return float(1.0 / (1.0 + np.exp(-(float(features @ self.coef) + self.intercept))))

    def _probability_gradient(self, features: np.ndarray, p: float) -> np.ndarray:
        return p * (1.0 - p) * self.coef
