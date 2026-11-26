"""Outcome objective for scikit-learn classifiers on dose and geometry features."""

from typing import Any, Literal

import numpy as np

from ._feature_outcome import FeatureOutcomeObjective


class OutcomeSklearn(FeatureOutcomeObjective):
    """Outcome objective for any binary scikit-learn classifier with ``predict_proba``.

    The feature gradient is taken by central finite differences, which needs
    ``2 * n_features`` predictions per gradient. Use it for smooth models (e.g. MLP,
    calibrated SVM); tree ensembles are piecewise constant and give no useful
    gradient. For logistic regression prefer :class:`OutcomeLR`.
    """

    name: Literal["Outcome scikit-learn"] = "Outcome scikit-learn"

    @classmethod
    def _check_estimator(cls, estimator: Any, data: dict) -> None:
        classes = getattr(estimator, "classes_", None)
        if not hasattr(estimator, "predict_proba") or classes is None or len(classes) != 2:
            raise ValueError(
                "OutcomeSklearn needs a binary classifier with predict_proba, got "
                f"{type(estimator).__name__}."
            )

    def _probability(self, features: np.ndarray) -> float:
        return float(self._predict_proba(features[np.newaxis, :])[0])

    def _probability_gradient(self, features: np.ndarray, p: float) -> np.ndarray:
        steps = 1e-4 * np.maximum(np.abs(features), 1.0)
        shifts = np.diag(steps)
        probed = self._predict_proba(np.concatenate([features + shifts, features - shifts]))
        n = features.size
        return (probed[:n] - probed[n:]) / (2.0 * steps)

    def _predict_proba(self, rows: np.ndarray) -> np.ndarray:
        return self.estimator.predict_proba(rows)[:, 1]
