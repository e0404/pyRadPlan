"""Base class for outcome objectives on dose and geometry features."""

from abc import abstractmethod
from typing import Any, Optional

import array_api_compat
import numpy as np
from pydantic import Field, PrivateAttr, model_validator

from pyRadPlan.ai.modelhub import load_pyanno4rt_model

from ._features import FeatureExtractor
from ._outcome import OutcomeObjective

_PROBABILITY_EPS = 1e-6


class FeatureOutcomeObjective(OutcomeObjective):
    """Outcome objective whose model consumes scalar features of dose and segments.

    The features see the dose of the VOI the objective is attached to, with 0 Gy
    everywhere else. Attach it to a helper VOI covering every segment the model uses,
    with ``apply_overlap=None``.

    Subclasses implement :meth:`_probability` and :meth:`_probability_gradient` on the
    feature vector.

    Parameters
    ----------
    model : str or tuple
        pyanno4rt snapshot folder (holding ``input.json``), or a preloaded
        ``(estimator, spec)`` tuple, see :func:`~pyRadPlan.ai.modelhub.load_pyanno4rt_model`.
    component : str, optional
        Outcome component to use when the configuration holds more than one.
    segment_map : dict[str, str], optional
        Mapping from the model's segment names to VOI names. Names are otherwise
        matched case-insensitively.

    Notes
    -----
    ``num_of_fractions`` defaults to the model's configuration, as pyanno4rt models
    expect total-course dose.
    """

    model: str = Field(description="pyanno4rt snapshot folder, or preloaded objects.")
    component: Optional[str] = Field(default=None)
    segment_map: Optional[dict[str, str]] = Field(default=None)

    estimator: Any = Field(default=None, exclude=True)
    spec: Any = Field(default=None, exclude=True)

    _extractor: Optional[FeatureExtractor] = PrivateAttr(default=None)

    model_config = {"arbitrary_types_allowed": True}

    @model_validator(mode="before")
    @classmethod
    def _load(cls, data: Any) -> Any:
        """Load the estimator and its feature specification."""
        if not isinstance(data, dict) or data.get("spec") is not None:
            return data

        model = data.get("model")
        if isinstance(model, str):
            estimator, spec = load_pyanno4rt_model(model, component=data.get("component"))
        else:
            estimator, spec = model
            data["model"] = "<preloaded>"
        if spec.outcome_type != "NTCP":
            raise ValueError(f"{cls.__name__} minimises NTCP, got a {spec.outcome_type} model.")

        cls._check_estimator(estimator, data)
        data["estimator"], data["spec"] = estimator, spec
        data.setdefault("num_of_fractions", spec.num_of_fractions)
        return data

    @classmethod
    def _check_estimator(cls, estimator: Any, data: dict) -> None:
        """Validate the estimator; may add derived fields to ``data``."""

    @property
    def feature_names(self) -> list[str]:
        """Model inputs, in the order the estimator expects them."""
        return [feature.name for feature in self.spec.features]

    @property
    def extractor(self) -> FeatureExtractor:
        """Feature extractor; available after :meth:`setup`."""
        if self._extractor is None:
            raise RuntimeError(
                f"{self.name} is not set up: the planning problem calls setup(ct, cst) "
                "before optimization."
            )
        return self._extractor

    def setup(self, ct, cst) -> None:
        """Resolve the segments and compute the geometry features on the CT grid."""
        self._extractor = FeatureExtractor(
            self.spec.features, ct, cst, segment_map=self.segment_map
        )

    def _set_dose_grid(self, dose_grid) -> None:
        """Resample the segments onto the dose grid and check the VOI covers them."""
        self.extractor.set_dose_grid(dose_grid)
        if self._dose_index is None:
            return

        covered = np.zeros(self._dose_n, dtype=bool)
        covered[self._dose_index] = True
        for segment, indices in self.extractor.segment_indices.items():
            missing = int(np.count_nonzero(~covered[indices]))
            if missing:
                raise ValueError(
                    f"The VOI of this objective misses {missing} of {indices.size} voxels of "
                    f"segment '{segment}', whose dose would be taken as 0. Attach it to a "
                    "helper VOI covering all segments, with apply_overlap=None."
                )

    def _predict(self, dose) -> float:
        xp = array_api_compat.array_namespace(dose)
        return self._clipped_probability(self.extractor.compute_values(dose, xp))

    def _predict_with_gradient(self, dose) -> tuple[float, Any]:
        xp = array_api_compat.array_namespace(dose)
        features = self.extractor.compute_values(dose, xp)
        p = self._clipped_probability(features)
        dp_dx = self._probability_gradient(features, p)

        grad = xp.zeros_like(dose)
        for position, indices, feature_grad in self.extractor.compute_dose_gradients(dose, xp):
            grad[indices] += float(dp_dx[position]) * feature_grad
        return p, grad

    def _clipped_probability(self, features: np.ndarray) -> float:
        p = self._probability(features)
        return min(max(p, _PROBABILITY_EPS), 1.0 - _PROBABILITY_EPS)

    @abstractmethod
    def _probability(self, features: np.ndarray) -> float:
        """Predicted probability for a feature vector."""

    @abstractmethod
    def _probability_gradient(self, features: np.ndarray, p: float) -> np.ndarray:
        """Gradient of the probability w.r.t. the features, given ``p`` at ``features``."""
