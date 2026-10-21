r"""Logistic-regression NTCP objective on dose and geometry features.

The model is fully described by its coefficients: ``p = sigmoid(w . x + b)`` with
``dp/dx = p (1 - p) w``. Only loading depends on where the model comes from; the
optimization itself needs neither scikit-learn nor torch.

Example
-------
>>> ntcp_obj = OutcomeLR(
...     priority=100.0,
...     model=r"...\\Head-and-neck-linear-lr",
...     transform=True,
...     ct=ct,
...     cst=cst,
... )
"""

from __future__ import annotations

from typing import Any, ClassVar, Literal, Optional

import array_api_compat
import numpy as np
from pydantic import Field, PrivateAttr, model_validator

from ....ai.modelhub import load_pyanno4rt_model
from .._objective import Objective
from ._features import FeatureExtractor

_PROBABILITY_EPS = 1e-6


def _logistic_coefficients(estimator: Any) -> tuple[np.ndarray, float]:
    """Return ``(w, b)`` of a binary scikit-learn ``LogisticRegression``."""
    coef = getattr(estimator, "coef_", None)
    if type(estimator).__name__ != "LogisticRegression" or coef is None or coef.shape[0] != 1:
        raise TypeError(
            f"OutcomeLR needs a binary LogisticRegression, got {type(estimator).__name__}."
        )
    return np.asarray(coef[0], dtype=np.float64), float(estimator.intercept_[0])


class OutcomeLR(Objective):
    """Logistic-regression NTCP objective.

    The features see the dose of the VOI the objective is attached to, with 0 Gy
    everywhere else. Attach it to a HELPER VOI covering every segment the model
    uses, with ``apply_overlap=None``.

    Parameters
    ----------
    model : str or tuple
        pyanno4rt snapshot folder (holding ``input.json``), or a preloaded
        ``(estimator, spec)`` tuple.
    component : str, optional
        Outcome component to use when the configuration holds more than one.
    monitor_only : bool
        When *True* the objective returns 0 (and a zero gradient) to the
        optimizer but still records the predicted probability.
    transform : bool
        Apply pyanno4rt's SALU transform: above ``p = 0.5`` the value becomes
        ``0.25 * logit(p) + 0.5``, which keeps the gradient from vanishing.
    segment_map : dict[str, str], optional
        Mapping from the model's segment names to VOI names. Names are otherwise
        matched case-insensitively.
    num_of_fractions : int, optional
        The model expects total-course dose, so the per-fraction dose of the
        optimizer is multiplied by this. Defaults to the pyanno4rt configuration.
    """

    name: Literal["Outcome LR"] = "Outcome LR"
    has_hessian: ClassVar[bool] = False

    model: str = Field(description="pyanno4rt snapshot folder, or preloaded objects.")
    component: Optional[str] = Field(default=None)
    monitor_only: bool = Field(default=False)
    transform: bool = Field(default=False)
    segment_map: Optional[dict[str, str]] = Field(default=None)
    num_of_fractions: Optional[int] = Field(default=None, gt=0)
    ntcp_history: list[float] = Field(default_factory=list, exclude=True)

    coef: Any = Field(default=None, exclude=True)
    intercept: float = Field(default=0.0, exclude=True)
    extractor: Any = Field(default=None, exclude=True)

    # VOI voxel indices on the dose grid; None when the VOI covers the whole grid.
    _dose_index: Any = PrivateAttr(default=None)
    _dose_n: Optional[int] = PrivateAttr(default=None)

    model_config = {"arbitrary_types_allowed": True}

    @model_validator(mode="before")
    @classmethod
    def _init(cls, data: Any) -> Any:
        """Read the coefficients and set up the feature extractor."""
        if not isinstance(data, dict) or data.get("extractor") is not None:
            return data

        ct = data.pop("ct", None)
        cst = data.pop("cst", None)
        model = data.get("model")
        if not isinstance(model, str):
            data["model"] = "<preloaded>"
        if ct is None or cst is None:
            return data

        if isinstance(model, str):
            estimator, spec = load_pyanno4rt_model(model, component=data.get("component"))
        else:
            estimator, spec = model
        if spec.outcome_type != "NTCP":
            raise ValueError(f"OutcomeLR minimises NTCP, got a {spec.outcome_type} model.")

        data["coef"], data["intercept"] = _logistic_coefficients(estimator)
        data.setdefault("num_of_fractions", spec.num_of_fractions)
        data["extractor"] = FeatureExtractor(
            spec.features, ct, cst, segment_map=data.get("segment_map")
        )
        return data

    def __deepcopy__(self, memo=None):
        """Share the extractor: it holds the resampled masks and never changes."""
        return self.model_copy(update={"ntcp_history": list(self.ntcp_history)})

    @property
    def feature_names(self) -> list[str]:
        """list[str]: Model inputs, in the order of the coefficients."""
        self._check_ready()
        return [feature.name for feature in self.extractor.features]

    def preprocess_image_reference_parameters(self, target_grid, index_list=None) -> None:
        """Resample the segments onto the dose grid and check the VOI covers them."""
        super().preprocess_image_reference_parameters(target_grid, index_list)
        if self.extractor is None:
            return
        self.extractor.set_dose_grid(target_grid)
        self._dose_n = int(target_grid.num_voxels)
        self._dose_index = None if index_list is None else np.asarray(index_list)
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

    def compute_objective(self, dose_values):
        """Compute the (transformed) NTCP; 0 when *monitor_only*."""
        self._check_ready()
        xp = array_api_compat.array_namespace(dose_values)
        probability = self._probability(self._features(dose_values, xp))
        self.ntcp_history.append(probability)

        value = 0.0 if self.monitor_only else self._transform(probability)[0]
        return xp.asarray(
            value, dtype=dose_values.dtype, device=array_api_compat.device(dose_values)
        )

    def compute_gradient(self, dose_values):
        """Compute the gradient w.r.t. the dose per fraction."""
        self._check_ready()
        xp = array_api_compat.array_namespace(dose_values)
        if self.monitor_only:
            return xp.zeros_like(dose_values)

        dose = self._full_total_dose(dose_values, xp)
        probability = self._probability(self.extractor.compute_values(dose, xp))
        slope = self._transform(probability)[1]
        # d value/dp * dp/dx * d total dose / d dose per fraction
        scale = slope * probability * (1.0 - probability) * self.coef * self.num_of_fractions

        grad = xp.zeros_like(dose)
        for position, indices, feature_grad in self.extractor.compute_dose_gradients(dose, xp):
            grad[indices] += float(scale[position]) * feature_grad
        if self._dose_index is None:
            return grad
        return grad[xp.asarray(self._dose_index, device=array_api_compat.device(dose_values))]

    def report_info(self) -> dict:
        """Return the NTCP history collected during optimization."""
        return {"ntcp_history": list(self.ntcp_history)} if self.ntcp_history else {}

    def reset_history(self) -> None:
        """Clear the recorded NTCP history."""
        self.ntcp_history.clear()

    def _check_ready(self) -> None:
        if self.extractor is None:
            raise RuntimeError("OutcomeLR not initialised -- provide ct and cst at construction.")

    def _full_total_dose(self, dose_values, xp):
        """Scatter the VOI dose onto the full grid (0 elsewhere) and scale to total dose."""
        if self._dose_index is not None:
            device = array_api_compat.device(dose_values)
            full = xp.zeros(self._dose_n, dtype=dose_values.dtype, device=device)
            full[xp.asarray(self._dose_index, device=device)] = dose_values
            dose_values = full
        return dose_values * self.num_of_fractions

    def _features(self, dose_values, xp) -> np.ndarray:
        return self.extractor.compute_values(self._full_total_dose(dose_values, xp), xp)

    def _probability(self, features: np.ndarray) -> float:
        p = 1.0 / (1.0 + np.exp(-(float(features @ self.coef) + self.intercept)))
        return min(max(p, _PROBABILITY_EPS), 1.0 - _PROBABILITY_EPS)

    def _transform(self, p: float) -> tuple[float, float]:
        """Return the objective value and its derivative w.r.t. ``p``."""
        if not self.transform or p <= 0.5:
            return p, 1.0
        return 0.25 * float(np.log(p / (1.0 - p))) + 0.5, 0.25 / (p - p**2)
