r"""Feature-based outcome objective (NTCP / TCP) backed by a scikit-learn model.

Example
-------
>>> ntcp_obj = OutcomeSklearn(
...     priority=1000.0,
...     model=r"...\\Head-and-neck-linear-lr\\input.json",
...     mode="NTCP",
...     transform=True,
...     ct=ct,
...     cst=cst,
...     segment_map={"PAROTID_LT": "Parotid_LT", "PAROTID_RT": "Parotid_RT"},
... )
"""

from __future__ import annotations

import copy
import logging
from typing import Any, ClassVar, Literal, Optional

import array_api_compat
import numpy as np
from pydantic import Field, PrivateAttr, model_validator
from pydantic_core import PydanticUndefined

from ....ai_models import load_pyanno4rt_model
from .._objective import Objective
from ._features import FeatureExtractor

logger = logging.getLogger(__name__)

#: Probabilities are clipped to this margin before the transform / gradient rescale.
_PROBABILITY_EPS = 1e-6


class OutcomeSklearn(Objective):
    """Feature-based outcome objective (NTCP or TCP).

    Parameters
    ----------
    model : str
        Path to a pyanno4rt ``input.json`` (or its directory), or a preloaded
        ``(estimator, spec)`` tuple used directly.
    component : str, optional
        Name, identifier or label of the outcome component to use. Required
        only when the configuration holds more than one.
    mode : ``"NTCP"`` or ``"TCP"``
        ``"NTCP"`` minimises the predicted probability, ``"TCP"`` maximises it.
    monitor_only : bool
        When *True* the objective returns 0 (and a zero gradient) to the
        optimizer but still records the predicted probability.
    transform : bool
        Apply pyanno4rt's SALU transform to the objective value: above
        ``p = 0.5`` the probability is replaced by ``0.25 * logit(p) + 0.5``,
        which keeps the gradient from vanishing once the probability saturates.
    segment_map : dict[str, str], optional
        Mapping from the model's segment names to VOI names in the structure
        set. Names are otherwise matched case-insensitively.
    static_features : dict[str, float], optional
        Overrides for the dose-independent (geometry) features, keyed by
        feature name.
    fd_step : float
        Relative step of the finite-difference fallback used for estimators
        without an analytic feature gradient.
    """

    # -- Objective metadata --
    name: Literal["Outcome Sklearn"] = "Outcome Sklearn"
    has_hessian: ClassVar[bool] = False

    # -- User-facing fields --
    model: str = Field(description="Path to a pyanno4rt input.json, or preloaded objects.")
    component: Optional[str] = Field(default=None)
    mode: Literal["NTCP", "TCP"] = Field(default="NTCP")
    monitor_only: bool = Field(default=False)
    transform: bool = Field(default=False)
    segment_map: Optional[dict[str, str]] = Field(default=None)
    static_features: Optional[dict[str, float]] = Field(default=None)
    fd_step: float = Field(default=1e-3, gt=0.0)
    ntcp_history: list[float] = Field(default_factory=list)

    # -- Internal state (excluded from serialisation) --
    estimator: Any = Field(default=None, exclude=True)
    spec: Any = Field(default=None, exclude=True)
    extractor: Any = Field(default=None, exclude=True)

    # Mapping of the (VOI-restricted) optimization vector onto the full dose grid.
    _dose_index: Any = PrivateAttr(default=None)
    _dose_n: Optional[int] = PrivateAttr(default=None)
    _dose_index_xp: Any = PrivateAttr(default=None)
    _dose_index_key: Any = PrivateAttr(default=None)

    model_config = {"arbitrary_types_allowed": True}

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    @model_validator(mode="before")
    @classmethod
    def _init(cls, data: Any) -> Any:
        """Load the estimator and configure the feature extractor with CT / structures."""
        if not isinstance(data, dict):
            return data

        # Already initialised (copy / re-validation).
        if data.get("extractor") is not None:
            return data

        ct = data.pop("ct", None)
        cst = data.pop("cst", None)
        data.pop("stf", None)  # accepted for convenience, not used
        data.pop("pln", None)  # the dose grid is resolved via the optimizer hook

        spec_input = data.get("model")
        if not isinstance(spec_input, str):
            # Preloaded objects -- keep the serialised field a plain string. This has to
            # happen even without planning data, or validation trips over the tuple.
            data["model"] = "<preloaded>"

        # Without planning data the extractor cannot be built (e.g. on deserialisation).
        # Leave the objective uninitialised; _check_ready() guards its use.
        if ct is None or cst is None:
            return data

        estimator, spec = cls._resolve_model(spec_input, data.get("component"))

        data.setdefault("mode", spec.outcome_type)
        data["estimator"] = estimator
        data["spec"] = spec
        data["extractor"] = FeatureExtractor(
            spec.features,
            ct,
            cst,
            segment_map=data.get("segment_map"),
            static_features=data.get("static_features"),
        )
        return data

    @staticmethod
    def _resolve_model(spec: Any, component: Optional[str]) -> tuple[Any, Any]:
        """Resolve ``model`` into an ``(estimator, spec)`` pair."""
        if isinstance(spec, (tuple, list)) and len(spec) == 2:
            return spec[0], spec[1]
        if isinstance(spec, str):
            return load_pyanno4rt_model(spec, component=component)
        raise TypeError(
            "OutcomeSklearn 'model' must be a path to a pyanno4rt input.json or a preloaded "
            "(estimator, spec) tuple."
        )

    def __deepcopy__(self, memo=None):
        """Deep-copy without duplicating the estimator and the feature extractor.

        The optimizer deep-copies the whole ``cst`` (e.g. in
        ``apply_overlap_priorities``). Estimator and extractor are read-only
        runtime handles holding the resampled masks, so the copy shares them;
        transient index caches are reset.
        """
        if memo is None:
            memo = {}
        new = self.__class__.__new__(self.__class__)
        memo[id(self)] = new

        shared = {"estimator", "spec", "extractor"}
        new_dict = {
            k: (v if k in shared else copy.deepcopy(v, memo)) for k, v in self.__dict__.items()
        }
        object.__setattr__(new, "__dict__", new_dict)
        object.__setattr__(new, "__pydantic_fields_set__", set(self.__pydantic_fields_set__))
        object.__setattr__(
            new,
            "__pydantic_extra__",
            None
            if self.__pydantic_extra__ is None
            else copy.deepcopy(self.__pydantic_extra__, memo),
        )
        if self.__pydantic_private__ is None:
            object.__setattr__(new, "__pydantic_private__", None)
        else:
            reset = {"_dose_index_xp", "_dose_index_key"}
            private = {
                k: (None if k in reset else copy.deepcopy(v, memo))
                for k, v in self.__pydantic_private__.items()
                if v is not PydanticUndefined
            }
            object.__setattr__(new, "__pydantic_private__", private)
        return new

    # ------------------------------------------------------------------
    # Objective interface
    # ------------------------------------------------------------------

    def preprocess_image_reference_parameters(self, target_grid, index_list=None) -> None:
        """Wire the optimization dose grid into the feature extractor.

        Called once by the planning problem for every objective with
        ``target_grid=dij.dose_grid``; this is where the objective learns the
        grid its dose vector lives on.
        """
        super().preprocess_image_reference_parameters(target_grid, index_list)
        if self.extractor is not None:
            self.extractor.set_dose_grid(target_grid)
            self._dose_n = int(target_grid.num_voxels)
            # The optimizer hands us only the voxels inside this objective's VOI
            # (index_list); remember them so we can scatter onto the full dose grid
            # the features are computed on, and gather the gradient back.
            self._dose_index = None if index_list is None else np.asarray(index_list)
            self._dose_index_xp = None
            self._dose_index_key = None
            self._check_segment_coverage()

    def _check_segment_coverage(self) -> None:
        """Reject a VOI that does not cover every segment the features read.

        Dose outside the objective's VOI is zero-filled by :meth:`_to_full_grid`, so an
        uncovered segment would contribute a zero dose feature instead of failing.
        """
        if self._dose_index is None:
            return  # objective sees the full grid, which covers every segment

        covered = np.zeros(self._dose_n, dtype=bool)
        covered[self._dose_index] = True
        for segment, indices in self.extractor.segment_indices.items():
            missing = int(np.count_nonzero(~covered[indices]))
            if missing:
                raise ValueError(
                    f"The VOI this objective is attached to misses {missing} of "
                    f"{indices.size} dose-grid voxels of segment '{segment}'. Dose "
                    "outside the VOI is treated as zero, so that segment's features "
                    "would be silently wrong. Attach the objective to a VOI covering "
                    "every segment the model uses (e.g. a helper VOI spanning their "
                    "union, with apply_overlap=None)."
                )

    def compute_objective(self, dose_values) -> float:
        """Compute the outcome probability (0.0 when *monitor_only*)."""
        self._check_ready()

        xp = array_api_compat.array_namespace(dose_values)
        probability = self._predict(self._to_full_grid(dose_values, xp), xp)
        self.ntcp_history.append(probability)

        if self.monitor_only:
            return 0.0

        value = self._transform_value(probability)
        return -value if self.mode == "TCP" else value

    def compute_gradient(self, dose_values):
        """Compute the gradient of the outcome probability w.r.t. dose.

        Returns a 1-D array matching ``dose_values`` in the same array namespace
        and device as the input.
        """
        self._check_ready()

        xp = array_api_compat.array_namespace(dose_values)
        n = array_api_compat.size(dose_values)

        if self.monitor_only:
            return xp.zeros(n, dtype=dose_values.dtype)

        dose = self._to_full_grid(dose_values, xp)
        features = self.extractor.compute_values(dose, xp)
        probability = self._clip(self._predict_from_features(features))
        scale = self._transform_scale(probability) * self._feature_gradient(features, probability)

        grad = xp.zeros(
            array_api_compat.size(dose),
            dtype=dose_values.dtype,
            device=array_api_compat.device(dose_values),
        )
        for position, indices, feature_gradient in self.extractor.compute_dose_gradients(dose, xp):
            grad[indices] += float(scale[position]) * feature_gradient

        if self.mode == "TCP":
            grad = -grad

        return self._from_full_grid(grad, dose_values, xp)

    def report_info(self) -> dict:
        """Return the NTCP history collected during optimization."""
        if self.ntcp_history:
            return {"ntcp_history": list(self.ntcp_history)}
        return {}

    def reset_history(self) -> None:
        """Clear the recorded probability history."""
        self.ntcp_history.clear()

    def get_ntcp_history(self) -> list[float]:
        """Return a copy of the probability history."""
        return list(self.ntcp_history)

    @property
    def feature_names(self) -> list[str]:
        """list[str]: Names of the model inputs, in the order the estimator expects them."""
        self._check_ready()
        return [feature.name for feature in self.spec.features]

    # ------------------------------------------------------------------
    # Model evaluation
    # ------------------------------------------------------------------

    def _predict(self, dose, xp) -> float:
        """Evaluate the model on a full-grid dose vector."""
        return self._clip(self._predict_from_features(self.extractor.compute_values(dose, xp)))

    def _predict_from_features(self, features: np.ndarray) -> float:
        """Predict the outcome probability from a ``(1, n_features)`` input vector."""
        if hasattr(self.estimator, "predict_proba"):
            probabilities = self.estimator.predict_proba(features)
            return float(probabilities[0, -1])
        return float(np.ravel(self.estimator.predict(features))[0])

    @staticmethod
    def _clip(probability: float) -> float:
        return min(max(probability, _PROBABILITY_EPS), 1.0 - _PROBABILITY_EPS)

    def _feature_gradient(self, features: np.ndarray, probability: float) -> np.ndarray:
        """Return ``dp / dfeature`` for all features."""
        coefficients = getattr(self.estimator, "coef_", None)
        if (
            type(self.estimator).__name__ == "LogisticRegression"
            and coefficients is not None
            and coefficients.shape[0] == 1
        ):
            return probability * (1.0 - probability) * coefficients[0]

        # Generic estimators: central differences, but only along the features that
        # actually move with the dose (typically two out of eight).
        gradient = np.zeros(features.shape[1], dtype=np.float64)
        for position in self.extractor.dose_feature_indices:
            step = self.fd_step * max(abs(float(features[0, position])), 1.0)
            forward = features.copy()
            forward[0, position] += step
            backward = features.copy()
            backward[0, position] -= step
            gradient[position] = (
                self._predict_from_features(forward) - self._predict_from_features(backward)
            ) / (2.0 * step)
        return gradient

    def _transform_value(self, probability: float) -> float:
        """Map the probability to the objective value (see ``transform``)."""
        if not self.transform or probability <= 0.5:
            return probability
        clipped = self._clip(probability)
        return 0.25 * float(np.log(clipped / (1.0 - clipped))) + 0.5

    def _transform_scale(self, probability: float) -> float:
        """Differentiate :meth:`_transform_value` w.r.t. the probability."""
        if not self.transform or probability <= 0.5:
            return 1.0
        return 0.25 / (probability - probability**2)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_ready(self) -> None:
        if self.extractor is None or self.estimator is None:
            raise RuntimeError(
                "OutcomeSklearn not initialised -- provide ct and cst at construction."
            )

    def _needs_scatter(self, dose_values) -> bool:
        """Whether ``dose_values`` is a VOI subset that must be scattered to the full grid."""
        return (
            self._dose_index is not None
            and self._dose_n is not None
            and array_api_compat.size(dose_values) != self._dose_n
        )

    def _index_on(self, dose_values, xp):
        """VOI indices in the namespace/device of ``dose_values`` (cached)."""
        device = array_api_compat.device(dose_values)
        key = (getattr(xp, "__name__", str(xp)), str(device))
        if self._dose_index_key != key:
            self._dose_index_xp = xp.asarray(self._dose_index, device=device)
            self._dose_index_key = key
        return self._dose_index_xp

    def _to_full_grid(self, dose_values, xp):
        """Scatter VOI-restricted dose onto the full dose grid (zeros elsewhere)."""
        if not self._needs_scatter(dose_values):
            return dose_values
        full = xp.zeros(
            self._dose_n, dtype=dose_values.dtype, device=array_api_compat.device(dose_values)
        )
        full[self._index_on(dose_values, xp)] = dose_values
        return full

    def _from_full_grid(self, grad_full, dose_values, xp):
        """Gather a full-grid gradient back to the objective's VOI voxels."""
        if not self._needs_scatter(dose_values):
            return grad_full
        return grad_full[self._index_on(dose_values, xp)]
