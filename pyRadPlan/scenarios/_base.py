"""
Abstract scenario model.

A scenario model is the pydantic configuration of an uncertainty space (the dimensions with
their marginal parameters and a joint distribution) together with a sampling rule. It produces a
:class:`~pyRadPlan.scenarios.ScenarioSet` lazily through :meth:`ScenarioModel.scenarios`; the
set is what dose engines, quantities and the Dij consume.
"""

from __future__ import annotations

import sys
from abc import ABC, abstractmethod
import warnings
from typing import Any, ClassVar, Optional, Union

import numpy as np
from pydantic import (
    Field,
    PrivateAttr,
    SerializationInfo,
    SerializerFunctionWrapHandler,
    field_serializer,
    model_serializer,
    model_validator,
)

from pyRadPlan.core import PyRadPlanBaseModel

from ._dimensions import (
    CTPhaseUncertainty,
    RangeUncertainty,
    SetupUncertainty,
    UncertaintyDimension,
)
from ._distributions import JointDistribution, create_distribution
from ._legacy import LegacyScenarioInterface
from ._set import MATRAD_AXES, AxisSpec, ScenarioSet

#: Legacy flat parameter names (matRad / previous pyRadPlan) and their nested targets.
_LEGACY_FIELD_MAP: dict[str, tuple[str, str]] = {
    "rangeRelSD": ("range", "rel_sd"),
    "range_rel_sd": ("range", "rel_sd"),
    "rangeAbsSD": ("range", "abs_sd"),
    "range_abs_sd": ("range", "abs_sd"),
    "combineRange": ("range", "combined"),
    "combine_range": ("range", "combined"),
    "shiftSD": ("setup", "sd"),
    "shift_sd": ("setup", "sd"),
}


class ScenarioModel(LegacyScenarioInterface, PyRadPlanBaseModel, ABC):
    """
    Base class of scenario models.

    Subclasses set :attr:`name` / :attr:`short_name` and implement :meth:`_generate`, which
    turns the configured dimensions and distribution into a :class:`ScenarioSet`. The set is
    cached and regenerated whenever the model's configuration changes (also through nested
    dimension objects), so generation is deferred until a consumer asks for it.

    Attributes
    ----------
    setup : SetupUncertainty
        Setup (isocenter shift) uncertainty.
    range : RangeUncertainty
        Range uncertainty (absolute / relative).
    ct_phase : CTPhaseUncertainty
        CT phases used and their probabilities.
    wc_sigma : float
        Worst-case extent in multiples of the standard deviation (gridded models).
    distribution : str or dict or JointDistribution
        Joint distribution of the continuous components; default independent Gaussian.
    """

    name: ClassVar[str]
    short_name: ClassVar[str]
    #: Legacy flat parameter names mapped onto top-level fields; subclasses extend this.
    legacy_aliases: ClassVar[dict[str, str]] = {}

    setup: SetupUncertainty = Field(default_factory=SetupUncertainty)
    range: RangeUncertainty = Field(default_factory=RangeUncertainty)
    ct_phase: CTPhaseUncertainty = Field(default_factory=CTPhaseUncertainty)
    wc_sigma: float = Field(
        default=1.0,
        ge=0,
        description="Worst-case definition in multiples of the standard deviation.",
    )
    distribution: Union[str, dict, JointDistribution] = Field(
        default="independent_gaussian",
        description="Joint distribution of the continuous uncertainty components.",
    )

    _set_cache: Optional[ScenarioSet] = PrivateAttr(default=None)
    _cache_fingerprint: Optional[str] = PrivateAttr(default=None)
    _num_available_ct_phases: Optional[int] = PrivateAttr(default=None)

    # ------------------------------------------------------------------ validation
    @model_validator(mode="before")
    @classmethod
    def _accept_legacy_fields(cls, data: Any) -> Any:
        """Map flat matRad / legacy parameter names onto the nested dimensions."""
        if not isinstance(data, dict):
            return data
        data = dict(data)
        data.pop("model", None)
        for key in ("wcFactor", "wc_factor"):
            if key in data:
                data.setdefault("wc_sigma", data.pop(key))
        for key, target in cls.legacy_aliases.items():
            if key in data:
                data.setdefault(target, data.pop(key))
        for key, (field, attr) in _LEGACY_FIELD_MAP.items():
            if key in data:
                _nested_set(data, field, attr, data.pop(key))
        for key in ("ctScenProb", "ct_scen_prob"):
            if key in data:
                pairs = data.pop(key)
                if pairs is not None:
                    data["ct_phase"] = CTPhaseUncertainty.from_pairs(pairs)
        for field in ("setup", "range", "ct_phase", "ctPhase"):
            value = data.get(field)
            if isinstance(value, dict):
                value = dict(value)
                value.pop("dimension", None)
                data[field] = value
        return data

    @field_serializer("distribution")
    def _serialize_distribution(self, value: Any, info: SerializationInfo) -> Any:
        if isinstance(value, JointDistribution):
            return value.to_dict()
        return value

    @model_serializer(mode="wrap")
    def _serialize(self, handler: SerializerFunctionWrapHandler, info: SerializationInfo) -> Any:
        """Add the model name; in a matRad context emit matRad's flat parameter struct."""
        if info.context is not None and "matRad" in info.context:
            return self._matrad_dict()
        output = handler(self)
        if isinstance(output, dict):
            output = {"model": self.short_name, **output}
        return output

    def __init__(self, ct: Any = None, **kwargs: Any):
        if ct is not None:
            warnings.warn(
                "Passing a CT to a scenario model is deprecated and has no effect; select the "
                "CT phases with ct_phase.",
                DeprecationWarning,
                stacklevel=2,
            )
        super().__init__(**kwargs)

    def _matrad_dict(self) -> dict[str, Any]:
        """Flat matRad ``multScen`` struct (1-based CT phases)."""
        out = {
            "model": self.short_name,
            "rangeRelSD": float(self.range.rel_sd),
            "rangeAbsSD": float(self.range.abs_sd),
            "shiftSD": [float(s) for s in self.setup.sd],
            "wcSigma": float(self.wc_sigma),
            "ctScenProb": [(p + 1, w) for p, w in self.ct_phase.pairs()],
        }
        out.update(self._matrad_extra())
        return out

    def _matrad_extra(self) -> dict[str, Any]:
        """Additional flat matRad parameters of a subclass (e.g. ``combinations``)."""
        return {}

    def to_matrad(self, context: Union[str, dict] = "mat-file") -> Any:
        """Return the matRad ``multScen`` struct of this model."""
        if isinstance(context, dict):
            context = context.get("matRad", "mat-file")
        if context != "mat-file":
            raise ValueError(f"Context {context} not supported")
        return self._matrad_dict()

    # ------------------------------------------------------------------ uncertainty space
    def dimensions(self) -> tuple[UncertaintyDimension, ...]:
        """Dimensions of the model in canonical order (CT phase, setup, range)."""
        return (self.ct_phase, self.setup, self.range)

    def distribution_object(self) -> JointDistribution:
        """Return the joint distribution instance for :meth:`dimensions`."""
        return create_distribution(self.distribution, self.dimensions())

    def default_axes(self) -> tuple[AxisSpec, ...]:
        """Return the default axis tying of the generated set (matRad: CT phase, setup, range)."""
        return MATRAD_AXES

    def validate_ct(self, ct: Any) -> None:
        """Check the dimensions against ``ct`` and remember its number of phases."""
        if ct is None:
            return
        for dim in self.dimensions():
            dim.validate_against_ct(ct)
        self._num_available_ct_phases = int(getattr(ct, "num_of_ct_scen", 1))

    @property
    def num_available_ct_phases(self) -> int:
        """Phases of the last CT validated against, else the highest used phase plus one."""
        if self._num_available_ct_phases is not None:
            return self._num_available_ct_phases
        return int(max(self.ct_phase.phases)) + 1

    # ------------------------------------------------------------------ generation
    def scenarios(self, ct: Any = None, *, refresh: bool = False) -> ScenarioSet:
        """
        Return the scenario set of this model (generated lazily, cached until it changes).

        Parameters
        ----------
        ct : CT, optional
            CT to validate the CT phases against.
        refresh : bool
            Force regeneration (relevant for random models).
        """
        self.validate_ct(ct)
        fingerprint = self._fingerprint()
        if refresh or self._set_cache is None or fingerprint != self._cache_fingerprint:
            scen_set = self._generate()
            if not isinstance(scen_set, ScenarioSet):
                raise TypeError(f"{type(self).__name__}._generate must return a ScenarioSet")
            self._set_cache = scen_set
            self._cache_fingerprint = fingerprint
        return self._set_cache

    def invalidate(self) -> None:
        """Drop the cached scenario set."""
        self._set_cache = None
        self._cache_fingerprint = None

    @abstractmethod
    def _generate(self) -> ScenarioSet:
        """Generate the scenario set from the current configuration."""

    def _make_set(
        self,
        points: dict[str, Any],
        *,
        tags: Optional[Any] = None,
        weights: Optional[Any] = None,
        probabilities: Optional[Any] = None,
        bases: Optional[Any] = None,
    ) -> ScenarioSet:
        """Build a set over :meth:`dimensions` with this model's distribution and axes."""
        return ScenarioSet.from_points(
            self.dimensions(),
            points,
            distribution=self.distribution_object(),
            probabilities=probabilities,
            weights=weights,
            tags=tags,
            bases=bases,
            axes=self.default_axes(),
        )

    def _fingerprint(self) -> str:
        try:
            return self.model_dump_json()
        except (TypeError, ValueError):  # pragma: no cover - fallback for exotic fields
            with np.printoptions(threshold=sys.maxsize):
                return repr(self.model_dump())


def _nested_set(data: dict, field: str, attr: str, value: Any) -> None:
    """Set ``attr`` of the nested dimension ``field`` in a raw input dict."""
    current = data.get(field)
    if current is None:
        current = {}
    elif isinstance(current, UncertaintyDimension):
        current = current.model_dump()
        current.pop("dimension", None)
    elif isinstance(current, dict):
        current = dict(current)
    else:
        raise ValueError(f"Cannot combine legacy parameter with {field}={current!r}")
    current[attr] = value
    data[field] = current
