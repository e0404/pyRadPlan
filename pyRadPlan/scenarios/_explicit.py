"""Scenario model wrapping an explicitly given scenario set."""

from __future__ import annotations

from typing import Any, ClassVar

from pydantic import field_validator

from ._base import ScenarioModel
from ._set import ScenarioSet


class ExplicitScenarioModel(ScenarioModel):
    """
    Scenario model that carries a ready-made :class:`ScenarioSet`.

    Used to pass hand-built or externally sampled realizations (e.g. quadrature points of a
    polynomial chaos expansion, or a single scenario extracted from another model) through the
    plan / engine interfaces that expect a model. The set is returned as is by
    :meth:`scenarios`; the dimension parameters only document the uncertainty space.
    """

    name: ClassVar[str] = "Explicit Scenarios"
    short_name: ClassVar[str] = "explicit"

    scenario_set: ScenarioSet

    @field_validator("scenario_set", mode="before")
    @classmethod
    def _coerce_set(cls, v: Any) -> ScenarioSet:
        if isinstance(v, ScenarioSet):
            return v
        return ScenarioSet.model_validate(v)

    @classmethod
    def from_set(cls, scenario_set: ScenarioSet, **kwargs: Any) -> "ExplicitScenarioModel":
        """Wrap ``scenario_set``; dimension parameters are taken from the set's dimensions."""
        params = {d.name: d for d in scenario_set.dimensions if d.name in ("setup", "range")}
        if "ct_phase" in scenario_set.dimension_names:
            params["ct_phase"] = scenario_set.dimension("ct_phase")
        params.update(kwargs)
        return cls(scenario_set=scenario_set, **params)

    def _generate(self) -> ScenarioSet:
        return self.scenario_set

    def _matrad_dict(self) -> dict[str, Any]:
        raise ValueError(
            "An explicit scenario set cannot be exported to matRad, whose multScen struct only "
            "holds the parameters of a model. Select one of the models for the export."
        )
