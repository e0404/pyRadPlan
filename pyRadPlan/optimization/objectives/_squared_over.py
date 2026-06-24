"""Squared Overdosing Objective."""

from typing import Annotated, Literal

from pydantic import Field

import array_api_compat

from ...core.xp_utils.typing import Array

from ._objective import Objective, ParameterMetadata

# %% Class definition


class SquaredOverdosing(Objective):
    """
    Squared Overdosing (piece-wise positive least-squares) objective.

    Attributes
    ----------
    d_max : float
        maximum values value (above which we penalize)
    """

    name: Literal["Squared Overdosing"] = "Squared Overdosing"

    d_max: Annotated[
        float,
        Field(default=30.0, ge=0.0, description="Maximum dose above which dose is penalized."),
        ParameterMetadata(kind="reference"),
    ]

    def compute_objective(self, values: Array) -> Array:
        xp = array_api_compat.array_namespace(values)
        overvalues = xp.clip(values - self.d_max, min=0.0, max=None)

        return (overvalues @ overvalues) / array_api_compat.size(values)

    def compute_gradient(self, values: Array) -> Array:
        xp = array_api_compat.array_namespace(values)
        overvalues = xp.clip(values - self.d_max, min=0.0, max=None)
        return 2.0 * overvalues / array_api_compat.size(overvalues)
