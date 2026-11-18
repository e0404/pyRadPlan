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

    def _overvalues(self, xp, values: Array) -> Array:
        # Not xp.clip, which masks and syncs on cupy; zeros_like as torch rejects a Python scalar
        excess = values - self.d_max
        return xp.maximum(excess, xp.zeros_like(excess))

    def compute_objective(self, values: Array) -> Array:
        xp = array_api_compat.array_namespace(values)
        overvalues = self._overvalues(xp, values)

        return (overvalues @ overvalues) / float(array_api_compat.size(values))

    def compute_gradient(self, values: Array) -> Array:
        xp = array_api_compat.array_namespace(values)
        overvalues = self._overvalues(xp, values)
        return 2.0 * overvalues / float(array_api_compat.size(overvalues))
