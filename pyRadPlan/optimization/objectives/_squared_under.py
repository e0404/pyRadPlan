"""Squared Underdosing Objective."""

from typing import Annotated

from pydantic import Field
import array_api_compat

from ...core.xp_utils.typing import Array

from ._objective import Objective, ParameterMetadata

# %% Class definition


class SquaredUnderdosing(Objective):
    """
    Squared Underdosing (piece-wise negative least-squares) objective.

    Attributes
    ----------
    d_min : float
        minimum values value (below which we penalize)
    """

    name = "Squared Underdosing"

    d_min: Annotated[float, Field(default=60.0, ge=0.0), ParameterMetadata(kind="reference")]

    def compute_objective(self, values: Array) -> Array:
        xp = array_api_compat.array_namespace(values)
        undervalues = xp.clip(values - self.d_min, min=None, max=0.0)

        return (undervalues @ undervalues) / array_api_compat.size(values)

    def compute_gradient(self, values: Array) -> Array:
        xp = array_api_compat.array_namespace(values)
        undervalues = xp.clip(values - self.d_min, min=None, max=0.0)
        return 2.0 * undervalues / array_api_compat.size(undervalues)
