"""Squared Underdosing Objective."""

from typing import Annotated, Literal

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

    name: Literal["Squared Underdosing"] = "Squared Underdosing"

    d_min: Annotated[
        float,
        Field(default=60.0, ge=0.0, description="Minimum dose below which dose is penalized."),
        ParameterMetadata(kind="reference"),
    ]

    def _undervalues(self, xp, values: Array) -> Array:
        # Not xp.clip, which masks and syncs on cupy; zeros_like as torch rejects a Python scalar
        deficit = values - self.d_min
        return xp.minimum(deficit, xp.zeros_like(deficit))

    def compute_objective(self, values: Array) -> Array:
        xp = array_api_compat.array_namespace(values)
        undervalues = self._undervalues(xp, values)

        return (undervalues @ undervalues) / float(array_api_compat.size(values))

    def compute_gradient(self, values: Array) -> Array:
        xp = array_api_compat.array_namespace(values)
        undervalues = self._undervalues(xp, values)
        return 2.0 * undervalues / float(array_api_compat.size(undervalues))
