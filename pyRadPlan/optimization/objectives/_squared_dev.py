"""Squared Deviation Objective."""

from typing import Annotated

from pydantic import Field

import array_api_compat

from ...core.xp_utils.typing import Array

from ._objective import Objective, ParameterMetadata

# %% Class definition


class SquaredDeviation(Objective):
    """
    Squared Deviation (least-squares) objective.

    Attributes
    ----------
    d_ref : float
        dose reference value
    """

    name = "Squared Deviation"

    d_ref: Annotated[float, Field(default=60.0, ge=0.0), ParameterMetadata(kind="reference")]

    def compute_objective(self, values: Array) -> Array:
        deviation = values - self.d_ref
        return (deviation @ deviation) / array_api_compat.size(values)

    def compute_gradient(self, values: Array) -> Array:
        return 2.0 * (values - self.d_ref) / array_api_compat.size(values)
