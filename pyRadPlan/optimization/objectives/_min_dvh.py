"""Minimum DVH objective."""

from typing import Annotated

from pydantic import Field

import array_api_compat

from ...core.xp_utils.typing import Array

from ._objective import Objective, ParameterMetadata
from ...core.xp_utils.compat import quantile


class MinDVH(Objective):
    """
    Minimum DVH objective.

    Attributes
    ----------
    d : float
        dose point
    v_min : float
        min. relative volume [%]
    """

    name = "Min DVH"

    d: Annotated[float, Field(default=30.0, ge=0.0), ParameterMetadata(kind="reference")]
    v_min: Annotated[
        float, Field(default=95.0, ge=0.0, le=100.0), ParameterMetadata(kind="relative_volume")
    ]

    def compute_objective(self, values: Array) -> Array:
        xp = array_api_compat.array_namespace(values)

        deviation = values - self.d
        values_quantile = quantile(
            values, 1.0 - self.v_min / 100.0, method="linear", is_sorted=False
        )
        deviation = xp.where(
            xp.logical_or(values > self.d, values < values_quantile), 0, deviation
        )

        return (deviation @ deviation) / array_api_compat.size(values)

    def compute_gradient(self, values: Array) -> Array:
        xp = array_api_compat.array_namespace(values)

        deviation = values - self.d
        values_quantile = quantile(
            values, 1.0 - self.v_min / 100.0, method="linear", is_sorted=False
        )
        deviation = xp.where(
            xp.logical_or(values > self.d, values < values_quantile), 0, deviation
        )

        return 2.0 * deviation / array_api_compat.size(values)
