"""LogSumExp ConvPOLO objective."""

from typing import Literal

from pydantic import Field
import array_api_compat

from ...core.xp_utils.typing import Array
from ._objective import Objective


class LogSumExpConvPOLO(Objective):
    """
    LogSumExp ConvPOLO objective.

    A convexified POLO objective using the Log-Sum-Exp approximation
    of the maximum function on the linear predictor (eta).
    """

    name: Literal["LogSumExp ConvPOLO"] = "LogSumExp ConvPOLO"
    quantity: str = Field(
        default="linear_polo",
        description="Quantity the objective is evaluated on (must be linear_polo or similar eta).",
    )

    def compute_objective(self, values: Array) -> Array:
        xp = array_api_compat.array_namespace(values)
        max_val = xp.max(values)
        # Log-Sum-Exp with max-subtraction for numerical stability
        lse = max_val + xp.log(xp.sum(xp.exp(values - max_val)))
        return lse

    def compute_gradient(self, values: Array) -> Array:
        xp = array_api_compat.array_namespace(values)
        max_val = xp.max(values)
        return xp.exp(values - max_val) / xp.sum(xp.exp(values - max_val))
