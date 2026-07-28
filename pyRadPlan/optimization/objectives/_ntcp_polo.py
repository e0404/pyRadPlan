"""NTCP POLO objective."""

from typing import Literal

from pydantic import Field
import array_api_compat

from ...core.xp_utils.typing import Array
from ._objective import Objective


class NTCPPOLO(Objective):
    """
    NTCP POLO objective.

    Evaluates the Normal Tissue Complication Probability (NTCP) based on the
    Probability of Lesion Origin (POLO) map.
    """

    name: Literal["NTCP POLO"] = "NTCP POLO"
    quantity: str = Field(
        default="polo", description="Quantity the objective is evaluated on (e.g. 'polo')."
    )

    def compute_objective(self, values: Array) -> Array:
        xp = array_api_compat.array_namespace(values)
        # Clip to avoid completely hitting 1.0 which makes the product exactly 0 (use 1e-6 for float32 compatibility)
        vals = xp.clip(values, 0.0, 1.0 - 1e-6)
        return 1.0 - xp.prod(1.0 - vals)

    def compute_gradient(self, values: Array) -> Array:
        xp = array_api_compat.array_namespace(values)
        vals = xp.clip(values, 0.0, 1.0 - 1e-6)
        prod_val = xp.prod(1.0 - vals)
        return prod_val / (1.0 - vals)
