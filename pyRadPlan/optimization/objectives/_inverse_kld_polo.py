"""Inverse KLD POLO objective."""

from typing import Literal

from pydantic import Field
import array_api_compat

from ...core.xp_utils.typing import Array
from ._objective import Objective


class InverseKLDPOLO(Objective):
    """
    Inverse Kullback-Leibler Divergence (KLD) POLO objective.

    Evaluates entropy/KLD-based penalty on POLO non-lesion probability (1 - p).
    """

    name: Literal["Inverse KLD POLO"] = "Inverse KLD POLO"
    quantity: str = Field(
        default="polo", description="Quantity the objective is evaluated on (e.g. 'polo')."
    )

    def compute_objective(self, values: Array) -> Array:
        xp = array_api_compat.array_namespace(values)
        vals = xp.clip(values, 0.0, 1.0 - 1e-12)
        one_minus_p = 1.0 - vals
        return -xp.sum(one_minus_p * xp.log(one_minus_p))

    def compute_gradient(self, values: Array) -> Array:
        xp = array_api_compat.array_namespace(values)
        vals = xp.clip(values, 0.0, 1.0 - 1e-12)
        one_minus_p = 1.0 - vals
        return 1.0 + xp.log(one_minus_p)
