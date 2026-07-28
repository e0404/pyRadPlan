"""Squared Deviation POLO objective."""

from typing import Literal

from pydantic import Field
import array_api_compat

from ...core.xp_utils.typing import Array
from ._objective import Objective


class SquaredDeviationPOLO(Objective):
    """
    Squared Deviation POLO objective.

    Penalizes squared deviation of POLO probabilities from a reference value.
    """

    name: Literal["Squared Deviation POLO"] = "Squared Deviation POLO"
    quantity: str = Field(
        default="polo", description="Quantity the objective is evaluated on (e.g. 'polo')."
    )
    polo_ref: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Reference POLO probability threshold."
    )

    def compute_objective(self, values: Array) -> Array:
        xp = array_api_compat.array_namespace(values)
        deviation = values - self.polo_ref
        return (deviation @ deviation) / array_api_compat.size(values)

    def compute_gradient(self, values: Array) -> Array:
        return 2.0 * (values - self.polo_ref) / array_api_compat.size(values)
