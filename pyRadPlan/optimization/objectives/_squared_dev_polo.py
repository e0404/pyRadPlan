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

    For model formulation and paper references, see :class:`pyRadPlan.quantities.POLO`
    (Ortkamp et al., Med Phys 2026, DOI: 10.1002/mp.70395).
    """

    name: Literal["Squared Deviation POLO"] = "Squared Deviation POLO"
    quantity: str = Field(
        default="polo", description="Quantity the objective is evaluated on (e.g. 'polo')."
    )
    risk_voi_name: str = Field(
        default="Ventricles_4mm",
        description="Name of the periventricular risk structure in CST (<= 4mm).",
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
