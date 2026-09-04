"""Hellinger POLO objective."""

from typing import ClassVar, Literal

from pydantic import Field
import array_api_compat

from ...core.xp_utils.typing import Array
from ._objective import Objective


class HellingerPOLO(Objective):
    """
    Hellinger POLO objective.

    Evaluates the sum of the POLO probabilities.

    For model formulation and paper references, see :class:`pyRadPlan.quantities.POLO`
    (Ortkamp et al., Med Phys 2026, DOI: 10.1002/mp.70395).
    """

    name: Literal["Hellinger POLO"] = "Hellinger POLO"
    quantity: str = Field(
        default="polo", description="Quantity the objective is evaluated on (e.g. 'polo')."
    )
    risk_voi_name: str = Field(
        default="Ventricles_4mm",
        description="Name of the periventricular risk structure in CST (<= 4mm).",
    )

    quantity_params: ClassVar[tuple[str, ...]] = ("risk_voi_name",)

    def compute_objective(self, values: Array) -> Array:
        xp = array_api_compat.array_namespace(values)
        return xp.sum(values)

    def compute_gradient(self, values: Array) -> Array:
        xp = array_api_compat.array_namespace(values)
        return xp.ones_like(values)
