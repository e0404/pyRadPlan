"""Hellinger ConvPOLO objective."""

from typing import Literal

from pydantic import Field
import array_api_compat

from ...core.xp_utils.typing import Array
from ._objective import Objective


class HellingerConvPOLO(Objective):
    """
    Hellinger ConvPOLO objective.

    Linear sum penalty over finite linear predictor values (eta).

    For model formulation and paper references, see :class:`pyRadPlan.quantities.POLO`
    (Ortkamp et al., Med Phys 2026, DOI: 10.1002/mp.70395).
    """

    name: Literal["Hellinger ConvPOLO"] = "Hellinger ConvPOLO"
    quantity: str = Field(
        default="linear_polo",
        description="Quantity the objective is evaluated on (must be linear_polo or similar eta).",
    )
    risk_voi_name: str = Field(
        default="Ventricles_4mm",
        description="Name of the periventricular risk structure in CST (<= 4mm).",
    )

    def compute_objective(self, values: Array) -> Array:
        xp = array_api_compat.array_namespace(values)
        finite_vals = xp.where(xp.isfinite(values), values, xp.zeros_like(values))
        return xp.sum(finite_vals)

    def compute_gradient(self, values: Array) -> Array:
        xp = array_api_compat.array_namespace(values)
        return xp.where(xp.isfinite(values), xp.ones_like(values), xp.zeros_like(values))
