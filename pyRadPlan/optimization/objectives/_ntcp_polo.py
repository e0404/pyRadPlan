"""NTCP POLO objective."""

from typing import Annotated, ClassVar, Literal

from pydantic import Field
import array_api_compat

from ...core.xp_utils.typing import Array
from ...quantities import P_MAX, total_ntcp
from ._objective import Objective, ParameterMetadata


class NTCPPOLO(Objective):
    """
    NTCP POLO objective.

    Evaluates the Normal Tissue Complication Probability (NTCP) based on the
    Probability of Lesion Origin (POLO) map.

    .. note::
       In float32 precision across large evaluation volumes, the product term
       exp(sum(log(1-p))) can underflow to 0.0, causing vanishing gradients.
       If optimization stalls, consider using :class:`HellingerPOLO` or
       :class:`LogSumExpConvPOLO` as alternative objectives.

    For model formulation and paper references, see :class:`pyRadPlan.quantities.POLO`
    (Ortkamp et al., Med Phys 2026, DOI: 10.1002/mp.70395).
    """

    name: Literal["NTCP POLO"] = "NTCP POLO"
    quantity: str = Field(
        default="polo", description="Quantity the objective is evaluated on (e.g. 'polo')."
    )
    risk_voi_name: Annotated[
        str,
        Field(
            default="Ventricles_4mm",
            description="Name of the periventricular risk structure in CST (<= 4mm).",
        ),
        ParameterMetadata(kind="voi_reference"),
    ]

    quantity_params: ClassVar[tuple[str, ...]] = ("risk_voi_name",)

    def compute_objective(self, values: Array) -> Array:
        return total_ntcp(values)

    def compute_gradient(self, values: Array) -> Array:
        # d/dp_i [1 - prod_j (1 - p_j)] = prod_{j != i} (1 - p_j)  (Ortkamp et al., Equation 10)
        xp = array_api_compat.array_namespace(values)
        vals = xp.clip(values, 0.0, P_MAX)
        prod_val = xp.exp(xp.sum(xp.log1p(-vals)))
        return prod_val / (1.0 - vals)
