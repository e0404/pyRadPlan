"""LogSumExp ConvPOLO objective."""

from typing import Annotated, ClassVar, Literal

from pydantic import Field
import array_api_compat

from ...core.xp_utils.typing import Array
from ._objective import Objective, ParameterMetadata


class LogSumExpConvPOLO(Objective):
    """
    LogSumExp ConvPOLO objective.

    A convexified POLO objective using the Log-Sum-Exp approximation of the maximum function
    on the linear predictor (eta), following Ortkamp et al. (Equation 11). The maximum is
    subtracted and re-added for numerical stability, as suggested there.

    Voxels outside the model's dose domain arrive as ``-inf`` from
    :class:`~pyRadPlan.quantities.LinearPOLO` and are excluded from both the objective and the
    gradient. If no voxel is inside the domain -- which is the case at the start of an
    optimization, where the fluence and hence the dose are zero -- the objective contributes
    nothing and carries no gradient.

    For model formulation and paper references, see :class:`pyRadPlan.quantities.POLO`
    (Ortkamp et al., Med Phys 2026, DOI: 10.1002/mp.70395).
    """

    name: Literal["LogSumExp ConvPOLO"] = "LogSumExp ConvPOLO"
    quantity: str = Field(
        default="linear_polo",
        description="Quantity the objective is evaluated on (must be linear_polo or similar eta).",
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

    @staticmethod
    def _in_domain_max(xp, values: Array) -> tuple[Array, Array]:
        """Return ``(finite_mask, max_over_finite)``; the max is -inf if nothing is finite."""
        finite = xp.isfinite(values)
        masked = xp.where(finite, values, xp.full_like(values, float("-inf")))
        return finite, xp.max(masked)

    def compute_objective(self, values: Array) -> Array:
        xp = array_api_compat.array_namespace(values)
        if array_api_compat.size(values) == 0:
            return xp.zeros((), dtype=values.dtype)

        finite, max_val = self._in_domain_max(xp, values)
        if not bool(xp.any(finite)):
            return xp.zeros((), dtype=values.dtype)

        # Log-Sum-Exp with max-subtraction for numerical stability. Non-finite entries are
        # excluded from the sum, consistently with compute_gradient and HellingerConvPOLO.
        exps = xp.where(finite, xp.exp(values - max_val), xp.zeros_like(values))
        return max_val + xp.log(xp.sum(exps))

    def compute_gradient(self, values: Array) -> Array:
        xp = array_api_compat.array_namespace(values)
        if array_api_compat.size(values) == 0:
            return xp.zeros_like(values)

        finite, max_val = self._in_domain_max(xp, values)
        if not bool(xp.any(finite)):
            return xp.zeros_like(values)

        exps = xp.where(finite, xp.exp(values - max_val), xp.zeros_like(values))
        return exps / xp.sum(exps)
