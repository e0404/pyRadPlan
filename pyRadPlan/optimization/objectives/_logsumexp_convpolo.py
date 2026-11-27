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

    **Deviation from the paper.** Ortkamp et al. (Sec. 2.3) evaluate every POLO objective
    only on voxels receiving at least 2 Gy. This objective keeps every voxel by default
    (``dose_threshold=0.0``), because the filter removes the convexity the objective was
    introduced for. Eta is affine in the fluence, so on a fixed set of voxels the objective
    is convex in the fluence. With a threshold the set depends on the fluence: a voxel just
    below the threshold is excluded while one just above enters with eta of about -26, so
    the objective jumps there and is neither convex nor continuous. An optimizer can then
    lower it by pushing voxels out of the domain, down to a single voxel just above the
    threshold, at the cost of the target dose.

    Dropping the filter does not change a reported NTCP: :class:`~pyRadPlan.quantities.POLO`
    and the objectives on the probability keep it. What the unfiltered voxels could add is
    bounded. Below 2 Gy course dose, ``eta < -24.73 + 0.036 * LETd`` (LETd in keV/um), which
    for 20 keV/um gives ``p < 4e-11`` per 1.08 mm^3 voxel of the fit, or less than 5e-5 NTCP
    summed over a whole brain (1.4 l).

    For a non-negative fluence the objective is bounded below by ``-26.3 + log(n)`` for
    ``n`` voxels, and its gradient with respect to eta is a softmax that sums to one. Its
    priority therefore belongs on the order of 1, not on that of :class:`NTCPPOLO`, whose
    gradient with respect to eta is of the order of the voxel probabilities.

    With ``dose_threshold > 0``, voxels below it arrive as ``-inf`` from
    :class:`~pyRadPlan.quantities.LinearPOLO` and are excluded from both the objective and the
    gradient. If no voxel is inside the domain -- which is then the case at the start of an
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
    dose_threshold: Annotated[
        float,
        Field(
            default=0.0,
            ge=0.0,
            description=(
                "Course dose in Gy below which a voxel is excluded from the objective. The "
                "default 0.0 keeps every voxel; see the class docstring before raising it."
            ),
        ),
        ParameterMetadata(kind="numeric"),
    ]

    quantity_params: ClassVar[tuple[str, ...]] = ("risk_voi_name", "dose_threshold")

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
