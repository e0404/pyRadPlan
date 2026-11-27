"""Hellinger ConvPOLO objective."""

from typing import Annotated, ClassVar, Literal

from pydantic import Field
import array_api_compat

from ...core.xp_utils.typing import Array
from ...quantities import POLOBase
from ._objective import Objective, ParameterMetadata


class HellingerConvPOLO(Objective):
    """
    Hellinger ConvPOLO objective.

    The sum of the linear predictor eta over the voxels of the region: the simplified
    Hellinger objective of Ortkamp et al. (Equation 14) evaluated on the decision function
    instead of the probability. The intercept of the model (-26.3) is subtracted from every
    voxel. It does not move the minimum, but as a constant of -26.3 per voxel it would
    dominate the objective value, and a solver's relative stopping criterion then fires
    long before the plan has converged.

    **Deviation from the paper.** Ortkamp et al. (Sec. 2.3) evaluate every POLO objective
    only on voxels receiving at least 2 Gy. This objective keeps every voxel by default
    (``dose_threshold=0.0``), because the filter removes the convexity the objective was
    introduced for. Eta is affine in the fluence, so on a fixed set of voxels the objective
    is affine in the fluence. With a threshold the set depends on the fluence: a voxel just
    below the threshold is excluded while one just above enters with eta of about -26, so
    the objective jumps there and is neither convex nor continuous. Each voxel crossing the
    threshold changes the objective by about 26 times the priority, which can stall a line
    search.

    Dropping the filter does not change a reported NTCP: :class:`~pyRadPlan.quantities.POLO`
    and the objectives on the probability keep it. What the unfiltered voxels could add is
    bounded. Below 2 Gy course dose, ``eta < -24.73 + 0.036 * LETd`` (LETd in keV/um), which
    for 20 keV/um gives ``p < 4e-11`` per 1.08 mm^3 voxel of the fit, or less than 5e-5 NTCP
    summed over a whole brain (1.4 l).

    The gradient with respect to eta is one per voxel, so value and gradient scale with the
    number of voxels in the region and the priority has to be chosen accordingly smaller.

    With ``dose_threshold > 0``, voxels below it arrive as ``-inf`` from
    :class:`~pyRadPlan.quantities.LinearPOLO` and are excluded from both the objective and the
    gradient.

    For model formulation and paper references, see :class:`pyRadPlan.quantities.POLO`
    (Ortkamp et al., Med Phys 2026, DOI: 10.1002/mp.70395).
    """

    name: Literal["Hellinger ConvPOLO"] = "Hellinger ConvPOLO"
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

    def compute_objective(self, values: Array) -> Array:
        xp = array_api_compat.array_namespace(values)
        shifted = xp.where(xp.isfinite(values), values - POLOBase.INTERCEPT, xp.zeros_like(values))
        return xp.sum(shifted)

    def compute_gradient(self, values: Array) -> Array:
        xp = array_api_compat.array_namespace(values)
        return xp.where(xp.isfinite(values), xp.ones_like(values), xp.zeros_like(values))
