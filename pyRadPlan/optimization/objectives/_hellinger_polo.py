"""Hellinger POLO objective."""

from typing import Annotated, ClassVar, Literal

from pydantic import Field
import array_api_compat

from ...core.xp_utils.typing import Array
from ._objective import Objective, ParameterMetadata


class HellingerPOLO(Objective):
    """
    Hellinger POLO objective.

    Evaluates the sum of the POLO probabilities. Voxels below :attr:`dose_threshold` (2 Gy
    course dose by default, the index filter of Ortkamp et al., Sec. 2.3) contribute a
    probability of zero.

    For model formulation and paper references, see :class:`pyRadPlan.quantities.POLO`
    (Ortkamp et al., Med Phys 2026, DOI: 10.1002/mp.70395).
    """

    name: Literal["Hellinger POLO"] = "Hellinger POLO"
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
    dose_threshold: Annotated[
        float,
        Field(
            default=2.0,
            ge=0.0,
            description=(
                "Course dose in Gy below which the model is treated as undefined and the voxel "
                "contributes a probability of zero. 0.0 disables the filter."
            ),
        ),
        ParameterMetadata(kind="numeric"),
    ]

    quantity_params: ClassVar[tuple[str, ...]] = ("risk_voi_name", "dose_threshold")

    def compute_objective(self, values: Array) -> Array:
        xp = array_api_compat.array_namespace(values)
        return xp.sum(values)

    def compute_gradient(self, values: Array) -> Array:
        xp = array_api_compat.array_namespace(values)
        return xp.ones_like(values)
