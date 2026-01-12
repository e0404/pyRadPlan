"""Mean dose objective."""

from typing import Annotated

from pydantic import Field

import array_api_compat

from ...core.xp_utils.typing import Array

from ._objective import Objective, ParameterMetadata

# %% Class definition


class MeanDose(Objective):
    """
    Mean Dose objective.

    Attributes
    ----------
    d_ref : float
        referene mean dose to achieve

    Notes
    -----
    While we implement a reference value, we suggest to only use 0 as reference
    """

    name = "Mean Dose"

    d_ref: Annotated[float, Field(default=0.0, ge=0.0), ParameterMetadata(kind="reference")]
    f_diff: Annotated[
        str,
        Field(default="quadratic", alias="f_\{diff\}"),
        ParameterMetadata(kind=["linear", "quadratic"]),
    ]

    def compute_objective(self, values: Array) -> Array:
        xp = array_api_compat.array_namespace(values)
        if self.f_diff == "linear":
            return xp.abs(xp.mean(values) - self.d_ref)
        else:
            return (xp.mean(values) - self.d_ref) ** 2

    def compute_gradient(self, values: Array) -> Array:
        xp = array_api_compat.array_namespace(values)
        grad = xp.full_like(values, 1.0 / array_api_compat.size(values))
        if self.f_diff == "linear":
            grad *= xp.sign(xp.mean(values) - self.d_ref)
        else:
            grad *= 2 * (xp.mean(values) - self.d_ref)
        return grad
