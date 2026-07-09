"""Equivalent uniform dose objective."""

from typing import Annotated, Literal

from pydantic import Field

import array_api_compat

from ...core.xp_utils.typing import Array
from ._objective import Objective, ParameterMetadata


class EUD(Objective):
    """
    Equivalent uniform dose (EUD) objective.

    Attributes
    ----------
    k : float
        exponent.
    eud_ref : float
        reference value
    """

    name: Literal["EUD"] = "EUD"

    eud_ref: Annotated[
        float,
        Field(default=0.0, ge=0.0, description="Reference equivalent uniform dose (EUD)."),
        ParameterMetadata(kind="reference"),
    ]
    k: Annotated[
        float,
        Field(default=1.0, description="EUD exponent (volume-effect parameter)."),
        ParameterMetadata(),
    ]
    f_diff: Annotated[
        Literal["linear", "quadratic"],
        Field(
            default="quadratic",
            alias="f_{diff}",
            description="Difference function penalizing deviation from the reference EUD.",
        ),
        ParameterMetadata(kind=["linear", "quadratic"]),
    ]

    def compute_objective(self, values: Array) -> Array:
        xp = array_api_compat.array_namespace(values)
        eud = (xp.sum(values ** (1 / self.k)) / array_api_compat.size(values)) ** self.k
        if self.f_diff == "linear":
            return xp.abs(eud - self.eud_ref)
        else:
            return (eud - self.eud_ref) ** 2

    def compute_gradient(self, values: Array) -> Array:
        xp = array_api_compat.array_namespace(values)
        eud = (xp.sum(values ** (1 / self.k)) / array_api_compat.size(values)) ** self.k
        eud_gradient = (
            xp.sum(values ** (1 / self.k)) ** (self.k - 1)
            * values ** (1 / self.k - 1)
            / (array_api_compat.size(values) ** self.k)
        )
        if self.f_diff == "linear":
            return xp.sign(eud - self.eud_ref) * eud_gradient
        else:
            return 2.0 * (eud - self.eud_ref) * eud_gradient
