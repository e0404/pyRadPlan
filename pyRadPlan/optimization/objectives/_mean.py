"""Mean dose objective."""

from typing import Annotated, Literal

from pydantic import Field

import array_api_compat

from ...core.xp_utils.typing import Array

from ._objective import Objective, ParameterMetadata

# %% Class definition


class MeanDose(Objective):
    """
    Mean Dose objective, pulling the mean towards d_ref from both sides (0: minimize).

    Attributes
    ----------
    d_ref : float
        reference mean dose to achieve

    Notes
    -----
    The deviation is penalized in both directions, so a mean below d_ref is pushed up.
    Use 0 as reference to minimize the mean, or :class:`MeanOverdosing` to penalize
    only a mean above a limit.
    """

    name: Literal["Mean Dose"] = "Mean Dose"

    d_ref: Annotated[
        float,
        Field(
            default=0.0,
            ge=0.0,
            description="Mean dose to pull towards, also from below; 0 minimizes the mean.",
        ),
        ParameterMetadata(kind="reference"),
    ]
    f_diff: Annotated[
        Literal["linear", "quadratic"],
        Field(
            default="quadratic",
            alias="f_{diff}",
            description="Difference function penalizing deviation from the reference mean dose.",
        ),
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


class MeanOverdosing(Objective):
    """
    Mean Overdosing objective, penalizing the mean dose only above d_max (one-sided).

    Attributes
    ----------
    d_max : float
        mean dose above which the squared excess of the mean is penalized
    """

    name: Literal["Mean Overdosing"] = "Mean Overdosing"

    d_max: Annotated[
        float,
        Field(default=0.0, ge=0.0, description="Mean dose above which the mean is penalized."),
        ParameterMetadata(kind="reference"),
    ]

    def compute_objective(self, values: Array) -> Array:
        xp = array_api_compat.array_namespace(values)
        excess = xp.clip(xp.mean(values) - self.d_max, min=0.0, max=None)
        return excess**2

    def compute_gradient(self, values: Array) -> Array:
        xp = array_api_compat.array_namespace(values)
        excess = xp.clip(xp.mean(values) - self.d_max, min=0.0, max=None)
        grad = xp.full_like(values, 2.0 / array_api_compat.size(values))
        grad *= excess
        return grad
