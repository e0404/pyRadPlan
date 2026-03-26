"""Dose uniformity."""

import array_api_compat

from ...core.xp_utils.typing import Array

from ._objective import Objective


class DoseUniformity(Objective):
    """Uniformity (minimize standard deviation) objective."""

    name = "Dose Uniformity"

    def compute_objective(self, values: Array) -> Array:
        xp = array_api_compat.array_namespace(values)
        return xp.std(values, correction=1)

    def compute_gradient(self, values: Array) -> Array:
        xp = array_api_compat.array_namespace(values)
        std_val = xp.std(values, correction=1)
        if std_val > 0.0:
            n = array_api_compat.size(values)
            grad = values - xp.mean(values)
            grad /= std_val * (n - 1)
        else:
            grad = xp.zeros_like(values)
        return grad
