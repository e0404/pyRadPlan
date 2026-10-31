"""Variance over scenarios objective."""

from typing import Literal, Optional

import numpy as np
import array_api_compat

from ...core import xp_utils
from ...core.xp_utils.typing import Array

from ._objective import Objective


class Variance(Objective):
    """
    Mean voxel-wise variance of the quantity over the scenarios.

    ``f = (1/n) sum_s w_s ||q_s - q_bar||^2`` with ``q_bar = sum_s w_s q_s``, i.e. the mean over
    the ``n`` voxels of the weighted variance over the scenarios. The objective is
    scenario-aware: it receives the quantity stacked over the scenarios, shape ``(S, n)``, and
    the scenario weights (uniform if omitted, normalized to sum to one). Its gradient is
    ``2 w_s (q_s - q_bar) / n``. It replaces the variance term of matRad's ``PROB`` robustness
    without the Omega matrices and has no matRad equivalent.

    A single scenario (a one-dimensional ``values``) has zero variance.
    """

    name: Literal["Variance"] = "Variance"
    scenario_aware = True

    @staticmethod
    def _deviation(values: Array, weights: Optional[Array]):
        xp = array_api_compat.array_namespace(values)
        num_scenarios = values.shape[0]
        if weights is None:
            weights = np.ones(num_scenarios)
        if array_api_compat.is_array_api_obj(weights) and (
            array_api_compat.array_namespace(weights) is xp
        ):
            weights = xp.asarray(weights, dtype=values.dtype)
        else:
            weights = xp.asarray(
                xp_utils.from_numpy(
                    xp,
                    np.asarray(xp_utils.to_numpy(weights), dtype=float),
                    device=array_api_compat.device(values),
                ),
                dtype=values.dtype,
            )
        if weights.shape != (num_scenarios,):
            raise ValueError(
                f"Variance needs one weight per scenario ({num_scenarios}), "
                f"got shape {tuple(weights.shape)}"
            )
        weights = weights / xp.sum(weights)
        deviation = values - xp.sum(weights[:, None] * values, axis=0)
        return xp, weights, deviation

    def compute_objective(self, values: Array, weights: Optional[Array] = None) -> Array:
        xp = array_api_compat.array_namespace(values)
        if values.ndim == 1:
            return xp.zeros((), dtype=values.dtype, device=array_api_compat.device(values))
        xp, weights, deviation = self._deviation(values, weights)
        return xp.sum(weights[:, None] * deviation**2) / values.shape[1]

    def compute_gradient(self, values: Array, weights: Optional[Array] = None) -> Array:
        if values.ndim == 1:
            return array_api_compat.array_namespace(values).zeros_like(values)
        _, weights, deviation = self._deviation(values, weights)
        return 2.0 * weights[:, None] * deviation / values.shape[1]
