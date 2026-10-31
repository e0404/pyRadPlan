"""Expected value of the objective over the scenarios."""

import array_api_compat

from ...core import xp_utils
from ._base import RobustnessOperator


class ExpectedValue(RobustnessOperator):
    """
    Weighted average of the objective over all scenarios (matRad's ``STOCH``).

    ``F = p * sum_s w_s f(q_s)`` with the set's normalized aggregation weights ``w_s`` (matRad
    multiplies by the raw densities ``scenProb`` instead).
    """

    name = "expected_value"
    aliases = ("STOCH", "stochastic", "probabilistic")
    matrad_name = "STOCH"

    def _weights_as(self, values):
        xp = array_api_compat.array_namespace(values)
        return xp.asarray(
            xp_utils.from_numpy(xp, self.weights, device=array_api_compat.device(values)),
            dtype=values.dtype,
        )

    def evaluate(self, entries, values):
        self._check(entries, values)
        objective = entries[0].objective
        xp = array_api_compat.array_namespace(values[0])
        per_scenario = xp.stack(
            [objective.compute_objective(values[0][s, ...]) for s in range(self.num_scenarios)]
        )
        weights = self._weights_as(values[0])
        return objective.priority * xp.sum(xp.astype(per_scenario, weights.dtype) * weights)

    def gradient(self, entries, values):
        self._check(entries, values)
        objective = entries[0].objective
        xp = array_api_compat.array_namespace(values[0])
        grads = xp.stack(
            [objective.compute_gradient(values[0][s, ...]) for s in range(self.num_scenarios)]
        )
        weights = self._weights_as(values[0])
        scaled = xp.astype(grads, weights.dtype) * weights[:, None]
        return [xp.asarray(objective.priority * scaled, dtype=values[0].dtype, copy=False)]
