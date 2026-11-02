"""Expected value of the objective over the scenarios."""

import array_api_compat

from ._base import RobustnessOperator
from ._utils import as_backend, scenario_value, stack_like


class ExpectedValue(RobustnessOperator):
    """
    Weighted average of the objective over all scenarios (matRad's ``STOCH``).

    ``F = p * sum_s w_s f(q_s)`` with the set's normalized aggregation weights ``w_s`` (matRad
    multiplies by the raw densities ``scenProb`` instead).
    """

    name = "expected_value"
    aliases = ("STOCH", "stochastic", "probabilistic")
    matrad_name = "STOCH"
    phase_contours = True

    def evaluate(self, entries, values):
        self._check(entries, values)
        objective = entries[0].objective
        first = scenario_value(values[0], 0)
        xp = array_api_compat.array_namespace(first)
        per_scenario = xp.stack(
            [
                objective.compute_objective(scenario_value(values[0], s))
                for s in range(self.num_scenarios)
            ]
        )
        weights = as_backend(self.weights, first)
        return objective.priority * xp.sum(xp.astype(per_scenario, weights.dtype) * weights)

    def gradient(self, entries, values):
        self._check(entries, values)
        objective = entries[0].objective
        first = scenario_value(values[0], 0)
        xp = array_api_compat.array_namespace(first)
        weights = as_backend(self.weights, first)
        grads = [
            objective.compute_gradient(scenario_value(values[0], s))
            for s in range(self.num_scenarios)
        ]
        scaled = [
            objective.priority * (xp.astype(grad, weights.dtype) * weights[s])
            for s, grad in enumerate(grads)
        ]
        return [stack_like(values[0], [xp.astype(g, first.dtype, copy=False) for g in scaled])]
