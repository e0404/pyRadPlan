"""Objective-wise worst case over the scenarios (matRad's ``OWC``)."""

import array_api_compat

from ._base import RobustnessOperator
from ._smooth_max import smooth_max
from ._utils import scenario_value, stack_like


class ObjectiveWorstCase(RobustnessOperator):
    """
    (Smooth) maximum of the objective over the scenarios (matRad's ``OWC``).

    ``F = smoothmax_s (p f_s)`` with the approximation of
    :attr:`RobustnessOptions.max_approximation`; the cotangent of scenario ``s`` is
    ``(d smoothmax / d f_s) * p * g_s``.
    """

    name = "objective_worst_case"
    label = "Objective-wise worst case"
    aliases = ("OWC",)
    matrad_name = "OWC"
    phase_contours = True

    def _scenario_values(self, objective, values):
        first = scenario_value(values, 0)
        xp = array_api_compat.array_namespace(first)
        per_scenario = xp.stack(
            [
                objective.compute_objective(scenario_value(values, s))
                for s in range(self.num_scenarios)
            ]
        )
        return objective.priority * xp.astype(per_scenario, first.dtype)

    def _smooth_max(self, values):
        return smooth_max(values, self.options.max_approximation, self.options.p_norm)

    def evaluate(self, entries, values):
        self._check(entries, values)
        value, _ = self._smooth_max(self._scenario_values(entries[0].objective, values[0]))
        return value

    def gradient(self, entries, values):
        self._check(entries, values)
        objective = entries[0].objective
        first = scenario_value(values[0], 0)
        xp = array_api_compat.array_namespace(first)
        _, dmax = self._smooth_max(self._scenario_values(objective, values[0]))
        grads = [
            objective.priority
            * dmax[s]
            * xp.astype(objective.compute_gradient(scenario_value(values[0], s)), first.dtype)
            for s in range(self.num_scenarios)
        ]
        return [stack_like(values[0], grads)]
