"""Objective-wise worst case over the scenarios (matRad's ``OWC``)."""

import array_api_compat

from ._base import RobustnessOperator
from ._smooth_max import smooth_max


class ObjectiveWorstCase(RobustnessOperator):
    """
    (Smooth) maximum of the objective over the scenarios (matRad's ``OWC``).

    ``F = smoothmax_s (p f_s)`` with the approximation of
    :attr:`RobustnessOptions.max_approximation`; the cotangent of scenario ``s`` is
    ``(d smoothmax / d f_s) * p * g_s``.
    """

    name = "objective_worst_case"
    aliases = ("OWC",)
    matrad_name = "OWC"

    def _scenario_values(self, objective, values):
        xp = array_api_compat.array_namespace(values)
        per_scenario = xp.stack(
            [objective.compute_objective(values[s, ...]) for s in range(self.num_scenarios)]
        )
        return objective.priority * xp.astype(per_scenario, values.dtype)

    def _smooth_max(self, values):
        return smooth_max(values, self.options.max_approximation, self.options.p_norm)

    def evaluate(self, entries, values):
        self._check(entries, values)
        value, _ = self._smooth_max(self._scenario_values(entries[0].objective, values[0]))
        return value

    def gradient(self, entries, values):
        self._check(entries, values)
        objective = entries[0].objective
        xp = array_api_compat.array_namespace(values[0])
        _, dmax = self._smooth_max(self._scenario_values(objective, values[0]))
        grads = xp.stack(
            [objective.compute_gradient(values[0][s, ...]) for s in range(self.num_scenarios)]
        )
        grads = xp.astype(grads, values[0].dtype)
        return [objective.priority * dmax[:, None] * grads]
