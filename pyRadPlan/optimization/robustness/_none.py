"""Non-robust evaluation on the nominal scenario(s)."""

import logging

import array_api_compat

from ._base import RobustnessOperator
from ._utils import scenario_value, stack_like

logger = logging.getLogger(__name__)


class NoRobustness(RobustnessOperator):
    """
    Evaluate the objective on the nominal scenario(s) only.

    With a 4D set this is one scenario per CT phase (matRad's ``useNominalCtScen``), summed
    without weights as matRad does. A set without a nominal scenario (e.g. random samples
    without the nominal point) evaluates the first computed scenario of each phase instead.
    """

    name = "none"
    label = "None"
    matrad_name = "none"
    phase_contours = True

    def _select_scenarios(self):
        try:
            return [int(i) for i in self.scenarios.nominal_indices()]
        except ValueError:
            indices = [int(i) for i in self.scenarios.reference_indices()]
            logger.warning(
                "The scenario set has no nominal scenario; objectives without robustness "
                "setting evaluate scenario(s) %s instead",
                indices,
            )
            return indices

    def evaluate(self, entries, values):
        self._check(entries, values)
        objective = entries[0].objective
        xp = array_api_compat.array_namespace(scenario_value(values[0], 0))
        total = xp.stack(
            [
                objective.compute_objective(scenario_value(values[0], s))
                for s in range(self.num_scenarios)
            ]
        )
        return objective.priority * xp.sum(total)

    def gradient(self, entries, values):
        self._check(entries, values)
        objective = entries[0].objective
        first = scenario_value(values[0], 0)
        xp = array_api_compat.array_namespace(first)
        grads = [
            objective.priority * objective.compute_gradient(scenario_value(values[0], s))
            for s in range(self.num_scenarios)
        ]
        return [stack_like(values[0], [xp.astype(g, first.dtype, copy=False) for g in grads])]
