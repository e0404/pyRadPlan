"""Non-robust evaluation on the nominal scenario(s)."""

import logging

import array_api_compat

from ._base import RobustnessOperator

logger = logging.getLogger(__name__)


class NoRobustness(RobustnessOperator):
    """
    Evaluate the objective on the nominal scenario(s) only.

    With a 4D set this is one scenario per CT phase (matRad's ``useNominalCtScen``), summed
    without weights as matRad does. A set without a nominal scenario (e.g. random samples
    without the nominal point) evaluates the first computed scenario of each phase instead.
    """

    name = "none"
    matrad_name = "none"

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
        xp = array_api_compat.array_namespace(values[0])
        total = xp.stack(
            [objective.compute_objective(values[0][s, ...]) for s in range(self.num_scenarios)]
        )
        return objective.priority * xp.sum(total)

    def gradient(self, entries, values):
        self._check(entries, values)
        objective = entries[0].objective
        xp = array_api_compat.array_namespace(values[0])
        grads = xp.stack(
            [objective.compute_gradient(values[0][s, ...]) for s in range(self.num_scenarios)]
        )
        return [xp.astype(objective.priority * grads, values[0].dtype, copy=False)]
