"""Objective of the expected quantity over the scenarios."""

import array_api_compat
import numpy as np

from ._base import RobustnessOperator
from ._utils import as_backend


class ExpectedQuantity(RobustnessOperator):
    """
    Objective evaluated on the weighted average quantity over all scenarios.

    ``F = p * phi(q_bar)`` with ``q_bar = sum_s w_s q_s`` and the set's aggregation weights
    normalized to sum to one, so that ``q_bar`` is a true mean also for sets whose weights do
    not (extended sets, subsets, partial CT phase probabilities). This is the expectation part
    of matRad's ``PROB``; matRad's variance term (the Omega matrices) is not included, use the
    ``Variance`` objective for it. Over several CT phases the objective reads the contour of the
    first phase, as ``PROB`` does. matRad's ``PROB`` is accepted as an alias on input; on export
    the operator has no matRad equivalent.
    """

    name = "expected_quantity"
    aliases = ("PROB",)
    matrad_name = None

    def _expected(self, values):
        xp = array_api_compat.array_namespace(values)
        total = float(np.sum(self.weights))
        weights = as_backend(self.weights / total if total > 0.0 else self.weights, values)
        return weights, xp.sum(weights[:, None] * values, axis=0)

    def evaluate(self, entries, values):
        self._check(entries, values)
        objective = entries[0].objective
        _, expected = self._expected(values[0])
        return objective.priority * objective.compute_objective(expected)

    def gradient(self, entries, values):
        self._check(entries, values)
        objective = entries[0].objective
        xp = array_api_compat.array_namespace(values[0])
        weights, expected = self._expected(values[0])
        grad = xp.asarray(objective.compute_gradient(expected), dtype=values[0].dtype)
        return [objective.priority * weights[:, None] * grad[None, :]]
