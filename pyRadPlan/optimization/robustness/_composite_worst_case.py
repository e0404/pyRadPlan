"""Composite worst case over the scenarios (matRad's ``COWC``)."""

import array_api_compat

from ._base import RobustnessOperator
from ._smooth_max import smooth_max
from ._utils import scenario_value, stack_like


class CompositeWorstCase(RobustnessOperator):
    """
    (Smooth) maximum over the scenarios of the summed objectives (matRad's ``COWC``).

    One instance acts on every objective of the problem carrying this robustness:
    ``F = smoothmax_s F_s`` with ``F_s = sum_j p_j f_js``; the cotangent of entry ``j`` in
    scenario ``s`` is ``(d smoothmax / d F_s) * p_j * g_js``. As in matRad, the exact maximum
    is used while all ``F_s <= 0``.
    """

    name = "composite_worst_case"
    label = "Composite worst case"
    aliases = ("COWC",)
    matrad_name = "COWC"
    scope = "composite"
    phase_contours = True

    def _composite(self, entries, values):
        xp = array_api_compat.array_namespace(scenario_value(values[0], 0))
        total = None
        for entry, value in zip(entries, values):
            objective = entry.objective
            per_scenario = xp.stack(
                [
                    objective.compute_objective(scenario_value(value, s))
                    for s in range(self.num_scenarios)
                ]
            )
            term = objective.priority * xp.astype(per_scenario, scenario_value(value, 0).dtype)
            total = term if total is None else total + term
        method = self.options.max_approximation
        # matRad applies the approximation only when the composite maximum is positive.
        if float(xp.max(total)) <= 0.0:
            method = "none"
        return smooth_max(total, method, self.options.p_norm)

    def evaluate(self, entries, values):
        self._check(entries, values)
        value, _ = self._composite(entries, values)
        return value

    def gradient(self, entries, values):
        self._check(entries, values)
        xp = array_api_compat.array_namespace(scenario_value(values[0], 0))
        _, dmax = self._composite(entries, values)
        cotangents = []
        for entry, value in zip(entries, values):
            objective = entry.objective
            dtype = scenario_value(value, 0).dtype
            scale = xp.astype(dmax, dtype)
            grads = [
                objective.priority
                * scale[s]
                * xp.astype(objective.compute_gradient(scenario_value(value, s)), dtype)
                for s in range(self.num_scenarios)
            ]
            cotangents.append(stack_like(value, grads))
        return cotangents
