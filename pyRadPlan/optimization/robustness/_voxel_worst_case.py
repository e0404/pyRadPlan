"""Voxel-wise worst case over the scenarios (matRad's ``VWWC`` / ``VWWC_INV``)."""

import logging
from typing import ClassVar, Literal

import numpy as np
import array_api_compat

from ._base import RobustnessOperator

logger = logging.getLogger(__name__)


class VoxelWorstCase(RobustnessOperator):
    """
    Objective evaluated on the voxel-wise worst-case quantity (matRad's ``VWWC``).

    ``F = p * phi(d~)`` with ``d~ = min_s q_s`` per voxel for targets and ``max_s q_s`` for
    organs at risk; other structure types are treated like organs at risk. The cotangent of
    scenario ``s`` is ``p * g(d~)`` on the voxels where ``s`` is the (first) extremal scenario.

    Raises
    ------
    NotImplementedError
        If the selected scenarios span several CT phases (voxels would have to be tracked
        between the phases, which is not implemented, as in matRad).
    """

    name = "voxel_worst_case"
    label = "Voxel-wise worst case"
    aliases = ("VWWC",)
    matrad_name = "VWWC"

    _target_reduction: ClassVar[Literal["min", "max"]] = "min"

    def __init__(self, scenarios, options=None):
        super().__init__(scenarios, options)
        if scenarios.has_dimension("ct_phase"):
            phases = np.unique(np.rint(scenarios.values("ct_phase")[self._scenario_indices, 0]))
            if phases.size > 1:
                raise NotImplementedError(
                    f"Robustness {self.name!r} on scenarios of several CT phases requires "
                    "voxel tracking between the phases, which is not implemented"
                )
        self._warned_types: set[str] = set()

    def _reduction(self, voi_type: str) -> Literal["min", "max"]:
        if voi_type not in ("TARGET", "OAR") and voi_type not in self._warned_types:
            self._warned_types.add(voi_type)
            logger.warning(
                "Robustness %r treats structures of type %r like organs at risk.",
                self.name,
                voi_type,
            )
        if voi_type == "TARGET":
            return self._target_reduction
        return "max" if self._target_reduction == "min" else "min"

    def _worst(self, entry, values):
        xp = array_api_compat.array_namespace(values)
        if self._reduction(entry.voi_type) == "min":
            return xp.min(values, axis=0), xp.argmin(values, axis=0)
        return xp.max(values, axis=0), xp.argmax(values, axis=0)

    def evaluate(self, entries, values):
        self._check(entries, values)
        objective = entries[0].objective
        worst, _ = self._worst(entries[0], values[0])
        return objective.priority * objective.compute_objective(worst)

    def gradient(self, entries, values):
        self._check(entries, values)
        objective = entries[0].objective
        xp = array_api_compat.array_namespace(values[0])
        worst, ix = self._worst(entries[0], values[0])
        grad = xp.asarray(objective.compute_gradient(worst), dtype=values[0].dtype)
        rows = xp.arange(self.num_scenarios, device=array_api_compat.device(values[0]))
        selected = xp.astype(rows[:, None] == ix[None, :], values[0].dtype)
        return [objective.priority * selected * grad[None, :]]


class VoxelWorstCaseInverted(VoxelWorstCase):
    """
    Voxel-wise "best case" (matRad's ``VWWC_INV``).

    As :class:`VoxelWorstCase` with the reductions swapped: the maximum over the scenarios in
    targets and the minimum in organs at risk.
    """

    name = "voxel_worst_case_inverted"
    label = "Voxel-wise worst case (inverted)"
    aliases = ("VWWC_INV",)
    matrad_name = "VWWC_INV"

    _target_reduction = "max"
