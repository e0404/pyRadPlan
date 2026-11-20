"""Nominal scenario model: one unperturbed scenario per CT phase."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from ._base import ScenarioModel
from ._set import TAG_NOMINAL, ScenarioSet


class NominalScenarioModel(ScenarioModel):
    """
    Nominal scenario model (matRad ``nomScen``).

    Produces one scenario per configured CT phase with zero setup and range shift. The
    probability is the phase probability times the density of the unperturbed point; weights
    are normalized over the phases.
    """

    name: ClassVar[str] = "Nominal Scenario"
    short_name: ClassVar[str] = "nomScen"

    def _generate(self) -> ScenarioSet:
        phases = np.asarray(self.ct_phase.phases, dtype=float).reshape(-1, 1)
        return self._make_set(
            {"ct_phase": phases},
            tags=[{TAG_NOMINAL}] * phases.shape[0],
        )
