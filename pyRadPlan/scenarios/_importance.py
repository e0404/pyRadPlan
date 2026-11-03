"""Gridded scenarios with importance weights (matRad ``impScen``)."""

from __future__ import annotations

from typing import ClassVar

from pydantic import Field

from ._gridded import GriddedScenarioModel


class ImportanceScenarioModel(GriddedScenarioModel):
    """Gridded scenarios (matRad ``impScen``), weighted by their probability."""

    name: ClassVar[str] = "Importance Grid"
    short_name: ClassVar[str] = "impScen"

    num_setup_grid_points: int = Field(default=9, ge=1)
    num_range_grid_points: int = Field(default=9, ge=1)
