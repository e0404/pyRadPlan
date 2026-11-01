"""Worst-case scenario model (matRad ``wcScen``)."""

from __future__ import annotations

from typing import ClassVar

from pydantic import Field, field_validator

from ._gridded import GriddedScenarioModel


class WorstCaseScenarioModel(GriddedScenarioModel):
    """
    Worst-case scenarios (matRad ``wcScen``): nominal plus the extremes at ``+- wc_sigma * SD``.

    Only one (nominal only) or three (nominal and both extremes) grid points are supported per
    uncertainty, as in matRad.
    """

    name: ClassVar[str] = "Worst Case Scenarios"
    short_name: ClassVar[str] = "wcScen"

    num_setup_grid_points: int = Field(default=3)
    num_range_grid_points: int = Field(default=3)

    @field_validator("num_setup_grid_points", "num_range_grid_points")
    @classmethod
    def _one_or_three(cls, v: int) -> int:
        if v not in (1, 3):
            raise ValueError(
                "WorstCaseScenarioModel only supports 1 (only nominal grid point) or 3 grid "
                "points (nominal + worst case)"
            )
        return v
