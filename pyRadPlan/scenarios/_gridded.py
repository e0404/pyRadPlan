"""Gridded scenario models (matRad ``wcScen`` / ``impScen`` share this base)."""

from __future__ import annotations

from abc import ABC
from typing import Any, ClassVar, Literal

import numpy as np
from pydantic import Field

from ._base import ScenarioModel
from ._set import TAG_NOMINAL, ScenarioSet


def unique_stable_rows(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the unique rows in order of first appearance and their first indices."""
    values = np.asarray(values, dtype=float)
    seen: dict[tuple, int] = {}
    for i, row in enumerate(values + 0.0):
        seen.setdefault(tuple(row.tolist()), i)
    index = np.fromiter(seen.values(), dtype=int, count=len(seen))
    return values[index], index


def _grid(center: np.ndarray, extent: np.ndarray, num_points: int) -> np.ndarray:
    """``(num_points, d)`` grid of linspaces ``center +- extent`` (the center for one point)."""
    if num_points <= 1:
        return np.tile(center, (1, 1))
    grid = np.column_stack([np.linspace(-e, e, num_points) for e in extent]) + center
    if num_points % 2 == 1:
        grid[num_points // 2] = center
    return grid


def _nominal_first(rows: np.ndarray, nominal: np.ndarray) -> np.ndarray:
    """Move the nominal row to the front (matRad), keeping the order of the others."""
    is_nominal = np.all(rows == nominal, axis=1)
    if not is_nominal.any():
        return rows
    return np.vstack([nominal, rows[~is_nominal]])


def _meshgrid_rows(grid: np.ndarray) -> np.ndarray:
    """Rows of MATLAB's ``meshgrid`` with ``(:)`` linearization (second index fastest)."""
    axes = np.meshgrid(*grid.T, indexing="xy")
    return np.column_stack([a.ravel(order="F") for a in axes])


class GriddedScenarioModel(ScenarioModel, ABC):
    """
    Base of scenario models placing scenarios on a grid within ``wc_sigma`` standard deviations.

    Replicates matRad's ``matRad_GriddedScenariosAbstract``: every setup direction and the range
    components get an equidistant grid of ``+- wc_sigma * SD``, which is combined according to
    :attr:`combinations`. The nominal setup / range shift is moved to the first grid position and
    the whole set is replicated for each CT phase (phases form the outer loop). Probabilities
    come from the model's distribution times the phase probability, weights are the normalized
    probabilities.

    Attributes
    ----------
    combinations : {"none", "shift", "all"}
        ``"none"``: one component at a time; ``"shift"``: full setup grid combined only with
        the nominal range; ``"all"``: setup and range grids combined.
    num_setup_grid_points, num_range_grid_points : int
        Grid points per setup direction / range component (subclasses constrain them). Whether
        the absolute and relative range components move together is ``range.combined``.
    """

    legacy_aliases: ClassVar[dict[str, str]] = {
        "numOfSetupGridPoints": "num_setup_grid_points",
        "numOfRangeGridPoints": "num_range_grid_points",
    }

    combinations: Literal["none", "shift", "all"] = Field(
        default="none", description="Combination of the setup and range grids."
    )
    num_setup_grid_points: int = Field(default=3, ge=1)
    num_range_grid_points: int = Field(default=3, ge=1)

    def _matrad_extra(self) -> dict[str, Any]:
        return {
            "combinations": self.combinations,
            "combineRange": bool(self.range.combined),
            "numOfSetupGridPoints": int(self.num_setup_grid_points),
            "numOfRangeGridPoints": int(self.num_range_grid_points),
        }

    def _setup_shifts(self) -> np.ndarray:
        center = self.setup.mean()
        grid = _grid(center, self.wc_sigma * self.setup.std(), self.num_setup_grid_points)
        if self.combinations == "none":
            blocks = []
            for i in range(grid.shape[1]):
                block = np.tile(center, (grid.shape[0], 1))
                block[:, i] = grid[:, i]
                blocks.append(block)
            shifts = np.vstack(blocks)
        else:
            shifts = _meshgrid_rows(grid)
        shifts, _ = unique_stable_rows(shifts)
        return _nominal_first(shifts, center)

    def _range_shifts(self) -> np.ndarray:
        center = self.range.mean()
        grid = _grid(center, self.wc_sigma * self.range.std(), self.num_range_grid_points)
        shifts = grid if self.range.combined else _meshgrid_rows(grid)
        shifts, _ = unique_stable_rows(shifts)
        return _nominal_first(shifts, center)

    def _generate(self) -> ScenarioSet:
        setup_shifts = self._setup_shifts()
        range_shifts = self._range_shifts()
        n_setup, n_range = setup_shifts.shape[0], range_shifts.shape[0]
        setup_nom, range_nom = self.setup.mean(), self.range.mean()

        if self.combinations == "all":
            setup = np.tile(setup_shifts, (n_range, 1))
            rng = np.repeat(range_shifts, n_setup, axis=0)
        else:
            setup = np.vstack([setup_shifts, np.tile(setup_nom, (n_range, 1))])
            rng = np.vstack([np.tile(range_nom, (n_setup, 1)), range_shifts])
        _, keep = unique_stable_rows(np.hstack([setup, rng]))
        setup, rng = setup[keep], rng[keep]

        phases = np.asarray(self.ct_phase.phases, dtype=float)
        n = setup.shape[0]
        is_nominal = np.all(setup == setup_nom, axis=1) & np.all(rng == range_nom, axis=1)
        tags = [{TAG_NOMINAL} if nom else set() for nom in is_nominal] * len(phases)
        return self._make_set(
            {
                "ct_phase": np.repeat(phases, n).reshape(-1, 1),
                "setup": np.tile(setup, (len(phases), 1)),
                "range": np.tile(rng, (len(phases), 1)),
            },
            tags=tags,
        )
