"""
matRad-style accessors of the previous scenario model interface.

Kept for one release so existing consumers (dose engines, stf generators, matRad import)
keep working while they migrate to :class:`~pyRadPlan.scenarios.ScenarioSet`. Everything here
is derived from the model's scenario set tied to matRad's axes (CT phase, setup, range).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ._dimensions import CTPhaseUncertainty
from ._set import MATRAD_AXES

if TYPE_CHECKING:  # pragma: no cover
    from ._set import ScenarioSet


class LegacyScenarioInterface:
    """Mixin providing the flat matRad-style properties on top of :meth:`scenarios`."""

    # -- flat parameters -------------------------------------------------------------
    @property
    def range_rel_sd(self) -> float:
        """Relative range SD in percent (``range.rel_sd``)."""
        return self.range.rel_sd

    @range_rel_sd.setter
    def range_rel_sd(self, value: float) -> None:
        self.range = self.range.model_validate({**self.range.model_dump(), "rel_sd": value})

    @property
    def range_abs_sd(self) -> float:
        """Absolute range SD in mm (``range.abs_sd``)."""
        return self.range.abs_sd

    @range_abs_sd.setter
    def range_abs_sd(self, value: float) -> None:
        self.range = self.range.model_validate({**self.range.model_dump(), "abs_sd": value})

    @property
    def shift_sd(self) -> tuple[float, float, float]:
        """Setup SD per direction in mm (``setup.sd``)."""
        return tuple(self.setup.sd)

    @shift_sd.setter
    def shift_sd(self, value: Any) -> None:
        self.setup = self.setup.model_validate({**self.setup.model_dump(), "sd": tuple(value)})

    @property
    def ct_scen_prob(self) -> list[tuple[int, float]]:
        """``[(phase, probability), ...]`` (0-based; ``ct_phase``)."""
        return self.ct_phase.pairs()

    @ct_scen_prob.setter
    def ct_scen_prob(self, value: Any) -> None:
        self.ct_phase = CTPhaseUncertainty.from_pairs(value)

    # -- derived scenario information ------------------------------------------------
    def _legacy_set(self) -> "ScenarioSet":
        return self.scenarios()

    def _matrad_positions(self) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int]]:
        """Per-scenario ``(phase, shift position, range position)`` and the matRad mask shape."""
        scen_set = self._legacy_set()
        layout = scen_set.layout_for(*MATRAD_AXES)
        phases = scen_set.ct_phases()
        pos = np.column_stack([phases, layout.position[:, 1], layout.position[:, 2]])
        shape = (
            max(self.num_available_ct_phases, int(phases.max(initial=-1)) + 1),
            layout.shape[1],
            layout.shape[2],
        )
        return phases, pos, shape

    @property
    def num_of_ct_scen(self) -> int:
        """Number of CT phases used."""
        return len(self.ct_phase.phases)

    @property
    def num_of_available_ct_scen(self) -> int:
        """Number of CT phases available in the CT (see ``num_available_ct_phases``)."""
        return self.num_available_ct_phases

    @property
    def ct_scen_ix(self) -> np.ndarray:
        """CT phase of each scenario."""
        return self._legacy_set().ct_phases()

    @property
    def iso_shift(self) -> np.ndarray:
        """``(num_scenarios, 3)`` isocenter shifts in mm."""
        return np.asarray(self._legacy_set().values("setup"), dtype=float)

    @property
    def rel_range_shift(self) -> np.ndarray:
        """Relative range shift (fraction) of each scenario."""
        return np.asarray(self._legacy_set().values("range")[:, 1], dtype=float)

    @property
    def abs_range_shift(self) -> np.ndarray:
        """Absolute range shift in mm of each scenario."""
        return np.asarray(self._legacy_set().values("range")[:, 0], dtype=float)

    @property
    def max_abs_range_shift(self) -> float:
        """Largest absolute range shift magnitude in mm."""
        return float(self._legacy_set().extent("range")[0])

    @property
    def max_rel_range_shift(self) -> float:
        """Largest relative range shift magnitude (fraction)."""
        return float(self._legacy_set().extent("range")[1])

    @property
    def tot_num_shift_scen(self) -> int:
        """Number of distinct setup shifts."""
        return self._matrad_positions()[2][1]

    @property
    def tot_num_range_scen(self) -> int:
        """Number of distinct range shifts."""
        return self._matrad_positions()[2][2]

    @property
    def tot_num_scen(self) -> int:
        """Total number of scenarios."""
        return len(self._legacy_set())

    @property
    def scen_for_prob(self) -> np.ndarray:
        """``(num_scenarios, 6)`` table ``[phase, x, y, z, abs range, rel range]``."""
        scen_set = self._legacy_set()
        return np.column_stack(
            [
                scen_set.ct_phases().astype(float),
                scen_set.values("setup"),
                scen_set.values("range"),
            ]
        )

    @property
    def scen_prob(self) -> np.ndarray:
        """Probability density of each scenario."""
        return np.asarray(self._legacy_set().probabilities, dtype=float)

    @property
    def scen_weight(self) -> np.ndarray:
        """Aggregation weight of each scenario."""
        return np.asarray(self._legacy_set().weights, dtype=float)

    @property
    def scen_mask(self) -> np.ndarray:
        """Return matRad's ``scenMask``: bool ``(available phases, shifts, ranges)``."""
        _, pos, shape = self._matrad_positions()
        mask = np.zeros(shape, dtype=bool)
        if pos.shape[0]:
            mask[tuple(pos.T)] = True
        return mask

    @property
    def linear_mask(self) -> np.ndarray:
        """Return matRad's ``linearMask``: ``(num_scenarios, 3)`` ``[phase, shift, range]``."""
        return self._matrad_positions()[1]

    def sub2scen_ix(self, ct_scen: int, shift_scen: int, range_shift_scen: int) -> int:
        """Linear index into :attr:`scen_mask` of the given subscripts."""
        return int(
            np.ravel_multi_index((ct_scen, shift_scen, range_shift_scen), self.scen_mask.shape)
        )

    def scen_num(self, full_scen_ix: int) -> int:
        """Flat scenario index of the scenario at linear mask index ``full_scen_ix``."""
        _, pos, shape = self._matrad_positions()
        sub = np.array(np.unravel_index(int(full_scen_ix), shape))
        hits = np.flatnonzero(np.all(pos == sub, axis=1))
        if hits.size == 0:
            raise IndexError(f"No scenario at mask index {full_scen_ix}")
        return int(hits[0])

    def update_scenarios(self) -> np.ndarray:
        """Regenerate the scenario set and return :attr:`scen_for_prob`."""
        self.invalidate()
        return self.scen_for_prob

    def list_all_scenarios(self) -> None:
        """Print a table of all scenarios."""
        print(self._legacy_set())

    def extract_single_scenario(self, scen_num: int) -> Any:
        """Model holding only scenario ``scen_num`` (an :class:`ExplicitScenarioModel`)."""
        from ._explicit import ExplicitScenarioModel  # noqa: PLC0415

        single = self._legacy_set().subset([int(scen_num)])
        return ExplicitScenarioModel(
            scenario_set=single,
            setup=self.setup,
            range=self.range,
            ct_phase=CTPhaseUncertainty(
                phases=[single.ct_phase(0)],
                probabilities=[
                    dict(self.ct_phase.pairs()).get(single.ct_phase(0), 1.0),
                ],
            ),
            wc_sigma=self.wc_sigma,
            distribution=self.distribution,
        )
