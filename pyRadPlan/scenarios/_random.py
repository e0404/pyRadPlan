"""Randomly sampled scenarios (matRad ``rndScen``)."""

from __future__ import annotations

from typing import Any, ClassVar, Optional

import numpy as np
from pydantic import Field

from ._base import ScenarioModel
from ._set import TAG_NOMINAL, AxisSpec, ScenarioSet


class RandomScenarioModel(ScenarioModel):
    """
    Randomly sampled scenarios (matRad ``rndScen``).

    ``n_samples`` realizations of the continuous components are drawn from the model's
    distribution and replicated for every CT phase (phases form the outer loop, all phases share
    the samples). Following matRad, the weight of a scenario is its phase probability divided by
    ``n_samples`` (the samples already follow the distribution); the probability is the density
    times the phase probability.

    The generated set is cached by the base class under a fingerprint of the model
    configuration, so a model with ``seed=None`` returns the same samples on every call to
    :meth:`scenarios` until the configuration changes or ``scenarios(refresh=True)`` /
    :meth:`invalidate` is used. A fixed ``seed`` makes every regeneration reproducible.

    Attributes
    ----------
    n_samples : int
        Number of random samples per CT phase.
    include_nominal : bool
        Replace the first sample by the nominal scenario.
    seed : int, optional
        Seed of the random generator.
    """

    name: ClassVar[str] = "Randomly sampled Scenarios"
    short_name: ClassVar[str] = "rndScen"
    legacy_aliases: ClassVar[dict[str, str]] = {"includeNominalScenario": "include_nominal"}

    n_samples: int = Field(default=10, ge=1)
    include_nominal: bool = False
    seed: Optional[int] = None

    def default_axes(self) -> tuple[AxisSpec, ...]:
        return ("ct_phase", ("setup", "range"))

    def _matrad_extra(self) -> dict[str, Any]:
        return {"nSamples": int(self.n_samples), "includeNominalScenario": self.include_nominal}

    def _generate(self) -> ScenarioSet:
        dist = self.distribution_object()
        rng = np.random.default_rng(self.seed)
        samples = np.asarray(dist.sample(self.n_samples, rng), dtype=float)
        if self.include_nominal:
            samples[0] = dist.mean()

        points: dict[str, np.ndarray] = {}
        start = 0
        for dim in dist.dimensions:
            points[dim.name] = samples[:, start : start + dim.num_components]
            start += dim.num_components

        phases = np.asarray(self.ct_phase.phases, dtype=float)
        probs = np.asarray(self.ct_phase.probabilities, dtype=float)
        num_phases = len(phases)
        tags = [set() for _ in range(self.n_samples * num_phases)]
        if self.include_nominal:
            for k in range(num_phases):
                tags[k * self.n_samples] = {TAG_NOMINAL}
        return self._make_set(
            {
                "ct_phase": np.repeat(phases, self.n_samples).reshape(-1, 1),
                **{name: np.tile(p, (num_phases, 1)) for name, p in points.items()},
            },
            tags=tags,
            weights=np.repeat(probs, self.n_samples) / self.n_samples,
        )
