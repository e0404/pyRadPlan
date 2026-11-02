"""Robustness operators: how per-scenario objective values combine into a robust objective."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Literal, Optional, Sequence

import numpy as np
import array_api_compat

from ...core.xp_utils.typing import Array
from ...scenarios import ScenarioSet

if TYPE_CHECKING:
    from ..objectives import Objective


@dataclass(frozen=True)
class RobustnessOptions:
    """
    Options shared by all robustness operators of a planning problem.

    Attributes
    ----------
    max_approximation : {"logsumexp", "pnorm", "none"}
        Smooth approximation of the maximum over scenarios used by the worst-case operators
        (matRad's ``useMaxApprox``). ``"none"`` is the exact maximum with a one-hot gradient.
    p_norm : float
        Exponent of the ``"pnorm"`` approximation (matRad's ``p``).
    """

    max_approximation: Literal["logsumexp", "pnorm", "none"] = "logsumexp"
    p_norm: float = 30.0


@dataclass(frozen=True)
class RobustTermEntry:
    """
    One objective an operator acts on.

    Attributes
    ----------
    objective : Objective
        The objective (``priority`` is applied by the operator).
    voi_type : str
        Type of the structure the objective belongs to (``"TARGET"``, ``"OAR"``, ...), needed
        by the voxel-wise worst-case operators.
    """

    objective: "Objective"
    voi_type: str = "OAR"


class RobustnessOperator(ABC):
    """
    Combine the per-scenario values of one or several objectives into a robust objective.

    An operator receives, for every entry it acts on, the quantity values of the entry's voxels
    stacked over the operator's scenarios (shape ``(S, n)``) and returns the robust objective
    value and the cotangents ``dF / dq_s`` per scenario, so that the planning problem can
    back-project each scenario's cotangent through its own influence matrix. Priorities are
    applied inside the operator.

    An operator with :attr:`phase_contours` evaluates every scenario on the contour of its own
    CT phase, so the voxel counts can differ between scenarios and an entry's values are then a
    list with one array per scenario instead of a stack. All others combine the scenarios per
    voxel and get a stack on the contour of the first scenario's phase.

    Parameters
    ----------
    scenarios : ScenarioSet
        The scenario set of the dose influence data.
    options : RobustnessOptions, optional
        Problem-wide operator options.

    Attributes
    ----------
    name : ClassVar[str]
        Canonical name used in :attr:`Objective.robustness`.
    aliases : ClassVar[tuple[str, ...]]
        Alternative names accepted on input (e.g. matRad's).
    matrad_name : ClassVar[Optional[str]]
        Name written to matRad's ``robustness`` property; ``None`` if matRad has no equivalent.
    scope : ClassVar[{"objective", "composite"}]
        Whether an instance acts on one objective or on all objectives carrying this name.
    phase_contours : ClassVar[bool]
        Whether the operator evaluates each scenario on the contour of its CT phase.
    """

    name: ClassVar[str]
    aliases: ClassVar[tuple[str, ...]] = ()
    matrad_name: ClassVar[Optional[str]] = None
    scope: ClassVar[Literal["objective", "composite"]] = "objective"
    phase_contours: ClassVar[bool] = False

    def __init__(self, scenarios: ScenarioSet, options: Optional[RobustnessOptions] = None):
        self.scenarios = scenarios
        self.options = options if options is not None else RobustnessOptions()
        self._scenario_indices = self._select_scenarios()
        self._weights = np.asarray(self.scenarios.weights, dtype=float)[self._scenario_indices]

    def _select_scenarios(self) -> list[int]:
        """Flat set indices the operator evaluates; by default every scenario with weight."""
        weights = np.asarray(self.scenarios.weights, dtype=float)
        return [int(i) for i in np.flatnonzero(weights > 0)] or list(range(len(self.scenarios)))

    @property
    def scenario_indices(self) -> list[int]:
        """Flat set indices the operator needs, in the order of the stacked values."""
        return list(self._scenario_indices)

    @property
    def weights(self) -> np.ndarray:
        """Set weights of :attr:`scenario_indices`."""
        return self._weights

    @property
    def num_scenarios(self) -> int:
        """Number of scenarios the operator evaluates."""
        return len(self._scenario_indices)

    def _check(self, entries: Sequence[RobustTermEntry], values: Sequence[Array]) -> None:
        if len(entries) != len(values):
            raise ValueError("one stacked value array per entry is required")
        if self.scope == "objective" and len(entries) != 1:
            raise ValueError(f"operator {self.name!r} acts on exactly one objective")
        for value in values:
            count = len(value) if isinstance(value, (list, tuple)) else value.shape[0]
            if count != self.num_scenarios:
                raise ValueError(
                    f"values must be stacked over the {self.num_scenarios} scenarios of the "
                    f"operator, got {count}"
                )

    @abstractmethod
    def evaluate(self, entries: Sequence[RobustTermEntry], values: Sequence[Array]) -> Array:
        """
        Robust objective value.

        Parameters
        ----------
        entries : Sequence[RobustTermEntry]
            The objectives acted on (one for objective-scope operators).
        values : Sequence[Array]
            Per entry the quantity values of its voxels stacked over :attr:`scenario_indices`,
            shape ``(S, n_entry)``.

        Returns
        -------
        Array
            Scalar value, priorities included.
        """

    @abstractmethod
    def gradient(self, entries: Sequence[RobustTermEntry], values: Sequence[Array]) -> list[Array]:
        """
        Cotangents of the robust objective w.r.t. the per-scenario quantity values.

        Parameters
        ----------
        entries : Sequence[RobustTermEntry]
            The objectives acted on.
        values : Sequence[Array]
            As in :meth:`evaluate`.

        Returns
        -------
        list[Array]
            Per entry ``dF / dq_s`` on its voxels, shape ``(S, n_entry)``, priorities included.
        """


class ScenarioStack(RobustnessOperator):
    """
    Pass the stacked scenarios through a scenario-aware objective.

    Scenario-aware objectives (:attr:`Objective.scenario_aware`) evaluate the whole set at once,
    e.g. the variance over scenarios; they receive the ``(S, n)`` values and the weights.
    """

    name = "scenario_stack"

    def evaluate(self, entries, values):
        self._check(entries, values)
        objective = entries[0].objective
        return objective.priority * objective.compute_objective(values[0], weights=self.weights)

    def gradient(self, entries, values):
        self._check(entries, values)
        objective = entries[0].objective
        xp = array_api_compat.array_namespace(values[0])
        grad = objective.compute_gradient(values[0], weights=self.weights)
        return [xp.astype(objective.priority * grad, values[0].dtype, copy=False)]
