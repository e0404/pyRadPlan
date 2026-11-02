"""Robust terms: the objectives of a planning problem grouped by their robustness operator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional, Sequence

import numpy as np

from ...scenarios import ScenarioSet
from ._base import RobustnessOperator, RobustnessOptions, RobustTermEntry, ScenarioStack
from ._factory import get_robustness_operator, get_robustness_operator_class

if TYPE_CHECKING:
    from ..objectives import Objective


@dataclass(frozen=True)
class TermItem:
    """
    One objective of a robust term together with the voxels it acts on.

    Attributes
    ----------
    mask : np.ndarray
        Boolean mask of the objective's VOI on the dose grid (linear, numpy order); the first
        CT phase's for a 4D VOI.
    objective : Objective
        The objective.
    quantity : str
        Identifier of the quantity the objective is evaluated on.
    voi_type : str
        Type of the VOI (``"TARGET"``, ``"OAR"``, ...).
    phase_masks : tuple[np.ndarray, ...]
        Per CT phase the VOI's mask, if the phases have different contours.
    """

    mask: np.ndarray
    objective: "Objective"
    quantity: str
    voi_type: str = "OAR"
    phase_masks: tuple[np.ndarray, ...] = ()

    @property
    def entry(self) -> RobustTermEntry:
        """The operator entry of this item."""
        return RobustTermEntry(objective=self.objective, voi_type=self.voi_type)


@dataclass
class RobustTerm:
    """
    One function of the planning problem: an operator over its items.

    A term contributes one value and one jacobian row to the problem. Objective-scope
    operators act on a single item, composite-scope operators on every objective carrying the
    operator's name.

    Attributes
    ----------
    operator : RobustnessOperator
        Combines the items' per-scenario values.
    items : list[TermItem]
        The objectives the operator acts on, in collection order.
    scenario_indices : list[int]
        Flat set indices the term evaluates (the operator's), in stacking order.
    entries : list[RobustTermEntry]
        The operator entries of :attr:`items`.
    scenario_masks : list[list[np.ndarray]]
        Per item the voxels of each of :attr:`scenario_indices`. The same mask object where
        scenarios share a contour.
    """

    operator: RobustnessOperator
    items: list[TermItem]
    scenario_indices: list[int] = field(init=False)
    entries: list[RobustTermEntry] = field(init=False)
    scenario_masks: list[list[np.ndarray]] = field(init=False)

    def __post_init__(self):
        self.scenario_indices = self.operator.scenario_indices
        self.entries = [item.entry for item in self.items]
        self.scenario_masks = [self._item_masks(item) for item in self.items]

    def _item_masks(self, item: TermItem) -> list[np.ndarray]:
        if not item.phase_masks:
            return [item.mask] * len(self.scenario_indices)
        scenarios = self.operator.scenarios
        phases = (
            np.rint(scenarios.values("ct_phase")[self.scenario_indices, 0]).astype(int).tolist()
            if scenarios.has_dimension("ct_phase")
            else [0] * len(self.scenario_indices)
        )
        missing = sorted({p for p in phases if p >= len(item.phase_masks)})
        if missing:
            raise ValueError(
                f"The scenarios use CT phases {missing}, but the structure has contours for "
                f"{len(item.phase_masks)} phases"
            )
        if not self.operator.phase_contours:
            phases = phases[:1] * len(phases)
        return [item.phase_masks[p] for p in phases]

    @property
    def quantities(self) -> list[str]:
        """Identifiers of the quantities the term reads, in order of first use."""
        return list(dict.fromkeys(item.quantity for item in self.items))


def voi_linear_mask(voi: Any, phase: int = 0) -> np.ndarray:
    """Boolean mask of a VOI's CT phase over the linear (numpy-ordered) voxels of one 3D grid."""
    mask = np.zeros(int(np.prod(voi.mask.GetSize()[:3])), dtype=np.bool_)
    mask[voi.scenario_indices(phase, order="numpy")] = True
    return mask


def voi_phase_masks(voi: Any) -> tuple[np.ndarray, ...]:
    """
    Per CT phase the VOI's linear mask, as matRad's ``cst{i,4}{phase}``.

    Empty for a structure with one contour on all phases (a 3D one or a repeated one).
    """
    if voi.num_of_scenarios <= 1:
        return ()
    masks = tuple(voi_linear_mask(voi, phase) for phase in range(voi.num_of_scenarios))
    return () if all(np.array_equal(masks[0], m) for m in masks[1:]) else masks


def build_robust_terms(
    voi_objectives: Sequence[tuple[Any, Sequence["Objective"]]],
    scenarios: ScenarioSet,
    options: Optional[RobustnessOptions] = None,
    masks: Optional[Sequence[np.ndarray]] = None,
) -> list[RobustTerm]:
    """
    Group objectives into robust terms.

    Every objective whose operator acts on a single objective gets its own term; all
    objectives sharing a composite-scope operator form one term, placed at the position of the
    first of them. Scenario-aware objectives always use :class:`ScenarioStack`.

    Parameters
    ----------
    voi_objectives : Sequence[tuple[VOI, Sequence[Objective]]]
        Per VOI its objectives, in collection order.
    scenarios : ScenarioSet
        The scenario set the operators evaluate.
    options : RobustnessOptions, optional
        Problem-wide operator options.
    masks : Sequence[np.ndarray], optional
        Per VOI one linear mask on the dose grid for all scenarios; if omitted, computed from
        the VOIs, per CT phase for a 4D VOI whose contours differ.

    Returns
    -------
    list[RobustTerm]
        The terms in collection order.
    """
    phase_masks = [() for _ in voi_objectives]
    if masks is None:
        masks = [voi_linear_mask(voi) for voi, _ in voi_objectives]
        phase_masks = [voi_phase_masks(voi) for voi, _ in voi_objectives]
    if len(masks) != len(voi_objectives):
        raise ValueError("one mask per VOI is required")

    slots: list[Any] = []
    composite: dict[str, list[TermItem]] = {}
    for (voi, objectives), mask, per_phase in zip(voi_objectives, masks, phase_masks):
        voi_type = getattr(voi, "voi_type", "OAR")
        for objective in objectives:
            item = TermItem(
                mask=mask,
                objective=objective,
                quantity=objective.quantity,
                voi_type=voi_type,
                phase_masks=per_phase,
            )
            name = (
                ScenarioStack.name
                if getattr(objective, "scenario_aware", False)
                else objective.robustness
            )
            operator_cls = get_robustness_operator_class(name)
            if operator_cls.scope == "composite":
                if operator_cls.name not in composite:
                    composite[operator_cls.name] = []
                    slots.append(operator_cls.name)
                composite[operator_cls.name].append(item)
            else:
                operator = get_robustness_operator(operator_cls.name, scenarios, options)
                slots.append(RobustTerm(operator=operator, items=[item]))

    terms = []
    for slot in slots:
        if isinstance(slot, RobustTerm):
            terms.append(slot)
        else:
            operator = get_robustness_operator(slot, scenarios, options)
            terms.append(RobustTerm(operator=operator, items=composite[slot]))
    return terms
