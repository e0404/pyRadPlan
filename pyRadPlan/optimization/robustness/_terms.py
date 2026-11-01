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
        Boolean mask of the objective's VOI on the dose grid (linear, numpy order).
    objective : Objective
        The objective.
    quantity : str
        Identifier of the quantity the objective is evaluated on.
    voi_type : str
        Type of the VOI (``"TARGET"``, ``"OAR"``, ...).
    """

    mask: np.ndarray
    objective: "Objective"
    quantity: str
    voi_type: str = "OAR"

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
    """

    operator: RobustnessOperator
    items: list[TermItem]
    scenario_indices: list[int] = field(init=False)
    entries: list[RobustTermEntry] = field(init=False)

    def __post_init__(self):
        self.scenario_indices = self.operator.scenario_indices
        self.entries = [item.entry for item in self.items]

    @property
    def quantities(self) -> list[str]:
        """Identifiers of the quantities the term reads, in order of first use."""
        return list(dict.fromkeys(item.quantity for item in self.items))


def voi_linear_mask(voi: Any) -> np.ndarray:
    """
    Boolean mask of a VOI over the linear (numpy-ordered) voxels of one 3D grid.

    A 4D structure contributes its first phase, as matRad optimizes on ``cst{i,4}{1}``.
    """
    mask = np.zeros(int(np.prod(voi.mask.GetSize()[:3])), dtype=np.bool_)
    mask[voi.scenario_indices(0, order="numpy")] = True
    return mask


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
        Per VOI its linear mask on the dose grid; computed from the VOIs if omitted.

    Returns
    -------
    list[RobustTerm]
        The terms in collection order.
    """
    if masks is None:
        masks = [voi_linear_mask(voi) for voi, _ in voi_objectives]
    if len(masks) != len(voi_objectives):
        raise ValueError("one mask per VOI is required")

    slots: list[Any] = []
    composite: dict[str, list[TermItem]] = {}
    for (voi, objectives), mask in zip(voi_objectives, masks):
        voi_type = getattr(voi, "voi_type", "OAR")
        for objective in objectives:
            item = TermItem(
                mask=mask, objective=objective, quantity=objective.quantity, voi_type=voi_type
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
