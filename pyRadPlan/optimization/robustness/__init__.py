"""
Robustness operators for optimization on scenario sets.

An operator decides how the per-scenario values of one objective (or of all objectives sharing
the operator) combine into the robust objective and provides the per-scenario cotangents for the
gradient. Objectives select one by name through :attr:`Objective.robustness`.
"""

from ._base import RobustnessOperator, RobustnessOptions, RobustTermEntry, ScenarioStack
from ._none import NoRobustness
from ._expected_value import ExpectedValue
from ._expected_quantity import ExpectedQuantity
from ._voxel_worst_case import VoxelWorstCase, VoxelWorstCaseInverted
from ._objective_worst_case import ObjectiveWorstCase
from ._composite_worst_case import CompositeWorstCase
from ._smooth_max import smooth_max
from ._factory import (
    available_robustness,
    get_robustness_operator,
    get_robustness_operator_class,
    register_robustness_operator,
    resolve_robustness_name,
)
from ._terms import RobustTerm, TermItem, build_robust_terms, voi_linear_mask, voi_phase_masks

register_robustness_operator(NoRobustness)
register_robustness_operator(ExpectedValue)
register_robustness_operator(ExpectedQuantity)
register_robustness_operator(VoxelWorstCase)
register_robustness_operator(VoxelWorstCaseInverted)
register_robustness_operator(ObjectiveWorstCase)
register_robustness_operator(CompositeWorstCase)
register_robustness_operator(ScenarioStack)

__all__ = [
    "RobustnessOperator",
    "RobustnessOptions",
    "RobustTermEntry",
    "ScenarioStack",
    "NoRobustness",
    "ExpectedValue",
    "ExpectedQuantity",
    "VoxelWorstCase",
    "VoxelWorstCaseInverted",
    "ObjectiveWorstCase",
    "CompositeWorstCase",
    "smooth_max",
    "available_robustness",
    "get_robustness_operator",
    "get_robustness_operator_class",
    "register_robustness_operator",
    "resolve_robustness_name",
    "RobustTerm",
    "TermItem",
    "build_robust_terms",
    "voi_linear_mask",
    "voi_phase_masks",
]
