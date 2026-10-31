"""
Robustness operators for optimization on scenario sets.

An operator decides how the per-scenario values of one objective (or of all objectives sharing
the operator) combine into the robust objective and provides the per-scenario cotangents for the
gradient. Objectives select one by name through :attr:`Objective.robustness`.
"""

from ._base import RobustnessOperator, RobustnessOptions, RobustTermEntry, ScenarioStack
from ._none import NoRobustness
from ._expected_value import ExpectedValue
from ._factory import (
    available_robustness,
    get_robustness_operator,
    get_robustness_operator_class,
    register_robustness_operator,
    resolve_robustness_name,
)

register_robustness_operator(NoRobustness)
register_robustness_operator(ExpectedValue)
register_robustness_operator(ScenarioStack)

__all__ = [
    "RobustnessOperator",
    "RobustnessOptions",
    "RobustTermEntry",
    "ScenarioStack",
    "NoRobustness",
    "ExpectedValue",
    "available_robustness",
    "get_robustness_operator",
    "get_robustness_operator_class",
    "register_robustness_operator",
    "resolve_robustness_name",
]
