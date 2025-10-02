"""Scalarization strategy module providing different scalarization strategies for pyRadPlan."""

from ._factory import (
    register_scalarization_strategy,
    get_available_scalarization_strategies,
    get_scalarization_strategy,
)


from ._base_scalarization_strategies import ScalarizationStrategyBase
from ._weighted_sum import WeightedSum

register_scalarization_strategy(WeightedSum)


__all__ = [
    "ScalarizationStrategyBase",
    "WeightedSum",
    "register_scalarization_strategy",
    "get_available_scalarization_strategies",
    "get_scalarization_strategy",
]
