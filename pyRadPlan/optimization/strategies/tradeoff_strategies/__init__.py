"""Tradeoff strategy module providing different tradeoff strategies for pyRadPlan."""

from ._factory import register_tradeoff_strategy, get_available_tradeoff_strategies, get_tradeoff_strategy
from ._base_tradeoff_strategies import TradeoffStrategyBase
from ._single_plan import SinglePlan

register_tradeoff_strategy(SinglePlan)


__all__ = [
    "TradeoffStrategyBase",
    "SinglePlan",
    "register_tradeoff_strategy",
    "get_available_tradeoff_strategies",
    "get_tradeoff_strategy",
]
