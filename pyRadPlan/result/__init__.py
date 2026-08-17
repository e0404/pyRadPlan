"""Provides functionality for handling treatment plan results."""

from ._quantity_distribution import (
    QuantityDistribution,
    create_distribution,
    validate_distribution,
)
from ._result import (
    PlanResult,
    create_result,
    validate_result,
)
from ._sequence import Sequence

__all__ = [
    "PlanResult",
    "QuantityDistribution",
    "Sequence",
    "create_distribution",
    "create_result",
    "validate_distribution",
    "validate_result",
]
