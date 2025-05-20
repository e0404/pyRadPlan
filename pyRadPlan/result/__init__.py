"""Provides functionality for handling treatment plan results."""

from ._result import PlanResult, ApertureInfo, validate_result, create_result
from ._quantity_distribution import (
    QuantityDistribution,
    validate_distribution,
    create_distribution,
)

__all__ = [
    "PlanResult",
    "QuantityDistribution",
    "validate_result",
    "create_result",
    "validate_distribution",
    "create_distribution",
]
