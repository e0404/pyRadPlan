"""Provides functionality for handling treatment plan results."""

from ._result import PlanResult, validate_result, create_result
from ._aperture_info import ApertureInfo
from ._quantity_distribution import (
    QuantityDistribution,
    validate_distribution,
    create_distribution,
)

__all__ = [
    "PlanResult",
    "QuantityDistribution",
    "ApertureInfo",
    "validate_result",
    "create_result",
    "validate_distribution",
    "create_distribution",
]
