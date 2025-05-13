"""
pyRadPlan Result Package.

------------------------

This package provides functionality for handling treatment plan results,
including generating, validating, and manipulating dose distributions
and plan metrics.

Modules:
    _result: Definition of the PlanResult model and related helpers.
    _quantity_distribution: Tools for validating and creating QuantityDistribution
                            models from input data.

Usage:
    from pyRadPlan.result import PlanResult, QuantityDistribution

For more details, see the documentation.
"""

from ._result import PlanResult, validate_result, create_result
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
