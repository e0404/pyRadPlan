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
from ._beam_sequence import BeamSequence, ControlPoint, MachineState, Segment
from ._sequence import Sequence
from ._aperture_info import ApertureGrid, ApertureInfo

__all__ = [
    "ApertureGrid",
    "ApertureInfo",
    "BeamSequence",
    "ControlPoint",
    "MachineState",
    "PlanResult",
    "QuantityDistribution",
    "Segment",
    "Sequence",
    "create_distribution",
    "create_result",
    "validate_distribution",
    "validate_result",
]
