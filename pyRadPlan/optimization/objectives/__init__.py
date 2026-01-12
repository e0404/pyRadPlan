"""Objective functions for treatment plan optimization."""

from ._objective import Objective
from ._std import DoseUniformity
from ._eud import EUD
from ._max_dvh import MaxDVH
from ._mean import MeanDose
from ._min_dvh import MinDVH
from ._squared_dev import SquaredDeviation
from ._squared_over import SquaredOverdosing
from ._squared_under import SquaredUnderdosing

from ._factory import get_available_objectives, get_objective, register_objective

register_objective(SquaredDeviation)
register_objective(SquaredOverdosing)
register_objective(SquaredUnderdosing)
register_objective(MeanDose)
register_objective(EUD)
register_objective(MinDVH)
register_objective(MaxDVH)
register_objective(DoseUniformity)

__all__ = [
    "Objective",
    "DoseUniformity",
    "EUD",
    "MaxDVH",
    "MeanDose",
    "MinDVH",
    "SquaredDeviation",
    "SquaredOverdosing",
    "SquaredUnderdosing",
    "get_available_objectives",
    "get_objective",
    "register_objective",
]
