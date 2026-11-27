"""Objective functions for treatment plan optimization."""

from ._objective import Objective
from ._std import DoseUniformity
from ._eud import EUD
from ._max_dvh import MaxDVH
from ._mean import MeanDose
from ._min_dvh import MinDVH
from ._squared_dev import SquaredDeviation
from ._squared_mimicking import SquaredMimicking
from ._squared_over import SquaredOverdosing
from ._squared_under import SquaredUnderdosing
from ._variance import Variance
from ._ntcp_polo import NTCPPOLO
from ._hellinger_polo import HellingerPOLO
from ._logsumexp_convpolo import LogSumExpConvPOLO
from ._hellinger_convpolo import HellingerConvPOLO

from ._factory import (
    get_available_objectives,
    get_objective,
    get_objectives_union,
    register_objective,
)

register_objective(SquaredDeviation)
register_objective(SquaredMimicking)
register_objective(SquaredOverdosing)
register_objective(SquaredUnderdosing)
register_objective(MeanDose)
register_objective(EUD)
register_objective(MinDVH)
register_objective(MaxDVH)
register_objective(DoseUniformity)
register_objective(Variance)
register_objective(NTCPPOLO)
register_objective(HellingerPOLO)
register_objective(LogSumExpConvPOLO)
register_objective(HellingerConvPOLO)

__all__ = [
    "Objective",
    "DoseUniformity",
    "EUD",
    "MaxDVH",
    "MeanDose",
    "MinDVH",
    "SquaredDeviation",
    "SquaredMimicking",
    "SquaredOverdosing",
    "SquaredUnderdosing",
    "Variance",
    "NTCPPOLO",
    "HellingerPOLO",
    "LogSumExpConvPOLO",
    "HellingerConvPOLO",
    "get_available_objectives",
    "get_objective",
    "get_objectives_union",
    "register_objective",
]
