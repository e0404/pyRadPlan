"""Beam geometry and steering information generators."""

from ._base import StfGeneratorBase
from ._externalbeam import StfGeneratorExternalBeamRayBixel, StfGeneratorExternalBeam
from ._ions import StfGeneratorIonSingleSpot, StfGeneratorIMPT
from ._photons import StfGeneratorPhotonIMRT, StfGeneratorPhotonCollimatedSquareFields

from ._factory import get_generator, get_available_generators, register_generator

register_generator(StfGeneratorIMPT)
register_generator(StfGeneratorPhotonIMRT)
register_generator(StfGeneratorIonSingleSpot)
register_generator(StfGeneratorPhotonCollimatedSquareFields)

__all__ = [
    "StfGeneratorBase",
    "StfGeneratorExternalBeam",
    "StfGeneratorExternalBeamRayBixel",
    "StfGeneratorPhotonIMRT",
    "StfGeneratorPhotonCollimatedSquareFields",
    "StfGeneratorIonSingleSpot",
    "StfGeneratorIMPT",
    "get_generator",
    "get_available_generators",
    "register_generator",
]
