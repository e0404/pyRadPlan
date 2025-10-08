"""Beam geometry and steering information generators."""

from ._base import StfGeneratorBase
from ._externalbeam import StfGeneratorExternalBeamRayBixel, StfGeneratorExternalBeam
from ._ions import StfGeneratorIonSingleSpot, StfGeneratorIMPT
from ._photons import StfGeneratorPhotonIMRT, StfGeneratorPhotonCollimatedSquareFields
from ._vhee import StfGeneratorVHEE


from ._factory import get_generator, get_available_generators, register_generator

register_generator(StfGeneratorIMPT)
register_generator(StfGeneratorPhotonIMRT)
register_generator(StfGeneratorIonSingleSpot)
register_generator(StfGeneratorPhotonCollimatedSquareFields)
register_generator(StfGeneratorVHEE)

__all__ = [
    "StfGeneratorBase",
    "StfGeneratorExternalBeam",
    "StfGeneratorExternalBeamRayBixel",
    "StfGeneratorPhotonIMRT",
    "StfGeneratorPhotonCollimatedSquareFields",
    "StfGeneratorIonSingleSpot",
    "StfGeneratorIMPT",
    "StfGeneratorVHEE",
    "get_generator",
    "get_available_generators",
    "register_generator",
]
