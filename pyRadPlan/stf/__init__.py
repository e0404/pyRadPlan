"""Beam configurations and irradiation geometries."""

from .generators import get_available_generators, get_generator, register_generator
from ._steeringinformation import SteeringInformation, create_stf, validate_stf
from ._beam import Beam  # , create_beam, validate_beam
from ._ray import Ray
from ._beamlet import Beamlet, IonSpot, PhotonBixel
from ._rangeshifter import RangeShifter

__all__ = [
    "get_available_generators",
    "get_generator",
    "register_generator",
    "SteeringInformation",
    "create_stf",
    "validate_stf",
    "Beam",
    "Ray",
    "Beamlet",
    "IonSpot",
    "PhotonBixel",
    "RangeShifter",
]
