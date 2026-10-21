"""Beam configurations and irradiation geometries."""

from .generators import (
    get_available_generators,
    get_generator,
    register_generator,
    resolve_iso_centers,
)
from ._steeringinformation import SteeringInformation, create_stf, validate_stf
from ._beam import Beam  # , create_beam, validate_beam
from ._ray import Ray
from ._beamlet import (
    Beamlet,
    IonSpot,
    PhotonBixel,
    FieldShape,
    FieldShapeAsBLD,
    FieldShapeAsMask,
    FieldShapeComposite,
)
from ._rangeshifter import RangeShifter
from ._presets import BeamAnglePreset, get_beam_angle_presets, register_beam_angle_preset

__all__ = [
    "get_available_generators",
    "get_generator",
    "register_generator",
    "resolve_iso_centers",
    "SteeringInformation",
    "create_stf",
    "validate_stf",
    "Beam",
    "Ray",
    "Beamlet",
    "IonSpot",
    "PhotonBixel",
    "FieldShape",
    "FieldShapeAsBLD",
    "FieldShapeAsMask",
    "FieldShapeComposite",
    "RangeShifter",
    "BeamAnglePreset",
    "get_beam_angle_presets",
    "register_beam_angle_preset",
]
