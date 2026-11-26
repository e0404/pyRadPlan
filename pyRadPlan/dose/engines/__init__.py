"""Dose calculation engines and algorithms."""

from ._base import DoseEngineBase
from ._svdpb import PhotonPencilBeamSVDEngine
from ._hongpb import ParticleHongPencilBeamEngine
from ._fredmc import ParticleFredMCEngine
from ._topasmc import ParticleTOPASMCEngine
from ._factory import get_engine, get_engine_class, get_available_engines, register_engine
from ._base_ai_beamlet import AIBeamletEngine

register_engine(PhotonPencilBeamSVDEngine)
register_engine(ParticleHongPencilBeamEngine)
register_engine(ParticleFredMCEngine)
register_engine(ParticleTOPASMCEngine)
register_engine(AIBeamletEngine)

__all__ = [
    "DoseEngineBase",
    "PhotonPencilBeamSVDEngine",
    "ParticleHongPencilBeamEngine",
    "ParticleFredMCEngine",
    "ParticleTOPASMCEngine",
    "AIBeamletEngine",
    "get_engine",
    "get_engine_class",
    "get_available_engines",
    "register_engine",
]
