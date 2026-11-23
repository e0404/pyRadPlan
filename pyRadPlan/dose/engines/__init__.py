"""Dose calculation engines and algorithms."""

from ._base import DoseEngineBase
from ._svdpb import PhotonPencilBeamSVDEngine
from ._ccc import PhotonCollapsedConeEngine
from ._hongpb import ParticleHongPencilBeamEngine
from ._fredmc import ParticleFredMCEngine
from ._topasmc import ParticleTOPASMCEngine
from ._factory import get_engine, get_engine_class, get_available_engines, register_engine

register_engine(PhotonPencilBeamSVDEngine)
register_engine(PhotonCollapsedConeEngine)
register_engine(ParticleHongPencilBeamEngine)
register_engine(ParticleFredMCEngine)
register_engine(ParticleTOPASMCEngine)

__all__ = [
    "DoseEngineBase",
    "PhotonPencilBeamSVDEngine",
    "PhotonCollapsedConeEngine",
    "ParticleHongPencilBeamEngine",
    "ParticleFredMCEngine",
    "ParticleTOPASMCEngine",
    "get_engine",
    "get_engine_class",
    "get_available_engines",
    "register_engine",
]
