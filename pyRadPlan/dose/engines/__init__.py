"""Dose calculation engines and algorithms."""

from ._base import DoseEngineBase
from ._svdpb import PhotonPencilBeamSVDEngine
from ._hongpb import ParticleHongPencilBeamEngine
from ._fredmc import ParticleFredMCEngine
from ._topasmc import ParticleTOPASMCEngine
from ._factory import get_engine, get_available_engines, register_engine
from ._base_ai_engine import AIEngine
from pyRadPlan.dose.ai_models import *

register_engine(PhotonPencilBeamSVDEngine)
register_engine(ParticleHongPencilBeamEngine)
register_engine(ParticleFredMCEngine)
register_engine(ParticleTOPASMCEngine)
register_engine(AIEngine)

__all__ = [
    "DoseEngineBase",
    "PhotonPencilBeamSVDEngine",
    "ParticleHongPencilBeamEngine",
    "ParticleFredMCEngine",
    "ParticleTOPASMCEngine",
    "get_engine",
    "get_available_engines",
    "register_engine",
]
