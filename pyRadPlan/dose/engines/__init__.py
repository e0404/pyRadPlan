"""Dose calculation engines and algorithms."""

from importlib.util import find_spec

from ._base import DoseEngineBase
from ._svdpb import PhotonPencilBeamSVDEngine
from ._hongpb import ParticleHongPencilBeamEngine
from ._fredmc import ParticleFredMCEngine
from ._topasmc import ParticleTOPASMCEngine
from ._pyradmc import PhotonPyRadMCEngine
from ._factory import get_engine, get_engine_class, get_available_engines, register_engine

register_engine(PhotonPencilBeamSVDEngine)
register_engine(ParticleHongPencilBeamEngine)
register_engine(ParticleFredMCEngine)
register_engine(ParticleTOPASMCEngine)

# pyRadMC is an optional dependency (pip install pyRadPlan[pyradmc])
if find_spec("pyradmc") is not None:
    register_engine(PhotonPyRadMCEngine)

__all__ = [
    "DoseEngineBase",
    "PhotonPencilBeamSVDEngine",
    "ParticleHongPencilBeamEngine",
    "ParticleFredMCEngine",
    "ParticleTOPASMCEngine",
    "PhotonPyRadMCEngine",
    "get_engine",
    "get_engine_class",
    "get_available_engines",
    "register_engine",
]
