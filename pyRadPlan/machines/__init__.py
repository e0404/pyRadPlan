"""Machine definitions for external beam radiotherapy."""

from .base import (
    Machine,
    ExternalBeamMachine,
    InternalBeamMachine,
    register_machine,
)
from .photons import PhotonLINAC, PhotonSVDKernel
from .particles import (
    ParticlePencilBeamKernel,
    LateralCutOff,
    IonAccelerator,
    VHEEAccelerator,
)
from ._validate import validate_machine
from ._load import load_from_name, load_machine_from_mat, load_machine

register_machine(PhotonLINAC)
register_machine(IonAccelerator)
register_machine(VHEEAccelerator)

__all__ = [
    "Machine",
    "ExternalBeamMachine",
    "InternalBeamMachine",
    "PhotonLINAC",
    "PhotonSVDKernel",
    "ParticlePencilBeamKernel",
    "LateralCutOff",
    "IonAccelerator",
    "validate_machine",
    "load_from_name",
    "load_machine_from_mat",
    "load_machine",
    "register_machine",
]
