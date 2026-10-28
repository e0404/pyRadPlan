"""Machine definitions for external beam radiotherapy."""

from .base import (
    Machine,
    ExternalBeamMachine,
    InternalBeamMachine,
    BeamLimitingDevice,
    MLC,
    Jaw,
    create_bld,
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
from ._load import (
    find_machine_file,
    load_from_name,
    load_machine_from_mat,
    load_machine,
    get_available_machines,
    add_machine_file,
    remove_machine_file,
    user_machine_dir,
)

register_machine(PhotonLINAC)
register_machine(IonAccelerator)
register_machine(VHEEAccelerator)

__all__ = [
    "Machine",
    "ExternalBeamMachine",
    "InternalBeamMachine",
    "BeamLimitingDevice",
    "MLC",
    "Jaw",
    "create_bld",
    "PhotonLINAC",
    "PhotonSVDKernel",
    "ParticlePencilBeamKernel",
    "LateralCutOff",
    "IonAccelerator",
    "validate_machine",
    "find_machine_file",
    "load_from_name",
    "load_machine_from_mat",
    "load_machine",
    "get_available_machines",
    "add_machine_file",
    "remove_machine_file",
    "user_machine_dir",
    "register_machine",
]
