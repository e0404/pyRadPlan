"""Placeholder for future implementation."""

from pydantic import Field, ValidationError
from typing import ClassVar, List, Optional

from .kernel import VHEEPencilBeamKernel
from ._base import ParticleAccelerator
from pyRadPlan.util.helpers import dl2ld


class VHEEAccelerator(ParticleAccelerator):
    """Machine Model for Very High Energy Electron (VHEE) Accelerators."""

    radiation_mode: str = Field(default="VHEE", pattern="^(VHEE)$", validate_default=True)
    _possible_radiation_modes: ClassVar[List[str]] = ["VHEE"]
    pb_kernels: Optional[dict[float, VHEEPencilBeamKernel]] = None

    @classmethod
    def _parse_tabulated_energy_data_from_mat(
        cls, tabulated_energy_data: dict, returned_data: dict
    ):
        """Parse the tabulated energy data from a matRad machine file."""

        super(VHEEAccelerator, cls)._parse_tabulated_energy_data_from_mat(
            tabulated_energy_data, returned_data
        )
        try:
            tmp = dl2ld(tabulated_energy_data)
            returned_data["pb_kernels"] = {
                entry["energy"]: VHEEPencilBeamKernel(**entry) for entry in tmp
            }
        except ValidationError:
            returned_data["pb_kernels"] = None
