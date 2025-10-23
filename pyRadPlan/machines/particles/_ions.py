from typing_extensions import Self
from typing import ClassVar, List, Optional
from pydantic import Field


# import warnings
import numpy as np
from pydantic import (
    model_validator,
    ValidationError,
)
from numpydantic import NDArray, Shape
from ._base import ParticleAccelerator
from .kernel import IonPencilBeamKernel
from pyRadPlan.util.helpers import dl2ld


class IonAccelerator(ParticleAccelerator):
    """Machine Model for Ion Accelerators.

    Defines minimum meta-data an ion machine must hold
    Provides multiple data storage formats, currently supported
    - Pencil-Beam Kernels
    - Monte Carlo Beam Models.

    Attributes
    ----------
    sad : float
        The source-to-axis (-isocenter) distance of the machine
    """

    # Annotated overrides for pydantic fields
    radiation_mode: str = Field(
        default="protons", pattern="^(protons|helium|carbon|oxygen)$", validate_default=True
    )
    _possible_radiation_modes: ClassVar[List[str]] = ["protons", "helium", "carbon", "oxygen"]

    peak_positions: NDArray[Shape["1-*"], np.float64]
    pb_kernels: Optional[dict[float, IonPencilBeamKernel]] = None

    @classmethod
    def _parse_tabulated_energy_data_from_mat(
        cls, tabulated_energy_data: dict, returned_data: dict
    ):
        """Parse the tabulated energy data from a matRad machine file."""

        super(IonAccelerator, cls)._parse_tabulated_energy_data_from_mat(
            tabulated_energy_data, returned_data
        )

        # extrac required quantities
        if "offset" in tabulated_energy_data:
            returned_data["peak_positions"] = np.array(
                tabulated_energy_data["peakPos"], dtype=np.float64
            ) + np.array(tabulated_energy_data["offset"], dtype=np.float64)
        # Extract the pencil beam kernels
        # Pencil beam kernels are not requried so we may skip them
        # if they can't be validated
        try:
            tmp = dl2ld(tabulated_energy_data)
            returned_data["pb_kernels"] = {
                entry["energy"]: IonPencilBeamKernel(**entry) for entry in tmp
            }
        except ValidationError:
            returned_data["pb_kernels"] = None

    @model_validator(mode="after")
    def _check_machine(self) -> Self:
        """Validate the machine model for consistency."""

        if self.bams_to_iso_dist > self.sad:
            raise ValueError("BAMS to iso distance must be small than SAD.")

        if self.pb_kernels is not None:
            # Check that the peak_positions are consistent between peak_positions of the model
            # and the kernel peak positions with offset applied
            for i, energy in enumerate(self.energies):
                kernel = self.pb_kernels[energy]
                if not np.isclose(self.peak_positions[i], kernel.peak_pos + kernel.offset):
                    raise ValueError(
                        f"Peak position of the model and the kernel data for energy {energy} "
                        "are inconsistent."
                    )

        return self
