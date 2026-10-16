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

from pyRadPlan.biology.tissues._base import (
    DEFAULT_ALPHA_X,
    DEFAULT_BETA_X,
    TissueProperties,
)
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

        # extract required quantities
        if "offset" in tabulated_energy_data:
            returned_data["peak_positions"] = np.array(
                tabulated_energy_data["peakPos"], dtype=np.float64
            ) + np.array(tabulated_energy_data["offset"], dtype=np.float64)
        # Extract the pencil beam kernels
        # Pencil beam kernels are not required so we may skip them
        # if they can't be validated
        # Kernel sets declared in the metadata may keep their data in fields of their own
        # (e.g. a second set of alpha/beta kernels), which the kernel model stores apart.
        from pyRadPlan.biology.models._kernel_sets import (  # noqa: PLC0415 - avoids an import cycle
            extra_kernel_fields,
        )

        known = set(IonPencilBeamKernel.model_fields) | {
            f.alias for f in IonPencilBeamKernel.model_fields.values() if f.alias
        }
        extra = extra_kernel_fields(returned_data.get("kernelSets"), known)
        try:
            tmp = dl2ld(tabulated_energy_data)
            for entry in tmp:
                additional = {name: entry.pop(name) for name in extra if name in entry}
                if additional:
                    entry["additional_data"] = additional
            returned_data["pb_kernels"] = {
                entry["energy"]: IonPencilBeamKernel(**entry) for entry in tmp
            }
        except ValidationError:
            returned_data["pb_kernels"] = None

    @classmethod
    def _parse_meta_data_from_mat(cls, meta_data: dict, returned_data: dict):
        """Parse the tabulated energy data from a matRad machine file."""

        super(IonAccelerator, cls)._parse_meta_data_from_mat(meta_data, returned_data)
        if meta_data.get("tissueParameters") is None:
            # check if pb_kernels are present and extract tissue properties from them
            if returned_data.get("pb_kernels") is not None:
                alpha_x = np.asarray(
                    [
                        returned_data["pb_kernels"][returned_data["energies"][0]].alpha_x
                        if returned_data["pb_kernels"][returned_data["energies"][0]].alpha_x
                        is not None
                        else DEFAULT_ALPHA_X
                    ]
                )
                beta_x = np.asarray(
                    [
                        returned_data["pb_kernels"][returned_data["energies"][0]].beta_x
                        if returned_data["pb_kernels"][returned_data["energies"][0]].beta_x
                        is not None
                        else DEFAULT_BETA_X
                    ]
                )
                pairs = np.vstack([alpha_x, beta_x]).T
                returned_data["tissue_properties"] = [
                    TissueProperties.from_base_data(alpha_x=a, beta_x=b) for a, b in pairs
                ]

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
