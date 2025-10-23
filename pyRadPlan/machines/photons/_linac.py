from typing import Any, Optional, ClassVar, cast, List
import numpy as np
from numpy.typing import NDArray

from pydantic import (
    Field,
    field_validator,
    ValidatorFunctionWrapHandler,
    ValidationInfo,
    ValidationError,
)

from pyRadPlan.machines.base import ExternalBeamMachine
from ._svd_kernel import PhotonSVDKernel


class PhotonLINAC(ExternalBeamMachine):
    """
    Base Class for Photon LINAC-like Machines.

    Defines minimum meta-data a photon LINAC machine must hold
    Provides multiple data storage formats, currently supported
    - SVD Kernel Data (Bortfeld 1993, 10.1118/1.597070).

    Attributes
    ----------
    scd : float
        The source-to-collimator distance of the machine
    pb_kernels : dict[float, PhotonSVDKernel]
        Pencil-beam kernels for the machine
    of : np.ndarray
        Output factor data
    tpr : np.ndarray
        Tissue-to-phantom ratio data
    """

    # Defining some naming properties (must keep type annotations when overriding pydantic fields)
    name: str = Field(default="Generic", alias="machine")
    key: str = "photon_linac_generic"
    radiation_mode: str = Field(default="photons", pattern="^(photons)$", validate_default=True)
    _possible_radiation_modes: ClassVar[List[str]] = ["photons"]

    # Machine geometry
    scd: float = Field(ge=0.0, description="Source to collimator distance", alias="SCD")

    # Optionally we can have pencil-beam kernels
    pb_kernels: Optional[dict[float, PhotonSVDKernel]] = None

    # Commissioning data
    of: Optional[NDArray] = Field(default=None, description="Output Factor")
    tpr: Optional[NDArray] = Field(default=None, description="Tissue-to-phantom-Ratio")

    # TODO: Beam model descriptions for Monte Carlo

    @classmethod
    def _parse_tabulated_energy_data_from_mat(
        cls, tabulated_energy_data: dict, returned_data: dict
    ):
        """Parse the tabulated energy data from a matRad machine file."""

        super(PhotonLINAC, cls)._parse_tabulated_energy_data_from_mat(
            tabulated_energy_data, returned_data
        )

        # parse kernel
        # returned_data["scd"] = tabulated_energy_data.get("scd", 0.0)
        returned_data["pb_kernels"] = tabulated_energy_data

    @field_validator("pb_kernels", mode="wrap")
    @classmethod
    def validate_pb_kernels(
        cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo
    ) -> PhotonSVDKernel:
        try:
            return handler(v, info)
        except ValidationError as err:
            # If we have a dict, we try to convert it to a PhotonSVDKernel object
            if isinstance(v, dict):
                # This check tries to check if we have a matRad kenrel data set
                if "kernel" in v and "kernel_data" not in v:
                    try:
                        kernel = cast(dict, v["kernel"])
                        v["kernel_ssds"] = np.asarray(kernel.pop("SSD"), dtype=np.float64)
                        kernel_list = [
                            np.asarray(kernel.pop(f"kernel{i}"), dtype=np.float64)
                            for i in range(1, len(kernel) + 1)
                        ]
                        v["kernel_data"] = np.ascontiguousarray(
                            np.moveaxis(np.asarray(kernel_list), 0, 1)
                        )

                        # kernel dictionary by energy
                        v = {float(v["energy"]): v}
                    except Exception as exc:
                        raise ValueError("Could not parse matRad kernel data") from exc

                    return handler(v, info)

            # Otherwise we don't know what to do
            raise err

    def get_kernel_by_index(self, ix_energy: int) -> PhotonSVDKernel:
        """Get the pencil beam kernel for a specific energy index.

        Parameters
        ----------
        ix_energy : int
            Index of the energy level in the stored data

        Returns
        -------
        IonPencilBeamKernel
            Pencil beam kernel data
        """
        return self.get_kernel_by_energy(self.energies[ix_energy])

    def get_kernel_by_energy(self, energy: float) -> PhotonSVDKernel:
        """Get the pencil beam kernel for a specific energy value.

        Parameters
        ----------
        energy : float
            Energy value to search for

        Returns
        -------
        IonPencilBeamKernel
            Pencil beam kernel data
        """
        if self.pb_kernels is None:
            raise ValueError("No pencil beam kernels available for this machine.")
        return self.pb_kernels[energy]
