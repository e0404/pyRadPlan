import warnings
from typing import Any, Optional, Annotated, cast
from typing_extensions import Self
import numpy as np
from pydantic import (
    Field,
    StringConstraints,
    computed_field,
    field_validator,
    model_validator,
    ValidatorFunctionWrapHandler,
    ValidationInfo,
    ValidationError,
)
from numpydantic import NDArray, Shape
from pyRadPlan.core import PyRadPlanBaseModel
from ._base import ExternalBeamMachine


class PhotonSVDKernel(PyRadPlanBaseModel):
    """Kernel data for photon beams.

    Attributes
    ----------
    energy : float
        The maximum energy of the photon beam in MeV (corresponds to the LINAC voltage in MV).
    """

    # some basic kernel information
    energy: float = Field(ge=0.0, description="The energy of the photon beam in MeV.")
    m: float = Field(description="attuneation factor")
    penumbra: float = Field(
        default=5.0,
        description="The penumbra of an open field as FWHM at Isocenter",
        alias="penumbarFWHMatIso",
    )

    # Tabulated kernel data
    kernel_betas: NDArray[Shape["1-*"], np.float64] = Field(
        description="weighting factors for kernel components", alias="betas"
    )
    kernel_ssds: NDArray[Shape["1-*"], np.float64] = Field(description="Kernel SSD points")
    kernel_pos: NDArray[Shape["1-*"], np.float64] = Field(description="Kernel position")
    kernel_data: NDArray[Shape["1-*,1-*, 1-*"], np.float64] = Field(
        description="Spatial grid of kernels"
    )

    primary_fluence: NDArray[Shape["1-*,2"], np.float64] = Field(description="Primary fluence")

    @computed_field(return_type=int)
    @property
    def num_kernel_components(self):
        return self.kernel_data.shape[1]

    def get_kernels_at_ssd(self, ssd: float) -> NDArray[Shape["1-*, 1-*"], np.float64]:
        """Get the kernel components at a specific SSD value.

        Parameters
        ----------
        ssd : float
            The SSD value to search for.

        Returns
        -------
        NDArray[Shape["1-*, 1-*"], np.float64]
            The kernels at the specified SSD value.
        """

        # For performance reasons we slice kernel_data at the closest SSD value
        if ssd < self.kernel_ssds.min() or ssd > self.kernel_ssds.max():
            warnings.warn(f"SSD value {ssd} is not in the kernel SSD range.")

        ix = np.argmin(np.abs(self.kernel_ssds - ssd))
        return self.kernel_data[ix]

    def kernel_interpolator(self):
        pass

    @field_validator("kernel_betas", "kernel_ssds", "kernel_pos", "primary_fluence", mode="wrap")
    @classmethod
    def validate_kernel_data_shapes(
        cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo
    ) -> NDArray:
        try:
            return handler(v, info)
        except ValidationError:
            try:
                v = np.asarray(v)
                return handler(v, info)
            except Exception as exc:
                raise ValueError("Input not an array") from exc

    @model_validator(mode="after")
    def validate_kernel_data(self) -> Self:
        if self.kernel_data.shape[1] != self.kernel_betas.shape[0]:
            raise ValueError("Kernel data and kernel betas do not match in length")

        if self.kernel_data.shape[0] != self.kernel_ssds.shape[0]:
            raise ValueError("Kernel data and kernel ssds do not match in length")

        if self.kernel_data.shape[2] != self.kernel_pos.shape[0]:
            raise ValueError("Kernel data and kernel positions do not match in length")

        return self


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

    radiation_mode: Annotated[str, StringConstraints(pattern="photons")] = Field(
        default="photons", validate_default=True
    )

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
                        # matRad kernels are named consequtively kernel1, kernel2, ...
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
            # return None
        return self.pb_kernels[energy]
