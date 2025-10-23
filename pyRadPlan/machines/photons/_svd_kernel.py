import warnings
from typing import Any
from typing_extensions import Self
import numpy as np
from pydantic import (
    Field,
    computed_field,
    field_validator,
    model_validator,
    ValidatorFunctionWrapHandler,
    ValidationInfo,
    ValidationError,
)
from numpydantic import NDArray, Shape
from pyRadPlan.core import PyRadPlanBaseModel


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
