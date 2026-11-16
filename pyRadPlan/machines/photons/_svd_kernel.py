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
    """Class representing a single photon beam kernel.

    The kernel is separated into a depth-dependent dose component and a radius-dependent
    lateral profile component. The depth-dependent dose component is represented by a set of
    exponential functions with different build-up coefficients (betas) and an effective
    attenuation coefficient. For each dose component, the radial lateral profile is represented by a set of
    kernel data at different source-to-surface distances (SSDs).

    Attributes
    ----------
    energy : float
        The nominal energy of the photon beam [MeV] (corresponds to the stated LINAC voltage [MV]).
    m : float
        The effective attenuation coefficient [1/mm] for the photon beam.
    penumbra : float
        The penumbra of an open field as FWHM at isocenter [mm].
    kernel_betas : np.ndarray, shape (n_components,)
        The build-up coefficients of the kernel depth dose components [1/mm], i.e. the
        exponents in ``exp(-beta * depth)``.
    kernel_ssds : np.ndarray, shape (n_ssds,)
        The source-to-surface distances (SSDs) at which the kernel data is defined [mm].
    kernel_pos : np.ndarray, shape (n_positions,)
        The radial positions at which the kernel data is defined [mm].
    kernel_data : np.ndarray, shape (n_ssds, n_components, n_positions)
        The radial lateral profile of the kernel components at different SSDs [1/mm^2].
    primary_fluence : np.ndarray, shape (n_fluence_points, 2)
        The relative photon fluence in the isocenter plane: radial distances from the
        central axis [mm], fluence relative to central axis.
    """

    # some basic kernel information
    energy: float = Field(ge=0.0, description="The energy of the photon beam in MeV.")
    m: float = Field(description="attenuation factor")
    penumbra: float = Field(
        default=5.0,
        description="The penumbra of an open field as FWHM at Isocenter",
        alias="penumbarFWHMatIso",
    )

    # Tabulated kernel data
    kernel_betas: NDArray[Shape["1-*"], np.float64] = Field(
        description="build-up coefficients (exponents) of the kernel components", alias="betas"
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
            return handler(v)
        except ValidationError:
            try:
                v = np.asarray(v)
                return handler(v)
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
