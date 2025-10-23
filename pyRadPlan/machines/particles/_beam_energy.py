from typing import Final

from pydantic import (
    Field,
)

from pyRadPlan.core import PyRadPlanBaseModel


class ChargedEnergySpectrum(PyRadPlanBaseModel):
    pass


class ChargedEnergySpectrumGaussian(ChargedEnergySpectrum):
    """
    Energy spectrum data for charged particle pencil beam kernels.

    Notes
    -----
    The energy spectrum is defined as a Gaussian distribution with a mean
    energy and a relative spread in percent.
    """

    type: Final[str] = "gaussian"

    mean: float = Field(ge=0.0)
    sigma: float = Field(
        ge=0.0, description="Relative spread as percentage sigma of the Gaussian energy spectrum"
    )

    @property
    def sigma_relative(self) -> float:
        """Relative spread of the energy spectrum."""
        return self.sigma / 100.0

    @property
    def sigma_absolute(self) -> float:
        """Absolute sigma of the energy spectrum."""
        return self.sigma_relative * self.mean

    @property
    def fwhm(self) -> float:
        """Full width at half maximum of the energy spectrum."""
        return 2.35482 * self.sigma_absolute
