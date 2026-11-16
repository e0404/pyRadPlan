from typing import Any, Literal

import numpy as np
from numpydantic import NDArray, Shape
from pydantic import (
    Field,
    field_validator,
    model_validator,
)

from pyRadPlan.core import PyRadPlanBaseModel

#: Factor between the standard deviation and the FWHM of a Gaussian.
_SIGMA_TO_FWHM = 2.35482


class ChargedEnergySpectrum(PyRadPlanBaseModel):
    """
    Energy spectrum of a charged particle pencil beam.

    Every spectrum provides its ``mean`` energy, its relative spread ``sigma`` in percent
    and the derived ``sigma_relative``, ``sigma_absolute`` and ``fwhm``, which is what the
    Monte Carlo engines read.
    """


class ChargedEnergySpectrumGaussian(ChargedEnergySpectrum):
    """
    Energy spectrum data for charged particle pencil beam kernels.

    Notes
    -----
    The energy spectrum is defined as a Gaussian distribution with a mean
    energy and a relative spread in percent.
    """

    type: Literal["gaussian"] = "gaussian"

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
        return _SIGMA_TO_FWHM * self.sigma_absolute


class ChargedEnergySpectrumHistogram(ChargedEnergySpectrum):
    """
    Energy spectrum given as a histogram of weights over energy bins.

    Notes
    -----
    The weights need not be normalized. ``mean`` and ``sigma`` are the weighted mean and
    standard deviation of the histogram, so a consumer expecting a Gaussian spectrum
    receives the one with the same first two moments.
    """

    type: Literal["histogram"] = "histogram"

    energy: NDArray[Shape["*"], np.float64] = Field(
        alias="energy_MeVpN", description="Energies of the histogram bins in MeV/u"
    )
    weight: NDArray[Shape["*"], np.float64] = Field(description="Weight of each energy bin")

    @field_validator("energy", "weight", mode="before")
    @classmethod
    def _to_1d_array(cls, v: Any) -> Any:
        return np.asarray(v, dtype=np.float64).reshape(-1)

    @model_validator(mode="after")
    def _check_histogram(self) -> "ChargedEnergySpectrumHistogram":
        if self.energy.shape != self.weight.shape:
            raise ValueError(
                f"Energy spectrum histogram has {self.energy.size} energies but "
                f"{self.weight.size} weights."
            )
        if np.any(self.weight < 0) or not np.sum(self.weight) > 0:
            raise ValueError("Energy spectrum histogram weights must be >= 0 and not all zero.")
        return self

    @property
    def mean(self) -> float:
        """Weighted mean energy of the histogram."""
        return float(np.average(self.energy, weights=self.weight))

    @property
    def sigma_absolute(self) -> float:
        """Weighted standard deviation of the energy."""
        variance = np.average((self.energy - self.mean) ** 2, weights=self.weight)
        return float(np.sqrt(variance))

    @property
    def sigma_relative(self) -> float:
        """Standard deviation relative to the mean energy."""
        return self.sigma_absolute / self.mean

    @property
    def sigma(self) -> float:
        """Relative spread in percent, as :attr:`ChargedEnergySpectrumGaussian.sigma`."""
        return 100.0 * self.sigma_relative

    @property
    def fwhm(self) -> float:
        """Full width at half maximum of the Gaussian with the histogram's moments."""
        return _SIGMA_TO_FWHM * self.sigma_absolute


def energy_spectrum_from_dict(data: dict) -> ChargedEnergySpectrum:
    """
    Create the energy spectrum of a matRad ``energySpectrum`` entry by its ``type``.

    Entries without a ``type`` are Gaussian, as matRad's default.
    """
    spectrum_type = str(data.get("type", "gaussian"))
    if spectrum_type == "gaussian":
        return ChargedEnergySpectrumGaussian(**data)
    if spectrum_type == "histogram":
        return ChargedEnergySpectrumHistogram(**data)
    raise ValueError(f"Unsupported energy spectrum type {spectrum_type!r}.")
