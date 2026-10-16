import numpy as np
import pytest

from pyRadPlan.machines.particles._beam_energy import (
    ChargedEnergySpectrumGaussian,
    ChargedEnergySpectrumHistogram,
    energy_spectrum_from_dict,
)


def test_gaussian_is_the_default_type():
    spectrum = energy_spectrum_from_dict({"mean": 100.0, "sigma": 1.0})
    assert isinstance(spectrum, ChargedEnergySpectrumGaussian)
    assert spectrum.sigma_absolute == pytest.approx(1.0)


def test_histogram_moments():
    energy = np.linspace(90.0, 110.0, 401)
    sigma_absolute = 2.0
    weight = np.exp(-0.5 * ((energy - 100.0) / sigma_absolute) ** 2) * 0.37
    spectrum = energy_spectrum_from_dict(
        {"energy_MeVpN": energy, "weight": weight, "type": "histogram"}
    )

    assert isinstance(spectrum, ChargedEnergySpectrumHistogram)
    assert spectrum.mean == pytest.approx(100.0)
    assert spectrum.sigma_absolute == pytest.approx(sigma_absolute, rel=1e-3)
    assert spectrum.sigma == pytest.approx(100 * sigma_absolute / 100.0, rel=1e-3)
    assert spectrum.sigma_relative == pytest.approx(spectrum.sigma / 100)
    assert spectrum.fwhm == pytest.approx(2.35482 * sigma_absolute, rel=1e-3)


@pytest.mark.parametrize(
    "energy, weight",
    [([1.0, 2.0, 3.0], [1.0, 1.0]), ([1.0, 2.0], [0.0, 0.0]), ([1.0, 2.0], [1.0, -1.0])],
)
def test_histogram_rejects_invalid_data(energy, weight):
    with pytest.raises(ValueError):
        ChargedEnergySpectrumHistogram(energy_MeVpN=energy, weight=weight)


def test_unknown_spectrum_type():
    with pytest.raises(ValueError, match="Unsupported energy spectrum type"):
        energy_spectrum_from_dict({"type": "lorentzian"})
