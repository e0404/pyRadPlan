import numpy as np
import pytest
from pydantic import ValidationError

from pyRadPlan.machines.particles import ParticlePencilBeamKernel

N_DEPTHS = 7


def _kernel(**extra):
    depths = np.linspace(0.0, 30.0, N_DEPTHS)
    return ParticlePencilBeamKernel(
        energy=100.0, depths=depths, Z=np.ones(N_DEPTHS), sigma=np.ones(N_DEPTHS), **extra
    )


@pytest.mark.parametrize("transposed", [False, True])
def test_multi_gaussian_kernel_orientation(transposed):
    sigma_multi = np.arange(3 * N_DEPTHS, dtype=float).reshape(3, N_DEPTHS) + 1
    weight_multi = np.full((2, N_DEPTHS), 0.1)
    if transposed:
        sigma_multi, weight_multi = sigma_multi.T, weight_multi.T
    kernel = _kernel(sigma_multi=sigma_multi, weight_multi=weight_multi)
    assert kernel.sigma_multi.shape == (3, N_DEPTHS)
    assert kernel.weight_multi.shape == (2, N_DEPTHS)
    assert kernel.sigma_multi[1, 0] == pytest.approx(N_DEPTHS + 1)


def test_multi_gaussian_kernel_single_weight():
    kernel = _kernel(sigma_multi=np.ones((2, N_DEPTHS)), weight_multi=np.full(N_DEPTHS, 0.3))
    assert kernel.weight_multi.shape == (N_DEPTHS,)


def test_multi_gaussian_kernel_errors():
    with pytest.raises(ValidationError, match="depth data length"):
        _kernel(sigma_multi=np.ones((2, N_DEPTHS + 1)), weight_multi=np.ones(N_DEPTHS))
    with pytest.raises(ValidationError, match="one sigma more than weights"):
        _kernel(sigma_multi=np.ones((3, N_DEPTHS)), weight_multi=np.ones(N_DEPTHS))


N_ENERGIES = 6


def _matrad_fluence(n_species=2):
    """matRad-style ``spectra`` struct with one aggregate and one explicit species."""
    rng = np.random.default_rng(3)
    energies = np.geomspace(1.0, 400.0, N_ENERGIES)
    spectra = [rng.random((N_ENERGIES, N_DEPTHS)) for _ in range(n_species)]
    return {
        "spectra": {
            "Z": np.asarray([1, 6][:n_species]),
            "A": np.asarray([1.0, np.nan][:n_species], dtype=float),
            "fluenceSpectrum": spectra,
            "energyBin": [energies] * n_species,
            "fluenceDepth": [s.sum(axis=0) for s in spectra],
        }
    }


def test_matrad_fluence_struct_is_still_parsed():
    """The matRad import keeps its orientation and units."""
    data = _matrad_fluence()
    kernel = _kernel(Fluence=data)

    assert len(kernel.fluence_spectrum.fragments) == 2
    first = kernel.fluence_spectrum.fragments[0]
    assert first.Z == 1 and first.A == 1.0
    assert first.fluence_spectrum.shape == (N_ENERGIES, N_DEPTHS)
    assert np.array_equal(first.fluence_spectrum, data["spectra"]["fluenceSpectrum"][0])
    assert np.array_equal(first.energy, data["spectra"]["energyBin"][0])
    assert np.isnan(kernel.fluence_spectrum.fragments[1].A)


@pytest.mark.parametrize("by_alias", [False, True])
def test_kernel_with_fluence_spectrum_round_trips(by_alias):
    """model_validate(model_dump()) reconstructs a spectrum-bearing kernel unchanged."""
    kernel = _kernel(Fluence=_matrad_fluence(), let=np.arange(N_DEPTHS, dtype=float))

    restored = ParticlePencilBeamKernel.model_validate(kernel.model_dump(by_alias=by_alias))

    assert np.array_equal(restored.idd, kernel.idd)
    assert np.array_equal(restored.let, kernel.let)
    assert restored.sigma_multi is None
    assert restored.fluence_spectrum is not None
    for original, copy_ in zip(
        kernel.fluence_spectrum.fragments, restored.fluence_spectrum.fragments
    ):
        assert copy_.Z == original.Z
        assert np.array_equal([copy_.A], [original.A], equal_nan=True)
        assert np.array_equal(copy_.fluence_spectrum, original.fluence_spectrum)
        assert np.array_equal(copy_.energy, original.energy)
        assert np.array_equal(copy_.fluenceZ, original.fluenceZ)


def test_kernel_accepts_an_existing_fluence_spectrum_instance():
    spectrum = _kernel(Fluence=_matrad_fluence()).fluence_spectrum

    kernel = _kernel(fluence_spectrum=spectrum)

    assert kernel.fluence_spectrum is not None
    assert np.array_equal(
        kernel.fluence_spectrum.fragments[0].fluence_spectrum,
        spectrum.fragments[0].fluence_spectrum,
    )


def test_optional_kernel_arrays_stay_none():
    """An absent optional array must not become a 0-d NaN array."""
    kernel = ParticlePencilBeamKernel.model_validate(_kernel().model_dump())
    assert kernel.let is None
    assert kernel.alpha is None and kernel.beta is None
    assert kernel.weight_multi is None
