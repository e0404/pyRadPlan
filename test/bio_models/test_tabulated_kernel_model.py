import numpy as np
import pytest
import array_api_strict as xp

from pyRadPlan.bio_models._base import BiologicalModelBase
from pyRadPlan.bio_models.models.tabulated_rbe_models import TabulatedAlphaBetaModel
from pyRadPlan.machines.particles import ParticlePencilBeamKernel


N_DEPTHS = 6
N_ENERGIES = 25


@pytest.fixture
def synthetic_kernel():
    """Pencil-beam kernel with a synthetic fragment fluence spectrum (H, C and electrons)."""
    rng = np.random.default_rng(42)
    depths = np.linspace(0.0, 50.0, N_DEPTHS)
    energies = np.geomspace(1.0, 400.0, N_ENERGIES)

    def spectrum():
        return rng.random((N_ENERGIES, N_DEPTHS))

    spectra = [spectrum(), spectrum(), spectrum()]
    fluence = {
        "spectra": {
            "Z": np.asarray([1, 6, -1]),
            "A": np.asarray([1.0, 12.0, np.nan]),
            "fluenceSpectrum": spectra,
            "energyBin": [energies, energies, energies],
            "fluenceDepth": [s.sum(axis=0) for s in spectra],
        }
    }
    return ParticlePencilBeamKernel(
        energy=100.0,
        depths=depths,
        Z=np.ones(N_DEPTHS),
        sigma=np.ones(N_DEPTHS),
        Fluence=fluence,
    )


def _expected_dose_averaged(model, kernel, quantity, transform=lambda x: x):
    """Reference: dose-weighted mean of the table quantity over energies and fragments."""
    numerator = np.zeros(N_DEPTHS)
    denominator = np.zeros(N_DEPTHS)
    for frag in kernel.fluence_spectrum.fragments:
        if frag.Z <= 0:
            continue
        sp_ix = np.flatnonzero(model._sp_table["fragments_AZ"][:, 1] == frag.Z)[0]
        q_ix = np.flatnonzero(model._q_table["fragments_AZ"][:, 1] == frag.Z)[0]
        sp = np.interp(
            frag.energy, model._sp_table["energies"][sp_ix], model._sp_table["dE_dx"][sp_ix]
        )
        q = transform(
            np.interp(
                frag.energy, model._q_table["energies"][q_ix], model._q_table[quantity][q_ix]
            )
        )
        denominator += (sp[:, None] * frag.fluence_spectrum).sum(axis=0)
        numerator += (q[:, None] * sp[:, None] * frag.fluence_spectrum).sum(axis=0)
    return numerator / denominator


def test_TabulatedAlphaBetaModel_constructor():
    tabulated_alpha_beta_model = TabulatedAlphaBetaModel()
    assert isinstance(tabulated_alpha_beta_model, BiologicalModelBase)
    assert tabulated_alpha_beta_model.model == "dose_average_alpha_beta"
    assert tabulated_alpha_beta_model.quantities_in_table == ["alpha", "beta"]
    assert tabulated_alpha_beta_model.quantities_in_kernel == ["alpha", "sqrt_beta"]
    assert tabulated_alpha_beta_model.required_quantities == ["fluence"]
    assert tabulated_alpha_beta_model.quantity_transforms == {"alpha": None, "beta": "sqrt"}
    assert tabulated_alpha_beta_model.table_alpha_x.shape == (1,)
    assert tabulated_alpha_beta_model.table_beta_x.shape == (1,)


def test_TabulatedAlphaBetaModel_load_fragments_skips_electrons(synthetic_kernel):
    model = TabulatedAlphaBetaModel()
    model.load_fragments(synthetic_kernel)
    assert model.fragments_to_include.tolist() == [[1.0, 1.0], [12.0, 6.0]]
    assert list(model.fragments_kernel_ix) == [0, 1]
    assert len(model.fragments_sp_table_ix) == len(model.fragments_q_table_ix) == 2


def test_TabulatedAlphaBetaModel_load_fragments_warns_on_missing_fragment(synthetic_kernel):
    model = TabulatedAlphaBetaModel()
    # Na (Z=11) is neither in the SP nor the RBE table and must be dropped with a warning
    model.fragments_to_include = np.asarray([[1.0, 1.0], [23.0, 11.0]])
    with pytest.warns(UserWarning, match="not present"):
        model.load_fragments(synthetic_kernel)
    assert model.fragments_to_include.tolist() == [[1.0, 1.0]]
    assert list(model.fragments_kernel_ix) == [0]


def test_TabulatedAlphaBetaModel_compute_kernel_quantities(synthetic_kernel):
    model = TabulatedAlphaBetaModel()
    model.load_fragments(synthetic_kernel)
    v_tissue_index = np.zeros((10, 1))
    kernel = model.compute_kernel_quantities(synthetic_kernel, v_tissue_index)

    assert kernel.quantities["alpha"].shape == (N_DEPTHS, 1)
    assert kernel.quantities["sqrt_beta"].shape == (N_DEPTHS, 1)
    assert np.allclose(
        kernel.quantities["alpha"][:, 0], _expected_dose_averaged(model, kernel, "alpha")
    )
    assert np.allclose(
        kernel.quantities["sqrt_beta"][:, 0],
        _expected_dose_averaged(model, kernel, "beta", np.sqrt),
    )


def test_TabulatedAlphaBetaModel_per_class_table(synthetic_kernel):
    """A table with one (alpha, beta) set per tissue class yields one kernel column per class."""
    model = TabulatedAlphaBetaModel()
    model._q_table["alpha_x"] = np.asarray([0.1, 0.5])
    model._q_table["beta_x"] = np.asarray([0.05, 0.05])
    model._q_table["alpha"] = np.stack([model._q_table["alpha"], 2 * model._q_table["alpha"]])
    model._q_table["beta"] = np.stack([model._q_table["beta"], 4 * model._q_table["beta"]])
    model.load_fragments(synthetic_kernel)
    kernel = model.compute_kernel_quantities(synthetic_kernel, np.zeros((10, 1)))

    assert kernel.quantities["alpha"].shape == (N_DEPTHS, 2)
    assert np.allclose(kernel.quantities["alpha"][:, 1], 2 * kernel.quantities["alpha"][:, 0])
    assert np.allclose(
        kernel.quantities["sqrt_beta"][:, 1], 2 * kernel.quantities["sqrt_beta"][:, 0]
    )


def test_TabulatedAlphaBetaModel_calc_biological_quantities_for_bixel():
    model = TabulatedAlphaBetaModel()
    bixel = {
        "rad_depths": xp.asarray([0.0, 1.0, 2.0, 3.0]),
        "v_tissue_index": xp.asarray([0, 1, 0, 1]),
        "v_alpha_x": xp.asarray([0.1, 0.5, 0.1, 0.5]),
        "v_beta_x": xp.asarray([0.05, 0.05, 0.05, 0.05]),
    }
    # interpolated kernel quantities have shape (n_voxels, n_tissue_classes)
    kernels = {
        "alpha": xp.asarray([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]]),
        "sqrt_beta": xp.asarray([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0], [1.0, 2.0]]),
    }
    result = model.calc_biological_quantities_for_bixel(bixel, kernels)
    assert np.allclose(np.asarray(result["alpha"]), [1.0, 20.0, 3.0, 40.0])
    assert np.allclose(np.asarray(result["beta"]), [1.0, 4.0, 1.0, 4.0])
