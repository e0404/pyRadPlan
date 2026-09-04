import numpy as np
import pytest
import array_api_strict as xp

from pyRadPlan.bio_models import (
    BiologicalModel,
    TabulatedAlphaBetaModel,
    TabulatedSpectrumEvaluator,
)
from pyRadPlan.machines.particles import ParticlePencilBeamKernel


N_DEPTHS = 6
N_ENERGIES = 25


def _synthetic_kernel(energy, seed):
    """Pencil-beam kernel with a synthetic fragment fluence spectrum (H, C and electrons)."""
    rng = np.random.default_rng(seed)
    depths = np.linspace(0.0, 50.0, N_DEPTHS)
    energies = np.geomspace(1.0, 400.0, N_ENERGIES)
    spectra = [rng.random((N_ENERGIES, N_DEPTHS)) for _ in range(3)]
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
        energy=energy,
        depths=depths,
        Z=np.ones(N_DEPTHS),
        sigma=np.ones(N_DEPTHS),
        Fluence=fluence,
    )


class _Machine:
    def __init__(self):
        self.energies = [100.0, 200.0]
        self.pb_kernels = {e: _synthetic_kernel(e, seed=i) for i, e in enumerate(self.energies)}


@pytest.fixture
def machine():
    return _Machine()


@pytest.fixture
def voxel_params():
    return {"alpha_x": np.asarray([[0.1], [0.0]]), "beta_x": np.asarray([[0.05], [0.0]])}


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
    model = TabulatedAlphaBetaModel()
    assert isinstance(model, BiologicalModel)
    assert model.model == "dose_average_alpha_beta"
    assert model.quantities_in_table == ["alpha", "beta"]
    assert model.quantities_in_kernel == ["alpha", "sqrt_beta"]
    assert model.required_quantities == ["fluence"]
    assert model.quantity_transforms == {"alpha": None, "beta": "sqrt"}
    assert model.provides_alpha_beta is True
    assert model.table_alpha_x.shape == (1,)
    assert model.table_beta_x.shape == (1,)


def test_TabulatedAlphaBetaModel_select_fragments_skips_electrons(machine):
    model = TabulatedAlphaBetaModel()
    selected = model.select_fragments(machine.pb_kernels[100.0].fluence_spectrum)
    assert selected["fragments_AZ"].tolist() == [[1.0, 1.0], [12.0, 6.0]]
    assert selected["kernel_ix"] == [0, 1]
    assert len(selected["sp_table_ix"]) == len(selected["q_table_ix"]) == 2


def test_TabulatedAlphaBetaModel_select_fragments_warns_on_missing_fragment(machine):
    # Na (Z=11) is neither in the SP nor the RBE table and must be dropped with a warning
    model = TabulatedAlphaBetaModel(fragments_to_include=[[1.0, 1.0], [23.0, 11.0]])
    with pytest.warns(UserWarning, match="not present"):
        selected = model.select_fragments(machine.pb_kernels[100.0].fluence_spectrum)
    assert selected["fragments_AZ"].tolist() == [[1.0, 1.0]]
    assert selected["kernel_ix"] == [0]


def test_TabulatedAlphaBetaModel_dose_average(machine):
    model = TabulatedAlphaBetaModel()
    kernel = machine.pb_kernels[100.0]
    tables = model.dose_average(kernel, model.select_fragments(kernel.fluence_spectrum))

    assert tables["alpha"].shape == (1, N_DEPTHS)
    assert tables["sqrt_beta"].shape == (1, N_DEPTHS)
    assert np.allclose(tables["alpha"][0], _expected_dose_averaged(model, kernel, "alpha"))
    assert np.allclose(
        tables["sqrt_beta"][0], _expected_dose_averaged(model, kernel, "beta", np.sqrt)
    )


def test_TabulatedAlphaBetaModel_per_class_table(machine):
    """A table with one (alpha, beta) set per tissue class yields one row per class."""
    model = TabulatedAlphaBetaModel()
    model._q_table["alpha_x"] = np.asarray([0.1, 0.5])
    model._q_table["beta_x"] = np.asarray([0.05, 0.05])
    model._q_table["alpha"] = np.stack([model._q_table["alpha"], 2 * model._q_table["alpha"]])
    model._q_table["beta"] = np.stack([model._q_table["beta"], 4 * model._q_table["beta"]])
    kernel = machine.pb_kernels[100.0]
    tables = model.dose_average(kernel, model.select_fragments(kernel.fluence_spectrum))

    assert tables["alpha"].shape == (2, N_DEPTHS)
    assert np.allclose(tables["alpha"][1], 2 * tables["alpha"][0])
    assert np.allclose(tables["sqrt_beta"][1], 2 * tables["sqrt_beta"][0])


def test_TabulatedAlphaBetaModel_evaluator(machine, voxel_params):
    model = TabulatedAlphaBetaModel()
    evaluator = model.evaluator(machine, voxel_params)
    assert isinstance(evaluator, TabulatedSpectrumEvaluator)

    # the machine is left untouched and one table per energy is precomputed
    assert not hasattr(machine.pb_kernels[100.0], "quantities")
    for energy in machine.energies:
        kernel_dict = machine.pb_kernels[energy].to_namespace(xp)
        quantities = evaluator.kernel_quantities(kernel_dict)
        assert set(quantities) == {"alpha", "sqrt_beta"}
        assert quantities["alpha"].shape == (1, N_DEPTHS)
        expected = _expected_dose_averaged(model, machine.pb_kernels[energy], "alpha")
        assert np.allclose(np.asarray(quantities["alpha"][0, :]), expected)


def test_TabulatedAlphaBetaModel_selects_fragments_per_energy(machine, voxel_params):
    second_kernel = machine.pb_kernels[200.0]
    second_kernel.fluence_spectrum.fragments = list(
        reversed(second_kernel.fluence_spectrum.fragments)
    )
    model = TabulatedAlphaBetaModel()

    evaluator = model.evaluator(machine, voxel_params)
    expected = model.dose_average(
        second_kernel,
        model.select_fragments(second_kernel.fluence_spectrum),
    )

    assert np.allclose(evaluator._tables[200.0]["alpha"], expected["alpha"])
    assert np.allclose(evaluator._tables[200.0]["sqrt_beta"], expected["sqrt_beta"])


def test_TabulatedAlphaBetaModel_bixel_alpha_beta(machine, voxel_params):
    evaluator = TabulatedAlphaBetaModel().evaluator(machine, voxel_params)
    bixel = {
        "rad_depths": xp.asarray([0.0, 1.0, 2.0, 3.0]),
        "v_alpha_x": xp.asarray([0.1, 0.1, 0.0, 0.1]),
        "v_beta_x": xp.asarray([0.05, 0.05, 0.0, 0.05]),
    }
    # interpolated kernels: (n_tissue_classes, n_voxels)
    kernels = {
        "alpha": xp.asarray([[1.0, 2.0, 3.0, 4.0]]),
        "sqrt_beta": xp.asarray([[1.0, 2.0, 1.0, 2.0]]),
    }
    alpha, beta = evaluator.bixel_alpha_beta(bixel, kernels)
    assert np.allclose(np.asarray(alpha), [1.0, 2.0, 3.0, 4.0])
    assert np.allclose(np.asarray(beta), [1.0, 4.0, 1.0, 4.0])
