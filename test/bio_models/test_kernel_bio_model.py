import pytest
import array_api_strict as xp
from pyRadPlan.bio_models._base import BiologicalModelBase
from pyRadPlan.bio_models.models.kernel_based_lq_model import KernelBasedLQModel
import numpy as np


@pytest.fixture
def sample_bixel():
    bixel_dict = {
        "rad_depths": xp.asarray([0.0, 1.0, 2.0]),
        "v_tissue_index": xp.asarray([0, 0, 0]),
        "v_alpha_x": xp.asarray([0.1, 0.1, 0.1]),
        "v_beta_x": xp.asarray([0.05, 0.05, 0.05]),
    }
    return bixel_dict


@pytest.fixture
def sample_kernel():
    kernel_dict = {
        "let": xp.asarray([0.0, 1.0, 2.0]),
        "alpha": xp.asarray([[0.0, 1.0, 2.0]]),  # (n_tissue_classes, n_voxels)
        "beta": xp.asarray([[0.0, 1.0, 2.0]]),
    }
    return kernel_dict


def test_KernelBasedLQModel_constructor():
    kernel_lq_model = KernelBasedLQModel()
    assert isinstance(kernel_lq_model, BiologicalModelBase)
    assert kernel_lq_model.model == "kernel_based_lq"
    assert kernel_lq_model.required_quantities == ["physical_dose", "alpha", "beta"]
    assert kernel_lq_model.possible_radiation_modes == ["protons", "helium", "carbon", "oxygen"]
    assert kernel_lq_model.kernel_quantities == ["alpha", "beta"]


def test_KernelBasedLQModel_calc_biological_quantities_for_bixel(sample_bixel, sample_kernel):
    kernel_lq_model = KernelBasedLQModel()
    bixel = sample_bixel
    kernels = sample_kernel
    result = kernel_lq_model.calc_biological_quantities_for_bixel(bixel, kernels)
    bixel["alpha"] = kernels["alpha"]
    bixel["beta"] = kernels["beta"]
    assert np.allclose(np.asarray(result["alpha"]), np.asarray(bixel["alpha"]))
    assert np.allclose(np.asarray(result["beta"]), np.asarray(bixel["beta"]))


def test_KernelBasedLQModel_mixed_tissue_classes():
    kernel_lq_model = KernelBasedLQModel()
    bixel = {
        "rad_depths": xp.asarray([0.0, 1.0, 2.0, 3.0]),
        "v_tissue_index": xp.asarray([0, 2, 0, 1]),
        "v_alpha_x": xp.asarray([0.1, 0.5, 0.1, 0.3]),
        "v_beta_x": xp.asarray([0.05, 0.05, 0.05, 0.05]),
    }
    kernels = {
        "alpha": xp.asarray([[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0], [3.0, 3.0, 3.0, 3.0]]),
        "beta": xp.asarray([[4.0, 4.0, 4.0, 4.0], [5.0, 5.0, 5.0, 5.0], [6.0, 6.0, 6.0, 6.0]]),
    }
    result = kernel_lq_model.calc_biological_quantities_for_bixel(bixel, kernels)
    assert np.allclose(np.asarray(result["alpha"]), [1.0, 3.0, 1.0, 2.0])
    assert np.allclose(np.asarray(result["beta"]), [4.0, 6.0, 4.0, 5.0])


def test_KernelBasedLQModel_match_tissue_classes():
    v_alpha_x = np.asarray([[0.1, 0.1], [0.5, 0.5], [0.0, 0.5]])
    v_beta_x = np.asarray([[0.05, 0.05], [0.05, 0.05], [0.0, 0.05]])
    ix = KernelBasedLQModel.match_tissue_classes(v_alpha_x, v_beta_x, [0.1, 0.5], [0.05, 0.05])
    assert np.array_equal(ix, [[0, 0], [1, 1], [0, 1]])

    with pytest.raises(ValueError, match="No matching tissue class"):
        KernelBasedLQModel.match_tissue_classes(
            np.asarray([[0.3]]), np.asarray([[0.05]]), [0.1], [0.05]
        )
