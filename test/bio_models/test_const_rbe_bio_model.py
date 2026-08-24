import pytest
import array_api_strict as xp
from pyRadPlan.bio_models._base import BiologicalModelBase
from pyRadPlan.bio_models.models.const_rbe import ConstantRBEModel


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
        "alpha": xp.asarray([0.0, 1.0, 2.0]),
        "beta": xp.asarray([0.0, 1.0, 2.0]),
    }
    return kernel_dict


def test_ConstantRBEModel_constructor():
    const_rbe_model = ConstantRBEModel()
    assert isinstance(const_rbe_model, BiologicalModelBase)
    assert const_rbe_model.model == "constant_rbe"
    assert const_rbe_model.required_quantities == ["physical_dose"]
    assert const_rbe_model.possible_radiation_modes == [
        "photons",
        "protons",
        "helium",
        "carbon",
        "oxygen",
        "VHEE",
    ]


def test_ConstantRBEModel_calc_biological_quantities_for_bixel(sample_bixel, sample_kernel):
    const_rbe_model = ConstantRBEModel()
    rbe = 1.1
    bixel = sample_bixel
    kernels = sample_kernel
    result = const_rbe_model.calc_biological_quantities_for_bixel(bixel, kernels)
    bixel["alpha"] = rbe * bixel["v_alpha_x"]
    bixel["beta"] = rbe**2 * bixel["v_beta_x"]
    assert result == bixel
