import pytest

import array_api_strict as xp


from pyRadPlan.bio_models._base import EmptyModel, BiologicalModelBase


@pytest.fixture
def sample_bixel():
    bixel_dict = {
        "rad_depths": xp.asarray([0.0, 1.0, 2.0]),
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


def test_EmptyModel_constructor():
    empty_model = EmptyModel()
    assert isinstance(empty_model, BiologicalModelBase)
    assert empty_model.model == "none"
    assert empty_model.required_quantities == []
    assert empty_model.possible_radiation_modes == [
        "photons",
        "protons",
        "helium",
        "carbon",
        "VHEE",
    ]


def test_EmptyModel_calc_biological_quantities_for_bixel(sample_bixel, sample_kernel):
    empty_model = EmptyModel()
    bixel = sample_bixel
    kernels = sample_kernel
    result = empty_model.calc_biological_quantities_for_bixel(bixel, kernels)
    assert result == bixel
