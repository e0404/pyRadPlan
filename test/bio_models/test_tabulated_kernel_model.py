import pytest
import array_api_strict as xp
from pyRadPlan.bio_models._base import BiologicalModelBase
from pyRadPlan.bio_models.models.tabulated_rbe_models import TabulatedAlphaBetaModel
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
        "alpha": xp.asarray([[0.0], [1.0], [2.0]]),
        "beta": xp.asarray([[0.0], [1.0], [2.0]]),
    }
    return kernel_dict


def test_TabulatedAlphaBetaModel_constructor():
    tabulated_alpha_beta_model = TabulatedAlphaBetaModel()
    assert isinstance(tabulated_alpha_beta_model, BiologicalModelBase)
    assert tabulated_alpha_beta_model.model == "dose_average_alpha_beta"
    assert tabulated_alpha_beta_model.quantities_in_table == ["alpha", "beta"]
    assert tabulated_alpha_beta_model.quantities_in_kernel == ["alpha", "sqrt_beta"]
    assert tabulated_alpha_beta_model.required_quantities == ["fluence"]
    assert tabulated_alpha_beta_model.quantity_transforms == {
        "alpha": None,
        "beta": "sqrt",
    }


# dont have the flunence spectrum in one of the generic machine files makes testing the full model difficult
