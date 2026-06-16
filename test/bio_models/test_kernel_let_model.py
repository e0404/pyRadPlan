from pyparsing import dblQuotedString
import pytest
import array_api_strict as xp
from pyRadPlan.bio_models._base import BiologicalModelBase
from pyRadPlan.bio_models.models.let_based_lq_models import (
    Wedenberg,
    MCNamara,
    Carabe,
    HeliumMairani,
    LinearScaling,
)
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


def test_Wedenberg_constructor():
    wedenberg_model = Wedenberg()
    assert isinstance(wedenberg_model, BiologicalModelBase)
    assert wedenberg_model.model == "WED"
    assert wedenberg_model.required_quantities == ["physical_dose", "let"]
    assert wedenberg_model.possible_radiation_modes == ["protons"]


def test_Wedenberg_calc_biological_quantities_for_bixel(sample_bixel, sample_kernel):
    wedenberg_model = Wedenberg()
    bixel = sample_bixel
    kernels = sample_kernel
    p0 = 1
    p1 = 0.434
    p2 = 1
    result = wedenberg_model.calc_biological_quantities_for_bixel(bixel, kernels)
    RBEmax = p0 + (p1 * kernels["let"]) / (bixel["v_alpha_x"] / bixel["v_beta_x"])
    RBEmin = p2
    expected_alpha = RBEmax * bixel["v_alpha_x"]
    expected_beta = RBEmin**2 * bixel["v_beta_x"]
    assert np.allclose(np.asarray(result["alpha"]), np.asarray(expected_alpha))
    assert np.allclose(np.asarray(result["beta"]), np.asarray(expected_beta))


def test_MCnamara_constructor():
    mcnamara_model = MCNamara()
    assert isinstance(mcnamara_model, BiologicalModelBase)
    assert mcnamara_model.model == "MCN"
    assert mcnamara_model.required_quantities == ["physical_dose", "let"]
    assert mcnamara_model.possible_radiation_modes == ["protons"]


def test_MCnamara_calc_biological_quantities_for_bixel(sample_bixel, sample_kernel):
    mcnamara_model = MCNamara()
    bixel = sample_bixel
    kernels = sample_kernel
    p0 = 0.999064
    p1 = 0.35605
    p2 = 1.1012
    p3 = -0.0038703
    result = mcnamara_model.calc_biological_quantities_for_bixel(bixel, kernels)
    RBEmax = p0 + (p1 * kernels["let"]) / (bixel["v_abr_x"])
    RBEmin = p2 + (p3 * kernels["let"] * xp.sqrt(bixel["v_abr_x"]))
    expected_alpha = RBEmax * bixel["v_alpha_x"]
    expected_beta = RBEmin**2 * bixel["v_beta_x"]
    assert np.allclose(np.asarray(result["alpha"]), np.asarray(expected_alpha))
    assert np.allclose(np.asarray(result["beta"]), np.asarray(expected_beta))


def test_Carabe_constructor():
    carabe_model = Carabe()
    assert isinstance(carabe_model, BiologicalModelBase)
    assert carabe_model.model == "CAR"
    assert carabe_model.required_quantities == ["physical_dose", "let"]
    assert carabe_model.possible_radiation_modes == ["protons"]


def test_Carabe_calc_biological_quantities_for_bixel(sample_bixel, sample_kernel):
    carabe_model = Carabe()
    bixel = sample_bixel
    kernels = sample_kernel
    p0 = 0.843
    p1 = 0.154
    p2 = 2.686
    p3 = 1.09
    p4 = 0.006
    result = carabe_model.calc_biological_quantities_for_bixel(bixel, kernels)
    RBEmax = p0 + ((p1 * p2) / bixel["v_abr_x"]) * kernels["let"]
    RBEmin = p3 + ((p4 * p2) / bixel["v_abr_x"]) * kernels["let"]
    expected_alpha = RBEmax * bixel["v_alpha_x"]
    expected_beta = RBEmin**2 * bixel["v_beta_x"]
    assert np.allclose(np.asarray(result["alpha"]), np.asarray(expected_alpha))
    assert np.allclose(np.asarray(result["beta"]), np.asarray(expected_beta))


def test_HeliumMairani_constructor():
    helium_mairani_model = HeliumMairani()
    assert isinstance(helium_mairani_model, BiologicalModelBase)
    assert helium_mairani_model.model == "HEL"
    assert helium_mairani_model.required_quantities == ["physical_dose", "let"]
    assert helium_mairani_model.possible_radiation_modes == ["helium"]


def test_HeliumMairani_calc_biological_quantities_for_bixel(sample_bixel, sample_kernel):
    helium_mairani_model = HeliumMairani()
    bixel = sample_bixel
    kernels = sample_kernel
    p0 = 1.36938e-1
    p1 = 9.73154e-3
    p2 = 1.51998e-2
    result = helium_mairani_model.calc_biological_quantities_for_bixel(bixel, kernels)
    f_QE = (p1 * kernels["let"] ** 2) * xp.exp(-p2 * kernels["let"])
    RBEmax_QE = 1 + (p0 + bixel["v_abr_x"]) * f_QE
    RBEmax = RBEmax_QE
    RBEmin = 1  #
    expected_alpha = RBEmax * bixel["v_alpha_x"]
    expected_beta = RBEmin**2 * bixel["v_beta_x"]
    assert np.allclose(np.asarray(result["alpha"]), np.asarray(expected_alpha))
    assert np.allclose(np.asarray(result["beta"]), np.asarray(expected_beta))


def test_LinearScaling_constructor():
    linear_scaling_model = LinearScaling()
    assert isinstance(linear_scaling_model, BiologicalModelBase)
    assert linear_scaling_model.model == "LSM"
    assert linear_scaling_model.required_quantities == ["physical_dose", "let"]
    assert linear_scaling_model.possible_radiation_modes == ["protons", "helium", "carbon"]


def test_LinearScaling_calc_biological_quantities_for_bixel(sample_bixel, sample_kernel):
    linear_scaling_model = LinearScaling()
    bixel = sample_bixel
    kernels = sample_kernel
    p_lamda_1_1 = 0.008
    p_corrFacEntranceRBE = 0.5  # [kev/mum]
    p_upperLETThreshold = 30  # [kev/mum]
    p_lowerLETThreshold = 0.3  # [kev/mum]
    result = linear_scaling_model.calc_biological_quantities_for_bixel(bixel, kernels)
    RBEmax = xp.full(bixel["v_alpha_x"].shape[0], 0.0)
    ix = (p_lowerLETThreshold < kernels["let"]) & (kernels["let"] < p_upperLETThreshold)
    alpha_0 = bixel["v_alpha_x"] - (p_lamda_1_1 * p_corrFacEntranceRBE)
    RBEmax[ix] = alpha_0[ix] + p_lamda_1_1 * kernels["let"][ix]
    if int(xp.count_nonzero(ix)) < kernels["let"].shape[0]:
        RBEmax[kernels["let"] > p_upperLETThreshold] = (
            alpha_0[kernels["let"] > p_upperLETThreshold] + p_lamda_1_1 * p_upperLETThreshold
        )
        RBEmax[kernels["let"] < p_lowerLETThreshold] = (
            alpha_0[kernels["let"] < p_lowerLETThreshold] + p_lamda_1_1 * p_lowerLETThreshold
        )
    RBEmax[ix] = RBEmax[ix] / bixel["v_alpha_x"][ix]
    RBEmin = 1
    expected_alpha = RBEmax * bixel["v_alpha_x"]
    expected_beta = RBEmin**2 * bixel["v_beta_x"]
    assert np.allclose(np.asarray(result["alpha"]), np.asarray(expected_alpha))
    assert np.allclose(np.asarray(result["beta"]), np.asarray(expected_beta))
