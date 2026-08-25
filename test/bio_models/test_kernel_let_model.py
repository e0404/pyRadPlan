import numpy as np
import pytest
import array_api_strict as xp

from pyRadPlan.bio_models import BiologicalModel, ParametricEvaluator
from pyRadPlan.bio_models.models.let_based_lq_models import (
    Wedenberg,
    MCNamara,
    Carabe,
    HeliumMairani,
    LinearScaling,
)


@pytest.fixture
def sample_bixel():
    return {
        "rad_depths": xp.asarray([0.0, 1.0, 2.0]),
        "v_alpha_x": xp.asarray([0.1, 0.1, 0.1]),
        "v_beta_x": xp.asarray([0.05, 0.05, 0.05]),
    }


@pytest.fixture
def sample_kernel():
    return {"let": xp.asarray([0.0, 1.0, 2.0])}


def _evaluate(model, bixel, kernels):
    evaluator = model.evaluator(machine=None, voxel_params={})
    assert isinstance(evaluator, ParametricEvaluator)
    assert evaluator.kernel_quantities({"let": 1}) == {}
    alpha, beta = evaluator.bixel_alpha_beta(bixel, kernels)
    return np.asarray(alpha), np.asarray(beta)


@pytest.mark.parametrize(
    "model_cls, name, modes",
    [
        (Wedenberg, "WED", ["protons"]),
        (MCNamara, "MCN", ["protons"]),
        (Carabe, "CAR", ["protons"]),
        (HeliumMairani, "HEL", ["helium"]),
        (LinearScaling, "LSM", ["protons", "helium", "carbon"]),
    ],
)
def test_let_model_constructor(model_cls, name, modes):
    model = model_cls()
    assert isinstance(model, BiologicalModel)
    assert model.model == name
    assert model.required_quantities == ["physical_dose", "let"]
    assert model.possible_radiation_modes == modes
    assert model.provides_alpha_beta is True
    assert model.requires_let is True


def test_Wedenberg_alpha_beta(sample_bixel, sample_kernel):
    p0, p1, p2 = 1, 0.434, 1
    let = sample_kernel["let"]
    abr = sample_bixel["v_alpha_x"] / sample_bixel["v_beta_x"]
    rbe_max = p0 + (p1 * let) / abr
    rbe_min = p2
    alpha, beta = _evaluate(Wedenberg(), sample_bixel, sample_kernel)
    assert np.allclose(alpha, np.asarray(rbe_max * sample_bixel["v_alpha_x"]))
    assert np.allclose(beta, np.asarray(rbe_min**2 * sample_bixel["v_beta_x"]))


def test_MCNamara_alpha_beta(sample_bixel, sample_kernel):
    p0, p1, p2, p3 = 0.999064, 0.35605, 1.1012, -0.0038703
    let = sample_kernel["let"]
    abr = sample_bixel["v_alpha_x"] / sample_bixel["v_beta_x"]
    rbe_max = p0 + (p1 * let) / abr
    rbe_min = p2 + (p3 * let * xp.sqrt(abr))
    alpha, beta = _evaluate(MCNamara(), sample_bixel, sample_kernel)
    assert np.allclose(alpha, np.asarray(rbe_max * sample_bixel["v_alpha_x"]))
    assert np.allclose(beta, np.asarray(rbe_min**2 * sample_bixel["v_beta_x"]))


def test_Carabe_alpha_beta(sample_bixel, sample_kernel):
    p0, p1, p2, p3, p4 = 0.843, 0.154, 2.686, 1.09, 0.006
    let = sample_kernel["let"]
    abr = sample_bixel["v_alpha_x"] / sample_bixel["v_beta_x"]
    rbe_max = p0 + ((p1 * p2) / abr) * let
    rbe_min = p3 + ((p4 * p2) / abr) * let
    alpha, beta = _evaluate(Carabe(), sample_bixel, sample_kernel)
    assert np.allclose(alpha, np.asarray(rbe_max * sample_bixel["v_alpha_x"]))
    assert np.allclose(beta, np.asarray(rbe_min**2 * sample_bixel["v_beta_x"]))


def test_HeliumMairani_alpha_beta(sample_bixel, sample_kernel):
    p0, p1, p2 = 1.36938e-1, 9.73154e-3, 1.51998e-2
    let = sample_kernel["let"]
    abr = sample_bixel["v_alpha_x"] / sample_bixel["v_beta_x"]
    f_qe = (p1 * let**2) * xp.exp(-p2 * let)
    rbe_max = 1 + (p0 + 1 / abr) * f_qe
    rbe_min = 1
    alpha, beta = _evaluate(HeliumMairani(), sample_bixel, sample_kernel)
    assert np.allclose(alpha, np.asarray(rbe_max * sample_bixel["v_alpha_x"]))
    assert np.allclose(beta, np.asarray(rbe_min**2 * sample_bixel["v_beta_x"]))


def test_LinearScaling_alpha_beta(sample_bixel):
    lam, corr, upper, lower = 0.008, 0.5, 30.0, 0.3
    # below, inside and above the LET thresholds
    kernels = {"let": xp.asarray([0.1, 5.0, 40.0])}
    let_clipped = np.asarray([lower, 5.0, upper])
    alpha_x = np.asarray(sample_bixel["v_alpha_x"])
    alpha_0 = alpha_x - lam * corr
    expected_alpha = alpha_0 + lam * let_clipped
    alpha, beta = _evaluate(LinearScaling(), sample_bixel, kernels)
    assert np.allclose(alpha, expected_alpha)
    assert np.allclose(beta, np.asarray(sample_bixel["v_beta_x"]))


def test_model_parameters_are_configurable():
    assert Wedenberg(p1=0.5).p1_WED == 0.5
    assert LinearScaling(upper_let_threshold=20.0).p_upperLETThreshold == 20.0
