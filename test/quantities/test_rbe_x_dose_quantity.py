import pytest

import numpy as np
import array_api_strict as xp
from scipy.sparse import csc_array

from pyRadPlan import dose
from pyRadPlan.dij import Dij
from pyRadPlan.quantities import RBExDose, FluenceDependentQuantity


@pytest.fixture
def sample_base_dij_dict():
    dij_dict = {
        "ct_grid": {
            "resolution": {"x": 1.5, "y": 1.5, "z": 1.5},
            "dimensions": (10, 10, 10),
            "num_of_voxels": 1000,
        },
        "dose_grid": {
            "resolution": {"x": 3.0, "y": 3.0, "z": 3.0},
            "dimensions": (5, 5, 5),
            "num_of_voxels": 125,
        },
        "num_of_beams": 1,
        "total_num_of_bixels": 10,
        "alpha_dose": np.empty((1, 1, 1), dtype=object),
        "sqrt_beta_dose": np.empty((1, 1, 1), dtype=object),
        "physical_dose": np.empty((1, 1, 1), dtype=object),
        "bixel_num": np.arange(10),
        "ray_num": np.arange(10),
        "beam_num": np.zeros((10,), dtype=np.int64),
        "alphax": np.ones((125,), dtype=np.float32),
        "betax": np.ones((125,), dtype=np.float32),
    }
    return dij_dict


@pytest.fixture
def sample_dij_dense(sample_base_dij_dict):
    sample_base_dij_dict["alpha_dose"].flat[0] = np.ones((125, 10), dtype=np.float32)
    sample_base_dij_dict["sqrt_beta_dose"].flat[0] = np.ones((125, 10), dtype=np.float32)
    sample_base_dij_dict["physical_dose"].flat[0] = np.ones((125, 10), dtype=np.float32)
    dij = Dij.model_validate(sample_base_dij_dict)
    return dij


@pytest.fixture
def sample_dij_sparse(sample_base_dij_dict):
    dense_mat = np.ones((125, 10), dtype=np.float32)
    dense_mat[:100] = 0
    np.random.shuffle(dense_mat)
    sample_base_dij_dict["alpha_dose"].flat[0] = csc_array(dense_mat)
    sample_base_dij_dict["sqrt_beta_dose"].flat[0] = csc_array(dense_mat)
    sample_base_dij_dict["physical_dose"].flat[0] = csc_array(dense_mat)
    dij = Dij.model_validate(sample_base_dij_dict)
    return dij


def test_RBExDose_constructor(sample_dij_dense):
    rbe_x_dose = RBExDose(sample_dij_dense)
    assert isinstance(rbe_x_dose, FluenceDependentQuantity)
    assert rbe_x_dose.mode == "indirect"
    assert rbe_x_dose.scenarios == [0]
    assert rbe_x_dose._dij == sample_dij_dense.to_namespace(xp)
    assert rbe_x_dose.dim == 1
    assert format(rbe_x_dose.unit, "~") == "Gy"
    assert rbe_x_dose.identifier == "rbe_x_dose"
    assert rbe_x_dose.name == "RBExDose"
    assert rbe_x_dose.required_dependencies == ("effect",)
    assert "effect" in rbe_x_dose.dependencies


def test_RBExDose_dense(sample_dij_dense):
    rbe_x_dose = RBExDose(sample_dij_dense)

    fluence = xp.arange(10, dtype=xp.float32)
    ret_callable = rbe_x_dose(fluence)
    assert np.array_equal(rbe_x_dose._w_cache, fluence)
    ret_compute = rbe_x_dose.compute(fluence)

    assert isinstance(ret_callable, np.ndarray)
    assert ret_callable.dtype == sample_dij_dense.physical_dose.dtype
    assert ret_callable.shape == sample_dij_dense.physical_dose.shape

    alpha_mat = sample_dij_dense.alpha_dose.flat[0]
    beta_mat = sample_dij_dense.sqrt_beta_dose.flat[0]
    effect = alpha_mat @ fluence + (beta_mat @ fluence) ** 2
    gamma = sample_dij_dense.alphax / sample_dij_dense.betax / 2
    rbe_x_dose_expected = np.zeros_like(effect)
    rbe_x_dose_expected = np.sqrt(gamma**2 + effect / sample_dij_dense.betax) - gamma
    assert np.allclose(ret_callable.flat[0], rbe_x_dose_expected)
    assert np.array_equal(ret_callable.flat[0], ret_compute.flat[0])

    dose_grad = xp.ones((1, 125), dtype=xp.float32)
    ret_deriv = rbe_x_dose.compute_chain_derivative(dose_grad, fluence)
    effect = effect + gamma
    betax = xp.asarray(sample_dij_dense.betax)
    effect = xp.asarray(effect)
    fgrad = dose_grad / (2 * betax * effect)
    calc_derivative = rbe_x_dose.dependencies["effect"]._compute_chain_derivative_single_scenario(
        fgrad, 0
    )
    assert np.array_equal(rbe_x_dose._w_grad_cache, fluence)
    assert np.array_equal(rbe_x_dose._qgrad_cache.flat[0], ret_deriv.flat[0])
    assert isinstance(ret_deriv, np.ndarray)
    assert ret_deriv.dtype == sample_dij_dense.physical_dose.dtype
    assert ret_deriv.shape == sample_dij_dense.physical_dose.shape
    assert np.allclose(ret_deriv.flat[0], calc_derivative)


def test_RBExDose_sparse(sample_dij_sparse):
    rbe_x_dose = RBExDose(sample_dij_sparse)

    fluence = xp.arange(10, dtype=xp.float32)
    ret_callable = rbe_x_dose(fluence)
    assert np.array_equal(rbe_x_dose._w_cache, fluence)
    ret_compute = rbe_x_dose.compute(rbe_x_dose._w_cache)
    alpha_mat = sample_dij_sparse.alpha_dose.flat[0]
    beta_mat = sample_dij_sparse.sqrt_beta_dose.flat[0]
    effect = alpha_mat @ fluence + (beta_mat @ fluence) ** 2
    ix = sample_dij_sparse.betax > 0
    gamma = np.zeros_like(sample_dij_sparse.betax)
    gamma[ix] = sample_dij_sparse.alphax[ix] / sample_dij_sparse.betax[ix] / 2
    rbe_x_dose_expected = np.zeros_like(effect)
    rbe_x_dose_expected[ix] = (
        np.sqrt(gamma[ix] ** 2 + effect[ix] / sample_dij_sparse.betax[ix]) - gamma[ix]
    )
    assert np.allclose(ret_callable.flat[0], rbe_x_dose_expected)
    assert np.array_equal(ret_callable.flat[0], ret_compute.flat[0])

    dose_grad = xp.ones((1, 125), dtype=xp.float32)
    ret_deriv = rbe_x_dose.compute_chain_derivative(dose_grad, fluence)
    effect = effect + gamma
    betax = xp.asarray(sample_dij_sparse.betax)
    effect = xp.asarray(effect)
    fgrad = dose_grad / (2 * betax * effect)
    calc_derivative = rbe_x_dose.dependencies["effect"]._compute_chain_derivative_single_scenario(
        fgrad, 0
    )
    assert np.array_equal(rbe_x_dose._w_grad_cache, fluence)
    assert np.array_equal(rbe_x_dose._qgrad_cache.flat[0], ret_deriv.flat[0])
    assert isinstance(ret_deriv, np.ndarray)
    assert ret_deriv.dtype == sample_dij_sparse.physical_dose.dtype
    assert ret_deriv.shape == sample_dij_sparse.physical_dose.shape
    assert np.allclose(ret_deriv.flat[0], calc_derivative)
