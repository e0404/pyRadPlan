from numpy.random import beta
import pytest

import numpy as np
import array_api_strict as xp
from scipy.sparse import csc_array

from pyRadPlan.dij import Dij
from pyRadPlan.quantities import Effect, FluenceDependentQuantity


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


def test_Effect_constructor(sample_dij_dense):
    effect = Effect(sample_dij_dense)
    assert isinstance(effect, FluenceDependentQuantity)
    assert effect.mode == "indirect"
    assert effect.scenarios == [0]
    assert effect._dij == sample_dij_dense.to_namespace(xp)
    assert effect.dim == 1
    assert effect.unit == []
    assert effect.identifier == "effect"
    assert effect.name == "Effect"
    assert effect.required_dependencies == ("alpha_dose", "sqrt_beta_dose")
    assert set(effect.dependencies) == {"alpha_dose", "sqrt_beta_dose"}


def test_Effect_dense(sample_dij_dense):
    effect = Effect(sample_dij_dense)

    fluence = xp.arange(10, dtype=xp.float32)
    ret_callable = effect(fluence)
    assert np.array_equal(effect._w_cache, fluence)
    ret_compute = effect.compute(fluence)

    assert isinstance(ret_callable, np.ndarray)
    assert ret_callable.dtype == sample_dij_dense.physical_dose.dtype
    assert ret_callable.shape == sample_dij_dense.physical_dose.shape

    alpha_mat = sample_dij_dense.alpha_dose.flat[0]
    beta_mat = sample_dij_dense.sqrt_beta_dose.flat[0]
    assert np.allclose(ret_callable.flat[0], alpha_mat @ fluence + (beta_mat @ fluence) ** 2)
    assert np.array_equal(ret_callable.flat[0], ret_compute.flat[0])

    dose_grad = xp.ones((1, 125), dtype=xp.float32)
    ret_deriv = effect.compute_chain_derivative(dose_grad, fluence)
    alpha_grad = effect.dependencies["alpha_dose"]._compute_chain_derivative_single_scenario(
        dose_grad, 0
    )
    sqrt_beta_dose = effect.dependencies["sqrt_beta_dose"].compute(effect._w_cache)
    fgrad_beta = 2 * dose_grad * sqrt_beta_dose.flat[0]
    beta_grad = effect.dependencies["sqrt_beta_dose"]._compute_chain_derivative_single_scenario(
        fgrad_beta, 0
    )
    calc_derivative = alpha_grad + beta_grad
    assert np.array_equal(effect._w_grad_cache, fluence)
    assert np.array_equal(effect._qgrad_cache.flat[0], ret_deriv.flat[0])
    assert isinstance(ret_deriv, np.ndarray)
    assert ret_deriv.dtype == sample_dij_dense.physical_dose.dtype
    assert ret_deriv.shape == sample_dij_dense.physical_dose.shape
    assert np.allclose(ret_deriv.flat[0], calc_derivative)


def test_effect_sparse(sample_dij_sparse):
    effect = Effect(sample_dij_sparse)

    fluence = xp.arange(10, dtype=xp.float32)
    ret_callable = effect(fluence)
    assert np.array_equal(effect._w_cache, fluence)
    ret_compute = effect.compute(fluence)

    assert isinstance(ret_callable, np.ndarray)
    assert ret_callable.dtype == sample_dij_sparse.physical_dose.dtype
    assert ret_callable.shape == sample_dij_sparse.physical_dose.shape

    alpha_mat = sample_dij_sparse.alpha_dose.flat[0]
    beta_mat = sample_dij_sparse.sqrt_beta_dose.flat[0]
    assert np.allclose(ret_callable.flat[0], alpha_mat @ fluence + (beta_mat @ fluence) ** 2)
    assert np.array_equal(ret_callable.flat[0], ret_compute.flat[0])

    dose_grad = xp.ones((1, 125), dtype=xp.float32)
    ret_deriv = effect.compute_chain_derivative(dose_grad, fluence)
    alpha_grad = effect.dependencies["alpha_dose"]._compute_chain_derivative_single_scenario(
        dose_grad, 0
    )
    sqrt_beta_dose = effect.dependencies["sqrt_beta_dose"].compute(effect._w_cache)
    fgrad_beta = 2 * dose_grad * sqrt_beta_dose.flat[0]
    beta_grad = effect.dependencies["sqrt_beta_dose"]._compute_chain_derivative_single_scenario(
        fgrad_beta, 0
    )
    calc_derivative = alpha_grad + beta_grad
    assert np.array_equal(effect._w_grad_cache, fluence)
    assert np.array_equal(effect._qgrad_cache.flat[0], ret_deriv.flat[0])
    assert isinstance(ret_deriv, np.ndarray)
    assert ret_deriv.dtype == sample_dij_sparse.physical_dose.dtype
    assert ret_deriv.shape == sample_dij_sparse.physical_dose.shape
    assert np.allclose(ret_deriv.flat[0], calc_derivative)
