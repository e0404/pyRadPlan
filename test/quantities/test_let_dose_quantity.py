import pytest

import numpy as np
from scipy.sparse import csc_array

from pyRadPlan.dij import Dij
from pyRadPlan.quantities import LETxDose, RTQuantity


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
        "let_dose": np.empty((1, 1, 1), dtype=object),
        "bixel_num": np.arange(10),
        "ray_num": np.arange(10),
        "beam_num": np.zeros((10,), dtype=np.int64),
    }
    return dij_dict


@pytest.fixture
def sample_dij_dense(sample_base_dij_dict):
    sample_base_dij_dict["let_dose"].flat[0] = np.ones((125, 10), dtype=np.float32)
    dij = Dij.model_validate(sample_base_dij_dict)
    return dij


@pytest.fixture
def sample_dij_sparse(sample_base_dij_dict):
    dense_mat = np.ones((125, 10), dtype=np.float32)
    dense_mat[:100] = 0
    np.random.shuffle(dense_mat)
    sample_base_dij_dict["let_dose"].flat[0] = csc_array(dense_mat)
    dij = Dij.model_validate(sample_base_dij_dict)
    return dij


def test_LETxDose_constructor(sample_dij_dense):
    let_dose = LETxDose(sample_dij_dense)
    assert isinstance(let_dose, RTQuantity)
    assert let_dose.scenarios == [0]
    assert let_dose._dij == sample_dij_dense
    assert let_dose.dim == 1
    assert format(let_dose.unit, "~") == "Gy * µm / keV"
    assert let_dose.identifier == "let_dose"
    assert let_dose.name == "LETxDose"


def test_let_dose_dense(sample_dij_dense):
    let_dose = LETxDose(sample_dij_dense)

    fluence = range(10)
    ret_callable = let_dose(fluence)
    assert np.array_equal(let_dose._w_cache, fluence)
    ret_compute = let_dose.compute(fluence)

    assert isinstance(ret_callable, np.ndarray)
    assert ret_callable.dtype == sample_dij_dense.let_dose.dtype
    assert ret_callable.shape == sample_dij_dense.let_dose.shape

    dij_mat = sample_dij_dense.let_dose.flat[0]
    assert np.allclose(ret_callable.flat[0], dij_mat @ fluence)
    assert np.array_equal(ret_callable.flat[0], ret_compute.flat[0])

    ret_deriv = let_dose.compute_chain_derivative(np.ones(125, dtype=np.float32), fluence)
    assert np.array_equal(let_dose._w_grad_cache, fluence)
    assert np.array_equal(let_dose._qgrad_cache.flat[0], ret_deriv.flat[0])
    assert isinstance(ret_deriv, np.ndarray)
    assert ret_deriv.dtype == sample_dij_dense.let_dose.dtype
    assert ret_deriv.shape == sample_dij_dense.let_dose.shape
    assert np.allclose(ret_deriv.flat[0], dij_mat.T @ np.ones(125, dtype=np.float32))


def test_let_dose_sparse(sample_dij_sparse):
    let_dose = LETxDose(sample_dij_sparse)

    fluence = range(10)
    ret_callable = let_dose(fluence)
    assert np.array_equal(let_dose._w_cache, fluence)
    ret_compute = let_dose.compute(fluence)

    assert isinstance(ret_callable, np.ndarray)
    assert ret_callable.dtype == sample_dij_sparse.let_dose.dtype
    assert ret_callable.shape == sample_dij_sparse.let_dose.shape

    dij_mat = sample_dij_sparse.let_dose.flat[0]
    assert np.allclose(ret_callable.flat[0], dij_mat @ fluence)
    assert np.array_equal(ret_callable.flat[0], ret_compute.flat[0])

    ret_deriv = let_dose.compute_chain_derivative(np.ones(125, dtype=np.float32), fluence)
    assert np.array_equal(let_dose._w_grad_cache, fluence)
    assert np.array_equal(let_dose._qgrad_cache.flat[0], ret_deriv.flat[0])
    assert isinstance(ret_deriv, np.ndarray)
    assert ret_deriv.dtype == sample_dij_sparse.let_dose.dtype
    assert ret_deriv.shape == sample_dij_sparse.let_dose.shape
    assert np.allclose(ret_deriv.flat[0], dij_mat.T @ np.ones(125, dtype=np.float32))
