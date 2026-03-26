import pytest

import numpy as np
import array_api_strict as xp
from scipy.sparse import csc_array

from pyRadPlan.dij import Dij
from pyRadPlan.quantities import Dose, RTQuantity


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
        "physical_dose": np.empty((1, 1, 1), dtype=object),
        "bixel_num": np.arange(10),
        "ray_num": np.arange(10),
        "beam_num": np.zeros((10,), dtype=np.int64),
    }
    return dij_dict


@pytest.fixture
def sample_dij_dense(sample_base_dij_dict):
    sample_base_dij_dict["physical_dose"].flat[0] = np.ones((125, 10), dtype=np.float32)
    dij = Dij.model_validate(sample_base_dij_dict)
    return dij


@pytest.fixture
def sample_dij_sparse(sample_base_dij_dict):
    dense_mat = np.ones((125, 10), dtype=np.float32)
    dense_mat[:100] = 0
    np.random.shuffle(dense_mat)
    sample_base_dij_dict["physical_dose"].flat[0] = csc_array(dense_mat)
    dij = Dij.model_validate(sample_base_dij_dict)
    return dij


def test_Dose_constructor(sample_dij_dense):
    dose = Dose(sample_dij_dense)
    assert isinstance(dose, RTQuantity)
    assert dose.scenarios == [0]
    assert dose._dij == sample_dij_dense.to_namespace(xp)
    assert dose.dim == 1
    assert format(dose.unit, "~") == "Gy"
    assert dose.identifier == "physical_dose"
    assert dose.name == "dose"


def test_dose_dense(sample_dij_dense):
    dose = Dose(sample_dij_dense)

    fluence = xp.arange(10, dtype=xp.float32)
    ret_callable = dose(fluence)
    assert np.array_equal(dose._w_cache, fluence)
    ret_compute = dose.compute(fluence)

    # assert isinstance(ret_callable, type(fluence))
    assert ret_callable.dtype == sample_dij_dense.physical_dose.dtype
    assert ret_callable.shape == sample_dij_dense.physical_dose.shape

    dij_mat = sample_dij_dense.physical_dose.flat[0]
    assert np.allclose(ret_callable.flat[0], dij_mat @ fluence)
    assert np.array_equal(ret_callable.flat[0], ret_compute.flat[0])

    ret_deriv = dose.compute_chain_derivative(xp.ones((1, 125), dtype=xp.float32), fluence)
    assert np.array_equal(dose._w_grad_cache, fluence)
    assert np.array_equal(dose._qgrad_cache.flat[0], ret_deriv.flat[0])
    # assert isinstance(ret_deriv, np.ndarray)
    assert ret_deriv.dtype == sample_dij_dense.physical_dose.dtype
    assert ret_deriv.shape == sample_dij_dense.physical_dose.shape
    assert np.allclose(ret_deriv.flat[0], dij_mat.T @ np.ones(125, dtype=np.float32))


def test_dose_sparse(sample_dij_sparse):
    dose = Dose(sample_dij_sparse)

    fluence = xp.arange(10, dtype=xp.float32)
    ret_callable = dose(fluence)
    assert np.array_equal(dose._w_cache, fluence)
    ret_compute = dose.compute(fluence)

    assert isinstance(ret_callable, np.ndarray)
    assert ret_callable.dtype == sample_dij_sparse.physical_dose.dtype
    assert ret_callable.shape == sample_dij_sparse.physical_dose.shape

    dij_mat = sample_dij_sparse.physical_dose.flat[0]
    assert np.allclose(ret_callable.flat[0], dij_mat @ fluence)
    assert np.array_equal(ret_callable.flat[0], ret_compute.flat[0])

    ret_deriv = dose.compute_chain_derivative(xp.ones((1, 125), dtype=xp.float32), fluence)
    assert np.array_equal(dose._w_grad_cache, fluence)
    assert np.array_equal(dose._qgrad_cache.flat[0], ret_deriv.flat[0])
    assert isinstance(ret_deriv, np.ndarray)
    assert ret_deriv.dtype == sample_dij_sparse.physical_dose.dtype
    assert ret_deriv.shape == sample_dij_sparse.physical_dose.shape
    assert np.allclose(ret_deriv.flat[0], dij_mat.T @ np.ones(125, dtype=np.float32))
