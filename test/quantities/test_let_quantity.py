import pytest
import numpy as np
import array_api_strict as xp

from pyRadPlan.dij import Dij
from pyRadPlan.quantities import DoseWeightedLET, FluenceDependentQuantity


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
        "physical_dose": np.empty((1, 1, 1), dtype=object),
        "bixel_num": np.arange(10),
        "ray_num": np.arange(10),
        "beam_num": np.zeros((10,), dtype=np.int64),
    }
    return dij_dict


@pytest.fixture
def sample_dij_dense(sample_base_dij_dict):
    sample_base_dij_dict["let_dose"].flat[0] = np.ones((125, 10), dtype=np.float32) * 5.0
    sample_base_dij_dict["physical_dose"].flat[0] = np.ones((125, 10), dtype=np.float32) * 2.0
    dij = Dij.model_validate(sample_base_dij_dict)
    return dij


def test_DoseWeightedLET_properties(sample_dij_dense):
    let = DoseWeightedLET(sample_dij_dense)
    assert isinstance(let, FluenceDependentQuantity)
    assert let.identifier == "let"
    assert let.name == "LETd"
    assert let.required_dependencies == ("let_dose", "physical_dose")
    assert format(let.unit, "~") == "keV / µm"


def test_DoseWeightedLET_compute_and_derivative(sample_dij_dense):
    let = DoseWeightedLET(sample_dij_dense)

    fluence = xp.ones(10, dtype=xp.float32)
    # let_dose = 5.0 * 10 = 50.0
    # physical_dose = 2.0 * 10 = 20.0
    # expected LETd = 50.0 / 20.0 = 2.5
    res = let.compute(fluence)
    assert np.allclose(res.flat[0], 2.5, atol=1e-5)

    # Test chain derivative
    d_q = xp.ones((1, 125), dtype=xp.float32)
    deriv = let.compute_chain_derivative(d_q, fluence)
    assert isinstance(deriv, np.ndarray)
    assert deriv.shape == (1, 1, 1)

    let_dose = let.dependencies["let_dose"].compute(let._w_cache).flat[0]
    dose = let.dependencies["physical_dose"].compute(let._w_cache).flat[0]
    let_dose_grad = let.dependencies["let_dose"]._compute_chain_derivative_single_scenario(
        d_q / dose, 0
    )
    dose_grad = let.dependencies["physical_dose"]._compute_chain_derivative_single_scenario(
        -d_q * let_dose / dose**2, 0
    )
    assert np.allclose(deriv.flat[0], let_dose_grad + dose_grad)


def test_DoseWeightedLET_derivative_without_forward_compute(sample_dij_dense):
    let = DoseWeightedLET(sample_dij_dense)

    fluence = xp.ones(10, dtype=xp.float32)
    d_q = xp.ones((1, 125), dtype=xp.float32)
    deriv = np.asarray(let.compute_chain_derivative(d_q, fluence).flat[0])

    assert np.all(np.isfinite(deriv))


def test_DoseWeightedLET_derivative_uses_current_fluence(sample_base_dij_dict):
    let_dose_matrix = np.tile(np.arange(1, 11, dtype=np.float32), (125, 1))
    dose_matrix = np.tile(np.arange(10, 0, -1, dtype=np.float32), (125, 1))
    sample_base_dij_dict["let_dose"].flat[0] = let_dose_matrix
    sample_base_dij_dict["physical_dose"].flat[0] = dose_matrix
    let = DoseWeightedLET(Dij.model_validate(sample_base_dij_dict))

    let.compute(xp.ones(10, dtype=xp.float32))

    fluence = xp.asarray(np.arange(1, 11, dtype=np.float32))
    d_q = xp.ones((1, 125), dtype=xp.float32)
    deriv = np.asarray(let.compute_chain_derivative(d_q, fluence).flat[0])

    let_dose = let_dose_matrix @ np.arange(1, 11, dtype=np.float32)
    dose = dose_matrix @ np.arange(1, 11, dtype=np.float32)
    d_q_numpy = np.asarray(d_q)
    expected = (d_q_numpy / dose) @ let_dose_matrix - (
        d_q_numpy * let_dose / dose**2
    ) @ dose_matrix
    assert np.allclose(deriv, expected)


@pytest.fixture
def sample_dij_partial_dose(sample_base_dij_dict):
    """Dij whose first 60 voxels receive no physical dose but a non-zero let_dose.

    Physically inconsistent on purpose: it pins the behaviour of the quotient where the
    denominator vanishes, which an epsilon-regularised division would turn into a huge
    number rather than zero.
    """
    let_dose = np.ones((125, 10), dtype=np.float32) * 5.0
    physical_dose = np.ones((125, 10), dtype=np.float32) * 2.0
    physical_dose[:60, :] = 0.0
    sample_base_dij_dict["let_dose"].flat[0] = let_dose
    sample_base_dij_dict["physical_dose"].flat[0] = physical_dose
    return Dij.model_validate(sample_base_dij_dict)


def test_DoseWeightedLET_zero_dose_voxels(sample_dij_partial_dose):
    """Voxels without dose must yield exactly zero, not a huge or non-finite value."""
    let = DoseWeightedLET(sample_dij_partial_dose)
    fluence = xp.ones(10, dtype=xp.float32)

    res = np.asarray(let.compute(fluence).flat[0])
    assert np.all(np.isfinite(res))
    assert np.all(res[:60] == 0.0)
    assert np.allclose(res[60:], 2.5, atol=1e-5)

    d_q = xp.ones((1, 125), dtype=xp.float32)
    deriv = np.asarray(let.compute_chain_derivative(d_q, fluence).flat[0])
    assert np.all(np.isfinite(deriv))
