import array_api_strict as xp
import numpy as np
import pytest
from scipy.sparse import csc_array

from pyRadPlan.bio_models import ConstantRBEModel, Wedenberg
from pyRadPlan.dij import Dij
from pyRadPlan.quantities import FluenceDependentQuantity
from pyRadPlan.quantities._rbe_x_dose import RBExDose, lq_inverse_dose


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
        "alphax": np.ones((125, 1), dtype=np.float32),
        "betax": np.ones((125, 1), dtype=np.float32),
    }
    return dij_dict


@pytest.fixture
def sample_base_dij_constant_rbe_dict():
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
        "rbe": 1.1,
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


@pytest.fixture
def sample_dij_dense_constant_rbe(sample_base_dij_constant_rbe_dict):
    sample_base_dij_constant_rbe_dict["physical_dose"].flat[0] = np.ones(
        (125, 10), dtype=np.float32
    )
    dij = Dij.model_validate(sample_base_dij_constant_rbe_dict)
    return dij


@pytest.fixture
def sample_dij_sparse_constant_rbe(sample_base_dij_constant_rbe_dict):
    dense_mat = np.ones((125, 10), dtype=np.float32)
    dense_mat[:100] = 0
    np.random.shuffle(dense_mat)
    sample_base_dij_constant_rbe_dict["physical_dose"].flat[0] = csc_array(dense_mat)
    dij = Dij.model_validate(sample_base_dij_constant_rbe_dict)
    return dij


def test_RBExDose_effect_constructor(sample_dij_dense):
    rbe_x_dose = RBExDose(sample_dij_dense)
    assert isinstance(rbe_x_dose, FluenceDependentQuantity)
    assert rbe_x_dose.mode == "indirect"
    assert rbe_x_dose.scenarios == [0]
    assert rbe_x_dose._dij == sample_dij_dense.to_namespace(xp)
    assert rbe_x_dose.dim == 1
    assert format(rbe_x_dose.unit, "~") == "Gy"
    assert rbe_x_dose.identifier == "rbe_x_dose"
    assert rbe_x_dose.name == "RBExDose"
    assert rbe_x_dose.optional_dependencies == ("effect", "physical_dose")
    assert "effect" in rbe_x_dose.dependencies
    assert rbe_x_dose.path == "effect"


def test_RBExDose_effect_dense(sample_dij_dense):
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
    gamma = sample_dij_dense.alphax[:, 0] / sample_dij_dense.betax[:, 0] / 2
    rbe_x_dose_expected = np.zeros_like(effect)
    rbe_x_dose_expected = np.sqrt(gamma**2 + effect / sample_dij_dense.betax[:, 0]) - gamma
    assert np.allclose(ret_callable.flat[0], rbe_x_dose_expected)
    assert np.array_equal(ret_callable.flat[0], ret_compute.flat[0])

    cached_effect = rbe_x_dose.dependencies["effect"].compute(fluence)
    effect_before_gradient = np.array(cached_effect.flat[0], copy=True)
    dose_grad = xp.ones((1, 125), dtype=xp.float32)
    ret_deriv = rbe_x_dose.compute_chain_derivative(dose_grad, fluence)
    betax = xp.asarray(sample_dij_dense.betax[:, 0])
    fgrad = dose_grad / (2 * betax * (xp.asarray(rbe_x_dose_expected) + xp.asarray(gamma)))
    calc_derivative = rbe_x_dose.dependencies["effect"]._compute_chain_derivative_single_scenario(
        fgrad, 0
    )
    assert np.array_equal(cached_effect.flat[0], effect_before_gradient)
    assert np.array_equal(rbe_x_dose._w_grad_cache, fluence)
    assert np.array_equal(rbe_x_dose._qgrad_cache.flat[0], ret_deriv.flat[0])
    assert isinstance(ret_deriv, np.ndarray)
    assert ret_deriv.dtype == sample_dij_dense.physical_dose.dtype
    assert ret_deriv.shape == sample_dij_dense.physical_dose.shape
    assert np.allclose(ret_deriv.flat[0], calc_derivative)


def test_RBExDose_effect_sparse(sample_dij_sparse):
    rbe_x_dose = RBExDose(sample_dij_sparse)

    fluence = xp.arange(10, dtype=xp.float32)
    ret_callable = rbe_x_dose(fluence)
    assert np.array_equal(rbe_x_dose._w_cache, fluence)
    ret_compute = rbe_x_dose.compute(rbe_x_dose._w_cache)
    alpha_mat = sample_dij_sparse.alpha_dose.flat[0]
    beta_mat = sample_dij_sparse.sqrt_beta_dose.flat[0]
    effect = alpha_mat @ fluence + (beta_mat @ fluence) ** 2
    ix = sample_dij_sparse.betax[:, 0] > 0
    gamma = np.zeros_like(sample_dij_sparse.betax[:, 0])
    gamma[ix] = sample_dij_sparse.alphax[:, 0][ix] / sample_dij_sparse.betax[:, 0][ix] / 2
    rbe_x_dose_expected = np.zeros_like(effect)
    rbe_x_dose_expected[ix] = (
        np.sqrt(gamma[ix] ** 2 + effect[ix] / sample_dij_sparse.betax[:, 0][ix]) - gamma[ix]
    )
    assert np.allclose(ret_callable.flat[0], rbe_x_dose_expected)
    assert np.array_equal(ret_callable.flat[0], ret_compute.flat[0])

    cached_effect = rbe_x_dose.dependencies["effect"].compute(fluence)
    effect_before_gradient = np.array(cached_effect.flat[0], copy=True)
    dose_grad = xp.ones((1, 125), dtype=xp.float32)
    ret_deriv = rbe_x_dose.compute_chain_derivative(dose_grad, fluence)
    betax = xp.asarray(sample_dij_sparse.betax[:, 0])
    fgrad = dose_grad / (2 * betax * (xp.asarray(rbe_x_dose_expected) + xp.asarray(gamma)))
    calc_derivative = rbe_x_dose.dependencies["effect"]._compute_chain_derivative_single_scenario(
        fgrad, 0
    )
    assert np.array_equal(cached_effect.flat[0], effect_before_gradient)
    assert np.array_equal(rbe_x_dose._w_grad_cache, fluence)
    assert np.array_equal(rbe_x_dose._qgrad_cache.flat[0], ret_deriv.flat[0])
    assert isinstance(ret_deriv, np.ndarray)
    assert ret_deriv.dtype == sample_dij_sparse.physical_dose.dtype
    assert ret_deriv.shape == sample_dij_sparse.physical_dose.shape
    assert np.allclose(ret_deriv.flat[0], calc_derivative)


def test_RBExDose_constant_constructor(sample_dij_dense_constant_rbe):
    const_rbe = RBExDose(sample_dij_dense_constant_rbe)
    assert isinstance(const_rbe, FluenceDependentQuantity)
    assert const_rbe.mode == "indirect"
    assert const_rbe.scenarios == [0]
    assert const_rbe._dij == sample_dij_dense_constant_rbe.to_namespace(xp)
    assert const_rbe.dim == 1
    assert format(const_rbe.unit, "~") == "Gy"
    assert const_rbe.identifier == "rbe_x_dose"
    assert const_rbe.name == "RBExDose"
    assert const_rbe.path == "constant"
    assert "effect" not in const_rbe.dependencies


def test_RBExDose_constant_dense(sample_dij_dense_constant_rbe):
    const_rbe = RBExDose(sample_dij_dense_constant_rbe)
    rbe = 1.1

    fluence = xp.arange(10, dtype=xp.float32)
    ret_callable = const_rbe(fluence)
    assert np.array_equal(const_rbe._w_cache, fluence)
    ret_compute = const_rbe.compute(fluence)

    # assert isinstance(ret_callable, type(fluence))
    assert ret_callable.dtype == sample_dij_dense_constant_rbe.physical_dose.dtype
    assert ret_callable.shape == sample_dij_dense_constant_rbe.physical_dose.shape

    dij_mat = sample_dij_dense_constant_rbe.physical_dose.flat[0]
    assert np.allclose(ret_callable.flat[0], rbe * dij_mat @ fluence)
    assert np.array_equal(ret_callable.flat[0], ret_compute.flat[0])

    ret_deriv = const_rbe.compute_chain_derivative(xp.ones((1, 125), dtype=xp.float32), fluence)
    assert np.array_equal(const_rbe._w_grad_cache, fluence)
    assert np.array_equal(const_rbe._qgrad_cache.flat[0], ret_deriv.flat[0])
    # assert isinstance(ret_deriv, np.ndarray)
    assert ret_deriv.dtype == sample_dij_dense_constant_rbe.physical_dose.dtype
    assert ret_deriv.shape == sample_dij_dense_constant_rbe.physical_dose.shape
    assert np.allclose(ret_deriv.flat[0], rbe * dij_mat.T @ np.ones(125, dtype=np.float32))


def test_RBExDose_constant_sparse(sample_dij_sparse_constant_rbe):
    const_rbe = RBExDose(sample_dij_sparse_constant_rbe)
    rbe = 1.1

    fluence = xp.arange(10, dtype=xp.float32)
    ret_callable = const_rbe(fluence)
    assert np.array_equal(const_rbe._w_cache, fluence)
    ret_compute = const_rbe.compute(fluence)

    assert isinstance(ret_callable, np.ndarray)
    assert ret_callable.dtype == sample_dij_sparse_constant_rbe.physical_dose.dtype
    assert ret_callable.shape == sample_dij_sparse_constant_rbe.physical_dose.shape

    dij_mat = sample_dij_sparse_constant_rbe.physical_dose.flat[0]
    assert np.allclose(ret_callable.flat[0], rbe * dij_mat @ fluence)
    assert np.array_equal(ret_callable.flat[0], ret_compute.flat[0])

    ret_deriv = const_rbe.compute_chain_derivative(xp.ones((1, 125), dtype=xp.float32), fluence)
    assert np.array_equal(const_rbe._w_grad_cache, fluence)
    assert np.array_equal(const_rbe._qgrad_cache.flat[0], ret_deriv.flat[0])
    assert isinstance(ret_deriv, np.ndarray)
    assert ret_deriv.dtype == sample_dij_sparse_constant_rbe.physical_dose.dtype
    assert ret_deriv.shape == sample_dij_sparse_constant_rbe.physical_dose.shape
    assert np.allclose(ret_deriv.flat[0], rbe * dij_mat.T @ np.ones(125, dtype=np.float32))


def test_dij_legacy_rbe_becomes_constant_model(
    sample_dij_sparse_constant_rbe, sample_base_dij_constant_rbe_dict
):
    dij = sample_dij_sparse_constant_rbe
    assert isinstance(dij.bio_model, ConstantRBEModel)
    assert dij.rbe == pytest.approx(1.1)
    assert dij.model_dump()["bio_model"] == {"model": "constant_rbe", "rbe": 1.1}
    matrad = dij.to_matrad()
    assert matrad["RBE"] == pytest.approx(1.1)
    assert "bioModel" not in matrad

    # the input dict is left untouched by the legacy conversion
    assert sample_base_dij_constant_rbe_dict["rbe"] == 1.1
    assert "bio_model" not in sample_base_dij_constant_rbe_dict

    # matRad's zero placeholder means "no constant RBE"
    del sample_base_dij_constant_rbe_dict["rbe"]
    sample_base_dij_constant_rbe_dict["RBE"] = np.array([0])
    assert Dij.model_validate(sample_base_dij_constant_rbe_dict).bio_model is None


def test_dij_bio_model_spec_and_precedence(sample_dij_dense):
    # no model: the LQ matrices decide
    assert sample_dij_dense.bio_model is None
    assert RBExDose(sample_dij_dense).path == "effect"

    # a constant-RBE model wins over present LQ matrices
    dij = sample_dij_dense.model_copy()
    dij.bio_model = {"model": "constant_rbe", "rbe": 1.2}
    assert isinstance(dij.bio_model, ConstantRBEModel) and dij.rbe == 1.2
    rbe_x = RBExDose(dij)
    assert rbe_x.path == "constant"
    w = np.arange(10, dtype=np.float32)
    assert np.allclose(
        rbe_x(xp.asarray(w)).flat[0], 1.2 * (dij.physical_dose.flat[0] @ w), rtol=1e-6
    )
    result = dij.get_result_arrays_from_intensity(w)
    assert np.allclose(result["rbe_x_dose"], 1.2 * result["physical_dose"])
    assert "effect" in result

    # an alpha/beta model without matrices cannot give RBE-weighted dose
    dij.bio_model = Wedenberg()
    dij.alpha_dose = None
    dij.sqrt_beta_dose = None
    with pytest.raises(ValueError, match="needs alpha_dose"):
        RBExDose(dij)
    assert "rbe_x_dose" not in dij.get_result_arrays_from_intensity(w)


def test_dij_without_rbe_path_raises(sample_dij_dense):
    dij = sample_dij_dense.model_copy()
    dij.alpha_dose = None
    dij.sqrt_beta_dose = None
    with pytest.raises(ValueError, match="neither"):
        RBExDose(dij)


def test_lq_inverse_dose_supports_linear_limit_and_immutable_arrays():
    effect = xp.asarray([2.0, 2.0, 2.0])
    alpha_x = xp.asarray([1.0, 2.0, 0.0])
    beta_x = xp.asarray([1.0, 0.0, 0.0])

    result = lq_inverse_dose(effect, alpha_x, beta_x)

    assert np.allclose(np.asarray(result), [1.0, 1.0, 0.0])


def test_RBExDose_constant_reuses_the_physical_dose_quantity(sample_dij_sparse_constant_rbe):
    """The constant path scales the dose vector instead of the influence matrix."""
    const_rbe = RBExDose(sample_dij_sparse_constant_rbe)
    matrix = const_rbe._dij.physical_dose.flat[0]
    fluence = xp.arange(10, dtype=xp.float32)

    quantity = const_rbe(fluence)

    physical_dose = const_rbe.dependencies["physical_dose"]
    assert np.allclose(physical_dose.compute(fluence).flat[0], matrix @ fluence)
    assert np.allclose(quantity.flat[0], 1.1 * np.asarray(physical_dose._q_cache.flat[0]))
    # the dij matrix itself is untouched
    assert const_rbe._dij.physical_dose.flat[0] is matrix


def _finite_difference_gradient(quantity, fluence, d_quantity, eps=1e-4):
    """Numerical d(d_quantity . q)/dw by central differences."""
    fluence = np.asarray(fluence, dtype=np.float64)
    weights = np.asarray(d_quantity, dtype=np.float64).reshape(-1)
    grad = np.zeros_like(fluence)
    for i in range(fluence.size):
        plus, minus = fluence.copy(), fluence.copy()
        plus[i] += eps
        minus[i] -= eps
        q_plus = np.asarray(quantity.compute(xp.asarray(plus, dtype=xp.float64)).flat[0])
        q_minus = np.asarray(quantity.compute(xp.asarray(minus, dtype=xp.float64)).flat[0])
        grad[i] = float(weights @ (q_plus - q_minus)) / (2 * eps)
    return grad


@pytest.fixture
def sample_dij_float64_constant_rbe(sample_base_dij_constant_rbe_dict):
    rng = np.random.default_rng(0)
    sample_base_dij_constant_rbe_dict["physical_dose"].flat[0] = csc_array(rng.random((125, 10)))
    return Dij.model_validate(sample_base_dij_constant_rbe_dict)


@pytest.fixture
def sample_dij_float64_effect(sample_base_dij_dict):
    rng = np.random.default_rng(1)
    sample_base_dij_dict["alphax"] = np.full((125, 1), 0.1)
    sample_base_dij_dict["betax"] = np.full((125, 1), 0.05)
    sample_base_dij_dict["physical_dose"].flat[0] = csc_array(rng.random((125, 10)))
    sample_base_dij_dict["alpha_dose"].flat[0] = csc_array(0.3 * rng.random((125, 10)))
    sample_base_dij_dict["sqrt_beta_dose"].flat[0] = csc_array(0.2 * rng.random((125, 10)))
    sample_base_dij_dict["bio_model"] = {"model": "WED"}
    return Dij.model_validate(sample_base_dij_dict)


def test_RBExDose_constant_gradient_matches_finite_differences(
    sample_dij_float64_constant_rbe,
):
    quantity = RBExDose(sample_dij_float64_constant_rbe)
    assert quantity.path == "constant"

    fluence = np.linspace(0.2, 1.4, 10)
    rng = np.random.default_rng(2)
    d_quantity = rng.random((1, 125))

    analytic = np.asarray(
        quantity.compute_chain_derivative(
            xp.asarray(d_quantity, dtype=xp.float64), xp.asarray(fluence, dtype=xp.float64)
        ).flat[0]
    ).reshape(-1)
    numeric = _finite_difference_gradient(quantity, fluence, d_quantity)

    assert np.allclose(analytic, numeric, rtol=1e-5, atol=1e-7)


def test_RBExDose_effect_gradient_matches_finite_differences(sample_dij_float64_effect):
    quantity = RBExDose(sample_dij_float64_effect)
    assert quantity.path == "effect"

    fluence = np.linspace(0.2, 1.4, 10)
    rng = np.random.default_rng(3)
    d_quantity = rng.random((1, 125))

    # the effect path linearises around the current fluence, so evaluate it first
    quantity.compute(xp.asarray(fluence, dtype=xp.float64))
    analytic = np.asarray(
        quantity.compute_chain_derivative(
            xp.asarray(d_quantity, dtype=xp.float64), xp.asarray(fluence, dtype=xp.float64)
        ).flat[0]
    ).reshape(-1)
    numeric = _finite_difference_gradient(quantity, fluence, d_quantity)

    assert np.allclose(analytic, numeric, rtol=1e-5, atol=1e-7)
