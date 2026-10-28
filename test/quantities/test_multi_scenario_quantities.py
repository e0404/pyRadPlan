"""Scenario-column handling of alphax/betax in the Dij and the LQ quantities."""

import array_api_compat
import array_api_strict as xp
import numpy as np
import pytest
from pydantic import ValidationError

from pyRadPlan.dij import Dij
from pyRadPlan.quantities import AlphaDose, Effect
from pyRadPlan.quantities import RBExDose

N_VOX = 125
N_BIX = 10


def _grids():
    return {
        "ct_grid": {
            "resolution": {"x": 1.5, "y": 1.5, "z": 1.5},
            "dimensions": (10, 10, 10),
            "num_of_voxels": 1000,
        },
        "dose_grid": {
            "resolution": {"x": 3.0, "y": 3.0, "z": 3.0},
            "dimensions": (5, 5, 5),
            "num_of_voxels": N_VOX,
        },
        "num_of_beams": 1,
        "total_num_of_bixels": N_BIX,
        "bixel_num": np.arange(N_BIX),
        "ray_num": np.arange(N_BIX),
        "beam_num": np.zeros((N_BIX,), dtype=np.int64),
    }


def _container(*mats):
    c = np.empty((len(mats), 1, 1), dtype=object)
    for i, m in enumerate(mats):
        c.flat[i] = m
    return c


@pytest.fixture
def two_scenario_dij():
    rng = np.random.default_rng(0)
    d = _grids()
    d["physical_dose"] = _container(
        rng.random((N_VOX, N_BIX), dtype=np.float32), rng.random((N_VOX, N_BIX), dtype=np.float32)
    )
    d["alpha_dose"] = _container(
        rng.random((N_VOX, N_BIX), dtype=np.float32), rng.random((N_VOX, N_BIX), dtype=np.float32)
    )
    d["sqrt_beta_dose"] = _container(
        rng.random((N_VOX, N_BIX), dtype=np.float32), rng.random((N_VOX, N_BIX), dtype=np.float32)
    )
    alphax = np.stack([0.1 * np.ones(N_VOX), 0.5 * np.ones(N_VOX)], axis=1).astype(np.float32)
    betax = np.stack([0.05 * np.ones(N_VOX), 0.02 * np.ones(N_VOX)], axis=1).astype(np.float32)
    d["alphax"] = alphax
    d["betax"] = betax
    return Dij.model_validate(d)


def test_dij_promotes_1d_alphax():
    d = _grids()
    d["physical_dose"] = _container(np.ones((N_VOX, N_BIX), dtype=np.float32))
    d["alphax"] = np.ones(N_VOX)
    d["betax"] = np.ones(N_VOX)
    dij = Dij.model_validate(d)
    assert dij.alphax.shape == (N_VOX, 1)
    assert dij.betax.shape == (N_VOX, 1)


def test_dij_rejects_scenario_mismatch():
    d = _grids()
    d["physical_dose"] = _container(
        np.ones((N_VOX, N_BIX), dtype=np.float32), np.ones((N_VOX, N_BIX), dtype=np.float32)
    )
    d["alphax"] = np.ones((N_VOX, 1))
    d["betax"] = np.ones((N_VOX, 1))
    with pytest.raises(ValidationError, match="number of scenarios"):
        Dij.model_validate(d)


def test_rbe_x_dose_uses_scenario_column(two_scenario_dij):
    dij = two_scenario_dij
    rbe = RBExDose(dij, scenarios=[0, 1])
    fluence = xp.arange(N_BIX, dtype=xp.float32)
    result = rbe(fluence)
    w = np.arange(N_BIX, dtype=np.float32)

    for s in range(2):
        effect = dij.alpha_dose.flat[s] @ w + (dij.sqrt_beta_dose.flat[s] @ w) ** 2
        ax, bx = dij.alphax[:, s], dij.betax[:, s]
        expected = (np.sqrt(ax**2 + 4 * bx * effect) - ax) / (2 * bx)
        assert np.allclose(result.flat[s], expected, rtol=1e-5)

    # scenarios differ in alphax/betax, so the results must not coincide
    assert not np.allclose(result.flat[0], result.flat[1])


def test_alpha_dose_indirect_uses_scenario_column(two_scenario_dij):
    dij = two_scenario_dij.model_copy()
    dij.alpha_dose = None  # force the alpha_x * dose fallback
    alpha_dose = AlphaDose(dij, scenarios=[0, 1])
    assert alpha_dose.mode == "indirect"

    fluence = xp.arange(N_BIX, dtype=xp.float32)
    result = alpha_dose(fluence)
    w = np.arange(N_BIX, dtype=np.float32)
    for s in range(2):
        expected = dij.alphax[:, s] * (dij.physical_dose.flat[s] @ w)
        assert np.allclose(result.flat[s], expected, rtol=1e-5)

    grad = xp.ones((1, N_VOX), dtype=xp.float32)
    deriv = alpha_dose.compute_chain_derivative(grad, fluence)
    for s in range(2):
        expected = dij.physical_dose.flat[s].T @ (dij.alphax[:, s] * np.ones(N_VOX, np.float32))
        assert np.allclose(np.asarray(deriv.flat[s]).ravel(), expected, rtol=1e-5)


def test_effect_scenarios_independent(two_scenario_dij):
    effect = Effect(two_scenario_dij, scenarios=[0, 1])
    fluence = xp.arange(N_BIX, dtype=xp.float32)
    result = effect(fluence)
    w = np.arange(N_BIX, dtype=np.float32)
    for s in range(2):
        d = two_scenario_dij
        expected = d.alpha_dose.flat[s] @ w + (d.sqrt_beta_dose.flat[s] @ w) ** 2
        assert np.allclose(result.flat[s], expected, rtol=1e-5)


def test_result_scaling_to_fractions(two_scenario_dij):
    dij = two_scenario_dij
    w = np.arange(N_BIX, dtype=np.float32)
    per_fraction = dij.get_result_arrays_from_intensity(w)
    total = dij.get_result_arrays_from_intensity(w, num_of_fractions=30)

    for key in ("physical_dose", "effect", "rbe_x_dose", "alpha_dose"):
        assert np.allclose(total[key], 30 * per_fraction[key])
    assert np.allclose(total["sqrt_beta_dose"], np.sqrt(30) * per_fraction["sqrt_beta_dose"])
    for key in ("rbe", "alpha", "beta"):
        assert np.allclose(total[key], per_fraction[key])
    assert np.allclose(total["physical_dose_beam"][0], 30 * per_fraction["physical_dose_beam"][0])
    assert np.allclose(total["effect"], total["alpha_dose"] + total["sqrt_beta_dose"] ** 2)


def test_result_arrays_preserve_array_api_backend(two_scenario_dij):
    dij = two_scenario_dij.to_namespace(xp, keep_sparse_compat=False)
    result = dij.get_result_arrays_from_intensity(xp.arange(N_BIX, dtype=xp.float32))

    for key in ("physical_dose", "effect", "alpha", "beta", "rbe_x_dose", "rbe"):
        assert array_api_compat.is_array_api_strict_namespace(result[key].__array_namespace__())
