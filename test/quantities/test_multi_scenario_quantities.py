"""Scenario handling of the Dij containers, voxel parameter columns and the quantities."""

from typing import ClassVar, Literal

import array_api_compat
import array_api_strict as xp
import numpy as np
import pytest
from pydantic import ValidationError

from pyRadPlan.dij import Dij
from pyRadPlan.quantities import AlphaDose, Dose, Effect, RBExDose
from pyRadPlan.scenarios import (
    AFFECTS_VOXEL_PARAMS,
    CTPhaseUncertainty,
    RangeUncertainty,
    ScenarioSet,
    SetupUncertainty,
    UncertaintyDimension,
    available_dimensions,
    register_dimension,
)

N_VOX = 125
N_BIX = 10


class BioTestUncertainty(UncertaintyDimension):
    """Discrete stand-in for a biological dimension: only the voxel parameters change."""

    name: ClassVar[str] = "test_bio"
    components: ClassVar[tuple[str, ...]] = ("variant",)
    kind: ClassVar[Literal["continuous", "discrete"]] = "discrete"
    units: ClassVar[tuple[str, ...]] = ("",)
    affects: ClassVar[frozenset[str]] = frozenset({AFFECTS_VOXEL_PARAMS})

    def support(self) -> np.ndarray:
        return np.array([[0.0], [1.0]])

    def pmf(self) -> np.ndarray:
        return np.array([0.5, 0.5])


if "test_bio" not in available_dimensions():
    register_dimension(BioTestUncertainty)
BioDimension = available_dimensions()["test_bio"]


def _dimensions():
    return [
        CTPhaseUncertainty(phases=[0, 1], probabilities=[0.5, 0.5]),
        SetupUncertainty(),
        RangeUncertainty(),
    ]


def _phase_set(phases, bases=None, setup=None) -> ScenarioSet:
    points = {"ct_phase": np.asarray(phases, dtype=float)}
    if setup is not None:
        points["setup"] = np.asarray(setup, dtype=float)
    return ScenarioSet.from_points(_dimensions(), points, bases=bases)


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
    c = np.empty((len(mats),), dtype=object)
    for i, m in enumerate(mats):
        c[i] = m
    return c


def _random_mats(rng, n):
    return [rng.random((N_VOX, N_BIX), dtype=np.float32) for _ in range(n)]


def _columns(*values):
    return np.stack([v * np.ones(N_VOX) for v in values], axis=1).astype(np.float32)


def _lq_dij(rng, scenarios, num_mats, alphas, betas, empty=()):
    d = _grids()
    for q in ("physical_dose", "alpha_dose", "sqrt_beta_dose"):
        mats = _random_mats(rng, num_mats)
        for i in empty:
            mats[i] = None
        d[q] = _container(*mats)
    d["alphax"] = _columns(*alphas)
    d["betax"] = _columns(*betas)
    d["scenarios"] = scenarios
    return Dij.model_validate(d)


def _rbe_reference(dij, s, column, w):
    effect = dij.alpha_dose.flat[s] @ w + (dij.sqrt_beta_dose.flat[s] @ w) ** 2
    ax, bx = dij.alphax[:, column], dij.betax[:, column]
    return (np.sqrt(ax**2 + 4 * bx * effect) - ax) / (2 * bx)


@pytest.fixture
def two_scenario_dij():
    return _lq_dij(
        np.random.default_rng(0), _phase_set([0, 1]), 2, alphas=(0.1, 0.5), betas=(0.05, 0.02)
    )


@pytest.fixture
def alias_dij():
    """Two scenarios sharing one set of matrices but differing in their voxel parameters."""
    scen_set = ScenarioSet.from_points(
        [*_dimensions(), BioDimension()],
        {"test_bio": np.array([0.0, 1.0])},
        bases=[None, {"coefficients": {0: 1.0}}],
    )
    return _lq_dij(
        np.random.default_rng(1), scen_set, 2, alphas=(0.1, 0.5), betas=(0.05, 0.02), empty=(1,)
    )


@pytest.fixture
def derived_dij():
    """Two computed CT phases and a derived scenario 0.25 * s0 + 0.75 * s1."""
    scen_set = _phase_set([0, 1, 0], bases=[None, None, {"coefficients": {0: 0.25, 1: 0.75}}])
    return _lq_dij(
        np.random.default_rng(2), scen_set, 3, alphas=(0.1, 0.5), betas=(0.05, 0.02), empty=(2,)
    )


# ---------------------------------------------------------------------- Dij validation
def test_dij_promotes_1d_alphax():
    d = _grids()
    d["physical_dose"] = _container(np.ones((N_VOX, N_BIX), dtype=np.float32))
    d["alphax"] = np.ones(N_VOX)
    d["betax"] = np.ones(N_VOX)
    dij = Dij.model_validate(d)
    assert dij.alphax.shape == (N_VOX, 1)
    assert dij.betax.shape == (N_VOX, 1)
    assert dij.scenarios is None
    assert dij.num_of_scenarios == 1


def test_dij_without_set_accepts_one_or_per_slot_columns():
    d = _grids()
    d["physical_dose"] = _container(*[np.ones((N_VOX, N_BIX), dtype=np.float32)] * 2)
    d["alphax"] = np.ones((N_VOX, 1))
    shared = Dij.model_validate(d)
    assert [shared.voxel_param_column(i) for i in range(2)] == [0, 0]

    d["alphax"] = np.ones((N_VOX, 2))
    per_slot = Dij.model_validate(d)
    assert [per_slot.voxel_param_column(i) for i in range(2)] == [0, 1]

    d["alphax"] = np.ones((N_VOX, 3))
    with pytest.raises(ValidationError, match="number of scenarios"):
        Dij.model_validate(d)


def test_dij_scenario_set_columns_count_voxel_param_sets(two_scenario_dij):
    dij = two_scenario_dij
    assert dij.num_of_scenarios == 2
    assert dij.scenarios.num_voxel_param_sets == 2
    assert [dij.voxel_param_column(i) for i in range(2)] == [0, 1]

    d = _grids()
    d["physical_dose"] = _container(*_random_mats(np.random.default_rng(0), 2))
    d["scenarios"] = _phase_set([0, 1])
    d["alphax"] = _columns(0.1)
    with pytest.raises(ValidationError, match="voxel parameter set"):
        Dij.model_validate(d)


def test_dij_scenario_set_columns_shared_by_scenarios_of_one_phase():
    d = _grids()
    d["physical_dose"] = _container(*_random_mats(np.random.default_rng(0), 3))
    d["scenarios"] = _phase_set([0, 0, 1], setup=[[0, 0, 0], [2, 0, 0], [0, 0, 0]])
    d["alphax"] = _columns(0.1, 0.5)
    dij = Dij.model_validate(d)
    assert [dij.voxel_param_column(i) for i in range(3)] == [0, 0, 1]

    d["alphax"] = _columns(0.1, 0.5, 0.3)
    with pytest.raises(ValidationError, match="voxel parameter set"):
        Dij.model_validate(d)


def test_dij_rejects_container_size_mismatch():
    d = _grids()
    d["physical_dose"] = _container(*_random_mats(np.random.default_rng(0), 2))
    d["scenarios"] = _phase_set([0, 1, 1], setup=[[0, 0, 0], [0, 0, 0], [1, 0, 0]])
    with pytest.raises(ValidationError, match="scenario slot"):
        Dij.model_validate(d)


def test_dij_accepts_container_of_any_shape_with_matching_size():
    d = _grids()
    container = np.empty((1, 2, 1), dtype=object)
    for i, m in enumerate(_random_mats(np.random.default_rng(0), 2)):
        container.flat[i] = m
    d["physical_dose"] = container
    d["scenarios"] = _phase_set([0, 1])
    dij = Dij.model_validate(d)
    assert dij.num_of_scenarios == 2


def test_dij_alias_slot_shares_matrix(alias_dij):
    dij = alias_dij
    for q in ("physical_dose", "alpha_dose", "sqrt_beta_dose"):
        container = getattr(dij, q)
        assert container.flat[1] is container.flat[0]

    converted = dij.to_namespace(xp, keep_sparse_compat=False)
    assert converted.physical_dose.flat[1] is converted.physical_dose.flat[0]

    d = _grids()
    a, b = _random_mats(np.random.default_rng(0), 2)
    d["physical_dose"] = _container(a, b.copy())
    d["scenarios"] = _phase_set([0, 1], bases=[None, {"coefficients": {0: 1.0}}])
    with pytest.raises(ValidationError, match="share the matrix"):
        Dij.model_validate(d)

    d["physical_dose"] = _container(a, a)
    assert Dij.model_validate(d).physical_dose.flat[1] is a


def test_dij_derived_slot_must_be_empty():
    d = _grids()
    mats = _random_mats(np.random.default_rng(0), 3)
    d["physical_dose"] = _container(*mats)
    d["scenarios"] = _phase_set([0, 1, 0], bases=[None, None, {"coefficients": {0: 0.5, 1: 0.5}}])
    with pytest.raises(ValidationError, match="derived scenario"):
        Dij.model_validate(d)

    d["physical_dose"] = _container(mats[0], mats[1], None)
    assert Dij.model_validate(d).physical_dose.flat[2] is None

    d["physical_dose"] = _container(mats[0], None, None)
    with pytest.raises(ValidationError, match="computed scenario 1"):
        Dij.model_validate(d)


def test_attach_scenarios_checks_layout():
    d = _grids()
    d["physical_dose"] = _container(*_random_mats(np.random.default_rng(0), 2))
    d["alphax"] = _columns(0.1, 0.5)
    dij = Dij.model_validate(d)
    assert dij.scenarios is None

    with pytest.raises(ValueError, match="scenario slot"):
        dij.attach_scenarios(_phase_set([0]))
    assert dij.scenarios is None

    single_phase = _phase_set([0, 0], setup=[[0, 0, 0], [1, 0, 0]])
    with pytest.raises(ValueError, match="voxel parameter set"):
        dij.attach_scenarios(single_phase)

    assert dij.attach_scenarios(_phase_set([0, 1])) is dij
    assert dij.num_of_scenarios == 2
    assert dij.voxel_param_column(1) == 1


def test_scenario_model_in_validation_context():
    d = _grids()
    d["physical_dose"] = _container(*_random_mats(np.random.default_rng(0), 2))
    scen_set = _phase_set([0, 1])
    dij = Dij.model_validate(d, context={"scenario_model": scen_set})
    assert dij.scenarios is scen_set

    with pytest.raises(ValidationError, match="scenario slot"):
        Dij.model_validate(d, context={"scenario_model": _phase_set([0])})


# ---------------------------------------------------------------------- quantities
def test_default_scenarios():
    d = _grids()
    d["physical_dose"] = _container(*_random_mats(np.random.default_rng(0), 2))
    assert list(Dose(Dij.model_validate(d)).scenarios) == [0, 1]

    d["scenarios"] = _phase_set([0, 1, 0], bases=[None, None, {"coefficients": {0: 1.0}}])
    d["physical_dose"] = _container(*_random_mats(np.random.default_rng(0), 2), None)
    # derived (here: alias) scenarios are part of the set, so they are evaluated by default
    assert list(Dose(Dij.model_validate(d)).scenarios) == [0, 1, 2]


def test_rbe_x_dose_uses_voxel_param_column(two_scenario_dij):
    dij = two_scenario_dij
    rbe = RBExDose(dij)
    assert list(rbe.scenarios) == [0, 1]
    result = rbe(xp.arange(N_BIX, dtype=xp.float32))
    w = np.arange(N_BIX, dtype=np.float32)

    for s in range(2):
        expected = _rbe_reference(dij, s, dij.voxel_param_column(s), w)
        assert np.allclose(result.flat[s], expected, rtol=1e-5)

    assert not np.allclose(result.flat[0], result.flat[1])


def test_alpha_dose_indirect_uses_voxel_param_column(two_scenario_dij):
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
    result = effect(xp.arange(N_BIX, dtype=xp.float32))
    w = np.arange(N_BIX, dtype=np.float32)
    for s in range(2):
        d = two_scenario_dij
        expected = d.alpha_dose.flat[s] @ w + (d.sqrt_beta_dose.flat[s] @ w) ** 2
        assert np.allclose(result.flat[s], expected, rtol=1e-5)


def test_alias_scenarios_share_dose_but_not_voxel_params(alias_dij):
    dij = alias_dij
    assert dij.voxel_param_column(0) == 0
    assert dij.voxel_param_column(1) == 1
    fluence = xp.arange(N_BIX, dtype=xp.float32)
    w = np.arange(N_BIX, dtype=np.float32)

    dose = Dose(dij, scenarios=[0, 1])
    dose_result = dose(fluence)
    assert dose_result.flat[1] is dose_result.flat[0]

    rbe = RBExDose(dij, scenarios=[0, 1])
    result = rbe(fluence)
    assert (
        rbe.dependencies["effect"].compute(fluence).flat[1]
        is (rbe.dependencies["effect"].compute(fluence).flat[0])
    )
    for s in range(2):
        expected = _rbe_reference(dij, 0, dij.voxel_param_column(s), w)
        assert np.allclose(result.flat[s], expected, rtol=1e-5)
    assert not np.allclose(result.flat[0], result.flat[1])

    grad = xp.ones((1, N_VOX), dtype=xp.float32)
    deriv = rbe.compute_chain_derivative(grad, fluence)
    assert not np.allclose(np.asarray(deriv.flat[0]), np.asarray(deriv.flat[1]))

    alpha_dose = AlphaDose(dij.model_copy(update={"alpha_dose": None}), scenarios=[0, 1])
    alpha_result = alpha_dose(fluence)
    for s in range(2):
        expected = dij.alphax[:, s] * (dij.physical_dose.flat[0] @ w)
        assert np.allclose(alpha_result.flat[s], expected, rtol=1e-5)

    result_arrays = dij.get_result_arrays_from_intensity(w, scenario_index=1)
    assert np.allclose(result_arrays["rbe_x_dose"], result.flat[1], rtol=1e-5)


@pytest.mark.parametrize("quantity_cls", [Dose, Effect, RBExDose])
def test_derived_scenario_is_linear_combination(derived_dij, quantity_cls):
    quantity = quantity_cls(derived_dij, scenarios=[0, 1, 2])
    fluence = xp.arange(N_BIX, dtype=xp.float32)
    result = quantity(fluence)
    expected = 0.25 * np.asarray(result.flat[0]) + 0.75 * np.asarray(result.flat[1])
    assert np.allclose(np.asarray(result.flat[2]), expected, rtol=1e-5)

    grad = xp.ones((1, N_VOX), dtype=xp.float32)
    deriv = quantity.compute_chain_derivative(grad, fluence)
    expected_grad = 0.25 * np.asarray(deriv.flat[0]) + 0.75 * np.asarray(deriv.flat[1])
    assert np.allclose(np.asarray(deriv.flat[2]), expected_grad, rtol=1e-5)

    reference = quantity_cls(derived_dij, scenarios=[0, 1])
    ref_result = reference(fluence)
    for s in range(2):
        assert np.allclose(np.asarray(ref_result.flat[s]), np.asarray(result.flat[s]))


def test_per_slot_cotangents_fold_derived_slots_onto_their_basis(derived_dij, monkeypatch):
    """One back-projection per computed slot; the derived slot's cotangent lands on its basis."""
    dose = Dose(derived_dij, scenarios=[0, 1, 2])
    fluence = xp.arange(N_BIX, dtype=xp.float32)
    rng = np.random.default_rng(5)
    g = [rng.standard_normal((1, N_VOX)).astype(np.float32) for _ in range(3)]
    cotangents = np.empty((3,), dtype=object)
    for s in range(3):
        cotangents.flat[s] = xp.asarray(g[s])

    calls = []
    original = Dose._compute_chain_derivative_single_scenario

    def counted(self, d_quantity, scenario_index):
        calls.append(scenario_index)
        return original(self, d_quantity, scenario_index)

    monkeypatch.setattr(Dose, "_compute_chain_derivative_single_scenario", counted)
    deriv = dose.compute_chain_derivative(cotangents, fluence)

    assert sorted(calls) == [0, 1]
    assert deriv.flat[2] is None
    matrices = [derived_dij.physical_dose.flat[s] for s in range(2)]
    expected = (g[0] + 0.25 * g[2]) @ matrices[0] + (g[1] + 0.75 * g[2]) @ matrices[1]
    total = np.asarray(deriv.flat[0]) + np.asarray(deriv.flat[1])
    assert np.allclose(total, expected, rtol=1e-4, atol=1e-5)

    # a slot without a cotangent is skipped, its basis slot still carries the derived one
    cotangents.flat[0] = None
    dose._w_grad_cache = None
    deriv = dose.compute_chain_derivative(cotangents, fluence)
    assert np.allclose(
        np.asarray(deriv.flat[0]), (0.25 * g[2]) @ matrices[0], rtol=1e-4, atol=1e-5
    )


def test_derived_scenario_alone_computes_its_basis(derived_dij):
    dose = Dose(derived_dij, scenarios=[2])
    result = dose(xp.arange(N_BIX, dtype=xp.float32))
    w = np.arange(N_BIX, dtype=np.float32)
    expected = 0.25 * (derived_dij.physical_dose.flat[0] @ w) + 0.75 * (
        derived_dij.physical_dose.flat[1] @ w
    )
    assert np.allclose(np.asarray(result.flat[2]), expected, rtol=1e-5)

    with pytest.raises(ValueError, match="derived"):
        derived_dij.get_result_arrays_from_intensity(w, scenario_index=2)


# ---------------------------------------------------------------------- extend_scenarios
def _single_dij(rng, scen_set, alphas):
    return _lq_dij(rng, scen_set, len(scen_set), alphas=alphas, betas=[0.05] * len(alphas))


def test_extend_scenarios_appends_containers_and_columns():
    rng = np.random.default_rng(3)
    first = _single_dij(rng, _phase_set([0]), alphas=(0.1,))
    second = _single_dij(rng, _phase_set([1]), alphas=(0.5,))

    merged = first.extend_scenarios(second)
    assert merged.num_of_scenarios == 2
    assert merged.physical_dose.shape == (2,)
    assert merged.physical_dose.flat[0] is first.physical_dose.flat[0]
    assert merged.physical_dose.flat[1] is second.physical_dose.flat[0]
    assert [merged.voxel_param_column(i) for i in range(2)] == [0, 1]
    assert np.allclose(merged.alphax[:, merged.voxel_param_column(1)], 0.5)
    assert np.allclose(merged.scenarios.ct_phases(), [0, 1])

    rbe = RBExDose(merged)
    result = rbe(xp.arange(N_BIX, dtype=xp.float32))
    w = np.arange(N_BIX, dtype=np.float32)
    assert np.allclose(result.flat[1], _rbe_reference(second, 0, 0, w), rtol=1e-5)


def test_extend_scenarios_merges_shared_voxel_param_sets():
    rng = np.random.default_rng(4)
    first = _single_dij(rng, _phase_set([0, 1]), alphas=(0.1, 0.5))
    second = _single_dij(rng, _phase_set([0], setup=[[1.0, 0, 0]]), alphas=(0.1,))

    merged = first.extend_scenarios(second)
    assert merged.num_of_scenarios == 3
    assert merged.alphax.shape == (N_VOX, 2)
    assert [merged.voxel_param_column(i) for i in range(3)] == [0, 1, 0]
    for i in range(3):
        expected = 0.5 if merged.scenarios.ct_phase(i) == 1 else 0.1
        assert np.allclose(merged.alphax[:, merged.voxel_param_column(i)], expected)

    conflicting = _single_dij(rng, _phase_set([0], setup=[[1.0, 0, 0]]), alphas=(0.3,))
    with pytest.raises(ValueError, match="columns differ"):
        first.extend_scenarios(conflicting)


def test_extend_scenarios_keeps_derived_bases():
    rng = np.random.default_rng(5)
    first = _single_dij(rng, _phase_set([0]), alphas=(0.1,))
    second = _lq_dij(
        rng,
        _phase_set([1, 1], bases=[None, {"coefficients": {0: 1.0}}], setup=[[0, 0, 0], [1, 0, 0]]),
        2,
        alphas=(0.5,),
        betas=(0.05,),
        empty=(1,),
    )
    merged = first.extend_scenarios(second)
    assert merged.scenarios.bases[2].coefficients == {1: 1.0}
    assert merged.physical_dose.flat[2] is merged.physical_dose.flat[1]


def test_extend_scenarios_rejects_incompatible_dijs():
    rng = np.random.default_rng(6)
    first = _single_dij(rng, _phase_set([0]), alphas=(0.1,))
    second = _single_dij(rng, _phase_set([1]), alphas=(0.5,))

    second.beam_num = np.zeros((N_BIX,), dtype=np.int64)
    other_bixels = second.model_copy(update={"bixel_num": np.arange(N_BIX) + 1})
    with pytest.raises(ValueError, match="bookkeeping"):
        first.extend_scenarios(other_bixels)

    no_set = second.model_copy(update={"scenarios": None})
    with pytest.raises(ValueError, match="scenario set"):
        first.extend_scenarios(no_set)

    fewer = second.model_copy(update={"alpha_dose": None})
    with pytest.raises(ValueError, match="same quantities"):
        first.extend_scenarios(fewer)


# ---------------------------------------------------------------------- results
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


def test_result_arrays_use_voxel_param_column(two_scenario_dij):
    dij = two_scenario_dij
    w = np.arange(N_BIX, dtype=np.float32)
    for s in range(2):
        out = dij.get_result_arrays_from_intensity(w, scenario_index=s)
        assert np.allclose(out["rbe_x_dose"], _rbe_reference(dij, s, s, w), rtol=1e-5)


def test_result_arrays_preserve_array_api_backend(two_scenario_dij):
    dij = two_scenario_dij.to_namespace(xp, keep_sparse_compat=False)
    result = dij.get_result_arrays_from_intensity(xp.arange(N_BIX, dtype=xp.float32))

    for key in ("physical_dose", "effect", "alpha", "beta", "rbe_x_dose", "rbe"):
        assert array_api_compat.is_array_api_strict_namespace(result[key].__array_namespace__())
