"""Random scenario model (rndScen)."""

from pathlib import Path

import numpy as np
import pytest
import scipy.io
from pydantic import ValidationError

from pyRadPlan.scenarios import (
    IndependentGaussian,
    RandomScenarioModel,
    available_scenario_models,
    create_scenario_model,
    validate_scenario_model,
)

FIXTURE = Path(__file__).parents[1] / "data" / "scenarios" / "rndScen_structure.mat"


class _FakeCT:
    def __init__(self, n):
        self.num_of_ct_scen = n


def test_defaults_and_registry():
    m = RandomScenarioModel()
    assert m.short_name == "rndScen"
    assert (m.n_samples, m.include_nominal, m.seed) == (10, False, None)
    assert "rndScen" in available_scenario_models()
    assert isinstance(create_scenario_model("rndScen"), RandomScenarioModel)
    assert m.default_axes() == ("ct_phase", ("setup", "range"))
    s = m.scenarios()
    assert len(s) == 10
    assert s.axes == (
        ("ct_phase.index",),
        ("setup.x", "setup.y", "setup.z", "range.abs", "range.rel"),
    )
    assert s.layout.shape == (1, 10)
    assert len(s.tagged("nominal")) == 0
    for bad in (0, -3):
        with pytest.raises(ValidationError):
            RandomScenarioModel(n_samples=bad)


def test_weights_probabilities_and_phases():
    m = RandomScenarioModel(
        n_samples=6, seed=1, ct_phase={"phases": [0, 2], "probabilities": [0.3, 0.7]}
    )
    s = m.scenarios()
    assert len(s) == 12
    assert np.array_equal(s.ct_phases(), [0] * 6 + [2] * 6)
    assert np.allclose(s.weights, [0.3 / 6] * 6 + [0.7 / 6] * 6)
    assert np.allclose(s.values("setup")[:6], s.values("setup")[6:])
    assert np.allclose(s.values("range")[:6], s.values("range")[6:])

    dist = IndependentGaussian(m.dimensions())
    points = np.hstack([s.values("setup"), s.values("range")])
    phase_prob = np.repeat([0.3, 0.7], 6)
    np.testing.assert_allclose(s.probabilities, dist.pdf(points) * phase_prob, rtol=1e-10)


def test_include_nominal_replaces_first_sample_per_phase():
    m = RandomScenarioModel(
        n_samples=5, include_nominal=True, seed=3, ct_scen_prob=[(0, 0.5), (1, 0.5)]
    )
    s = m.scenarios()
    assert list(s.tagged("nominal")) == [0, 5]
    assert s.nominal_index() == 0
    assert np.all(s.values("setup")[[0, 5]] == 0)
    assert np.all(s.values("range")[[0, 5]] == 0)
    assert np.all(np.any(s.values("setup")[[1, 2, 3, 4]] != 0, axis=1))
    plain = RandomScenarioModel(n_samples=5, seed=3).scenarios()
    np.testing.assert_allclose(s.values("setup")[1:5], plain.values("setup")[1:5])


def test_moments_of_large_sample():
    m = RandomScenarioModel(n_samples=20000, seed=7, shift_sd=(1.0, 2.0, 3.0))
    s = m.scenarios()
    setup, rng = s.values("setup"), s.values("range")
    np.testing.assert_allclose(setup.std(axis=0), [1.0, 2.0, 3.0], rtol=0.15)
    np.testing.assert_allclose(rng.std(axis=0), [1.0, 0.035], rtol=0.15)
    assert np.all(np.abs(setup.mean(axis=0)) < 0.1 * np.array([1.0, 2.0, 3.0]) + 0.05)
    assert np.isclose(s.weights.sum(), 1.0)


def test_seed_reproducibility_and_refresh():
    a = RandomScenarioModel(n_samples=8, seed=42).scenarios()
    b = RandomScenarioModel(n_samples=8, seed=42).scenarios()
    assert np.array_equal(a.values("setup"), b.values("setup"))
    c = RandomScenarioModel(n_samples=8, seed=43).scenarios()
    assert not np.array_equal(a.values("setup"), c.values("setup"))

    seeded = RandomScenarioModel(n_samples=8, seed=42)
    first = seeded.scenarios()
    again = seeded.scenarios(refresh=True)
    assert again is not first
    assert np.array_equal(first.values("setup"), again.values("setup"))


def test_unseeded_model_is_cached_until_refresh_or_change():
    m = RandomScenarioModel(n_samples=8)
    first = m.scenarios()
    assert m.scenarios() is first
    refreshed = m.scenarios(refresh=True)
    assert refreshed is not first
    assert not np.array_equal(first.values("setup"), refreshed.values("setup"))
    assert m.scenarios() is refreshed
    m.n_samples = 9
    assert len(m.scenarios()) == 9
    m.invalidate()
    assert m.scenarios() is not refreshed


def test_legacy_mask_is_diagonal():
    m = RandomScenarioModel(
        n_samples=4, seed=0, ct_phase={"phases": [0, 1], "probabilities": [0.3, 0.7]}
    )
    m.validate_ct(_FakeCT(3))
    assert m.scen_mask.shape == (3, 4, 4)
    assert m.tot_num_shift_scen == m.tot_num_range_scen == 4
    assert m.tot_num_scen == 8
    expected = np.zeros((3, 4, 4), dtype=bool)
    expected[0] = expected[1] = np.eye(4, dtype=bool)
    assert np.array_equal(m.scen_mask, expected)
    assert np.array_equal(m.linear_mask, [[p, i, i] for p in (0, 1) for i in range(4)])
    assert m.scen_num(m.sub2scen_ix(1, 2, 2)) == 6


def test_matches_matrad_structure():
    ref = scipy.io.loadmat(FIXTURE)
    probs = ref["ctScenProb"]
    model = validate_scenario_model(
        {
            "model": "rndScen",
            "nSamples": int(ref["nSamples"].item()),
            "includeNominalScenario": bool(ref["includeNominalScenario"].item()),
            "ctScenProb": probs,
            "rangeRelSD": float(ref["rangeRelSD"].item()),
            "rangeAbsSD": float(ref["rangeAbsSD"].item()),
            "shiftSD": ref["shiftSD"].reshape(-1),
        },
        _FakeCT(int(ref["numOfAvailableCtScen"].item())),
    )
    assert isinstance(model, RandomScenarioModel)
    s = model.scenarios()
    assert np.array_equal(model.linear_mask, ref["linearMask"].astype(int) - 1)
    assert np.array_equal(model.scen_mask, ref["scenMask"].astype(bool))
    assert model.scen_mask.shape == tuple(int(v) for v in ref["scenMaskSize"].reshape(-1))
    np.testing.assert_allclose(model.scen_weight, ref["scenWeight"].reshape(-1), rtol=1e-10)
    assert model.tot_num_shift_scen == int(ref["totNumShiftScen"].item())
    assert model.tot_num_range_scen == int(ref["totNumRangeScen"].item())
    assert model.tot_num_scen == int(ref["totNumScen"].item()) == len(s)
    np.testing.assert_array_equal(model.scen_for_prob[:, 0], ref["scenForProb"][:, 0] - 1)
    # nominal scenario of each phase is exactly zero in both implementations
    assert np.all(ref["scenForProb"][[0, 4], 1:] == 0)
    assert np.all(model.scen_for_prob[[0, 4], 1:] == 0)


def test_matrad_dict_round_trip():
    m = validate_scenario_model(
        {
            "model": "rndScen",
            "nSamples": 7,
            "includeNominalScenario": 1,
            "ctScenProb": [(1, 1.0)],
        }
    )
    assert m.n_samples == 7 and m.include_nominal is True
    mat = m.to_matrad()
    assert mat["nSamples"] == 7 and mat["includeNominalScenario"] is True
    assert mat["model"] == "rndScen"
    back = validate_scenario_model(mat)
    assert back == m
    assert m.model_dump(context={"matRad": "mat-file"}) == mat
    assert RandomScenarioModel.model_validate(m.model_dump()) == m
