"""Legacy (matRad-style) scenario model interface kept during the migration to ScenarioSet."""

import warnings

import numpy as np
import pytest
import SimpleITK as sitk

from pyRadPlan.ct import CT
from pyRadPlan.scenarios import NominalScenario, NominalScenarioModel, ScenarioModel
from pyRadPlan.scenarios import validate_scenario_model


@pytest.fixture
def sample_model_dict_matrad():
    return {
        "model": "nomScen",
        "rangeRelSD": 5.0,
        "rangeAbsSD": 2.5,
        "shiftSD": (3.0, 3.0, 3.0),
        "ctScenProb": [(1, 0.5)],
        "wcSigma": 5.0,
    }


def helper_mvar_gauss(scenario_model):
    """Multivariate Gaussian probability including the ct phase probability (uncorrelated)."""
    sigma = np.diag(
        np.hstack(
            (
                scenario_model.shift_sd,
                scenario_model.range_abs_sd,
                scenario_model.range_rel_sd / 100.0,
            )
        )
        ** 2
    )
    d = sigma.shape[0]
    cs = np.linalg.cholesky(sigma)
    tmp_standardized = np.linalg.solve(cs, scenario_model.scen_for_prob[:, 1:].T).T
    p = (
        (2 * np.pi) ** (-d / 2)
        * np.exp(-0.5 * np.sum(tmp_standardized**2, axis=1))
        / np.prod(np.diag(cs))
    )
    tmp_phase_prob = np.array(
        [
            [x[1] for x in scenario_model.ct_scen_prob if x[0] == phase][0]
            for phase in scenario_model.scen_for_prob[:, 0]
        ]
    )
    return p * tmp_phase_prob


@pytest.fixture
def sample_ct():
    sample_array = np.random.rand(50, 100, 100) * 1000
    image = sitk.GetImageFromArray(sample_array)
    image.SetOrigin((0, 0, 0))
    image.SetSpacing((1, 1, 2))
    image.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))
    return CT(cube_hu=image)


def _check_nominal_legacy_view(scenario):
    assert isinstance(scenario, ScenarioModel)
    assert isinstance(scenario, NominalScenario)
    assert scenario.name == "Nominal Scenario"
    assert scenario.short_name == "nomScen"
    assert scenario.ct_scen_prob == [(0, 1.0)]
    assert scenario.num_of_ct_scen == 1
    assert scenario.tot_num_scen == 1
    assert scenario.tot_num_shift_scen == 1
    assert scenario.tot_num_range_scen == 1
    assert scenario.rel_range_shift == 0.0
    assert scenario.abs_range_shift == 0.0
    assert np.array_equal(scenario.iso_shift, np.array([[0.0, 0.0, 0.0]]))
    assert scenario.max_abs_range_shift == 0.0
    assert scenario.max_rel_range_shift == 0.0
    assert np.array_equal(scenario.scen_mask, np.ones((1, 1, 1)).astype(bool))
    assert np.array_equal(scenario.linear_mask, np.array([[0, 0, 0]]))
    assert np.array_equal(scenario.ct_scen_ix, [0])
    assert np.allclose(scenario.scen_prob, helper_mvar_gauss(scenario))
    assert np.array_equal(scenario.scen_for_prob, np.zeros((1, 6)))
    assert np.array_equal(scenario.scen_weight, np.ones((1,), dtype=float))
    assert scenario.sub2scen_ix(0, 0, 0) == 0
    assert scenario.scen_num(0) == 0


def test_nominal_scenario_construct():
    _check_nominal_legacy_view(NominalScenario())


def test_nominal_scenario_construct_with_ct(sample_ct):
    scenario = NominalScenarioModel()
    scenario.validate_ct(sample_ct)
    _check_nominal_legacy_view(scenario)
    assert scenario.num_of_available_ct_scen == 1


def test_legacy_view_with_two_phases():
    scenario = NominalScenarioModel(ct_scen_prob=[(0, 0.5), (2, 0.5)])
    assert scenario.num_of_ct_scen == 2
    assert scenario.num_of_available_ct_scen == 3
    assert scenario.tot_num_scen == 2
    assert scenario.scen_mask.shape == (3, 1, 1)
    assert np.array_equal(scenario.scen_mask[:, 0, 0], [True, False, True])
    assert np.array_equal(scenario.linear_mask, [[0, 0, 0], [2, 0, 0]])
    assert np.array_equal(scenario.ct_scen_ix, [0, 2])
    assert scenario.sub2scen_ix(2, 0, 0) == 2
    assert scenario.scen_num(2) == 1
    with pytest.raises(IndexError):
        scenario.scen_num(1)
    assert np.allclose(scenario.scen_prob, helper_mvar_gauss(scenario))
    assert np.allclose(scenario.scen_weight, [0.5, 0.5])
    assert np.array_equal(scenario.update_scenarios(), scenario.scen_for_prob)


def test_nominal_scenario_list(capsys):
    NominalScenario().list_all_scenarios()
    assert "nominal" in capsys.readouterr().out


def test_nominal_scenario_creation_by_name():
    scenario = validate_scenario_model("nomScen")
    assert isinstance(scenario, NominalScenario)
    with pytest.raises(ValueError):
        validate_scenario_model("unknown")


def test_nominal_scenario_creation_by_legacy_dict():
    model_dict = {
        "model": "nomScen",
        "range_rel_sd": 5.0,
        "range_abs_sd": 2.5,
        "shift_sd": (3.0, 3.0, 3.0),
        "ct_scen_prob": [(0, 0.5)],
        "wc_sigma": 5.0,
    }
    scenario = validate_scenario_model(model_dict)
    assert isinstance(scenario, NominalScenario)
    assert scenario.range_rel_sd == 5.0
    assert scenario.range_abs_sd == 2.5
    assert scenario.shift_sd == (3.0, 3.0, 3.0)
    assert scenario.ct_scen_prob == [(0, 0.5)]
    assert scenario.wc_sigma == 5.0
    # the dump is nested now, but validates back to the same model
    assert validate_scenario_model(scenario.model_dump()) == scenario


def test_nominal_scenario_creation_by_matrad_dict(sample_model_dict_matrad):
    scenario = validate_scenario_model(sample_model_dict_matrad)
    assert isinstance(scenario, NominalScenario)
    assert scenario.range_rel_sd == 5.0
    assert scenario.range_abs_sd == 2.5
    assert scenario.shift_sd == (3.0, 3.0, 3.0)
    assert scenario.ct_scen_prob == [(0, 0.5)]
    assert scenario.wc_sigma == 5.0

    return_dict = scenario.to_matrad()
    for key, value in sample_model_dict_matrad.items():
        assert key in return_dict
        assert np.all(np.asarray(return_dict[key]) == np.asarray(value))


def test_scenario_serialization_emits_no_pydantic_warning():
    scenario = NominalScenario()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        scenario.model_dump()
        scenario.model_dump_json()
        scenario.to_matrad()
    messages = [str(w.message) for w in caught]
    assert not any("PydanticSerializationUnexpectedValue" in m for m in messages), messages
    assert np.array_equal(scenario.rel_range_shift, np.zeros(1))
    assert np.array_equal(scenario.abs_range_shift, np.zeros(1))


def test_nominal_scenario_extract_single_scenario():
    scenario = NominalScenario()
    single = scenario.extract_single_scenario(0)
    assert isinstance(single, ScenarioModel)
    assert single.tot_num_scen == 1
    with pytest.raises(IndexError):
        scenario.extract_single_scenario(1)
