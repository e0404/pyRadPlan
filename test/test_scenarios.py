import numpy as np
import SimpleITK as sitk
from pyRadPlan.scenarios import ScenarioModel, NominalScenario, validate_scenario_model
from pyRadPlan.ct import CT
import pytest


@pytest.fixture
def sample_model_dict_matrad():
    model_dict = {
        "model": "nomScen",
        "rangeRelSD": 5.0,
        "rangeAbsSD": 2.5,
        "shiftSD": (3.0, 3.0, 3.0),
        # TODO!: MatRad export has e.g. (1, 0.5) as indexing.
        # The Model does not yet respect that (converting 1 to 0)
        # because it can't distinguish whether its from the user or matRad. (adding '1' or keeping '0')
        "ctScenProb": [(1, 0.5)],
        "wcSigma": 5.0,
    }
    return model_dict


def helper_mvar_gauss(scenarioModel):
    # Computes multivariate Gaussian probability for scenario model
    # Can be used to test correct scenario probabilities. Also considers used
    # ct phases for probability computation (uncorrelated)

    sigma = np.diag(
        np.hstack(
            (
                scenarioModel.shift_sd,
                scenarioModel.range_abs_sd,
                scenarioModel.range_rel_sd / 100.0,
            )
        )
        ** 2
    )
    d = sigma.shape[0]
    cs = np.linalg.cholesky(sigma)

    tmp_standardized = np.linalg.solve(cs, scenarioModel.scen_for_prob[:, 1:].T).T
    p = (
        (2 * np.pi) ** (-d / 2)
        * np.exp(-0.5 * np.sum(tmp_standardized**2, axis=1))
        / np.prod(np.diag(cs))
    )

    # Now multiply with the phase probability
    tmp_phase_prob = np.array(
        [
            [x[1] for x in scenarioModel.ct_scen_prob if x[0] == phase]
            for phase in scenarioModel.scen_for_prob[:, 0]
        ]
    )
    p = p * tmp_phase_prob

    return p


@pytest.fixture
def sample_ct():
    """Create a sample SimpleITK image for testing."""
    sample_array = np.random.rand(50, 100, 100) * 1000  # Random HU values
    image = sitk.GetImageFromArray(sample_array)
    image.SetOrigin((0, 0, 0))
    image.SetSpacing((1, 1, 2))
    image.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))

    ct = CT(cube_hu=image)
    return ct


# Test Abstract Scenario Model
def test_scenario_model():
    with pytest.raises(TypeError):
        ScenarioModel()


def test_nominal_scenario_construct():
    scenario = NominalScenario()
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
    assert np.allclose(scenario.scen_prob, helper_mvar_gauss(scenario))
    assert np.array_equal(
        scenario.scen_for_prob,
        np.hstack((np.zeros((1, 1), dtype=float), np.zeros((1, 5), dtype=float))),
    )
    assert np.array_equal(scenario.scen_weight, np.ones((1,), dtype=float))


def test_nominal_scenario_construct_with_ct(sample_ct):
    scenario = NominalScenario(sample_ct)
    assert isinstance(scenario, ScenarioModel)
    assert isinstance(scenario, NominalScenario)
    assert scenario.name == "Nominal Scenario"
    assert scenario.short_name == "nomScen"
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
    assert np.allclose(scenario.scen_prob, helper_mvar_gauss(scenario))
    assert np.array_equal(
        scenario.scen_for_prob,
        np.hstack((np.zeros((1, 1), dtype=float), np.zeros((1, 5), dtype=float))),
    )
    assert np.array_equal(scenario.scen_weight, np.ones((1,), dtype=float))


def test_nominal_scenario_list():
    scenario = NominalScenario()
    scenario.list_all_scenarios()


def test_nominal_scenario_creation_by_name():
    scenario = validate_scenario_model("nomScen")
    assert isinstance(scenario, ScenarioModel)
    assert isinstance(scenario, NominalScenario)

    with pytest.raises(ValueError):
        validate_scenario_model("unknown")

    with pytest.raises(NotImplementedError):
        validate_scenario_model("wcScen")

    with pytest.raises(NotImplementedError):
        validate_scenario_model("impScen")

    with pytest.raises(NotImplementedError):
        validate_scenario_model("rndScen")


def test_nominal_scenario_creation_by_dict():
    model_dict = {
        "model": "nomScen",
        "range_rel_sd": 5.0,
        "range_abs_sd": 2.5,
        "shift_sd": (3.0, 3.0, 3.0),
        "ct_scen_prob": [(0, 0.5)],
        "wc_sigma": 5.0,
    }

    scenario = validate_scenario_model(model_dict)

    assert isinstance(scenario, ScenarioModel)
    assert isinstance(scenario, NominalScenario)
    assert scenario.range_rel_sd == 5.0
    assert scenario.range_abs_sd == 2.5
    assert np.array_equal(scenario.shift_sd, (3.0, 3.0, 3.0))
    assert scenario.ct_scen_prob == [(0, 0.5)]
    assert scenario.wc_sigma == 5.0

    return_dict = scenario.model_dump()

    model_dict.pop("model")

    for key in model_dict.keys():
        assert key in return_dict
        assert return_dict[key] == model_dict[key]


def test_nominal_scenario_creation_by_matrad_dict(sample_model_dict_matrad):
    scenario = validate_scenario_model(sample_model_dict_matrad)

    assert isinstance(scenario, ScenarioModel)
    assert isinstance(scenario, NominalScenario)
    assert scenario.range_rel_sd == 5.0
    assert scenario.range_abs_sd == 2.5
    assert np.array_equal(scenario.shift_sd, [3.0, 3.0, 3.0])
    assert scenario.ct_scen_prob == [(0, 0.5)]
    assert scenario.wc_sigma == 5.0

    return_dict = scenario.to_matrad()

    sample_model_dict_matrad.pop("model")

    for key in sample_model_dict_matrad.keys():
        assert key in return_dict
        assert return_dict[key] == sample_model_dict_matrad[key]


def test_nominal_scenario_extract_single_scenario():
    scenario = NominalScenario()
    assert isinstance(scenario.extract_single_scenario(0), ScenarioModel)
    assert isinstance(scenario.extract_single_scenario(0), NominalScenario)

    with pytest.raises(NotImplementedError):
        scenario.extract_single_scenario(1)
