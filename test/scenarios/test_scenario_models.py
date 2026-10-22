"""ScenarioModel base behaviour, the nominal and explicit models and the factory."""

import numpy as np
import pytest
from pydantic import ValidationError

from pyRadPlan.scenarios import (
    MATRAD_AXES,
    CTPhaseUncertainty,
    ExplicitScenarioModel,
    IndependentGaussian,
    MultivariateGaussian,
    NominalScenario,
    NominalScenarioModel,
    RangeUncertainty,
    ScenarioModel,
    ScenarioSet,
    SetupUncertainty,
    available_scenario_models,
    create_scenario_model,
    get_scenario_model_class,
    register_scenario_model,
    validate_scenario_model,
)


class _FakeCT:
    def __init__(self, n):
        self.num_of_ct_scen = n


def test_abstract_base_cannot_be_instantiated():
    with pytest.raises(TypeError):
        ScenarioModel()


def test_nominal_defaults_and_set():
    m = NominalScenarioModel()
    assert m.short_name == "nomScen" and m.name == "Nominal Scenario"
    assert m.setup == SetupUncertainty()
    assert m.range == RangeUncertainty()
    assert m.ct_phase == CTPhaseUncertainty()
    assert m.wc_sigma == 1.0
    assert [d.name for d in m.dimensions()] == ["ct_phase", "setup", "range"]
    assert isinstance(m.distribution_object(), IndependentGaussian)

    s = m.scenarios()
    assert isinstance(s, ScenarioSet)
    assert len(s) == 1
    assert s.nominal_index() == 0 and s[0].is_nominal
    assert np.array_equal(s.values("setup"), np.zeros((1, 3)))
    assert np.array_equal(s.values("range"), np.zeros((1, 2)))
    assert np.isclose(s.weights[0], 1.0)
    expected = IndependentGaussian(m.dimensions()).pdf(np.zeros((1, 5)))
    assert np.allclose(s.probabilities, expected)
    assert s.axes == (
        ("ct_phase.index",),
        ("setup.x", "setup.y", "setup.z"),
        ("range.abs", "range.rel"),
    )
    assert s.layout.shape == (1, 1, 1)


def test_nominal_with_several_phases_weights():
    m = NominalScenarioModel(ct_phase={"phases": [0, 1, 2], "probabilities": [0.2, 0.3, 0.5]})
    s = m.scenarios()
    assert len(s) == 3
    assert np.array_equal(s.ct_phases(), [0, 1, 2])
    assert np.allclose(s.weights, [0.2, 0.3, 0.5])
    assert all(sc.is_nominal for sc in s)
    assert s.num_voxel_param_sets == 3
    assert s.layout.shape == (3, 1, 1)


def test_scenarios_are_cached_and_regenerated_on_change():
    m = NominalScenarioModel()
    s1 = m.scenarios()
    assert m.scenarios() is s1
    m.wc_sigma = 2.0  # field assignment
    s2 = m.scenarios()
    assert s2 is not s1
    m.setup.sd = (1.0, 1.0, 1.0)  # nested mutation also invalidates
    s3 = m.scenarios()
    assert s3 is not s2
    assert s3.probabilities[0] > s2.probabilities[0]
    m.invalidate()
    s4 = m.scenarios()
    assert s4 is not s3
    assert m.scenarios(refresh=True) is not s4


def test_validate_ct():
    m = NominalScenarioModel(ct_phase={"phases": [0, 1], "probabilities": [0.5, 0.5]})
    assert m.num_available_ct_phases == 2
    m.validate_ct(_FakeCT(4))
    assert m.num_available_ct_phases == 4
    with pytest.raises(ValueError, match="only has 1"):
        m.scenarios(_FakeCT(1))
    assert m.scenarios(_FakeCT(3)) is not None


def test_legacy_flat_input_names_map_to_dimensions():
    m = NominalScenarioModel(
        rangeRelSD=5.0, rangeAbsSD=2.5, shiftSD=(3.0, 3.0, 3.0), ctScenProb=[(0, 0.5)], wcSigma=5.0
    )
    assert m.range.rel_sd == 5.0 and m.range.abs_sd == 2.5
    assert m.setup.sd == (3.0, 3.0, 3.0)
    assert m.ct_phase.pairs() == [(0, 0.5)]
    assert m.wc_sigma == 5.0
    m2 = NominalScenarioModel(
        range_rel_sd=4.0, shift_sd=[1, 2, 3], ct_scen_prob=[(0, 1.0)], wc_factor=2
    )
    assert m2.range.rel_sd == 4.0 and m2.setup.sd == (1.0, 2.0, 3.0) and m2.wc_sigma == 2
    # legacy and nested at once: legacy value lands inside the nested dimension
    m3 = NominalScenarioModel(range={"abs_sd": 2.0}, rangeRelSD=1.0)
    assert m3.range.abs_sd == 2.0 and m3.range.rel_sd == 1.0
    m4 = NominalScenarioModel(range=RangeUncertainty(abs_sd=2.0), rangeRelSD=1.0)
    assert m4.range.abs_sd == 2.0 and m4.range.rel_sd == 1.0
    with pytest.raises(ValidationError):
        NominalScenarioModel(rangeRelSD=-1)


def test_legacy_flat_properties_read_and_write():
    m = NominalScenarioModel()
    assert m.range_rel_sd == 3.5 and m.range_abs_sd == 1.0
    assert m.shift_sd == (2.25, 2.25, 2.25)
    assert m.ct_scen_prob == [(0, 1.0)]
    m.range_rel_sd = 7.0
    m.range_abs_sd = 0.5
    m.shift_sd = (1, 1, 1)
    m.ct_scen_prob = [(0, 0.5), (1, 0.5)]
    assert m.range == RangeUncertainty(abs_sd=0.5, rel_sd=7.0)
    assert m.setup.sd == (1.0, 1.0, 1.0)
    assert m.ct_phase.phases == [0, 1]
    assert len(m.scenarios()) == 2
    with pytest.raises(ValidationError):
        m.range_abs_sd = -1


def test_model_dump_and_matrad_export():
    m = NominalScenarioModel(rangeRelSD=5.0, ctScenProb=[(0, 0.5)])
    d = m.model_dump()
    assert d["model"] == "nomScen"
    assert d["range"]["rel_sd"] == 5.0
    assert d["ct_phase"]["phases"] == [0]
    assert d["distribution"] == "independent_gaussian"
    assert NominalScenarioModel.model_validate(d) == m
    assert validate_scenario_model(d) == m

    mat = m.to_matrad()
    assert mat == {
        "model": "nomScen",
        "rangeRelSD": 5.0,
        "rangeAbsSD": 1.0,
        "shiftSD": [2.25, 2.25, 2.25],
        "wcSigma": 1.0,
        "ctScenProb": [(1, 0.5)],
    }
    # the matRad struct validates back (1-based phases are converted)
    back = validate_scenario_model(mat)
    assert back == m
    # serialization inside another model with matRad context yields the same struct
    assert m.model_dump(context={"matRad": "mat-file"}) == mat


def test_distribution_field_accepts_instance_and_dict():
    dims = (CTPhaseUncertainty(), SetupUncertainty(), RangeUncertainty())
    cov = np.diag(IndependentGaussian(dims).std() ** 2)
    m = NominalScenarioModel(distribution=MultivariateGaussian(dims, cov=cov))
    assert isinstance(m.distribution_object(), MultivariateGaussian)
    assert m.model_dump()["distribution"]["distribution"] == "multivariate_gaussian"
    m2 = NominalScenarioModel(distribution={"distribution": "multivariate_gaussian", "cov": cov})
    assert np.allclose(
        m2.scenarios().probabilities, NominalScenarioModel().scenarios().probabilities
    )
    with pytest.raises(ValueError):
        NominalScenarioModel(distribution="copula").scenarios()


def test_explicit_model_wraps_a_set():
    s = ScenarioSet.from_points(
        (CTPhaseUncertainty(), SetupUncertainty(), RangeUncertainty()),
        {"setup": [[0, 0, 0], [3, 0, 0]]},
        axes=MATRAD_AXES,
    )
    m = ExplicitScenarioModel.from_set(s)
    assert m.short_name == "explicit"
    assert m.scenarios() is s
    assert m.setup == SetupUncertainty()
    assert create_scenario_model(s).scenarios() is s
    d = m.model_dump()
    assert d["model"] == "explicit"
    m2 = ExplicitScenarioModel.model_validate(d)
    assert m2.scenarios() == s


def test_extract_single_scenario_returns_explicit_model():
    m = NominalScenarioModel(ct_phase={"phases": [0, 1], "probabilities": [0.25, 0.75]})
    single = m.extract_single_scenario(1)
    assert isinstance(single, ExplicitScenarioModel)
    assert len(single.scenarios()) == 1
    assert single.scenarios().ct_phase(0) == 1
    assert single.ct_phase.pairs() == [(1, 0.75)]
    assert single.scenarios()[0].index == 0


def test_factory_and_registry():
    assert available_scenario_models()[0] == "nomScen"
    assert "explicit" not in available_scenario_models()
    assert get_scenario_model_class("nomScen") is NominalScenarioModel
    assert isinstance(create_scenario_model(), NominalScenarioModel)
    assert isinstance(create_scenario_model(None), NominalScenarioModel)
    assert isinstance(create_scenario_model("nomScen"), NominalScenarioModel)
    m = NominalScenarioModel()
    assert create_scenario_model(m) is m
    assert create_scenario_model({"model": "nomScen", "wc_sigma": 2}).wc_sigma == 2
    with pytest.raises(ValueError):
        create_scenario_model({"wc_sigma": 2})
    with pytest.raises(ValueError):
        create_scenario_model("unknown")
    with pytest.raises(ValueError):
        create_scenario_model(5)
    assert {"wcScen", "impScen", "rndScen"} <= set(available_scenario_models())
    for name in ("wcScen", "impScen", "rndScen"):
        assert create_scenario_model(name).short_name == name
    created = create_scenario_model("nomScen", ct=_FakeCT(3))
    assert created.num_available_ct_phases == 3
    assert NominalScenario is NominalScenarioModel


def test_register_custom_model():
    @register_scenario_model
    class _Tiny(ScenarioModel):
        name = "Tiny"
        short_name = "tinyScen_test"

        def _generate(self):
            return self._make_set({"setup": [[0, 0, 0], [1, 0, 0]]})

    try:
        m = create_scenario_model("tinyScen_test")
        assert len(m.scenarios()) == 2
        assert m.scenarios().layout.shape == (1, 2, 1)
        assert "tinyScen_test" in available_scenario_models()
    finally:
        from pyRadPlan import scenarios

        scenarios._MODEL_REGISTRY.pop("tinyScen_test", None)


@pytest.mark.parametrize(
    "ct_scen_prob",
    [np.array([1.0, 1.0]), np.array([[1.0, 1.0]]), [(1, 1.0)]],
)
def test_validate_scenario_model_ctscenprob_shapes(ct_scen_prob):
    scenario = validate_scenario_model({"model": "nomScen", "ctScenProb": ct_scen_prob})
    assert isinstance(scenario, NominalScenarioModel)
    assert scenario.ct_scen_prob == [(0, 1.0)]
