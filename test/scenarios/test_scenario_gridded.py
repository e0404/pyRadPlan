"""Gridded scenario models (wcScen / impScen) against matRad reference tables."""

from pathlib import Path

import numpy as np
import pytest
import scipy.io
from pydantic import ValidationError

from pyRadPlan.scenarios import (
    GriddedScenarioModel,
    ImportanceScenarioModel,
    ScenarioModel,
    WorstCaseScenarioModel,
    available_scenario_models,
    create_scenario_model,
    validate_scenario_model,
)

FIXTURE_DIR = Path(__file__).parents[1] / "data" / "scenarios"
GRIDDED_FIXTURES = sorted(
    p.stem for p in FIXTURE_DIR.glob("*.mat") if p.stem.startswith(("wcScen", "impScen"))
)


class _FakeCT:
    def __init__(self, n):
        self.num_of_ct_scen = n


#: matRad maps the nominal-setup range scenarios onto its first setup grid point, which is not the
#: nominal shift when the grid has no nominal point (even number of points). The realizations agree;
#: the mask layout is value-derived here and therefore differs.
MATRAD_MASK_QUIRK = {"impScen_even_shift"}


def _load(name):
    data = scipy.io.loadmat(FIXTURE_DIR / f"{name}.mat", squeeze_me=False)
    return {k: v for k, v in data.items() if not k.startswith("__")}


def _scalar(value):
    return np.asarray(value).reshape(-1)[0]


def _matrad_dict(ref):
    out = {
        "model": str(_scalar(ref["model"])),
        "rangeRelSD": float(_scalar(ref["rangeRelSD"])),
        "rangeAbsSD": float(_scalar(ref["rangeAbsSD"])),
        "shiftSD": ref["shiftSD"].reshape(-1),
        "wcSigma": float(_scalar(ref["wcSigma"])),
        "ctScenProb": ref["ctScenProb"],
    }
    if str(_scalar(ref["model"])) in ("wcScen", "impScen"):
        out.update(
            combinations=str(_scalar(ref["combinations"])),
            combineRange=float(_scalar(ref["combineRange"])),
            numOfSetupGridPoints=int(_scalar(ref["numOfSetupGridPoints"])),
            numOfRangeGridPoints=int(_scalar(ref["numOfRangeGridPoints"])),
        )
    return out


def test_fixtures_present():
    assert len(GRIDDED_FIXTURES) >= 30


@pytest.mark.parametrize("name", GRIDDED_FIXTURES)
def test_matches_matrad_reference(name):
    ref = _load(name)
    model = validate_scenario_model(
        _matrad_dict(ref), _FakeCT(int(_scalar(ref["numOfAvailableCtScen"])))
    )
    assert isinstance(model, GriddedScenarioModel)
    scen = model.scenarios()

    # realizations: [phase (1-based in matRad), x, y, z, abs range, rel range]
    expected = ref["scenForProb"].copy()
    expected[:, 0] -= 1
    assert np.array_equal(model.scen_for_prob.shape, expected.shape)
    np.testing.assert_allclose(model.scen_for_prob, expected, rtol=1e-12, atol=1e-12)

    np.testing.assert_allclose(model.scen_prob, ref["scenProb"].reshape(-1), rtol=1e-10)
    np.testing.assert_allclose(model.scen_weight, ref["scenWeight"].reshape(-1), rtol=1e-10)
    np.testing.assert_allclose(scen.weights.sum(), 1.0)

    if name in MATRAD_MASK_QUIRK:
        return
    linear = ref["linearMask"].astype(int) - 1
    assert np.array_equal(model.linear_mask, linear)
    mask_shape = tuple(int(s) for s in ref["scenMaskSize"].reshape(-1))
    assert model.scen_mask.shape == mask_shape
    assert np.array_equal(model.scen_mask, ref["scenMask"].astype(bool))
    assert model.tot_num_shift_scen == int(_scalar(ref["totNumShiftScen"]))
    assert model.tot_num_range_scen == int(_scalar(ref["totNumRangeScen"]))
    assert model.tot_num_scen == int(_scalar(ref["totNumScen"])) == len(scen)

    # new API views
    layout = scen.layout
    assert np.array_equal(layout.position, linear)
    assert np.array_equal(layout.mask, ref["scenMask"].astype(bool)[np.unique(linear[:, 0])])
    for i in range(model.tot_num_scen):
        assert model.sub2scen_ix(*linear[i]) == np.ravel_multi_index(
            linear[i], mask_shape, order="C"
        )
        assert model.scen_num(model.sub2scen_ix(*linear[i])) == i

    # nominal scenario: first in every CT phase, tagged
    nominal = scen.tagged("nominal")
    assert nominal.size == len(model.ct_phase.phases)
    assert scen.nominal_index() == nominal[0] == 0
    for idx in nominal:
        assert np.all(scen.values("setup")[idx] == 0)
        assert np.all(scen.values("range")[idx] == 0)
        assert scen.tags[idx] == frozenset({"nominal"})
    assert sum(1 for t in scen.tags if t) == nominal.size


def test_even_grid_without_nominal_point_has_value_consistent_layout():
    m = ImportanceScenarioModel(
        combinations="shift", num_setup_grid_points=2, num_range_grid_points=2
    )
    s = m.scenarios()
    assert len(s) == 8 + 2
    assert np.all(np.any(s.values("setup") != 0, axis=1)[:8])
    pos = s.layout.position
    assert pos[8, 1] == 0 and pos[0, 1] == 1
    assert m.tot_num_shift_scen == 9
    assert not s.has_nominal()


def test_wc_scen_defaults_and_counts():
    m = WorstCaseScenarioModel()
    assert m.short_name == "wcScen"
    assert m.combinations == "none"
    assert (m.num_setup_grid_points, m.num_range_grid_points) == (3, 3)
    assert m.range.combined is True
    assert len(m.scenarios()) == 9
    assert (m.tot_num_shift_scen, m.tot_num_range_scen) == (7, 3)
    assert m.scenarios().axes[0] == ("ct_phase.index",)

    m.combinations = "all"
    assert len(m.scenarios()) == 27 * 3
    m.range.combined = False
    assert len(m.scenarios()) == 27 * 9


def test_wc_scen_grid_point_validation():
    assert WorstCaseScenarioModel(num_setup_grid_points=1).num_setup_grid_points == 1
    for bad in (0, 2, 5):
        with pytest.raises(ValidationError, match="1 .*or 3"):
            WorstCaseScenarioModel(num_range_grid_points=bad)
    with pytest.raises(ValidationError):
        WorstCaseScenarioModel(combinations="everything")


def test_imp_scen_defaults_and_validation():
    m = ImportanceScenarioModel()
    assert m.short_name == "impScen"
    assert (m.num_setup_grid_points, m.num_range_grid_points) == (9, 9)
    s = m.scenarios()
    assert len(s) == (9 - 1) * 3 + 1 + 8
    for bad in (0, -1):
        with pytest.raises(ValidationError):
            ImportanceScenarioModel(num_setup_grid_points=bad)
    with pytest.raises(ValidationError):
        ImportanceScenarioModel(num_range_grid_points=2.5)


def test_zero_sd_collapses_duplicates():
    m = ImportanceScenarioModel(
        combinations="all", num_setup_grid_points=3, num_range_grid_points=3, range={"abs_sd": 0.0}
    )
    m.range.rel_sd = 0.0
    assert len(m.scenarios().tagged("nominal")) == 1
    assert m.tot_num_range_scen == 1


def test_phases_are_outer_loop_and_weights_follow_phase_probability():
    m = WorstCaseScenarioModel(ct_phase={"phases": [0, 1], "probabilities": [0.25, 0.75]})
    s = m.scenarios()
    n = len(s) // 2
    assert np.array_equal(s.ct_phases(), [0] * n + [1] * n)
    assert np.allclose(s.values("setup")[:n], s.values("setup")[n:])
    np.testing.assert_allclose(s.probabilities[n:] / s.probabilities[:n], 3.0)
    assert np.isclose(s.weights[n:].sum(), 0.75)
    assert s.layout.shape[0] == 2


def test_registry_and_factory():
    for name, cls in (("wcScen", WorstCaseScenarioModel), ("impScen", ImportanceScenarioModel)):
        assert name in available_scenario_models()
        assert isinstance(create_scenario_model(name), cls)
        assert isinstance(create_scenario_model(name), ScenarioModel)


def test_legacy_matrad_dict_round_trip():
    m = validate_scenario_model(
        {
            "model": "wcScen",
            "combinations": "all",
            "combineRange": 0,
            "numOfSetupGridPoints": 1,
            "numOfRangeGridPoints": 3,
            "rangeRelSD": 2.0,
            "ctScenProb": [(1, 0.4), (2, 0.6)],
            "wcSigma": 2,
        }
    )
    assert isinstance(m, WorstCaseScenarioModel)
    assert m.combinations == "all"
    assert m.range.combined is False
    assert m.num_setup_grid_points == 1 and m.num_range_grid_points == 3
    assert m.ct_phase.pairs() == [(0, 0.4), (1, 0.6)]

    mat = m.to_matrad()
    assert mat["model"] == "wcScen"
    assert mat["combinations"] == "all"
    assert mat["combineRange"] is False
    assert mat["numOfSetupGridPoints"] == 1 and mat["numOfRangeGridPoints"] == 3
    assert mat["ctScenProb"] == [(1, 0.4), (2, 0.6)]
    assert validate_scenario_model(mat) == m
    assert m.model_dump(context={"matRad": "mat-file"}) == mat

    snake = WorstCaseScenarioModel(combine_range=False, combinations="shift")
    assert snake.range.combined is False
    assert WorstCaseScenarioModel.model_validate(m.model_dump()) == m


def test_cache_invalidation_on_grid_change():
    m = ImportanceScenarioModel(num_setup_grid_points=3, num_range_grid_points=3)
    s = m.scenarios()
    assert m.scenarios() is s
    m.num_setup_grid_points = 5
    assert len(m.scenarios()) > len(s)
