"""matRad cell-array import / export of multi-scenario Dijs."""

import numpy as np
import pytest
import scipy.sparse as sp

from pyRadPlan.dij import Dij, validate_dij
from pyRadPlan.io.matlab import _matfile
from pyRadPlan.scenarios import (
    CTPhaseUncertainty,
    NominalScenarioModel,
    RangeUncertainty,
    ScenarioSet,
    SetupUncertainty,
)

DIMS = (4, 3, 2)
N_VOX = int(np.prod(DIMS))
N_BIX = 6


def _dimensions():
    return [CTPhaseUncertainty(), SetupUncertainty(), RangeUncertainty()]


def _set(setup, range_shift) -> ScenarioSet:
    return ScenarioSet.from_points(
        _dimensions(),
        {"setup": np.asarray(setup, dtype=float), "range": np.asarray(range_shift, dtype=float)},
    )


@pytest.fixture
def two_scenario_set():
    return _set([[0, 0, 0], [2.25, 0, 0]], [[0, 0], [0, 0]])


@pytest.fixture
def sparse_grid_set():
    """Nominal, one setup shift, one range shift: a 1 x 2 x 2 matRad grid with an empty cell."""
    return _set([[0, 0, 0], [2.25, 0, 0], [0, 0, 0]], [[0, 0], [0, 0], [1.0, 0.035]])


def _dij(scen_set: ScenarioSet) -> Dij:
    rng = np.random.default_rng(0)
    container = np.empty((len(scen_set),), dtype=object)
    for i in range(len(scen_set)):
        container[i] = sp.csc_matrix(sp.random(N_VOX, N_BIX, density=0.3, random_state=rng))
    grid = {"resolution": {"x": 2.0, "y": 3.0, "z": 4.0}, "dimensions": DIMS}
    return Dij(
        ct_grid=grid,
        dose_grid=grid,
        physical_dose=container,
        num_of_beams=1,
        bixel_num=np.arange(1, N_BIX + 1),
        ray_num=np.arange(1, N_BIX + 1),
        beam_num=np.ones(N_BIX, dtype=int),
        scenarios=scen_set,
    )


def _round_trip(dij: Dij, tmp_path) -> dict:
    path = tmp_path / "dij.mat"
    _matfile.save(path, {"dij": dij.to_matrad()})
    return _matfile.load(path)["dij"]


def test_export_arranges_cells_on_matrad_axes(sparse_grid_set):
    dij = _dij(sparse_grid_set)
    cells = dij.to_matrad()["physicalDose"]
    assert cells.shape == (1, 2, 2)
    assert dij.to_matrad()["numOfScenarios"] == 3
    assert "scenarios" not in dij.to_matrad()

    layout = sparse_grid_set.layout_for(("ct_phase",), ("setup",), ("range",))
    filled = {tuple(p) for p in layout.position}
    for index in np.ndindex(cells.shape):
        if index in filled:
            assert sp.issparse(cells[index])
        else:
            assert cells[index].size == 0


@pytest.mark.parametrize("set_name", ["two_scenario_set", "sparse_grid_set"])
def test_matrad_round_trip_with_scenario_model(set_name, tmp_path, request):
    scen_set = request.getfixturevalue(set_name)
    dij = _dij(scen_set)
    loaded = _round_trip(dij, tmp_path)

    restored = validate_dij(loaded, scenario_model=scen_set)
    assert restored.scenarios is scen_set
    assert restored.physical_dose.shape == (len(scen_set),)
    for i in range(len(scen_set)):
        diff = restored.physical_dose.flat[i] - dij.physical_dose.flat[i]
        assert abs(diff).max() == 0


def test_matrad_import_without_model_flattens_column_major(sparse_grid_set, tmp_path):
    dij = _dij(sparse_grid_set)
    restored = validate_dij(_round_trip(dij, tmp_path))
    assert restored.scenarios is None
    assert restored.num_of_scenarios == 3

    layout = sparse_grid_set.layout_for(("ct_phase",), ("setup",), ("range",))
    order = np.argsort(
        [np.ravel_multi_index(tuple(p), layout.shape, order="F") for p in layout.position]
    )
    for flat, scen in enumerate(order):
        diff = restored.physical_dose.flat[flat] - dij.physical_dose.flat[scen]
        assert abs(diff).max() == 0

    restored.attach_scenarios(sparse_grid_set)
    assert restored.num_of_scenarios == 3


def test_matrad_import_checks_scenario_count(two_scenario_set, sparse_grid_set, tmp_path):
    loaded = _round_trip(_dij(two_scenario_set), tmp_path)
    with pytest.raises(ValueError):
        validate_dij(loaded, scenario_model=sparse_grid_set)


def test_matrad_round_trip_nominal_model(tmp_path):
    model = NominalScenarioModel()
    dij = _dij(model.scenarios())
    restored = validate_dij(_round_trip(dij, tmp_path), scenario_model=model)
    assert len(restored.scenarios) == 1
    assert abs(restored.physical_dose.flat[0] - dij.physical_dose.flat[0]).max() == 0


def test_matrad_export_rejects_derived_and_foreign_dimensions():
    derived = ScenarioSet.from_points(
        _dimensions(),
        {"setup": np.array([[0, 0, 0], [1.0, 0, 0], [0.5, 0, 0]])},
        bases=[None, None, {"coefficients": {0: 0.5, 1: 0.5}}],
    )
    dij = _dij(derived.computed())
    container = np.empty((3,), dtype=object)
    container[:2] = list(dij.physical_dose.flat)
    dij_derived = dij.model_copy(update={"physical_dose": container, "scenarios": derived})
    with pytest.raises(ValueError, match="Derived scenarios"):
        dij_derived.to_matrad()
