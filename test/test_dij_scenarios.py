"""Multi-scenario Dijs: matRad cell-array import / export, per-scenario results and labels."""

import numpy as np
import pytest
import scipy.sparse as sp
import SimpleITK as sitk

from pyRadPlan.core import xp_utils
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


def _derived_dij() -> Dij:
    """Two computed scenarios and a derived one (their mean)."""
    derived = ScenarioSet.from_points(
        _dimensions(),
        {"setup": np.array([[0, 0, 0], [1.0, 0, 0], [0.5, 0, 0]])},
        bases=[None, None, {"coefficients": {0: 0.5, 1: 0.5}}],
    )
    dij = _dij(derived.computed())
    container = np.empty((3,), dtype=object)
    container[:2] = list(dij.physical_dose.flat)
    return dij.model_copy(update={"physical_dose": container, "scenarios": derived})


def test_results_per_scenario(sparse_grid_set):
    dij = _dij(sparse_grid_set)
    w = np.linspace(0.5, 1.5, N_BIX)

    out = dij.get_result_arrays_from_intensity(w, scenarios=True)

    per_scenario = out["physical_dose_scenario"]
    assert len(per_scenario) == len(sparse_grid_set)
    for i, dose in enumerate(per_scenario):
        np.testing.assert_allclose(dose, dij.physical_dose.flat[i] @ w)
    np.testing.assert_allclose(out["physical_dose"], per_scenario[0])
    assert len(out["physical_dose_beam"]) == 1
    assert not any(key.endswith("_beam_scenario") for key in out)


def test_results_default_to_the_nominal_scenario():
    """The unsuffixed results are the nominal scenario's, wherever it sits in the set."""
    dij = _dij(_set([[2.25, 0, 0], [0, 0, 0]], [[0, 0], [0, 0]]))
    w = np.linspace(0.5, 1.5, N_BIX)
    assert dij.nominal_scenario_index() == 1

    out = dij.get_result_arrays_from_intensity(w, scenarios=True)
    nominal = dij.physical_dose.flat[1] @ w
    np.testing.assert_allclose(out["physical_dose"], nominal)
    np.testing.assert_allclose(out["physical_dose_scenario"][1], nominal)
    assert not np.allclose(out["physical_dose_scenario"][0], nominal)

    image = dij.compute_result_ct_grid(w)["physical_dose"]
    np.testing.assert_allclose(sitk.GetArrayFromImage(image).ravel(), nominal, rtol=1e-6)

    # without a nominal scenario the first computed one stands in
    shifted = _dij(_set([[2.25, 0, 0], [-2.25, 0, 0]], [[0, 0], [0, 0]]))
    assert shifted.nominal_scenario_index() == 0
    np.testing.assert_allclose(
        shifted.get_result_arrays_from_intensity(w)["physical_dose"],
        shifted.physical_dose.flat[0] @ w,
    )


def test_results_per_scenario_follow_the_nominal_index(sparse_grid_set):
    dij = _dij(sparse_grid_set)
    w = np.ones(N_BIX)
    out = dij.get_result_arrays_from_intensity(w, scenario_index=2, scenarios=[0, 2])

    np.testing.assert_allclose(out["physical_dose"], dij.physical_dose.flat[2] @ w)
    assert out["physical_dose_scenario"][1] is None
    np.testing.assert_allclose(out["physical_dose_scenario"][0], dij.physical_dose.flat[0] @ w)


@pytest.mark.parametrize("scenarios", [False, [1]])
def test_results_without_several_scenarios_have_no_scenario_keys(sparse_grid_set, scenarios):
    dij = _dij(sparse_grid_set)
    out = dij.get_result_arrays_from_intensity(np.ones(N_BIX), scenarios=scenarios)
    assert not any(key.endswith("_scenario") for key in out)


def test_results_per_scenario_with_a_single_scenario_set(two_scenario_set):
    dij = _dij(two_scenario_set.subset([0]))
    out = dij.get_result_arrays_from_intensity(np.ones(N_BIX), scenarios=True)
    assert "physical_dose_scenario" not in out


def test_results_per_scenario_scale_with_fractions(sparse_grid_set):
    dij = _dij(sparse_grid_set)
    w = np.ones(N_BIX)
    single = dij.get_result_arrays_from_intensity(w, scenarios=True)
    course = dij.get_result_arrays_from_intensity(w, num_of_fractions=30, scenarios=True)
    for one, total in zip(single["physical_dose_scenario"], course["physical_dose_scenario"]):
        np.testing.assert_allclose(total, 30 * one)


def test_results_per_scenario_skip_derived_scenarios():
    dij = _derived_dij()
    w = np.ones(N_BIX)

    out = dij.get_result_arrays_from_intensity(w, scenarios=True)
    assert [d is None for d in out["physical_dose_scenario"]] == [False, False, True]

    images = dij.compute_result_ct_grid(w, scenarios=True)["physical_dose_scenario"]
    assert images[2] is None
    scenario_1 = dij.compute_result_ct_grid(w, scenario_index=1)["physical_dose"]
    np.testing.assert_allclose(
        sitk.GetArrayFromImage(images[1]), sitk.GetArrayFromImage(scenario_1)
    )

    with pytest.raises(ValueError, match="derived"):
        dij.get_result_arrays_from_intensity(w, scenarios=[0, 2])
    with pytest.raises(IndexError):
        dij.get_result_arrays_from_intensity(w, scenarios=[0, 3])


def test_scenario_labels(sparse_grid_set):
    assert _dij(sparse_grid_set).scenario_labels() == [
        "nominal",
        "setup (+2.25, 0.0, 0.0) mm",
        "range +1.0 mm / +3.5 %",
    ]
    assert _derived_dij().scenario_labels() == [
        "nominal",
        "setup (+1.0, 0.0, 0.0) mm",
        "derived",
    ]
    no_set = _dij(sparse_grid_set).model_copy(update={"scenarios": None})
    assert no_set.scenario_labels() == ["nominal", "scenario 1", "scenario 2"]


def test_scenario_labels_list_perturbed_dimensions_only():
    phases = CTPhaseUncertainty(phases=[0, 1], probabilities=[0.5, 0.5])
    scen_set = ScenarioSet.from_points(
        [phases, SetupUncertainty(), RangeUncertainty()],
        {
            "ct_phase": np.array([[0], [1], [1]]),
            "setup": np.array([[0, 0, 0], [0, 0, 0], [-0.25, 0, 1.5]]),
            "range": np.array([[0, 0], [0, 0], [0, -0.01]]),
        },
    )
    assert scen_set.labels() == [
        "nominal",
        "phase 1",
        "phase 1, setup (-0.25, 0.0, +1.5) mm, range 0.0 mm / -1.0 %",
    ]
    assert scen_set.label(-1) == scen_set.labels()[-1]


def _union_nnz(matrices) -> int:
    pattern = sum((sp.csc_array(m) != 0).astype(int) for m in matrices)
    return (pattern != 0).nnz


def _with_let_dose(dij: Dij) -> Dij:
    rng = np.random.default_rng(1)
    container = np.empty_like(dij.physical_dose)
    for i in range(container.size):
        container[i] = sp.csc_matrix(
            sp.random(N_VOX, N_BIX, density=0.3, random_state=rng, dtype=np.float32)
        )
    return dij.model_copy(update={"let_dose": container})


def test_share_scenario_structure_puts_all_matrices_on_one_pattern(sparse_grid_set):
    dij = _with_let_dose(_dij(sparse_grid_set))
    shared = dij.share_scenario_structure()

    ref = shared.physical_dose[0]
    assert ref.indices.dtype == np.int32
    assert ref.indices.size == _union_nnz([*dij.physical_dose, *dij.let_dose])
    for q in ("physical_dose", "let_dose"):
        for before, after in zip(getattr(dij, q), getattr(shared, q)):
            assert type(after) is type(before)
            assert (sp.csc_array(before) != sp.csc_array(after)).nnz == 0
            assert np.shares_memory(after.indices, ref.indices)
            assert np.shares_memory(after.indptr, ref.indptr)
    assert shared.scenarios is dij.scenarios
    # the source keeps its separate matrices
    assert not np.shares_memory(dij.physical_dose[0].indices, dij.physical_dose[1].indices)


def test_share_scenario_structure_keeps_aliases_and_derived_slots():
    scen_set = ScenarioSet.from_points(
        _dimensions(),
        {"setup": np.array([[0, 0, 0], [1.0, 0, 0], [0.5, 0, 0], [1.0, 0, 0]])},
        bases=[None, None, {"coefficients": {0: 0.5, 1: 0.5}}, {"coefficients": {1: 1.0}}],
    )
    computed = _dij(scen_set.computed())
    container = np.empty((4,), dtype=object)
    container[:2] = list(computed.physical_dose.flat)
    container[3] = container[1]
    dij = computed.model_copy(update={"physical_dose": container, "scenarios": scen_set})

    shared = dij.share_scenario_structure()

    assert shared.physical_dose[2] is None
    assert shared.physical_dose[3] is shared.physical_dose[1]
    assert np.shares_memory(shared.physical_dose[0].indices, shared.physical_dose[1].indices)
    assert (sp.csc_array(shared.physical_dose[1]) != sp.csc_array(container[1])).nnz == 0


def test_share_scenario_structure_sums_duplicates_without_touching_the_source(two_scenario_set):
    dij = _dij(two_scenario_set)
    duplicated = sp.csc_matrix(
        (np.array([1.0, 2.0, 3.0]), np.array([0, 0, 1]), np.array([0, 2, 3] + [3] * (N_BIX - 2))),
        shape=(N_VOX, N_BIX),
    )
    assert not duplicated.has_canonical_format
    dij.physical_dose[0] = duplicated

    shared = dij.share_scenario_structure()

    assert shared.physical_dose[0][0, 0] == 3.0
    assert shared.physical_dose[0][1, 1] == 3.0
    assert shared.physical_dose[0].has_canonical_format
    assert duplicated.nnz == 3 and not duplicated.has_canonical_format


def test_share_scenario_structure_with_a_single_matrix_returns_self():
    dij = _dij(ScenarioSet.from_points(_dimensions(), {"setup": np.zeros((1, 3))}))
    assert dij.share_scenario_structure() is dij


def test_share_scenario_structure_rejects_dense_matrices(two_scenario_set):
    dij = _dij(two_scenario_set)
    dij.physical_dose[1] = np.zeros((N_VOX, N_BIX), dtype=np.float32)
    with pytest.raises(TypeError, match="compressed sparse"):
        dij.share_scenario_structure()


@pytest.mark.skipif(not xp_utils.cupy_available(), reason="cupy not installed")
def test_shared_structure_is_uploaded_once_to_cupy(sparse_grid_set):
    import cupy as cp

    dij = _with_let_dose(_dij(sparse_grid_set))
    shared = dij.share_scenario_structure()

    on_gpu = shared.to_namespace("cupy")
    pointers = {
        mat.indices.data.ptr for q in ("physical_dose", "let_dose") for mat in getattr(on_gpu, q)
    }
    assert len(pointers) == 1
    assert on_gpu.physical_dose[0].indices.dtype == cp.int32
    for before, after in zip(shared.physical_dose, on_gpu.physical_dose):
        assert (sp.csc_array(after.get()) != sp.csc_array(before)).nnz == 0

    # the union can also be built from cupy matrices
    rebuilt = dij.to_namespace("cupy").share_scenario_structure()
    assert len({mat.indices.data.ptr for mat in rebuilt.physical_dose}) == 1
    for before, after in zip(dij.physical_dose, rebuilt.physical_dose):
        assert (sp.csc_array(after.get()) != sp.csc_array(before)).nnz == 0


def _dij_kwargs(num_matrices: int) -> dict:
    rng = np.random.default_rng(1)
    container = np.empty((num_matrices,), dtype=object)
    for i in range(num_matrices):
        container[i] = sp.csc_matrix(sp.random(N_VOX, N_BIX, density=0.3, random_state=rng))
    grid = {"resolution": {"x": 2.0, "y": 3.0, "z": 4.0}, "dimensions": DIMS}
    return dict(
        ct_grid=grid,
        dose_grid=grid,
        physical_dose=container,
        num_of_beams=1,
        bixel_num=np.arange(1, N_BIX + 1),
        ray_num=np.arange(1, N_BIX + 1),
        beam_num=np.ones(N_BIX, dtype=int),
    )


def test_scenario_set_input_forms(two_scenario_set):
    from_dict = Dij(**_dij_kwargs(2), scenarios=two_scenario_set.model_dump())
    assert from_dict.scenarios == two_scenario_set
    with pytest.raises(ValueError, match="Cannot interpret"):
        Dij(**_dij_kwargs(2), scenarios=3)


def test_scenario_queries_without_a_scenario_set():
    dij = Dij(**_dij_kwargs(2))
    assert dij.scenarios is None
    assert dij.num_of_scenarios == 2
    assert dij.voxel_param_column(1) == 0
    assert dij.nominal_scenario_index() == 0
    out = dij.get_result_arrays_from_intensity(np.ones(N_BIX), scenarios=True)
    assert len(out["physical_dose_scenario"]) == 2
    np.testing.assert_allclose(
        out["physical_dose_scenario"][1], dij.physical_dose[1] @ np.ones(N_BIX)
    )


def test_extend_scenarios_rejects_mismatched_dijs(two_scenario_set):
    dij = _dij(two_scenario_set)
    other_grid = {"resolution": {"x": 1.0, "y": 3.0, "z": 4.0}, "dimensions": DIMS}
    with pytest.raises(ValueError, match="identical dose and CT grids"):
        dij.extend_scenarios(
            Dij(**{**_dij_kwargs(2), "dose_grid": other_grid}, scenarios=two_scenario_set)
        )
    with_bio = Dij(
        **_dij_kwargs(2),
        scenarios=two_scenario_set,
        alphax=np.full((N_VOX, 1), 0.1),
        betax=np.full((N_VOX, 1), 0.05),
    )
    with pytest.raises(ValueError, match="present on only one"):
        dij.extend_scenarios(with_bio)


def test_validate_dij_attaches_a_scenario_model(two_scenario_set):
    from pyRadPlan.scenarios import ExplicitScenarioModel

    model = ExplicitScenarioModel.from_set(two_scenario_set)
    assert validate_dij(Dij(**_dij_kwargs(2)), scenario_model=model).scenarios == two_scenario_set
    assert validate_dij(scenario_model=model, **_dij_kwargs(2)).scenarios == two_scenario_set
